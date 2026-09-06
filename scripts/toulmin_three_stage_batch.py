#!/usr/bin/env python3
"""Prepare and ingest a strict, no-repair, three-stage Toulmin Qwen Batch run."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

MODEL = "qwen3.7-max"
VERSION = "toulmin-three-stage-locked-v2"
OUTCOMES = {"full_support", "partial_support", "rejected", "not_addressed"}
OPERATIONS = {"AWARD_MONEY", "ORDER_ACT", "DISMISS", "NOT_ADDRESS"}

SYSTEM = {
    1: "你只负责诉请和事实抽取，不预测结果，不补造材料。严格按字段契约输出单个JSON对象。",
    2: "你只负责规则适用和抗辩。D1/D2已经锁定，所有ID必须原样引用，不得修改、缩写或新增。严格按字段契约输出单个JSON对象。",
    3: "你只负责限定条件和最终裁判。D1-D5已经锁定，不得新增事实、法条、诉请、金额来源或当事人。严格按字段契约输出单个JSON对象。",
}

INSTRUCTION = {
    1: """仅输出：
{"D1_claim":[{"claim_id":"M1或N1并分别连续编号","claim_type":"monetary或non_monetary","item":"简短项目","request_text":"原告诉请中的连续原文","requested_amount":"数字或null","parties":{"plaintiff_names":["原文名称"],"defendant_names":["原文名称"]}}],"D2_data":[{"fact_id":"F1并连续编号","fact_text":"court_found_facts中的连续原文","related_claim_ids":["已有claim_id"]}]}
规则：逐项拆分全部实体诉请；诉讼费、受理费、保全费、鉴定费等程序费用不得建立claim。金钱诉请用M1、M2；非金钱诉请用N1、N2。requested_amount不得推算。事实只保留裁判必需内容，共同事实只定义一次。不得输出结果、比例或判赔金额。""",
    2: """仅输出：
{"D3_warrant":[{"rule_id":"R1并连续编号","rule_text":"法律规则","related_claim_ids":["已有claim_id"],"element_findings":[{"element":"要件","finding":"satisfied或partially_satisfied或not_satisfied或not_addressed","fact_ids":["已有fact_id"]}]}],"D4_backing":[{"rule_id":"已有rule_id","law_ids":["候选law_id"]}],"D5_rebuttal":[{"defense_id":"D1并连续编号","defense_text":"答辩原文或忠实概括","defendant_names":["答辩中名称"],"related_claim_ids":["已有claim_id"],"predicted_effect":"accepted或partially_accepted或rejected或not_addressed","reason":"简短理由"}]}
规则：claim_id和fact_id必须逐字符复制锁定输入；不得使用M、N等缩写，不得添加新诉请或事实。每个element_findings只能引用已有fact_id。D4只能使用候选law_id。材料没有实质抗辩时D5_rebuttal为空数组。不得输出最终outcome、operation或金额。""",
    3: """仅输出以下完整结构，不得改名、缺字段或增加替代字段：
{"reasoning":{"D6_qualifier":[{"claim_id":"已有claim_id","strength":"high或medium或low","limitations":["限定"],"uncertainty":"说明"}],"D7_conclusion":[{"claim_id":"已有claim_id","decision_reason":"结论理由","fact_ids":["已有fact_id"],"rule_ids":["已有rule_id"],"law_ids":["候选law_id"],"defense_ids":["已有defense_id"]}]},"claim_results":[{"claim_id":"已有claim_id","claim_type":"monetary或non_monetary","item":"复制D1","request_text":"复制D1","requested_amount":"复制D1数字或null","outcome":"full_support或partial_support或rejected或not_addressed","awarded_amount":"数字或null","operation":"AWARD_MONEY或ORDER_ACT或DISMISS或NOT_ADDRESS","decision_reason":"简短理由"}],"payment_result":{"has_payment":"boolean","total_awarded_amount":"数字或null","obligations":[{"obligation_id":"O1并连续编号","payer_names":["D1/D5已有名称"],"payee_names":["D1已有名称"],"related_claim_ids":["已有且获金钱支持的claim_id"],"amount":"数字","liability_mode":"individual或joint或several或supplementary或insurance_limit或unknown","reason":"简短理由"}]}}
规则：每个D1 claim_id在claim_results和D7中恰好出现一次且逐字符复制。monetary rejected的awarded_amount=0；not_addressed=null；non_monetary始终为null。金钱给付用AWARD_MONEY，行为救济用ORDER_ACT，驳回用DISMISS。has_payment=false时total_awarded_amount=null且obligations=[]；true时总额必须与付款义务一致。不得把payment_result输出为数组。""",
}


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def indexed(path: Path | None) -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in read(path)} if path else {}


def select_cases(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Same deterministic stratified selector used by MAIN-1."""
    if len(rows) < limit:
        raise ValueError(f"requested {limit} cases, found {len(rows)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("stratum") or "unknown")].append(row)
    for values in groups.values():
        values.sort(key=lambda row: (int(row.get("queue_index") or 0), str(row.get("case_id"))))
    quotas = {key: 1 for key in groups}
    remaining = limit - len(quotas)
    if remaining < 0:
        raise ValueError("limit is smaller than the number of strata")
    exact = {key: remaining * len(values) / len(rows) for key, values in groups.items()}
    for key, value in exact.items():
        quotas[key] += int(value)
    left = limit - sum(quotas.values())
    order = sorted(groups, key=lambda key: (exact[key] - int(exact[key]), len(groups[key]), key), reverse=True)
    for key in order[:left]:
        quotas[key] += 1
    selected = [row for key, values in sorted(groups.items()) for row in values[: min(quotas[key], len(values))]]
    if len(selected) < limit:
        chosen = {str(row["case_id"]) for row in selected}
        remainder_rows = sorted((row for row in rows if str(row["case_id"]) not in chosen), key=lambda row: (int(row.get("queue_index") or 0), str(row["case_id"])))
        selected.extend(remainder_rows[: limit - len(selected)])
    return sorted(selected, key=lambda row: (int(row.get("queue_index") or 0), str(row.get("case_id"))))


def stage_payload(stage: int, case: dict[str, Any], s1: dict[str, Any] | None, s2: dict[str, Any] | None) -> dict[str, Any]:
    material = case["model_input"]
    if stage == 1:
        return {key: material.get(key) for key in ("case_type", "plaintiff_statement", "court_found_facts")}
    if stage == 2:
        return {"locked_D1_D2": s1["stage_output"], "defendant_statement": material.get("defendant_statement"), "law_candidates": material.get("law_candidates", [])}
    return {"locked_D1_D2": s1["stage_output"], "locked_D3_D5": s2["stage_output"]}


def _ids(values: Any, key: str) -> list[str]:
    return [str(value.get(key)) for value in values if isinstance(value, dict)] if isinstance(values, list) else []


def validate(stage: int, value: Any, prior1: dict[str, Any] | None = None, prior2: dict[str, Any] | None = None, laws: set[str] | None = None) -> None:
    if not isinstance(value, dict):
        raise ValueError("output must be an object")
    if stage == 1:
        claims, facts = value.get("D1_claim"), value.get("D2_data")
        if not isinstance(claims, list) or not claims or not isinstance(facts, list):
            raise ValueError("D1_claim/D2_data invalid")
        claim_ids = _ids(claims, "claim_id")
        if len(claim_ids) != len(set(claim_ids)) or any(not re.fullmatch(r"[MN][1-9]\d*", cid) for cid in claim_ids):
            raise ValueError("claim_ids must be unique M1/N1 sequences")
        required = {"claim_id", "claim_type", "item", "request_text", "requested_amount", "parties"}
        if any(not required <= set(claim) or claim.get("claim_type") not in {"monetary", "non_monetary"} for claim in claims):
            raise ValueError("D1 claim contract invalid")
        fact_ids = _ids(facts, "fact_id")
        if len(fact_ids) != len(set(fact_ids)) or any(not re.fullmatch(r"F[1-9]\d*", fid) for fid in fact_ids):
            raise ValueError("fact_ids invalid")
        if any(not {"fact_id", "fact_text", "related_claim_ids"} <= set(fact) or not set(fact["related_claim_ids"]) <= set(claim_ids) for fact in facts):
            raise ValueError("D2 fact contract/reference invalid")
        return
    claim_ids = set(_ids(prior1.get("D1_claim"), "claim_id")); fact_ids = set(_ids(prior1.get("D2_data"), "fact_id"))
    if stage == 2:
        warrants, backing, rebuttals = value.get("D3_warrant"), value.get("D4_backing"), value.get("D5_rebuttal")
        if not all(isinstance(part, list) for part in (warrants, backing, rebuttals)):
            raise ValueError("D3/D4/D5 must be arrays")
        rule_ids = set(_ids(warrants, "rule_id"))
        if not rule_ids or any(not set(w.get("related_claim_ids", [])) <= claim_ids for w in warrants):
            raise ValueError("D3 claim references invalid")
        for warrant in warrants:
            findings = warrant.get("element_findings")
            if not isinstance(findings, list) or any(not isinstance(f, dict) or not set(f.get("fact_ids", [])) <= fact_ids for f in findings):
                raise ValueError("D3 fact references invalid")
        if any(b.get("rule_id") not in rule_ids or not set(b.get("law_ids", [])) <= (laws or set()) for b in backing):
            raise ValueError("D4 rule/law references invalid")
        if any(not set(d.get("related_claim_ids", [])) <= claim_ids for d in rebuttals):
            raise ValueError("D5 claim references invalid")
        return
    claims, payment, reasoning = value.get("claim_results"), value.get("payment_result"), value.get("reasoning")
    if not isinstance(claims, list) or not isinstance(payment, dict) or not isinstance(reasoning, dict):
        raise ValueError("final output types invalid")
    if set(_ids(claims, "claim_id")) != claim_ids or len(claims) != len(claim_ids):
        raise ValueError("final claim_ids must exactly equal locked D1")
    required = {"claim_id", "claim_type", "item", "request_text", "requested_amount", "outcome", "awarded_amount", "operation", "decision_reason"}
    if any(not required <= set(claim) or claim.get("outcome") not in OUTCOMES or claim.get("operation") not in OPERATIONS for claim in claims):
        raise ValueError("claim_results contract invalid")
    if not {"has_payment", "total_awarded_amount", "obligations"} <= set(payment) or not isinstance(payment.get("obligations"), list):
        raise ValueError("payment_result contract invalid")
    if any(not set(ob.get("related_claim_ids", [])) <= claim_ids for ob in payment["obligations"] if isinstance(ob, dict)):
        raise ValueError("payment claim references invalid")
    if not isinstance(reasoning.get("D6_qualifier"), list) or not isinstance(reasoning.get("D7_conclusion"), list):
        raise ValueError("reasoning contract invalid")


def prepare(args: argparse.Namespace) -> int:
    cases = select_cases(read(args.blind), args.limit)
    p1, p2 = indexed(args.stage1), indexed(args.stage2)
    rows = []
    for case in cases:
        cid = str(case["case_id"])
        if (args.stage >= 2 and cid not in p1) or (args.stage == 3 and cid not in p2):
            raise SystemExit(f"missing prior stage for {cid}")
        payload = stage_payload(args.stage, case, p1.get(cid), p2.get(cid))
        rows.append({"custom_id": f"toulmin3:s{args.stage}:{cid}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": args.model, "messages": [{"role": "system", "content": SYSTEM[args.stage]}, {"role": "user", "content": INSTRUCTION[args.stage] + "\n输入：" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}], "temperature": 0, "enable_thinking": False, "max_tokens": args.max_tokens or (6000 if args.stage == 3 else 3500)}})
    write(args.output, rows)
    print(json.dumps({"version": VERSION, "stage": args.stage, "requests": len(rows), "model": args.model, "selected_case_ids": [str(case["case_id"]) for case in cases]}, ensure_ascii=False))
    return 0


def ingest(args: argparse.Namespace) -> int:
    prior1, prior2 = indexed(args.stage1), indexed(args.stage2)
    blind = indexed(args.blind) if args.blind else {}
    out, errors = [], []
    for row in read(args.batch_result):
        cid = str(row.get("custom_id", "")).split(":", 2)[-1]
        try:
            body = row["response"]["body"]
            raw = body["choices"][0]["message"]["content"].strip()
            value = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I))
            s1 = prior1.get(cid, {}).get("stage_output")
            s2 = prior2.get(cid, {}).get("stage_output")
            laws = {str(x.get("law_id")) for x in blind.get(cid, {}).get("model_input", {}).get("law_candidates", []) if isinstance(x, dict)}
            validate(args.stage, value, s1, s2, laws)
            item = {"case_id": cid, "stage": args.stage, "prompt_version": VERSION, "stage_output": value, "usage": body.get("usage", {})}
            if args.stage == 3:
                item.update({"condition": "T2_TOULMIN_THREE_STAGE", "model_output": value})
            out.append(item)
        except Exception as exc:
            errors.append({"case_id": cid, "error": f"{type(exc).__name__}: {exc}"})
    write(args.output, out); write(args.errors, errors)
    print(json.dumps({"stage": args.stage, "successful": len(out), "errors": len(errors)}, ensure_ascii=False))
    return 0 if not errors else 2


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare"); command.add_argument("--stage", type=int, choices=(1, 2, 3), required=True); command.add_argument("--blind", type=Path, required=True)
    command.add_argument("--stage1", type=Path); command.add_argument("--stage2", type=Path); command.add_argument("--output", type=Path, required=True)
    command.add_argument("--limit", type=int, default=50); command.add_argument("--model", default=MODEL); command.add_argument("--max-tokens", type=int); command.set_defaults(handler=prepare)
    command = sub.add_parser("ingest"); command.add_argument("--stage", type=int, choices=(1, 2, 3), required=True); command.add_argument("--batch-result", type=Path, required=True)
    command.add_argument("--blind", type=Path); command.add_argument("--stage1", type=Path); command.add_argument("--stage2", type=Path); command.add_argument("--output", type=Path, required=True); command.add_argument("--errors", type=Path, required=True); command.set_defaults(handler=ingest)
    return parser


if __name__ == "__main__":
    args = parser().parse_args(); raise SystemExit(args.handler(args))
