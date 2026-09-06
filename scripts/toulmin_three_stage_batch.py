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
VERSION = "toulmin-three-stage-locked-v4-remedy-bridge"
OUTCOMES = {"full_support", "partial_support", "rejected", "not_addressed"}
OPERATIONS = {"AWARD_MONEY", "ORDER_ACT", "DISMISS", "NOT_ADDRESS"}
LIABILITY_MODES = {"individual", "joint", "several", "supplementary", "insurance_limit", "unknown"}
FINDINGS = {"satisfied", "partially_satisfied", "not_satisfied", "not_addressed"}
DEFENSE_EFFECTS = {"accepted", "partially_accepted", "rejected", "not_addressed"}

SYSTEM = {
    1: "你只负责诉请和事实抽取，不预测结果，不补造材料。严格按字段契约输出单个JSON对象。",
    2: "你只负责规则适用和抗辩。D1/D2已经锁定，所有ID必须原样引用，不得修改、缩写或新增。严格按字段契约输出单个JSON对象。",
    3: "你只负责限定条件和最终裁判。D1-D5已经锁定，不得新增事实、法条、诉请、金额来源或当事人。严格按字段契约输出单行紧凑JSON对象，不要Markdown、缩进或换行。字符串内部不得出现未转义的双引号、反斜杠或控制字符。",
}

INSTRUCTION = {
    1: """仅输出：
{"claim_source_units":[{"source_no":1,"source_text":"诉请连续原文","disposition":"substantive_claim或procedural_excluded","claim_ids":["已有claim_id；排除项为空"]}],"D1_claim":[{"claim_id":"M1或N1并分别连续编号","claim_type":"monetary或non_monetary","item":"简短项目","request_text":"原告诉请中的连续原文","requested_amount":"数字或null","parties":{"plaintiff_names":["原文名称"],"defendant_names":["原文名称"]}}],"D2_data":[{"fact_id":"F1并连续编号","fact_text":"court_found_facts中的连续原文","related_claim_ids":["已有claim_id"]}]}
规则：先按编号或语义分项扫描全部诉请并建立claim_source_units，再拆分全部实体诉请；诉讼费、受理费、保全费、鉴定费等程序费用标为procedural_excluded且不得建立claim。每个substantive_claim必须链接至少一个claim_id，每个D1 claim_id必须被来源单元链接。金钱诉请用M1、M2；非金钱诉请用N1、N2。requested_amount不得推算。事实只保留裁判必需内容，共同事实只定义一次。不得输出结果、比例或判赔金额。""",
    2: """仅输出：
{"D3_warrant":[{"rule_id":"R1并连续编号","rule_text":"法律规则","related_claim_ids":["已有claim_id"],"element_findings":[{"element":"要件","finding":"satisfied或partially_satisfied或not_satisfied或not_addressed","fact_ids":["已有fact_id"]}]}],"D4_backing":[{"rule_id":"已有rule_id","law_ids":["候选law_id"]}],"D5_rebuttal":[{"defense_id":"D1并连续编号","defense_text":"答辩原文或忠实概括","defendant_names":["答辩中名称"],"related_claim_ids":["已有claim_id"],"predicted_effect":"accepted或partially_accepted或rejected或not_addressed","reason":"简短理由"}],"remedy_analysis":[{"claim_id":"已有claim_id","requested_amount":"复制D1数字或null","provable_base_amounts":[{"amount":"数字","fact_ids":["已有fact_id"],"meaning":"简短含义"}],"applicable_ratios":[{"ratio":"0至1数字","fact_ids":["已有fact_id"],"rule_ids":["已有rule_id"]}],"discretion_factors":["最多3个短语"],"payer_candidates":["锁定材料已有名称"],"liability_mode_candidate":"individual或joint或several或supplementary或insurance_limit或unknown","upper_bound":"数字或null","calculation_note":"一句计算方法"}],"liability_matrix":[{"party_name":"锁定材料已有名称","related_claim_ids":["已有claim_id"],"role":"责任角色","liability_mode":"individual或joint或several或supplementary或insurance_limit或unknown","share_or_limit":"数字或null","fact_ids":["已有fact_id"],"rule_ids":["已有rule_id"]}]}
规则：claim_id和fact_id必须逐字符复制锁定输入；不得使用M、N等缩写，不得添加新诉请或事实。每个element_findings只能引用已有fact_id。D4只能使用候选law_id。材料没有实质抗辩时D5_rebuttal为空数组。每个claim_id必须有且仅有一条remedy_analysis；金额、比例、限额必须有事实或规则ID依据，无法确定时用空数组或null，不得猜测。liability_matrix只列可能承担责任的主体。不得输出最终outcome、operation或awarded_amount。""",
    3: """仅输出以下完整结构，不得改名、缺字段或增加替代字段：
{"reasoning":{"decision_matrix":[{"claim_id":"已有claim_id","outcome":"合法枚举","requested_amount":"复制D1","calculated_amount":"数字或null","operation":"合法枚举","payer_names":["责任矩阵已有名称"],"liability_mode":"合法责任枚举","fact_ids":["已有fact_id"],"rule_ids":["已有rule_id"]}],"D6_qualifier":[{"claim_id":"已有claim_id","strength":"high或medium或low","limitations":["限定"],"uncertainty":"说明"}],"D7_conclusion":[{"claim_id":"已有claim_id","decision_reason":"结论理由","fact_ids":["已有fact_id"],"rule_ids":["已有rule_id"],"law_ids":["候选law_id"],"defense_ids":["已有defense_id"]}]},"claim_results":[{"claim_id":"已有claim_id","claim_type":"monetary或non_monetary","item":"复制D1","request_text":"复制D1","requested_amount":"复制D1数字或null","outcome":"full_support或partial_support或rejected或not_addressed","awarded_amount":"数字或null","operation":"AWARD_MONEY或ORDER_ACT或DISMISS或NOT_ADDRESS","decision_reason":"简短理由"}],"payment_result":{"has_payment":"boolean","total_awarded_amount":"数字或null","obligations":[{"obligation_id":"O1并连续编号","payer_names":["责任矩阵已有名称"],"payee_names":["D1已有名称"],"related_claim_ids":["已有且获金钱支持的claim_id"],"amount":"数字","liability_mode":"individual或joint或several或supplementary或insurance_limit或unknown","reason":"简短理由"}]}}
规则：先依据remedy_analysis和liability_matrix为每项诉请生成唯一decision_matrix行，再逐行复制为claim_results并汇总payment_result。每个D1 claim_id在decision_matrix、claim_results和D7中恰好出现一次且逐字符复制。D7_conclusion只能填写ID列表和一句简短decision_reason，不得重复事实、规则、法条正文或D1-D5内容；D6每项limitations最多2条，uncertainty最多一句；其他reason字段也只写一句。awarded_amount/calculated_amount不得超过requested_amount；有明确基数和比例时必须计算；无可靠金额依据时才可酌定且须引用discretion_factors；不得把请求金额直接当作支持金额。monetary rejected的金额=0；not_addressed=null；non_monetary始终为null。金钱给付用AWARD_MONEY，行为救济用ORDER_ACT，驳回用DISMISS。has_payment=false时total_awarded_amount=null且obligations=[]；true时总额必须与付款义务一致。不得把payment_result输出为数组。输出必须是单行紧凑合法JSON；所有字符串中的双引号、反斜杠和换行必须按JSON规则转义，禁止尾随逗号。""",
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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _exact_ids(values: Any, key: str, expected: set[str], label: str) -> None:
    ids = _ids(values, key)
    if len(ids) != len(expected) or set(ids) != expected:
        raise ValueError(f"{label} ids must exactly equal locked claims")


def _sequential(ids: list[str], prefix: str) -> bool:
    return sorted(int(value[len(prefix):]) for value in ids if value.startswith(prefix)) == list(range(1, len(ids) + 1))


def validate(stage: int, value: Any, prior1: dict[str, Any] | None = None, prior2: dict[str, Any] | None = None, laws: set[str] | None = None) -> None:
    if not isinstance(value, dict):
        raise ValueError("output must be an object")
    if stage == 1:
        units, claims, facts = value.get("claim_source_units"), value.get("D1_claim"), value.get("D2_data")
        if not isinstance(units, list) or not units or not isinstance(claims, list) or not claims or not isinstance(facts, list):
            raise ValueError("claim_source_units/D1_claim/D2_data invalid")
        claim_ids = _ids(claims, "claim_id")
        money_ids = [cid for cid in claim_ids if cid.startswith("M")]
        nonmoney_ids = [cid for cid in claim_ids if cid.startswith("N")]
        if (len(claim_ids) != len(set(claim_ids)) or any(not re.fullmatch(r"[MN][1-9]\d*", cid) for cid in claim_ids)
                or not _sequential(money_ids, "M") or not _sequential(nonmoney_ids, "N")):
            raise ValueError("claim_ids must be unique M1/N1 sequences")
        required = {"claim_id", "claim_type", "item", "request_text", "requested_amount", "parties"}
        if any(not required <= set(claim) or claim.get("claim_type") not in {"monetary", "non_monetary"} for claim in claims):
            raise ValueError("D1 claim contract invalid")
        fact_ids = _ids(facts, "fact_id")
        if len(fact_ids) != len(set(fact_ids)) or any(not re.fullmatch(r"F[1-9]\d*", fid) for fid in fact_ids):
            raise ValueError("fact_ids invalid")
        if any(not {"fact_id", "fact_text", "related_claim_ids"} <= set(fact) or not set(fact["related_claim_ids"]) <= set(claim_ids) for fact in facts):
            raise ValueError("D2 fact contract/reference invalid")
        covered: set[str] = set()
        for unit in units:
            if not isinstance(unit, dict) or not {"source_no", "source_text", "disposition", "claim_ids"} <= set(unit):
                raise ValueError("claim source unit contract invalid")
            refs = unit.get("claim_ids")
            if not isinstance(refs, list) or not set(refs) <= set(claim_ids):
                raise ValueError("claim source unit references invalid")
            if unit.get("disposition") == "procedural_excluded" and refs:
                raise ValueError("procedural source unit cannot create claims")
            if unit.get("disposition") == "substantive_claim" and not refs:
                raise ValueError("substantive source unit must link claims")
            if unit.get("disposition") not in {"substantive_claim", "procedural_excluded"}:
                raise ValueError("claim source unit disposition invalid")
            if unit.get("disposition") == "substantive_claim":
                covered.update(refs)
        if covered != set(claim_ids):
            raise ValueError("claim source coverage incomplete")
        return
    claim_ids = set(_ids(prior1.get("D1_claim"), "claim_id")); fact_ids = set(_ids(prior1.get("D2_data"), "fact_id"))
    if stage == 2:
        warrants, backing, rebuttals = value.get("D3_warrant"), value.get("D4_backing"), value.get("D5_rebuttal")
        remedies, liability = value.get("remedy_analysis"), value.get("liability_matrix")
        if not all(isinstance(part, list) for part in (warrants, backing, rebuttals, remedies, liability)):
            raise ValueError("D3/D4/D5/remedy/liability must be arrays")
        rule_id_list = _ids(warrants, "rule_id"); rule_ids = set(rule_id_list)
        if not rule_ids or len(rule_ids) != len(rule_id_list) or any(not re.fullmatch(r"R[1-9]\d*", rid) for rid in rule_ids) or any(not set(w.get("related_claim_ids", [])) <= claim_ids for w in warrants):
            raise ValueError("D3 claim references invalid")
        for warrant in warrants:
            findings = warrant.get("element_findings")
            if not isinstance(findings, list) or any(not isinstance(f, dict) or f.get("finding") not in FINDINGS or not set(f.get("fact_ids", [])) <= fact_ids for f in findings):
                raise ValueError("D3 fact references invalid")
        if any(b.get("rule_id") not in rule_ids or not set(b.get("law_ids", [])) <= (laws or set()) for b in backing):
            raise ValueError("D4 rule/law references invalid")
        defense_ids = _ids(rebuttals, "defense_id")
        if len(defense_ids) != len(set(defense_ids)) or any(d.get("predicted_effect") not in DEFENSE_EFFECTS or not set(d.get("related_claim_ids", [])) <= claim_ids for d in rebuttals):
            raise ValueError("D5 claim references invalid")
        _exact_ids(remedies, "claim_id", claim_ids, "remedy_analysis")
        locked_claims = {str(c["claim_id"]): c for c in prior1["D1_claim"]}
        for remedy in remedies:
            cid = str(remedy.get("claim_id"))
            if remedy.get("requested_amount") != locked_claims[cid].get("requested_amount") or remedy.get("liability_mode_candidate") not in LIABILITY_MODES:
                raise ValueError("remedy locked value invalid")
            for base in remedy.get("provable_base_amounts", []):
                if not _is_number(base.get("amount")) or not set(base.get("fact_ids", [])) <= fact_ids:
                    raise ValueError("remedy base invalid")
            for ratio in remedy.get("applicable_ratios", []):
                if not _is_number(ratio.get("ratio")) or not 0 <= ratio["ratio"] <= 1 or not set(ratio.get("fact_ids", [])) <= fact_ids or not set(ratio.get("rule_ids", [])) <= rule_ids:
                    raise ValueError("remedy ratio invalid")
        for row in liability:
            if row.get("liability_mode") not in LIABILITY_MODES or not set(row.get("related_claim_ids", [])) <= claim_ids or not set(row.get("fact_ids", [])) <= fact_ids or not set(row.get("rule_ids", [])) <= rule_ids:
                raise ValueError("liability matrix invalid")
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
    matrix, d6, d7 = reasoning.get("decision_matrix"), reasoning.get("D6_qualifier"), reasoning.get("D7_conclusion")
    if not isinstance(matrix, list) or not isinstance(d6, list) or not isinstance(d7, list):
        raise ValueError("reasoning contract invalid")
    _exact_ids(matrix, "claim_id", claim_ids, "decision_matrix"); _exact_ids(d6, "claim_id", claim_ids, "D6"); _exact_ids(d7, "claim_id", claim_ids, "D7")
    locked = {str(c["claim_id"]): c for c in prior1["D1_claim"]}
    matrix_by_id = {str(row["claim_id"]): row for row in matrix}
    rule_ids = set(_ids(prior2.get("D3_warrant", []), "rule_id")); defense_ids = set(_ids(prior2.get("D5_rebuttal", []), "defense_id"))
    for claim in claims:
        cid = str(claim["claim_id"]); source = locked[cid]; decision = matrix_by_id[cid]
        if any(claim.get(key) != source.get(key) for key in ("claim_type", "item", "request_text", "requested_amount")):
            raise ValueError("claim_results must copy locked D1")
        if decision.get("outcome") != claim.get("outcome") or decision.get("operation") != claim.get("operation") or decision.get("requested_amount") != claim.get("requested_amount") or decision.get("calculated_amount") != claim.get("awarded_amount"):
            raise ValueError("decision matrix and claim result mismatch")
        if decision.get("outcome") not in OUTCOMES or decision.get("operation") not in OPERATIONS or decision.get("liability_mode") not in LIABILITY_MODES or not set(decision.get("fact_ids", [])) <= fact_ids or not set(decision.get("rule_ids", [])) <= rule_ids:
            raise ValueError("decision matrix contract/reference invalid")
        award, requested = claim.get("awarded_amount"), claim.get("requested_amount")
        if claim.get("claim_type") == "non_monetary" and award is not None:
            raise ValueError("non-monetary award must be null")
        if claim.get("outcome") == "rejected" and claim.get("claim_type") == "monetary" and award != 0:
            raise ValueError("rejected monetary award must be zero")
        if claim.get("outcome") == "not_addressed" and award is not None:
            raise ValueError("not-addressed award must be null")
        if _is_number(award) and _is_number(requested) and award > requested:
            raise ValueError("award exceeds request")
    for row in d7:
        if not set(row.get("fact_ids", [])) <= fact_ids or not set(row.get("rule_ids", [])) <= rule_ids or not set(row.get("defense_ids", [])) <= defense_ids or not set(row.get("law_ids", [])) <= (laws or set()):
            raise ValueError("D7 references invalid")
    if payment.get("has_payment") is False and (payment.get("total_awarded_amount") is not None or payment.get("obligations")):
        raise ValueError("no-payment fields inconsistent")
    obligation_ids = _ids(payment.get("obligations"), "obligation_id")
    if len(obligation_ids) != len(set(obligation_ids)) or any(ob.get("liability_mode") not in LIABILITY_MODES or not _is_number(ob.get("amount")) for ob in payment.get("obligations", [])):
        raise ValueError("payment obligation invalid")


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
