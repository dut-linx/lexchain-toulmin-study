#!/usr/bin/env python3
"""Prepare/ingest a token-controlled, locked three-stage Toulmin Qwen Batch run."""

from __future__ import annotations

import argparse, json, re
from pathlib import Path
from typing import Any

MODEL = "qwen3.7-max"
VERSION = "toulmin-three-stage-locked-v1"

SYSTEM = {
    1: "你只负责诉请与事实抽取。不得预测结果，不得补造材料。只输出JSON对象。",
    2: "你只负责规则适用与抗辩。D1/D2已锁定，不得改写；不得输出裁判结果。只输出JSON对象。",
    3: "你只负责限定条件和最终裁判。D1-D5已锁定，不得新增事实、法条、诉请、金额来源或当事人。只输出JSON对象。",
}
INSTRUCTION = {
    1: (
        "输出D1_claim和D2_data。D1逐项拆分实体诉请并分配M/N claim_id；D2只提取裁判所需事实，"
        "为其分配fact_id并列related_claim_ids。共同事实只定义一次。"
    ),
    2: (
        "输出D3_warrant、D4_backing、D5_rebuttal。每个rule_id列related_claim_ids，element_findings只能引用已有fact_ids；"
        "D4只能链接候选law_id；真实抗辩用defense_id链接claim_id，无实质抗辩用空数组。"
    ),
    3: (
        "输出reasoning、claim_results、payment_result。reasoning含D6_qualifier与D7_conclusion；D7按claim_id总结且与最终两表一致。"
        "claim_results/payment_result字段和枚举遵循LexChain既定格式。必须保持claim_id→fact_id→rule_id→law_id及defense_id→claim_id可追踪。"
    ),
}

def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in rows), encoding="utf-8")

def indexed(path: Path | None) -> dict[str, dict[str, Any]]:
    return {str(x["case_id"]): x for x in read(path)} if path else {}

def stage_payload(stage: int, case: dict[str, Any], s1: dict[str, Any] | None, s2: dict[str, Any] | None) -> dict[str, Any]:
    material = case["model_input"]
    if stage == 1:
        keep = ("case_type", "plaintiff_statement", "court_found_facts")
        return {k: material.get(k) for k in keep}
    if stage == 2:
        return {
            "locked_D1_D2": s1["stage_output"],
            "defendant_statement": material.get("defendant_statement"),
            "law_candidates": material.get("law_candidates", []),
        }
    return {"locked_D1_D2": s1["stage_output"], "locked_D3_D5": s2["stage_output"]}

def prepare(args: argparse.Namespace) -> int:
    cases = read(args.blind)[:args.limit]
    p1, p2 = indexed(args.stage1), indexed(args.stage2)
    rows = []
    for case in cases:
        cid = str(case["case_id"])
        if args.stage >= 2 and cid not in p1 or args.stage == 3 and cid not in p2:
            raise SystemExit(f"missing prior stage for {cid}")
        payload = stage_payload(args.stage, case, p1.get(cid), p2.get(cid))
        rows.append({
            "custom_id": f"toulmin3:s{args.stage}:{cid}", "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": args.model, "messages": [
                {"role": "system", "content": SYSTEM[args.stage]},
                {"role": "user", "content": INSTRUCTION[args.stage] + "\n输入：" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ], "temperature": 0, "enable_thinking": False, "max_tokens": args.max_tokens},
        })
    write(args.output, rows)
    print(json.dumps({"version": VERSION, "stage": args.stage, "requests": len(rows), "model": args.model}, ensure_ascii=False))
    return 0

def ingest(args: argparse.Namespace) -> int:
    out, errors = [], []
    for row in read(args.batch_result):
        cid = str(row.get("custom_id", "")).split(":", 2)[-1]
        try:
            body = row["response"]["body"]
            raw = body["choices"][0]["message"]["content"].strip()
            value = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I))
            required = ("D1_claim", "D2_data") if args.stage == 1 else (("D3_warrant", "D4_backing", "D5_rebuttal") if args.stage == 2 else ("reasoning", "claim_results", "payment_result"))
            if not isinstance(value, dict) or not all(k in value for k in required):
                raise ValueError("required fields missing")
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
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare"); a.add_argument("--stage", type=int, choices=(1,2,3), required=True); a.add_argument("--blind", type=Path, required=True)
    a.add_argument("--stage1", type=Path); a.add_argument("--stage2", type=Path); a.add_argument("--output", type=Path, required=True)
    a.add_argument("--limit", type=int, default=50); a.add_argument("--model", default=MODEL); a.add_argument("--max-tokens", type=int, default=3500); a.set_defaults(handler=prepare)
    a = sub.add_parser("ingest"); a.add_argument("--stage", type=int, choices=(1,2,3), required=True); a.add_argument("--batch-result", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True); a.add_argument("--errors", type=Path, required=True); a.set_defaults(handler=ingest)
    return p

if __name__ == "__main__":
    ns = parser().parse_args(); raise SystemExit(ns.handler(ns))
