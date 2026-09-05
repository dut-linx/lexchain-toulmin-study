#!/usr/bin/env python3
"""Prepare and ingest the seven-condition MAIN-1 Qwen Batch pilot."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


MODEL = "qwen3.7-max"
PROMPT_VERSION = "main1-seven-conditions-v1"
CONDITIONS = ("B0_DIRECT", "B1_COT", "B2_IRAC", "B3_SYLLOGISM", "B4_SCHEMA", "B5_LEXCHAIN", "T1_TOULMIN")
COMMON = (
    "你是审理中国民事侵权纠纷一审案件的法律分析者。只能使用给定材料，不得假设法院最终如何裁判，不得补造事实、证据、法条或金额。"
    "逐项回应原告的实体诉请；诉讼费等程序事项不作为实体诉请。输出必须是单个JSON对象且不得使用Markdown。"
    "无论采用何种分析方法，最终都必须给出reasoning、claim_results和payment_result三个顶层字段。"
    "claim_results中每项包含claim_id、claim_type、item、request_text、requested_amount、outcome、awarded_amount、operation、decision_reason；"
    "outcome仅可为full_support、partial_support、rejected、not_addressed，operation仅可为AWARD_MONEY、ORDER_ACT、DISMISS、NOT_ADDRESS。"
    "payment_result包含has_payment、total_awarded_amount、obligations；每项义务包含obligation_id、payer_names、payee_names、related_claim_ids、amount、liability_mode、reason。"
    "对无法从材料可靠确定的结果或金额，明确说明不确定性，不得把未知伪装成确定事实。"
)
INSTRUCTIONS = {
    "B0_DIRECT": "直接给出案件分析、逐项结论和付款关系。reasoning使用一段连贯文字，不套用指定推理框架。",
    "B1_COT": "公开展示逐步法律分析。reasoning按有序步骤说明识别诉请、核对事实、选择规则、适用规则、处理抗辩和形成结论的过程。",
    "B2_IRAC": "使用IRAC。reasoning按每项诉请分别给出Issue、Rule、Application、Conclusion，并说明主体、抗辩和救济。",
    "B3_SYLLOGISM": "使用法律三段论。reasoning按每项诉请分别给出规范大前提、事实小前提、涵摄过程和法律结论。",
    "B4_SCHEMA": "使用中性结构化表格思路，但不要使用IRAC、法律三段论、LexChain或图尔敏术语。reasoning按材料要点、规则要点、对应分析、反向检查、结论依据五个中性栏目组织。",
    "B5_LEXCHAIN": "使用贴近中国裁判文书的请求级法律推理链。reasoning逐项组织为诉请与主体、认定事实、裁判规范、要件涵摄、抗辩处理、责任与救济、结论。",
    "T1_TOULMIN": "使用完整图尔敏论证。reasoning对每项诉请分别给出Claim、Grounds、Warrant、Backing、Qualifier、Rebuttal和Conclusion，并明确事实如何经规则支持结论、抗辩如何影响责任和救济。",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def select_cases(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) < limit:
        raise ValueError(f"requested {limit} cases, found {len(rows)}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("stratum") or "unknown")].append(row)
    for values in groups.values():
        values.sort(key=lambda r: (int(r.get("queue_index") or 0), str(r.get("case_id"))))
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
        selected_ids = {str(r.get("case_id")) for r in selected}
        remainder = [r for r in rows if str(r.get("case_id")) not in selected_ids]
        remainder.sort(key=lambda r: (int(r.get("queue_index") or 0), str(r.get("case_id"))))
        selected.extend(remainder[: limit - len(selected)])
    return sorted(selected, key=lambda r: (int(r.get("queue_index") or 0), str(r.get("case_id"))))


def prepare(args: argparse.Namespace) -> int:
    cases = select_cases(read_jsonl(args.blind), args.limit)
    requests = []
    for case in cases:
        for condition in CONDITIONS:
            payload = {"case_id": case["case_id"], "case_material": case["model_input"]}
            requests.append({
                "custom_id": f"main1:{condition}:{case['case_id']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": COMMON + INSTRUCTIONS[condition]},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                    ],
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    "enable_thinking": False,
                },
            })
    ids = [r["custom_id"] for r in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate custom_id")
    write_jsonl(args.output, requests)
    report = {
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "cases": len(cases),
        "conditions": list(CONDITIONS),
        "requests": len(requests),
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "enable_thinking": False,
        "selected_case_ids": [c["case_id"] for c in cases],
        "strata": dict(Counter(c.get("stratum") for c in cases)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_content(text: str) -> Any:
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    return json.loads(value)


def ingest(args: argparse.Namespace) -> int:
    by_condition: dict[str, list[dict[str, Any]]] = {key: [] for key in CONDITIONS}
    errors: list[dict[str, Any]] = []
    usage = Counter()
    for row in read_jsonl(args.batch_result):
        custom_id = str(row.get("custom_id", ""))
        parts = custom_id.split(":", 2)
        try:
            if len(parts) != 3 or parts[0] != "main1" or parts[1] not in CONDITIONS:
                raise ValueError("invalid custom_id")
            response = row["response"]
            if response.get("status_code") != 200:
                raise ValueError(f"status {response.get('status_code')}")
            body = response["body"]
            content = body["choices"][0]["message"]["content"]
            parsed = parse_content(content)
            if not isinstance(parsed, dict) or not all(k in parsed for k in ("reasoning", "claim_results", "payment_result")):
                raise ValueError("required output fields missing")
            condition, case_id = parts[1], parts[2]
            by_condition[condition].append({
                "schema_version": "lexchain-main1-candidate-v1",
                "case_id": case_id,
                "condition": condition,
                "prompt_version": PROMPT_VERSION,
                "model": body.get("model"),
                "model_output": parsed,
            })
            usage.update(body.get("usage", {}))
        except Exception as exc:
            errors.append({"custom_id": custom_id, "error": f"{type(exc).__name__}: {exc}"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for condition, rows in by_condition.items():
        write_jsonl(args.output_dir / f"{condition}.jsonl", rows)
    write_jsonl(args.output_dir / "errors.jsonl", errors)
    report = {
        "rows": sum(len(v) for v in by_condition.values()) + len(errors),
        "successful": sum(len(v) for v in by_condition.values()),
        "errors": len(errors),
        "by_condition": {k: len(v) for k, v in by_condition.items()},
        "usage": dict(usage),
    }
    (args.output_dir / "ingest_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--blind", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--report", type=Path, required=True)
    prep.add_argument("--limit", type=int, default=50)
    prep.add_argument("--model", default=MODEL)
    prep.add_argument("--max-tokens", type=int, default=4500)
    prep.set_defaults(handler=prepare)
    restore = commands.add_parser("ingest")
    restore.add_argument("--batch-result", type=Path, required=True)
    restore.add_argument("--output-dir", type=Path, required=True)
    restore.set_defaults(handler=ingest)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    raise SystemExit(args.handler(args))
