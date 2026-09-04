#!/usr/bin/env python3
"""Prepare Qwen Batch Chat requests and restore Batch results to LexChain V1."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import lexchain_core_gold_api_v2 as api_v2
import lexchain_core_gold_local_v1 as local_v1


BATCH_URL = "/v1/chat/completions"
DEFAULT_LINE_LIMIT = 1_000_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(local_v1.iter_input_records(path))


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            target.write("\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(value, target, ensure_ascii=False, indent=2)
        target.write("\n")


def custom_id(queue_index: Any, case_id: Any) -> str:
    return f"lexchain:{int(queue_index)}:{local_v1.clean(case_id)}"


def parse_custom_id(value: Any) -> tuple[int, str]:
    text = local_v1.clean(value)
    prefix, queue, case_id = text.split(":", 2)
    if prefix != "lexchain" or not case_id:
        raise ValueError(f"invalid custom_id: {text}")
    return int(queue), case_id


def completed_case_ids(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        for record in read_jsonl(path):
            case_id = local_v1.clean(record.get("case_id"))
            if case_id:
                result.add(case_id)
    return result


def batch_request(
    record: dict[str, Any], model: str, enable_thinking: bool
) -> dict[str, Any]:
    body = json.loads(
        api_v2.build_request_body(model, api_v2.system_prompt(), record["payload"])
    )
    body["enable_thinking"] = enable_thinking
    return {
        "custom_id": custom_id(record["queue_index"], record["case_id"]),
        "method": "POST",
        "url": BATCH_URL,
        "body": body,
    }


def prepare(args: argparse.Namespace) -> int:
    done = completed_case_ids(args.completed)
    source = read_jsonl(args.input)
    pending = [r for r in source if local_v1.clean(r.get("case_id")) not in done]
    if args.start:
        pending = pending[args.start:]
    if args.limit is not None:
        pending = pending[:args.limit]
    requests = [batch_request(r, args.model, args.enable_thinking) for r in pending]
    ids = [r["custom_id"] for r in requests]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate custom_id in Batch input")
    encoded = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        for record in requests
    ]
    oversized = [ids[i] for i, line in enumerate(encoded) if len(line) > args.line_limit]
    if oversized:
        raise SystemExit(f"Batch lines exceed byte limit: {oversized[:10]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as target:
        for line in encoded:
            target.write(line + b"\n")
    report = {
        "schema_version": "lexchain-qwen-batch-input-v1",
        "source_records": len(source),
        "completed_case_ids": len(done),
        "batch_requests": len(requests),
        "model": args.model,
        "enable_thinking": args.enable_thinking,
        "prompt_version": api_v2.PROMPT_VERSION,
        "bytes": args.output.stat().st_size,
        "maximum_line_bytes": max(map(len, encoded), default=0),
        "url": BATCH_URL,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def batch_content(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    response = row.get("response")
    if not isinstance(response, dict) or response.get("status_code") != 200:
        raise ValueError("batch_response_not_successful")
    body = response.get("body")
    if not isinstance(body, dict):
        raise ValueError("batch_response_body_missing")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("batch_response_choices_missing")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("batch_response_content_missing")
    return content, body


def ingest(args: argparse.Namespace) -> int:
    sources = {
        (int(r["queue_index"]), local_v1.clean(r["case_id"])): api_v2.prepare_source_record(r, i)
        for i, r in enumerate(read_jsonl(args.source), 1)
    }
    passed: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for row in read_jsonl(args.batch_result):
        key = parse_custom_id(row.get("custom_id"))
        if key in seen:
            raise SystemExit(f"duplicate Batch result custom_id: {row.get('custom_id')}")
        seen.add(key)
        source = sources.get(key)
        if source is None:
            raise SystemExit(f"Batch result has no matching source: {row.get('custom_id')}")
        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        body = response.get("body") if isinstance(response.get("body"), dict) else {}
        audit = {
            "queue_index": key[0],
            "case_id": key[1],
            "status": "received",
            "transport": "qwen_batch_chat",
            "model": body.get("model"),
            "prompt_version": api_v2.PROMPT_VERSION,
            "request_id": response.get("request_id"),
            "usage": api_v2._normalized_usage(body.get("usage")),
            "semantic_calls": 1,
        }
        try:
            content, _ = batch_content(row)
            raw = api_v2.parse_json_content(content)
            result = api_v2.assemble_result(source, raw)
            result["quality_control"] = api_v2.validate_output(source, result)
        except Exception as exc:
            issue = f"batch_ingest_error:{type(exc).__name__}:{exc}"
            result = {
                "schema_version": api_v2.SCHEMA_VERSION,
                "queue_index": key[0],
                "case_id": key[1],
                "train_input": source.get("train_input"),
                "gold_output": {},
                "quality_control": api_v2._error_quality(issue),
            }
            audit["status"] = "error"
            audit["issue"] = issue
        audits.append(audit)
        (passed if result["quality_control"]["passed"] else review).append(result)

    write_jsonl(args.output, passed)
    write_jsonl(args.review_output, review)
    write_jsonl(args.audit_output, audits)
    issues = Counter()
    for result in review:
        issues.update(result["quality_control"].get("issues", []))
    report = {
        "schema_version": api_v2.SCHEMA_VERSION,
        "batch_results": len(seen),
        "qc_passed": len(passed),
        "review_required": len(review),
        "ingest_errors": sum(a["status"] == "error" for a in audits),
        "prompt_tokens": sum(a["usage"].get("prompt_tokens", 0) for a in audits),
        "completion_tokens": sum(a["usage"].get("completion_tokens", 0) for a in audits),
        "total_tokens": sum(a["usage"].get("total_tokens", 0) for a in audits),
        "top_qc_issues": dict(issues.most_common(20)),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--input", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--report", type=Path, required=True)
    prep.add_argument("--completed", type=Path, nargs="*", default=[])
    prep.add_argument("--model", default=api_v2.DEFAULT_MODEL)
    prep.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    prep.add_argument("--start", type=int, default=0)
    prep.add_argument("--limit", type=int)
    prep.add_argument("--line-limit", type=int, default=DEFAULT_LINE_LIMIT)
    prep.set_defaults(handler=prepare)

    restore = commands.add_parser("ingest")
    restore.add_argument("--source", type=Path, required=True)
    restore.add_argument("--batch-result", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    restore.add_argument("--review-output", type=Path, required=True)
    restore.add_argument("--audit-output", type=Path, required=True)
    restore.add_argument("--report", type=Path, required=True)
    restore.set_defaults(handler=ingest)
    return root


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "start", 0) < 0:
        raise SystemExit("--start must be non-negative")
    if getattr(args, "limit", None) is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
