#!/usr/bin/env python3
"""Merge split LexChain QC outputs into one source-ordered aligned JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected object")
                rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            target.write("\n")


def key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["queue_index"]), str(row["case_id"]).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--passed", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    passed = read_jsonl(args.passed)
    review = read_jsonl(args.review)
    combined = [*passed, *review]
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for row in combined:
        row_key = key(row)
        if row_key in by_key:
            raise SystemExit(f"duplicate result key: {row_key}")
        by_key[row_key] = row

    ordered: list[dict[str, Any]] = []
    source_seen: set[tuple[int, str]] = set()
    for row in read_jsonl(args.source):
        row_key = key(row)
        if row_key in source_seen:
            raise SystemExit(f"duplicate source key: {row_key}")
        source_seen.add(row_key)
        result = by_key.get(row_key)
        if result is not None:
            ordered.append(result)

    missing_from_source = set(by_key) - source_seen
    missing_results = set(by_key) - {key(row) for row in ordered}
    if missing_from_source or missing_results or len(ordered) != len(combined):
        raise SystemExit(
            "alignment failure: "
            f"ordered={len(ordered)} combined={len(combined)} "
            f"unknown={len(missing_from_source)} missing={len(missing_results)}"
        )

    write_jsonl(args.output, ordered)
    report = {
        "schema_version": "lexchain-reference-v1",
        "source_records": len(source_seen),
        "aligned_records": len(ordered),
        "qc_passed": len(passed),
        "review_required": len(review),
        "unique_keys": len(by_key),
        "duplicates": len(combined) - len(by_key),
        "missing_results": 0,
        "unknown_results": 0,
        "ordered_like_source": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
