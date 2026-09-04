#!/usr/bin/env python3
"""Freeze QC-passed references and create leakage-safe development/test views."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ALLOWED_INPUT_FIELDS = (
    "case_type",
    "plaintiff_statement",
    "defendant_statement",
    "court_found_facts",
    "law_candidates",
)
FORBIDDEN_KEYS = {
    "court_reasoning",
    "judgment_result",
    "gold_output",
    "quality_control",
    "decision_text",
    "ruling_text",
    "outcome",
    "awarded_amount",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(value, target, ensure_ascii=False, indent=2)
        target.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split())


def complexity_stratum(record: dict[str, Any]) -> str:
    case_type = clean(record.get("train_input", {}).get("case_type"))
    gold = record.get("gold_output", {})
    claims = gold.get("claim_results", []) if isinstance(gold, dict) else []
    payment = gold.get("payment_result", {}) if isinstance(gold, dict) else {}
    obligations = payment.get("obligations", []) if isinstance(payment, dict) else []
    monetary = [c for c in claims if c.get("claim_type") == "monetary"]
    non_monetary = [c for c in claims if c.get("claim_type") == "non_monetary"]
    payer_count = max(
        (len(o.get("payer_names", [])) for o in obligations if isinstance(o, dict)),
        default=0,
    )
    liability_modes = {
        clean(o.get("liability_mode")) for o in obligations if isinstance(o, dict)
    }
    if "保险" in case_type:
        return "insurance"
    if payer_count > 1 or any(m not in {"", "individual"} for m in liability_modes):
        return "multi_party_liability"
    if len(obligations) > 1:
        return "multiple_payment_obligations"
    if any(len(o.get("related_claim_ids", [])) > 1 for o in obligations if isinstance(o, dict)):
        return "combined_award"
    if any(word in case_type for word in ("著作权", "商标", "专利", "知识产权")) and non_monetary:
        return "ip_non_monetary_relief"
    if len(monetary) > 1:
        return "multiple_monetary_items"
    return "ordinary_single_relief"


def allocate_dev(rows: list[dict[str, Any]], dev_size: int, seed: int) -> set[str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[complexity_stratum(row)].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        values.sort(key=lambda r: clean(r.get("case_id")))
        rng.shuffle(values)
    exact = {key: dev_size * len(values) / len(rows) for key, values in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = dev_size - sum(quotas.values())
    order = sorted(groups, key=lambda key: (exact[key] - quotas[key], len(groups[key]), key), reverse=True)
    for key in order[:remaining]:
        quotas[key] += 1
    return {
        clean(row.get("case_id"))
        for key, values in groups.items()
        for row in values[: quotas[key]]
    }


def assert_no_leakage(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden leakage key at {path}.{key}")
            assert_no_leakage(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_leakage(child, f"{path}[{index}]")


def prepare(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit("input is empty")
    ids = [clean(row.get("case_id")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise SystemExit("case_id must be present and unique")
    invalid_qc = [case_id for case_id, row in zip(ids, rows) if row.get("quality_control", {}).get("passed") is not True]
    if invalid_qc:
        raise SystemExit(f"input contains non-passed QC rows: {invalid_qc[:5]}")
    if args.dev_size < 1 or args.dev_size >= len(rows):
        raise SystemExit("--dev-size must be between 1 and record_count-1")

    dev_ids = allocate_dev(rows, args.dev_size, args.seed)
    blind: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for row in rows:
        case_id = clean(row["case_id"])
        split = "development" if case_id in dev_ids else "test"
        source_input = row.get("train_input", {})
        model_input = {key: source_input.get(key) for key in ALLOWED_INPUT_FIELDS}
        blind_row = {
            "schema_version": "lexchain-experiment-blind-v1",
            "queue_index": row.get("queue_index"),
            "case_id": case_id,
            "split": split,
            "stratum": complexity_stratum(row),
            "model_input": model_input,
        }
        assert_no_leakage(blind_row)
        blind.append(blind_row)
        references.append({
            "schema_version": "lexchain-experiment-reference-v1",
            "queue_index": row.get("queue_index"),
            "case_id": case_id,
            "split": split,
            "stratum": blind_row["stratum"],
            "gold_output": row["gold_output"],
        })
        split_rows.append({"case_id": case_id, "split": split, "stratum": blind_row["stratum"]})

    prefix = args.output_dir
    paths = {
        "blind_all": prefix / "blind_all.jsonl",
        "blind_development": prefix / "blind_development.jsonl",
        "blind_test": prefix / "blind_test.jsonl",
        "reference_all": prefix / "reference_all.jsonl",
        "split_manifest": prefix / "split_manifest.jsonl",
    }
    write_jsonl(paths["blind_all"], blind)
    write_jsonl(paths["blind_development"], (r for r in blind if r["split"] == "development"))
    write_jsonl(paths["blind_test"], (r for r in blind if r["split"] == "test"))
    write_jsonl(paths["reference_all"], references)
    write_jsonl(paths["split_manifest"], split_rows)
    report = {
        "schema_version": "lexchain-experiment-freeze-v1",
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "seed": args.seed,
        "records": len(rows),
        "development": len(dev_ids),
        "test": len(rows) - len(dev_ids),
        "strata_all": dict(sorted(Counter(r["stratum"] for r in blind).items())),
        "strata_development": dict(sorted(Counter(r["stratum"] for r in blind if r["split"] == "development").items())),
        "allowed_model_input_fields": list(ALLOWED_INPUT_FIELDS),
        "artifacts": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()},
    }
    write_json(prefix / "freeze_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--dev-size", type=int, default=300)
    result.add_argument("--seed", type=int, default=20260905)
    return result


if __name__ == "__main__":
    raise SystemExit(prepare(parser().parse_args()))
