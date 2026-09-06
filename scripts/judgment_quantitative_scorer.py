#!/usr/bin/env python3
"""Deterministically score predicted claim/payment outcomes against QC references."""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def text(value: Any) -> str:
    return re.sub(r"\s+|[，。；：、,.!?！？()（）\[\]【】]", "", str(value or "")).lower()


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def amount_similarity(predicted: Any, gold: Any) -> float:
    p, g = number(predicted), number(gold)
    if p is None or g is None:
        return 1.0 if p is None and g is None else 0.0
    error = abs(p - g)
    if error <= 0.01:
        return 1.0
    return max(0.0, 1.0 - error / max(abs(g), 100.0))


def set_similarity(left: Any, right: Any) -> float:
    a = {text(v) for v in left or [] if text(v)}
    b = {text(v) for v in right or [] if text(v)}
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def claim_similarity(predicted: dict[str, Any], gold: dict[str, Any]) -> float:
    item = SequenceMatcher(None, text(predicted.get("item")), text(gold.get("item"))).ratio()
    request = SequenceMatcher(None, text(predicted.get("request_text")), text(gold.get("request_text"))).ratio()
    type_score = float(predicted.get("claim_type") == gold.get("claim_type"))
    requested = amount_similarity(predicted.get("requested_amount"), gold.get("requested_amount"))
    return 0.35 * item + 0.30 * request + 0.15 * type_score + 0.20 * requested


def greedy_match(predicted: list[dict[str, Any]], gold: list[dict[str, Any]], threshold: float = 0.35) -> list[tuple[int, int, float]]:
    candidates = sorted(
        ((claim_similarity(p, g), pi, gi) for pi, p in enumerate(predicted) for gi, g in enumerate(gold)),
        reverse=True,
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    result = []
    for score, pi, gi in candidates:
        if score < threshold or pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        result.append((pi, gi, score))
    return result


def f1(matches: int, predicted: int, gold: int) -> float:
    if predicted == gold == 0:
        return 1.0
    precision = matches / predicted if predicted else 0.0
    recall = matches / gold if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def obligation_similarity(predicted: dict[str, Any], gold: dict[str, Any], id_map: dict[str, str]) -> float:
    related = [id_map.get(str(v), str(v)) for v in predicted.get("related_claim_ids", [])]
    return mean([
        set_similarity(predicted.get("payer_names"), gold.get("payer_names")),
        set_similarity(predicted.get("payee_names"), gold.get("payee_names")),
        float(text(predicted.get("liability_mode")) == text(gold.get("liability_mode"))),
        amount_similarity(predicted.get("amount"), gold.get("amount")),
        set_similarity(related, gold.get("related_claim_ids")),
    ])


def score_case(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    output = candidate.get("model_output", candidate)
    gold_output = reference["gold_output"]
    predicted_claims = output.get("claim_results", []) if isinstance(output, dict) else []
    if not isinstance(predicted_claims, list):
        predicted_claims = []
    predicted_claims = [value for value in predicted_claims if isinstance(value, dict)]
    gold_claims = gold_output.get("claim_results", [])
    matches = greedy_match(predicted_claims, gold_claims)
    claim_f1 = f1(len(matches), len(predicted_claims), len(gold_claims))
    denom = max(len(gold_claims), 1)
    outcome = sum(predicted_claims[pi].get("outcome") == gold_claims[gi].get("outcome") for pi, gi, _ in matches) / denom
    operation = sum(predicted_claims[pi].get("operation") == gold_claims[gi].get("operation") for pi, gi, _ in matches) / denom
    award = sum(amount_similarity(predicted_claims[pi].get("awarded_amount"), gold_claims[gi].get("awarded_amount")) for pi, gi, _ in matches) / denom
    requested = sum(amount_similarity(predicted_claims[pi].get("requested_amount"), gold_claims[gi].get("requested_amount")) for pi, gi, _ in matches) / denom
    id_map = {str(predicted_claims[pi].get("claim_id")): str(gold_claims[gi].get("claim_id")) for pi, gi, _ in matches}
    pp = output.get("payment_result", {}) if isinstance(output, dict) else {}
    if not isinstance(pp, dict):
        pp = {}
    gp = gold_output.get("payment_result", {})
    payment_total = 0.5 * float(pp.get("has_payment") == gp.get("has_payment")) + 0.5 * amount_similarity(pp.get("total_awarded_amount"), gp.get("total_awarded_amount"))
    pred_ob = pp.get("obligations", []) if isinstance(pp, dict) else []
    gold_ob = gp.get("obligations", []) if isinstance(gp, dict) else []
    pairs = sorted(((obligation_similarity(p, g, id_map), pi, gi) for pi, p in enumerate(pred_ob) for gi, g in enumerate(gold_ob)), reverse=True)
    used_p: set[int] = set(); used_g: set[int] = set(); relation_sum = 0.0
    for similarity, pi, gi in pairs:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi); used_g.add(gi); relation_sum += similarity
    payment_relation = relation_sum / max(len(pred_ob), len(gold_ob), 1)
    components = {
        "claim_alignment": round(20 * claim_f1, 2),
        "outcome_classification": round(20 * outcome, 2),
        "awarded_amount": round(25 * award, 2),
        "operation": round(10 * operation, 2),
        "payment_total": round(15 * payment_total, 2),
        "payment_relation": round(10 * payment_relation, 2),
    }
    exact_awards = [abs((number(predicted_claims[pi].get("awarded_amount")) or 0) - (number(gold_claims[gi].get("awarded_amount")) or 0)) <= 0.01 for pi, gi, _ in matches if number(gold_claims[gi].get("awarded_amount")) is not None]
    return {
        "case_id": candidate.get("case_id"),
        "condition": candidate.get("condition"),
        "total_score": round(sum(components.values()), 2),
        "components": components,
        "raw_metrics": {
            "predicted_claims": len(predicted_claims), "gold_claims": len(gold_claims), "matched_claims": len(matches),
            "claim_f1": round(claim_f1, 4), "outcome_accuracy": round(outcome, 4), "requested_amount_accuracy": round(requested, 4),
            "awarded_amount_similarity": round(award, 4), "awarded_amount_exact_rate": round(mean(exact_awards), 4) if exact_awards else None,
            "operation_accuracy": round(operation, 4), "payment_total_similarity": round(payment_total, 4), "payment_relation_similarity": round(payment_relation, 4),
        },
        "claim_matches": [{"predicted_index": pi, "gold_index": gi, "similarity": round(score, 4)} for pi, gi, score in matches],
    }


def main(args: argparse.Namespace) -> int:
    references = {str(r["case_id"]): r for r in read_jsonl(args.references)}
    candidates = read_jsonl(args.candidates)
    missing = [r.get("case_id") for r in candidates if str(r.get("case_id")) not in references]
    if missing:
        raise SystemExit(f"missing references: {missing[:10]}")
    scores = [score_case(r, references[str(r["case_id"])]) for r in candidates]
    write_jsonl(args.output, scores)
    summary = {
        "schema_version": "lexchain-judgment-quantitative-score-v1",
        "cases": len(scores),
        "condition": scores[0].get("condition") if scores else None,
        "mean_total_score": round(mean(r["total_score"] for r in scores), 2) if scores else None,
        "mean_components": {key: round(mean(r["components"][key] for r in scores), 2) for key in next(iter(scores), {}).get("components", {})},
        "mean_raw_metrics": {key: round(mean(v), 4) for key in next(iter(scores), {}).get("raw_metrics", {}) for v in [[r["raw_metrics"][key] for r in scores if isinstance(r["raw_metrics"][key], (int, float))]] if v},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    return p


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
