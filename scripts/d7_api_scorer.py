#!/usr/bin/env python3
"""Build Qwen Batch judge requests and ingest D1-D6 percentage scores."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


BATCH_URL = "/v1/chat/completions"
DEFAULT_MODEL = "qwen3.7-max"
RUBRIC_VERSION = "d7-six-dimensions-percent-v1"
DIMENSIONS = {
    "D1": "诉请、主体与决定性争点覆盖",
    "D2": "责任规范选择、法源适格与解释",
    "D3": "事实—规范—法律效果连接",
    "D4": "抗辩、例外与责任限缩处理",
    "D5": "请求级结论、责任分配与救济证成",
    "D6": "整体公共证立与可审查性",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def index_by_case(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id or case_id in result:
            raise ValueError(f"{label}: missing or duplicate case_id: {case_id}")
        result[case_id] = row
    return result


def candidate_value(row: dict[str, Any]) -> Any:
    for key in ("model_output", "candidate_output", "prediction", "output"):
        if key in row:
            return row[key]
    raise ValueError(f"candidate has no output field: {row.get('case_id')}")


def system_prompt() -> str:
    return (
        "你是中国民事侵权纠纷一审裁判说理的盲评员。严格依据给定案件材料、QC参考答案和待评模型输出评分。"
        "不要猜测材料外事实，不因文字长或术语多而加分。六个维度各自独立评分，直接输出0至100整数。"
        "锚点换算：0分对应原手册0级，33分对应1级，67分对应2级，100分对应3级；可在锚点间取整数。"
        "D4仅在材料确无实质抗辩时输出applicable=false且score=null；不能因模型遗漏抗辩而标N/A。"
        "每个维度必须给出简短理由、案件材料中的证据和待评输出中的证据；找不到证据时使用空字符串。"
        "严重错误上限：D2出现虚构、失效或明显不适格法源时D2不得高于25；D3依赖材料外核心事实时D3不得高于25；"
        "D5与QC参考答案在主要诉请结论、责任主体、责任形态或金额上实质冲突时D5不得高于25。"
        "只输出一个JSON对象，不要Markdown。"
    )


def rubric() -> dict[str, Any]:
    return {
        "D1": "逐项核对诉请、权利主体、责任主体和决定性争点。完整清晰=100；仅次要遗漏=67；遗漏一个主要项=33；多项遗漏、错配或超裁=0。",
        "D2": "核对责任规范、一般法与特别法、法源真实性有效性及必要解释。全部正确充分=100；轻微瑕疵=67；重要规范或解释缺口=33；归责基础错误或虚构失效法源=0。",
        "D3": "核对证据—事实—要件—法律效果链。完整可追溯=100；次要环节可补足=67；决定性事实或核心涵摄缺失=33；核心事实臆造或结论无法推出=0。",
        "D4": "仅评价材料中实际存在的抗辩、例外和限责。全部实质回应且效果准确=100；次要缺口=67；遗漏决定性抗辩或门槛混淆=33；整体不回应或错误适用=0；确无实质抗辩=N/A。",
        "D5": "逐项核对结论理由、责任主体/形态/份额和金钱或非金钱救济。准确充分可执行=100；次要说明不足=67；主要请求仅列结论或责任救济含混=33；多项无理由、明显错误或不可执行=0。",
        "D6": "核对公开可理解、整体融贯、路径清晰、繁简适度和可复核性。完全达到=100；少量重复跳转=67；需费力重构或依赖口号=33；核心矛盾或结构严重混乱=0。",
    }


def build_payload(blind: dict[str, Any], reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "rubric_version": RUBRIC_VERSION,
        "scoring_rubric": rubric(),
        "case_material": blind["model_input"],
        "qc_reference_answer": reference["gold_output"],
        "candidate_to_score": candidate_value(candidate),
        "required_output": {
            "case_id": blind["case_id"],
            "dimensions": {
                key: {"applicable": "boolean", "score": "integer 0..100 or null", "reason": "string", "case_evidence": "string", "candidate_evidence": "string"}
                for key in DIMENSIONS
            },
            "fatal_errors": ["string"],
        },
    }


def prepare(args: argparse.Namespace) -> int:
    blind = index_by_case(read_jsonl(args.blind), "blind")
    references = index_by_case(read_jsonl(args.references), "references")
    candidates = index_by_case(read_jsonl(args.candidates), "candidates")
    ids = list(candidates)
    missing = [case_id for case_id in ids if case_id not in blind or case_id not in references]
    if missing:
        raise SystemExit(f"candidate cases missing blind/reference rows: {missing[:10]}")
    requests = []
    for case_id in ids:
        payload = build_payload(blind[case_id], references[case_id], candidates[case_id])
        requests.append({
            "custom_id": f"d7score:{case_id}",
            "method": "POST",
            "url": BATCH_URL,
            "body": {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                ],
                "temperature": 0,
                "enable_thinking": False,
            },
        })
    write_jsonl(args.output, requests)
    print(json.dumps({"requests": len(requests), "model": args.model, "rubric_version": RUBRIC_VERSION}, ensure_ascii=False))
    return 0


def parse_json_text(text: str) -> dict[str, Any]:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("judge output is not an object")
    return parsed


def response_content(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    response = row.get("response", {})
    if response.get("status_code") != 200:
        raise ValueError(f"non-200 response: {response.get('status_code')}")
    body = response.get("body", {})
    return body["choices"][0]["message"]["content"], body


def normalize_score(raw: dict[str, Any], case_id: str) -> dict[str, Any]:
    dimensions = raw.get("dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError("dimensions missing")
    normalized: dict[str, Any] = {}
    for key, name in DIMENSIONS.items():
        item = dimensions.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"{key} missing")
        applicable = item.get("applicable") is not False
        score = item.get("score")
        if not applicable:
            if key != "D4":
                raise ValueError(f"only D4 may be N/A, got {key}")
            score = None
        elif isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
            raise ValueError(f"{key} score outside 0..100")
        normalized[key] = {
            "name": name,
            "applicable": applicable,
            "score": None if score is None else int(round(score)),
            "reason": str(item.get("reason", "")).strip(),
            "case_evidence": str(item.get("case_evidence", "")).strip(),
            "candidate_evidence": str(item.get("candidate_evidence", "")).strip(),
        }
    substantive = [normalized[k]["score"] for k in ("D1", "D2", "D3", "D4", "D5") if normalized[k]["score"] is not None]
    overall = [normalized[k]["score"] for k in DIMENSIONS if normalized[k]["score"] is not None]
    return {
        "schema_version": "lexchain-d7-api-score-v1",
        "case_id": case_id,
        "rubric_version": RUBRIC_VERSION,
        "dimensions": normalized,
        "substantive_score": round(mean(substantive), 2),
        "overall_score": round(mean(overall), 2),
        "fatal_errors": [str(v) for v in raw.get("fatal_errors", [])] if isinstance(raw.get("fatal_errors", []), list) else [],
    }


def ingest(args: argparse.Namespace) -> int:
    scores: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in read_jsonl(args.batch_result):
        custom_id = str(row.get("custom_id", ""))
        case_id = custom_id.split(":", 1)[1] if custom_id.startswith("d7score:") else ""
        try:
            content, body = response_content(row)
            score = normalize_score(parse_json_text(content), case_id)
            score["model"] = body.get("model")
            score["usage"] = body.get("usage", {})
            scores.append(score)
        except Exception as exc:
            errors.append({"custom_id": custom_id, "case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})
    write_jsonl(args.output, scores)
    write_jsonl(args.errors, errors)
    report = {
        "batch_rows": len(scores) + len(errors),
        "scored": len(scores),
        "errors": len(errors),
        "mean_substantive_score": round(mean(s["substantive_score"] for s in scores), 2) if scores else None,
        "mean_overall_score": round(mean(s["overall_score"] for s in scores), 2) if scores else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--blind", type=Path, required=True)
    prep.add_argument("--references", type=Path, required=True)
    prep.add_argument("--candidates", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--model", default=DEFAULT_MODEL)
    prep.set_defaults(handler=prepare)
    restore = commands.add_parser("ingest")
    restore.add_argument("--batch-result", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    restore.add_argument("--errors", type=Path, required=True)
    restore.add_argument("--report", type=Path, required=True)
    restore.set_defaults(handler=ingest)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    raise SystemExit(args.handler(args))
