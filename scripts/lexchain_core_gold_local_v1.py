#!/usr/bin/env python3
"""Network-free, high-precision CoreGold extraction for Chinese judgments."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "lexchain-reference-v1"
METHOD = "python_rules_v1"
PROCEDURAL_COST_PATTERN = re.compile(
    r"诉讼费|案件受理费|受理费|保全费|申请费|公告费|送达费|鉴定费"
)
MONEY_ITEMS = (
    "住院伙食补助费",
    "精神损害抚慰金",
    "精神抚慰金",
    "残疾赔偿金",
    "伤残赔偿金",
    "死亡赔偿金",
    "被扶养人生活费",
    "医疗费",
    "医药费",
    "误工费",
    "护理费",
    "交通费",
    "营养费",
    "住宿费",
    "伙食费",
    "后续治疗费",
    "康复费",
    "丧葬费",
    "财产损失",
    "经济损失",
    "衣物损失",
    "车辆维修费",
    "维权合理支出",
    "合理支出",
    "律师费",
    "复印费",
    "伙食补助费",
    "维修费",
    "修理费",
    "拖车费",
    "施救费",
    "停运损失",
    "租金",
    "货款",
    "借款本金",
    "本金",
    "利息",
    "违约金",
    "赔偿款",
)
MONEY_ITEM_PATTERN = re.compile(
    rf"(?P<item>{'|'.join(map(re.escape, sorted(MONEY_ITEMS, key=len, reverse=True)))})"
    r"\s*(?:人民币)?(?P<amount>\d[\d,，]*(?:\.\d+)?)\s*元"
)
NON_MONEY_PATTERNS = (
    ("删除侵权内容", re.compile(r"删除[^，,；;。]{0,30}(?:图片|内容|文章|作品|链接|信息)")),
    (
        "提供用户身份信息",
        re.compile(r"提供[^，,；;。]{0,30}(?:真实身份信息|身份信息|联系方式|地址)"),
    ),
    ("停止侵害", re.compile(r"停止(?:侵害|侵权)")),
    ("返还", re.compile(r"返还(?:原告)?(?P<object>[^，,；;。]{1,20})")),
    ("继续履行", re.compile(r"继续履行(?:[^，,；;。]{0,20})")),
    ("解除合同", re.compile(r"解除(?:双方签订的)?[^，,；;。]{0,16}合同")),
    ("确认无效", re.compile(r"确认[^，,；;。]{0,20}(?:无效|不成立)")),
    ("赔礼道歉", re.compile(r"赔礼道歉")),
    ("消除影响", re.compile(r"消除影响")),
    ("恢复原状", re.compile(r"恢复原状")),
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\u3000", " ").split())


def nested_get(obj: Any, *path: str) -> Any:
    current = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_text(*values: Any) -> str:
    for value in values:
        text = clean(value)
        if text:
            return text
    return ""


def compact_case_type(value: Any) -> str:
    if isinstance(value, list):
        values = [clean(item) for item in value if clean(item)]
        return values[-1] if values else ""
    return clean(value)


IDENTIFIED_INLINE_FACT_MARKERS = (
    "本院经审理查明",
    "经审理查明",
    "本院查明",
    "经查明",
    "另查明",
    "经本院核实",
)
IDENTIFIED_FACT_BLOCK_PATTERN = re.compile(
    r"(?:^|[\n\t。！？；])\s*((?:本院)?(?:经审理)?查明(?:事实)?如下[：:])"
)
IDENTIFIED_REASONING_BOUNDARIES = ("本院认为", "综上", "依照")


def extract_facts_from_identified(value: Any) -> str:
    text = str(value or "")
    block = IDENTIFIED_FACT_BLOCK_PATTERN.search(text)
    if block:
        end = len(text)
        for boundary in IDENTIFIED_REASONING_BOUNDARIES:
            position = text.find(boundary, block.end())
            if position >= 0:
                end = min(end, position)
        return clean(text[block.start(1):end])
    starts: list[int] = []
    for marker in IDENTIFIED_INLINE_FACT_MARKERS:
        match = re.search(
            rf"(?:^|[\n\t。！？；])\s*({re.escape(marker)})",
            text,
        )
        if match:
            starts.append(match.start(1))
    if not starts:
        return ""
    start = min(starts)
    ending = re.search(r"[。！？\n]", text[start:])
    end = start + ending.end() if ending else len(text)
    return clean(text[start:end])


def _evidence(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = nested_get(case, "source_evidence", "evidence_items")
    if not isinstance(items, list):
        return {}
    return {
        clean(item.get("source_id")): item
        for item in items
        if isinstance(item, dict) and clean(item.get("source_id"))
    }


def _law_items(case: dict[str, Any], supplied: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = (
        supplied.get("law_candidates"),
        case.get("law_candidates"),
        nested_get(case, "model_input", "A_court_found_facts_only", "given_sources", "items"),
        nested_get(case, "model_input", "B_pleading_aware", "given_sources", "items"),
    )
    for value in candidates:
        if isinstance(value, list) and value:
            return [item for item in value if isinstance(item, dict)]
    return []


def _law_entry(item: dict[str, Any]) -> dict[str, str] | None:
    law_id = first_text(item.get("law_id"), item.get("source_id"), item.get("id"))
    citation = first_text(item.get("citation"), item.get("label"))
    if not law_id or not citation:
        return None
    return {
        "law_id": law_id,
        "law_title": first_text(item.get("law_title"), item.get("source_title")),
        "article": first_text(item.get("article"), item.get("article_number")),
        "citation": citation,
        "full_text": first_text(item.get("full_text"), item.get("text")),
        "timeliness": clean(item.get("timeliness")),
        "source_url": first_text(item.get("source_url"), item.get("url")),
    }


def build_source_record(case: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    supplied = case.get("train_input") if isinstance(case.get("train_input"), dict) else case
    evidence = _evidence(case)
    metadata = case.get("case_metadata") if isinstance(case.get("case_metadata"), dict) else {}
    pleading = nested_get(case, "model_input", "B_pleading_aware") or {}
    facts_only = nested_get(case, "model_input", "A_court_found_facts_only") or {}
    entries = [
        entry
        for item in _law_items(case, supplied)
        if (entry := _law_entry(item)) is not None
    ]
    train_input = {
        "case_type": first_text(
            supplied.get("case_type"), case.get("case_type"), compact_case_type(metadata.get("category"))
        ),
        "plaintiff_statement": first_text(
            supplied.get("plaintiff_statement"),
            evidence.get("E-PlaintiffClaims", {}).get("raw_span"),
            nested_get(pleading, "plaintiff_claims", "text"),
        ),
        "defendant_statement": first_text(
            supplied.get("defendant_statement"),
            evidence.get("E-DefenseViewpoint", {}).get("raw_span"),
            nested_get(pleading, "defense_viewpoint", "text"),
        ),
        "court_found_facts": first_text(
            supplied.get("court_found_facts"),
            evidence.get("E-Ascertain", {}).get("raw_span"),
            nested_get(pleading, "facts", "text"),
            nested_get(facts_only, "facts", "text"),
            extract_facts_from_identified(
                evidence.get("E-Identified", {}).get("raw_span")
            ),
        ),
        "law_candidates": [
            {"law_id": entry["law_id"], "citation": entry["citation"]}
            for entry in entries
        ],
    }
    return {
        "queue_index": case.get("queue_index", fallback_index),
        "case_id": first_text(case.get("case_id"), metadata.get("case_id")),
        "train_input": train_input,
        "court_reasoning": first_text(
            case.get("court_reasoning"),
            evidence.get("E-Identified", {}).get("raw_span"),
            evidence.get("E-ControversialFocus", {}).get("raw_span"),
        ),
        "judgment_result": first_text(
            case.get("judgment_result"),
            evidence.get("E-RefereeResult", {}).get("raw_span"),
        ),
        "law_catalog_entries": entries,
    }


def collect_law_catalog(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    by_id: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for record in records:
        entries = record.get("law_catalog_entries")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            law_id = clean(entry.get("law_id"))
            if not law_id:
                continue
            if law_id not in by_id:
                by_id[law_id] = entry
            elif by_id[law_id] != entry:
                issue = f"law_catalog_conflict_{law_id}"
                if issue not in issues:
                    issues.append(issue)
    for law_id in sorted(by_id):
        if not clean(by_id[law_id].get("full_text")):
            issues.append(f"law_catalog_missing_full_text_{law_id}")
    return [by_id[law_id] for law_id in sorted(by_id)], issues


def _request_section(statement: str) -> str:
    text = clean(statement)
    marker = re.search(r"诉讼请求\s*[:：]", text)
    start = marker.end() if marker else 0
    tail = text[start:]
    boundary = re.search(r"(?:事实和理由|事实与理由|事实及理由)\s*[:：]", tail)
    return tail[: boundary.start()] if boundary else tail


def _amount(value: str) -> float:
    return float(value.replace(",", "").replace("，", ""))


def extract_claims(
    plaintiff_statement: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    section = _request_section(plaintiff_statement)
    claims: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for match in MONEY_ITEM_PATTERN.finditer(section):
        item = match.group("item")
        if PROCEDURAL_COST_PATTERN.search(item):
            continue
        claims.append(
            {
                "claim_type": "monetary",
                "item": item,
                "request_text": match.group(0),
                "requested_amount": _amount(match.group("amount")),
                "source_order": match.start(),
            }
        )
        occupied.append(match.span())

    for label, pattern in NON_MONEY_PATTERNS:
        for match in pattern.finditer(section):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            matched = match.group(0)
            if label == "返还":
                item = matched
            else:
                item = label
            claims.append(
                {
                    "claim_type": "non_monetary",
                    "item": item,
                    "request_text": matched,
                    "requested_amount": None,
                    "source_order": match.start(),
                }
            )

    claims.sort(key=lambda item: item["source_order"])
    issues: list[str] = []
    if not claims and re.search(r"\d[\d,，]*(?:\.\d+)?\s*元", section):
        issues.append("claim_split_ambiguous")
    elif not claims and section and not PROCEDURAL_COST_PATTERN.fullmatch(section):
        issues.append("claim_split_ambiguous")
    return claims, issues


def _clauses(text: str) -> list[str]:
    return [clean(part) for part in re.split(r"[；;。！？\n]+", text) if clean(part)]


def _claim_ids(claims: list[dict[str, Any]]) -> dict[int, str]:
    result: dict[int, str] = {}
    money = 0
    non_money = 0
    for index, claim in enumerate(claims):
        if claim.get("claim_type") == "monetary":
            money += 1
            result[index] = f"M{money}"
        else:
            non_money += 1
            result[index] = f"N{non_money}"
    return result


def _residual_dismissal_is_unambiguous(
    residual_clause: str,
    court_reasoning: str,
    judgment_result: str,
) -> bool:
    residual_start = judgment_result.find(residual_clause)
    judgment_prefix = (
        judgment_result[:residual_start] if residual_start >= 0 else judgment_result
    )
    relevant_text = f"{court_reasoning}\n{judgment_prefix}"
    if re.search(r"撤回|撤诉|反诉", relevant_text):
        return False
    return True


def _money_after_item(clause: str, item: str) -> float | None:
    match = re.search(
        rf"{re.escape(item)}[^，,；;。]{{0,24}}?(\d[\d,，]*(?:\.\d+)?)\s*元",
        clause,
    )
    return _amount(match.group(1)) if match else None


def match_claim_results(
    claims: list[dict[str, Any]],
    court_reasoning: str,
    judgment_result: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    ids = _claim_ids(claims)
    court_clauses = _clauses(court_reasoning)
    judgment_clauses = _clauses(judgment_result)
    results: list[dict[str, Any]] = []
    unresolved: list[int] = []
    for index, claim in enumerate(claims):
        item = clean(claim.get("item"))
        candidates = [clause for clause in court_clauses if item and item in clause]
        candidates.extend(
            clause for clause in judgment_clauses if item and item in clause
        )
        outcome = "not_addressed"
        awarded: float | None = None
        operation = "NOT_ADDRESSED"
        decision_text = ""
        rejected = next(
            (
                clause
                for clause in reversed(candidates)
                if re.search(r"不予支持|不予采纳|予以驳回|驳回", clause)
            ),
            None,
        )
        withdrawn = next(
            (
                clause
                for clause in reversed(candidates)
                if re.search(r"撤回|撤诉", clause)
            ),
            None,
        )
        if rejected:
            outcome = "rejected"
            awarded = 0.0 if claim.get("claim_type") == "monetary" else None
            operation = "DISMISS"
            decision_text = rejected
        elif withdrawn:
            outcome = "withdrawn"
            operation = "WITHDRAWN"
            decision_text = withdrawn
        elif claim.get("claim_type") == "monetary":
            amount_candidate: tuple[str, float] | None = None
            for clause in reversed(candidates):
                if clause not in judgment_clauses and not re.search(
                    r"本院|法院|认定|支持|酌定|应赔|判令|判决", clause
                ):
                    continue
                value = _money_after_item(clause, item)
                if value is not None:
                    amount_candidate = (clause, value)
                    break
            if amount_candidate is not None:
                decision_text, awarded = amount_candidate
                requested = claim.get("requested_amount")
                if isinstance(requested, (int, float)) and abs(awarded - float(requested)) <= 0.01:
                    outcome = "full_support"
                elif isinstance(requested, (int, float)) and 0 <= awarded < float(requested):
                    outcome = "partial_support"
                else:
                    outcome = "not_addressed"
                    awarded = None
                    decision_text = ""
                if outcome != "not_addressed":
                    operation = "AWARD_MONEY"
        else:
            positive = next(
                (
                    clause
                    for clause in reversed(candidates)
                    if re.search(r"判令|应当|应予|予以支持|停止|返还|继续履行|解除|确认", clause)
                ),
                None,
            )
            if positive:
                outcome = "full_support"
                operation = "ORDER_PERFORMANCE"
                decision_text = positive

        if outcome == "not_addressed":
            unresolved.append(index)
        results.append(
            {
                "claim_id": ids[index],
                "claim_type": claim.get("claim_type"),
                "item": item,
                "request_text": clean(claim.get("request_text")),
                "requested_amount": claim.get("requested_amount"),
                "outcome": outcome,
                "awarded_amount": awarded,
                "operation": operation,
                "decision_text": decision_text,
            }
        )

    issues: list[str] = []
    residual_clause = next(
        (
            clause
            for clause in judgment_clauses
            if re.search(r"驳回.+其他诉讼请求", clause)
        ),
        None,
    )
    if residual_clause and unresolved and _residual_dismissal_is_unambiguous(
        residual_clause, court_reasoning, judgment_result
    ):
        for index in unresolved:
            item = results[index]
            item["outcome"] = "rejected"
            item["awarded_amount"] = (
                0.0 if item["claim_type"] == "monetary" else None
            )
            item["operation"] = "DISMISS"
            item["decision_text"] = residual_clause
        unresolved.clear()
    elif residual_clause and unresolved:
        issues.append("residual_dismissal_ambiguous")
    if unresolved and "residual_dismissal_ambiguous" not in issues:
        issues.append("claim_outcome_ambiguous")
    return results, issues


def _party_name(value: str) -> str:
    text = clean(value)
    text = re.sub(r"^(?:被告|原告|第三人|保险人)", "", text)
    return text.strip("，,、 ")


def _claims_named_in_clause(
    claims: list[dict[str, Any]], clause: str
) -> list[dict[str, Any]]:
    return [
        item
        for item in claims
        if clean(item.get("item")) and clean(item.get("item")) in clause
    ]


def _party_names(value: str, liability_mode: str) -> list[str]:
    name = _party_name(value)
    if liability_mode not in {"joint", "several"}:
        return [name]
    parts = [clean(part) for part in re.split(r"[、，]|和|与", name) if clean(part)]
    return parts or [name]


def extract_payment_result(
    claim_results: list[dict[str, Any]],
    judgment_result: str,
) -> tuple[dict[str, Any], list[str]]:
    positive = [
        item
        for item in claim_results
        if item.get("claim_type") == "monetary"
        and item.get("outcome") in {"full_support", "partial_support"}
        and isinstance(item.get("awarded_amount"), (int, float))
    ]
    clauses = [
        re.sub(r"^[一二三四五六七八九十\d]+[、.]\s*", "", clause)
        for clause in _clauses(judgment_result)
        if not PROCEDURAL_COST_PATTERN.search(clause)
        and not re.search(r"迟延履行|执行通知|高消费|失信名单", clause)
    ]
    obligations: list[dict[str, Any]] = []
    issues: list[str] = []
    for clause in clauses:
        if not re.search(r"赔偿|支付|给付", clause):
            continue
        amount_match = re.search(r"(\d[\d,，]*(?:\.\d+)?)\s*元", clause)
        payer_match = re.search(
            r"^(?:由)?(?:被告|第三人|保险人)?(?P<name>.+?)(?=(?:于|在|自)本判决|向(?:原告|申请人)|赔偿|支付|给付)",
            clause,
        )
        payee_markers = "|".join(
            map(re.escape, sorted((*MONEY_ITEMS, "损失", "费用", "广告牌制作费用"), key=len, reverse=True))
        )
        payee_match = re.search(
            rf"(?:赔偿|支付|给付)(?:给)?(?:原告|申请人)?(?P<name>.+?)(?=的?(?:相关|各项|{payee_markers})|人民币|\d)",
            clause,
        )
        if amount_match is None:
            continue
        if payer_match is None or payee_match is None:
            issues.append("payment_party_ambiguous")
            continue
        if "连带" in clause:
            mode = "joint"
        elif "按份" in clause or "分别" in clause or "各自" in clause:
            mode = "several"
        elif "补充责任" in clause or "补充赔偿" in clause:
            mode = "supplementary"
        else:
            mode = "individual"
        linked_claims = _claims_named_in_clause(positive, clause)
        obligations.append(
            {
                "obligation_id": f"O{len(obligations) + 1}",
                "payer_names": _party_names(payer_match.group("name"), mode),
                "payee_names": [_party_name(payee_match.group("name"))],
                "related_claim_ids": [item["claim_id"] for item in linked_claims],
                "amount": _amount(amount_match.group(1)),
                "liability_mode": mode,
                "ruling_text": clause,
            }
        )
    if not obligations:
        if positive:
            issues.append("payment_obligation_missing")
        return {
            "has_payment": False,
            "total_awarded_amount": None,
            "obligations": [],
        }, list(dict.fromkeys(issues))
    if len(obligations) == 1 and not obligations[0]["related_claim_ids"]:
        obligations[0]["related_claim_ids"] = [item["claim_id"] for item in positive]

    linked_ids: list[str] = []
    for obligation in obligations:
        related_ids = obligation["related_claim_ids"]
        if not related_ids:
            issues.append("payment_claim_link_ambiguous")
            continue
        if any(claim_id in linked_ids for claim_id in related_ids):
            issues.append("complex_multi_party_liability")
            continue
        linked_ids.extend(related_ids)
        linked = [item for item in positive if item["claim_id"] in related_ids]
        component_total = sum(float(item["awarded_amount"]) for item in linked)
        if abs(component_total - float(obligation["amount"])) > 0.01:
            issues.append("payment_total_mismatch")

    total = (
        sum(float(item["amount"]) for item in obligations)
        if set(linked_ids) == {item["claim_id"] for item in positive}
        else None
    )
    if set(linked_ids) != {item["claim_id"] for item in positive}:
        issues.append("payment_claim_link_ambiguous")
    return {
        "has_payment": True,
        "total_awarded_amount": total,
        "obligations": obligations,
    }, list(dict.fromkeys(issues))


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _grounded(value: Any, sources: Iterable[Any]) -> bool:
    text = clean(value)
    return bool(text) and any(text in clean(source) for source in sources)


def validate_local_result(
    source_record: dict[str, Any],
    result: dict[str, Any],
    catalog_ids_with_text: set[str],
) -> dict[str, Any]:
    issues = [clean(item) for item in result.get("_rule_issues", []) if clean(item)]
    train_input = result.get("train_input") if isinstance(result.get("train_input"), dict) else {}
    gold = result.get("gold_output")
    if not isinstance(gold, dict):
        gold = result.get("partial_gold_output")
    gold = gold if isinstance(gold, dict) else {}
    claims = _dict_list(gold.get("claim_results"))
    payment = gold.get("payment_result") if isinstance(gold.get("payment_result"), dict) else {}
    if not clean(train_input.get("court_found_facts")):
        issues.append("court_fact_missing")
    if not claims:
        issues.append("claim_split_ambiguous")
    claim_ids: set[str] = set()
    for claim in claims:
        claim_id = clean(claim.get("claim_id")) or "unknown_claim"
        if claim_id in claim_ids:
            issues.append("duplicate_claim_id")
        claim_ids.add(claim_id)
        if not _grounded(claim.get("request_text"), [train_input.get("plaintiff_statement")]):
            issues.append(f"{claim_id}_request_text_not_grounded")
        if claim.get("outcome") == "not_addressed":
            issues.append("claim_outcome_ambiguous")
        decision = clean(claim.get("decision_text"))
        if claim.get("outcome") != "not_addressed" and not decision:
            issues.append(f"{claim_id}_decision_text_missing")
        elif decision and not _grounded(
            decision,
            [source_record.get("court_reasoning"), source_record.get("judgment_result")],
        ):
            issues.append(f"{claim_id}_decision_text_not_grounded")
        requested = claim.get("requested_amount")
        awarded = claim.get("awarded_amount")
        if claim.get("claim_type") == "non_monetary" and (
            requested is not None or awarded is not None
        ):
            issues.append(f"{claim_id}_non_monetary_amount_forbidden")
        if claim.get("outcome") == "rejected" and claim.get("claim_type") == "monetary" and awarded != 0.0:
            issues.append(f"{claim_id}_rejected_amount_must_be_zero")

    obligations = _dict_list(payment.get("obligations"))
    for obligation in obligations:
        obligation_id = clean(obligation.get("obligation_id")) or "unknown_obligation"
        if not _grounded(obligation.get("ruling_text"), [source_record.get("judgment_result")]):
            issues.append(f"{obligation_id}_ruling_text_not_grounded")
        related = obligation.get("related_claim_ids")
        if not isinstance(related, list) or any(clean(value) not in claim_ids for value in related):
            issues.append(f"{obligation_id}_invalid_claim_reference")
        if not obligation.get("payer_names") or not obligation.get("payee_names"):
            issues.append("payment_party_ambiguous")

    for candidate in _dict_list(train_input.get("law_candidates")):
        law_id = clean(candidate.get("law_id"))
        if law_id and law_id not in catalog_ids_with_text:
            issues.append(f"law_catalog_missing_full_text_{law_id}")
    issues = list(dict.fromkeys(issues))
    checks = {
        "all_claims_addressed": bool(claims)
        and all(item.get("outcome") != "not_addressed" for item in claims),
        "amounts_consistent": not any(
            "amount" in issue or "total_mismatch" in issue for issue in issues
        ),
        "payment_grounded": not any(
            issue.startswith("payment_") or "ruling_text" in issue for issue in issues
        ),
        "all_spans_grounded": not any(
            "not_grounded" in issue or "_missing" in issue for issue in issues
        ),
    }
    confidence = round(max(0.0, 1.0 - 0.20 * len(issues)), 2)
    return {
        "passed": not issues and confidence >= 0.90,
        "method": METHOD,
        "confidence": confidence,
        "issues": issues,
        "checks": checks,
    }


def process_record(
    source_record: dict[str, Any],
    catalog_ids_with_text: set[str],
) -> dict[str, Any]:
    train_input = source_record.get("train_input")
    train_input = train_input if isinstance(train_input, dict) else {}
    claims, claim_issues = extract_claims(clean(train_input.get("plaintiff_statement")))
    claim_results, outcome_issues = match_claim_results(
        claims,
        clean(source_record.get("court_reasoning")),
        clean(source_record.get("judgment_result")),
    )
    payment_result, payment_issues = extract_payment_result(
        claim_results, clean(source_record.get("judgment_result"))
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "queue_index": source_record.get("queue_index"),
        "case_id": clean(source_record.get("case_id")),
        "train_input": train_input,
        "gold_output": {
            "claim_results": claim_results,
            "payment_result": payment_result,
        },
        "_rule_issues": [*claim_issues, *outcome_issues, *payment_issues],
    }
    quality = validate_local_result(source_record, result, catalog_ids_with_text)
    result.pop("_rule_issues", None)
    result["quality_control"] = quality
    if not quality["passed"]:
        result["partial_gold_output"] = result.pop("gold_output")
    return result


def iter_input_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise TypeError(f"JSONL line {line_number} must be an object")
                yield item
        return
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        yield value
    elif isinstance(value, list):
        for position, item in enumerate(value, 1):
            if not isinstance(item, dict):
                raise TypeError(f"JSON item {position} must be an object")
            yield item
    else:
        raise TypeError("Input JSON must be an object or array")


def load_source_records(
    path: Path,
    *,
    start: int,
    limit: int | None,
    existing_keys: set[tuple[Any, str]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    end = None if limit is None else start + limit
    for zero_index, case in enumerate(iter_input_records(path)):
        if zero_index < start:
            continue
        if end is not None and zero_index >= end:
            break
        record = build_source_record(case, zero_index + 1)
        if (record["queue_index"], record["case_id"]) not in existing_keys:
            result.append(record)
    return result


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"JSONL line {line_number} must be an object")
            result.append(item)
    return result


def read_existing_keys(path: Path) -> set[tuple[Any, str]]:
    return {
        (item.get("queue_index"), clean(item.get("case_id")))
        for item in read_jsonl_records(path)
        if clean(item.get("case_id"))
    }


def repair_trailing_jsonl(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return False
    last_newline = data.rfind(b"\n")
    tail = data[last_newline + 1 :]
    try:
        item = json.loads(tail.decode("utf-8"))
        if not isinstance(item, dict):
            raise TypeError("trailing JSON is not an object")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        repaired = data[: last_newline + 1] if last_newline >= 0 else b""
    else:
        repaired = data + b"\n"
    with path.open("wb") as target:
        target.write(repaired)
        target.flush()
        os.fsync(target.fileno())
    return True


class JsonlCommitWriter:
    def __init__(self, path: Path, *, append: bool) -> None:
        self.path = path
        self.append = append
        self._target: Any = None

    def __enter__(self) -> "JsonlCommitWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._target = self.path.open("ab" if self.append else "wb")
        return self

    def commit(self, record: dict[str, Any]) -> None:
        if self._target is None:
            raise RuntimeError("writer is not open")
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        self._target.write(line)
        self._target.flush()
        os.fsync(self._target.fileno())

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._target is not None:
            self._target.close()
            self._target = None


def build_report(
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    law_catalog_issues: list[str] | None = None,
) -> dict[str, Any]:
    claim_types: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    for record in [*successes, *failures]:
        gold = record.get("gold_output") or record.get("partial_gold_output") or {}
        for claim in _dict_list(gold.get("claim_results")):
            claim_types[clean(claim.get("claim_type"))] += 1
            outcomes[clean(claim.get("outcome"))] += 1
    for record in failures:
        quality = record.get("quality_control") if isinstance(record.get("quality_control"), dict) else {}
        for issue in quality.get("issues", []):
            if clean(issue):
                failure_reasons[clean(issue)] += 1
    selected = len(successes) + len(failures)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "selected": selected,
        "passed": len(successes),
        "failed": len(failures),
        "pass_rate": round(len(successes) / selected, 4) if selected else 0.0,
        "claim_counts": dict(sorted(claim_types.items())),
        "outcome_counts": dict(sorted(outcomes.items())),
        "top_failure_reasons": dict(failure_reasons.most_common()),
        "law_catalog_issues": law_catalog_issues or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="纯 Python 抽取请求级 CoreGold；不调用 API 或网络。"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failed-output", type=Path)
    parser.add_argument("--law-catalog-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--pretty-output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start 不能小于 0")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须大于 0")
    if args.workers <= 0:
        parser.error("--workers 必须大于 0")
    failed_output = args.failed_output or args.output.with_name(
        f"{args.output.stem}.qc-failed.jsonl"
    )
    law_catalog_output = args.law_catalog_output or args.output.with_name(
        "lexchain_law_catalog_local_v1.jsonl"
    )
    report_output = args.report_output or args.output.with_name(
        f"{args.output.stem}.report.json"
    )
    if len(
        {
            args.output.resolve(),
            failed_output.resolve(),
            law_catalog_output.resolve(),
            report_output.resolve(),
        }
    ) != 4:
        parser.error("成功、失败、法条目录和报告必须使用不同文件")

    if args.resume:
        repair_trailing_jsonl(args.output)
        repair_trailing_jsonl(failed_output)
        existing = read_existing_keys(args.output) | read_existing_keys(failed_output)
    else:
        existing = set()
    all_selected = load_source_records(
        args.input, start=args.start, limit=args.limit, existing_keys=set()
    )
    pending = [
        record
        for record in all_selected
        if (record["queue_index"], record["case_id"]) not in existing
    ]
    catalog, catalog_issues = collect_law_catalog(all_selected)
    write_jsonl(law_catalog_output, catalog)
    catalog_ids_with_text = {
        clean(item.get("law_id")) for item in catalog if clean(item.get("full_text"))
    }

    stats = {"written": 0, "passed": 0, "failed": 0}

    def commit(
        item: dict[str, Any],
        success_writer: JsonlCommitWriter,
        failure_writer: JsonlCommitWriter,
    ) -> None:
        if item.get("quality_control", {}).get("passed") is True:
            success_writer.commit(item)
            stats["passed"] += 1
            status = "passed"
        else:
            failure_writer.commit(item)
            stats["failed"] += 1
            status = "failed"
        stats["written"] += 1
        print(
            f"[{stats['written']}/{len(pending)}] queue_index={item.get('queue_index')} "
            f"case_id={item.get('case_id')} status={status}",
            flush=True,
        )

    with JsonlCommitWriter(args.output, append=args.resume) as success_writer, JsonlCommitWriter(
        failed_output, append=args.resume
    ) as failure_writer:
        if args.workers == 1:
            for record in pending:
                commit(
                    process_record(record, catalog_ids_with_text),
                    success_writer,
                    failure_writer,
                )
        elif pending:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(args.workers, len(pending))
            ) as executor:
                futures = [
                    executor.submit(process_record, record, catalog_ids_with_text)
                    for record in pending
                ]
                for future in concurrent.futures.as_completed(futures):
                    commit(future.result(), success_writer, failure_writer)

    successes = read_jsonl_records(args.output)
    failures = read_jsonl_records(failed_output)
    report = build_report(successes, failures, catalog_issues)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.pretty_output:
        args.pretty_output.parent.mkdir(parents=True, exist_ok=True)
        args.pretty_output.write_text(
            json.dumps(successes, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
