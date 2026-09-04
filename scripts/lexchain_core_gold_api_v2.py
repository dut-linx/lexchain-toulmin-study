"""Auditable one-shot API pipeline for LexChain CoreGold V2."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import lexchain_core_gold_local_v1 as local_v1


SCHEMA_VERSION = "lexchain-reference-v1"
METHOD = "single_api_extraction_v2"
PROMPT_VERSION = "lexchain-core-gold-single-api-v4.1-source-boundary"
DEFAULT_MODEL = "qwen3.7-max-2026-05-17"
DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_API_KEY_ENV = "QWEN_API_KEY"
DEFAULT_MAX_TOKENS = 8192
MODELS_WITHOUT_JSON_RESPONSE_FORMAT = frozenset({
    "qwen3.7-max-2026-05-17",
})
RETRYABLE_HTTP_STATUSES = {408, 409, 429, 500, 502, 503, 504}

NUMERIC_TEXT = re.compile(r"^-?\d+(?:\.\d+)?$")
CLAIM_ID = re.compile(r"^[MN][1-9]\d*$")
CLAIM_TYPES = {"monetary", "non_monetary"}
OUTCOMES = {
    "full_support", "partial_support", "rejected", "withdrawn", "not_addressed",
}
OPERATIONS = {
    "AWARD_MONEY", "DISMISS", "ORDER_PERFORMANCE", "DECLARATION",
    "WITHDRAWN", "NOT_ADDRESSED", "OTHER",
}
NON_MONEY_SUPPORT_OPERATIONS = {"ORDER_PERFORMANCE", "DECLARATION", "OTHER"}
LIABILITY_MODES = {"individual", "joint", "several", "supplementary", "unknown"}
CLAIM_TEXT_KEYS = {
    "claim_id", "claim_type", "item", "request_text", "outcome", "operation",
    "decision_text",
}
CLAIM_NUMBER_KEYS = {"requested_amount", "awarded_amount"}
DIRECTIVE_PARTY_PREFIXES = (
    "限被告", "由被告", "判令被告", "原告", "被告",
    "闄愯鍛", "鐢辫鍛", "鍒や护琚憡", "鍘熷憡", "琚憡",
)


def system_prompt() -> str:
    """Return the instruction for one complete semantic generation."""
    return (
        "你是中国裁判文书金标准抽取员，只进行一次完整抽取。"
        "完整输出原告全部实体诉请的claim_results和裁判主文中的payment_result。"
        "claim只能由plaintiff_statement诉请部分的明确文字建立；"
        "court_reasoning和judgment_result只能用于判断裁判结果、支持金额和付款义务，"
        "绝对不得用其中的项目或金额补造、改写或细分原告诉请。"
        "每个编号诉请至少成项；同一编号中可以独立执行的救济分别成项，"
        "例如停止与删除、赔礼道歉与消除影响。"
        "金钱项目只有在诉请原文中同时具有独立项目名称和独立请求金额时才分别输出；"
        "若诉请只列多个项目名称但仅给一个总额，必须保留为一个复合金额诉请，"
        "item保留这些项目，requested_amount使用该总额，禁止从说理分项或相减计算来拆分。"
        "每个requested_amount必须逐字可由本条request_text中的金额换算得到。"
        "若原告只有一个复合金钱诉请，且裁判主文给出唯一一笔明确对应的总判赔，"
        "必须将该总判赔写入该诉请的awarded_amount；不得因项目未分项而留空。"
        "明确金额明细存在时不得用‘各项损失’、‘赔偿合计’替代，也不得另建重复合计诉请。"
        "法院对不同项目分别作出全额支持、部分支持或驳回时，更必须逐项输出其结果和金额。"
        "消除影响、恢复名誉、赔礼道歉等不同救济分别输出，不得因位于同一编号而合并。"
        "排除单纯诉讼费承担、执行告知和迟延履行提示。"
        "不得使用输入外的事实、金额、主体、法条或结果。"
        "request_text必须是单个诉请中的连续原文，禁止跨编号拼接；"
        "其他连续原文字段也必须原样引用。"
        "非金钱诉请只有救济行为、对象、平台、范围、公开方式和期限均未实质缩减时，"
        "才是full_support；任一维度缩减均为partial_support，完全未获支持才是rejected。"
        "简单合并判赔只有唯一一笔付款义务且无法按诉请从原文可靠分配时，"
        "各金钱诉请保持独立，"
        "统一使用partial_support、awarded_amount为null、AWARD_MONEY和共同裁判原文；"
        "付款义务关联其共同覆盖的全部金钱诉请并记录可知总额，"
        "不得平均分配、重复总额或猜测。"
        "仅要求保险人或其他主体在责任限额内承担付款顺序、但没有独立请求金额的内容，"
        "不得虚构为requested_amount为null的金钱诉请；必要时作为非金钱责任诉请，"
        "实际付款仍关联其所清偿的基础金钱诉请。"
        "payment_result只记录为满足原告诉请而向原告支付的义务；"
        "不得纳入共同被告之间返还垫付款、追偿款或内部结算。"
        "payment_result必须独立依据judgment_result的给付主文提取最终应付款，"
        "已付款、垫付款或抵扣后的净额以主文为准；不同付款人或不同责任方式分别建立obligation。"
        "obligation只能关联其实际清偿的受支持金钱诉请，严禁关联rejected、withdrawn或not_addressed诉请。"
        "total_awarded_amount必须与所有obligation金额精确相加一致。"
        "无法确定时保留null或not_addressed，使记录进入本地复核。"
        "只输出严格JSON object，不输出Markdown或解释。"
    )


def extraction_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Build the compact source-only payload for the API."""
    train = record["train_input"]
    return {
        "task": "一次性抽取请求级LexChain金标准",
        "input": {
            "case_type": local_v1.clean(train.get("case_type")),
            "plaintiff_statement": local_v1.clean(train.get("plaintiff_statement")),
            "court_reasoning": local_v1.clean(record.get("court_reasoning")),
            "judgment_result": local_v1.clean(record.get("judgment_result")),
        },
        "output_schema": {
            "claim_results": [{
                "claim_id": "M1",
                "claim_type": "monetary/non_monetary",
                "item": "",
                "request_text": "",
                "requested_amount": None,
                "outcome": "full_support/partial_support/rejected/withdrawn/not_addressed",
                "awarded_amount": None,
                "operation": (
                    "AWARD_MONEY/DISMISS/ORDER_PERFORMANCE/DECLARATION/"
                    "WITHDRAWN/NOT_ADDRESSED/OTHER"
                ),
                "decision_text": "",
            }],
            "payment_result": {
                "has_payment": False,
                "total_awarded_amount": None,
                "obligations": [{
                    "obligation_id": "O1",
                    "payer_names": [],
                    "payee_names": [],
                    "related_claim_ids": ["M1"],
                    "amount": None,
                    "liability_mode": "individual/joint/several/supplementary/unknown",
                    "ruling_text": "",
                }],
            },
        },
        "instructions": [
            "完整覆盖全部实体诉请；每个编号诉请至少成项。",
            (
                "claim只能来自plaintiff_statement诉请原文；court_reasoning和judgment_result只用于判断"
                "outcome、awarded_amount、operation、decision_text和payment_result，禁止据此补造诉请项目或金额。"
            ),
            "同一编号中可独立执行的救济分别成项，例如停止与删除、赔礼道歉与消除影响。",
            (
                "只有诉请原文同时给出独立项目名称和独立金额时才拆分金钱项目；"
                "只列多个项目名称但仅给一个总额时，保留一个复合金额诉请，禁止从法院说理补充分项金额。"
            ),
            "法院逐项判断支持程度时，逐项输出full_support、partial_support或rejected及对应金额。",
            (
                "原告只有一个复合金钱诉请且裁判主文有唯一明确对应总判赔时，"
                "该总判赔直接作为此诉请的awarded_amount；只有多个独立诉请共享且无法分配时才填null。"
            ),
            (
                "request_text必须连续引用单个诉请原文，禁止跨编号拼接；"
                "requested_amount必须能由本条request_text中的金额直接换算得到。"
            ),
            (
                "非金钱诉请仅在行为、对象、平台、范围、方式和期限均未缩减时为full_support；"
                "任一维度缩减使用partial_support。"
            ),
            (
                "claim_id只能是M后接正整数或N后接正整数，例如金额诉请M1、非金额诉请N1；"
                "禁止输出斜杠、候选值或复合ID。M和N分别从1连续编号；"
                "付款义务只引用有效的金钱诉请ID。"
            ),
            "程序费用不进入claim_results或payment_result。",
            (
                "简单合并判赔仅有唯一obligation且无法可靠分项时，各诉请保持独立并使用partial_support、"
                "awarded_amount=null、AWARD_MONEY和共同decision_text；"
                "obligation关联全部覆盖诉请并保留总额，不得平均、重复或猜测。"
            ),
            (
                "无独立请求金额、仅指定保险责任主体或赔付顺序的请求不是金钱明细诉请；"
                "付款义务应关联基础损失项目。"
            ),
            "payment_result只含向原告履行的付款，不含被告之间返还垫付款、追偿或内部结算。",
            (
                "payment_result独立依据judgment_result提取最终净付款；不同付款人或责任方式分别建立obligation；"
                "related_claim_ids只包含实际获支持且被该付款清偿的金钱诉请，禁止关联被驳回或撤回诉请。"
            ),
            "total_awarded_amount必须严格等于全部obligation.amount之和。",
        ],
    }


def _reject_nonstandard_json_constant(constant: str) -> None:
    raise json.JSONDecodeError("invalid JSON constant", constant, 0)


def parse_json_content(content: str) -> dict[str, Any]:
    """Parse a plain strict JSON object."""
    parsed = json.loads(content, parse_constant=_reject_nonstandard_json_constant)
    if not isinstance(parsed, dict):
        raise TypeError("JSON content must be an object")
    return parsed


def prepare_source_record(case: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    """Normalize a source case with the established V1 source boundary."""
    return local_v1.build_source_record(case, fallback_index)


def source_exclusion_reasons(record: dict[str, Any]) -> list[str]:
    """Return deterministic source-incompleteness reasons."""
    reasons: list[str] = []
    train_input = record.get("train_input")
    train = train_input if isinstance(train_input, dict) else {}
    if not local_v1.clean(record.get("case_id")):
        reasons.append("source_missing_case_id")
    if not local_v1.clean(train.get("court_found_facts")):
        reasons.append("source_missing_court_facts")
    if not local_v1.clean(train.get("plaintiff_statement")):
        reasons.append("source_missing_plaintiff_statement")
    if not local_v1.clean(record.get("judgment_result")):
        reasons.append("source_missing_judgment_result")
    return reasons


def normalize_number(value: Any) -> Any:
    """Convert real numbers and numeric text without semantic inference."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    text = local_v1.clean(value).replace(",", "").replace("，", "")
    return float(text) if NUMERIC_TEXT.fullmatch(text) else value


def normalize_model_output(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply mechanical container, text, number, and safe claim-ID normalization."""
    result: dict[str, Any] = {}
    claim_id_mapping: dict[str, str] = {}
    raw_claims = raw.get("claim_results")
    if isinstance(raw_claims, list):
        claims: list[Any] = []
        for value in raw_claims:
            if not isinstance(value, dict):
                claims.append(value)
                continue
            claim = dict(value)
            for key in CLAIM_TEXT_KEYS & claim.keys():
                claim[key] = local_v1.clean(claim[key])
            for key in CLAIM_NUMBER_KEYS & claim.keys():
                claim[key] = normalize_number(claim[key])
            if (
                local_v1.clean(claim.get("claim_type")) == "monetary"
                and local_v1.clean(claim.get("outcome")) == "rejected"
                and claim.get("awarded_amount") is None
            ):
                claim["awarded_amount"] = 0.0
            claims.append(claim)

        claim_rows = [claim for claim in claims if isinstance(claim, dict)]
        old_ids = [local_v1.clean(claim.get("claim_id")) for claim in claim_rows]
        types = [local_v1.clean(claim.get("claim_type")) for claim in claim_rows]
        old_id_set = set(old_ids)
        raw_payment = raw.get("payment_result")
        payment_refs_safe = isinstance(raw_payment, dict)
        raw_obligations = raw_payment.get("obligations") if payment_refs_safe else None
        has_payment = raw_payment.get("has_payment") if payment_refs_safe else None
        if not isinstance(has_payment, bool) or not isinstance(raw_obligations, list):
            payment_refs_safe = False
        elif has_payment is False:
            payment_refs_safe = raw_obligations == []
        else:
            payment_refs_safe = bool(raw_obligations)
            for obligation in raw_obligations:
                if not isinstance(obligation, dict):
                    payment_refs_safe = False
                    break
                related = obligation.get("related_claim_ids")
                if not isinstance(related, list) or not related:
                    payment_refs_safe = False
                    break
                cleaned_related = [
                    local_v1.clean(value) if isinstance(value, str) else value
                    for value in related
                ]
                if (
                    not all(isinstance(value, str) and value for value in cleaned_related)
                    or len(cleaned_related) != len(set(cleaned_related))
                    or any(value not in old_id_set for value in cleaned_related)
                ):
                    payment_refs_safe = False
                    break
        remap_safe = (
            len(claim_rows) == len(claims)
            and bool(claim_rows)
            and all(old_ids)
            and all(CLAIM_ID.fullmatch(old_id) for old_id in old_ids)
            and len(old_ids) == len(set(old_ids))
            and all(claim_type in CLAIM_TYPES for claim_type in types)
            and payment_refs_safe
        )
        if remap_safe:
            counters = {"monetary": 0, "non_monetary": 0}
            prefixes = {"monetary": "M", "non_monetary": "N"}
            for claim, old_id, claim_type in zip(claim_rows, old_ids, types):
                counters[claim_type] += 1
                new_id = f"{prefixes[claim_type]}{counters[claim_type]}"
                claim_id_mapping[old_id] = new_id
                claim["claim_id"] = new_id
        result["claim_results"] = claims

    if "payment_result" in raw:
        payment = dict(raw["payment_result"]) if isinstance(raw["payment_result"], dict) else raw["payment_result"]
        if isinstance(payment, dict):
            if "total_awarded_amount" in payment:
                payment["total_awarded_amount"] = normalize_number(payment["total_awarded_amount"])
            if isinstance(payment.get("obligations"), list):
                obligations: list[Any] = []
                for value in payment["obligations"]:
                    obligation = dict(value) if isinstance(value, dict) else value
                    if isinstance(obligation, dict):
                        for key in {"obligation_id", "liability_mode", "ruling_text"} & obligation.keys():
                            obligation[key] = local_v1.clean(obligation[key])
                        if "amount" in obligation:
                            obligation["amount"] = normalize_number(obligation["amount"])
                        for key in {"payer_names", "payee_names", "related_claim_ids"} & obligation.keys():
                            if isinstance(obligation[key], list):
                                obligation[key] = [
                                    (
                                        claim_id_mapping.get(local_v1.clean(item), local_v1.clean(item))
                                        if key == "related_claim_ids" and isinstance(item, str)
                                        else local_v1.clean(item) if isinstance(item, str)
                                        else item
                                    )
                                    for item in obligation[key]
                                ]
                    obligations.append(obligation)
                payment["obligations"] = obligations
            if (
                payment.get("has_payment") is False
                and payment.get("obligations") == []
                and isinstance(payment.get("total_awarded_amount"), (int, float))
                and not isinstance(payment.get("total_awarded_amount"), bool)
                and float(payment["total_awarded_amount"]) == 0.0
            ):
                payment["total_awarded_amount"] = None
        result["payment_result"] = payment

    # One monetary claim linked to one smaller positive obligation has an
    # unambiguous total award, even if the claim itself names several items.
    claims = result.get("claim_results")
    payment = result.get("payment_result")
    if isinstance(claims, list) and isinstance(payment, dict):
        monetary = [
            claim for claim in claims
            if isinstance(claim, dict) and claim.get("claim_type") == "monetary"
        ]
        obligations = payment.get("obligations")
        if len(monetary) == 1 and isinstance(obligations, list) and len(obligations) == 1:
            claim = monetary[0]
            obligation = obligations[0]
            requested = claim.get("requested_amount")
            amount = obligation.get("amount") if isinstance(obligation, dict) else None
            related = obligation.get("related_claim_ids") if isinstance(obligation, dict) else None
            if (
                claim.get("outcome") == "partial_support"
                and claim.get("awarded_amount") is None
                and isinstance(requested, (int, float)) and not isinstance(requested, bool)
                and isinstance(amount, (int, float)) and not isinstance(amount, bool)
                and 0 < float(amount) < float(requested)
                and related == [claim.get("claim_id")]
            ):
                claim["awarded_amount"] = float(amount)
    return result


def assemble_result(source_record: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """Assemble local identity and normalized API output."""
    return {
        "schema_version": SCHEMA_VERSION,
        "queue_index": source_record.get("queue_index"),
        "case_id": source_record.get("case_id"),
        "train_input": source_record.get("train_input"),
        "gold_output": normalize_model_output(raw),
    }


def _append_issue(issues: list[str], issue: str) -> None:
    if issue not in issues:
        issues.append(issue)


def _claim_label(claim: dict[str, Any], index: int) -> str:
    return local_v1.clean(claim.get("claim_id")) or f"claim_{index}"


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _as_decimal(value: int | float) -> Decimal:
    return Decimal(str(value))


def validate_claims(source_record: dict[str, Any], claims: list[Any]) -> list[str]:
    """Validate stable claim structure and exact source grounding only."""
    issues: list[str] = []
    if not claims:
        issues.append("claim_results_empty")
        return issues

    train_input = source_record.get("train_input")
    train = train_input if isinstance(train_input, dict) else {}
    plaintiff_statement = local_v1.clean(train.get("plaintiff_statement"))
    decision_sources = (
        local_v1.clean(source_record.get("court_reasoning")),
        local_v1.clean(source_record.get("judgment_result")),
    )
    seen_ids: set[str] = set()
    typed_ids: dict[str, list[str]] = {"monetary": [], "non_monetary": []}

    for index, claim in enumerate(claims, 1):
        if not isinstance(claim, dict):
            _append_issue(issues, f"claim_{index}_not_object")
            continue

        label = _claim_label(claim, index)
        claim_id = local_v1.clean(claim.get("claim_id"))
        claim_type = local_v1.clean(claim.get("claim_type"))
        outcome = local_v1.clean(claim.get("outcome"))
        operation = local_v1.clean(claim.get("operation"))
        item = local_v1.clean(claim.get("item"))
        request_text = local_v1.clean(claim.get("request_text"))
        decision_text = local_v1.clean(claim.get("decision_text"))

        if not claim_id:
            _append_issue(issues, f"{label}_claim_id_empty")
        elif not CLAIM_ID.fullmatch(claim_id):
            _append_issue(issues, f"{label}_claim_id_invalid")
        if claim_id in seen_ids:
            _append_issue(issues, f"{label}_duplicate_claim_id")
        elif claim_id:
            seen_ids.add(claim_id)

        if claim_type not in CLAIM_TYPES:
            _append_issue(issues, f"{label}_claim_type_invalid")
        else:
            typed_ids[claim_type].append(claim_id)
            if CLAIM_ID.fullmatch(claim_id):
                prefix = "M" if claim_type == "monetary" else "N"
                if not claim_id.startswith(prefix):
                    _append_issue(issues, f"{label}_claim_id_type_mismatch")

        if outcome not in OUTCOMES:
            _append_issue(issues, f"{label}_outcome_invalid")
        if operation not in OPERATIONS:
            _append_issue(issues, f"{label}_operation_invalid")
        if not item:
            _append_issue(issues, f"{label}_item_empty")
        if not request_text:
            _append_issue(issues, f"{label}_request_text_empty")
        elif request_text not in plaintiff_statement:
            _append_issue(issues, f"{label}_request_text_not_grounded")

        if outcome != "not_addressed" and not decision_text:
            _append_issue(issues, f"{label}_decision_text_empty")
        elif decision_text and not any(decision_text in source for source in decision_sources if source):
            _append_issue(issues, f"{label}_decision_text_not_grounded")
        if outcome == "not_addressed":
            _append_issue(issues, f"{label}_not_addressed_review")

    for claim_type, prefix in (("monetary", "M"), ("non_monetary", "N")):
        expected = [f"{prefix}{number}" for number in range(1, len(typed_ids[claim_type]) + 1)]
        if typed_ids[claim_type] != expected:
            _append_issue(issues, f"{claim_type}_claim_id_sequence_invalid")
    return issues


def _valid_unallocated_partial_ids(
    source_record: dict[str, Any],
    claims: list[Any],
    payment: dict[str, Any],
) -> set[str]:
    """Return null partial claims covered by one strict mechanical payment pattern."""
    if (
        not claims
        or not all(isinstance(claim, dict) for claim in claims)
        or not isinstance(payment, dict)
        or payment.get("has_payment") is not True
        or not _is_number(payment.get("total_awarded_amount"))
    ):
        return set()

    claim_ids = [local_v1.clean(claim.get("claim_id")) for claim in claims]
    claim_types = [local_v1.clean(claim.get("claim_type")) for claim in claims]
    if (
        any(not CLAIM_ID.fullmatch(claim_id) for claim_id in claim_ids)
        or len(claim_ids) != len(set(claim_ids))
        or any(claim_type not in CLAIM_TYPES for claim_type in claim_types)
    ):
        return set()

    candidates: dict[str, Decimal] = {}
    for claim, claim_id in zip(claims, claim_ids):
        if claim.get("claim_type") != "monetary" or claim.get("outcome") not in {
            "full_support", "partial_support",
        }:
            continue
        if (
            claim.get("outcome") != "partial_support"
            or claim.get("awarded_amount") is not None
            or claim.get("operation") != "AWARD_MONEY"
            or not _is_number(claim.get("requested_amount"))
            or float(claim["requested_amount"]) <= 0
        ):
            return set()
        candidates[claim_id] = _as_decimal(claim["requested_amount"])
    if len(candidates) < 2:
        return set()

    obligations = payment.get("obligations")
    if not isinstance(obligations, list) or len(obligations) != 1:
        return set()
    obligation = obligations[0]
    if not isinstance(obligation, dict) or not _is_number(obligation.get("amount")):
        return set()
    related = obligation.get("related_claim_ids")
    if (
        not isinstance(related, list)
        or not related
        or not all(isinstance(value, str) and local_v1.clean(value) for value in related)
    ):
        return set()
    cleaned_related = [local_v1.clean(value) for value in related]
    if (
        len(cleaned_related) != len(set(cleaned_related))
        or set(cleaned_related) != set(candidates)
    ):
        return set()

    ruling_text = local_v1.clean(obligation.get("ruling_text"))
    judgment = local_v1.clean(source_record.get("judgment_result"))
    if not ruling_text or ruling_text not in judgment:
        return set()
    for claim in claims:
        claim_id = local_v1.clean(claim.get("claim_id"))
        if claim_id in candidates and local_v1.clean(claim.get("decision_text")) != ruling_text:
            return set()

    total = _as_decimal(payment["total_awarded_amount"])
    obligation_amount = _as_decimal(obligation["amount"])
    if total != obligation_amount:
        return set()
    requested_total = sum(candidates.values(), Decimal("0"))
    if not (Decimal("0") < total < requested_total):
        return set()
    return set(candidates)


def validate_claim_amounts(
    claims: list[Any],
    allowed_unallocated: set[str] | None = None,
) -> list[str]:
    """Validate basic claim amount and outcome/operation invariants."""
    issues: list[str] = []
    allowed = allowed_unallocated or set()
    for index, claim in enumerate(claims, 1):
        if not isinstance(claim, dict):
            continue
        label = _claim_label(claim, index)
        claim_type = local_v1.clean(claim.get("claim_type"))
        outcome = local_v1.clean(claim.get("outcome"))
        operation = local_v1.clean(claim.get("operation"))
        requested = claim.get("requested_amount")
        awarded = claim.get("awarded_amount")

        if claim_type == "monetary":
            if not _is_number(requested) or float(requested) < 0:
                _append_issue(issues, f"{label}_requested_amount_invalid")
            if outcome == "full_support":
                if not _is_number(awarded) or not _is_number(requested) or float(awarded) != float(requested):
                    _append_issue(issues, f"{label}_full_support_amount_invalid")
                if operation != "AWARD_MONEY":
                    _append_issue(issues, f"{label}_full_support_operation_invalid")
            elif outcome == "partial_support":
                if (
                    not (awarded is None and local_v1.clean(claim.get("claim_id")) in allowed)
                    and (
                        not _is_number(awarded)
                        or not _is_number(requested)
                        or not (0 < float(awarded) < float(requested))
                    )
                ):
                    _append_issue(issues, f"{label}_partial_support_amount_invalid")
                if operation != "AWARD_MONEY":
                    _append_issue(issues, f"{label}_partial_support_operation_invalid")
            elif outcome == "rejected":
                if not _is_number(awarded) or float(awarded) != 0.0:
                    _append_issue(issues, f"{label}_rejected_amount_must_be_zero")
                if operation != "DISMISS":
                    _append_issue(issues, f"{label}_rejected_operation_invalid")
            elif outcome == "withdrawn":
                if awarded is not None:
                    _append_issue(issues, f"{label}_withdrawn_amount_must_be_null")
                if operation != "WITHDRAWN":
                    _append_issue(issues, f"{label}_withdrawn_operation_invalid")
            elif outcome == "not_addressed":
                if awarded is not None:
                    _append_issue(issues, f"{label}_not_addressed_amount_must_be_null")
                if operation != "NOT_ADDRESSED":
                    _append_issue(issues, f"{label}_not_addressed_operation_invalid")

        elif claim_type == "non_monetary":
            if requested is not None or awarded is not None:
                _append_issue(issues, f"{label}_non_monetary_amount_forbidden")
            expected_operation: str | None = None
            if outcome == "rejected":
                expected_operation = "DISMISS"
            elif outcome == "withdrawn":
                expected_operation = "WITHDRAWN"
            elif outcome == "not_addressed":
                expected_operation = "NOT_ADDRESSED"
            elif outcome in {"full_support", "partial_support"}:
                if operation not in NON_MONEY_SUPPORT_OPERATIONS:
                    _append_issue(issues, f"{label}_non_monetary_support_operation_invalid")
            if expected_operation is not None and operation != expected_operation:
                _append_issue(issues, f"{label}_{outcome}_operation_invalid")
    return issues


def _valid_party_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and bool(local_v1.clean(item)) for item in value)


def validate_payment(
    source_record: dict[str, Any],
    claims: list[Any],
    payment: dict[str, Any],
    allowed_unallocated: set[str] | None = None,
) -> list[str]:
    """Validate basic payment totals, parties, links, and ruling grounding."""
    issues: list[str] = []
    if not isinstance(payment, dict):
        return ["payment_result_not_object"]

    has_payment = payment.get("has_payment")
    total = payment.get("total_awarded_amount")
    obligations = payment.get("obligations")
    if not isinstance(has_payment, bool):
        _append_issue(issues, "payment_has_payment_not_boolean")
        return issues
    if has_payment is False:
        if total is not None:
            _append_issue(issues, "payment_false_total_must_be_null")
        if obligations != []:
            _append_issue(issues, "payment_false_obligations_must_be_empty")
    else:
        if not _is_number(total) or float(total) < 0:
            _append_issue(issues, "payment_total_invalid")
        if not isinstance(obligations, list) or not obligations:
            _append_issue(issues, "payment_obligations_empty")

    allowed = (
        _valid_unallocated_partial_ids(source_record, claims, payment)
        if allowed_unallocated is None
        else allowed_unallocated
    )
    supported: dict[str, Decimal] = {}
    existing_ids: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = local_v1.clean(claim.get("claim_id"))
        if claim_id:
            existing_ids.add(claim_id)
        if (
            claim.get("claim_type") == "monetary"
            and claim.get("outcome") in {"full_support", "partial_support"}
            and _is_number(claim.get("awarded_amount"))
        ):
            supported[claim_id] = _as_decimal(claim["awarded_amount"])

    judgment = local_v1.clean(source_record.get("judgment_result"))
    seen_obligation_ids: set[str] = set()
    reference_counts: Counter[str] = Counter()
    obligation_total = Decimal("0")
    obligation_amounts_valid = True
    obligation_rows = obligations if isinstance(obligations, list) else []
    for index, obligation in enumerate(obligation_rows, 1):
        if not isinstance(obligation, dict):
            _append_issue(issues, f"obligation_{index}_not_object")
            obligation_amounts_valid = False
            continue
        obligation_id = local_v1.clean(obligation.get("obligation_id"))
        label = obligation_id or f"obligation_{index}"
        if not obligation_id:
            _append_issue(issues, f"{label}_obligation_id_empty")
        elif obligation_id in seen_obligation_ids:
            _append_issue(issues, f"{label}_duplicate_obligation_id")
        else:
            seen_obligation_ids.add(obligation_id)

        for role in ("payer", "payee"):
            names = obligation.get(f"{role}_names")
            if not _valid_party_list(names):
                _append_issue(issues, f"{label}_{role}_names_invalid")
                continue
            for name in names:
                if local_v1.clean(name).startswith(DIRECTIVE_PARTY_PREFIXES):
                    _append_issue(issues, f"{label}_{role}_name_contains_directive")

        related = obligation.get("related_claim_ids")
        if not isinstance(related, list) or not related:
            _append_issue(issues, f"{label}_related_claim_ids_invalid")
        else:
            related_types_valid = all(isinstance(value, str) for value in related)
            cleaned_related = [
                local_v1.clean(value) if isinstance(value, str) else value
                for value in related
            ]
            if (
                not related_types_valid
                or any(not value for value in cleaned_related)
                or len(cleaned_related) != len(set(cleaned_related))
            ):
                _append_issue(issues, f"{label}_related_claim_ids_invalid")
            for claim_id in cleaned_related:
                if not isinstance(claim_id, str):
                    continue
                reference_counts[claim_id] += 1
                if claim_id not in existing_ids or claim_id not in (set(supported) | allowed):
                    _append_issue(issues, f"{label}_related_claim_id_not_supported")

        amount = obligation.get("amount")
        if not _is_number(amount) or float(amount) < 0:
            _append_issue(issues, f"{label}_amount_invalid")
            obligation_amounts_valid = False
        else:
            obligation_total += _as_decimal(amount)
        if local_v1.clean(obligation.get("liability_mode")) not in LIABILITY_MODES:
            _append_issue(issues, f"{label}_liability_mode_invalid")
        ruling_text = local_v1.clean(obligation.get("ruling_text"))
        if not ruling_text:
            _append_issue(issues, f"{label}_ruling_text_empty")
        elif ruling_text not in judgment:
            _append_issue(issues, f"{label}_ruling_text_not_grounded")

    null_partial_ids = {
        local_v1.clean(claim.get("claim_id"))
        for claim in claims
        if isinstance(claim, dict)
        and claim.get("claim_type") == "monetary"
        and claim.get("outcome") == "partial_support"
        and claim.get("awarded_amount") is None
        and claim.get("operation") == "AWARD_MONEY"
    }
    supported_or_allowed = set(supported) | allowed
    tracked_ids = supported_or_allowed | null_partial_ids
    ordered_tracked_ids: list[str] = []
    seen_tracked_ids: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = local_v1.clean(claim.get("claim_id"))
        if claim_id in tracked_ids and claim_id not in seen_tracked_ids:
            ordered_tracked_ids.append(claim_id)
            seen_tracked_ids.add(claim_id)
    for claim_id in ordered_tracked_ids:
        count = reference_counts[claim_id]
        if claim_id in supported_or_allowed and count == 0:
            _append_issue(issues, f"payment_supported_claim_unreferenced_{claim_id}")
        elif count > 1:
            _append_issue(issues, f"payment_supported_claim_referenced_multiple_{claim_id}")

    if _is_number(total):
        total_value = _as_decimal(total)
        if obligation_amounts_valid and total_value != obligation_total:
            _append_issue(issues, "payment_total_vs_obligations_mismatch")
        if not allowed and total_value != sum(supported.values(), Decimal("0")):
            _append_issue(issues, "payment_total_vs_claim_awards_mismatch")
    return issues


MONEY_AMOUNT_IN_REQUEST = re.compile(
    r"(?<!\d)(\d[\d,，]*(?:\.\d+)?)\s*(万元|万余元|元|余元)(?![\d年月日])"
)
INDEPENDENT_RELIEF_MARKERS = (
    "停止侵害", "停止侵权", "停止使用", "删除", "下架", "销毁",
    "消除影响", "恢复名誉", "赔礼道歉", "返还", "交付", "腾退",
    "排除妨害", "恢复原状", "继续履行", "解除合同", "确认无效",
)
NONMONEY_SCOPE_MARKERS = (
    "抖音", "微信朋友圈", "公众号", "微博", "快手", "小红书",
    "淘宝", "天猫", "拼多多", "京东", "QQ群", "微信群", "业主群",
    "网站", "网页", "报纸", "报刊", "电视",
)
NONMONEY_DURATION = re.compile(
    r"(?:置顶|持续|连续|保留|刊登)?\s*[一二三四五六七八九十百\d]+\s*(?:日|天|个月|月|年)"
)


def _claim_request_section(plaintiff_statement: str) -> str:
    """Return the leading requests block, excluding later factual amount narration."""
    text = local_v1.clean(plaintiff_statement)
    boundaries = [
        text.find(marker) for marker in ("事实和理由", "事实与理由", "事实及理由")
        if text.find(marker) >= 0
    ]
    return text[:min(boundaries)] if boundaries else text


def _money_amounts_in_text(value: Any) -> set[Decimal]:
    amounts: set[Decimal] = set()
    for number, unit in MONEY_AMOUNT_IN_REQUEST.findall(local_v1.clean(value)):
        normalized = number.replace(",", "").replace("，", "")
        amount = Decimal(normalized)
        if unit.startswith("万"):
            amount *= Decimal("10000")
        amounts.add(amount)
    return amounts


def validate_claim_granularity(source_record: dict[str, Any], claims: list[Any]) -> list[str]:
    """Route obvious semantic claim collapses to review without rewriting API output."""
    issues: list[str] = []
    train = source_record.get("train_input")
    plaintiff_statement = local_v1.clean(
        train.get("plaintiff_statement") if isinstance(train, dict) else ""
    )
    request_section = _claim_request_section(plaintiff_statement)
    source_amounts = {value for value in _money_amounts_in_text(request_section) if value > 0}
    monetary_claim_count = sum(
        1 for claim in claims
        if isinstance(claim, dict) and claim.get("claim_type") == "monetary"
    )
    if len(source_amounts) >= 3 and monetary_claim_count <= 1:
        _append_issue(issues, "source_itemized_money_collapsed_review")

    for index, claim in enumerate(claims, 1):
        if not isinstance(claim, dict) or claim.get("claim_type") != "monetary":
            continue
        requested = claim.get("requested_amount")
        if _is_number(requested) and _as_decimal(requested) not in _money_amounts_in_text(
            claim.get("request_text")
        ):
            _append_issue(
                issues,
                f"{_claim_label(claim, index)}_requested_amount_not_in_request_text",
            )

    for index, claim in enumerate(claims, 1):
        if not isinstance(claim, dict) or claim.get("claim_type") != "non_monetary":
            continue
        combined_text = local_v1.clean(claim.get("item")) + local_v1.clean(claim.get("request_text"))
        present = {marker for marker in INDEPENDENT_RELIEF_MARKERS if marker in combined_text}
        # Longer phrases subsume their generic prefix (for example 停止侵害 / 停止使用).
        if "停止侵权" in present:
            present.discard("停止侵害")
        if len(present) >= 2:
            _append_issue(issues, f"{_claim_label(claim, index)}_independent_reliefs_collapsed_review")
    return issues


def validate_nonmoney_scope(claims: list[Any]) -> list[str]:
    """Flag mechanically visible scope reductions mislabeled as full support."""
    issues: list[str] = []
    for index, claim in enumerate(claims, 1):
        if (
            not isinstance(claim, dict)
            or claim.get("claim_type") != "non_monetary"
            or claim.get("outcome") != "full_support"
        ):
            continue
        request_text = local_v1.clean(claim.get("request_text"))
        decision_text = local_v1.clean(claim.get("decision_text"))
        request_scope = {marker for marker in NONMONEY_SCOPE_MARKERS if marker in request_text}
        if any(marker not in decision_text for marker in request_scope):
            _append_issue(
                issues,
                f"{_claim_label(claim, index)}_full_support_scope_narrowed_review",
            )
        request_durations = set(NONMONEY_DURATION.findall(request_text))
        if request_durations and not all(value in decision_text for value in request_durations):
            _append_issue(
                issues,
                f"{_claim_label(claim, index)}_full_support_duration_narrowed_review",
            )
    return issues


def validate_output(source_record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic QC metadata without altering semantic output."""
    issues: list[str] = []
    gold = result.get("gold_output")
    if not isinstance(gold, dict):
        gold = {}
        issues.append("gold_output_not_object")
    claims_value = gold.get("claim_results")
    if not isinstance(claims_value, list):
        issues.append("claim_results_not_list")
        claims: list[Any] = []
    else:
        claims = claims_value
    payment_value = gold.get("payment_result")
    if not isinstance(payment_value, dict):
        issues.append("payment_result_not_object")
        payment: dict[str, Any] = {}
    else:
        payment = payment_value
    if not local_v1.clean(source_record.get("court_reasoning")):
        _append_issue(issues, "source_missing_court_reasoning_review")
    allowed_unallocated = _valid_unallocated_partial_ids(source_record, claims, payment)
    if allowed_unallocated:
        _append_issue(issues, "combined_award_unallocated_review")
    for issue in (
        *validate_claims(source_record, claims),
        *validate_claim_granularity(source_record, claims),
        *validate_nonmoney_scope(claims),
        *validate_claim_amounts(claims, allowed_unallocated),
        *validate_payment(source_record, claims, payment, allowed_unallocated),
    ):
        _append_issue(issues, issue)
    return {
        "passed": not issues,
        "method": METHOD,
        "confidence": 1.0 if not issues else 0.0,
        "issues": issues,
        "checks": {
            "all_claims_addressed": not any("not_addressed" in issue for issue in issues),
            "amounts_consistent": not any("amount" in issue or "total" in issue for issue in issues),
            "payment_grounded": not any(issue.startswith("O") or issue.startswith("obligation_") or issue.startswith("payment_") for issue in issues),
            "all_spans_grounded": not any("not_grounded" in issue for issue in issues),
        },
    }


def build_request_body(model: str, system_prompt_text: str, payload: dict[str, Any]) -> bytes:
    """Build the complete request exactly once as deterministic UTF-8 bytes."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        "temperature": 0,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }
    if model not in MODELS_WITHOUT_JSON_RESPONSE_FORMAT:
        body["response_format"] = {"type": "json_object"}
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _normalized_usage(value: Any) -> dict[str, Any]:
    usage = dict(value) if isinstance(value, dict) else {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if not _is_number(usage.get(key)):
            usage[key] = 0
    return usage


class ChatApiFailure(RuntimeError):
    """Carry audit-safe metadata from a failed transport or response."""

    def __init__(self, issue_type: str, audit: dict[str, Any]):
        super().__init__(issue_type)
        self.issue_type = issue_type
        self.audit = audit


def call_chat_api(
    *,
    api_key: str,
    endpoint: str,
    model: str,
    system_prompt_text: str,
    payload: dict[str, Any],
    timeout: int,
    retries: int,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make one semantic call with byte-identical retryable transports."""
    encoded = build_request_body(model, system_prompt_text, payload)
    request_sha256 = hashlib.sha256(encoded).hexdigest()
    attempts = 0
    for attempt in range(retries + 1):
        attempts += 1
        request = urllib.request.Request(
            endpoint,
            data=encoded,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        try:
            with opener(request, timeout=timeout) as response:
                response_bytes = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in RETRYABLE_HTTP_STATUSES and attempt < retries:
                sleeper(min(2 ** attempt, 8))
                continue
            audit = {
                "request_sha256": request_sha256,
                "response_sha256": "",
                "response_id": "",
                "network_attempts": attempts,
                "semantic_calls": 1,
                "usage": _normalized_usage({}),
                "status": "error",
            }
            raise ChatApiFailure(type(exc).__name__, audit) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < retries:
                sleeper(min(2 ** attempt, 8))
                continue
            audit = {
                "request_sha256": request_sha256,
                "response_sha256": "",
                "response_id": "",
                "network_attempts": attempts,
                "semantic_calls": 1,
                "usage": _normalized_usage({}),
                "status": "error",
            }
            raise ChatApiFailure(type(exc).__name__, audit) from exc

        response_sha256 = hashlib.sha256(response_bytes).hexdigest()
        response_id = ""
        usage = _normalized_usage({})
        try:
            response_obj = json.loads(
                response_bytes.decode("utf-8"),
                parse_constant=_reject_nonstandard_json_constant,
            )
            if not isinstance(response_obj, dict):
                raise TypeError("response envelope must be an object")
            response_id = local_v1.clean(response_obj.get("id"))
            usage = _normalized_usage(response_obj.get("usage"))
            content = response_obj["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("response content must be text")
            raw = parse_json_content(content)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            audit = {
                "request_sha256": request_sha256,
                "response_sha256": response_sha256,
                "response_id": response_id,
                "network_attempts": attempts,
                "semantic_calls": 1,
                "usage": usage,
                "status": "error",
            }
            raise ChatApiFailure(type(exc).__name__, audit) from exc

        audit = {
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
            "response_id": response_id,
            "network_attempts": attempts,
            "semantic_calls": 1,
            "usage": usage,
            "status": "received",
        }
        return raw, audit
    raise RuntimeError("unreachable retry state")


def _error_quality(issue: str) -> dict[str, Any]:
    return {
        "passed": False,
        "method": METHOD,
        "confidence": 0.0,
        "issues": [issue],
        "checks": {
            "all_claims_addressed": False,
            "amounts_consistent": False,
            "payment_grounded": False,
            "all_spans_grounded": False,
        },
    }


def run_record(
    source_record: dict[str, Any],
    *,
    api_key: str,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    timeout: int = 180,
    retries: int = 5,
    api_caller: Callable[..., tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the semantic API once, assemble, validate, and return audit data."""
    caller = api_caller or call_chat_api
    base_audit = {
        "queue_index": source_record.get("queue_index"),
        "case_id": source_record.get("case_id"),
        "model": model,
        "prompt_version": PROMPT_VERSION,
    }
    try:
        raw, call_audit = caller(
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            system_prompt_text=system_prompt(),
            payload=extraction_payload(source_record),
            timeout=timeout,
            retries=retries,
        )
        if not isinstance(raw, dict):
            raise TypeError("API output must be an object")
        result = assemble_result(source_record, raw)
        result["quality_control"] = validate_output(source_record, result)
        audit = {**base_audit, **call_audit}
        audit.setdefault("request_sha256", "")
        audit.setdefault("response_sha256", "")
        audit.setdefault("response_id", "")
        audit.setdefault("network_attempts", 1)
        audit["semantic_calls"] = 1
        audit["usage"] = _normalized_usage(audit.get("usage"))
        audit.setdefault("status", "received")
        return result, audit
    except Exception as exc:
        if isinstance(exc, ChatApiFailure):
            issue_type = exc.issue_type
            call_audit = exc.audit
        else:
            issue_type = type(exc).__name__
            call_audit = {
                "request_sha256": "",
                "response_sha256": "",
                "response_id": "",
                "network_attempts": 1,
                "semantic_calls": 1,
                "usage": _normalized_usage({}),
                "status": "error",
            }
        issue = f"api_error_{issue_type}"
        result = {
            "schema_version": SCHEMA_VERSION,
            "queue_index": source_record.get("queue_index"),
            "case_id": source_record.get("case_id"),
            "train_input": source_record.get("train_input"),
            "gold_output": {},
            "quality_control": _error_quality(issue),
        }
        audit = {**base_audit, **call_audit, "status": "error", "semantic_calls": 1}
        return result, audit


def iter_input_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON or JSONL source records."""
    yield from local_v1.iter_input_records(path)


def select_input_cases(
    path: Path,
    start: int,
    limit: int | None,
    queue_indices: set[int] | None,
) -> list[tuple[int, dict[str, Any]]]:
    """Select input records before API preparation."""
    selected: list[tuple[int, dict[str, Any]]] = []
    if start < 0:
        raise ValueError("start must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    for zero_index, case in enumerate(iter_input_records(path)):
        if zero_index < start:
            continue
        if limit is not None and zero_index >= start + limit:
            break
        fallback = zero_index + 1
        queue_value = case.get("queue_index", fallback)
        try:
            queue_number = int(queue_value)
        except (TypeError, ValueError):
            queue_number = fallback
        if queue_indices is not None and queue_number not in queue_indices:
            continue
        selected.append((fallback, case))
    return selected


def record_key(record: dict[str, Any]) -> tuple[Any, str]:
    return record.get("queue_index"), local_v1.clean(record.get("case_id"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(iter_input_records(path))


def read_keys(path: Path) -> set[tuple[Any, str]]:
    return {record_key(record) for record in _read_jsonl(path) if record_key(record)[1]}


def attempt_journal_path(audit_output: Path) -> Path:
    return Path(str(audit_output) + ".attempt-journal.jsonl")


def _latest_journal_events(
    journal_records: list[dict[str, Any]],
) -> dict[tuple[Any, str], dict[str, Any]]:
    latest: dict[tuple[Any, str], dict[str, Any]] = {}
    for record in journal_records:
        key = record_key(record)
        if key[1]:
            latest[key] = record
    return latest


def _journal_event(
    record: dict[str, Any],
    state: str,
    *,
    model: Any,
    prompt_version: Any,
) -> dict[str, Any]:
    return {
        "queue_index": record.get("queue_index"),
        "case_id": record.get("case_id"),
        "state": state,
        "model": model if isinstance(model, str) and model else None,
        "prompt_version": (
            prompt_version
            if isinstance(prompt_version, str) and prompt_version
            else None
        ),
    }


def _unknown_usage() -> dict[str, None]:
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def _is_ambiguous_review(record: dict[str, Any]) -> bool:
    quality = record.get("quality_control")
    issues = quality.get("issues") if isinstance(quality, dict) else None
    return (
        isinstance(issues, list)
        and "runtime_ambiguous_attempt" in issues
    )


def _local_missing_audit_markers(
    completed_records: list[dict[str, Any]],
    audit_records: list[dict[str, Any]],
    journal_events: dict[tuple[Any, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    audited_keys = {record_key(record) for record in audit_records}
    by_key = {record_key(record): record for record in completed_records}
    markers: list[dict[str, Any]] = []
    for key, record in by_key.items():
        if key in audited_keys:
            continue
        journal_event = journal_events.get(key, {})
        was_ambiguous = (
            _is_ambiguous_review(record)
            or journal_event.get("state") == "ambiguous_review"
        )
        semantic_calls = 1 if journal_event and not was_ambiguous else None
        markers.append({
            "queue_index": record.get("queue_index"),
            "case_id": record.get("case_id"),
            "model": journal_event.get("model"),
            "prompt_version": journal_event.get("prompt_version"),
            "request_sha256": "",
            "response_sha256": "",
            "response_id": "",
            "network_attempts": None,
            "semantic_calls": semantic_calls,
            "usage": _unknown_usage(),
            "status": "ambiguous_attempt" if was_ambiguous else "local_audit_missing",
        })
    return markers


def _ambiguous_review_record(source_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "queue_index": source_record.get("queue_index"),
        "case_id": source_record.get("case_id"),
        "train_input": source_record.get("train_input"),
        "gold_output": {},
        "quality_control": _error_quality("runtime_ambiguous_attempt"),
    }


def _ambiguous_attempt_audit(
    source_record: dict[str, Any],
    journal_event: dict[str, Any],
) -> dict[str, Any]:
    return {
        "queue_index": source_record.get("queue_index"),
        "case_id": source_record.get("case_id"),
        "model": journal_event.get("model"),
        "prompt_version": journal_event.get("prompt_version"),
        "request_sha256": "",
        "response_sha256": "",
        "response_id": "",
        "network_attempts": None,
        "semantic_calls": None,
        "usage": _unknown_usage(),
        "status": "ambiguous_attempt",
    }


def _complete_trailing_jsonl(path: Path) -> None:
    """Keep complete rows and discard only an incomplete final row."""
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return
    last_newline = data.rfind(b"\n")
    tail = data[last_newline + 1:]
    try:
        item = json.loads(tail.decode("utf-8"))
        valid = isinstance(item, dict)
    except (UnicodeDecodeError, json.JSONDecodeError):
        valid = False
    complete = data + b"\n" if valid else (data[:last_newline + 1] if last_newline >= 0 else b"")
    with path.open("wb") as target:
        target.write(complete)
        target.flush()
        os.fsync(target.fileno())


class JsonlCommitWriter:
    """Incremental JSONL writer that flushes and fsyncs every row."""

    def __init__(self, path: Path, *, append: bool):
        self.path = path
        self.append = append
        self._target: Any = None
        self._lock = threading.Lock()

    def __enter__(self) -> "JsonlCommitWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._target = self.path.open("ab" if self.append else "wb")
        return self

    def commit(self, record: dict[str, Any]) -> None:
        if self._target is None:
            raise RuntimeError("writer is not open")
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        with self._lock:
            self._target.write(line)
            self._target.flush()
            os.fsync(self._target.fileno())

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._target is not None:
            self._target.close()
            self._target = None


def write_excluded_records(path: Path, records: list[dict[str, Any]]) -> None:
    with JsonlCommitWriter(path, append=False) as writer:
        for record in records:
            writer.commit({
                "queue_index": record.get("queue_index"),
                "case_id": record.get("case_id"),
                "train_input": record.get("train_input"),
                "exclusion_reasons": source_exclusion_reasons(record),
            })


def write_api_inputs(path: Path, records: list[dict[str, Any]]) -> None:
    with JsonlCommitWriter(path, append=False) as writer:
        for record in records:
            writer.commit({
                "queue_index": record.get("queue_index"),
                "case_id": record.get("case_id"),
                "payload": extraction_payload(record),
            })


def process_records(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    attempt_starter: Callable[[dict[str, Any]], None] | None = None,
    result_committer: Callable[[dict[str, Any]], None] | None = None,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    """Process pending records sequentially or with a bounded thread pool."""
    def process(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if attempt_starter is not None:
            attempt_starter(record)
        result, audit = run_record(
            record,
            api_key=args.api_key,
            endpoint=args.endpoint,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
        )
        if result_committer is not None:
            result_committer(result)
        return result, audit

    if args.workers <= 1:
        for record in records:
            yield process(record)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        yield from executor.map(process, records)


def build_report(
    selected_records: list[dict[str, Any]],
    eligible_records: list[dict[str, Any]],
    excluded_records: list[dict[str, Any]],
    passed_records: list[dict[str, Any]],
    review_records: list[dict[str, Any]],
    audit_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build complete operational counts from output records."""
    issue_counts: Counter[str] = Counter()
    for record in review_records:
        quality = record.get("quality_control")
        if isinstance(quality, dict) and isinstance(quality.get("issues"), list):
            issue_counts.update(local_v1.clean(issue) for issue in quality["issues"] if local_v1.clean(issue))
    usage_totals = Counter()
    for audit in audit_records:
        usage = audit.get("usage") if isinstance(audit.get("usage"), dict) else {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key, 0)
            if _is_number(value):
                usage_totals[key] += int(value)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "selected": len(selected_records),
        "eligible": len(eligible_records),
        "source_excluded": len(excluded_records),
        "api_attempted": len(audit_records),
        "api_received": sum(audit.get("status") == "received" for audit in audit_records),
        "api_errors": sum(audit.get("status") == "error" for audit in audit_records),
        "qc_passed": len(passed_records),
        "review_required": len(review_records),
        "network_retries": sum(
            max(int(audit["network_attempts"]) - 1, 0)
            for audit in audit_records
            if _is_number(audit.get("network_attempts"))
        ),
        "local_audit_missing": sum(
            audit.get("status") == "local_audit_missing"
            for audit in audit_records
        ),
        "ambiguous_attempts": sum(
            audit.get("status") == "ambiguous_attempt"
            for audit in audit_records
        ),
        "prompt_tokens": usage_totals["prompt_tokens"],
        "completion_tokens": usage_totals["completion_tokens"],
        "total_tokens": usage_totals["total_tokens"],
        "top_qc_issues": dict(issue_counts.most_common(20)),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())


def write_report_from_complete_files(
    args: argparse.Namespace,
    selected_records: list[dict[str, Any]],
    eligible_records: list[dict[str, Any]],
    excluded_records: list[dict[str, Any]],
) -> dict[str, Any]:
    report = build_report(
        selected_records,
        eligible_records,
        excluded_records,
        _read_jsonl(args.output),
        _read_jsonl(args.review_output),
        _read_jsonl(args.audit_output),
    )
    write_report(args.report_output, report)
    return report


def _parse_queue_indices(value: str) -> set[int]:
    try:
        result = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("queue indices must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("queue indices cannot be empty")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--excluded-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--api-input-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--pretty-output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--queue-indices", type=_parse_queue_indices)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=5)
    return parser


def _nonempty_output_bundle_paths(args: argparse.Namespace) -> list[Path]:
    paths = {
        args.output,
        args.review_output,
        args.excluded_output,
        args.audit_output,
        args.api_input_output,
        args.report_output,
        attempt_journal_path(args.audit_output),
    }
    if args.pretty_output is not None:
        paths.add(args.pretty_output)
    return sorted(
        (path for path in paths if path.exists() and path.stat().st_size > 0),
        key=str,
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.retries < 0:
        raise SystemExit("--retries must be non-negative")
    if args.prepare_only:
        occupied = _nonempty_output_bundle_paths(args)
        if occupied:
            joined = ", ".join(str(path) for path in occupied)
            raise SystemExit(
                "--prepare-only requires an empty output bundle; nonempty paths: "
                + joined
            )

    selected_cases = select_input_cases(args.input, args.start, args.limit, args.queue_indices)
    source_records = [prepare_source_record(case, fallback) for fallback, case in selected_cases]
    excluded_records = [record for record in source_records if source_exclusion_reasons(record)]
    eligible_records = [record for record in source_records if not source_exclusion_reasons(record)]
    eligible_keys = [record_key(record) for record in eligible_records]
    if len(eligible_keys) != len(set(eligible_keys)):
        raise SystemExit("eligible records contain duplicate completion keys")
    write_excluded_records(args.excluded_output, excluded_records)
    write_api_inputs(args.api_input_output, eligible_records)

    if args.prepare_only:
        report = build_report(source_records, eligible_records, excluded_records, [], [], [])
        write_report(args.report_output, report)
        if args.pretty_output:
            write_report(args.pretty_output, {"records": []})
        return 0

    journal_path = attempt_journal_path(args.audit_output)
    if args.resume:
        for path in (args.output, args.review_output, args.audit_output, journal_path):
            _complete_trailing_jsonl(path)
    if args.resume:
        completed_records = [*_read_jsonl(args.output), *_read_jsonl(args.review_output)]
        existing_audits = _read_jsonl(args.audit_output)
        journal_events = _latest_journal_events(_read_jsonl(journal_path))
        completed_keys = {record_key(record) for record in completed_records}
        missing_audit_markers = _local_missing_audit_markers(
            completed_records,
            existing_audits,
            journal_events,
        )
        ambiguous_records = [
            record
            for record in eligible_records
            if record_key(record) not in completed_keys
            and record_key(record) in journal_events
        ]
    else:
        completed_keys = set()
        missing_audit_markers = []
        journal_events = {}
        ambiguous_records = []
    ambiguous_keys = {record_key(record) for record in ambiguous_records}
    pending_records = [
        record
        for record in eligible_records
        if record_key(record) not in completed_keys
        and record_key(record) not in ambiguous_keys
    ]
    args.api_key = os.environ.get(args.api_key_env, "")
    if pending_records and not args.api_key:
        raise SystemExit(f"API key environment variable is empty: {args.api_key_env}")

    with (
        JsonlCommitWriter(args.output, append=args.resume) as passed_writer,
        JsonlCommitWriter(args.review_output, append=args.resume) as review_writer,
        JsonlCommitWriter(args.audit_output, append=args.resume) as audit_writer,
        JsonlCommitWriter(journal_path, append=args.resume) as journal_writer,
    ):
        audit_failure: Exception | None = None

        def commit_audit(audit: dict[str, Any]) -> None:
            nonlocal audit_failure
            if audit_failure is None:
                try:
                    audit_writer.commit(audit)
                except Exception as exc:
                    audit_failure = exc

        for marker in missing_audit_markers:
            commit_audit(marker)

        for record in ambiguous_records:
            prior_event = journal_events[record_key(record)]
            review_record = _ambiguous_review_record(record)
            review_writer.commit(review_record)
            journal_writer.commit(
                _journal_event(
                    record,
                    "ambiguous_review",
                    model=prior_event.get("model"),
                    prompt_version=prior_event.get("prompt_version"),
                )
            )
            commit_audit(_ambiguous_attempt_audit(record, prior_event))

        def start_attempt(record: dict[str, Any]) -> None:
            journal_writer.commit(
                _journal_event(
                    record,
                    "started",
                    model=args.model,
                    prompt_version=PROMPT_VERSION,
                )
            )

        def commit_result(result: dict[str, Any]) -> None:
            if result["quality_control"]["passed"]:
                passed_writer.commit(result)
            else:
                review_writer.commit(result)
            journal_writer.commit(
                _journal_event(
                    result,
                    "completed",
                    model=args.model,
                    prompt_version=PROMPT_VERSION,
                )
            )

        for result, audit in process_records(
            pending_records,
            args,
            attempt_starter=start_attempt,
            result_committer=commit_result,
        ):
            del result
            commit_audit(audit)
        if audit_failure is not None:
            raise audit_failure

    write_report_from_complete_files(args, source_records, eligible_records, excluded_records)
    if args.pretty_output:
        args.pretty_output.parent.mkdir(parents=True, exist_ok=True)
        with args.pretty_output.open("w", encoding="utf-8", newline="\n") as target:
            json.dump(_read_jsonl(args.output), target, ensure_ascii=False, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
