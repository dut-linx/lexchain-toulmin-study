from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import lexchain_core_gold_api_v2 as v2


def raw_case() -> dict:
    return {
        "queue_index": 1,
        "case_metadata": {"case_id": "case-001", "category": ["民事", "饲养动物损害责任纠纷"]},
        "model_input": {
            "A_court_found_facts_only": {
                "facts": {"text": ""},
                "given_sources": {"items": [{
                    "source_id": "MC_1179",
                    "citation": "《中华人民共和国民法典》第一千一百七十九条",
                    "text": "侵害他人造成人身损害的，应当赔偿。",
                }]},
            },
        },
        "source_evidence": {"evidence_items": [
            {"source_id": "E-PlaintiffClaims", "raw_span": "原告提出诉讼请求：1.赔偿医疗费1257元；2.赔偿误工费680元；3.赔偿交通费200元；4.诉讼费由被告承担。事实和理由：原告受伤。"},
            {"source_id": "E-DefenseViewpoint", "raw_span": "被告不同意原告诉请。"},
            {"source_id": "E-Ascertain", "raw_span": "本院查明原告因本案受伤。"},
            {"source_id": "E-Identified", "raw_span": "本院认为医疗费1257元予以支持，误工费不予支持，交通费酌情支持100元。"},
            {"source_id": "E-RefereeResult", "raw_span": "一、被告赵某赔偿原告钱某1357元；二、驳回其他诉讼请求。"},
        ]},
    }


def evidence_by_id(case: dict) -> dict[str, dict]:
    return {item["source_id"]: item for item in case["source_evidence"]["evidence_items"]}


def eligible_source() -> dict:
    return v2.prepare_source_record(raw_case(), 1)


def complete_model_output() -> dict:
    return {
        "claim_results": [
            {"claim_id": "M1", "claim_type": "monetary", "item": "医疗费", "request_text": "医疗费1257元", "requested_amount": 1257.0, "outcome": "full_support", "awarded_amount": 1257.0, "operation": "AWARD_MONEY", "decision_text": "医疗费1257元予以支持"},
            {"claim_id": "M2", "claim_type": "monetary", "item": "误工费", "request_text": "误工费680元", "requested_amount": 680.0, "outcome": "rejected", "awarded_amount": 0.0, "operation": "DISMISS", "decision_text": "误工费不予支持"},
            {"claim_id": "M3", "claim_type": "monetary", "item": "交通费", "request_text": "交通费200元", "requested_amount": 200.0, "outcome": "partial_support", "awarded_amount": 100.0, "operation": "AWARD_MONEY", "decision_text": "交通费酌情支持100元"},
        ],
        "payment_result": {
            "has_payment": True,
            "total_awarded_amount": 1357.0,
            "obligations": [{
                "obligation_id": "O1", "payer_names": ["赵某"], "payee_names": ["钱某"],
                "related_claim_ids": ["M1", "M3"], "amount": 1357.0,
                "liability_mode": "individual", "ruling_text": "被告赵某赔偿原告钱某1357元",
            }],
        },
    }


def invalid_but_parseable_output() -> dict:
    value = complete_model_output()
    value["claim_results"][0].update(outcome="not_addressed", awarded_amount=None, operation="NOT_ADDRESSED", decision_text="")
    value["payment_result"]["obligations"][0]["related_claim_ids"] = ["M3"]
    value["payment_result"]["obligations"][0]["amount"] = 100.0
    value["payment_result"]["total_awarded_amount"] = 100.0
    return value


def combined_award_source() -> dict:
    source = copy.deepcopy(eligible_source())
    source["train_input"]["plaintiff_statement"] = (
        "原告提出诉讼请求：1.停止侵权并删除涉案图片；"
        "2.赔偿合理费用3000元；3.赔偿经济损失7000元；4.诉讼费由被告承担。"
    )
    source["court_reasoning"] = "本院认为侵权成立，经济损失及合理费用合并酌定1800元。"
    source["judgment_result"] = (
        "一、被告停止侵权并删除涉案图片；"
        "二、被告赔偿原告经济损失及合理费用共计1800元；"
        "三、驳回其他诉讼请求。"
    )
    return source


def combined_award_output() -> dict:
    common_ruling = "被告赔偿原告经济损失及合理费用共计1800元"
    return {
        "claim_results": [
            {
                "claim_id": "N1", "claim_type": "non_monetary",
                "item": "停止侵权", "request_text": "停止侵权",
                "requested_amount": None, "outcome": "full_support",
                "awarded_amount": None, "operation": "ORDER_PERFORMANCE",
                "decision_text": "被告停止侵权并删除涉案图片",
            },
            {
                "claim_id": "N2", "claim_type": "non_monetary",
                "item": "删除涉案图片", "request_text": "删除涉案图片",
                "requested_amount": None, "outcome": "full_support",
                "awarded_amount": None, "operation": "ORDER_PERFORMANCE",
                "decision_text": "被告停止侵权并删除涉案图片",
            },
            {
                "claim_id": "M1", "claim_type": "monetary",
                "item": "合理费用", "request_text": "合理费用3000元",
                "requested_amount": 3000.0, "outcome": "partial_support",
                "awarded_amount": None, "operation": "AWARD_MONEY",
                "decision_text": common_ruling,
            },
            {
                "claim_id": "M2", "claim_type": "monetary",
                "item": "经济损失", "request_text": "经济损失7000元",
                "requested_amount": 7000.0, "outcome": "partial_support",
                "awarded_amount": None, "operation": "AWARD_MONEY",
                "decision_text": common_ruling,
            },
        ],
        "payment_result": {
            "has_payment": True,
            "total_awarded_amount": 1800.0,
            "obligations": [{
                "obligation_id": "O1", "payer_names": ["海南公司"],
                "payee_names": ["彭某"], "related_claim_ids": ["M1", "M2"],
                "amount": 1800.0, "liability_mode": "individual",
                "ruling_text": common_ruling,
            }],
        },
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class SourcePartitionTests(unittest.TestCase):
    def test_missing_fact_is_excluded_before_api(self) -> None:
        case = raw_case()
        evidence = evidence_by_id(case)
        evidence["E-Ascertain"]["raw_span"] = ""
        evidence["E-Identified"]["raw_span"] = "本院认为，被告应承担责任。"
        self.assertEqual(v2.source_exclusion_reasons(v2.prepare_source_record(case, 1)), ["source_missing_court_facts"])

    def test_missing_judgment_is_excluded_before_api(self) -> None:
        case = raw_case()
        evidence_by_id(case)["E-RefereeResult"]["raw_span"] = ""
        self.assertEqual(v2.source_exclusion_reasons(v2.prepare_source_record(case, 1)), ["source_missing_judgment_result"])

    def test_missing_reasoning_is_still_api_eligible(self) -> None:
        case = raw_case()
        evidence_by_id(case)["E-Identified"]["raw_span"] = ""
        self.assertEqual(v2.source_exclusion_reasons(v2.prepare_source_record(case, 1)), [])

    def test_missing_case_id_is_excluded_before_api(self) -> None:
        case = raw_case()
        case["case_metadata"]["case_id"] = ""
        self.assertEqual(
            v2.source_exclusion_reasons(v2.prepare_source_record(case, 1)),
            ["source_missing_case_id"],
        )


class PromptContractTests(unittest.TestCase):
    def test_payload_contains_only_required_sources_and_final_gold_schema(self) -> None:
        payload = v2.extraction_payload(eligible_source())
        self.assertEqual(
            set(payload["input"]),
            {"case_type", "plaintiff_statement", "court_reasoning", "judgment_result"},
        )
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("defendant_statement", serialized)
        self.assertNotIn("court_found_facts", serialized)
        self.assertNotIn("law_candidates", serialized)
        self.assertNotIn("partial_gold_output", serialized)
        self.assertNotIn("quality_control", serialized)
        self.assertEqual(set(payload["output_schema"]), {"claim_results", "payment_result"})
        self.assertEqual(payload["output_schema"]["claim_results"][0]["claim_id"], "M1")
        self.assertIn("禁止输出斜杠", serialized)

    def test_prompt_forbids_second_generation_and_unallocated_guessing(self) -> None:
        prompt_text = v2.system_prompt()
        self.assertIn("一次", prompt_text)
        self.assertIn("不得平均分配", prompt_text)
        self.assertIn("每个编号诉请至少成项", prompt_text)
        self.assertIn("停止与删除", prompt_text)
        self.assertIn("独立项目名称和独立请求金额", prompt_text)
        self.assertIn("保留为一个复合金额诉请", prompt_text)
        self.assertIn("必须将该总判赔写入该诉请的awarded_amount", prompt_text)
        self.assertIn("绝对不得用其中的项目或金额补造", prompt_text)
        self.assertIn("任一维度缩减均为partial_support", prompt_text)
        self.assertIn("独立依据judgment_result", prompt_text)
        self.assertIn("不得纳入共同被告之间返还垫付款", prompt_text)
        self.assertIn("禁止跨编号拼接", prompt_text)
        self.assertIn("只输出严格JSON", prompt_text)

    def test_parse_json_content_is_strict_object_json(self) -> None:
        self.assertEqual(v2.parse_json_content('{"claim_results":[],"payment_result":{}}')["claim_results"], [])
        with self.assertRaises(TypeError):
            v2.parse_json_content("[]")
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant), self.assertRaises(json.JSONDecodeError):
                v2.parse_json_content(f'{{"value":{constant}}}')


class NormalizationTests(unittest.TestCase):
    def test_normalization_cleans_strings_and_numeric_strings_only(self) -> None:
        raw = complete_model_output()
        raw["claim_results"][0]["claim_id"] = "  M1  "
        raw["claim_results"][0]["requested_amount"] = "1,257.00"
        normalized = v2.normalize_model_output(raw)
        self.assertEqual(normalized["claim_results"][0]["claim_id"], "M1")
        self.assertEqual(normalized["claim_results"][0]["requested_amount"], 1257.0)

    def test_normalization_does_not_infer_missing_values(self) -> None:
        raw = complete_model_output()
        del raw["claim_results"][0]["outcome"]
        raw["claim_results"][0]["claim_id"] = ""
        normalized = v2.normalize_model_output(raw)
        self.assertNotIn("outcome", normalized["claim_results"][0])
        self.assertEqual(normalized["claim_results"][0]["claim_id"], "")

    def test_rejected_monetary_null_award_is_normalized_to_zero(self) -> None:
        raw = complete_model_output()
        raw["claim_results"][1]["awarded_amount"] = None
        normalized = v2.normalize_model_output(raw)
        self.assertEqual(normalized["claim_results"][1]["awarded_amount"], 0.0)

    def test_false_payment_zero_total_is_normalized_to_null(self) -> None:
        raw = complete_model_output()
        raw["payment_result"] = {
            "has_payment": False,
            "total_awarded_amount": 0,
            "obligations": [],
        }
        normalized = v2.normalize_model_output(raw)
        self.assertIsNone(normalized["payment_result"]["total_awarded_amount"])

    def test_false_payment_zero_is_not_changed_when_obligations_exist(self) -> None:
        raw = complete_model_output()
        raw["payment_result"]["has_payment"] = False
        raw["payment_result"]["total_awarded_amount"] = 0
        normalized = v2.normalize_model_output(raw)
        self.assertEqual(normalized["payment_result"]["total_awarded_amount"], 0.0)

    def test_single_monetary_claim_uses_its_only_linked_obligation_amount(self) -> None:
        raw = complete_model_output()
        claim = raw["claim_results"][2]
        claim["claim_id"] = "M1"
        claim["requested_amount"] = 1000.0
        claim["awarded_amount"] = None
        raw["claim_results"] = [claim]
        raw["payment_result"]["total_awarded_amount"] = 600.0
        raw["payment_result"]["obligations"][0]["related_claim_ids"] = ["M1"]
        raw["payment_result"]["obligations"][0]["amount"] = 600.0
        normalized = v2.normalize_model_output(raw)
        self.assertEqual(normalized["claim_results"][0]["awarded_amount"], 600.0)

    def test_single_claim_award_is_not_inferred_from_ambiguous_obligations(self) -> None:
        raw = complete_model_output()
        claim = raw["claim_results"][2]
        claim["claim_id"] = "M1"
        claim["requested_amount"] = 1000.0
        claim["awarded_amount"] = None
        raw["claim_results"] = [claim]
        raw["payment_result"]["obligations"][0]["related_claim_ids"] = ["M1"]
        raw["payment_result"]["obligations"].append(dict(raw["payment_result"]["obligations"][0]))
        normalized = v2.normalize_model_output(raw)
        self.assertIsNone(normalized["claim_results"][0]["awarded_amount"])

    def test_mixed_claim_types_are_mechanically_renumbered_and_refs_updated(self) -> None:
        raw = complete_model_output()
        raw["claim_results"] = [
            {**raw["claim_results"][0], "claim_id": "M1"},
            {
                "claim_id": "M2", "claim_type": "non_monetary", "item": "停止侵权",
                "request_text": "医疗费1257元", "requested_amount": None,
                "outcome": "full_support", "awarded_amount": None,
                "operation": "ORDER_PERFORMANCE", "decision_text": "医疗费1257元予以支持",
            },
            {
                "claim_id": "M3", "claim_type": "non_monetary", "item": "删除内容",
                "request_text": "误工费680元", "requested_amount": None,
                "outcome": "rejected", "awarded_amount": None,
                "operation": "DISMISS", "decision_text": "误工费不予支持",
            },
            {**raw["claim_results"][2], "claim_id": "M4"},
        ]
        raw["payment_result"]["obligations"][0]["related_claim_ids"] = ["M1", "M4"]

        normalized = v2.normalize_model_output(raw)

        self.assertEqual(
            [claim["claim_id"] for claim in normalized["claim_results"]],
            ["M1", "N1", "N2", "M2"],
        )
        self.assertEqual(
            normalized["payment_result"]["obligations"][0]["related_claim_ids"],
            ["M1", "M2"],
        )

    def test_duplicate_old_ids_are_not_guessed_or_remapped(self) -> None:
        raw = complete_model_output()
        raw["claim_results"][1]["claim_id"] = "M1"
        raw["payment_result"]["obligations"][0]["related_claim_ids"] = ["M1"]

        normalized = v2.normalize_model_output(raw)

        self.assertEqual(
            [claim["claim_id"] for claim in normalized["claim_results"]],
            ["M1", "M1", "M3"],
        )
        self.assertEqual(
            normalized["payment_result"]["obligations"][0]["related_claim_ids"],
            ["M1"],
        )
        result = v2.assemble_result(eligible_source(), raw)
        quality = v2.validate_output(eligible_source(), result)
        self.assertIn("M1_duplicate_claim_id", quality["issues"])

    def test_invalid_type_blocks_all_renumbering_and_reference_changes(self) -> None:
        raw = complete_model_output()
        raw["claim_results"][1]["claim_type"] = "other"
        raw["claim_results"][2]["claim_id"] = "N7"
        raw["payment_result"]["obligations"][0]["related_claim_ids"] = ["N7"]

        normalized = v2.normalize_model_output(raw)

        self.assertEqual(
            [claim["claim_id"] for claim in normalized["claim_results"]],
            ["M1", "M2", "N7"],
        )
        self.assertEqual(
            normalized["payment_result"]["obligations"][0]["related_claim_ids"],
            ["N7"],
        )
        result = v2.assemble_result(eligible_source(), raw)
        self.assertIn(
            "M2_claim_type_invalid",
            v2.validate_output(eligible_source(), result)["issues"],
        )

    def test_unknown_related_reference_is_not_rewritten(self) -> None:
        raw = complete_model_output()
        raw["claim_results"][2]["claim_id"] = "M9"
        raw["payment_result"]["obligations"][0]["related_claim_ids"] = ["M1", "M404"]

        normalized = v2.normalize_model_output(raw)

        self.assertEqual(
            [claim["claim_id"] for claim in normalized["claim_results"]],
            ["M1", "M2", "M9"],
        )
        self.assertEqual(
            normalized["payment_result"]["obligations"][0]["related_claim_ids"],
            ["M1", "M404"],
        )
        result = v2.assemble_result(eligible_source(), raw)
        self.assertIn(
            "O1_related_claim_id_not_supported",
            v2.validate_output(eligible_source(), result)["issues"],
        )

    def test_unknown_reference_cannot_collide_with_generated_new_id(self) -> None:
        raw = complete_model_output()
        raw["claim_results"][2]["claim_id"] = "M9"
        raw["payment_result"]["obligations"][0]["related_claim_ids"] = ["M1", "M3"]

        normalized = v2.normalize_model_output(raw)

        self.assertEqual(
            [claim["claim_id"] for claim in normalized["claim_results"]],
            ["M1", "M2", "M9"],
        )
        self.assertEqual(
            normalized["payment_result"]["obligations"][0]["related_claim_ids"],
            ["M1", "M3"],
        )
        result = v2.assemble_result(eligible_source(), raw)
        quality = v2.validate_output(eligible_source(), result)
        self.assertFalse(quality["passed"])
        self.assertIn("O1_related_claim_id_not_supported", quality["issues"])

    def test_invalid_old_id_blocks_all_renumbering(self) -> None:
        raw = complete_model_output()
        raw["claim_results"][0]["claim_id"] = "claim-one"
        raw["payment_result"]["obligations"][0]["related_claim_ids"] = [
            "claim-one", "M3",
        ]

        normalized = v2.normalize_model_output(raw)

        self.assertEqual(
            [claim["claim_id"] for claim in normalized["claim_results"]],
            ["claim-one", "M2", "M3"],
        )
        self.assertEqual(
            normalized["payment_result"]["obligations"][0]["related_claim_ids"],
            ["claim-one", "M3"],
        )
        result = v2.assemble_result(eligible_source(), raw)
        self.assertIn(
            "claim-one_claim_id_invalid",
            v2.validate_output(eligible_source(), result)["issues"],
        )

    def test_assemble_result_preserves_exact_pre_qc_shape(self) -> None:
        source = eligible_source()
        original_train_input = copy.deepcopy(source["train_input"])
        assembled = v2.assemble_result(source, complete_model_output())
        self.assertEqual(set(assembled), {"schema_version", "queue_index", "case_id", "train_input", "gold_output"})
        self.assertEqual(assembled["train_input"], original_train_input)


class BasicClaimQCTests(unittest.TestCase):
    def test_claims_must_be_nonempty_objects(self) -> None:
        self.assertIn("claim_results_empty", v2.validate_claims(eligible_source(), []))
        self.assertIn("claim_1_not_object", v2.validate_claims(eligible_source(), ["bad"]))

    def test_ids_enums_required_text_and_grounding_are_checked(self) -> None:
        claims = complete_model_output()["claim_results"]
        claims[0].update(claim_id="M2", item="", request_text="不存在", outcome="bad")
        issues = v2.validate_claims(eligible_source(), claims)
        self.assertIn("monetary_claim_id_sequence_invalid", issues)
        self.assertIn("M2_item_empty", issues)
        self.assertIn("M2_request_text_not_grounded", issues)
        self.assertIn("M2_outcome_invalid", issues)

    def test_not_addressed_is_review_required(self) -> None:
        issues = v2.validate_claims(eligible_source(), invalid_but_parseable_output()["claim_results"])
        self.assertIn("M1_not_addressed_review", issues)
        self.assertNotIn("M1_decision_text_empty", issues)

    def test_removed_semantic_completeness_rules_do_not_block(self) -> None:
        issues = v2.validate_claims(eligible_source(), complete_model_output()["claim_results"][:2])
        self.assertFalse(any(issue.startswith("uncovered_") for issue in issues))


class ClaimGranularityQCTests(unittest.TestCase):
    def test_three_itemized_source_amounts_collapsed_to_one_claim_routes_review(self) -> None:
        output = complete_model_output()
        output["claim_results"] = [{
            "claim_id": "M1", "claim_type": "monetary", "item": "各项损失",
            "request_text": "赔偿医疗费1257元；2.赔偿误工费680元；3.赔偿交通费200元",
            "requested_amount": 2137.0, "outcome": "partial_support",
            "awarded_amount": 1357.0, "operation": "AWARD_MONEY",
            "decision_text": "被告赵某赔偿原告钱某1357元",
        }]
        output["payment_result"]["obligations"][0]["related_claim_ids"] = ["M1"]
        result = v2.assemble_result(eligible_source(), output)
        self.assertIn(
            "source_itemized_money_collapsed_review",
            v2.validate_output(eligible_source(), result)["issues"],
        )

    def test_distinct_nonmoney_reliefs_collapsed_routes_review(self) -> None:
        source = combined_award_source()
        output = combined_award_output()
        output["claim_results"][:2] = [{
            "claim_id": "N1", "claim_type": "non_monetary",
            "item": "停止侵权并删除涉案图片",
            "request_text": "停止侵权并删除涉案图片",
            "requested_amount": None, "outcome": "full_support",
            "awarded_amount": None, "operation": "ORDER_PERFORMANCE",
            "decision_text": "被告停止侵权并删除涉案图片",
        }]
        result = v2.assemble_result(source, output)
        self.assertIn(
            "N1_independent_reliefs_collapsed_review",
            v2.validate_output(source, result)["issues"],
        )

    def test_two_money_requests_are_not_overblocked(self) -> None:
        source = combined_award_source()
        self.assertEqual(
            v2.validate_claim_granularity(source, combined_award_output()["claim_results"]),
            [],
        )

    def test_requested_amount_must_appear_in_its_request_span(self) -> None:
        claims = complete_model_output()["claim_results"]
        claims[0]["request_text"] = "请求赔偿各项损失共计3000元"
        claims[0]["requested_amount"] = 1257.0
        self.assertIn(
            "M1_requested_amount_not_in_request_text",
            v2.validate_claim_granularity(eligible_source(), claims),
        )

    def test_comma_and_wan_amounts_are_recognized_in_request_span(self) -> None:
        claim = {
            "claim_id": "M1",
            "claim_type": "monetary",
            "request_text": "赔偿经济损失1万元及费用12,000元",
            "requested_amount": 10000.0,
        }
        self.assertNotIn(
            "M1_requested_amount_not_in_request_text",
            v2.validate_claim_granularity(eligible_source(), [claim]),
        )

    def test_full_nonmoney_support_with_missing_platform_routes_review(self) -> None:
        claim = {
            "claim_id": "N1",
            "claim_type": "non_monetary",
            "request_text": "删除抖音、微信朋友圈、公众号发布的内容",
            "outcome": "full_support",
            "decision_text": "删除微信朋友圈文章以及抖音作品",
        }
        self.assertIn(
            "N1_full_support_scope_narrowed_review",
            v2.validate_nonmoney_scope([claim]),
        )

    def test_full_nonmoney_support_with_shorter_duration_routes_review(self) -> None:
        claim = {
            "claim_id": "N1",
            "claim_type": "non_monetary",
            "request_text": "在微信朋友圈公开置顶三个月",
            "outcome": "full_support",
            "decision_text": "在微信朋友圈公开七日",
        }
        self.assertIn(
            "N1_full_support_duration_narrowed_review",
            v2.validate_nonmoney_scope([claim]),
        )


class AmountAndPaymentQCTests(unittest.TestCase):
    def test_monetary_outcome_amount_and_operation_pairs(self) -> None:
        claims = complete_model_output()["claim_results"]
        claims[0]["awarded_amount"] = 1200.0
        claims[1]["awarded_amount"] = None
        claims[2]["awarded_amount"] = 200.0
        issues = v2.validate_claim_amounts(claims)
        self.assertIn("M1_full_support_amount_invalid", issues)
        self.assertIn("M2_rejected_amount_must_be_zero", issues)
        self.assertIn("M3_partial_support_amount_invalid", issues)

    def test_requested_amount_is_numeric_nonnegative_and_nonmoney_is_null(self) -> None:
        claims = complete_model_output()["claim_results"]
        claims[0]["requested_amount"] = -1.0
        claims.append({"claim_id": "N1", "claim_type": "non_monetary", "item": "停止侵害", "request_text": "停止侵害", "requested_amount": 1.0, "outcome": "rejected", "awarded_amount": None, "operation": "DISMISS", "decision_text": "驳回停止侵害"})
        issues = v2.validate_claim_amounts(claims)
        self.assertIn("M1_requested_amount_invalid", issues)
        self.assertIn("N1_non_monetary_amount_forbidden", issues)

    def test_valid_payment_and_combined_obligation_pass(self) -> None:
        output = complete_model_output()
        self.assertEqual(v2.validate_payment(eligible_source(), output["claim_results"], output["payment_result"]), [])

    def test_queue265_style_unallocated_combined_award_has_single_review_issue(self) -> None:
        source = combined_award_source()
        result = v2.assemble_result(source, combined_award_output())
        quality = v2.validate_output(source, result)
        self.assertFalse(quality["passed"])
        self.assertEqual(quality["issues"], ["combined_award_unallocated_review"])
        self.assertEqual(
            set(quality["checks"]),
            {
                "all_claims_addressed", "amounts_consistent",
                "payment_grounded", "all_spans_grounded",
            },
        )

    def test_isolated_unallocated_partial_still_requires_review(self) -> None:
        source = combined_award_source()
        output = combined_award_output()
        output["payment_result"]["obligations"][0]["related_claim_ids"] = ["M1"]
        result = v2.assemble_result(source, output)
        issues = v2.validate_output(source, result)["issues"]
        self.assertIn("M1_partial_support_amount_invalid", issues)
        self.assertIn("M2_partial_support_amount_invalid", issues)

    def test_unallocated_residual_equal_to_or_over_requests_requires_review(self) -> None:
        source = combined_award_source()
        for total in (10000.0, 10001.0):
            with self.subTest(total=total):
                output = combined_award_output()
                output["payment_result"]["total_awarded_amount"] = total
                output["payment_result"]["obligations"][0]["amount"] = total
                result = v2.assemble_result(source, output)
                issues = v2.validate_output(source, result)["issues"]
                self.assertIn("M1_partial_support_amount_invalid", issues)
                self.assertIn("payment_total_vs_claim_awards_mismatch", issues)

    def test_unallocated_claim_referenced_by_multiple_obligations_stays_review(self) -> None:
        source = combined_award_source()
        output = combined_award_output()
        first = output["payment_result"]["obligations"][0]
        first["amount"] = 800.0
        second = copy.deepcopy(first)
        second.update(obligation_id="O2", amount=1000.0)
        output["payment_result"]["obligations"] = [first, second]
        result = v2.assemble_result(source, output)
        quality = v2.validate_output(source, result)
        issues = quality["issues"]
        self.assertFalse(quality["passed"])
        self.assertIn("M1_partial_support_amount_invalid", issues)
        self.assertIn("M2_partial_support_amount_invalid", issues)
        self.assertIn("payment_supported_claim_referenced_multiple_M1", issues)
        self.assertIn("payment_supported_claim_referenced_multiple_M2", issues)

    def test_multiple_reference_issues_follow_original_claim_order(self) -> None:
        source = combined_award_source()
        output = combined_award_output()
        monetary = output["claim_results"][2:]
        monetary.append(copy.deepcopy(monetary[0]))
        for claim, claim_id in zip(monetary, ("M10", "M2", "M1")):
            claim["claim_id"] = claim_id
        output["claim_results"] = monetary
        first = output["payment_result"]["obligations"][0]
        first["related_claim_ids"] = ["M10", "M2", "M1"]
        first["amount"] = 800.0
        second = copy.deepcopy(first)
        second.update(obligation_id="O2", amount=1000.0)
        output["payment_result"]["obligations"] = [first, second]

        issues = v2.validate_payment(
            source,
            output["claim_results"],
            output["payment_result"],
        )

        self.assertEqual(
            [issue for issue in issues if "referenced_multiple" in issue],
            [
                "payment_supported_claim_referenced_multiple_M10",
                "payment_supported_claim_referenced_multiple_M2",
                "payment_supported_claim_referenced_multiple_M1",
            ],
        )

    def test_numeric_supported_claim_cannot_mix_with_unallocated_group(self) -> None:
        source = combined_award_source()
        source["train_input"]["plaintiff_statement"] += "另赔偿交通费200元。"
        source["court_reasoning"] += "交通费支持100元。"
        source["judgment_result"] = source["judgment_result"].replace(
            "共计1800元", "共计1900元"
        )
        output = combined_award_output()
        output["claim_results"].append({
            "claim_id": "M3", "claim_type": "monetary", "item": "交通费",
            "request_text": "交通费200元", "requested_amount": 200.0,
            "outcome": "partial_support", "awarded_amount": 100.0,
            "operation": "AWARD_MONEY", "decision_text": "交通费支持100元",
        })
        obligation = output["payment_result"]["obligations"][0]
        obligation["related_claim_ids"].append("M3")
        obligation["amount"] = 1900.0
        obligation["ruling_text"] = "被告赔偿原告经济损失及合理费用共计1900元"
        output["payment_result"]["total_awarded_amount"] = 1900.0
        result = v2.assemble_result(source, output)
        quality = v2.validate_output(source, result)
        self.assertFalse(quality["passed"])
        self.assertIn("M1_partial_support_amount_invalid", quality["issues"])

    def test_non_string_party_survives_normalization_and_fails_qc(self) -> None:
        raw = complete_model_output()
        raw["payment_result"]["obligations"][0]["payer_names"] = [123]
        result = v2.assemble_result(eligible_source(), raw)
        self.assertEqual(
            result["gold_output"]["payment_result"]["obligations"][0]["payer_names"],
            [123],
        )
        quality = v2.validate_output(eligible_source(), result)
        self.assertIn("O1_payer_names_invalid", quality["issues"])

    def test_decimal_obligation_and_claim_sums_equal_declared_total(self) -> None:
        output = complete_model_output()
        first, _, third = output["claim_results"]
        first.update(requested_amount=0.1, awarded_amount=0.1)
        third.update(
            requested_amount=0.2,
            outcome="full_support",
            awarded_amount=0.2,
        )
        ruling = output["payment_result"]["obligations"][0]["ruling_text"]
        output["payment_result"] = {
            "has_payment": True,
            "total_awarded_amount": 0.3,
            "obligations": [
                {"obligation_id": "O1", "payer_names": ["赵某"], "payee_names": ["钱某"], "related_claim_ids": ["M1"], "amount": 0.1, "liability_mode": "individual", "ruling_text": ruling},
                {"obligation_id": "O2", "payer_names": ["赵某"], "payee_names": ["钱某"], "related_claim_ids": ["M3"], "amount": 0.2, "liability_mode": "individual", "ruling_text": ruling},
            ],
        }
        self.assertEqual(
            v2.validate_payment(
                eligible_source(),
                output["claim_results"],
                output["payment_result"],
            ),
            [],
        )

    def test_payment_rejects_directive_party_bad_refs_and_bad_total(self) -> None:
        output = complete_model_output()
        obligation = output["payment_result"]["obligations"][0]
        obligation["payer_names"] = ["限被告赵某"]
        obligation["related_claim_ids"] = ["M2", "M9"]
        output["payment_result"]["total_awarded_amount"] = 1.0
        issues = v2.validate_payment(eligible_source(), output["claim_results"], output["payment_result"])
        self.assertIn("O1_payer_name_contains_directive", issues)
        self.assertIn("O1_related_claim_id_not_supported", issues)
        self.assertIn("payment_total_vs_obligations_mismatch", issues)
        self.assertIn("payment_total_vs_claim_awards_mismatch", issues)

    def test_false_payment_requires_null_total_and_empty_obligations(self) -> None:
        issues = v2.validate_payment(eligible_source(), [], {"has_payment": False, "total_awarded_amount": 0.0, "obligations": [{}]})
        self.assertIn("payment_false_total_must_be_null", issues)
        self.assertIn("payment_false_obligations_must_be_empty", issues)

    def test_validate_output_adds_exact_final_shape(self) -> None:
        result = v2.assemble_result(eligible_source(), complete_model_output())
        result["quality_control"] = v2.validate_output(eligible_source(), result)
        self.assertTrue(result["quality_control"]["passed"])
        self.assertEqual(set(result), {"schema_version", "queue_index", "case_id", "train_input", "gold_output", "quality_control"})

    def test_missing_court_reasoning_is_always_review_required(self) -> None:
        source = eligible_source()
        source["court_reasoning"] = ""
        source["judgment_result"] += "医疗费1257元予以支持；误工费不予支持；交通费酌情支持100元。"
        result = v2.assemble_result(source, complete_model_output())
        quality = v2.validate_output(source, result)
        self.assertFalse(quality["passed"])
        self.assertIn("source_missing_court_reasoning_review", quality["issues"])


class FakeHttpResponse:
    def __init__(self, value: dict):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.value, ensure_ascii=False).encode("utf-8")


def successful_response(raw: dict) -> FakeHttpResponse:
    return FakeHttpResponse({"id": "response-1", "choices": [{"message": {"content": json.dumps(raw, ensure_ascii=False)}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})


class RecordingOpener:
    def __init__(self, actions: list):
        self.actions = list(actions)
        self.request_bodies: list[bytes] = []

    def __call__(self, request, timeout: int):
        self.request_bodies.append(request.data)
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action


class RuntimeTests(unittest.TestCase):
    def test_request_body_has_deterministic_one_shot_contract(self) -> None:
        body = json.loads(v2.build_request_body("glm-test", "system", {"x": 1}))
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 8192)
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_qwen_37_max_snapshot_omits_unsupported_response_format(self) -> None:
        body = json.loads(
            v2.build_request_body(
                "qwen3.7-max-2026-05-17",
                "system",
                {"x": 1},
            )
        )
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 8192)
        self.assertNotIn("response_format", body)

    def test_qwen_37_plus_enables_supported_json_response_format(self) -> None:
        body = json.loads(
            v2.build_request_body("qwen3.7-plus", "输出JSON", {"x": 1})
        )
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_transport_retries_reuse_byte_identical_body(self) -> None:
        opener = RecordingOpener([TimeoutError(), successful_response(complete_model_output())])
        raw, audit = v2.call_chat_api(api_key="secret", endpoint="https://example.test", model="glm-test", system_prompt_text=v2.system_prompt(), payload={"x": 1}, timeout=1, retries=1, opener=opener, sleeper=lambda _: None)
        self.assertEqual(raw["claim_results"][0]["claim_id"], "M1")
        self.assertEqual(opener.request_bodies[0], opener.request_bodies[1])
        self.assertEqual(audit["network_attempts"], 2)
        self.assertEqual(audit["semantic_calls"], 1)
        self.assertNotIn("secret", json.dumps(audit))

    def test_qc_failure_does_not_trigger_second_generation(self) -> None:
        caller = Mock(return_value=(invalid_but_parseable_output(), {"network_attempts": 1, "semantic_calls": 1, "status": "received", "usage": {}}))
        result, audit = v2.run_record(eligible_source(), api_key="secret", api_caller=caller)
        self.assertEqual(caller.call_count, 1)
        self.assertFalse(result["quality_control"]["passed"])
        self.assertEqual(audit["semantic_calls"], 1)

    def test_api_error_returns_review_and_audit(self) -> None:
        caller = Mock(side_effect=ValueError("bad response"))
        result, audit = v2.run_record(eligible_source(), api_key="secret", api_caller=caller)
        self.assertEqual(caller.call_count, 1)
        self.assertEqual(result["quality_control"]["issues"], ["api_error_ValueError"])
        self.assertEqual(audit["status"], "error")

    def test_content_parse_error_keeps_envelope_id_and_usage(self) -> None:
        opener = RecordingOpener([FakeHttpResponse({
            "id": "resp-kept",
            "choices": [{"message": {"content": "{not-json"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        })])
        with self.assertRaises(v2.ChatApiFailure) as caught:
            v2.call_chat_api(
                api_key="secret", endpoint="https://example.test", model="glm-test",
                system_prompt_text=v2.system_prompt(), payload={"x": 1}, timeout=1,
                retries=0, opener=opener, sleeper=lambda _: None,
            )
        self.assertEqual(caught.exception.audit["response_id"], "resp-kept")
        self.assertEqual(
            caught.exception.audit["usage"],
            {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        )


def cli_paths(tmp: str) -> dict[str, Path]:
    root = Path(tmp)
    return {name: root / filename for name, filename in {"input": "input.jsonl", "passed": "passed.jsonl", "review": "review.jsonl", "excluded": "excluded.jsonl", "audit": "audit.jsonl", "api_input": "api-input.jsonl", "report": "report.json", "pretty": "pretty.json"}.items()}


def invoke_main(
    tmp: str,
    cases: list[dict],
    *,
    prepare_only: bool = False,
    resume: bool = False,
    outputs: list[dict] | None = None,
    model: str | None = None,
) -> tuple[dict[str, Path], int, int]:
    paths = cli_paths(tmp)
    if not resume:
        write_jsonl(paths["input"], cases)
    argv = ["lexchain_core_gold_api_v2.py", "--input", str(paths["input"]), "--output", str(paths["passed"]), "--review-output", str(paths["review"]), "--excluded-output", str(paths["excluded"]), "--audit-output", str(paths["audit"]), "--api-input-output", str(paths["api_input"]), "--report-output", str(paths["report"]), "--pretty-output", str(paths["pretty"]), "--api-key-env", "TEST_API_KEY"]
    if prepare_only:
        argv.append("--prepare-only")
    if resume:
        argv.append("--resume")
    if model is not None:
        argv.extend(["--model", model])
    side_effect = [(value, {"network_attempts": 1, "semantic_calls": 1, "status": "received", "usage": {}}) for value in (outputs or [])]
    with patch.object(sys, "argv", argv), patch.dict("os.environ", {"TEST_API_KEY": "secret"}), patch.object(v2, "call_chat_api", side_effect=side_effect) as caller:
        exit_code = v2.main()
    return paths, caller.call_count, exit_code


class CliTests(unittest.TestCase):
    def test_prepare_only_partitions_without_api(self) -> None:
        excluded = copy.deepcopy(raw_case())
        excluded["queue_index"] = 2
        excluded["case_metadata"]["case_id"] = "case-002"
        evidence_by_id(excluded)["E-Ascertain"]["raw_span"] = ""
        evidence_by_id(excluded)["E-Identified"]["raw_span"] = "本院认为被告担责。"
        with tempfile.TemporaryDirectory() as tmp:
            paths, calls, code = invoke_main(tmp, [raw_case(), excluded], prepare_only=True)
            report = read_json(paths["report"])
            self.assertEqual((report["selected"], report["eligible"], report["source_excluded"]), (2, 1, 1))
            self.assertEqual((calls, code), (0, 0))
            self.assertEqual(len(read_jsonl(paths["api_input"])), 1)
            self.assertEqual(len(read_jsonl(paths["excluded"])), 1)

    def test_pass_review_routing_and_resume(self) -> None:
        second = copy.deepcopy(raw_case())
        second["queue_index"] = 2
        second["case_metadata"]["case_id"] = "case-002"
        with tempfile.TemporaryDirectory() as tmp:
            paths, calls, code = invoke_main(tmp, [raw_case(), second], outputs=[complete_model_output(), invalid_but_parseable_output()])
            self.assertEqual((calls, code), (2, 0))
            self.assertEqual(len(read_jsonl(paths["passed"])), 1)
            self.assertEqual(len(read_jsonl(paths["review"])), 1)
            self.assertEqual(len(read_jsonl(paths["audit"])), 2)
            _, resumed_calls, resumed_code = invoke_main(tmp, [], resume=True)
            self.assertEqual((resumed_calls, resumed_code), (0, 0))
            self.assertEqual(read_json(paths["report"])["api_attempted"], 2)

    def test_result_is_completion_key_when_audit_commit_fails_then_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = cli_paths(tmp)
            second = copy.deepcopy(raw_case())
            second["queue_index"] = 2
            second["case_metadata"]["case_id"] = "case-002"
            write_jsonl(paths["input"], [raw_case(), second])
            argv = [
                "lexchain_core_gold_api_v2.py", "--input", str(paths["input"]),
                "--output", str(paths["passed"]), "--review-output", str(paths["review"]),
                "--excluded-output", str(paths["excluded"]), "--audit-output", str(paths["audit"]),
                "--api-input-output", str(paths["api_input"]), "--report-output", str(paths["report"]),
                "--api-key-env", "TEST_API_KEY", "--workers", "2",
                "--model", "model-a",
            ]
            original_commit = v2.JsonlCommitWriter.commit

            def fail_audit_commit(writer, record):
                if writer.path == paths["audit"]:
                    raise OSError("simulated audit disk failure")
                return original_commit(writer, record)

            first_audit = {
                "network_attempts": 1, "semantic_calls": 1,
                "status": "received", "usage": {},
            }
            with (
                patch.object(sys, "argv", argv),
                patch.dict("os.environ", {"TEST_API_KEY": "secret"}),
                patch.object(v2, "PROMPT_VERSION", "prompt-a"),
                patch.object(v2, "call_chat_api", return_value=(complete_model_output(), first_audit)),
                patch.object(v2.JsonlCommitWriter, "commit", autospec=True, side_effect=fail_audit_commit),
                self.assertRaises(OSError),
            ):
                v2.main()

            self.assertEqual(len(read_jsonl(paths["passed"])), 2)
            with patch.object(v2, "PROMPT_VERSION", "prompt-b"):
                _, resumed_calls, resumed_code = invoke_main(
                    tmp,
                    [],
                    resume=True,
                    model="model-b",
                )
            self.assertEqual((resumed_calls, resumed_code), (0, 0))
            recovered_audits = read_jsonl(paths["audit"])
            self.assertEqual(len(recovered_audits), 2)
            self.assertTrue(
                all(audit["status"] == "local_audit_missing" for audit in recovered_audits)
            )
            self.assertTrue(all(audit["model"] == "model-a" for audit in recovered_audits))
            self.assertTrue(
                all(audit["prompt_version"] == "prompt-a" for audit in recovered_audits)
            )

    def test_result_commit_failure_becomes_ambiguous_review_without_second_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = cli_paths(tmp)
            write_jsonl(paths["input"], [raw_case()])
            argv = [
                "lexchain_core_gold_api_v2.py", "--input", str(paths["input"]),
                "--output", str(paths["passed"]), "--review-output", str(paths["review"]),
                "--excluded-output", str(paths["excluded"]), "--audit-output", str(paths["audit"]),
                "--api-input-output", str(paths["api_input"]), "--report-output", str(paths["report"]),
                "--api-key-env", "TEST_API_KEY", "--model", "model-a",
            ]
            original_commit = v2.JsonlCommitWriter.commit

            def fail_result_commit(writer, record):
                if writer.path == paths["passed"]:
                    raise OSError("simulated result disk failure")
                return original_commit(writer, record)

            first_audit = {
                "network_attempts": 1, "semantic_calls": 1,
                "status": "received", "usage": {},
            }
            with (
                patch.object(sys, "argv", argv),
                patch.dict("os.environ", {"TEST_API_KEY": "secret"}),
                patch.object(v2, "PROMPT_VERSION", "prompt-a"),
                patch.object(v2, "call_chat_api", return_value=(complete_model_output(), first_audit)) as first_caller,
                patch.object(v2.JsonlCommitWriter, "commit", autospec=True, side_effect=fail_result_commit),
                self.assertRaises(OSError),
            ):
                v2.main()
            self.assertEqual(first_caller.call_count, 1)
            self.assertEqual(read_jsonl(paths["passed"]), [])
            self.assertEqual(read_jsonl(paths["review"]), [])
            journal_path = Path(str(paths["audit"]) + ".attempt-journal.jsonl")
            journal_rows = read_jsonl(journal_path)
            self.assertEqual([row["state"] for row in journal_rows], ["started"])
            self.assertEqual(journal_rows[0]["model"], "model-a")
            self.assertEqual(journal_rows[0]["prompt_version"], "prompt-a")

            with patch.object(v2, "PROMPT_VERSION", "prompt-b"):
                _, resumed_calls, resumed_code = invoke_main(
                    tmp,
                    [],
                    resume=True,
                    model="model-b",
                )
            self.assertEqual((resumed_calls, resumed_code), (0, 0))
            reviews = read_jsonl(paths["review"])
            self.assertEqual(len(reviews), 1)
            self.assertEqual(
                reviews[0]["quality_control"]["issues"],
                ["runtime_ambiguous_attempt"],
            )
            audits = read_jsonl(paths["audit"])
            self.assertEqual(len(audits), 1)
            self.assertEqual(audits[0]["status"], "ambiguous_attempt")
            self.assertEqual(audits[0]["model"], "model-a")
            self.assertEqual(audits[0]["prompt_version"], "prompt-a")

    def test_ambiguous_review_survives_journal_completion_failure_truthfully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = cli_paths(tmp)
            write_jsonl(paths["input"], [raw_case()])
            journal_path = Path(str(paths["audit"]) + ".attempt-journal.jsonl")
            base_argv = [
                "lexchain_core_gold_api_v2.py", "--input", str(paths["input"]),
                "--output", str(paths["passed"]), "--review-output", str(paths["review"]),
                "--excluded-output", str(paths["excluded"]), "--audit-output", str(paths["audit"]),
                "--api-input-output", str(paths["api_input"]), "--report-output", str(paths["report"]),
                "--api-key-env", "TEST_API_KEY", "--model", "model-a",
            ]
            original_commit = v2.JsonlCommitWriter.commit

            def fail_result_commit(writer, record):
                if writer.path == paths["passed"]:
                    raise OSError("simulated result disk failure")
                return original_commit(writer, record)

            with (
                patch.object(sys, "argv", base_argv),
                patch.dict("os.environ", {"TEST_API_KEY": "secret"}),
                patch.object(v2, "PROMPT_VERSION", "prompt-a"),
                patch.object(
                    v2,
                    "call_chat_api",
                    return_value=(
                        complete_model_output(),
                        {"network_attempts": 1, "semantic_calls": 1, "status": "received", "usage": {}},
                    ),
                ),
                patch.object(v2.JsonlCommitWriter, "commit", autospec=True, side_effect=fail_result_commit),
                self.assertRaises(OSError),
            ):
                v2.main()
            self.assertEqual([row["state"] for row in read_jsonl(journal_path)], ["started"])

            def fail_ambiguous_completion(writer, record):
                if (
                    writer.path == journal_path
                    and record.get("state") == "ambiguous_review"
                ):
                    raise OSError("simulated ambiguous journal failure")
                return original_commit(writer, record)

            with (
                patch.object(sys, "argv", [*base_argv, "--resume"]),
                patch.dict("os.environ", {"TEST_API_KEY": "secret"}),
                patch.object(v2, "PROMPT_VERSION", "prompt-b"),
                patch.object(v2, "call_chat_api") as second_caller,
                patch.object(v2.JsonlCommitWriter, "commit", autospec=True, side_effect=fail_ambiguous_completion),
                self.assertRaises(OSError),
            ):
                v2.main()
            self.assertEqual(second_caller.call_count, 0)
            self.assertEqual(len(read_jsonl(paths["review"])), 1)
            self.assertEqual(read_jsonl(paths["audit"]), [])

            with patch.object(v2, "PROMPT_VERSION", "prompt-c"):
                _, third_calls, third_code = invoke_main(
                    tmp,
                    [],
                    resume=True,
                    model="model-c",
                )
            self.assertEqual((third_calls, third_code), (0, 0))
            self.assertEqual(len(read_jsonl(paths["review"])), 1)
            marker = read_jsonl(paths["audit"])[0]
            self.assertEqual(marker["status"], "ambiguous_attempt")
            self.assertIsNone(marker["semantic_calls"])
            self.assertIsNone(marker["network_attempts"])
            self.assertEqual(
                marker["usage"],
                {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            )
            self.assertEqual(marker["model"], "model-a")
            self.assertEqual(marker["prompt_version"], "prompt-a")

    def test_api_is_not_called_when_pre_call_journal_commit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = cli_paths(tmp)
            write_jsonl(paths["input"], [raw_case()])
            journal_path = Path(str(paths["audit"]) + ".attempt-journal.jsonl")
            argv = [
                "lexchain_core_gold_api_v2.py", "--input", str(paths["input"]),
                "--output", str(paths["passed"]), "--review-output", str(paths["review"]),
                "--excluded-output", str(paths["excluded"]), "--audit-output", str(paths["audit"]),
                "--api-input-output", str(paths["api_input"]), "--report-output", str(paths["report"]),
                "--api-key-env", "TEST_API_KEY",
            ]
            original_commit = v2.JsonlCommitWriter.commit

            def fail_journal_commit(writer, record):
                if writer.path == journal_path:
                    raise OSError("simulated journal fsync failure")
                return original_commit(writer, record)

            with (
                patch.object(sys, "argv", argv),
                patch.dict("os.environ", {"TEST_API_KEY": "secret"}),
                patch.object(v2, "call_chat_api") as caller,
                patch.object(v2.JsonlCommitWriter, "commit", autospec=True, side_effect=fail_journal_commit),
                self.assertRaises(OSError),
            ):
                v2.main()
            self.assertEqual(caller.call_count, 0)

    def test_missing_historical_journal_provenance_is_null_not_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = cli_paths(tmp)
            write_jsonl(paths["input"], [raw_case()])
            completed = v2.assemble_result(eligible_source(), complete_model_output())
            completed["quality_control"] = v2.validate_output(eligible_source(), completed)
            write_jsonl(paths["passed"], [completed])
            with patch.object(v2, "PROMPT_VERSION", "new-prompt"):
                _, calls, code = invoke_main(
                    tmp,
                    [],
                    resume=True,
                    model="new-model",
                )
            self.assertEqual((calls, code), (0, 0))
            marker = read_jsonl(paths["audit"])[0]
            self.assertIsNone(marker["model"])
            self.assertIsNone(marker["prompt_version"])
            self.assertIsNone(marker["semantic_calls"])

    def test_prepare_only_refuses_nonempty_existing_bundle_without_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = cli_paths(tmp)
            write_jsonl(paths["input"], [raw_case()])
            bundle_paths = [
                paths["passed"], paths["review"], paths["excluded"],
                paths["audit"], paths["api_input"], paths["report"], paths["pretty"],
                Path(str(paths["audit"]) + ".attempt-journal.jsonl"),
            ]
            for index, path in enumerate(bundle_paths, 1):
                path.write_bytes(f"sentinel-{index}".encode("utf-8"))
            before = {path: path.read_bytes() for path in bundle_paths}
            argv = [
                "lexchain_core_gold_api_v2.py", "--input", str(paths["input"]),
                "--output", str(paths["passed"]), "--review-output", str(paths["review"]),
                "--excluded-output", str(paths["excluded"]), "--audit-output", str(paths["audit"]),
                "--api-input-output", str(paths["api_input"]), "--report-output", str(paths["report"]),
                "--pretty-output", str(paths["pretty"]), "--prepare-only",
            ]
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                v2.main()
            self.assertEqual(
                {path: path.read_bytes() for path in bundle_paths},
                before,
            )

    def test_prepare_only_refuses_existing_attempt_journal_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = cli_paths(tmp)
            write_jsonl(paths["input"], [raw_case()])
            journal_path = Path(str(paths["audit"]) + ".attempt-journal.jsonl")
            journal_path.write_bytes(b"durable-journal-sentinel")
            argv = [
                "lexchain_core_gold_api_v2.py", "--input", str(paths["input"]),
                "--output", str(paths["passed"]), "--review-output", str(paths["review"]),
                "--excluded-output", str(paths["excluded"]), "--audit-output", str(paths["audit"]),
                "--api-input-output", str(paths["api_input"]), "--report-output", str(paths["report"]),
                "--prepare-only",
            ]
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                v2.main()
            self.assertEqual(journal_path.read_bytes(), b"durable-journal-sentinel")
            self.assertFalse(paths["api_input"].exists())
            self.assertFalse(paths["excluded"].exists())

    def test_cli_exposes_only_the_bound_one_shot_options(self) -> None:
        option_strings = {option for action in v2.build_arg_parser()._actions for option in action.option_strings}
        self.assertEqual(
            {option for option in option_strings if option.startswith("--")},
            {
                "--help", "--input", "--output", "--review-output",
                "--excluded-output", "--audit-output", "--api-input-output",
                "--report-output", "--pretty-output", "--start", "--limit",
                "--queue-indices", "--workers", "--resume", "--prepare-only",
                "--model", "--endpoint", "--api-key-env", "--timeout", "--retries",
            },
        )


class ProductionPartitionTests(unittest.TestCase):
    def test_annotation_ready_2440_partition(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "annotation_ready_2440.jsonl"
        if not path.exists():
            self.skipTest("private production dataset is not included in the repository")
        cases = list(v2.iter_input_records(path))
        records = [v2.prepare_source_record(case, index + 1) for index, case in enumerate(cases)]
        reason_lists = [v2.source_exclusion_reasons(record) for record in records]
        self.assertEqual(len(records), 2440)
        self.assertEqual(sum(bool(reasons) for reasons in reason_lists), 578)
        self.assertEqual(sum(not reasons for reasons in reason_lists), 1862)
        self.assertEqual(sum("source_missing_court_facts" in reasons for reasons in reason_lists), 574)
        self.assertEqual(sum("source_missing_judgment_result" in reasons for reasons in reason_lists), 5)
        self.assertEqual(sum("source_missing_judgment_result" in reasons and "source_missing_court_facts" not in reasons for reasons in reason_lists), 4)


if __name__ == "__main__":
    unittest.main()
