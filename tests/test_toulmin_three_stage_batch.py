import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("toulmin_three_stage_batch", ROOT / "scripts" / "toulmin_three_stage_batch.py")
module = importlib.util.module_from_spec(spec); assert spec.loader
sys.modules["toulmin_three_stage_batch"] = module; spec.loader.exec_module(module)


class ThreeStageTests(unittest.TestCase):
    def test_final_prompt_requires_compact_escaped_json(self):
        prompt = module.SYSTEM[3] + module.INSTRUCTION[3]
        self.assertIn("单行紧凑JSON", prompt)
        self.assertIn("不得重复事实、规则、法条正文", prompt)
        self.assertIn("未转义", prompt)

    def test_stage1_rejects_short_or_duplicate_claim_ids(self):
        value = {
            "claim_source_units": [{"source_no": 1, "source_text": "赔偿100元", "disposition": "substantive_claim", "claim_ids": ["M"]}],
            "D1_claim": [{"claim_id": "M", "claim_type": "monetary", "item": "损失", "request_text": "赔偿100元", "requested_amount": 100, "parties": {}}],
            "D2_data": [],
        }
        with self.assertRaisesRegex(ValueError, "claim_ids"):
            module.validate(1, value)

    def test_stage3_requires_object_payment_and_exact_claim_ids(self):
        prior1 = {"D1_claim": [{"claim_id": "M1"}], "D2_data": [{"fact_id": "F1"}]}
        value = {"reasoning": {"D6_qualifier": [], "D7_conclusion": []}, "claim_results": [], "payment_result": []}
        with self.assertRaisesRegex(ValueError, "final output types"):
            module.validate(3, value, prior1, {})

    def test_stage1_requires_complete_source_coverage(self):
        value = {
            "claim_source_units": [{"source_no": 1, "source_text": "承担诉讼费", "disposition": "procedural_excluded", "claim_ids": []}],
            "D1_claim": [{"claim_id": "M1", "claim_type": "monetary", "item": "损失", "request_text": "赔偿100元", "requested_amount": 100, "parties": {}}],
            "D2_data": [],
        }
        with self.assertRaisesRegex(ValueError, "coverage"):
            module.validate(1, value)

    def test_stage2_requires_remedy_and_liability_arrays(self):
        prior1 = {"D1_claim": [{"claim_id": "M1", "requested_amount": 100}], "D2_data": [{"fact_id": "F1"}]}
        value = {"D3_warrant": [{"rule_id": "R1", "related_claim_ids": ["M1"], "element_findings": []}], "D4_backing": [], "D5_rebuttal": []}
        with self.assertRaisesRegex(ValueError, "remedy/liability"):
            module.validate(2, value, prior1, laws=set())

    def test_stage3_rejects_decision_result_mismatch(self):
        prior1 = {"D1_claim": [{"claim_id": "M1", "claim_type": "monetary", "item": "损失", "request_text": "赔偿100元", "requested_amount": 100}], "D2_data": [{"fact_id": "F1"}]}
        prior2 = {"D3_warrant": [{"rule_id": "R1"}], "D5_rebuttal": []}
        value = {
            "reasoning": {
                "decision_matrix": [{"claim_id": "M1", "outcome": "full_support", "requested_amount": 100, "calculated_amount": 90, "operation": "AWARD_MONEY", "payer_names": ["被告"], "liability_mode": "individual", "fact_ids": ["F1"], "rule_ids": ["R1"]}],
                "D6_qualifier": [{"claim_id": "M1"}],
                "D7_conclusion": [{"claim_id": "M1", "fact_ids": ["F1"], "rule_ids": ["R1"], "law_ids": [], "defense_ids": []}],
            },
            "claim_results": [{"claim_id": "M1", "claim_type": "monetary", "item": "损失", "request_text": "赔偿100元", "requested_amount": 100, "outcome": "full_support", "awarded_amount": 100, "operation": "AWARD_MONEY", "decision_reason": "支持"}],
            "payment_result": {"has_payment": True, "total_awarded_amount": 100, "obligations": []},
        }
        with self.assertRaisesRegex(ValueError, "mismatch"):
            module.validate(3, value, prior1, prior2, set())


if __name__ == "__main__":
    unittest.main()
