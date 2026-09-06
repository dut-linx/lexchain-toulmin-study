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


if __name__ == "__main__":
    unittest.main()
