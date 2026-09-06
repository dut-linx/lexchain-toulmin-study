import importlib.util
import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("judgment_quantitative_scorer", ROOT / "scripts" / "judgment_quantitative_scorer.py")
scorer = importlib.util.module_from_spec(spec)
assert spec.loader
sys.modules["judgment_quantitative_scorer"] = scorer
spec.loader.exec_module(scorer)


class JudgmentQuantitativeScorerTests(unittest.TestCase):
    def fixture(self):
        gold = {
            "gold_output": {
                "claim_results": [{"claim_id": "M1", "claim_type": "monetary", "item": "医疗费", "request_text": "赔偿医疗费1000元", "requested_amount": 1000, "outcome": "full_support", "awarded_amount": 1000, "operation": "AWARD_MONEY"}],
                "payment_result": {"has_payment": True, "total_awarded_amount": 1000, "obligations": [{"payer_names": ["被告"], "payee_names": ["原告"], "related_claim_ids": ["M1"], "amount": 1000, "liability_mode": "individual"}]},
            }
        }
        candidate = {"case_id": "c1", "condition": "test", "model_output": copy.deepcopy(gold["gold_output"])}
        return candidate, gold

    def test_identical_prediction_scores_100(self):
        candidate, gold = self.fixture()
        result = scorer.score_case(candidate, gold)
        self.assertEqual(result["total_score"], 100)
        self.assertEqual(result["raw_metrics"]["awarded_amount_exact_rate"], 1)

    def test_wrong_outcome_and_amount_lose_points(self):
        candidate, gold = self.fixture()
        claim = candidate["model_output"]["claim_results"][0]
        claim["outcome"] = "rejected"
        claim["awarded_amount"] = 0
        candidate["model_output"]["payment_result"]["total_awarded_amount"] = 0
        result = scorer.score_case(candidate, gold)
        self.assertLess(result["total_score"], 70)

    def test_generation_failure_always_scores_zero(self):
        candidate, gold = self.fixture()
        candidate["generation_error"] = True
        result = scorer.score_case(candidate, gold)
        self.assertEqual(result["total_score"], 0)
        self.assertTrue(result["generation_error"])


if __name__ == "__main__":
    unittest.main()
