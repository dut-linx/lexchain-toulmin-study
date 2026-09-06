import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scorer = load("d7_api_scorer", "d7_api_scorer.py")


class ScoringTests(unittest.TestCase):
    def test_v4_does_not_make_amount_difference_automatically_fatal(self):
        prompt = scorer.system_prompt()
        self.assertIn("金额精确性另由确定性定量评分器", prompt)
        self.assertIn("单纯金额数值差异不是当然致命错误", scorer.rubric()["D5"])
        self.assertIn("v4-separate-amount", scorer.RUBRIC_VERSION)

    def test_percentage_normalization_and_na(self):
        raw = {
            "dimensions": {
                key: {"applicable": key != "D4", "score": None if key == "D4" else 80, "reason": "r"}
                for key in scorer.DIMENSIONS
            },
            "fatal_errors": [],
        }
        result = scorer.normalize_score(raw, "case")
        self.assertEqual(result["substantive_score"], 80)
        self.assertEqual(result["overall_score"], 80)
        self.assertIsNone(result["dimensions"]["D4"]["score"])

    def test_only_d4_can_be_na(self):
        raw = {
            "dimensions": {
                key: {"applicable": key != "D3", "score": None if key == "D3" else 80}
                for key in scorer.DIMENSIONS
            }
        }
        with self.assertRaises(ValueError):
            scorer.normalize_score(raw, "case")

    def test_accepts_dimensions_at_top_level(self):
        raw = {
            key: {"applicable": True, "score": 80, "reason": "r"}
            for key in scorer.DIMENSIONS
        }
        result = scorer.normalize_score(raw, "case")
        self.assertEqual(result["overall_score"], 80)

    def test_accepts_score_inside_band(self):
        raw = {
            "dimensions": {
                key: {"applicable": True, "score": 75}
                for key in scorer.DIMENSIONS
            }
        }
        result = scorer.normalize_score(raw, "case")
        self.assertEqual(result["overall_score"], 75)

    def test_rejects_out_of_range_score(self):
        raw = {
            "dimensions": {
                key: {"applicable": True, "score": 101}
                for key in scorer.DIMENSIONS
            }
        }
        with self.assertRaises(ValueError):
            scorer.normalize_score(raw, "case")


if __name__ == "__main__":
    unittest.main()
