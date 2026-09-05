import importlib.util
import json
import sys
import tempfile
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


prepare = load("prepare_blind_experiment", "prepare_blind_experiment.py")
scorer = load("d7_api_scorer", "d7_api_scorer.py")


class BlindDatasetTests(unittest.TestCase):
    def sample(self, index: int):
        return {
            "queue_index": index,
            "case_id": f"c{index}",
            "train_input": {
                "case_type": "侵权责任纠纷",
                "plaintiff_statement": "请求赔偿100元",
                "defendant_statement": "不同意",
                "court_found_facts": "查明事实",
                "court_reasoning": "不得泄漏",
                "judgment_result": "不得泄漏",
                "law_candidates": [],
            },
            "gold_output": {"claim_results": [], "payment_result": {"obligations": []}},
            "quality_control": {"passed": True},
        }

    def test_prepare_is_deterministic_and_blind(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "source.jsonl"
            source.write_text("\n".join(json.dumps(self.sample(i), ensure_ascii=False) for i in range(10)) + "\n", encoding="utf-8")
            args = type("Args", (), {"input": source, "output_dir": tmp / "out", "dev_size": 3, "seed": 7})
            self.assertEqual(prepare.prepare(args), 0)
            blind = prepare.read_jsonl(tmp / "out" / "blind_all.jsonl")
            self.assertEqual(sum(r["split"] == "development" for r in blind), 3)
            encoded = json.dumps(blind, ensure_ascii=False)
            self.assertNotIn("court_reasoning", encoded)
            self.assertNotIn("judgment_result", encoded)
            self.assertNotIn("gold_output", encoded)


class ScoringTests(unittest.TestCase):
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
