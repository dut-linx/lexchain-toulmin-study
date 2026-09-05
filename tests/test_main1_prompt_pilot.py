import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("main1_prompt_pilot", ROOT / "scripts" / "main1_prompt_pilot.py")
pilot = importlib.util.module_from_spec(spec)
assert spec.loader
sys.modules["main1_prompt_pilot"] = pilot
spec.loader.exec_module(pilot)


class Main1PilotTests(unittest.TestCase):
    def test_seven_conditions_share_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            blind = tmp / "blind.jsonl"
            rows = [{"queue_index": i, "case_id": f"c{i}", "stratum": "ordinary", "model_input": {"case_type": "侵权"}} for i in range(2)]
            blind.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            args = type("Args", (), {"blind": blind, "output": tmp / "batch.jsonl", "report": tmp / "report.json", "limit": 2, "model": "qwen3.7-max", "max_tokens": 4500})
            self.assertEqual(pilot.prepare(args), 0)
            requests = pilot.read_jsonl(args.output)
            self.assertEqual(len(requests), 14)
            self.assertEqual({r["body"]["model"] for r in requests}, {"qwen3.7-max"})
            self.assertEqual({r["body"]["max_tokens"] for r in requests}, {4500})
            self.assertEqual({r["body"]["temperature"] for r in requests}, {0})
            self.assertEqual(len({r["custom_id"] for r in requests}), 14)


if __name__ == "__main__":
    unittest.main()
