from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import lexchain_qwen_batch_v1 as batch  # noqa: E402


class QwenBatchTests(unittest.TestCase):
    def test_custom_id_round_trip(self) -> None:
        value = batch.custom_id(17, "case:with:colon")
        self.assertEqual(batch.parse_custom_id(value), (17, "case:with:colon"))

    def test_request_uses_batch_chat_contract(self) -> None:
        source = {
            "queue_index": 2,
            "case_id": "abc",
            "payload": {"task": "test", "input": {}, "output_schema": {}},
        }
        request = batch.batch_request(source, "qwen3.7-max", True)
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["url"], "/v1/chat/completions")
        self.assertEqual(request["body"]["model"], "qwen3.7-max")
        self.assertTrue(request["body"]["enable_thinking"])
        self.assertEqual(request["body"]["response_format"], {"type": "json_object"})
        self.assertEqual(len(request["body"]["messages"]), 2)

    def test_batch_content_reads_official_result_shape(self) -> None:
        content, body = batch.batch_content({
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": json.dumps({"ok": True})}}]
                },
            }
        })
        self.assertEqual(json.loads(content), {"ok": True})
        self.assertIn("choices", body)

    def test_unsuccessful_batch_row_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_response_not_successful"):
            batch.batch_content({"response": {"status_code": 400}})


if __name__ == "__main__":
    unittest.main()
