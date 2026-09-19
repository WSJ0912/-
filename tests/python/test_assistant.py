from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from inference_service.assistant import AssistantUnavailableError, OpenAIAssistant


class _FakeResponse:
    def __init__(self, value: dict[str, object]) -> None:
        output_text = json.dumps(value, ensure_ascii=False)
        self.payload = json.dumps(
            {"output": [{"content": [{"type": "output_text", "text": output_text}]}]}
        ).encode()

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class AssistantTests(unittest.TestCase):
    def test_report_request_is_stateless_tool_free_and_structured(self) -> None:
        assistant = OpenAIAssistant(api_key="test-key")
        response = _FakeResponse({"draft": "草稿", "cautions": ["需医生确认"]})
        with patch("urllib.request.urlopen", return_value=response) as request_call:
            result = assistant.report_draft(
                {
                    "observations": [{"label": "Atelectasis", "probability": 0.2}],
                    "review": {"Atelectasis": "denied"},
                    "clinicianText": "未见明确肺不张。",
                }
            )
        self.assertEqual(result["draft"], "草稿")
        request = request_call.call_args.args[0]
        body = json.loads(request.data)
        self.assertIs(body["store"], False)
        self.assertEqual(body["tools"], [])
        self.assertEqual(body["model"], "gpt-4o-mini")
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertIs(body["text"]["format"]["strict"], True)

    def test_controlled_identifiers_are_rejected(self) -> None:
        assistant = OpenAIAssistant(api_key="test-key")
        with self.assertRaises(ValueError):
            assistant.report_draft(
                {"observations": [], "review": {}, "clinicianText": "", "studyId": "ST-1"}
            )

    def test_invalid_structured_response_is_rejected_locally(self) -> None:
        assistant = OpenAIAssistant(api_key="test-key")
        response = _FakeResponse({"draft": "草稿", "cautions": [], "unexpected": True})
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaises(AssistantUnavailableError):
                assistant.report_draft(
                    {"observations": [], "review": {}, "clinicianText": ""}
                )


if __name__ == "__main__":
    unittest.main()
