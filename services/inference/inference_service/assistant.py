from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Mapping

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["draft", "cautions"],
    "properties": {
        "draft": {"type": "string"},
        "cautions": {"type": "array", "items": {"type": "string"}},
    },
}

EXPERIMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "limitations"],
    "properties": {
        "summary": {"type": "string"},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
}


class AssistantUnavailableError(RuntimeError):
    pass


def _validate_structured_response(
    value: Any,
    text_field: str,
    list_field: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {text_field, list_field}:
        raise AssistantUnavailableError("助手返回的结构化响应无效")
    if not isinstance(value[text_field], str):
        raise AssistantUnavailableError("助手返回的结构化响应无效")
    items = value[list_field]
    if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
        raise AssistantUnavailableError("助手返回的结构化响应无效")
    return value


def _reject_sensitive_keys(value: Any, path: str = "root") -> None:
    prohibited = {
        "path", "filepath", "image", "pixels", "dicom", "patientid",
        "patientname", "accessionnumber", "mimicid", "subjectid", "studyid",
        "seriesid", "sopinstanceuid", "studyinstanceuid", "seriesinstanceuid",
        "predictionid", "reviewid", "reportid", "userid", "doctorid",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in prohibited:
                raise ValueError(f"assistant payload contains prohibited field at {path}.{key}")
            _reject_sensitive_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive_keys(child, f"{path}[{index}]")


class OpenAIAssistant:
    """Optional text-only adapter. It has no tools and cannot change local state."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def report_draft(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        _reject_sensitive_keys(payload)
        result = self._call(
            "你是阅片报告文字助手。只能整理给定的去标识化结构化观察和医生复核意见，不能作出最终诊断、不能宣称已确认报告。",
            payload,
            "report_draft",
            REPORT_SCHEMA,
        )
        return _validate_structured_response(result, "draft", "cautions")

    def experiment_summary(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        _reject_sensitive_keys(payload)
        result = self._call(
            "你是科研结果写作助手。只解释聚合指标和实验记录，不推断患者级结果，不夸大临床有效性。",
            payload,
            "experiment_summary",
            EXPERIMENT_SCHEMA,
        )
        return _validate_structured_response(result, "summary", "limitations")

    def _call(self, instruction: str, payload: Mapping[str, Any], name: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise AssistantUnavailableError("未配置 OpenAI 密钥；本地识别、复核和报告功能仍可使用")
        request_body = {
            "model": self.model,
            "store": False,
            "tools": [],
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": instruction}]},
                {"role": "user", "content": [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                result = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise AssistantUnavailableError("助手请求失败；不会影响离线核心流程") from exc
        output_text = result.get("output_text")
        if not output_text:
            for item in result.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        output_text = content.get("text")
                        break
        try:
            return json.loads(output_text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise AssistantUnavailableError("助手返回的结构化响应无效") from exc
