"""Memory-side OpenAI-compatible client for control-layer LLM calls."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from ..base import BaselineEvalConfig


def memory_chat_extra_body() -> dict[str, Any]:
    return {
        "reasoning": {"enabled": False},
        "thinking": {"type": "disabled"},
    }


def memory_chat_client(eval_config: BaselineEvalConfig) -> OpenAI:
    api_key = (eval_config.api_key or os.environ.get("MEMORY_API_KEY") or "").strip()
    base_url = (eval_config.base_url or os.environ.get("MEMORY_BASE_URL") or "").strip()
    if not api_key:
        raise RuntimeError(
            "MemZero control methods need MEMORY_API_KEY (or BaselineEvalConfig.api_key)."
        )
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    timeout = float(os.environ.get("MEMORY_REQUEST_TIMEOUT", "60"))
    if timeout > 0:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)


def memory_llm_model(eval_config: BaselineEvalConfig) -> str:
    return (
        (eval_config.llm_model or "").strip()
        or os.environ.get("MEMORY_LLM_MODEL", "").strip()
        or "deepseek-v4-flash"
    )


def chat_completion(
    eval_config: BaselineEvalConfig,
    *,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
) -> str:
    client = memory_chat_client(eval_config)
    resp = client.chat.completions.create(
        model=memory_llm_model(eval_config),
        messages=messages,
        temperature=temperature,
        extra_body=memory_chat_extra_body(),
    )
    return (resp.choices[0].message.content or "").strip()


def chat_json(
    eval_config: BaselineEvalConfig,
    *,
    system: str,
    user: str,
    temperature: float = 0.0,
) -> dict[str, Any]:
    text = chat_completion(
        eval_config,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return parse_json_object(text)


def parse_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty LLM response")
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    if fence:
        raw = fence.group(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    return data
