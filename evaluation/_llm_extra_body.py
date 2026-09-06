"""Shared chat-completion extra_body for generation and judge calls."""

from __future__ import annotations

import os
from typing import Any, Literal

ThinkingRole = Literal["generation", "judge"]

_TRUE = {"1", "true", "yes", "on", "enabled"}
_FALSE = {"0", "false", "no", "off", "disabled"}


def _parse_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return None


def thinking_role_from_purpose(purpose: str) -> ThinkingRole:
    if "judge" in (purpose or "").lower():
        return "judge"
    return "generation"


def thinking_enabled(role: ThinkingRole = "generation") -> bool:
    if role == "judge":
        specific = _parse_bool(
            os.environ.get("JUDGE_ENABLE_THINKING") or os.environ.get("JUDGE_THINKING")
        )
    else:
        specific = _parse_bool(
            os.environ.get("GENERATION_ENABLE_THINKING")
            or os.environ.get("GENERATION_THINKING")
        )
    if specific is not None:
        return specific
    shared = _parse_bool(
        os.environ.get("ENABLE_THINKING") or os.environ.get("THINKING_ENABLED")
    )
    return bool(shared)


def chat_extra_body(role: ThinkingRole | str = "generation") -> dict[str, Any]:
    """Thinking extra_body for OpenRouter and DeepSeek official APIs.

    Generation reads ``GENERATION_ENABLE_THINKING``; judge reads
    ``JUDGE_ENABLE_THINKING``. Either side can fall back to ``ENABLE_THINKING``.
    """
    enabled = thinking_enabled("judge" if role == "judge" else "generation")
    return {
        "reasoning": {"enabled": enabled},
        "thinking": {"type": "enabled" if enabled else "disabled"},
    }
