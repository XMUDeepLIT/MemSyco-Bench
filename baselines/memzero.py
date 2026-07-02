from __future__ import annotations

from .base import BaselineContext, BaselineEvalConfig
from .toolkit.runner import build_lightmem_toolkit_context


METHODS = ("MemZero",)


def build_context(
    prior_dialogue: str,
    user_question: str,
    eval_config: BaselineEvalConfig,
    *,
    sample_key: str | int | None = None,
) -> BaselineContext:
    if eval_config.method not in METHODS:
        raise ValueError(f"Unsupported MemZero adapter method: {eval_config.method!r}")
    return build_lightmem_toolkit_context(
        eval_config.method,
        prior_dialogue,
        user_question,
        eval_config,
        sample_key=sample_key,
    )
