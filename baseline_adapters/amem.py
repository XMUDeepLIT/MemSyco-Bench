from __future__ import annotations

from .base import BaselineContext, BaselineEvalConfig
from .lightmem_toolkit import build_lightmem_toolkit_context


METHOD = "A-MEM"


def build_context(
    prior_dialogue: str,
    user_question: str,
    eval_config: BaselineEvalConfig,
    *,
    sample_key: str | int | None = None,
) -> BaselineContext:
    return build_lightmem_toolkit_context(METHOD, prior_dialogue, user_question, eval_config, sample_key=sample_key)
