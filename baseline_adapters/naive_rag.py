from __future__ import annotations

from .base import BaselineContext, BaselineEvalConfig
from .lightmem_toolkit import build_lightmem_toolkit_context


METHOD = "NaiveRAG"


def build_context(
    prior_dialogue: str,
    user_question: str,
    eval_config: BaselineEvalConfig,
    *,
    sample_key: str | int | None = None,
) -> BaselineContext:
    """Simple turn-level RAG: one dialogue turn -> one embedded chunk -> top-k retrieve."""
    return build_lightmem_toolkit_context(METHOD, prior_dialogue, user_question, eval_config, sample_key=sample_key)
