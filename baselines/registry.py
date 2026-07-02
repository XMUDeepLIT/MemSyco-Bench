from __future__ import annotations

from collections.abc import Callable

from . import amem, lightmem, memgpt, memorybank, memzero, naive_rag, supermemory
from .base import BaselineContext, BaselineEvalConfig


BuildFn = Callable[[str, str, BaselineEvalConfig], BaselineContext]

_BUILDERS = {
    lightmem.METHOD: lightmem.build_context,
    naive_rag.METHOD: naive_rag.build_context,
    amem.METHOD: amem.build_context,
    "MemZero": memzero.build_context,
    memorybank.METHOD: memorybank.build_context,
    supermemory.METHOD: supermemory.build_context,
    memgpt.METHOD: memgpt.build_context,
}

BASELINE_METHODS = tuple(_BUILDERS)


def build_baseline_context(
    prior_dialogue: str,
    user_question: str,
    eval_config: BaselineEvalConfig,
    *,
    sample_key: str | int | None = None,
) -> BaselineContext:
    try:
        builder = _BUILDERS[eval_config.method]
    except KeyError as exc:
        raise ValueError(f"Unsupported baseline method: {eval_config.method!r}") from exc
    return builder(prior_dialogue, user_question, eval_config, sample_key=sample_key)
