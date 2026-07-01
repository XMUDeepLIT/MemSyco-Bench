from __future__ import annotations

from collections.abc import Callable

from . import amem, full_context, langmem, letta, lightmem, memgpt_minimal, memorybank, memzero, naive_rag, supermemory, zep_cloud, zep_legacy
from .base import BaselineContext, BaselineEvalConfig


BuildFn = Callable[[str, str, BaselineEvalConfig], BaselineContext]

_BUILDERS = {
    lightmem.METHOD: lightmem.build_context,
    lightmem.FULL_METHOD: lightmem.build_context,
    full_context.METHOD: full_context.build_context,
    naive_rag.METHOD: naive_rag.build_context,
    amem.METHOD: amem.build_context,
    "MemZero": memzero.build_context,
    "MemZeroGraph": memzero.build_context,
    memorybank.METHOD: memorybank.build_context,
    supermemory.METHOD: supermemory.build_context,
    langmem.METHOD: langmem.build_context,
    letta.METHOD: letta.build_context,
    memgpt_minimal.METHOD: memgpt_minimal.build_context,
    zep_cloud.METHOD: zep_cloud.build_context,
    zep_legacy.METHOD: zep_legacy.build_context,
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
