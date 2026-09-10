"""MemZero plus post-retrieval control baselines."""

from __future__ import annotations

from dataclasses import replace

from .base import BaselineContext, BaselineEvalConfig
from .common import format_retrieved_memories, jsonable_memories
from .controls import (
    CONSTRUCTION_METHOD,
    MEMZERO_CONTROL_METHODS,
    METHOD_DYN_PARTITION,
    METHOD_MEMGATE,
    METHOD_SELF_RECHECK,
    apply_memzero_control,
)
from .memzero import build_context as build_memzero_context


METHODS = MEMZERO_CONTROL_METHODS


def memzero_construction_config(eval_config: BaselineEvalConfig) -> BaselineEvalConfig:
    return replace(eval_config, method=CONSTRUCTION_METHOD)


def build_context(
    prior_dialogue: str,
    user_question: str,
    eval_config: BaselineEvalConfig,
    *,
    sample_key: str | int | None = None,
) -> BaselineContext:
    if eval_config.method not in METHODS:
        raise ValueError(f"Unsupported MemZero control method: {eval_config.method!r}")
    memzero_ctx = build_memzero_context(
        prior_dialogue,
        user_question,
        memzero_construction_config(eval_config),
        sample_key=sample_key,
    )
    filtered = apply_memzero_control(
        eval_config.method,
        user_question,
        list(memzero_ctx.retrieved_memories),
        eval_config,
    )
    return BaselineContext(
        context_text=format_retrieved_memories(filtered),
        retrieved_memories=jsonable_memories(filtered),
        user_id=memzero_ctx.user_id,
        save_dir=memzero_ctx.save_dir,
        method=eval_config.method,
        top_k=eval_config.top_k,
    )
