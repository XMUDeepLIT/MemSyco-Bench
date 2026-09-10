"""Post-retrieval MemZero control wrappers."""

from __future__ import annotations

from typing import Any, Callable

from ..base import BaselineEvalConfig

METHOD_SELF_RECHECK = "MemZero+SelfReCheck"
METHOD_MEMGATE = "MemZero+MemGate"
METHOD_DYN_PARTITION = "MemZero+DynPartition"

MEMZERO_CONTROL_METHODS = (
    METHOD_SELF_RECHECK,
    METHOD_MEMGATE,
    METHOD_DYN_PARTITION,
)

CONSTRUCTION_METHOD = "MemZero"


def is_memzero_control_method(method: str) -> bool:
    return method in MEMZERO_CONTROL_METHODS


def apply_memzero_control(
    method: str,
    query: str,
    memories: list[dict[str, Any]],
    eval_config: BaselineEvalConfig,
    *,
    layer: Any | None = None,
) -> list[dict[str, Any]]:
    if method == METHOD_SELF_RECHECK:
        from .self_recheck import filter_memories

        return filter_memories(query, memories, eval_config)
    if method == METHOD_MEMGATE:
        from .memgate_filter import filter_memories

        return filter_memories(query, memories, eval_config)
    if method == METHOD_DYN_PARTITION:
        from .dyn_partition import filter_memories

        return filter_memories(query, memories, eval_config, layer=layer)
    raise ValueError(f"Unsupported MemZero control method: {method!r}")
