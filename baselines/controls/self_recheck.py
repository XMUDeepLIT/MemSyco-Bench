"""Mem0 + Self-ReCheck post-retrieval filter.

Self-ReCheck (OP-Bench, arXiv:2601.13722) keeps a retrieved memory only when an
LLM judges it directly useful for the current query. Construction and retrieval
stay MemZero; this module only filters the retrieved set.
"""

from __future__ import annotations

from typing import Any

from ..base import BaselineEvalConfig
from .llm import chat_json

SELF_RECHECK_SYSTEM = """You are a memory filter for a conversational agent.
For each retrieved memory, decide whether it is DIRECTLY useful for answering the current user query.

Keep a memory only if it supplies information that is necessary or clearly helpful for answering this query.
Drop a memory if it is merely topically related, redundant, outdated for this query, likely to cause over-personalization, or not needed to answer the question.

Return JSON only:
{"keep": [true or false for each memory in order]}
The keep array length MUST equal the number of memories."""


def filter_memories(
    query: str,
    memories: list[dict[str, Any]],
    eval_config: BaselineEvalConfig,
) -> list[dict[str, Any]]:
    if not memories:
        return []
    numbered = []
    for i, mem in enumerate(memories, start=1):
        text = str(mem.get("used_content") or mem.get("content") or "").strip()
        numbered.append(f"{i}. {text}")
    user = (
        f"User query:\n{query.strip()}\n\n"
        f"Retrieved memories:\n" + "\n".join(numbered)
    )
    try:
        parsed = chat_json(eval_config, system=SELF_RECHECK_SYSTEM, user=user)
        keep_flags = parsed.get("keep")
        if not isinstance(keep_flags, list) or len(keep_flags) != len(memories):
            raise ValueError("keep array missing or wrong length")
    except Exception as exc:
        print(f"[Self-ReCheck] parse/LLM failed, keeping all memories: {exc}", flush=True)
        return memories

    kept: list[dict[str, Any]] = []
    for mem, flag in zip(memories, keep_flags):
        if flag is True or flag == 1 or str(flag).strip().lower() in {"true", "yes", "keep"}:
            meta = dict(mem.get("metadata") or {})
            meta["control"] = {"method": "MemZero+SelfReCheck", "kept": True}
            item = dict(mem)
            item["metadata"] = meta
            kept.append(item)
    return kept
