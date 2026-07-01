from __future__ import annotations

import asyncio
import os
from typing import Any

from .base import BaselineContext, BaselineEvalConfig
from .common import jsonable_memories, parse_dialogue_to_messages, sample_user_id, timestamp_for_turn


METHOD = "ZepCloud"

CONTEXT_TEMPLATE = """FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts and their valid date ranges. If the fact is about an event, the event takes place during this time.
# format: FACT (Date range: from - to)
<FACTS>
{facts}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>"""


def _run(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "ZepCloud adapter is synchronous and cannot run inside an active event loop. "
        "Call it from the benchmark scripts' normal synchronous path."
    )


def _ignore_exists(exc: Exception) -> None:
    text = str(exc).lower()
    if "already" in text or "exists" in text or "409" in text:
        return
    raise exc


def _make_context(edges: list[Any], nodes: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    facts: list[str] = []
    entities: list[str] = []
    memories: list[dict[str, Any]] = []

    for edge in edges:
        fact = getattr(edge, "fact", None)
        if not fact:
            continue
        start = getattr(edge, "valid_at", None) or "date unknown"
        end = getattr(edge, "invalid_at", None) or "present"
        used_content = f"{fact} ({start} - {end})"
        facts.append(f"  - {used_content}")
        memories.append(
            {
                "type": "fact",
                "content": fact,
                "used_content": used_content,
                "valid_at": start,
                "invalid_at": end,
            }
        )

    for node in nodes:
        name = getattr(node, "name", None)
        summary = getattr(node, "summary", None)
        if not name and not summary:
            continue
        used_content = f"{name or '[unknown entity]'}: {summary or ''}".strip()
        entities.append(f"  - {used_content}")
        memories.append(
            {
                "type": "entity",
                "content": summary or "",
                "used_content": used_content,
                "name": name,
            }
        )

    context = CONTEXT_TEMPLATE.format(facts="\n".join(facts), entities="\n".join(entities))
    return context, memories


async def _build_context_async(
    prior_dialogue: str,
    user_question: str,
    eval_config: BaselineEvalConfig,
    *,
    sample_key: str | int | None,
) -> BaselineContext:
    try:
        from zep_cloud import Message
        from zep_cloud.client import AsyncZep
    except ImportError as exc:
        raise RuntimeError("ZepCloud requires the optional Python package: pip install zep-cloud") from exc

    api_key = eval_config.api_key or os.getenv("ZEP_API_KEY")
    if not api_key:
        raise RuntimeError("ZepCloud requires --memory-api-key or ZEP_API_KEY.")

    base_url = eval_config.base_url or os.getenv("ZEP_CLOUD_BASE_URL")
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    zep = AsyncZep(**client_kwargs)

    user_id = sample_user_id(METHOD, sample_key, prior_dialogue, user_question)
    session_id = f"{user_id}_session"

    try:
        await zep.user.add(user_id=user_id)
    except Exception as exc:  # pragma: no cover - depends on remote SDK errors
        _ignore_exists(exc)

    try:
        await zep.memory.add_session(user_id=user_id, session_id=session_id)
    except Exception as exc:  # pragma: no cover - depends on remote SDK errors
        _ignore_exists(exc)

    messages = parse_dialogue_to_messages(prior_dialogue)
    for idx, message in enumerate(messages):
        role = message.get("role") or "user"
        payload = Message(
            role=role,
            role_type=role if role in {"system", "assistant", "user", "tool", "function"} else "user",
            content=message.get("content", ""),
            created_at=timestamp_for_turn(idx),
        )
        await zep.memory.add(session_id=session_id, messages=[payload])

    wait_sec = float(os.getenv("ZEP_INGEST_WAIT_SEC", "2"))
    if wait_sec > 0:
        await asyncio.sleep(wait_sec)

    edges_results = await zep.graph.search(user_id=user_id, query=user_question, limit=eval_config.top_k)
    nodes_results = await zep.graph.search(
        user_id=user_id,
        query=user_question,
        search_scope="nodes",
        limit=eval_config.top_k,
    )
    context_text, memories = _make_context(edges_results.edges or [], nodes_results.nodes or [])

    return BaselineContext(
        context_text=context_text,
        retrieved_memories=jsonable_memories(memories),
        user_id=user_id,
        save_dir=base_url or "zep-cloud",
        method=METHOD,
        top_k=eval_config.top_k,
    )


def build_context(
    prior_dialogue: str,
    user_question: str,
    eval_config: BaselineEvalConfig,
    *,
    sample_key: str | int | None = None,
) -> BaselineContext:
    return _run(
        _build_context_async(
            prior_dialogue,
            user_question,
            eval_config,
            sample_key=sample_key,
        )
    )
