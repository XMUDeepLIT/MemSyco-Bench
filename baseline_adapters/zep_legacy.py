from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .base import BaselineContext, BaselineEvalConfig
from .common import jsonable_memories, parse_dialogue_to_messages, sample_user_id, timestamp_for_turn


METHOD = "ZepLegacy"
DEFAULT_BASE_URL = "http://localhost:8000/api/v2"
MAX_MESSAGES_PER_BATCH = 30


def _headers(secret: str) -> dict[str, str]:
    return {
        "Authorization": f"Api-Key {secret}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _request(
    method: str,
    url: str,
    *,
    secret: str,
    payload: dict[str, Any] | None = None,
    ok_statuses: set[int] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=body, headers=_headers(secret), method=method)
    ok = ok_statuses or {200, 201, 204}
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read().decode("utf-8")
            if resp.status not in ok:
                raise RuntimeError(f"ZepLegacy request failed: {method} {url} -> {resp.status} {data}")
            return json.loads(data) if data.strip() else {}
    except urllib.error.HTTPError as exc:
        data = exc.read().decode("utf-8", errors="replace")
        if exc.code in ok:
            return json.loads(data) if data.strip() else {}
        raise RuntimeError(f"ZepLegacy request failed: {method} {url} -> {exc.code} {data}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach ZepLegacy at {url}. Start a Zep Community Edition server "
            "(see https://github.com/getzep/zep for deployment) and pass --memory-base-url "
            "if it is not on localhost:8000."
        ) from exc


def _try_create(method: str, url: str, *, secret: str, payload: dict[str, Any]) -> None:
    try:
        _request(method, url, secret=secret, payload=payload)
    except RuntimeError as exc:
        text = str(exc).lower()
        if "already" in text or "exists" in text or "409" in text:
            return
        raise


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _format_context(results: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    memories: list[dict[str, Any]] = []
    facts: list[str] = []
    for item in results:
        fact = item.get("fact") or {}
        fact_text = fact.get("fact") if isinstance(fact, dict) else None
        if not fact_text:
            continue
        created_at = fact.get("created_at") if isinstance(fact, dict) else None
        used_content = f"{fact_text} ({created_at})" if created_at else fact_text
        facts.append(f"  - {used_content}")
        memories.append(
            {
                "type": "fact",
                "content": fact_text,
                "used_content": used_content,
                "created_at": created_at,
                "raw": item,
            }
        )

    context = """FACTS represent relevant context to the current conversation.

<FACTS>
{facts}
</FACTS>""".format(facts="\n".join(facts))
    return context, memories


def build_context(
    prior_dialogue: str,
    user_question: str,
    eval_config: BaselineEvalConfig,
    *,
    sample_key: str | int | None = None,
) -> BaselineContext:
    base_url = (eval_config.base_url or os.getenv("ZEP_LEGACY_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    secret = eval_config.api_key or os.getenv("ZEP_API_SECRET") or os.getenv("ZEP_API_KEY")
    if not secret:
        raise RuntimeError(
            "ZepLegacy requires --memory-api-key, ZEP_API_SECRET, or ZEP_API_KEY. "
            "It must match legacy/zep.yaml api_secret."
        )

    user_id = sample_user_id(METHOD, sample_key, prior_dialogue, user_question)
    session_id = f"{user_id}_session"

    _try_create(
        "POST",
        f"{base_url}/users",
        secret=secret,
        payload={"user_id": user_id, "metadata": {"baseline": METHOD}},
    )
    _try_create(
        "POST",
        f"{base_url}/sessions",
        secret=secret,
        payload={"session_id": session_id, "user_id": user_id, "metadata": {"baseline": METHOD}},
    )

    messages: list[dict[str, Any]] = []
    for idx, message in enumerate(parse_dialogue_to_messages(prior_dialogue)):
        role = message.get("role") or "user"
        messages.append(
            {
                "role": role,
                "role_type": role if role in {"system", "assistant", "user", "tool", "function"} else "user",
                "content": message.get("content", ""),
                "created_at": timestamp_for_turn(idx),
            }
        )

    batch_size = min(int(os.getenv("ZEP_LEGACY_BATCH_SIZE", str(MAX_MESSAGES_PER_BATCH))), MAX_MESSAGES_PER_BATCH)
    for batch in _chunks(messages, max(batch_size, 1)):
        _request(
            "POST",
            f"{base_url}/sessions/{urllib.parse.quote(session_id, safe='')}/memory",
            secret=secret,
            payload={"messages": batch},
        )

    wait_sec = float(os.getenv("ZEP_INGEST_WAIT_SEC", "2"))
    if wait_sec > 0:
        time.sleep(wait_sec)

    search = _request(
        "POST",
        f"{base_url}/sessions/search?limit={max(eval_config.top_k, 0)}",
        secret=secret,
        payload={"text": user_question, "user_id": user_id, "session_ids": [session_id]},
    )
    context_text, memories = _format_context(search.get("results") or [])

    return BaselineContext(
        context_text=context_text,
        retrieved_memories=jsonable_memories(memories),
        user_id=user_id,
        save_dir=base_url,
        method=METHOD,
        top_k=eval_config.top_k,
    )
