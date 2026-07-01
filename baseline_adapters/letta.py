from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .base import BaselineContext, BaselineEvalConfig
from .common import (
    format_retrieved_memories,
    jsonable_memories,
    parse_dialogue_to_messages,
    timestamp_for_turn,
)


METHOD = "Letta"

# Letta (formerly MemGPT) is a self-hosted stateful-agent server. Like the Zep
# adapters, this baseline talks to a running server over its REST API instead of
# pulling the heavy `letta` package in-process. Start the server first, e.g.:
#
#   pip install letta            # or: pip install -U letta
#   letta server                 # serves on http://localhost:8283
#
# and make sure the server itself can reach an LLM + embedding provider (set its
# OPENAI_API_KEY / OPENAI_BASE_URL, etc.). The adapter only needs the server URL
# (and an optional access token when LETTA_SERVER_PASSWORD is configured).
#
# IMPORTANT: the Letta server URL/token are resolved exclusively from LETTA_*
# environment variables (see `_resolve_base_url` / `_resolve_token`). They are
# intentionally NOT taken from the generic MEMORY_BASE_URL / MEMORY_API_KEY used
# by the embedding/LLM-backed baselines, because those point at the LLM provider
# (e.g. https://api.deepseek.com), not at the Letta server.
DEFAULT_BASE_URL = "http://localhost:8283"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_EMBEDDING = "openai/text-embedding-3-small"
DEFAULT_INGEST_BATCH_SIZE = 8
DEFAULT_REQUEST_TIMEOUT = 120.0

# How prior dialogue is written into Letta's memory:
#   "messages"  -> feed the transcript to the agent so its own LLM loop edits
#                  core/archival memory (faithful to how Letta works; costs LLM
#                  calls). This is the default.
#   "archival"  -> insert each turn directly as an archival passage (cheap,
#                  RAG-style; bypasses Letta's self-editing memory).
INGEST_MODE_MESSAGES = "messages"
INGEST_MODE_ARCHIVAL = "archival"

CONTEXT_TEMPLATE = """The following is the agent's persistent memory about the user and conversation.

# Core memory blocks the Letta agent maintains in-context.
<CORE_MEMORY>
{core_memory}
</CORE_MEMORY>

# The most relevant passages retrieved from the agent's archival memory.
<ARCHIVAL_MEMORY>
{archival_memory}
</ARCHIVAL_MEMORY>"""


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _resolve_base_url(eval_config: BaselineEvalConfig) -> str:
    """Resolve the Letta server URL from Letta-specific settings ONLY.

    The Letta server endpoint is deliberately decoupled from the generic
    ``MEMORY_BASE_URL`` (which carries the LLM provider endpoint, e.g.
    ``https://api.deepseek.com``). The evaluation scripts forward
    ``MEMORY_BASE_URL`` into ``eval_config.base_url`` with the *highest* priority,
    so honoring it here would point the adapter at the LLM provider instead of the
    Letta server and cause every request to fail (typically 401/403). We therefore
    read ``LETTA_BASE_URL`` / ``LETTA_SERVER_URL`` directly and never fall back to
    ``eval_config.base_url``.
    """
    base_url = _env("LETTA_BASE_URL", "LETTA_SERVER_URL") or DEFAULT_BASE_URL
    return base_url.rstrip("/")


def _resolve_token(eval_config: BaselineEvalConfig) -> str | None:
    """Resolve the Letta access token from Letta-specific settings ONLY.

    As with the base URL, ``eval_config.api_key`` carries the generic
    ``MEMORY_API_KEY`` (the LLM provider key), which is unrelated to the Letta
    server. Forwarding it as a ``Bearer`` token to a password-protected server is
    exactly what triggers the 403 auth failures, so we only honor the
    Letta-specific credentials and ignore ``eval_config.api_key``.
    """
    return _env("LETTA_API_KEY", "LETTA_SERVER_PASSWORD", "LETTA_TOKEN")


def _as_handle(value: str, default_provider: str = "openai") -> str:
    value = value.strip()
    return value if "/" in value else f"{default_provider}/{value}"


def _bare_name(value: str) -> str:
    return value.split("/", 1)[1] if "/" in value else value


def _resolve_model(eval_config: BaselineEvalConfig) -> str:
    model = eval_config.llm_model or _env("LETTA_MODEL") or DEFAULT_MODEL
    return _as_handle(model)


def _resolve_embedding(eval_config: BaselineEvalConfig) -> str:
    embedding = eval_config.embedding_model or _env("LETTA_EMBEDDING_MODEL") or DEFAULT_EMBEDDING
    return _as_handle(embedding)


def _explicit_llm_config(model_handle: str) -> dict[str, Any] | None:
    """Build an explicit LLMConfig when LETTA_LLM_ENDPOINT is set.

    This lets the Letta agent talk directly to a custom OpenAI-compatible endpoint
    (e.g. a local vLLM server) without configuring the Letta server's providers.
    The endpoint's API key is taken from the Letta server env (OPENAI_API_KEY);
    local servers usually ignore it.
    """
    endpoint = _env("LETTA_LLM_ENDPOINT", "LETTA_LLM_BASE_URL")
    if not endpoint:
        return None
    return {
        "model": _bare_name(model_handle),
        "model_endpoint_type": _env("LETTA_LLM_ENDPOINT_TYPE") or "openai",
        "model_endpoint": endpoint.rstrip("/"),
        "context_window": int(_env("LETTA_CONTEXT_WINDOW") or "32768"),
        "put_inner_thoughts_in_kwargs": False,
    }


def _explicit_embedding_config(embedding_handle: str, eval_config: BaselineEvalConfig) -> dict[str, Any] | None:
    """Build an explicit EmbeddingConfig when LETTA_EMBEDDING_ENDPOINT is set."""
    endpoint = _env("LETTA_EMBEDDING_ENDPOINT", "LETTA_EMBEDDING_BASE_URL")
    if not endpoint:
        return None
    dims = _env("LETTA_EMBEDDING_DIMS", "MEMORY_EMBEDDING_DIMS")
    embedding_dim = int(dims) if dims else (eval_config.embedding_dims or 1024)
    return {
        "embedding_endpoint_type": _env("LETTA_EMBEDDING_ENDPOINT_TYPE") or "openai",
        "embedding_endpoint": endpoint.rstrip("/"),
        "embedding_model": _bare_name(embedding_handle),
        "embedding_dim": int(embedding_dim),
        "embedding_chunk_size": int(_env("LETTA_EMBEDDING_CHUNK_SIZE") or "300"),
    }


def _ingest_mode() -> str:
    mode = (_env("LETTA_INGEST_MODE") or INGEST_MODE_MESSAGES).lower()
    if mode not in {INGEST_MODE_MESSAGES, INGEST_MODE_ARCHIVAL}:
        raise ValueError(
            f"Unsupported LETTA_INGEST_MODE={mode!r}; use {INGEST_MODE_MESSAGES!r} or {INGEST_MODE_ARCHIVAL!r}."
        )
    return mode


def _request_timeout() -> float:
    raw = _env("LETTA_REQUEST_TIMEOUT")
    return float(raw) if raw else DEFAULT_REQUEST_TIMEOUT


def _headers(token: str | None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request(
    method: str,
    url: str,
    *,
    token: str | None,
    payload: dict[str, Any] | None = None,
    timeout: float,
    ok_statuses: set[int] | None = None,
) -> Any:
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=body, headers=_headers(token), method=method)
    ok = ok_statuses or {200, 201, 204}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
            if resp.status not in ok:
                raise RuntimeError(f"Letta request failed: {method} {url} -> {resp.status} {data}")
            return json.loads(data) if data.strip() else {}
    except urllib.error.HTTPError as exc:
        data = exc.read().decode("utf-8", errors="replace")
        if exc.code in ok:
            return json.loads(data) if data.strip() else {}
        raise RuntimeError(f"Letta request failed: {method} {url} -> {exc.code} {data}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach the Letta server at {url}. Start it with `letta server` "
            "(default http://localhost:8283) and pass --memory-base-url / set LETTA_BASE_URL "
            "if it is hosted elsewhere."
        ) from exc


def _dialogue_digest(model: str, embedding: str, ingest_mode: str, messages: list[dict[str, str]]) -> str:
    h = hashlib.sha1()
    for part in (METHOD, model, embedding, ingest_mode):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    for message in messages:
        h.update(str(message.get("role") or "").encode("utf-8"))
        h.update(b"\0")
        h.update(str(message.get("content") or "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:24]


def _agent_name(digest: str) -> str:
    return f"pm_letta_{digest}"


def _find_agent(base_url: str, token: str | None, name: str, timeout: float) -> str | None:
    query = urllib.parse.urlencode({"name": name})
    data = _request("GET", f"{base_url}/v1/agents/?{query}", token=token, timeout=timeout)
    if isinstance(data, list):
        for agent in data:
            if isinstance(agent, dict) and agent.get("name") == name and agent.get("id"):
                return str(agent["id"])
    return None


def _create_agent(base_url: str, token: str | None, name: str, model: str, embedding: str, eval_config: BaselineEvalConfig, timeout: float) -> str:
    payload: dict[str, Any] = {
        "name": name,
        "include_base_tools": True,
        "memory_blocks": [
            {"label": "human", "value": ""},
            {
                "label": "persona",
                "value": (
                    "I am a helpful assistant. I carefully read the conversation and persist any "
                    "important facts about the user into my core and archival memory so I can recall "
                    "them later."
                ),
            },
        ],
    }

    # Prefer explicit endpoint configs (direct wiring to local/custom servers);
    # otherwise fall back to provider handles configured on the Letta server.
    llm_config = _explicit_llm_config(model)
    if llm_config is not None:
        payload["llm_config"] = llm_config
    else:
        payload["model"] = model

    embedding_config = _explicit_embedding_config(embedding, eval_config)
    if embedding_config is not None:
        payload["embedding_config"] = embedding_config
    else:
        payload["embedding"] = embedding

    data = _request("POST", f"{base_url}/v1/agents/", token=token, payload=payload, timeout=timeout)
    agent_id = data.get("id") if isinstance(data, dict) else None
    if not agent_id:
        raise RuntimeError(f"Letta agent creation returned no id: {data!r}")
    return str(agent_id)


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    size = max(int(size), 1)
    return [items[i : i + size] for i in range(0, len(items), size)]


def _transcript_lines(messages: list[dict[str, str]]) -> list[str]:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip().capitalize()
        content = str(message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return lines


def _ingest_via_messages(
    base_url: str,
    token: str | None,
    agent_id: str,
    messages: list[dict[str, str]],
    timeout: float,
) -> None:
    lines = _transcript_lines(messages)
    if not lines:
        return
    batch_size = int(_env("LETTA_INGEST_BATCH_SIZE") or DEFAULT_INGEST_BATCH_SIZE)
    max_steps = int(_env("LETTA_MAX_STEPS") or "20")
    for batch in _chunks(lines, batch_size):
        transcript = "\n".join(batch)
        content = (
            "Here is part of a prior conversation. Read it and store any important details about "
            "the user into your memory for future recall. Do not ask me questions.\n\n" + transcript
        )
        _request(
            "POST",
            f"{base_url}/v1/agents/{urllib.parse.quote(agent_id, safe='')}/messages",
            token=token,
            payload={
                "messages": [{"role": "user", "content": content}],
                "max_steps": max_steps,
            },
            timeout=timeout,
        )


def _ingest_via_archival(
    base_url: str,
    token: str | None,
    agent_id: str,
    messages: list[dict[str, str]],
    timeout: float,
) -> None:
    for idx, message in enumerate(messages):
        role = str(message.get("role") or "user").strip()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        _request(
            "POST",
            f"{base_url}/v1/agents/{urllib.parse.quote(agent_id, safe='')}/archival-memory",
            token=token,
            payload={"text": f"{role}: {content}", "created_at": timestamp_for_turn(idx)},
            timeout=timeout,
        )


def _fetch_core_blocks(base_url: str, token: str | None, agent_id: str, timeout: float) -> list[dict[str, Any]]:
    data = _request(
        "GET",
        f"{base_url}/v1/agents/{urllib.parse.quote(agent_id, safe='')}/core-memory/blocks",
        token=token,
        timeout=timeout,
    )
    return data if isinstance(data, list) else []


def _search_archival(
    base_url: str,
    token: str | None,
    agent_id: str,
    query: str,
    top_k: int,
    timeout: float,
) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({"search": query, "limit": max(int(top_k), 1)})
    data = _request(
        "GET",
        f"{base_url}/v1/agents/{urllib.parse.quote(agent_id, safe='')}/archival-memory?{params}",
        token=token,
        timeout=timeout,
    )
    return data if isinstance(data, list) else []


def _make_context(
    blocks: list[dict[str, Any]],
    passages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    memories: list[dict[str, Any]] = []
    core_lines: list[str] = []
    archival_lines: list[str] = []

    for block in blocks:
        label = block.get("label") or ""
        value = str(block.get("value") or "").strip()
        if not value:
            continue
        used_content = f"{label}: {value}" if label else value
        core_lines.append(f"  - {used_content}")
        memories.append(
            {
                "type": "core_memory",
                "label": label,
                "content": value,
                "used_content": used_content,
            }
        )

    for passage in passages:
        text = str(passage.get("text") or "").strip()
        if not text:
            continue
        archival_lines.append(f"  - {text}")
        memories.append(
            {
                "type": "archival_memory",
                "content": text,
                "used_content": text,
                "id": passage.get("id"),
                "created_at": passage.get("created_at"),
            }
        )

    context = CONTEXT_TEMPLATE.format(
        core_memory="\n".join(core_lines) if core_lines else "  (empty)",
        archival_memory="\n".join(archival_lines) if archival_lines else "  (no relevant passages)",
    )
    return context, memories


def build_context(
    prior_dialogue: str,
    user_question: str,
    eval_config: BaselineEvalConfig,
    *,
    sample_key: str | int | None = None,
) -> BaselineContext:
    base_url = _resolve_base_url(eval_config)
    token = _resolve_token(eval_config)
    model = _resolve_model(eval_config)
    embedding = _resolve_embedding(eval_config)
    ingest_mode = _ingest_mode()
    timeout = _request_timeout()

    messages = parse_dialogue_to_messages(prior_dialogue)
    # The agent name is keyed on the dialogue + config so that every question that
    # shares the same prior dialogue reuses one agent: memory is ingested once and
    # then searched many times (cheaper, and correct since memory only depends on
    # the dialogue, not the question).
    digest = _dialogue_digest(model, embedding, ingest_mode, messages)
    agent_name = _agent_name(digest)

    force_rebuild = (_env("LETTA_FORCE_REBUILD") or "").lower() in {"1", "true", "yes", "on"}
    agent_id = None if force_rebuild else _find_agent(base_url, token, agent_name, timeout)
    needs_ingest = agent_id is None
    if agent_id is None:
        agent_id = _create_agent(base_url, token, agent_name, model, embedding, eval_config, timeout)

    if needs_ingest and messages:
        if ingest_mode == INGEST_MODE_MESSAGES:
            _ingest_via_messages(base_url, token, agent_id, messages, timeout)
        else:
            _ingest_via_archival(base_url, token, agent_id, messages, timeout)
        wait_sec = float(_env("LETTA_INGEST_WAIT_SEC") or "0")
        if wait_sec > 0:
            time.sleep(wait_sec)

    blocks = _fetch_core_blocks(base_url, token, agent_id, timeout)
    passages = _search_archival(base_url, token, agent_id, user_question, eval_config.top_k, timeout)
    context_text, memories = _make_context(blocks, passages)
    if not memories:
        context_text = format_retrieved_memories([])

    return BaselineContext(
        context_text=context_text,
        retrieved_memories=jsonable_memories(memories),
        user_id=agent_name,
        save_dir=base_url,
        method=METHOD,
        top_k=eval_config.top_k,
    )
