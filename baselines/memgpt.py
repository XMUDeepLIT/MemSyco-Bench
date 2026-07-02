from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import BaselineContext, BaselineEvalConfig
from .common import jsonable_memories, parse_dialogue_to_messages, retry_embedding_query


# ---------------------------------------------------------------------------
# MemGPT baseline adapter — lightweight in-repo re-implementation of the MemGPT
# memory mechanism (core memory blocks + archival vector store via tool calls).
# ---------------------------------------------------------------------------

METHOD = "MemGPT"

DEFAULT_LLM_MODEL = "deepseek-v4-flash"
DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_EMBED_BATCH = 64
DEFAULT_REQUEST_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_STEPS = 12
DEFAULT_INGEST_BATCH_SIZE = 6
DEFAULT_ARCHIVAL_SEARCH_LIMIT = 5
# 公平性设置：所有 baseline 的 LLM 摄入调用统一 max_tokens / temperature。
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.7

_CACHE_VERSION = 1
_COMPLETE_MARKER = ".memgpt_complete.json"

_cache_guard = threading.Lock()
_inproc_cache: dict[str, "_MemoryIndex"] = {}


# ---------------------------------------------------------------------------
# agent 运行时状态（纯内存）
# ---------------------------------------------------------------------------
@dataclass
class _AgentState:
    core_blocks: dict[str, str] = field(default_factory=dict)
    archival_docs: list[dict[str, Any]] = field(default_factory=list)   # [{"text","created_at"}]
    archival_vectors: Any = None                                        # np.ndarray (n, dim) L2-normalized


@dataclass
class _MemoryIndex:
    user_id: str
    save_dir: str
    core_blocks: dict[str, str]
    archival_docs: list[dict[str, Any]]
    archival_vectors: Any  # np.ndarray (n, dim) L2-normalized


# ---------------------------------------------------------------------------
# 配置解析（全部来自 MEMORY_* 环境变量，与 MemoryBank / Supermemory 一致）
# ---------------------------------------------------------------------------
def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _request_timeout() -> float:
    raw = _env("MEMGPT_REQUEST_TIMEOUT", "MEMORY_REQUEST_TIMEOUT")
    return float(raw) if raw else DEFAULT_REQUEST_TIMEOUT


def _max_retries() -> int:
    raw = _env("MEMGPT_MAX_RETRIES", "MEMORY_API_MAX_RETRIES")
    return max(1, int(raw)) if raw else DEFAULT_MAX_RETRIES


def _max_steps() -> int:
    raw = _env("MEMGPT_MAX_STEPS")
    return max(1, int(raw)) if raw else DEFAULT_MAX_STEPS


def _ingest_batch_size() -> int:
    raw = _env("MEMGPT_INGEST_BATCH_SIZE")
    return max(1, int(raw)) if raw else DEFAULT_INGEST_BATCH_SIZE


def _language() -> str:
    lang = (_env("MEMGPT_LANGUAGE", "MEMORYBANK_LANGUAGE", "SUPERMEMORY_LANGUAGE") or "en").lower()
    return "cn" if lang in {"cn", "zh", "zh-cn", "chinese"} else "en"


def _resolve_llm_model(eval_config: BaselineEvalConfig) -> str:
    return eval_config.llm_model or _env("MEMORY_LLM_MODEL") or DEFAULT_LLM_MODEL


def _resolve_embedding_model(eval_config: BaselineEvalConfig) -> str:
    return eval_config.embedding_model or _env("MEMORY_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


def _resolve_embedding_dims(eval_config: BaselineEvalConfig) -> int | None:
    if eval_config.embedding_dims is not None:
        return int(eval_config.embedding_dims)
    raw = _env("MEMORY_EMBEDDING_DIMS")
    return int(raw) if raw else None


# ---------------------------------------------------------------------------
# OpenAI 兼容客户端（直连 LLM + embedding 端点，使用 MEMORY_* 凭据）
# ---------------------------------------------------------------------------
def _make_clients(eval_config: BaselineEvalConfig) -> tuple[Any, Any]:
    import openai  # noqa: PLC0415

    timeout = _request_timeout()
    max_retries = _max_retries()

    chat_client = openai.OpenAI(
        api_key=eval_config.api_key or os.environ.get("OPENAI_API_KEY") or "sk-local",
        base_url=eval_config.base_url or os.environ.get("OPENAI_API_BASE") or None,
        timeout=timeout,
        max_retries=max_retries,
    )
    embed_client = openai.OpenAI(
        api_key=eval_config.embedding_api_key or eval_config.api_key or "sk-local",
        base_url=eval_config.embedding_base_url or eval_config.base_url or None,
        timeout=timeout,
        max_retries=max_retries,
    )
    return chat_client, embed_client


def _l2_normalize(arr: Any) -> Any:
    import numpy as np  # noqa: PLC0415

    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms


def _embed_texts(embed_client: Any, model: str, texts: list[str], dims: int | None) -> Any:
    import numpy as np  # noqa: PLC0415

    if not texts:
        return np.zeros((0, dims or 1), dtype=np.float32)

    vectors: list[list[float]] = []
    extra: dict[str, Any] = {"dimensions": dims} if dims else {}

    def _one_batch(batch: list[str]) -> list[list[float]]:
        resp = embed_client.embeddings.create(model=model, input=batch, **extra)
        return [item.embedding for item in resp.data]

    for start in range(0, len(texts), DEFAULT_EMBED_BATCH):
        batch = texts[start : start + DEFAULT_EMBED_BATCH]
        out = retry_embedding_query(
            lambda b=batch: _one_batch(b),
            method=METHOD,
        )
        vectors.extend(out)

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return _l2_normalize(arr)


def _embed_one(embed_client: Any, model: str, text: str, dims: int | None) -> Any:
    return _embed_texts(embed_client, model, [text], dims)[0]


# ---------------------------------------------------------------------------
# MemGPT 记忆工具（OpenAI function-calling schema）
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "core_memory_replace",
            "description": (
                "Replace the string `old_str` with `new_str` inside a core memory block "
                "(identified by its label, e.g. 'human' or 'persona'). Use this to update "
                "or correct a fact about the user stored in core memory. `old_str` must "
                "appear verbatim in the block."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "the core memory block label"},
                    "old_str": {"type": "string", "description": "the exact substring to replace"},
                    "new_str": {"type": "string", "description": "the replacement string"},
                },
                "required": ["label", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "core_memory_append",
            "description": (
                "Append `content` to the end of a core memory block (identified by its label, "
                "e.g. 'human'). Use this to record a new durable fact about the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "the core memory block label"},
                    "content": {"type": "string", "description": "the text to append"},
                },
                "required": ["label", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archival_memory_insert",
            "description": (
                "Insert a passage into archival memory (an external vector store). Use this to "
                "store longer evidence, verbatim quotes, or detailed context that does not belong "
                "in the compact core memory. The passage will be retrievable later via "
                "archival_memory_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "the passage to store"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archival_memory_search",
            "description": (
                "Search archival memory for passages semantically related to `query`, returning "
                "the top `limit` results. Use this to recall earlier stored evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the search query"},
                    "limit": {"type": "integer", "description": "max number of results (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
]


def _dispatch_tool(
    name: str,
    args: dict[str, Any],
    state: _AgentState,
    embed_client: Any,
    embedding_model: str,
    embedding_dims: int | None,
) -> str:
    """Execute one memory tool call against the in-memory agent state; return the tool result string."""
    import numpy as np  # noqa: PLC0415

    if name == "core_memory_replace":
        label = str(args.get("label") or "")
        old_str = str(args.get("old_str") or "")
        new_str = str(args.get("new_str") or "")
        block = state.core_blocks.get(label)
        if block is None:
            return f"ERROR: no core memory block named {label!r}. Available: {list(state.core_blocks)}."
        if old_str not in block:
            return f"ERROR: old_str not found verbatim in block {label!r}."
        state.core_blocks[label] = block.replace(old_str, new_str, 1)
        return f"OK: replaced in core memory block {label!r}."

    if name == "core_memory_append":
        label = str(args.get("label") or "")
        content = str(args.get("content") or "")
        if label not in state.core_blocks:
            return f"ERROR: no core memory block named {label!r}. Available: {list(state.core_blocks)}."
        if not content.strip():
            return "ERROR: empty content."
        sep = "\n" if state.core_blocks[label] and not state.core_blocks[label].endswith("\n") else ""
        state.core_blocks[label] = (state.core_blocks[label] or "") + sep + content.strip()
        return f"OK: appended to core memory block {label!r}."

    if name == "archival_memory_insert":
        text = str(args.get("text") or "").strip()
        if not text:
            return "ERROR: empty archival passage."
        vec = _embed_one(embed_client, embedding_model, text, embedding_dims)
        doc = {"text": text, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        state.archival_docs.append(doc)
        if state.archival_vectors is None or state.archival_vectors.shape[0] == 0:
            state.archival_vectors = vec.reshape(1, -1)
        else:
            state.archival_vectors = np.vstack([state.archival_vectors, vec.reshape(1, -1)])
        return f"OK: inserted 1 passage into archival memory (total={len(state.archival_docs)})."

    if name == "archival_memory_search":
        query = str(args.get("query") or "")
        limit = int(args.get("limit") or DEFAULT_ARCHIVAL_SEARCH_LIMIT)
        if not query.strip() or state.archival_vectors is None or state.archival_vectors.shape[0] == 0:
            return "no archival memories found."
        qvec = _embed_one(embed_client, embedding_model, query, embedding_dims)
        scores = state.archival_vectors @ qvec
        k = max(1, min(limit, len(state.archival_docs)))
        top_idx = np.argsort(-scores)[:k]
        lines = []
        for rank, i in enumerate(top_idx, start=1):
            lines.append(f"[{rank}] (score={float(scores[int(i)]):.3f}) {state.archival_docs[int(i)]['text']}")
        return "\n".join(lines) if lines else "no archival memories found."

    return f"ERROR: unknown tool {name!r}."


# ---------------------------------------------------------------------------
# MemGPT system prompt（把 core memory 内联进去，指示 LLM 自主管理记忆）
# ---------------------------------------------------------------------------
_PERSONA = (
    "You are MemGPT, a helpful assistant that manages its own memory. You have two kinds of memory:\n"
    "- CORE MEMORY: shown to you below inside <CORE_MEMORY>. It holds the most important, durable "
    "facts about the user. You can edit it with the tools core_memory_replace and core_memory_append.\n"
    "- ARCHIVAL MEMORY: an external store you cannot see directly. You can write to it with "
    "archival_memory_insert and recall from it with archival_memory_search.\n"
    "When you learn something durable about the user (preferences, identity, goals, constraints, "
    "decisions), proactively record it: concise facts go into the 'human' core memory block; longer "
    "evidence, verbatim quotes, or detailed context go into archival memory via archival_memory_insert. "
    "Resolve contradictions by updating the existing core memory entry rather than appending duplicates."
)

_CORE_MEMORY_TEMPLATE = """<CORE_MEMORY>
{blocks}</CORE_MEMORY>"""


def _render_core_memory(blocks: dict[str, str]) -> str:
    parts = []
    for label, value in blocks.items():
        parts.append(f"[block: {label}]\n{(value or '').strip() or '(empty)'}")
    return _CORE_MEMORY_TEMPLATE.format(blocks="\n\n".join(parts)) if parts else "<CORE_MEMORY>\n(empty)\n</CORE_MEMORY>"


def _system_prompt(state: _AgentState) -> str:
    return f"{_PERSONA}\n\n{_render_core_memory(state.core_blocks)}"


# ---------------------------------------------------------------------------
# agent loop（OpenAI tool-calling）
# ---------------------------------------------------------------------------
def _run_agent_loop(
    chat_client: Any,
    model: str,
    user_text: str,
    state: _AgentState,
    embed_client: Any,
    embedding_model: str,
    embedding_dims: int | None,
    max_steps: int,
) -> str:
    """Run a MemGPT-style tool-calling loop for one user turn. Returns the final assistant text."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(state)},
        {"role": "user", "content": user_text},
    ]
    final_text = ""
    for _ in range(max_steps):
        resp = chat_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        choice = resp.choices[0]
        msg = choice.message
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        if not tool_calls:
            final_text = (msg.content or "").strip()
            break

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            try:
                result = _dispatch_tool(
                    tc.function.name, args, state, embed_client, embedding_model, embedding_dims
                )
            except Exception as exc:  # noqa: BLE001
                result = f"ERROR: tool execution failed: {exc}"
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result}
            )
        # refresh the inlined core memory in the system prompt after edits
        messages[0] = {"role": "system", "content": _system_prompt(state)}
    return final_text


# ---------------------------------------------------------------------------
# 摄入：把 prior dialogue 分批喂给 agent loop，让 LLM 自主写记忆
# ---------------------------------------------------------------------------
def _transcript(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role") or "user").strip().capitalize()
        content = str(message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _initial_state() -> _AgentState:
    return _AgentState(core_blocks={"human": "", "persona": "You are a helpful, memory-managing assistant."})


def _ingest(
    chat_client: Any,
    model: str,
    embed_client: Any,
    embedding_model: str,
    embedding_dims: int | None,
    messages: list[dict[str, str]],
    max_steps: int,
    batch_size: int,
    language: str,
) -> _AgentState:
    state = _initial_state()
    if language == "cn":
        preamble = (
            "下面是一段用户与助手的过往对话（节选）。请阅读并把你认为重要的、关于用户的持久信息"
            "写进记忆：简短事实用 core_memory_append 写入 'human' 块，较长的证据/原文用 "
            "archival_memory_insert 存入 archival memory。不要向我提问，存储完后简要确认即可。"
        )
    else:
        preamble = (
            "Here is part of a prior conversation between a user and an assistant. Read it and store "
            "any important, durable information about the user into your memory: concise facts go into "
            "the 'human' core memory block via core_memory_append (or core_memory_replace to update an "
            "existing fact); longer evidence or verbatim passages go into archival memory via "
            "archival_memory_insert. Do not ask me questions; briefly confirm once you have stored what "
            "matters."
        )

    for start in range(0, len(messages), batch_size):
        batch = messages[start : start + batch_size]
        transcript = _transcript(batch)
        if not transcript.strip():
            continue
        user_text = f"{preamble}\n\n{transcript}"
        _run_agent_loop(
            chat_client, model, user_text, state, embed_client, embedding_model, embedding_dims, max_steps
        )
    return state


# ---------------------------------------------------------------------------
# 检索（对 query 做 top-k 余弦）
# ---------------------------------------------------------------------------
def _search_archival(
    index: _MemoryIndex,
    query: str,
    embed_client: Any,
    embedding_model: str,
    embedding_dims: int | None,
    top_k: int,
) -> list[dict[str, Any]]:
    import numpy as np  # noqa: PLC0415

    if index.archival_vectors is None or index.archival_vectors.shape[0] == 0 or not query.strip():
        return []
    qvec = _embed_one(embed_client, embedding_model, query, embedding_dims)
    scores = index.archival_vectors @ qvec
    k = max(1, min(int(top_k), len(index.archival_docs)))
    top_idx = np.argsort(-scores)[:k]
    results = []
    for rank, i in enumerate(top_idx, start=1):
        i = int(i)
        text = index.archival_docs[i].get("text", "")
        results.append(
            {
                "type": "archival",
                "content": text,
                "used_content": text,
                "score": float(scores[i]),
                "metadata": {"rank": rank, "created_at": index.archival_docs[i].get("created_at")},
            }
        )
    return results


# ---------------------------------------------------------------------------
# 磁盘缓存（对齐 supermemory.py / memorybank.py）
# ---------------------------------------------------------------------------
def _digest(
    transcript: str,
    *,
    model: str,
    embedding_model: str,
    embedding_dims: int | None,
    max_steps: int,
    batch_size: int,
    language: str,
) -> str:
    h = hashlib.sha1()
    for part in (
        METHOD,
        model,
        embedding_model,
        str(embedding_dims or ""),
        str(max_steps),
        str(batch_size),
        str(LLM_MAX_TOKENS),
        str(LLM_TEMPERATURE),
        language,
        f"v{_CACHE_VERSION}",
    ):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    h.update(transcript.encode("utf-8"))
    return h.hexdigest()[:24]


def _marker_path(save_dir: str) -> Path:
    return Path(save_dir) / _COMPLETE_MARKER


def _memory_path(save_dir: str) -> Path:
    return Path(save_dir) / "memory.json"


def _vectors_path(save_dir: str) -> Path:
    return Path(save_dir) / "embeddings.npy"


def _load_from_disk(save_dir: str, digest: str, user_id: str) -> _MemoryIndex | None:
    import numpy as np  # noqa: PLC0415

    marker = _marker_path(save_dir)
    if not marker.is_file():
        return None
    try:
        meta = json.loads(marker.read_text(encoding="utf-8-sig"))
        if meta.get("digest") != digest or meta.get("method") != METHOD:
            return None
        payload = json.loads(_memory_path(save_dir).read_text(encoding="utf-8"))
        core_blocks = payload.get("core_blocks") or {}
        archival_docs = payload.get("archival_docs") or []
        vectors = np.load(_vectors_path(save_dir))
    except Exception:  # noqa: BLE001
        return None
    if len(archival_docs) != len(vectors):
        return None
    return _MemoryIndex(
        user_id=user_id,
        save_dir=save_dir,
        core_blocks=core_blocks,
        archival_docs=archival_docs,
        archival_vectors=vectors,
    )


def _save_to_disk(
    save_dir: str,
    digest: str,
    core_blocks: dict[str, str],
    archival_docs: list[dict[str, Any]],
    vectors: Any,
) -> None:
    import numpy as np  # noqa: PLC0415

    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    _memory_path(save_dir).write_text(
        json.dumps({"core_blocks": core_blocks, "archival_docs": archival_docs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(_vectors_path(save_dir), vectors)
    marker = {
        "method": METHOD,
        "digest": digest,
        "core_blocks": list(core_blocks.keys()),
        "archival_count": len(archival_docs),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": _CACHE_VERSION,
    }
    tmp = _marker_path(save_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_marker_path(save_dir))


def _empty_vectors(dims: int | None) -> Any:
    import numpy as np  # noqa: PLC0415

    return np.zeros((0, dims or 1), dtype=np.float32)


def _get_or_build_index(
    transcript: str,
    messages: list[dict[str, str]],
    eval_config: BaselineEvalConfig,
) -> _MemoryIndex:
    model = _resolve_llm_model(eval_config)
    embedding_model = _resolve_embedding_model(eval_config)
    embedding_dims = _resolve_embedding_dims(eval_config)
    max_steps = _max_steps()
    batch_size = _ingest_batch_size()
    language = _language()

    digest = _digest(
        transcript,
        model=model,
        embedding_model=embedding_model,
        embedding_dims=embedding_dims,
        max_steps=max_steps,
        batch_size=batch_size,
        language=language,
    )
    user_id = f"pm_memgpt_{digest}"
    save_dir = str((eval_config.save_root / METHOD / user_id).resolve())

    with _cache_guard:
        cached = _inproc_cache.get(digest)
        if cached is not None:
            return cached

    disk = _load_from_disk(save_dir, digest, user_id)
    if disk is not None:
        print(f"[memory-cache] disk hit method={METHOD} user_id={user_id} save_dir={save_dir}", flush=True)
        with _cache_guard:
            _inproc_cache[digest] = disk
        return disk

    print(f"[memory-cache] disk miss; building method={METHOD} user_id={user_id} save_dir={save_dir}", flush=True)
    chat_client, embed_client = _make_clients(eval_config)
    state = _ingest(
        chat_client,
        model,
        embed_client,
        embedding_model,
        embedding_dims,
        messages,
        max_steps,
        batch_size,
        language,
    )
    if state.archival_vectors is None:
        state.archival_vectors = _empty_vectors(embedding_dims)

    try:
        _save_to_disk(save_dir, digest, state.core_blocks, state.archival_docs, state.archival_vectors)
    except Exception as exc:  # noqa: BLE001
        print(f"[memory-cache] save failed method={METHOD} save_dir={save_dir}: {exc}", flush=True)

    index = _MemoryIndex(
        user_id=user_id,
        save_dir=save_dir,
        core_blocks=state.core_blocks,
        archival_docs=state.archival_docs,
        archival_vectors=state.archival_vectors,
    )
    with _cache_guard:
        _inproc_cache[digest] = index
    return index


# ---------------------------------------------------------------------------
# 上下文组装（对齐其他 baseline：把记忆拼成 context_text 注入答案模型）
# ---------------------------------------------------------------------------
CONTEXT_TEMPLATE = """The following is the persistent memory that a MemGPT-style agent maintains about the user.

# Core memory (editable facts the agent keeps in-context).
<CORE_MEMORY>
{core_memory}
</CORE_MEMORY>

# Archival memory passages most relevant to the current question.
<ARCHIVAL_MEMORY>
{archival_memory}
</ARCHIVAL_MEMORY>"""


def _bullets(items: list[str]) -> str:
    return "\n".join(f"  - {item}" for item in items if str(item).strip())


def _build_context_text(core_blocks: dict[str, str], archival: list[dict[str, Any]]) -> str:
    core_lines = []
    for label, value in core_blocks.items():
        value = (value or "").strip()
        if value:
            core_lines.append(f"[{label}] {value}")
    core_text = _bullets(core_lines) if core_lines else "  (empty)"
    archival_lines = [
        f"  - {str(m.get('used_content') or '').strip()}"
        for m in archival
        if str(m.get("used_content") or "").strip()
    ]
    archival_text = "\n".join(archival_lines) if archival_lines else "  (no relevant passages)"
    return CONTEXT_TEMPLATE.format(core_memory=core_text, archival_memory=archival_text)


# ---------------------------------------------------------------------------
# 适配器入口
# ---------------------------------------------------------------------------
def build_context(
    prior_dialogue: str,
    user_question: str,
    eval_config: BaselineEvalConfig,
    *,
    sample_key: str | int | None = None,
) -> BaselineContext:
    messages = parse_dialogue_to_messages(prior_dialogue)
    transcript = _transcript(messages)
    index = _get_or_build_index(transcript, messages, eval_config)

    _, embed_client = _make_clients(eval_config)
    archival = _search_archival(
        index,
        user_question,
        embed_client,
        _resolve_embedding_model(eval_config),
        _resolve_embedding_dims(eval_config),
        eval_config.top_k,
    )

    context_text = _build_context_text(index.core_blocks, archival)

    core_memories = [
        {"type": "core", "content": v, "used_content": v, "metadata": {"block": k}}
        for k, v in index.core_blocks.items()
        if (v or "").strip()
    ]

    return BaselineContext(
        context_text=context_text,
        retrieved_memories=jsonable_memories(core_memories + archival),
        user_id=index.user_id,
        save_dir=index.save_dir,
        method=METHOD,
        top_k=eval_config.top_k,
    )
