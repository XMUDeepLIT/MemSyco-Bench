from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import BaselineContext, BaselineEvalConfig
from .common import jsonable_memories, parse_dialogue_to_messages


# ---------------------------------------------------------------------------
# Supermemory baseline adapter —— 本地复刻版（不依赖 supermemory 官方 API / SDK）。
#
# Supermemory（https://supermemory.ai）的核心引擎是闭源服务端，但其机制在官方文档中有
# 清晰说明。这里在本地忠实复刻它的记忆机制，对接本仓库的评测基础设施（与 MemoryBank /
# A-MEM 一致：LLM 与 embedding 分别直连 MEMORY_* 和 MEMORY_EMBEDDING_* 配置的服务，
# 记忆只构建一次并落盘缓存，后续同一段对话的不同问题复用）。
#
# 复刻的 supermemory 机制（对齐官方 user-profiles / add-memories / search 文档）：
#   1. 摄入：把 prior dialogue 交给 LLM，抽取关于「用户」的原子事实/记忆，并按
#      static（稳定长期事实：身份、背景、偏好、专长）/ dynamic（近期、临时状态：当前在做
#      什么、最近事件）两层分类；解决矛盾/更新，只保留最新版本。
#   2. 索引：把每条记忆用 bge-m3 向量化、L2 归一化后落盘。
#   3. 检索（默认 profile 模式，对齐 client.profile）：一次返回
#        - profile.static  = 全部 static 记忆
#        - profile.dynamic = 全部 dynamic 记忆
#        - search_results  = 对 query 做 top-k 语义检索得到的最相关记忆（混合）
#      并把三者组合成注入答案模型的上下文（对齐官方把 profile + search 合并的用法）。
#      也可设 SUPERMEMORY_RETRIEVAL_MODE=search 只用 top-k 语义检索。
#
# 说明：本地没有独立的 RAG 文档库，因此 hybrid / memories 两种检索都退化为对「抽取出的
# 记忆」做语义检索（与 supermemory「memory 不是 RAG」的语义一致：检索的是关于用户的事实）。
# ---------------------------------------------------------------------------


METHOD = "Supermemory"

DEFAULT_LLM_MODEL = "deepseek-v4-flash"
DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_EMBED_BATCH = 64
DEFAULT_REQUEST_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_MEMORIES = 40
DEFAULT_RETRIEVAL_MODE = "profile"  # profile | search
# 公平性设置：所有 baseline 的 LLM 摄入调用统一 max_tokens / temperature。
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.7

_COMPLETE_MARKER = ".supermemory_complete.json"
_CACHE_VERSION = 1

_cache_guard = threading.Lock()
_inproc_cache: dict[str, "_MemoryIndex"] = {}


@dataclass
class _MemoryIndex:
    user_id: str
    save_dir: str
    memories: list[dict[str, Any]]          # [{"text", "type": static|dynamic}]
    vectors: Any                            # numpy.ndarray (n, dim), L2-normalized
    static: list[str] = field(default_factory=list)
    dynamic: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 配置解析（与 MemoryBank 一致：LLM 与 embedding 使用各自的 MEMORY_* 配置）
# ---------------------------------------------------------------------------
def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _language() -> str:
    lang = (_env("SUPERMEMORY_LANGUAGE", "MEMORYBANK_LANGUAGE") or "en").lower()
    return "cn" if lang in {"cn", "zh", "zh-cn", "chinese"} else "en"


def _retrieval_mode() -> str:
    mode = (_env("SUPERMEMORY_RETRIEVAL_MODE") or DEFAULT_RETRIEVAL_MODE).lower()
    if mode not in {"profile", "search"}:
        raise ValueError(
            f"Unsupported SUPERMEMORY_RETRIEVAL_MODE={mode!r}; use 'profile' or 'search'."
        )
    return mode


def _max_memories() -> int:
    raw = _env("SUPERMEMORY_MAX_MEMORIES")
    return max(1, int(raw)) if raw else DEFAULT_MAX_MEMORIES


def _request_timeout() -> float:
    raw = _env("SUPERMEMORY_REQUEST_TIMEOUT", "MEMORY_REQUEST_TIMEOUT")
    return float(raw) if raw else DEFAULT_REQUEST_TIMEOUT


def _max_retries() -> int:
    raw = _env("SUPERMEMORY_MAX_RETRIES", "MEMORY_API_MAX_RETRIES")
    return max(1, int(raw)) if raw else DEFAULT_MAX_RETRIES


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


def _embed_texts(embed_client: Any, model: str, texts: list[str], dims: int | None) -> Any:
    import numpy as np  # noqa: PLC0415

    vectors: list[list[float]] = []
    extra: dict[str, Any] = {"dimensions": dims} if dims else {}
    for start in range(0, len(texts), DEFAULT_EMBED_BATCH):
        batch = texts[start : start + DEFAULT_EMBED_BATCH]
        last_exc: Exception | None = None
        for _ in range(_max_retries()):
            try:
                resp = embed_client.embeddings.create(model=model, input=batch, **extra)
                vectors.extend([item.embedding for item in resp.data])
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(1.0)
        if last_exc is not None:
            raise RuntimeError(f"Supermemory embedding call failed: {last_exc}") from last_exc
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return _l2_normalize(arr)


def _l2_normalize(arr: Any) -> Any:
    import numpy as np  # noqa: PLC0415

    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms


# ---------------------------------------------------------------------------
# 记忆抽取（LLM）：从对话抽取关于用户的原子事实，并分 static / dynamic
# ---------------------------------------------------------------------------
def _transcript(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip().capitalize()
        content = str(message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _extraction_messages(transcript: str, language: str, max_memories: int) -> list[dict[str, str]]:
    if language == "cn":
        system = (
            "你是一个记忆抽取引擎（类似 Supermemory）。阅读用户与助手之间的对话，"
            "抽取关于【用户】的原子化、可独立成立的事实/记忆（偏好、身份、背景、目标、"
            "当前在做的事、观点、决定、约束等）。\n"
            "为每条记忆分类：\n"
            "- \"static\"：长期稳定、很少变化的事实（身份、背景、长期偏好、专长）。\n"
            "- \"dynamic\"：近期上下文或临时状态（当前正在做的事、最近发生的事、短期目标）。\n"
            "规则：\n"
            "- 只抽取关于用户（及其相关情况）的事实，不要抽取助手的泛泛陈述。\n"
            "- 每条记忆是一句简洁的话，用第三人称书写（如「用户……」）。\n"
            "- 处理矛盾与更新：若后文改变了前文的事实，只保留最新版本。\n"
            "- 忽略没有持久信息价值的寒暄闲聊。\n"
            f"- 最多输出 {max_memories} 条记忆。\n"
            "只返回合法 JSON，格式："
            "{\"memories\": [{\"text\": \"……\", \"type\": \"static|dynamic\"}, ...]}"
        )
        user = f"对话内容：\n{transcript}"
    else:
        system = (
            "You are a memory extraction engine (like Supermemory). Read the conversation "
            "between a user and an assistant and extract atomic, self-contained facts/memories "
            "about the USER (preferences, identity, background, goals, current activities, "
            "opinions, decisions, constraints).\n"
            "Classify each memory:\n"
            "- \"static\": long-term, stable facts that rarely change (identity, background, "
            "durable preferences, expertise).\n"
            "- \"dynamic\": recent context or temporary states (what they are currently doing, "
            "recent events, short-term goals).\n"
            "Rules:\n"
            "- Only extract facts about the user (and their world), not the assistant's generic "
            "statements.\n"
            "- Each memory is one concise sentence written in the third person (\"The user ...\").\n"
            "- Resolve contradictions/updates: if a later message changes an earlier fact, keep "
            "only the latest version.\n"
            "- Ignore chit-chat with no durable information.\n"
            f"- Output at most {max_memories} memories.\n"
            "Return ONLY valid JSON of the form: "
            "{\"memories\": [{\"text\": \"...\", \"type\": \"static|dynamic\"}, ...]}"
        )
        user = f"Conversation:\n{transcript}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_json_memories(raw: str, max_memories: int) -> list[dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return []
    # 去掉 ```json ... ``` 之类的代码块包裹
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    data: Any = None
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:  # noqa: BLE001
                data = None
    if data is None:
        return []

    items = data.get("memories") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []

    memories: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            mem_text, mem_type = item, "dynamic"
        elif isinstance(item, dict):
            mem_text = str(item.get("text") or item.get("memory") or item.get("fact") or "").strip()
            mem_type = str(item.get("type") or item.get("category") or "dynamic").strip().lower()
        else:
            continue
        mem_text = mem_text.strip()
        if not mem_text or mem_text in seen:
            continue
        if mem_type not in {"static", "dynamic"}:
            mem_type = "dynamic"
        memories.append({"text": mem_text, "type": mem_type})
        seen.add(mem_text)
        if len(memories) >= max_memories:
            break
    return memories


def _extract_memories(
    chat_client: Any,
    model: str,
    transcript: str,
    language: str,
    max_memories: int,
) -> list[dict[str, Any]]:
    if not transcript.strip():
        return []
    messages = _extraction_messages(transcript, language, max_memories)
    last_exc: Exception | None = None
    for _ in range(_max_retries()):
        try:
            resp = chat_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                top_p=1.0,
            )
            content = resp.choices[0].message.content or ""
            return _parse_json_memories(content, max_memories)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.0)
    raise RuntimeError(f"Supermemory memory extraction LLM call failed: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# 磁盘缓存（对齐 memorybank.py）
# ---------------------------------------------------------------------------
def _digest(
    transcript: str,
    *,
    model: str,
    embedding_model: str,
    embedding_dims: int | None,
    language: str,
    max_memories: int,
) -> str:
    h = hashlib.sha1()
    for part in (
        METHOD,
        model,
        embedding_model,
        str(embedding_dims or ""),
        str(LLM_MAX_TOKENS),
        str(LLM_TEMPERATURE),
        language,
        str(max_memories),
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
        memories = payload.get("memories") or []
        vectors = np.load(_vectors_path(save_dir))
    except Exception:  # noqa: BLE001
        return None
    if len(memories) != len(vectors):
        return None
    return _MemoryIndex(
        user_id=user_id,
        save_dir=save_dir,
        memories=memories,
        vectors=vectors,
        static=[m["text"] for m in memories if m.get("type") == "static"],
        dynamic=[m["text"] for m in memories if m.get("type") == "dynamic"],
    )


def _save_to_disk(save_dir: str, digest: str, memories: list[dict[str, Any]], vectors: Any) -> None:
    import numpy as np  # noqa: PLC0415

    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    _memory_path(save_dir).write_text(
        json.dumps({"memories": memories}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(_vectors_path(save_dir), vectors)
    marker = {
        "method": METHOD,
        "digest": digest,
        "memories_count": len(memories),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": _CACHE_VERSION,
    }
    tmp = _marker_path(save_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_marker_path(save_dir))


def _get_or_build_index(
    transcript: str,
    eval_config: BaselineEvalConfig,
) -> _MemoryIndex | None:
    if not transcript.strip():
        return None

    language = _language()
    model = _resolve_llm_model(eval_config)
    embedding_model = _resolve_embedding_model(eval_config)
    embedding_dims = _resolve_embedding_dims(eval_config)
    max_memories = _max_memories()

    digest = _digest(
        transcript,
        model=model,
        embedding_model=embedding_model,
        embedding_dims=embedding_dims,
        language=language,
        max_memories=max_memories,
    )
    user_id = f"pm_sm_{digest}"
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
    memories = _extract_memories(chat_client, model, transcript, language, max_memories)
    if not memories:
        return None
    vectors = _embed_texts(embed_client, embedding_model, [m["text"] for m in memories], embedding_dims)

    index = _MemoryIndex(
        user_id=user_id,
        save_dir=save_dir,
        memories=memories,
        vectors=vectors,
        static=[m["text"] for m in memories if m.get("type") == "static"],
        dynamic=[m["text"] for m in memories if m.get("type") == "dynamic"],
    )
    try:
        _save_to_disk(save_dir, digest, memories, vectors)
    except Exception as exc:  # noqa: BLE001
        print(f"[memory-cache] save failed method={METHOD} save_dir={save_dir}: {exc}", flush=True)
    with _cache_guard:
        _inproc_cache[digest] = index
    return index


# ---------------------------------------------------------------------------
# 检索（对 query 做 top-k 余弦相似，对齐 search.memories）
# ---------------------------------------------------------------------------
def _search(index: _MemoryIndex, query: str, eval_config: BaselineEvalConfig, top_k: int) -> list[dict[str, Any]]:
    import numpy as np  # noqa: PLC0415

    if not index.memories:
        return []
    _, embed_client = _make_clients(eval_config)
    query_vec = _embed_texts(
        embed_client,
        _resolve_embedding_model(eval_config),
        [query],
        _resolve_embedding_dims(eval_config),
    )[0]
    scores = index.vectors @ query_vec
    k = max(1, min(int(top_k), len(index.memories)))
    top_idx = np.argsort(-scores)[:k]
    results: list[dict[str, Any]] = []
    for rank, i in enumerate(top_idx, start=1):
        mem = index.memories[int(i)]
        text = str(mem.get("text") or "").strip()
        if not text:
            continue
        results.append(
            {
                "type": "search_result",
                "content": text,
                "used_content": text,
                "score": float(scores[int(i)]),
                "metadata": {"rank": rank, "memory_type": mem.get("type", "dynamic")},
            }
        )
    return results


# ---------------------------------------------------------------------------
# 上下文组装（对齐 client.profile：static + dynamic + relevant memories）
# ---------------------------------------------------------------------------
CONTEXT_TEMPLATE = """The following is the persistent memory Supermemory maintains about the user.

# Stable, long-term facts about the user (static profile).
<STATIC_PROFILE>
{static_profile}
</STATIC_PROFILE>

# Recent / changing context about the user (dynamic profile).
<DYNAMIC_PROFILE>
{dynamic_profile}
</DYNAMIC_PROFILE>

# The most relevant memories retrieved for the current question.
<RELEVANT_MEMORIES>
{relevant_memories}
</RELEVANT_MEMORIES>"""


def _bullets(items: list[str]) -> str:
    return "\n".join(f"  - {item}" for item in items if str(item).strip())


def _build_context_text(static: list[str], dynamic: list[str], memories: list[dict[str, Any]]) -> str:
    if not static and not dynamic and not memories:
        return "[NO RETRIEVED MEMORIES]"
    relevant_lines = [
        f"  - {str(m.get('used_content') or '').strip()}"
        for m in memories
        if str(m.get("used_content") or "").strip()
    ]
    return CONTEXT_TEMPLATE.format(
        static_profile=_bullets(static) if static else "  (empty)",
        dynamic_profile=_bullets(dynamic) if dynamic else "  (empty)",
        relevant_memories="\n".join(relevant_lines) if relevant_lines else "  (no relevant memories)",
    )


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
    index = _get_or_build_index(transcript, eval_config)

    if index is None:
        return BaselineContext(
            context_text="[NO RETRIEVED MEMORIES]",
            retrieved_memories=[],
            user_id="",
            save_dir="",
            method=METHOD,
            top_k=eval_config.top_k,
        )

    memories = _search(index, user_question, eval_config, eval_config.top_k)

    if _retrieval_mode() == "profile":
        static, dynamic = index.static, index.dynamic
    else:
        # 纯 search 模式：不暴露完整 profile，只给检索到的相关记忆。
        static, dynamic = [], []

    profile_memories = [
        {"type": "static_profile", "content": text, "used_content": text} for text in static
    ] + [
        {"type": "dynamic_profile", "content": text, "used_content": text} for text in dynamic
    ]

    context_text = _build_context_text(static, dynamic, memories)
    return BaselineContext(
        context_text=context_text,
        retrieved_memories=jsonable_memories(profile_memories + memories),
        user_id=index.user_id,
        save_dir=index.save_dir,
        method=METHOD,
        top_k=eval_config.top_k,
    )
