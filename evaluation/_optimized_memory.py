"""Shared optimized memory-cache helpers for the active evaluation tasks.

This module patches baseline memory construction so identical complete prior
dialogues are built once per process. Each sample still retrieves from one
memory layer containing the full context, preserving baseline retrieval
semantics.
"""

from __future__ import annotations

import hashlib
import gc
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from baselines import BaselineContext, BaselineEvalConfig
from baselines.common import (
    format_retrieved_memories,
    jsonable_memories,
    parse_dialogue_to_messages,
    retry_embedding_query,
    timestamp_for_turn,
)
import _objective_base as base_eval


_TOOLKIT_METHODS = {"NaiveRAG", "A-MEM", "MemZero"}
_NATIVE_LIGHTMEM_METHODS = {"LightMem"}
_cache_guard = threading.Lock()
_entry_locks: dict[tuple[Any, ...], threading.Lock] = {}
_entries: dict[tuple[Any, ...], "_MemoryEntry"] = {}
_COMPLETE_MARKER = ".memory_complete.json"
_SECRET_FINGERPRINT_VALUE = "<secret>"
_IGNORED_FINGERPRINT_VALUE = "<ignored>"
_IGNORED_FINGERPRINT_LABELS = {
    "save_root",
    "memory_base_url",
    "embedding_base_url",
}
_EMBEDDING_MODEL_ALIASES = {
    "baai/bge-m3": "bge-m3",
}
_DEFAULT_NATIVE_CACHE_MAX_ENTRIES = 1
# Local Qdrant (mem0) cannot keep one client per sample in RAM for full datasets.
_DEFAULT_TOOLKIT_CACHE_MAX_ENTRIES = 1
_DISK_QDRANT_TOOLKIT_METHODS = {"NaiveRAG", "MemZero", "A-MEM"}


def _normalize_embedding_model_for_fingerprint(model: Any) -> str:
    text = "" if model is None else str(model).strip()
    folded = text.lower()
    return _EMBEDDING_MODEL_ALIASES.get(folded, folded)


@dataclass
class _MemoryEntry:
    layer: Any
    user_id: str
    save_dir: str
    method: str
    lock: threading.Lock


def _sha1_parts(parts: list[str]) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _message_key(message: dict[str, str]) -> str:
    role = str(message.get("role") or "")
    content = str(message.get("content") or "")
    h = hashlib.sha1()
    h.update(role.encode("utf-8"))
    h.update(b"\0")
    h.update(content.encode("utf-8"))
    return h.hexdigest()


def _config_fingerprint(eval_config: BaselineEvalConfig) -> tuple[Any, ...]:
    return (
        eval_config.method,
        eval_config.config_path,
        eval_config.save_root,
        _SECRET_FINGERPRINT_VALUE if eval_config.api_key else "",
        _IGNORED_FINGERPRINT_VALUE,
        eval_config.llm_model,
        _normalize_embedding_model_for_fingerprint(eval_config.embedding_model),
        eval_config.embedding_dims,
        _SECRET_FINGERPRINT_VALUE if eval_config.embedding_api_key else "",
        _IGNORED_FINGERPRINT_VALUE,
    )


def _lock_for_key(key: tuple[Any, ...]) -> threading.Lock:
    with _cache_guard:
        lock = _entry_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _entry_locks[key] = lock
        return lock


def _context_user_id(method: str, digest: str) -> str:
    h = hashlib.sha1()
    h.update(method.encode("utf-8"))
    h.update(b"\0")
    h.update(digest.encode("utf-8"))
    return "pmfull_" + h.hexdigest()[:24]


def _marker_path(save_dir: str) -> Path:
    return Path(save_dir) / _COMPLETE_MARKER


def _fingerprint_for_marker(eval_config: BaselineEvalConfig) -> list[str]:
    return ["" if value is None else str(value) for value in _config_fingerprint(eval_config)]


_FINGERPRINT_LABELS = (
    "method",
    "config_path",
    "save_root",
    "memory_api_key",
    "memory_base_url",
    "memory_llm_model",
    "embedding_model",
    "embedding_dims",
    "embedding_api_key",
    "embedding_base_url",
)
_SENSITIVE_FINGERPRINT_LABELS = {"memory_api_key", "embedding_api_key"}


def _display_fingerprint_value(label: str, value: Any) -> str:
    text = "" if value is None else str(value)
    if label not in _SENSITIVE_FINGERPRINT_LABELS:
        return repr(text)
    if text == _SECRET_FINGERPRINT_VALUE:
        return repr(text)
    if not text:
        return "''"
    if len(text) <= 12:
        masked = "*" * len(text)
    else:
        masked = f"{text[:6]}...{text[-4:]}"
    return f"{masked!r} len={len(text)}"


def _fingerprint_values_equal(label: str, stored: Any, current: Any) -> bool:
    if label in _IGNORED_FINGERPRINT_LABELS:
        return True
    if label in _SENSITIVE_FINGERPRINT_LABELS:
        stored_text = "" if stored is None else str(stored)
        current_text = "" if current is None else str(current)
        if stored_text == _SECRET_FINGERPRINT_VALUE or current_text == _SECRET_FINGERPRINT_VALUE:
            return True
        return bool(stored_text) == bool(current_text)
    if label == "embedding_model":
        return _normalize_embedding_model_for_fingerprint(stored) == _normalize_embedding_model_for_fingerprint(current)
    return stored == current


def _fingerprints_match(stored: Any, current: list[str]) -> bool:
    if not isinstance(stored, list):
        return False
    if len(stored) != len(current):
        return False
    for idx, expected in enumerate(current):
        label = _FINGERPRINT_LABELS[idx] if idx < len(_FINGERPRINT_LABELS) else f"extra_{idx}"
        if not _fingerprint_values_equal(label, stored[idx], expected):
            return False
    return True


def _log_marker_mismatch(save_dir: str, method: str, digest: str, eval_config: BaselineEvalConfig) -> None:
    marker = _marker_path(save_dir)
    expected = _fingerprint_for_marker(eval_config)
    try:
        data = json.loads(marker.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(
            f"[memory-cache] marker mismatch method={method} save_dir={save_dir}: "
            f"failed to read {marker}: {exc}",
            flush=True,
        )
        return

    actual_method = data.get("method")
    actual_digest = data.get("digest")
    actual_fp = data.get("config_fingerprint")
    if not isinstance(actual_fp, list):
        actual_fp = []

    print(
        f"[memory-cache] marker mismatch method={method} save_dir={save_dir}",
        flush=True,
    )
    print(
        f"[memory-cache] marker method: stored={actual_method!r} expected={method!r} same={actual_method == method}",
        flush=True,
    )
    print(
        f"[memory-cache] marker digest: stored={actual_digest!r} expected={digest!r} same={actual_digest == digest}",
        flush=True,
    )

    max_len = max(len(expected), len(actual_fp), len(_FINGERPRINT_LABELS))
    for idx in range(max_len):
        label = _FINGERPRINT_LABELS[idx] if idx < len(_FINGERPRINT_LABELS) else f"extra_{idx}"
        stored = actual_fp[idx] if idx < len(actual_fp) else None
        current = expected[idx] if idx < len(expected) else None
        same = _fingerprint_values_equal(label, stored, current)
        if same:
            continue
        print(
            "[memory-cache] marker fingerprint mismatch "
            f"[{idx}] {label}: "
            f"stored={_display_fingerprint_value(label, stored)} "
            f"current={_display_fingerprint_value(label, current)}",
            flush=True,
        )


def _is_complete_marker_valid(save_dir: str, method: str, digest: str, eval_config: BaselineEvalConfig) -> bool:
    marker = _marker_path(save_dir)
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    return data.get("method") == method and data.get("digest") == digest


def _warn_marker_fingerprint_drift(
    save_dir: str,
    method: str,
    eval_config: BaselineEvalConfig,
) -> None:
    marker = _marker_path(save_dir)
    try:
        data = json.loads(marker.read_text(encoding="utf-8-sig"))
    except Exception:
        return
    stored_fp = data.get("config_fingerprint")
    current_fp = _fingerprint_for_marker(eval_config)
    if _fingerprints_match(stored_fp, current_fp):
        return
    print(
        f"[memory-cache] marker fingerprint drift (reusing disk store) method={method} save_dir={save_dir}",
        flush=True,
    )


def _write_complete_marker(
    save_dir: str,
    *,
    method: str,
    digest: str,
    eval_config: BaselineEvalConfig,
    messages_count: int,
) -> None:
    path = _marker_path(save_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "digest": digest,
        "config_fingerprint": _fingerprint_for_marker(eval_config),
        "messages_count": messages_count,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": 1,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False,
                   indent=2), encoding="utf-8")
    tmp.replace(path)


def _toolkit_layer_has_memory(layer: Any, user_id: str) -> bool:
    if hasattr(layer, "_has_any_memory"):
        try:
            return bool(layer._has_any_memory(user_id=user_id))
        except TypeError:
            return bool(layer._has_any_memory(user_id))
        except Exception:
            return False
    memory_layer = getattr(layer, "memory_layer", None)
    if memory_layer is not None:
        memories = getattr(memory_layer, "memories", None)
        if isinstance(memories, dict) and memories:
            return True
        retriever = getattr(memory_layer, "retriever", None)
        collection = getattr(retriever, "collection", None)
        if collection is not None:
            try:
                return bool(collection.count())
            except Exception:
                try:
                    got = collection.get(limit=1)
                    return bool(got.get("ids"))
                except Exception:
                    return False
    return False


def _native_lightmem_point_count(lightmem: Any) -> int:
    retriever = getattr(lightmem, "embedding_retriever", None)
    if retriever is None:
        return 0
    collection_name = getattr(retriever, "collection_name", None)
    client = getattr(retriever, "client", None)
    if not collection_name or client is None:
        return 0
    try:
        info = client.get_collection(collection_name=collection_name)
        count = getattr(info, "points_count", None)
        if count is not None:
            return int(count)
    except Exception:
        pass
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(points)
    except Exception:
        return 0


def _native_lightmem_has_memory(lightmem: Any) -> bool:
    return _native_lightmem_point_count(lightmem) > 0


def _remove_native_lightmem_save_dir(save_dir: str) -> None:
    import shutil

    path = Path(save_dir)
    if not path.exists():
        return
    shutil.rmtree(path)
    print(
        f"[memory-cache] removed stale native LightMem dir save_dir={save_dir}",
        flush=True,
    )


def _open_native_lightmem_for_retrieval(config: dict[str, Any], *, method: str) -> Any:
    from baselines.lightmem import _is_full_method, retrieval_only_lightmem_config
    from baselines.toolkit.runner import _prepare_import_path

    _prepare_import_path()
    from lightmem.memory.lightmem import LightMemory  # type: ignore

    load_config = (
        retrieval_only_lightmem_config(config)
        if _is_full_method(method)
        else config
    )
    return LightMemory.from_config(load_config)


def _get_entry(
    messages: list[dict[str, str]],
    eval_config: BaselineEvalConfig,
    *,
    digest: str,
) -> _MemoryEntry | None:
    if not messages:
        return None
    method = eval_config.method
    key = _config_fingerprint(eval_config) + ("full_context", digest)
    key_lock = _lock_for_key(key)
    with key_lock:
        entry = _entries.get(key)
        if entry is None:
            entry = (
                _build_native_lightmem_entry(
                    messages,
                    eval_config,
                    digest=digest,
                )
                if method in _NATIVE_LIGHTMEM_METHODS
                else _build_toolkit_entry(
                    messages,
                    eval_config,
                    digest=digest,
                )
            )
            _entries[key] = entry
            _evict_cached_entries_if_needed(eval_config, keep_key=key)
        return entry


def _cache_max_entries(eval_config: BaselineEvalConfig) -> int | None:
    raw = os.environ.get("BASELINE_OPT_MEMORY_CACHE_MAX_ENTRIES")
    if raw is not None:
        value = int(raw)
        return None if value <= 0 else value
    if eval_config.method in _NATIVE_LIGHTMEM_METHODS:
        return _DEFAULT_NATIVE_CACHE_MAX_ENTRIES
    if eval_config.method in _DISK_QDRANT_TOOLKIT_METHODS:
        return _DEFAULT_TOOLKIT_CACHE_MAX_ENTRIES
    return None


def _evict_cached_entries_if_needed(eval_config: BaselineEvalConfig, *, keep_key: tuple[Any, ...]) -> None:
    max_entries = _cache_max_entries(eval_config)
    if max_entries is None:
        return

    while len(_entries) > max_entries:
        victim_key = next((key for key in _entries if key != keep_key), None)
        if victim_key is None:
            return
        victim = _entries.pop(victim_key)
        _entry_locks.pop(victim_key, None)
        with victim.lock:
            if victim.method in _NATIVE_LIGHTMEM_METHODS:
                _close_native_lightmem(victim.layer)
            else:
                _close_toolkit_layer(victim.layer)
        print(
            f"[memory-cache] evicted method={victim.method} user_id={victim.user_id} "
            f"save_dir={victim.save_dir}",
            flush=True,
        )


def _build_toolkit_entry(
    messages: list[dict[str, str]],
    eval_config: BaselineEvalConfig,
    *,
    digest: str,
) -> _MemoryEntry:
    from baselines.toolkit.runner import (
        _apply_api_overrides,
        _format_llm_model_for_method,
        _load_method_config,
        _prepare_import_path,
    )

    method = eval_config.method
    _prepare_import_path()
    config_dict = _load_method_config(method, eval_config)
    user_id = _context_user_id(method, digest)
    save_dir = str((eval_config.save_root / method / user_id).resolve())

    config_dict["user_id"] = user_id
    config_dict["save_dir"] = save_dir
    if "collection_name" in config_dict or method in {"NaiveRAG", "MemZero"}:
        config_dict["collection_name"] = user_id
    if eval_config.llm_model:
        config_dict["llm_model"] = _format_llm_model_for_method(
            method, eval_config.llm_model)
    if eval_config.embedding_model:
        config_dict["retriever_name_or_path"] = eval_config.embedding_model
    if eval_config.embedding_dims is not None:
        config_dict["embedding_model_dims"] = eval_config.embedding_dims
    _apply_api_overrides(method, config_dict, eval_config)

    from memories import CONFIG_MAPPING, MEMORY_LAYERS_MAPPING  # type: ignore

    config = CONFIG_MAPPING[method](**config_dict)
    layer = MEMORY_LAYERS_MAPPING[method](config)
    marker_valid = _is_complete_marker_valid(
        save_dir, method, digest, eval_config)
    if method == "A-MEM" and marker_valid:
        try:
            if layer.load_memory(user_id):
                print(
                    f"[memory-cache] disk hit method={method} user_id={user_id} save_dir={save_dir}", flush=True)
                return _MemoryEntry(layer=layer, user_id=user_id, save_dir=save_dir, method=method, lock=threading.Lock())
        except Exception as exc:
            print(
                f"[memory-cache] disk load failed method={method} user_id={user_id}: {exc}", flush=True)
    has_existing_memory = _toolkit_layer_has_memory(layer, user_id)
    if has_existing_memory and (marker_valid or not _marker_path(save_dir).exists()):
        if not marker_valid:
            _write_complete_marker(
                save_dir,
                method=method,
                digest=digest,
                eval_config=eval_config,
                messages_count=len(messages),
            )
            print(
                f"[memory-cache] adopted existing disk store without marker method={method} "
                f"user_id={user_id} save_dir={save_dir}",
                flush=True,
            )
        else:
            print(
                f"[memory-cache] disk hit method={method} user_id={user_id} save_dir={save_dir}", flush=True)
        return _MemoryEntry(layer=layer, user_id=user_id, save_dir=save_dir, method=method, lock=threading.Lock())
    if has_existing_memory:
        _log_marker_mismatch(save_dir, method, digest, eval_config)
        raise RuntimeError(
            f"Existing memory store has an incompatible completion marker: method={method} save_dir={save_dir}. "
            "Use a different --memory-save-root or remove the stale store before rebuilding."
        )

    print(
        f"[memory-cache] disk miss; building method={method} user_id={user_id} save_dir={save_dir}", flush=True)
    for idx, message in enumerate(messages):
        add_kwargs: dict[str, Any] = {"timestamp": timestamp_for_turn(idx)}
        if method == "NaiveRAG":
            add_kwargs["turn_index"] = idx
        layer.add_message(message, **add_kwargs)
    if hasattr(layer, "save_memory"):
        layer.save_memory()
    _write_complete_marker(
        save_dir,
        method=method,
        digest=digest,
        eval_config=eval_config,
        messages_count=len(messages),
    )

    return _MemoryEntry(layer=layer, user_id=user_id, save_dir=save_dir, method=method, lock=threading.Lock())


def _build_native_lightmem_entry(
    messages: list[dict[str, str]],
    eval_config: BaselineEvalConfig,
    *,
    digest: str,
) -> _MemoryEntry:
    from baselines.lightmem import (
        _is_full_method,
        _load_native_lightmem_config,
        ingest_full_lightmem,
        ingest_raw_dialogue_lightmem,
    )
    from baselines.toolkit.runner import _prepare_import_path

    _prepare_import_path()
    from lightmem.memory.lightmem import LightMemory  # type: ignore

    method = eval_config.method
    user_id = _context_user_id(method, digest)
    save_dir = str((eval_config.save_root / method / user_id).resolve())
    save_path = Path(save_dir)
    config = _load_native_lightmem_config(
        eval_config, user_id=user_id, save_dir=save_dir)
    marker_valid = _is_complete_marker_valid(
        save_dir, method, digest, eval_config)
    marker_exists = _marker_path(save_dir).exists()

    def _make_entry(lightmem: Any, *, mode: str) -> _MemoryEntry:
        print(
            f"[memory-cache] disk hit ({mode}) method={method} user_id={user_id} save_dir={save_dir}",
            flush=True,
        )
        return _MemoryEntry(
            layer=lightmem,
            user_id=user_id,
            save_dir=save_dir,
            method=method,
            lock=threading.Lock(),
        )

    if marker_valid:
        _warn_marker_fingerprint_drift(save_dir, method, eval_config)
        return _make_entry(
            _open_native_lightmem_for_retrieval(config, method=method),
            mode="retrieval-only",
        )

    if marker_exists:
        _log_marker_mismatch(save_dir, method, digest, eval_config)
        raise RuntimeError(
            f"Existing memory store has an incompatible completion marker: method={method} save_dir={save_dir}. "
            "Use a different --memory-save-root or remove the stale store before rebuilding."
        )

    if save_path.exists():
        print(
            f"[memory-cache] removing incomplete store without marker method={method} "
            f"user_id={user_id} save_dir={save_dir}",
            flush=True,
        )
        _remove_native_lightmem_save_dir(save_dir)

    lightmem = LightMemory.from_config(config)
    print(
        f"[memory-cache] disk miss (fresh) method={method} user_id={user_id} "
        f"save_dir={save_dir} messages={len(messages)}",
        flush=True,
    )
    if _is_full_method(method):
        ingest_full_lightmem(lightmem, messages)
    else:
        ingest_raw_dialogue_lightmem(lightmem, messages)
    _write_complete_marker(
        save_dir,
        method=method,
        digest=digest,
        eval_config=eval_config,
        messages_count=len(messages),
    )

    return _MemoryEntry(
        layer=lightmem,
        user_id=user_id,
        save_dir=save_dir,
        method=method,
        lock=threading.Lock(),
    )


def _retrieve_from_entry(entry: _MemoryEntry, user_question: str, top_k: int) -> list[dict[str, Any]]:
    with entry.lock:
        if entry.method in _NATIVE_LIGHTMEM_METHODS:
            retrieved_strings = retry_embedding_query(
                lambda: entry.layer.retrieve(user_question, limit=top_k),
                method=entry.method,
            )
            return [
                {"content": text, "metadata": {"rank": i}, "used_content": text}
                for i, text in enumerate(retrieved_strings, start=1)
            ]
        return retry_embedding_query(
            lambda: entry.layer.retrieve(user_question, k=top_k),
            method=entry.method,
        )


def _close_quietly(obj: Any) -> None:
    close = getattr(obj, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        pass


def _close_toolkit_layer(layer: Any) -> None:
    memory_layer = getattr(layer, "memory_layer", None)
    if memory_layer is None:
        return
    for attr in ("vector_store", "_telemetry_vector_store"):
        store = getattr(memory_layer, attr, None)
        client = getattr(store, "client", None)
        if client is not None:
            _close_quietly(client)


def _close_native_lightmem(lightmem: Any) -> None:
    for attr in ("embedding_retriever", "context_retriever", "summary_retriever"):
        retriever = getattr(lightmem, attr, None)
        client = getattr(retriever, "client", None)
        if client is not None:
            _close_quietly(client)
        _close_quietly(retriever)
    for attr in (
        "compressor",
        "segmenter",
        "manager",
        "text_embedder",
        "embedding_retriever",
        "context_retriever",
        "summary_retriever",
        "senmem_buffer_manager",
        "shortmem_buffer_manager",
    ):
        if hasattr(lightmem, attr):
            try:
                delattr(lightmem, attr)
            except Exception:
                pass
    _close_quietly(lightmem)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def close_cached_memory_entries() -> None:
    with _cache_guard:
        entries = list(_entries.values())
        _entries.clear()
        _entry_locks.clear()
    for entry in entries:
        with entry.lock:
            if entry.method in _NATIVE_LIGHTMEM_METHODS:
                _close_native_lightmem(entry.layer)
            else:
                _close_toolkit_layer(entry.layer)


def _parse_full_context(prior_dialogue: str) -> tuple[list[dict[str, str]], str]:
    messages = parse_dialogue_to_messages(prior_dialogue)
    keys = [_message_key(message) for message in messages]
    return messages, _sha1_parts(keys)


def build_cached_baseline_context(
    prior_dialogue: str,
    user_question: str,
    eval_config: BaselineEvalConfig,
    *,
    sample_key: str | int | None = None,
) -> BaselineContext:
    if eval_config.method not in _TOOLKIT_METHODS and eval_config.method not in _NATIVE_LIGHTMEM_METHODS:
        from baselines.registry import build_baseline_context

        return build_baseline_context(
            prior_dialogue,
            user_question,
            eval_config,
            sample_key=sample_key,
        )

    messages, digest = _parse_full_context(prior_dialogue)
    entry = _get_entry(messages, eval_config, digest=digest)
    retrieved = (
        _retrieve_from_entry(entry, user_question, eval_config.top_k)
        if entry is not None
        else []
    )

    return BaselineContext(
        context_text=format_retrieved_memories(retrieved),
        retrieved_memories=jsonable_memories(retrieved),
        user_id=entry.user_id if entry is not None else "",
        save_dir=entry.save_dir if entry is not None else "",
        method=eval_config.method,
        top_k=eval_config.top_k,
    )


def patch_eval_memory_context(eval_module: Any) -> None:
    eval_module.build_baseline_context = build_cached_baseline_context
    import baselines as _baselines

    _baselines.build_baseline_context = build_cached_baseline_context


def patch_eval_args(eval_module: Any) -> None:
    patch_eval_memory_context(eval_module)
    original_parse_args = eval_module.parse_args

    def _parse_args_cached() -> Any:
        args = original_parse_args()
        if hasattr(args, "parallel_dual"):
            args.parallel_dual = False
        if hasattr(args, "workers"):
            args.workers = 1
        return args

    eval_module.parse_args = _parse_args_cached


def main() -> None:
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    base_eval.build_baseline_context = build_cached_baseline_context
    patch_eval_args(base_eval)
    try:
        base_eval.main()
    finally:
        close_cached_memory_entries()


if __name__ == "__main__":
    main()
