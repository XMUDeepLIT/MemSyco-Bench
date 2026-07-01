#!/usr/bin/env python3
"""Inspect every memory stored for one evaluated query.

This script understands the persistent formats used by MemZero, A-MEM and
LightMemFull.  Token counts deliberately exclude embedding vectors:

* content_tokens: text that is (approximately) exposed to retrieval/generation
* record_tokens: the semantic JSON record, including useful metadata

Pickle files must only be inspected when they were generated locally and are
trusted.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sqlite3
import statistics
from pathlib import Path
from typing import Any, Iterable


METHOD_ALIASES = {
    "a-mem": "amem",
    "amem": "amem",
    "mem0": "memzero",
    "memzero": "memzero",
    "lightmem": "lightmem_full",
    "lightmemfull": "lightmem_full",
    "lightmem_full": "lightmem_full",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_text(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected an object in {path}")
    return data


def _row_id(row: dict[str, Any], index: int) -> str:
    for key in ("query_id", "id", "source_query_id", "sample_id"):
        if row.get(key) is not None:
            return str(row[key])
    return str(index)


def _find_result(path: Path, query_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _load_json(path)
    rows = data.get("results", [])
    for index, row in enumerate(rows):
        if isinstance(row, dict) and _row_id(row, index) == query_id:
            return data, row
    raise KeyError(f"Query {query_id!r} not found in {path}")


def _canonical_method(data: dict[str, Any], row: dict[str, Any]) -> str:
    details = row.get("lightmem") if isinstance(row.get("lightmem"), dict) else {}
    raw = details.get("method") or data.get("memory_method") or data.get("method")
    key = str(raw or "").lower().replace("-", "_").replace(" ", "")
    key = key.replace("_", "") if key not in METHOD_ALIASES else key
    method = METHOD_ALIASES.get(key)
    if method is None:
        raise ValueError(f"Unsupported or missing memory method: {raw!r}")
    return method


def _store_location(row: dict[str, Any]) -> tuple[Path, str]:
    details = row.get("lightmem")
    if not isinstance(details, dict):
        raise ValueError("Result row has no 'lightmem' store metadata")
    save_dir = details.get("save_dir")
    user_id = details.get("user_id")
    if not save_dir or not user_id:
        raise ValueError("Result row is missing lightmem.save_dir or lightmem.user_id")
    return Path(save_dir), str(user_id)


def _find_pickle(save_dir: Path, user_id: str) -> Path:
    preferred = save_dir / f"{user_id}.pkl"
    if preferred.is_file():
        return preferred
    candidates = sorted(save_dir.glob("*.pkl"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not identify memory pickle under {save_dir}")


def _semantic_amem_record(note: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _jsonable(note.get(key))
        for key in (
            "id",
            "content",
            "context",
            "keywords",
            "tags",
            "category",
            "timestamp",
            "last_accessed",
            "retrieval_count",
            "links",
            "evolution_history",
        )
        if key in note
    }


def _amem_content(note: dict[str, Any]) -> str:
    return "\n".join(
        (
            f"memory content: {note.get('content', '')}",
            f"memory context: {note.get('context', '')}",
            f"memory keywords: {note.get('keywords', [])}",
            f"memory tags: {note.get('tags', [])}",
            f"talk start time: {note.get('timestamp', '')}",
        )
    )


def _read_amem(save_dir: Path, user_id: str) -> list[dict[str, Any]]:
    with _find_pickle(save_dir, user_id).open("rb") as handle:
        state = pickle.load(handle)  # Trusted, locally generated evaluation data.
    notes = state.get("notes", []) if isinstance(state, dict) else []
    return [
        {
            "source": "main",
            "id": str(note.get("id", index)),
            "content": _amem_content(note),
            "record": _semantic_amem_record(note),
        }
        for index, note in enumerate(notes)
        if isinstance(note, dict)
    ]


def _memzero_content(memory: dict[str, Any]) -> str:
    metadata = memory.get("metadata") if isinstance(memory.get("metadata"), dict) else {}
    timestamp = metadata.get("timestamp", "")
    return f"Memory: {memory.get('memory', '')}\nTime: {timestamp}"


def _read_memzero(save_dir: Path, user_id: str) -> list[dict[str, Any]]:
    with _find_pickle(save_dir, user_id).open("rb") as handle:
        state = pickle.load(handle)  # Trusted, locally generated evaluation data.
    if isinstance(state, dict):
        state = state.get("memories", state.get("results", []))
    if not isinstance(state, list):
        raise ValueError(f"Unexpected MemZero pickle shape in {save_dir}")
    records = []
    for index, memory in enumerate(state):
        if not isinstance(memory, dict):
            continue
        semantic = {k: _jsonable(v) for k, v in memory.items() if k not in {"embedding", "vector"}}
        records.append(
            {
                "source": "main",
                "id": str(memory.get("id", index)),
                "content": _memzero_content(memory),
                "record": semantic,
            }
        )
    return records


class _QdrantObject:
    """Placeholder for Qdrant's pydantic classes when qdrant-client is absent."""

    def __setstate__(self, state: Any) -> None:
        self._state = state


class _QdrantUnpickler(pickle.Unpickler):
    _classes: dict[tuple[str, str], type] = {}

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("qdrant_client."):
            key = (module, name)
            if key not in self._classes:
                self._classes[key] = type(name, (_QdrantObject,), {})
            return self._classes[key]
        return super().find_class(module, name)


def _unpack_qdrant(value: Any) -> Any:
    if isinstance(value, _QdrantObject):
        state = getattr(value, "_state", {})
        if isinstance(state, dict) and isinstance(state.get("__dict__"), dict):
            state = state["__dict__"]
        return _unpack_qdrant(state)
    if isinstance(value, dict):
        return {str(k): _unpack_qdrant(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_unpack_qdrant(v) for v in value]
    return value


def _read_qdrant_sqlite(path: Path, source: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        blobs = connection.execute("SELECT point FROM points ORDER BY id").fetchall()
    records = []
    for index, (blob,) in enumerate(blobs):
        point = _unpack_qdrant(_QdrantUnpickler(io.BytesIO(blob)).load())
        if not isinstance(point, dict):
            continue
        payload = point.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        semantic = {
            k: _jsonable(v)
            for k, v in payload.items()
            if k not in {"embedding", "vector", "update_queue"}
        }
        content = " ".join(
            str(v).strip()
            for v in (
                payload.get("time_stamp", ""),
                payload.get("weekday", ""),
                payload.get("memory", ""),
            )
            if str(v).strip()
        )
        records.append(
            {
                "source": source,
                "id": str(point.get("id", index)),
                "content": content,
                "record": semantic,
            }
        )
    return records


def _read_lightmem(save_dir: Path, user_id: str) -> list[dict[str, Any]]:
    main = save_dir / "collection" / user_id / "storage.sqlite"
    summary = save_dir / "summaries" / "collection" / f"{user_id}_summary" / "storage.sqlite"
    records = _read_qdrant_sqlite(main, "main")
    records.extend(_read_qdrant_sqlite(summary, "summary"))
    if not records:
        raise FileNotFoundError(f"No LightMem Qdrant points found under {save_dir}")
    return records


def _read_store(method: str, save_dir: Path, user_id: str) -> list[dict[str, Any]]:
    if method == "amem":
        return _read_amem(save_dir, user_id)
    if method == "memzero":
        return _read_memzero(save_dir, user_id)
    if method == "lightmem_full":
        return _read_lightmem(save_dir, user_id)
    raise AssertionError(method)


def _stats(values: Iterable[int]) -> dict[str, Any]:
    values = list(values)
    if not values:
        return {"count": 0, "total": 0, "min": 0, "median": 0, "mean": 0, "max": 0}
    return {
        "count": len(values),
        "total": sum(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 2),
        "max": max(values),
    }


def _question(row: dict[str, Any]) -> str:
    for key in ("objective_question", "question", "query", "input"):
        if row.get(key):
            return str(row[key])
    return ""


def _print_report(report: dict[str, Any], preview_chars: int, show_all: bool) -> None:
    print(f"query_id: {report['query_id']}")
    if report.get("question"):
        print(f"question: {report['question']}")
    print(f"token_encoding: {report['token_encoding']}")
    for store in report["stores"]:
        stats = store["content_token_stats"]
        rstats = store["record_token_stats"]
        print(
            f"\n[{store['method']}] {store['model']} | {store['user_id']}\n"
            f"store: {store['save_dir']}\n"
            f"memories: {stats['count']} "
            f"content_tokens(total/mean/median/min/max)="
            f"{stats['total']}/{stats['mean']}/{stats['median']}/{stats['min']}/{stats['max']} "
            f"record_tokens(total/mean/median/min/max)="
            f"{rstats['total']}/{rstats['mean']}/{rstats['median']}/{rstats['min']}/{rstats['max']}"
        )
        for index, memory in enumerate(store["memories"], 1):
            content = memory["content"]
            if not show_all and len(content) > preview_chars:
                content = content[:preview_chars].rstrip() + "…"
            print(
                f"  {index:03d} source={memory['source']} id={memory['id']} "
                f"content_tokens={memory['content_tokens']} "
                f"record_tokens={memory['record_tokens']}\n"
                f"      {content.replace(chr(10), chr(10) + '      ')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        type=Path,
        required=True,
        help="Evaluation result JSON; repeat once per method",
    )
    parser.add_argument("--query-id", required=True, help="Query/sample ID to inspect")
    parser.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding")
    parser.add_argument("--export-json", type=Path, help="Write the complete structured report")
    parser.add_argument("--preview-chars", type=int, default=300)
    parser.add_argument("--show-all", action="store_true", help="Print full memory text")
    args = parser.parse_args()

    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit("tiktoken is required to count tokens") from exc
    encoding = tiktoken.get_encoding(args.encoding)

    report: dict[str, Any] = {
        "query_id": args.query_id,
        "question": "",
        "token_encoding": args.encoding,
        "token_counting": {
            "content_tokens": "retrieval-facing text",
            "record_tokens": "semantic JSON metadata, excluding vectors and LightMem update_queue",
        },
        "stores": [],
    }
    seen_methods: set[str] = set()
    for result_path in args.result:
        data, row = _find_result(result_path, args.query_id)
        method = _canonical_method(data, row)
        if method in seen_methods:
            raise ValueError(f"Duplicate result method {method!r}: {result_path}")
        seen_methods.add(method)
        save_dir, user_id = _store_location(row)
        memories = _read_store(method, save_dir, user_id)
        for memory in memories:
            memory["content_tokens"] = len(encoding.encode(memory["content"]))
            memory["record_tokens"] = len(encoding.encode(_json_text(memory["record"])))
        store = {
            "method": method,
            "model": str(data.get("model", "")),
            "result_path": str(result_path),
            "save_dir": str(save_dir),
            "user_id": user_id,
            "content_token_stats": _stats(m["content_tokens"] for m in memories),
            "record_token_stats": _stats(m["record_tokens"] for m in memories),
            "memories": memories,
        }
        report["stores"].append(store)
        report["question"] = report["question"] or _question(row)

    _print_report(report, args.preview_chars, args.show_all)
    if args.export_json:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        with args.export_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
