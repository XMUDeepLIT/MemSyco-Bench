"""Mem0 + MemGate admission filter.

Uses the official MemGate checkpoint (arXiv:2606.06054) as a query-conditioned
gate over MemZero retrieval candidates. The checkpoint expects 384-d
sentence-transformer embeddings (all-MiniLM-L6-v2), not the MemZero store's
bge-m3 vectors.

Inference code is ``memgate_big.py``. The weight file is not in git.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ..base import BaselineEvalConfig, REPO_ROOT

_LOCK = threading.Lock()
_DEFENDER: Any = None
_DEFENDER_KEY: tuple[Any, ...] | None = None

DEFAULT_CHECKPOINT = REPO_ROOT / "output_data" / "checkpoints" / "Memgate.pt"
DEFAULT_THRESHOLD = 0.07


def _option(eval_config: BaselineEvalConfig, key: str, default: Any = None) -> Any:
    options = eval_config.options or {}
    if key not in options or options[key] is None:
        return default
    return options[key]


def _load_defender(eval_config: BaselineEvalConfig) -> Any:
    global _DEFENDER, _DEFENDER_KEY
    with _LOCK:
        raw_checkpoint = _option(eval_config, "checkpoint", DEFAULT_CHECKPOINT)
        checkpoint = Path(raw_checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = REPO_ROOT / checkpoint
        device = str(_option(eval_config, "device", "cpu") or "cpu").strip() or "cpu"
        threshold = float(_option(eval_config, "threshold", DEFAULT_THRESHOLD))
        key = (str(checkpoint), device, threshold)
        if _DEFENDER is not None and _DEFENDER_KEY == key:
            return _DEFENDER
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"MemGate checkpoint not found: {checkpoint}. "
                "Run ./scripts/fetch_memgate_checkpoint.sh"
            )
        from .memgate_big import BIGDefender

        defender = BIGDefender(
            embed_encoder=None,
            embed_dim=384,
            intent_dim=256,
            threshold=threshold,
            device=device,
            use_memgate_single_sided=True,
        )
        defender.load(str(checkpoint))
        defender.threshold = threshold
        # Do not inject MemGate's extra safety/objective strings into benchmark context.
        defender.safety_memory = None
        defender.objective_memory = None
        defender.eval()
        print(
            f"[MemGate] loaded {checkpoint} threshold={threshold} device={defender.device}",
            flush=True,
        )
        _DEFENDER = defender
        _DEFENDER_KEY = key
        return _DEFENDER


def filter_memories(
    query: str,
    memories: list[dict[str, Any]],
    eval_config: BaselineEvalConfig,
) -> list[dict[str, Any]]:
    if not memories:
        return []
    defender = _load_defender(eval_config)
    texts = [
        str(mem.get("used_content") or mem.get("content") or "").strip()
        for mem in memories
    ]
    scores = defender.score_candidates(query, texts)
    threshold = float(defender.threshold)
    out: list[dict[str, Any]] = []
    for mem, score in zip(memories, scores):
        if score <= threshold:
            continue
        meta = dict(mem.get("metadata") or {})
        meta["control"] = {
            "method": "MemZero+MemGate",
            "score": score,
            "threshold": threshold,
        }
        item = dict(mem)
        item["metadata"] = meta
        out.append(item)
    return out
