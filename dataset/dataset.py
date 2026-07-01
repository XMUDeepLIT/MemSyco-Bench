"""Small, dependency-free loader for the MemSyco-Bench release data."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

TASK_DISPLAY_NAMES = {
    "objective_fact_judgment": "Objective Fact Judgment",
    "contextual_scope_control": "Contextual Scope Control",
    "memory_evidence_conflict": "Memory-Evidence Conflict",
    "valid_memory_selection": "Valid Memory Selection",
    "personalized_memory_use": "Personalized Memory Use",
}


def load_manifest(data_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def task_names(data_dir: str | Path | None = None) -> tuple[str, ...]:
    return tuple(load_manifest(data_dir)["tasks"])


def iter_dataset(
    task: str | None = None,
    *,
    data_dir: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    manifest = load_manifest(root)
    tasks = manifest["tasks"]
    selected = tuple(tasks) if task is None else (task,)
    for name in selected:
        if name not in tasks:
            choices = ", ".join(tasks)
            raise ValueError(f"Unknown task {name!r}. Choose one of: {choices}")
        with (root / tasks[name]["file"]).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_dataset(
    task: str | None = None,
    *,
    data_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    return list(iter_dataset(task, data_dir=data_dir))
