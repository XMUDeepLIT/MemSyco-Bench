"""Unified command-line entry point for the five MemSyco-Bench tasks."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TASK_MODULES = {
    "personalized_memory_use": "task_personalized_memory_use",
    "valid_memory_selection": "task_valid_memory_selection",
    "memory_evidence_conflict": "task_memory_evidence_conflict",
    "contextual_scope_control": "task_contextual_scope_control",
    "objective_fact_judgment": "task_objective_fact_judgment",
}


def _usage() -> str:
    tasks = "\n".join(f"  {name}" for name in TASK_MODULES)
    return f"""Usage: python evaluation/run_task.py TASK [--optimized] [TASK OPTIONS]\n\nTasks:\n{tasks}\n\nUse TASK --help to see task-specific options.\n"""


def _load_task(task: str) -> tuple[str, Any]:
    try:
        module_name = TASK_MODULES[task]
    except KeyError as exc:
        choices = ", ".join(TASK_MODULES)
        raise SystemExit(f"Unknown task {task!r}. Choose one of: {choices}") from exc
    return task, importlib.import_module(module_name)


def _run_optimized(module: Any) -> None:
    from _optimized_memory import (
        close_cached_memory_entries,
        patch_eval_args,
    )

    target = getattr(module, "base", module)
    patch_eval_args(target)
    try:
        module.main()
    finally:
        close_cached_memory_entries()


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return

    task = sys.argv[1]
    remaining = sys.argv[2:]
    optimized = "--optimized" in remaining
    remaining = [arg for arg in remaining if arg != "--optimized"]
    _, module = _load_task(task)
    sys.argv = [sys.argv[0], *remaining]

    if optimized:
        _run_optimized(module)
    else:
        module.main()


if __name__ == "__main__":
    main()
