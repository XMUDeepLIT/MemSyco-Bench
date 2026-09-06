"""Paper metric names attached to generated evaluation results."""

from __future__ import annotations

from typing import Any

PAPER_SPECS: dict[str, dict[str, Any]] = {
    "objective_fact_judgment": {
        "paper_task": "Objective Fact Judgment",
        "acc": {
            "name": "Acc.",
            "field": "objective_correctness",
            "avg_key": "objective_correctness_avg",
            "direction": "higher_is_better",
        },
        "secondary": {
            "name": "Syco. Rate",
            "alias": "syco_rate",
            "field": "preference_answer_selected",
            "avg_key": "preference_answer_selected_avg",
            "direction": "lower_is_better",
        },
    },
    "contextual_scope_control": {
        "paper_task": "Contextual Scope Limits",
        "acc": {
            "name": "Acc.",
            "field": "accuracy",
            "avg_key": "accuracy_avg",
            "direction": "higher_is_better",
        },
        "secondary": {
            "name": "Syco. Rate",
            "alias": "syco_rate",
            "field": "incorrectly_used_preference",
            "avg_key": "incorrectly_used_preference_avg",
            "direction": "lower_is_better",
        },
    },
    "memory_evidence_conflict": {
        "paper_task": "Preference-Fact Conflict",
        "acc": {
            "name": "Acc.",
            "field": "accuracy",
            "avg_key": "accuracy_avg",
            "direction": "higher_is_better",
        },
        "secondary": {
            "name": "Syco. Rate",
            "alias": "syco_rate",
            "field": "misled_by_conflicting_memory",
            "avg_key": "misled_by_conflicting_memory_avg",
            "direction": "lower_is_better",
        },
    },
    "personalized_memory_use": {
        "paper_task": "Personalized Recommendation",
        "acc": {
            "name": "Acc.",
            "field": "answer_accuracy",
            "avg_key": "answer_accuracy_avg",
            "direction": "higher_is_better",
        },
        "secondary": {
            "name": "Correct Pref. Use",
            "alias": "correct_pref_use",
            "field": "preference_used",
            "avg_key": "preference_used_avg",
            "direction": "higher_is_better",
        },
    },
    "valid_memory_selection": {
        "paper_task": "Preference Change",
        "acc": {
            "name": "Acc.",
            "field": "uses_latest_preference",
            "avg_key": "uses_latest_preference_avg",
            "direction": "higher_is_better",
        },
        "secondary": {
            "name": "Outdated Pref.",
            "alias": "outdated_pref",
            "field": "outdated_preference_contamination",
            "avg_key": "outdated_preference_contamination_avg",
            "direction": "lower_is_better",
        },
    },
}

_SETTING_KEYS = ("no_memory", "with_memory")


def attach_paper_metrics(task: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Copy paper aliases onto setting blocks and return a paper_metrics block."""
    spec = PAPER_SPECS[task]
    acc = spec["acc"]
    secondary = spec["secondary"]
    paper: dict[str, Any] = {
        "paper_task": spec["paper_task"],
        "code_task": task,
        "metric_map": {
            acc["name"]: {
                "field": acc["field"],
                "direction": acc["direction"],
            },
            secondary["name"]: {
                "field": secondary["field"],
                "direction": secondary["direction"],
            },
        },
    }
    for setting in _SETTING_KEYS:
        block = metrics.get(setting)
        if not isinstance(block, dict):
            continue
        if acc["avg_key"] not in block and secondary["avg_key"] not in block:
            continue
        n_used = block.get("n_judged", block.get("n_scored"))
        if n_used == 0:
            continue
        acc_val = block.get(acc["avg_key"])
        secondary_val = block.get(secondary["avg_key"])
        block["acc"] = acc_val
        block[secondary["alias"]] = secondary_val
        paper[setting] = {
            acc["name"]: acc_val,
            secondary["name"]: secondary_val,
        }
    return paper


def paper_secondary_summary(paper_metrics: dict[str, Any], setting: str) -> str:
    values = paper_metrics.get(setting) or {}
    code_task = paper_metrics.get("code_task")
    if not values or code_task not in PAPER_SPECS:
        return ""
    spec = PAPER_SPECS[code_task]
    name = spec["secondary"]["name"]
    value = values.get(name)
    if value is None:
        return ""
    arrow = "↓" if spec["secondary"]["direction"] == "lower_is_better" else "↑"
    return f", {name}={value:.4f} ({arrow})"
