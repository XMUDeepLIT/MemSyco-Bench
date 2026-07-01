"""Summarize retrieval/answer quadrants from finalized baseline results."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_ROOT = REPO_ROOT / "output_data" / "baseline_opt_v2_runs_short"
OUTPUT_ROOT = REPO_ROOT / "output_data" / "extra_exp"

TASKS = {
    "consensus_judgment": ("interfering_preference_seed", "objective_correctness"),
    "evidence_memory_conflict_noisy": ("decisive_evidence_seed", "accuracy"),
    "memory_scope_overgeneralization": ("interfering_preference_seed", "accuracy"),
    "recommend": ("interfering_preference_seed", "answer_accuracy"),
    "recommend_change": ("updated_preference_seed", "update_pass"),
}

QUADRANTS = ("R+/A+", "R+/A-", "R-/A-", "R-/A+")


def method_of(data: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> str:
    for row in rows:
        method = (row.get("lightmem") or {}).get("method")
        if method:
            return str(method)
    name = path.name
    if name.startswith("raw_dialogue_"):
        return "RawDialogue"
    if name.startswith("no_memory_"):
        return "NoMemory"
    return "Unknown"


def model_family(model: str) -> str:
    normalized = model.lower().strip()
    if normalized in {"qwen/qwen3-8b", "qwen3-8b"}:
        return "qwen3-8b"
    if normalized in {"deepseek-v4-flash", "deepseek/deepseek-v4-flash"}:
        return "deepseek-v4-flash"
    return normalized


def answer_value(row: dict[str, Any], field: str) -> Any:
    # The post-hoc annotation copies the answer judgment. Prefer it so the two
    # axes come from the same analysis pass, with the original judge as fallback.
    seed = row.get("seed_retrieval_judge") or {}
    annotated = seed.get("answer_judge") or {}
    if field in annotated:
        return annotated[field]
    return (((row.get("with_memory") or {}).get("judge") or {}).get(field))


def classify(row: dict[str, Any], group_name: str, answer_field: str) -> tuple[str | None, str]:
    seed = row.get("seed_retrieval_judge")
    if not isinstance(seed, dict):
        return None, "missing_seed_retrieval_judge"
    if seed.get("judge_parse_ok") is not True:
        return None, "seed_judge_parse_failed"
    group = (seed.get("groups") or {}).get(group_name)
    if not isinstance(group, dict) or not isinstance(group.get("present"), bool):
        return None, "missing_required_seed_group"
    answer = answer_value(row, answer_field)
    if answer not in (0, 1, False, True):
        return None, "missing_answer_accuracy"
    retrieval_ok = bool(group["present"])
    answer_ok = bool(answer)
    if retrieval_ok and answer_ok:
        return "R+/A+", ""
    if retrieval_ok and not answer_ok:
        return "R+/A-", ""
    if not retrieval_ok and not answer_ok:
        return "R-/A-", ""
    return "R-/A+", ""


def blank_counts() -> dict[str, int]:
    return {key: 0 for key in QUADRANTS}


def output_row(labels: dict[str, Any], counts: dict[str, int], total: int, excluded: int) -> dict[str, Any]:
    valid = sum(counts.values())
    row = {**labels, "n_total": total, "n_valid": valid, "n_excluded": excluded}
    for key in QUADRANTS:
        row[f"{key}_count"] = counts[key]
        row[f"{key}_pct"] = round(100 * counts[key] / valid, 2) if valid else ""
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(records: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = tuple(record[name] for name in keys)
        bucket = buckets.setdefault(key, {"counts": blank_counts(), "total": 0, "excluded": 0, "files": set()})
        bucket["total"] += 1
        bucket["files"].add(record["file"])
        if record["quadrant"]:
            bucket["counts"][record["quadrant"]] += 1
        else:
            bucket["excluded"] += 1
    result = []
    for key, bucket in sorted(buckets.items()):
        labels = dict(zip(keys, key))
        labels["n_files"] = len(bucket["files"])
        result.append(output_row(labels, bucket["counts"], bucket["total"], bucket["excluded"]))
    return result


def wide_model_table(records: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    methods = ("A-MEM", "LightMemFull", "MemZero")
    selected = [r for r in records if r["model_family"] == family and r["method"] in methods]
    summarized = aggregate(selected, ("task", "method"))
    indexed = {(r["task"], r["method"]): r for r in summarized}
    rows: list[dict[str, Any]] = []
    for method in methods:
        out: dict[str, Any] = {"framework": method}
        for task in TASKS:
            current = indexed.get((task, method))
            out[f"{task}_n_total"] = current["n_total"] if current else 0
            out[f"{task}_n_valid"] = current["n_valid"] if current else 0
            out[f"{task}_n_excluded"] = current["n_excluded"] if current else 0
            for quadrant in QUADRANTS:
                out[f"{task}_{quadrant}_count"] = current[f"{quadrant}_count"] if current else 0
                out[f"{task}_{quadrant}_pct"] = current[f"{quadrant}_pct"] if current else ""
        rows.append(out)
    return rows


def markdown_model_table(rows: list[dict[str, Any]], family: str) -> str:
    def cell(row: dict[str, Any], task: str) -> str:
        valid = int(row[f"{task}_n_valid"])
        total = int(row[f"{task}_n_total"])
        excluded = int(row[f"{task}_n_excluded"])
        if valid == 0:
            return f"N/A (valid 0/{total}; excluded {excluded})"
        values = []
        for quadrant in QUADRANTS:
            count = row[f"{task}_{quadrant}_count"]
            pct = row[f"{task}_{quadrant}_pct"]
            values.append(f"{quadrant} {count} ({pct}%)")
        return "<br>".join(values) + f"<br>valid {valid}/{total}"

    lines = [
        f"# {family}: Retrieval / Answer 四象限",
        "",
        "百分比以每个任务、框架的 `n_valid` 为分母。`N/A` 表示当前 final 文件没有可用的检索判定。",
        "",
        "| Framework | " + " | ".join(TASKS) + " |",
        "|---|" + "---|" * len(TASKS),
    ]
    for row in rows:
        lines.append("| " + str(row["framework"]) + " | " + " | ".join(cell(row, task) for task in TASKS) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sample_records: list[dict[str, Any]] = []
    exclusion_reasons: dict[str, int] = defaultdict(int)

    for task, (group_name, answer_field) in TASKS.items():
        for path in sorted((INPUT_ROOT / task).glob("*final*.json")):
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            rows = data.get("results") or []
            if not isinstance(rows, list):
                continue
            method = method_of(data, rows, path)
            model = str(data.get("model") or "Unknown")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                quadrant, reason = classify(row, group_name, answer_field)
                if reason:
                    exclusion_reasons[reason] += 1
                sample_records.append(
                    {
                        "task": task,
                        "method": method,
                        "model": model,
                        "model_family": model_family(model),
                        "file": path.name,
                        "quadrant": quadrant,
                    }
                )

    by_file = aggregate(sample_records, ("task", "method", "model", "file"))
    by_config = aggregate(sample_records, ("task", "method", "model_family"))
    by_task = aggregate(sample_records, ("task",))
    write_csv(OUTPUT_ROOT / "retrieval_answer_by_file.csv", by_file)
    write_csv(OUTPUT_ROOT / "retrieval_answer_by_category_method_model.csv", by_config)
    write_csv(OUTPUT_ROOT / "retrieval_answer_by_category.csv", by_task)

    for family, stem in (("qwen3-8b", "qwen3_8b"), ("deepseek-v4-flash", "deepseek_v4_flash")):
        wide_rows = wide_model_table(sample_records, family)
        write_csv(OUTPUT_ROOT / f"{stem}_five_tasks.csv", wide_rows)
        (OUTPUT_ROOT / f"{stem}_five_tasks.md").write_text(
            markdown_model_table(wide_rows, family), encoding="utf-8"
        )

    metadata = {
        "input_root": str(INPUT_ROOT.relative_to(REPO_ROOT)),
        "selected_files": "*final*.json in the five requested task directories",
        "task_rules": {
            task: {"retrieval_group": group, "answer_field": answer}
            for task, (group, answer) in TASKS.items()
        },
        "model_family_aliases": {
            "qwen3-8b": ["qwen/qwen3-8b", "qwen3-8b"],
            "deepseek-v4-flash": ["deepseek-v4-flash", "deepseek/deepseek-v4-flash"],
        },
        "quadrants": {
            "R+/A+": "检索到且答对",
            "R+/A-": "检索到但答错（generation/calibration failure）",
            "R-/A-": "未检索到且答错",
            "R-/A+": "未检索到但答对",
        },
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "notes": [
            "Percentages use n_valid as denominator.",
            "Missing or unparsable retrieval/answer judgments are excluded, not treated as failures.",
            "Counts are pooled at sample level; files with different sample sizes therefore have proportional weight.",
        ],
    }
    (OUTPUT_ROOT / "retrieval_answer_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
