"""Summarize evidence-conflict metrics by retrieved seed combination."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_ROOT / "output_data" / "baseline_opt_v2_runs_short" / "evidence_memory_conflict_noisy"
OUTPUT_DIR = REPO_ROOT / "output_data" / "extra_exp" / "scenario"

FAMILIES = {
    "qwen3-8b": {"qwen/qwen3-8b", "qwen3-8b"},
    "deepseek-v4-flash": {"deepseek-v4-flash", "deepseek/deepseek-v4-flash"},
}
PREFERRED_METHOD_ORDER = ("A-MEM", "LightMemFull", "MemZero")
CONDITIONS = {
    "fact_only": (True, False),
    "preference_only": (False, True),
    "both": (True, True),
}


def method_of(rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        method = (row.get("lightmem") or {}).get("method")
        if method:
            return str(method)
    return None


def family_of(model: Any) -> str | None:
    normalized = str(model or "").lower().strip()
    for family, aliases in FAMILIES.items():
        if normalized in aliases:
            return family
    return None


def binary(value: Any) -> bool:
    return value in (0, 1, False, True)


def empty_bucket() -> dict[str, int]:
    return {"n": 0, "acc_sum": 0, "misled_sum": 0}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: list[dict[str, Any]], family: str) -> str:
    def cell(row: dict[str, Any], condition: str) -> str:
        n = int(row[f"{condition}_n"])
        if not n:
            return f"Acc N/A<br>Misled N/A<br>Share {row[f'{condition}_share_pct']}%<br>n=0"
        return (
            f"Acc {row[f'{condition}_acc_pct']}%<br>"
            f"Misled {row[f'{condition}_misled_pct']}%<br>"
            f"Share {row[f'{condition}_share_pct']}%<br>n={n}"
        )

    lines = [
        f"# {family}: evidence_memory_conflict_noisy",
        "",
        "事实 = `decisive_evidence_seed`；偏好 = `interfering_preference_seed`。",
        "Acc 与 Misled Rate 使用同时具有有效检索判定、accuracy 和 "
        "misled_by_conflicting_memory 的样本。",
        "",
        "| Framework | 仅事实被检索 | 仅偏好被检索 | 事实与偏好均被检索 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['framework']} | {cell(row, 'fact_only')} | "
            f"{cell(row, 'preference_only')} | {cell(row, 'both')} |"
        )
    lines.extend(
        [
            "",
            "## 如何从现有结果复现",
            "",
            "输入文件来自：",
            "",
            "```text",
            "output_data/baseline_opt_v2_runs_short/evidence_memory_conflict_noisy/*final*.json",
            "```",
            "",
            "脚本会按顶层 `model` 筛选模型系列，并从每个结果项的 "
            "`lightmem.method` 自动识别 memory framework。因此加入新框架时无需修改框架列表；"
            "只需保证其 final 文件具有相同字段结构。模型名称映射为：",
            "",
            "- Qwen3-8B：`qwen/qwen3-8b` 或 `qwen3-8b`",
            "- DeepSeek：`deepseek-v4-flash` 或 `deepseek/deepseek-v4-flash`",
            "",
            "对 `results` 中的每个样本读取：",
            "",
            "```text",
            "seed = result.seed_retrieval_judge",
            "fact_retrieved = seed.groups.decisive_evidence_seed.present",
            "preference_retrieved = seed.groups.interfering_preference_seed.present",
            "acc = seed.answer_judge.accuracy",
            "misled = seed.answer_judge.misled_by_conflicting_memory",
            "```",
            "",
            "若 `seed.answer_judge` 中缺少答案指标，则回退读取：",
            "",
            "```text",
            "result.with_memory.judge.accuracy",
            "result.with_memory.judge.misled_by_conflicting_memory",
            "```",
            "",
            "仅保留满足以下条件的样本：",
            "",
            "1. `seed_retrieval_judge.judge_parse_ok == true`；",
            "2. 两个 seed group 的 `present` 均为布尔值；",
            "3. `accuracy` 和 `misled_by_conflicting_memory` 均为 0/1。",
            "",
            "然后根据两个检索布尔值分组：",
            "",
            "| 分组 | `fact_retrieved` | `preference_retrieved` |",
            "|---|---:|---:|",
            "| 仅事实 | true | false |",
            "| 仅偏好 | false | true |",
            "| 两者均被检索 | true | true |",
            "",
            "`false / false` 属于“两者均未检索”，不进入本表。每组使用同一批有效样本计算：",
            "",
            "```text",
            "Acc (%)         = sum(accuracy) / n * 100",
            "Misled Rate (%) = sum(misled_by_conflicting_memory) / n * 100",
            "```",
            "",
            "其中 `n` 是该框架、该检索组合下同时具有两个有效答案指标的样本数。"
            "缺失判定、judge 解析失败或答案指标缺失的样本记入 CSV 的 `n_excluded`，"
            "不会被当作 0。",
            "",
            "组合占比以包括“两者均未检索”在内的全部有效样本作为分母：",
            "",
            "```text",
            "Share (%) = 该检索组合的 n / (n_total - n_excluded) * 100",
            "```",
            "",
            "由于本表不展示 `false / false`，存在“两者均未检索”样本时，表中三项 Share "
            "不会加总到 100%。",
            "",
            "在仓库根目录运行以下命令即可重新生成两个 Markdown、CSV 和 metadata：",
            "",
            "```bash",
            "python evaluation/summarize_evidence_retrieval_combinations.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    buckets: dict[tuple[str, str, str], dict[str, int]] = defaultdict(empty_bucket)
    diagnostics: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"n_total": 0, "n_excluded": 0}
    )
    methods_by_family: dict[str, set[str]] = defaultdict(set)

    for path in sorted(INPUT_DIR.glob("*final*.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        family = family_of(data.get("model"))
        rows = data.get("results") or []
        if not family or not isinstance(rows, list):
            continue
        method = method_of(rows)
        if not method:
            continue
        methods_by_family[family].add(method)
        for row in rows:
            if not isinstance(row, dict):
                continue
            diagnostics[(family, method)]["n_total"] += 1
            seed = row.get("seed_retrieval_judge")
            groups = (seed or {}).get("groups") or {} if isinstance(seed, dict) else {}
            fact = groups.get("decisive_evidence_seed")
            preference = groups.get("interfering_preference_seed")
            answer = (seed or {}).get("answer_judge") or {} if isinstance(seed, dict) else {}
            judge = ((row.get("with_memory") or {}).get("judge") or {})
            acc = answer.get("accuracy", judge.get("accuracy"))
            misled = answer.get(
                "misled_by_conflicting_memory", judge.get("misled_by_conflicting_memory")
            )
            valid = (
                isinstance(seed, dict)
                and seed.get("judge_parse_ok") is True
                and isinstance(fact, dict)
                and isinstance(fact.get("present"), bool)
                and isinstance(preference, dict)
                and isinstance(preference.get("present"), bool)
                and binary(acc)
                and binary(misled)
            )
            if not valid:
                diagnostics[(family, method)]["n_excluded"] += 1
                continue
            state = (bool(fact["present"]), bool(preference["present"]))
            condition = next((name for name, expected in CONDITIONS.items() if state == expected), None)
            if condition is None:  # Neither retrieved; outside the requested three conditions.
                continue
            bucket = buckets[(family, method, condition)]
            bucket["n"] += 1
            bucket["acc_sum"] += int(acc)
            bucket["misled_sum"] += int(misled)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {"diagnostics": {}}
    for family, stem in (("qwen3-8b", "qwen3_8b"), ("deepseek-v4-flash", "deepseek_v4_flash")):
        output_rows: list[dict[str, Any]] = []
        preferred = {name: index for index, name in enumerate(PREFERRED_METHOD_ORDER)}
        methods = sorted(
            methods_by_family[family], key=lambda name: (preferred.get(name, len(preferred)), name.lower())
        )
        for method in methods:
            row: dict[str, Any] = {"framework": method}
            diag = diagnostics[(family, method)]
            row.update(diag)
            metadata["diagnostics"][f"{family}/{method}"] = diag
            for condition in CONDITIONS:
                bucket = buckets[(family, method, condition)]
                n = bucket["n"]
                valid_total = diag["n_total"] - diag["n_excluded"]
                row[f"{condition}_n"] = n
                row[f"{condition}_share_pct"] = round(100 * n / valid_total, 2) if valid_total else ""
                row[f"{condition}_acc_count"] = bucket["acc_sum"]
                row[f"{condition}_acc_pct"] = round(100 * bucket["acc_sum"] / n, 2) if n else ""
                row[f"{condition}_misled_count"] = bucket["misled_sum"]
                row[f"{condition}_misled_pct"] = round(100 * bucket["misled_sum"] / n, 2) if n else ""
            output_rows.append(row)
        write_csv(OUTPUT_DIR / f"{stem}_evidence_retrieval_combinations.csv", output_rows)
        (OUTPUT_DIR / f"{stem}_evidence_retrieval_combinations.md").write_text(
            markdown(output_rows, family), encoding="utf-8"
        )

    metadata.update(
        {
            "task": "evidence_memory_conflict_noisy",
            "conditions": {
                "fact_only": "decisive evidence present, interfering preference absent",
                "preference_only": "decisive evidence absent, interfering preference present",
                "both": "both seed groups present",
            },
            "notes": [
                "Neither-retrieved samples are outside the requested table.",
                "These are observational strata of natural retrieval outputs, not randomized interventions.",
            ],
        }
    )
    (OUTPUT_DIR / "evidence_retrieval_combinations_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
