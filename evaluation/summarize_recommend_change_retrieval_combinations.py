"""Summarize preference-update accuracy by old/new retrieval combination."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_ROOT / "output_data" / "baseline_opt_v2_runs_short" / "recommend_change"
OUTPUT_DIR = REPO_ROOT / "output_data" / "extra_exp" / "scenario"

FAMILIES = {
    "qwen3-8b": {"qwen/qwen3-8b", "qwen3-8b"},
    "deepseek-v4-flash": {"deepseek-v4-flash", "deepseek/deepseek-v4-flash"},
}
PREFERRED_METHOD_ORDER = ("A-MEM", "LightMemFull", "MemZero")
CONDITIONS = {
    "old_only": (True, False),
    "new_only": (False, True),
    "both": (True, True),
    "neither": (False, False),
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: list[dict[str, Any]], family: str) -> str:
    def cell(row: dict[str, Any], condition: str) -> str:
        n = int(row[f"{condition}_n"])
        share = row[f"{condition}_share_pct"]
        if not n:
            return f"Acc N/A<br>Share {share}%<br>n=0"
        return f"Acc {row[f'{condition}_acc_pct']}%<br>Share {share}%<br>n={n}"

    lines = [
        f"# {family}: recommend_change",
        "",
        "旧偏好 = `old_preference_seed`；新偏好 = `updated_preference_seed`；"
        "Acc 使用任务的 `update_pass`。",
        "",
        "| Framework | 仅旧偏好 | 仅新偏好 | 新旧偏好都有 | 新旧偏好都无 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['framework']} | {cell(row, 'old_only')} | {cell(row, 'new_only')} | "
            f"{cell(row, 'both')} | {cell(row, 'neither')} |"
        )
    lines.extend(
        [
            "",
            "## 如何从现有结果复现",
            "",
            "输入文件：",
            "",
            "```text",
            "output_data/baseline_opt_v2_runs_short/recommend_change/*final*.json",
            "```",
            "",
            "脚本按顶层 `model` 筛选 Qwen3-8B 或 DeepSeek，并通过每个结果项的 "
            "`lightmem.method` 自动识别 memory framework；新增框架无需修改框架名单。",
            "模型别名与 evidence 统计相同。",
            "",
            "逐样本读取：",
            "",
            "```text",
            "seed = result.seed_retrieval_judge",
            "old_retrieved = seed.groups.old_preference_seed.present",
            "new_retrieved = seed.groups.updated_preference_seed.present",
            "acc = seed.answer_judge.update_pass",
            "```",
            "",
            "如果 annotation 中没有 `update_pass`，回退读取 "
            "`result.with_memory.judge.update_pass`。仅保留：",
            "",
            "1. `seed.judge_parse_ok == true`；",
            "2. old/new 两个 `present` 均为布尔值；",
            "3. `update_pass` 为 0/1。",
            "",
            "分组规则：",
            "",
            "| 分组 | `old_retrieved` | `new_retrieved` |",
            "|---|---:|---:|",
            "| 仅旧偏好 | true | false |",
            "| 仅新偏好 | false | true |",
            "| 新旧偏好都有 | true | true |",
            "| 新旧偏好都无 | false | false |",
            "",
            "计算公式：",
            "",
            "```text",
            "Acc (%)   = sum(update_pass) / 该组合 n * 100",
            "Share (%) = 该组合 n / 四种组合的有效样本总数 * 100",
            "```",
            "",
            "四项 Share 在舍入误差范围内合计为 100%。缺失检索判定、解析失败或缺少 "
            "`update_pass` 的样本计入 CSV 的 `n_excluded`，不会作为错误答案。",
            "",
            "重新生成：",
            "",
            "```bash",
            "python evaluation/summarize_recommend_change_retrieval_combinations.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "acc_sum": 0}
    )
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
            groups = seed.get("groups") or {} if isinstance(seed, dict) else {}
            old = groups.get("old_preference_seed")
            new = groups.get("updated_preference_seed")
            annotation = seed.get("answer_judge") or {} if isinstance(seed, dict) else {}
            judge = ((row.get("with_memory") or {}).get("judge") or {})
            acc = annotation.get("update_pass", judge.get("update_pass"))
            valid = (
                isinstance(seed, dict)
                and seed.get("judge_parse_ok") is True
                and isinstance(old, dict)
                and isinstance(old.get("present"), bool)
                and isinstance(new, dict)
                and isinstance(new.get("present"), bool)
                and acc in (0, 1, False, True)
            )
            if not valid:
                diagnostics[(family, method)]["n_excluded"] += 1
                continue
            state = (bool(old["present"]), bool(new["present"]))
            condition = next(name for name, expected in CONDITIONS.items() if state == expected)
            bucket = counts[(family, method, condition)]
            bucket["n"] += 1
            bucket["acc_sum"] += int(acc)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {"task": "recommend_change", "diagnostics": {}}
    preferred = {name: index for index, name in enumerate(PREFERRED_METHOD_ORDER)}
    for family, stem in (("qwen3-8b", "qwen3_8b"), ("deepseek-v4-flash", "deepseek_v4_flash")):
        methods = sorted(
            methods_by_family[family], key=lambda name: (preferred.get(name, len(preferred)), name.lower())
        )
        output_rows: list[dict[str, Any]] = []
        for method in methods:
            diag = diagnostics[(family, method)]
            valid_total = diag["n_total"] - diag["n_excluded"]
            row: dict[str, Any] = {"framework": method, **diag, "n_valid": valid_total}
            metadata["diagnostics"][f"{family}/{method}"] = diag
            for condition in CONDITIONS:
                bucket = counts[(family, method, condition)]
                n = bucket["n"]
                row[f"{condition}_n"] = n
                row[f"{condition}_share_pct"] = round(100 * n / valid_total, 2) if valid_total else ""
                row[f"{condition}_acc_count"] = bucket["acc_sum"]
                row[f"{condition}_acc_pct"] = round(100 * bucket["acc_sum"] / n, 2) if n else ""
            output_rows.append(row)
        write_csv(OUTPUT_DIR / f"{stem}_recommend_change_retrieval_combinations.csv", output_rows)
        (OUTPUT_DIR / f"{stem}_recommend_change_retrieval_combinations.md").write_text(
            markdown(output_rows, family), encoding="utf-8"
        )

    metadata["conditions"] = {name: list(state) for name, state in CONDITIONS.items()}
    metadata["accuracy_field"] = "update_pass"
    (OUTPUT_DIR / "recommend_change_retrieval_combinations_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
