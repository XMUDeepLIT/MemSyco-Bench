"""Write seed_retrieval_judge annotations back into result JSON files from
the cached analysis jsonl produced by analyze_retrieved_seed_hits.py.

This is a pure local patch: no API calls. It only fills result items whose
seed_retrieval_judge is missing AND whose analysis_key has a parse_ok record
in the analysis jsonl. Samples that retrieved no memories (and were therefore
never judged) are left untouched.

Reuses helpers from analyze_retrieved_seed_hits.py so the analysis_key
computation and annotation shape stay identical to the original annotator.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from analyze_retrieved_seed_hits import (
    REPO_ROOT,
    annotation_from_record,
    normalized_test_jsonl,
    result_analysis_key,
)


def load_jsonl_index(path: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    if not path.exists():
        return index
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("analysis_key")
            if key:
                index[str(key)] = record
    return index


def patch_file(
    path: Path,
    records: dict[str, dict],
    *,
    judge_model: str,
    judge_base_url: str,
) -> tuple[int, int, int]:
    """Return (n_missing, n_patched, n_already_present)."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    results = data.get("results") or []
    if not isinstance(results, list):
        return 0, 0, 0
    test_jsonl = normalized_test_jsonl(data.get("test_jsonl"))
    n_missing = 0
    n_patched = 0
    n_already = 0
    changed = False
    for idx, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        if result.get("seed_retrieval_judge") is not None:
            n_already += 1
            continue
        n_missing += 1
        query_id = str(result.get("query_id") or result.get("id") or idx)
        key = result_analysis_key(path.parent.name, path, test_jsonl, query_id)
        record = records.get(key)
        if not record:
            continue
        seed_judge = record.get("seed_judge") or {}
        if seed_judge.get("judge_parse_ok") is not True:
            continue
        result["seed_retrieval_judge"] = annotation_from_record(
            record, model=judge_model, base_url=judge_base_url
        )
        n_patched += 1
        changed = True
    if changed:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return n_missing, n_patched, n_already


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "output_data" / "baseline_opt_v2_runs_short",
    )
    parser.add_argument(
        "--analysis-jsonl",
        type=Path,
        default=REPO_ROOT / "output_data" / "analysis" / "v3_seed_hits.jsonl",
    )
    parser.add_argument(
        "--tasks",
        default="recommend",
        help="Comma-separated task folder names to patch.",
    )
    parser.add_argument(
        "--methods",
        default="memgpt_minimal",
        help="Comma-separated filename prefixes to patch (e.g. memgpt_minimal,lightmem_full).",
    )
    parser.add_argument(
        "--result-models",
        default="qwen/qwen3-8b",
        help="Comma-separated answer-model names to restrict to. Empty = all.",
    )
    parser.add_argument(
        "--judge-model",
        default="deepseek-v4-flash",
        help="Judge model name to record in the written annotation metadata.",
    )
    parser.add_argument(
        "--judge-base-url",
        default="https://api.deepseek.com",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be patched without writing files.",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    prefixes = [p.strip().rstrip("_") + "_" for p in args.methods.split(",") if p.strip()]
    result_models = {m.strip() for m in args.result_models.split(",") if m.strip()}

    records = load_jsonl_index(args.analysis_jsonl)
    print(f"Loaded {len(records)} records from {args.analysis_jsonl}")
    if not records:
        print("No analysis records found; nothing to patch.")
        return

    total_patched = 0
    for task in tasks:
        task_dir = args.input_root / task
        if not task_dir.exists():
            print(f"[skip] task dir not found: {task_dir}")
            continue
        for path in sorted(task_dir.glob("*final*.json")):
            if not any(path.name.startswith(p) for p in prefixes):
                continue
            data_head = json.loads(path.read_text(encoding="utf-8-sig"))
            result_model = str(data_head.get("model") or "").strip()
            if result_models and result_model not in result_models:
                continue
            if args.dry_run:
                # still need to parse to count, but don't write
                pass
            n_missing, n_patched, n_already = patch_file(
                path,
                records,
                judge_model=args.judge_model,
                judge_base_url=args.judge_base_url,
            ) if not args.dry_run else (0, 0, 0)
            if args.dry_run:
                # recompute counts without writing
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                test_jsonl = normalized_test_jsonl(data.get("test_jsonl"))
                n_missing = n_already = 0
                would_patch = 0
                for idx, r in enumerate(data.get("results") or []):
                    if not isinstance(r, dict):
                        continue
                    if r.get("seed_retrieval_judge") is not None:
                        n_already += 1
                        continue
                    n_missing += 1
                    qid = str(r.get("query_id") or r.get("id") or idx)
                    key = result_analysis_key(task, path, test_jsonl, qid)
                    rec = records.get(key)
                    if rec and (rec.get("seed_judge") or {}).get("judge_parse_ok") is True:
                        would_patch += 1
                tag = "DRY-RUN"
                print(f"[{tag}] {path.name}: missing={n_missing} already={n_already} would_patch={would_patch}")
                total_patched += would_patch
                continue
            tag = "PATCHED" if n_patched else "no-op"
            print(f"[{tag}] {path.name}: missing={n_missing} patched={n_patched} already={n_already}")
            total_patched += n_patched

    print(f"\nTotal {'would-patch' if args.dry_run else 'patched'}: {total_patched}")


if __name__ == "__main__":
    main()
