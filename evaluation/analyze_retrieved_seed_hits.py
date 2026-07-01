"""Judge whether retrieved memories contain task seed memories.

This is a post-hoc analyzer for baseline evaluation outputs. It does not rerun
the answer model. For each result with recorded retrieved memories, it asks an
LLM judge whether those retrieved memories semantically contain the seed
preference/evidence that the benchmark intended to test.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from _dataset_compat import to_legacy_row

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - handled at runtime.
    OpenAI = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - progress is optional.
    tqdm = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output_data" / "baseline_opt_v2_runs_short"
DEFAULT_ANALYSIS_JSONL = (
    REPO_ROOT / "output_data" / "analysis" / "retrieved_seed_hits.jsonl"
)
DEFAULT_SUMMARY_JSON = (
    REPO_ROOT / "output_data" / "analysis" / "retrieved_seed_hits_summary.json"
)

TASK_SOURCE_JSONL = {
    "recommend": REPO_ROOT / "data" / "personalized_recommendation.jsonl",
    "recommend_change": REPO_ROOT / "data" / "preference_change.jsonl",
    "evidence_memory_conflict_noisy": REPO_ROOT / "data" / "preference_fact_conflict.jsonl",
    "memory_scope_overgeneralization": REPO_ROOT / "data" / "contextual_scope_limits.jsonl",
    "consensus_judgment": REPO_ROOT / "data" / "objective_fact_judgment.jsonl",
}

DEFAULT_TASKS = tuple(TASK_SOURCE_JSONL)

SYSTEM_PROMPT = """You judge whether retrieved memories contain benchmark seed information.

You will receive seed_groups and retrieved_memories. A seed group is present
when at least one retrieved memory preserves the same user preference, outdated
preference, updated preference, misconception, or evidence as that seed group.
It does not need exact wording. Count concise summaries as present when the
meaning is clear. Do not count a memory as present merely because it mentions
the same broad topic while omitting the decisive preference/evidence.

Return JSON only:
{
  "groups": {
    "<group_name>": {
      "present": true,
      "supporting_indices": [0],
      "best_rank": 1,
      "confidence": "high",
      "brief_rationale": "Short reason."
    }
  }
}

Use zero-based supporting_indices. best_rank is one-based retrieval rank, or
null when absent. confidence is "high", "medium", or "low".
""".strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = to_legacy_row(json.loads(line))
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def index_source_rows(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        for key in (
            row.get("id"),
            row.get("query_id"),
            row.get("source_query_id"),
        ):
            if key:
                index[str(key)] = row
    return index


def compact_text(value: Any, *, max_chars: int = 4000) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def memory_text(memory: Any) -> str:
    if not isinstance(memory, dict):
        return compact_text(memory)
    parts: list[str] = []
    for key in ("content", "used_content"):
        value = memory.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    metadata = memory.get("metadata")
    if isinstance(metadata, dict):
        score = metadata.get("score")
        if score is not None:
            parts.append(f"score={score}")
    return compact_text("\n".join(parts), max_chars=2500)


def retrieved_memories(result: dict[str, Any]) -> list[str]:
    lightmem = result.get("lightmem") or {}
    memories = lightmem.get("retrieved_memories") or []
    if not isinstance(memories, list):
        return []
    return [memory_text(item) for item in memories if memory_text(item)]


def extracted_memories(row: dict[str, Any]) -> str:
    extraction = row.get("extraction") or {}
    memories = extraction.get("memories") if isinstance(
        extraction, dict) else None
    if not memories:
        return ""
    return compact_text(json.dumps(memories, ensure_ascii=False), max_chars=3000)


def seed_groups(task: str, result: dict[str, Any], source: dict[str, Any] | None) -> dict[str, str]:
    source = source or {}
    generated = result.get("generated_question") or source.get(
        "generated_question") or {}
    groups: dict[str, str] = {}

    if task == "recommend_change":
        groups["old_preference_seed"] = compact_text(
            "\n".join(
                [
                    str(result.get("old_preference")
                        or source.get("old_preference") or ""),
                    str(source.get("original_dialogue") or ""),
                    str(result.get("old_preference_trap")
                        or source.get("old_preference_trap") or ""),
                ]
            ),
            max_chars=5000,
        )
        groups["updated_preference_seed"] = compact_text(
            "\n".join(
                [
                    str(result.get("updated_preference")
                        or source.get("updated_preference") or ""),
                    str(source.get("later_user_utterance") or ""),
                    str(result.get("updated_preference_basis")
                        or source.get("updated_preference_basis") or ""),
                ]
            ),
            max_chars=5000,
        )
        return {k: v for k, v in groups.items() if v}

    preference_memory = result.get("preference_memory") or []
    groups["interfering_preference_seed"] = compact_text(
        "\n".join(
            [
                str(source.get("original_memory") or ""),
                extracted_memories(source),
                json.dumps(preference_memory, ensure_ascii=False),
                str(generated.get("memory_interference") or ""),
                str(generated.get("preference_interference") or ""),
                str(generated.get("evidence_from_memory") or ""),
                str(result.get("memory_trap")
                    or generated.get("memory_trap") or ""),
            ]
        ),
        max_chars=6000,
    )

    if task == "evidence_memory_conflict_noisy":
        groups["decisive_evidence_seed"] = compact_text(
            "\n".join(
                [
                    str(result.get("decisive_evidence")
                        or generated.get("decisive_evidence") or ""),
                    str(result.get("evidence_basis")
                        or generated.get("evidence_basis") or ""),
                    str(generated.get("evidence_from_memory") or ""),
                    str(result.get("open_answer_target")
                        or generated.get("open_answer_target") or ""),
                ]
            ),
            max_chars=5000,
        )

    return {k: v for k, v in groups.items() if v}


def result_failure(task: str, result: dict[str, Any]) -> dict[str, Any]:
    judge = ((result.get("with_memory") or {}).get("judge") or {})
    if task == "recommend":
        failed = (
            judge.get("answer_accuracy") == 0
            or judge.get("preference_used") == 0
        )
        return {
            "judge_parse_ok": judge.get("judge_parse_ok"),
            "answer_failed": bool(failed),
            "answer_accuracy": judge.get("answer_accuracy"),
            "preference_used": judge.get("preference_used"),
            "recommend_pass": judge.get("recommend_pass"),
        }
    if task == "recommend_change":
        failed = (
            judge.get("uses_latest_preference") == 0
            or judge.get("outdated_preference_contamination") == 1
        )
        return {
            "judge_parse_ok": judge.get("judge_parse_ok"),
            "answer_failed": bool(failed),
            "uses_latest_preference": judge.get("uses_latest_preference"),
            "outdated_preference_contamination": judge.get(
                "outdated_preference_contamination"
            ),
            "update_pass": judge.get("update_pass"),
        }
    if task == "consensus_judgment":
        failed = (
            judge.get("objective_correctness") == 0
            or judge.get("preference_contamination") == 1
            or judge.get("preference_answer_selected") == 1
        )
        return {
            "judge_parse_ok": judge.get("judge_parse_ok"),
            "answer_failed": bool(failed),
            "objective_correctness": judge.get("objective_correctness"),
            "preference_contamination": judge.get("preference_contamination"),
            "preference_answer_selected": judge.get("preference_answer_selected"),
            "suppress_pass": judge.get("suppress_pass"),
        }
    if task == "memory_scope_overgeneralization":
        failed = (
            judge.get("accuracy") == 0
            or judge.get("incorrectly_used_preference") == 1
        )
        return {
            "judge_parse_ok": judge.get("judge_parse_ok"),
            "answer_failed": bool(failed),
            "accuracy": judge.get("accuracy"),
            "incorrectly_used_preference": judge.get("incorrectly_used_preference"),
            "scope_pass": judge.get("scope_pass"),
        }
    failed = judge.get("accuracy") == 0 or judge.get(
        "misled_by_conflicting_memory") == 1
    return {
        "judge_parse_ok": judge.get("judge_parse_ok"),
        "answer_failed": bool(failed),
        "accuracy": judge.get("accuracy"),
        "misled_by_conflicting_memory": judge.get("misled_by_conflicting_memory"),
        "evidence_pass": judge.get("evidence_pass"),
    }


def parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```\w*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    match = re.search(r"\{[\s\S]*\}\s*$", stripped) or re.search(
        r"\{[\s\S]*\}", stripped
    )
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def judge_seed_hit(
    client: Any,
    *,
    model: str,
    seed_payload: dict[str, Any],
    max_retries: int,
) -> dict[str, Any]:
    last_error = ""
    for _ in range(max(1, max_retries)):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(seed_payload, ensure_ascii=False, indent=2),
                    },
                ],
                temperature=0.0,
                extra_body={"reasoning": {"enabled": False}},
            )
            raw = (response.choices[0].message.content or "").strip()
            parsed = parse_json_object(raw)
            return {
                "judge_parse_ok": parsed is not None,
                "judge_raw": raw,
                "groups": (parsed or {}).get("groups") if parsed else {},
                "judge_error": None if parsed is not None else "Could not parse JSON.",
            }
        except Exception as exc:  # noqa: BLE001 - report API failures in output.
            last_error = f"{type(exc).__name__}: {exc}"
    return {
        "judge_parse_ok": False,
        "judge_raw": "",
        "groups": {},
        "judge_error": last_error,
    }


def normalize_seed_judge(seed_judge: dict[str, Any], group_names: list[str]) -> dict[str, Any]:
    groups = seed_judge.get("groups")
    if not isinstance(groups, dict):
        groups = {}
    normalized: dict[str, Any] = {}
    for name in group_names:
        group = groups.get(name)
        if not isinstance(group, dict):
            group = {}
        present = group.get("present")
        if isinstance(present, str):
            present = present.strip().lower() in {"1", "true", "yes"}
        else:
            present = bool(present)
        supporting = group.get("supporting_indices")
        if not isinstance(supporting, list):
            supporting = []
        normalized[name] = {
            "present": present,
            "supporting_indices": supporting,
            "best_rank": group.get("best_rank") if present else None,
            "confidence": group.get("confidence") or ("low" if present else "high"),
            "brief_rationale": str(group.get("brief_rationale") or "").strip(),
        }
    seed_judge["groups"] = normalized
    return seed_judge


def normalized_test_jsonl(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def matches_task_dataset(task: str, data: dict[str, Any]) -> bool:
    expected_path = TASK_SOURCE_JSONL.get(task)
    if expected_path is None:
        return False
    expected = expected_path.relative_to(REPO_ROOT).as_posix()
    actual = normalized_test_jsonl(data.get("test_jsonl"))
    return actual == expected or actual.endswith(f"/{expected}")


def result_analysis_key(
    task: str,
    result_file: Path,
    test_jsonl: str,
    query_id: str,
) -> str:
    # `_final` marks a finalized artifact; it does not change sample identity.
    # Keep resume keys compatible with judgments written before files were renamed.
    result_name = result_file.name
    if result_file.stem.endswith("_final"):
        result_name = result_file.stem[: -len("_final")] + result_file.suffix
    return f"{task}/{test_jsonl}/{result_name}/{query_id}"


def iter_result_files(
    output_root: Path,
    tasks: tuple[str, ...],
    result_models: frozenset[str] | None = None,
    name_prefixes: frozenset[str] | None = None,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    files: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for task in tasks:
        task_dir = output_root / task
        if task_dir.exists():
            for result_file in sorted(task_dir.glob("*final*.json")):
                if name_prefixes is not None and not any(
                    result_file.name.startswith(prefix) for prefix in name_prefixes
                ):
                    continue
                data = read_json(result_file)
                if matches_task_dataset(task, data):
                    result_model = str(data.get("model") or "").strip()
                    if result_models is not None and result_model not in result_models:
                        continue
                    files.append(result_file)
                else:
                    skipped.append(
                        (result_file, normalized_test_jsonl(data.get("test_jsonl")))
                    )
    return files, skipped


def count_pending_records(
    files: list[Path],
    *,
    done: set[str],
    max_samples: int,
    include_raw_dialogue: bool,
) -> int:
    total = 0
    for result_file in files:
        task = result_file.parent.name
        if not include_raw_dialogue and (
            result_file.name.startswith("raw_dialogue_")
            or result_file.name.startswith("no_memory_")
        ):
            continue
        data = read_json(result_file)
        test_jsonl = normalized_test_jsonl(data.get("test_jsonl"))
        results = data.get("results") or []
        if not isinstance(results, list):
            continue
        for idx, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            if not retrieved_memories(result):
                continue
            query_id = str(result.get("query_id") or result.get("id") or idx)
            analysis_key = result_analysis_key(
                task, result_file, test_jsonl, query_id
            )
            if analysis_key in done:
                continue
            total += 1
            if max_samples > 0 and total >= max_samples:
                return total
    return total


def load_done_keys(path: Path, *, retry_failed: bool = False) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if retry_failed and ((row.get("seed_judge") or {}).get("judge_parse_ok") is not True):
            continue
        key = row.get("analysis_key")
        if key:
            done.add(str(key))
    return done


def update_summary(summary: dict[str, Any], record: dict[str, Any]) -> None:
    file_key = f"{record['task']}/{record['result_file']}"
    bucket = summary.setdefault(
        file_key, {"n": 0, "answer_failed": 0, "groups": {}})
    bucket["n"] += 1
    if (record.get("answer_judge") or {}).get("answer_failed"):
        bucket["answer_failed"] += 1
    for name, group in ((record.get("seed_judge") or {}).get("groups") or {}).items():
        gb = bucket["groups"].setdefault(
            name,
            {
                "present": 0,
                "absent": 0,
                "present_and_answer_failed": 0,
                "absent_and_answer_failed": 0,
            },
        )
        present = bool((group or {}).get("present"))
        failed = bool((record.get("answer_judge") or {}).get("answer_failed"))
        if present:
            gb["present"] += 1
            if failed:
                gb["present_and_answer_failed"] += 1
        else:
            gb["absent"] += 1
            if failed:
                gb["absent_and_answer_failed"] += 1


def annotation_from_record(record: dict[str, Any], *, model: str, base_url: str) -> dict[str, Any]:
    seed_judge = record.get("seed_judge") or {}
    return {
        "analysis_version": 1,
        "judge_model": model,
        "judge_base_url": base_url,
        "judge_parse_ok": seed_judge.get("judge_parse_ok"),
        "judge_error": seed_judge.get("judge_error"),
        "groups": seed_judge.get("groups") or {},
        "answer_judge": record.get("answer_judge") or {},
        "retrieved_memory_count": record.get("retrieved_memory_count"),
    }


def build_pending_jobs(
    files: list[Path],
    *,
    source_indexes: dict[str, dict[str, dict[str, Any]]],
    done: set[str],
    max_samples: int,
    include_raw_dialogue: bool,
) -> tuple[list[dict[str, Any]], dict[Path, dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    loaded_files: dict[Path, dict[str, Any]] = {}
    for result_file in files:
        task = result_file.parent.name
        if not include_raw_dialogue and (
            result_file.name.startswith("raw_dialogue_")
            or result_file.name.startswith("no_memory_")
        ):
            continue
        data = read_json(result_file)
        loaded_files[result_file] = data
        test_jsonl = normalized_test_jsonl(data.get("test_jsonl"))
        results = data.get("results") or []
        if not isinstance(results, list):
            continue
        for idx, result in enumerate(results):
            if max_samples > 0 and len(jobs) >= max_samples:
                return jobs, loaded_files
            if not isinstance(result, dict):
                continue
            memories = retrieved_memories(result)
            if not memories:
                continue
            query_id = str(result.get("query_id") or result.get("id") or idx)
            analysis_key = result_analysis_key(
                task, result_file, test_jsonl, query_id
            )
            if analysis_key in done:
                continue

            source = source_indexes.get(task, {}).get(query_id)
            groups = seed_groups(task, result, source)
            payload = {
                "task": task,
                "query_id": query_id,
                "seed_groups": groups,
                "retrieved_memories": [
                    {"index": i, "rank": i + 1, "text": text}
                    for i, text in enumerate(memories)
                ],
            }
            jobs.append(
                {
                    "analysis_key": analysis_key,
                    "task": task,
                    "test_jsonl": test_jsonl,
                    "result_file": result_file,
                    "result_name": result_file.name,
                    "result": result,
                    "query_id": query_id,
                    "memory_method": (result.get("lightmem") or {}).get("method"),
                    "model": data.get("model"),
                    "retrieved_memory_count": len(memories),
                    "answer_judge": result_failure(task, result),
                    "groups": groups,
                    "payload": payload,
                }
            )
    return jobs, loaded_files


def run_seed_job(
    job: dict[str, Any],
    *,
    client: Any,
    model: str,
    max_retries: int,
    dry_run: bool,
) -> dict[str, Any]:
    groups = job["groups"]
    seed_judge = (
        {
            "judge_parse_ok": None,
            "judge_raw": "",
            "groups": {},
            "judge_error": "dry_run",
        }
        if dry_run
        else judge_seed_hit(
            client,
            model=model,
            seed_payload=job["payload"],
            max_retries=max_retries,
        )
    )
    seed_judge = normalize_seed_judge(seed_judge, list(groups))
    return {
        "analysis_key": job["analysis_key"],
        "task": job["task"],
        "test_jsonl": job["test_jsonl"],
        "result_file": job["result_name"],
        "query_id": job["query_id"],
        "memory_method": job["memory_method"],
        "model": job["model"],
        "retrieved_memory_count": job["retrieved_memory_count"],
        "answer_judge": job["answer_judge"],
        "seed_payload": job["payload"] if dry_run else None,
        "seed_judge": seed_judge,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--analysis-jsonl", type=Path,
                        default=DEFAULT_ANALYSIS_JSONL)
    parser.add_argument("--summary-json", type=Path,
                        default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--model", default=os.environ.get("SEED_JUDGE_MODEL")
                        or os.environ.get("JUDGE_MODEL") or "deepseek-v4-flash")
    parser.add_argument(
        "--result-models",
        default="",
        help=(
            "Comma-separated answer-model names to include, matched against "
            "the top-level model field in each result JSON. Empty means all models."
        ),
    )
    parser.add_argument(
        "--methods",
        default="",
        help=(
            "Comma-separated memory-method filename prefixes to include, "
            "matched against the start of each result filename "
            "(e.g. memgpt_minimal,memorybank,supermemory). Empty means all methods."
        ),
    )
    parser.add_argument(
        "--base-url", default=os.environ.get("JUDGE_BASE_URL") or "https://api.deepseek.com")
    parser.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY")
                        or os.environ.get("DEEPSEEK_JUDGE_API_KEY") or "")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of concurrent seed-judge API requests.",
    )
    parser.add_argument("--api-max-retries", type=int, default=2)
    parser.add_argument("--include-raw-dialogue", action="store_true")
    parser.add_argument(
        "--annotate-results",
        action="store_true",
        help="Write seed retrieval judgments back into each result item.",
    )
    parser.add_argument(
        "--annotation-field",
        default="seed_retrieval_judge",
        help="Field name to add to each result item when --annotate-results is set.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="When resuming, retry records whose previous seed judge did not parse successfully.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    result_models = frozenset(
        model.strip() for model in args.result_models.split(",") if model.strip()
    )
    name_prefixes = frozenset(
        prefix.strip().rstrip("_") + "_"
        for prefix in args.methods.split(",")
        if prefix.strip()
    )
    unknown_tasks = sorted(set(tasks) - set(TASK_SOURCE_JSONL))
    if unknown_tasks:
        raise ValueError(
            "Unsupported tasks: "
            + ", ".join(unknown_tasks)
            + ". Supported tasks: "
            + ", ".join(DEFAULT_TASKS)
        )
    source_indexes = {
        task: index_source_rows(path)
        for task, path in TASK_SOURCE_JSONL.items()
        if task in tasks
    }

    files, skipped_files = iter_result_files(
        args.output_root,
        tasks,
        result_models=result_models or None,
        name_prefixes=name_prefixes or None,
    )
    if args.max_files > 0:
        files = files[: args.max_files]

    print(
        f"Selected {len(files)} result files matching the configured test_jsonl.")
    if result_models:
        print("Answer-model filter: " + ", ".join(sorted(result_models)))
    if name_prefixes:
        print("Method filter: " + ", ".join(sorted(name_prefixes)))
    if skipped_files:
        skipped_counts = Counter(
            (path.parent.name, test_jsonl or "<missing>")
            for path, test_jsonl in skipped_files
        )
        print(
            f"Skipped {len(skipped_files)} result files with another test_jsonl:")
        for (task, test_jsonl), count in sorted(skipped_counts.items()):
            print(f"  {task}: {test_jsonl} ({count} files)")

    if not args.dry_run and OpenAI is None:
        raise RuntimeError(
            "Install the OpenAI Python SDK to run the seed judge.")
    if not args.dry_run and not args.api_key:
        raise RuntimeError(
            "Missing judge API key. Set JUDGE_API_KEY or pass --api-key.")

    client = None
    if not args.dry_run:
        # type: ignore[misc]
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    args.analysis_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.no_resume else load_done_keys(
        args.analysis_jsonl,
        retry_failed=args.retry_failed,
    )
    summary: dict[str, Any] = {}
    written = 0
    jobs, loaded_files = build_pending_jobs(
        files,
        source_indexes=source_indexes,
        done=done,
        max_samples=args.max_samples,
        include_raw_dialogue=args.include_raw_dialogue,
    )
    progress = None
    if not args.no_progress and tqdm is not None:
        progress = tqdm(total=len(jobs),
                        desc="Judging seed retrieval", unit="sample")

    changed_files: set[Path] = set()
    with args.analysis_jsonl.open("a", encoding="utf-8") as out:
        try:
            workers = max(1, int(args.workers))
            if workers == 1:
                completed = (
                    (
                        job,
                        run_seed_job(
                            job,
                            client=client,
                            model=args.model,
                            max_retries=args.api_max_retries,
                            dry_run=args.dry_run,
                        ),
                    )
                    for job in jobs
                )
                for job, record in completed:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    update_summary(summary, record)
                    if args.annotate_results and not args.dry_run:
                        job["result"][args.annotation_field] = annotation_from_record(
                            record,
                            model=args.model,
                            base_url=args.base_url,
                        )
                        changed_files.add(job["result_file"])
                    written += 1
                    if progress is not None:
                        progress.update(1)
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_job = {
                        executor.submit(
                            run_seed_job,
                            job,
                            client=client,
                            model=args.model,
                            max_retries=args.api_max_retries,
                            dry_run=args.dry_run,
                        ): job
                        for job in jobs
                    }
                    for future in as_completed(future_to_job):
                        job = future_to_job[future]
                        record = future.result()
                        out.write(json.dumps(
                            record, ensure_ascii=False) + "\n")
                        out.flush()
                        update_summary(summary, record)
                        if args.annotate_results and not args.dry_run:
                            job["result"][args.annotation_field] = annotation_from_record(
                                record,
                                model=args.model,
                                base_url=args.base_url,
                            )
                            changed_files.add(job["result_file"])
                        written += 1
                        if progress is not None:
                            progress.update(1)
        finally:
            if progress is not None:
                progress.close()

    for result_file in sorted(changed_files):
        data = loaded_files[result_file]
        result_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {written} new records to {args.analysis_jsonl}")
    print(f"Wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
