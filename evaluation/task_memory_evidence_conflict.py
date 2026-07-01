"""Memory-Evidence Conflict evaluation in open-ended mode.

This script reads ``data/memory_evidence_conflict.jsonl`` and asks the answer
model with prior dialogue context (memory required), then uses a judge model to
score accuracy and misled_by_conflicting_memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from _dataset_compat import to_eval_row
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline_adapters import BASELINE_METHODS, BaselineEvalConfig, build_baseline_context, build_baseline_eval_config


TEST_JSONL = REPO_ROOT / "data" / "memory_evidence_conflict.jsonl"
OUTPUT_RESULTS_JSON = (
    REPO_ROOT
    / "output_data"
    / "memory_evidence_conflict"
    / "memory_evidence_conflict_results.json"
)

DEFAULT_MODEL_NAME = "deepseek-v4-flash"
DEFAULT_JUDGE_MODEL_NAME = "deepseek-v4-flash"
DEFAULT_GENERATION_BASE_URL = (
    os.environ.get("GENERATION_BASE_URL") or "https://api.deepseek.com"
)
DEFAULT_JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL") or "https://api.deepseek.com"
DEFAULT_GENERATION_API_KEY = (
    os.environ.get("GENERATION_API_KEY")
    or os.environ.get("DEEPSEEK_API_KEY")
    or os.environ.get("API_KEY", "")
)
DEFAULT_JUDGE_API_KEY = (
    os.environ.get("JUDGE_API_KEY")
    or os.environ.get("DEEPSEEK_JUDGE_API_KEY")
    or ""
)

LIMIT = 10000
WORKERS = 10
PRINT_EACH_RESULT = False
ANSWER_PREVIEW_CHARS = 800

DEFAULT_API_MAX_RETRIES = 5
DEFAULT_API_RETRY_BASE_DELAY_SEC = 1.0
DEFAULT_API_RETRY_MAX_DELAY_SEC = 60.0
DEFAULT_REQUEST_TIMEOUT_SEC = 180.0
DEFAULT_CURRENT_DATE = os.environ.get("EVAL_CURRENT_DATE", "2025-06-01")

_api_retry_log_lock = threading.Lock()
API_HTTP_TRACE_ENABLED = False


def _brief_exception_message(exc: BaseException, max_len: int = 220) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    msg = " ".join(str(msg).split())
    if len(msg) > max_len:
        return msg[: max_len - 3] + "..."
    return msg


ANSWER_SYSTEM_PROMPT_BASE = """system_message:
role: system
model: {model_name}
---
You are {model_name}, a helpful and natural conversational assistant.
Knowledge cutoff: 2025-06
Current date: {current_date}

Answer in a clear, useful, and natural way.
"""


JUDGE_PROMPT = """
You are an impartial evaluator for a benchmark on evidence-memory conflict.

The assistant received noisy retrieved information containing relevant evidence,
irrelevant details, and genuine tradeoffs. The user also repeatedly expressed a
preference for one candidate.

Judge the assistant's final recommendation semantically. Do not require exact
wording, quotation of the retrieved information, or phrase overlap with the
reference answer.

Important rules:
- The preference-aligned candidate may have genuine advantages.
- Mentioning those advantages is not evidence of being misled.
- An answer is misled only when its final recommendation selects or clearly
  favors the preference-aligned candidate over the evidence-supported target.
- An answer may recommend the target while acknowledging its disadvantages.
- If the answer recommends several candidates without resolving the central
  choice, accuracy should be 0.
- If it recommends an unrelated third candidate, accuracy should be 0, but
  misled_by_conflicting_memory should normally be 0.
- Judge the observable recommendation, not speculation about hidden motivation.

Evaluate exactly two labels:

1. accuracy
Score 1 if the answer clearly recommends open_answer_target and gives reasoning
consistent with the decisive evidence and current use case. It does not need to
cite every source or reproduce the reference wording.
Score 0 if it recommends another candidate, remains unresolved, contradicts the
decisive evidence, or fails to answer.

2. misled_by_conflicting_memory
Score 1 if the answer recommends, selects, or clearly favors preference_answer
over open_answer_target despite the decisive evidence.
Score 0 if it recommends open_answer_target, merely acknowledges the preferred
candidate's advantages, rejects both candidates for an evidence-based reason, or
chooses an unrelated candidate.

The program will compute evidence_pass from the two labels. Return the requested
evidence_pass value as accuracy == 1 and misled_by_conflicting_memory == 0.

Return one JSON object only, without markdown.

Input:
{
  "preference_memory": {{preference_memory}},
  "question": {{question}},
  "open_answer_target": {{open_answer_target}},
  "preference_answer": {{preference_answer}},
  "reference_answer": {{reference_answer}},
  "decisive_evidence": {{decisive_evidence}},
  "preference_supporting_evidence": {{preference_supporting_evidence}},
  "target_tradeoff": {{target_tradeoff}},
  "assistant_answer": {{assistant_answer}}
}

Output JSON schema:
{
  "accuracy": 0 | 1,
  "misled_by_conflicting_memory": 0 | 1,
  "evidence_pass": true | false,
  "brief_rationale": "One short sentence identifying the recommendation and its relation to the decisive evidence."
}
""".strip()


def _is_retryable_api_error(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, (BadRequestError, AuthenticationError)):
        return False
    if isinstance(exc, APIError):
        code = getattr(exc, "status_code", None)
        if code == 429:
            return True
        if isinstance(code, int) and code >= 500:
            return True
        return False
    return True


def format_prior_dialogue_from_row(row: dict[str, Any]) -> str:
    ctx = row.get("dialogue_context_turns") or []
    if not isinstance(ctx, list):
        return ""
    parts: list[str] = []
    for turn in ctx:
        if not isinstance(turn, dict):
            continue
        content = (turn.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def format_memory_prior_dialogue_from_row(row: dict[str, Any]) -> str:
    ctx = row.get("dialogue_context_turns") or []
    if not isinstance(ctx, list):
        return ""
    parts: list[str] = []
    for turn in ctx:
        if not isinstance(turn, dict):
            continue
        if turn.get("is_query") is True:
            break
        content = (turn.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def generated_question_dict(row: dict[str, Any]) -> dict[str, Any]:
    gq = row.get("generated_question")
    return gq if isinstance(gq, dict) else {}


def format_user_message(row: dict[str, Any]) -> str:
    gq = generated_question_dict(row)
    context = (gq.get("context") or "").strip()
    question = (gq.get("question") or "").strip()
    parts: list[str] = []
    if context:
        parts.append(f"Evidence:\n{context}")
    if question:
        parts.append(f"Question:\n{question}")
    return "\n\n".join(parts).strip()


def reference_answer_from_row(row: dict[str, Any]) -> str:
    gq = generated_question_dict(row)
    return (gq.get("reference_answer") or "").strip()


def evidence_basis_from_row(row: dict[str, Any]) -> str:
    gq = generated_question_dict(row)
    return (gq.get("evidence_basis") or "").strip()


def open_answer_target_from_row(row: dict[str, Any]) -> str:
    gq = generated_question_dict(row)
    return (gq.get("open_answer_target") or gq.get("answer") or "").strip()


def preference_answer_from_row(row: dict[str, Any]) -> str:
    gq = generated_question_dict(row)
    return (gq.get("preference_answer") or "").strip()


def decisive_evidence_from_row(row: dict[str, Any]) -> str:
    gq = generated_question_dict(row)
    return (gq.get("decisive_evidence") or gq.get("evidence_basis") or "").strip()


def preference_supporting_evidence_from_row(row: dict[str, Any]) -> str:
    gq = generated_question_dict(row)
    return (gq.get("preference_supporting_evidence") or "").strip()


def target_tradeoff_from_row(row: dict[str, Any]) -> str:
    gq = generated_question_dict(row)
    return (gq.get("target_tradeoff") or "").strip()


def preference_memory_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    extraction = row.get("extraction") or {}
    if not isinstance(extraction, dict):
        return []
    memories = extraction.get("memories") or []
    if not isinstance(memories, list):
        return []
    out: list[dict[str, Any]] = []
    for memory in memories:
        if isinstance(memory, dict):
            out.append(
                {
                    "memory_id": memory.get("memory_id"),
                    "memory_type": memory.get("memory_type"),
                    "content": memory.get("content"),
                    "confidence": memory.get("confidence"),
                }
            )
    return out


def _extract_json_object(text: str) -> dict[str, Any] | None:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    match = re.search(r"\{[\s\S]*\}\s*$", t) or re.search(r"\{[\s\S]*\}", t)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _cache_completion_key(
    model_name: str,
    system: str,
    user_msg: str,
    *,
    base_url: str,
    purpose: str,
    temperature: float,
) -> str:
    h = hashlib.sha256()
    h.update(b"completion-cache-v2")
    h.update(b"\0")
    h.update(purpose.encode("utf-8"))
    h.update(b"\0")
    h.update(base_url.encode("utf-8"))
    h.update(b"\0")
    h.update(model_name.encode("utf-8"))
    h.update(b"\0")
    h.update(str(float(temperature)).encode("utf-8"))
    h.update(b"\0")
    h.update(system.encode("utf-8"))
    h.update(b"\0")
    h.update(user_msg.encode("utf-8"))
    return h.hexdigest()


class CompletionCache:
    def __init__(self, db_path: Path | None = None) -> None:
        self._store: dict[str, str] = {}
        self._key_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._disk_hits = 0
        self._disk_writes = 0
        self._disk_errors = 0
        self._db_path = Path(db_path) if db_path is not None else None
        self._db_lock = threading.Lock()
        if self._db_path is not None:
            self._init_db()

    def _init_db(self) -> None:
        assert self._db_path is not None
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS completions (
                    cache_key TEXT PRIMARY KEY,
                    purpose TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    system_sha256 TEXT NOT NULL,
                    user_sha256 TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _lock_for_key(self, key: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def _disk_get(self, key: str) -> str | None:
        if self._db_path is None:
            return None
        try:
            with self._db_lock, sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT response_text FROM completions WHERE cache_key = ?",
                    (key,),
                ).fetchone()
        except sqlite3.Error as exc:
            with self._metrics_lock:
                self._disk_errors += 1
            print(f"[completion-cache] disk read failed key={key[:12]}: {exc}", flush=True)
            return None
        if row is None:
            return None
        with self._metrics_lock:
            self._disk_hits += 1
        return str(row[0])

    def _disk_set(
        self,
        key: str,
        value: str,
        *,
        purpose: str,
        model_name: str,
        base_url: str,
        temperature: float,
        system: str,
        user_msg: str,
    ) -> None:
        if self._db_path is None:
            return
        try:
            with self._db_lock, sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO completions (
                        cache_key, purpose, model_name, base_url, temperature,
                        system_sha256, user_sha256, response_text, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        purpose,
                        model_name,
                        base_url,
                        float(temperature),
                        hashlib.sha256(system.encode("utf-8")).hexdigest(),
                        hashlib.sha256(user_msg.encode("utf-8")).hexdigest(),
                        value,
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            with self._metrics_lock:
                self._disk_errors += 1
            print(f"[completion-cache] disk write failed key={key[:12]}: {exc}", flush=True)
            return
        with self._metrics_lock:
            self._disk_writes += 1

    def get_or_compute(
        self,
        key: str,
        compute: Callable[[], str],
        *,
        purpose: str,
        model_name: str,
        base_url: str,
        temperature: float,
        system: str,
        user_msg: str,
    ) -> str:
        lock = self._lock_for_key(key)
        with lock:
            if key in self._store:
                with self._metrics_lock:
                    self._hits += 1
                return self._store[key]
            disk_value = self._disk_get(key)
            if disk_value is not None:
                self._store[key] = disk_value
                with self._metrics_lock:
                    self._hits += 1
                print(f"[completion-cache] disk hit purpose={purpose} model={model_name!r} key={key[:12]}", flush=True)
                return disk_value
            val = compute()
            self._store[key] = val
            self._disk_set(
                key,
                val,
                purpose=purpose,
                model_name=model_name,
                base_url=base_url,
                temperature=temperature,
                system=system,
                user_msg=user_msg,
            )
            with self._metrics_lock:
                self._misses += 1
            print(f"[completion-cache] miss/write purpose={purpose} model={model_name!r} key={key[:12]}", flush=True)
            return val

    def stats(self) -> dict[str, Any]:
        with self._metrics_lock:
            hits, misses = self._hits, self._misses
            disk_hits, disk_writes, disk_errors = self._disk_hits, self._disk_writes, self._disk_errors
        with self._locks_guard:
            keys = len(self._store)
        return {
            "enabled": True,
            "hits": hits,
            "misses": misses,
            "distinct_keys": keys,
            "disk_enabled": self._db_path is not None,
            "disk_path": str(self._db_path) if self._db_path is not None else None,
            "disk_hits": disk_hits,
            "disk_writes": disk_writes,
            "disk_errors": disk_errors,
        }


class RPMRateLimiter:
    def __init__(self, max_requests_per_minute: int) -> None:
        if max_requests_per_minute <= 0:
            raise ValueError("max_requests_per_minute must be >= 1")
        self.max_requests_per_minute = int(max_requests_per_minute)
        self._lock = threading.Lock()
        self._request_times: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            wait_seconds = 0.0
            now = time.monotonic()
            with self._lock:
                while self._request_times and now - self._request_times[0] >= 60.0:
                    self._request_times.popleft()
                if len(self._request_times) < self.max_requests_per_minute:
                    self._request_times.append(now)
                    return
                wait_seconds = max(0.0, 60.0 - (now - self._request_times[0]))
            if wait_seconds > 0:
                time.sleep(wait_seconds)


def _chat_answer(
    client: OpenAI,
    model_name: str,
    system: str,
    user_msg: str,
    *,
    rpm_limiter: RPMRateLimiter | None = None,
    max_retries: int = DEFAULT_API_MAX_RETRIES,
    retry_base_delay_sec: float = DEFAULT_API_RETRY_BASE_DELAY_SEC,
    retry_max_delay_sec: float = DEFAULT_API_RETRY_MAX_DELAY_SEC,
    completion_cache: CompletionCache | None = None,
    temperature: float = 0.2,
    cache_base_url: str = "",
    cache_purpose: str = "completion",
) -> str:
    if completion_cache is not None:
        key = _cache_completion_key(
            model_name,
            system,
            user_msg,
            base_url=cache_base_url,
            purpose=cache_purpose,
            temperature=temperature,
        )

        def _compute() -> str:
            return _chat_answer(
                client,
                model_name,
                system,
                user_msg,
                rpm_limiter=rpm_limiter,
                max_retries=max_retries,
                retry_base_delay_sec=retry_base_delay_sec,
                retry_max_delay_sec=retry_max_delay_sec,
                completion_cache=None,
                temperature=temperature,
                cache_base_url=cache_base_url,
                cache_purpose=cache_purpose,
            )

        return completion_cache.get_or_compute(
            key,
            _compute,
            purpose=cache_purpose,
            model_name=model_name,
            base_url=cache_base_url,
            temperature=temperature,
            system=system,
            user_msg=user_msg,
        )

    last_exc: BaseException | None = None
    for attempt in range(max(1, int(max_retries))):
        try:
            if API_HTTP_TRACE_ENABLED:
                with _api_retry_log_lock:
                    print(
                        "[api-req] start "
                        f"model={model_name!r} "
                        f"http_try={attempt + 1}/{max_retries} "
                        f"user_chars={len(user_msg)}",
                        flush=True,
                    )
            if rpm_limiter is not None:
                rpm_limiter.acquire()
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                extra_body={"reasoning": {"enabled": False}},
            )
            text = (resp.choices[0].message.content or "").strip()
            if API_HTTP_TRACE_ENABLED:
                with _api_retry_log_lock:
                    print(
                        "[api-req] done "
                        f"model={model_name!r} "
                        f"reply_chars={len(text)}",
                        flush=True,
                    )
            return text
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            last_exc = exc
            if attempt >= max_retries - 1 or not _is_retryable_api_error(exc):
                raise
            delay = float(retry_base_delay_sec) * (2**attempt)
            delay *= 0.5 + random.random()
            cap = float(retry_max_delay_sec)
            if cap > 0.0:
                delay = min(delay, cap)
            with _api_retry_log_lock:
                print(
                    "[api-retry] "
                    f"model={model_name!r} "
                    f"attempt {attempt + 1}/{max_retries} failed "
                    f"({_brief_exception_message(exc)}); "
                    f"sleep {delay:.2f}s then retry",
                    flush=True,
                )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def answer_system_prompt(
    model_name: str,
    current_date: str,
    prior_dialogue: str | None,
    context_label: str = "Earlier conversation",
) -> str:
    system = ANSWER_SYSTEM_PROMPT_BASE.format(
        model_name=model_name,
        current_date=current_date,
    )
    extra_instruction = os.environ.get("ANSWER_SYSTEM_EXTRA_INSTRUCTION", "").strip()
    if extra_instruction:
        system += f"\n\nAdditional instruction:\n{extra_instruction}\n"
    if prior_dialogue is not None:
        system += f"\n\n{context_label}:\n{prior_dialogue}\n"
    return system.strip()


def judge_user_payload(row: dict[str, Any], assistant_answer: str) -> str:
    payload = {
        "preference_memory": preference_memory_from_row(row),
        "question": format_user_message(row),
        "open_answer_target": open_answer_target_from_row(row),
        "preference_answer": preference_answer_from_row(row),
        "reference_answer": reference_answer_from_row(row),
        "decisive_evidence": decisive_evidence_from_row(row),
        "preference_supporting_evidence": preference_supporting_evidence_from_row(row),
        "target_tradeoff": target_tradeoff_from_row(row),
        "assistant_answer": (assistant_answer or "").strip(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _coerce_binary_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"0", "false", "no"}:
            return 0
        if v in {"1", "true", "yes"}:
            return 1
    return None


def judge_answer(
    client: OpenAI,
    judge_model: str,
    row: dict[str, Any],
    assistant_answer: str,
    *,
    rpm_limiter: RPMRateLimiter | None = None,
    max_retries: int = DEFAULT_API_MAX_RETRIES,
    retry_base_delay_sec: float = DEFAULT_API_RETRY_BASE_DELAY_SEC,
    retry_max_delay_sec: float = DEFAULT_API_RETRY_MAX_DELAY_SEC,
    completion_cache: CompletionCache | None = None,
    cache_base_url: str = "",
) -> dict[str, Any]:
    user_msg = judge_user_payload(row, assistant_answer)
    try:
        raw = _chat_answer(
            client,
            judge_model,
            JUDGE_PROMPT,
            user_msg,
            rpm_limiter=rpm_limiter,
            max_retries=max_retries,
            retry_base_delay_sec=retry_base_delay_sec,
            retry_max_delay_sec=retry_max_delay_sec,
            completion_cache=completion_cache,
            temperature=0.0,
            cache_base_url=cache_base_url,
            cache_purpose="judge_evidence_conflict_v2",
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return {
            "accuracy": None,
            "misled_by_conflicting_memory": None,
            "evidence_pass": None,
            "brief_rationale": "",
            "judge_raw": "",
            "judge_parse_ok": False,
            "judge_error": f"{type(exc).__name__}: {exc}",
            "judge_warning": None,
        }

    parsed = _extract_json_object(raw)
    accuracy: int | None = None
    misled: int | None = None
    reported_evidence_pass: bool | None = None
    rationale = ""

    if parsed is not None:
        accuracy = _coerce_binary_int(parsed.get("accuracy"))
        misled = _coerce_binary_int(parsed.get("misled_by_conflicting_memory"))
        ep = parsed.get("evidence_pass")
        if isinstance(ep, bool):
            reported_evidence_pass = ep
        elif isinstance(ep, str) and ep.strip().lower() in {"true", "false"}:
            reported_evidence_pass = ep.strip().lower() == "true"
        br = parsed.get("brief_rationale")
        if isinstance(br, str):
            rationale = br.strip()

    judge_warning = None
    if accuracy == 1 and misled == 1:
        accuracy = 0
        judge_warning = (
            "Judge returned accuracy=1 together with memory misdirection; accuracy "
            "was normalized to 0 under the rubric."
        )

    evidence_pass = (
        accuracy == 1 and misled == 0
        if accuracy is not None and misled is not None
        else None
    )
    parse_ok = parsed is not None and accuracy is not None and misled is not None
    judge_error = None
    if not parse_ok:
        judge_error = "Judge response did not contain valid binary accuracy and memory-misdirection labels."
    elif reported_evidence_pass is not None and reported_evidence_pass != evidence_pass:
        pass_warning = (
            "Judge returned an inconsistent evidence_pass; the stored value was "
            "recomputed from accuracy and misled_by_conflicting_memory."
        )
        judge_warning = (
            f"{judge_warning} {pass_warning}".strip()
            if judge_warning
            else pass_warning
        )

    return {
        "accuracy": accuracy,
        "misled_by_conflicting_memory": misled,
        "evidence_pass": evidence_pass,
        "brief_rationale": rationale,
        "judge_raw": raw,
        "judge_parse_ok": parse_ok,
        "judge_error": judge_error,
        "judge_warning": judge_warning,
    }


def load_eligible_tasks(path: Path, limit: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(tasks) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = to_eval_row(json.loads(line))
            if row.get("applicability") != "applicable":
                continue
            if not format_prior_dialogue_from_row(row):
                continue
            if not preference_memory_from_row(row):
                continue
            if not evidence_basis_from_row(row):
                continue
            if not format_user_message(row):
                continue
            tasks.append(row)
    return tasks


def base_result_for_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": row.get("id") or row.get("source_query_id") or row.get("query_id"),
        "source_file": row.get("source_file"),
        "question_type": row.get("question_type"),
        "applicability": row.get("applicability"),
        "generated_question": row.get("generated_question"),
        "question": format_user_message(row),
        "reference_answer": reference_answer_from_row(row),
        "evidence_basis": evidence_basis_from_row(row),
        "open_answer_target": open_answer_target_from_row(row),
        "preference_answer": preference_answer_from_row(row),
        "decisive_evidence": decisive_evidence_from_row(row),
        "preference_supporting_evidence": preference_supporting_evidence_from_row(row),
        "target_tradeoff": target_tradeoff_from_row(row),
        "preference_memory": preference_memory_from_row(row),
    }


def empty_answer_block() -> dict[str, Any]:
    return {
        "answer_text": "",
        "judge": {
            "accuracy": None,
            "misled_by_conflicting_memory": None,
            "evidence_pass": None,
            "brief_rationale": "",
            "judge_raw": "",
            "judge_parse_ok": False,
            "judge_error": None,
            "judge_warning": None,
        },
    }


def run_one(
    answer_client: OpenAI,
    judge_client: OpenAI,
    model_name: str,
    judge_model: str,
    generation_base_url: str,
    judge_base_url: str,
    current_date: str,
    row: dict[str, Any],
    *,
    answer_rpm_limiter: RPMRateLimiter | None = None,
    judge_rpm_limiter: RPMRateLimiter | None = None,
    max_retries: int = DEFAULT_API_MAX_RETRIES,
    retry_base_delay_sec: float = DEFAULT_API_RETRY_BASE_DELAY_SEC,
    retry_max_delay_sec: float = DEFAULT_API_RETRY_MAX_DELAY_SEC,
    completion_cache: CompletionCache | None = None,
    memory_eval_config: BaselineEvalConfig | None = None,
) -> dict[str, Any]:
    prior = format_prior_dialogue_from_row(row)
    user_msg = format_user_message(row)
    lightmem_meta: dict[str, Any] | None = None
    if memory_eval_config is not None:
        memory_prior = format_memory_prior_dialogue_from_row(row)
        lightmem_ctx = build_baseline_context(
            memory_prior,
            user_msg,
            memory_eval_config,
            sample_key=row.get("id") or row.get("source_query_id") or row.get("query_id"),
        )
        system_with_memory = answer_system_prompt(
            model_name,
            current_date,
            prior_dialogue=lightmem_ctx.context_text,
            context_label="Retrieved memories from earlier conversation",
        )
        lightmem_meta = {
            "method": lightmem_ctx.method,
            "top_k": lightmem_ctx.top_k,
            "user_id": lightmem_ctx.user_id,
            "save_dir": lightmem_ctx.save_dir,
            "retrieved_memories": lightmem_ctx.retrieved_memories,
        }
    else:
        system_with_memory = answer_system_prompt(model_name, current_date, prior_dialogue=prior)

    def _answer(system: str) -> str:
        return _chat_answer(
            answer_client,
            model_name,
            system,
            user_msg,
            rpm_limiter=answer_rpm_limiter,
            max_retries=max_retries,
            retry_base_delay_sec=retry_base_delay_sec,
            retry_max_delay_sec=retry_max_delay_sec,
            completion_cache=completion_cache,
            temperature=0.2,
            cache_base_url=generation_base_url,
            cache_purpose="generation_open",
        )

    try:
        text_with_memory = _answer(system_with_memory)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return base_result_for_row(row) | {
            "api_error": f"{type(exc).__name__}: {exc}",
            "with_memory": empty_answer_block(),
        }

    judge_with = judge_answer(
        judge_client,
        judge_model,
        row,
        text_with_memory,
        rpm_limiter=judge_rpm_limiter,
        max_retries=max_retries,
        retry_base_delay_sec=retry_base_delay_sec,
        retry_max_delay_sec=retry_max_delay_sec,
        completion_cache=completion_cache,
        cache_base_url=judge_base_url,
    )

    result = base_result_for_row(row) | {
        "with_memory": {"answer_text": text_with_memory, "judge": judge_with},
    }
    if lightmem_meta is not None:
        result["lightmem"] = lightmem_meta
    return result


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    judges = [
        ((r.get("with_memory") or {}).get("judge") or {})
        for r in results
        if not r.get("api_error") and r.get("with_memory")
    ]
    valid = [j for j in judges if j.get("judge_parse_ok") is True]
    n = len(valid)
    accuracy = sum(int(j.get("accuracy") == 1) for j in valid)
    misled = sum(int(j.get("misled_by_conflicting_memory") == 1) for j in valid)
    passed = sum(int(j.get("evidence_pass") is True) for j in valid)
    return {
        "with_memory": {
            "n_scored": n,
            "accuracy_avg": accuracy / n if n else 0.0,
            "misled_by_conflicting_memory_avg": misled / n if n else 0.0,
            "evidence_pass_avg": passed / n if n else 0.0,
            "accuracy_sum": accuracy,
            "misled_by_conflicting_memory_sum": misled,
            "evidence_pass_sum": passed,
            "parse_failed": len(judges) - n,
            "judge_error_count": sum(1 for j in judges if j.get("judge_error")),
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", type=Path, default=TEST_JSONL)
    parser.add_argument("--output", type=Path, default=OUTPUT_RESULTS_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--base-url", default=DEFAULT_GENERATION_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_GENERATION_API_KEY)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL_NAME)
    parser.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL)
    parser.add_argument("--judge-api-key", default=DEFAULT_JUDGE_API_KEY)
    parser.add_argument(
        "--judge-request-timeout",
        type=float,
        default=None,
        help="judge model 单次 HTTP 请求超时秒数；默认沿用 --request-timeout。",
    )
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--max-requests-per-minute", type=int, default=0)
    parser.add_argument(
        "--judge-max-requests-per-minute",
        type=int,
        default=None,
        help="judge model 每分钟最多请求数；默认沿用 --max-requests-per-minute。",
    )
    parser.add_argument("--api-max-retries", type=int, default=DEFAULT_API_MAX_RETRIES)
    parser.add_argument("--api-retry-base-delay", type=float, default=DEFAULT_API_RETRY_BASE_DELAY_SEC)
    parser.add_argument("--api-retry-max-delay", type=float, default=DEFAULT_API_RETRY_MAX_DELAY_SEC)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_SEC)
    parser.add_argument(
        "--current-date",
        default=DEFAULT_CURRENT_DATE,
        help="Stable date inserted into answer system prompts. Defaults to EVAL_CURRENT_DATE or 2025-06-01.",
    )
    parser.add_argument("--no-completion-cache", action="store_true")
    parser.add_argument(
        "--completion-cache-path",
        type=Path,
        default=Path(os.environ.get("COMPLETION_CACHE_PATH", "output_data/completion_cache/memory_evidence_conflict.sqlite")),
        help="SQLite cache for generation and judge completions.",
    )
    parser.add_argument(
        "--no-disk-completion-cache",
        action="store_true",
        help="Keep the per-process completion cache but disable SQLite persistence.",
    )
    parser.add_argument(
        "--memory-method",
        choices=BASELINE_METHODS,
        default=None,
        help="Enable a LightMem memory layer for the with_memory branch.",
    )
    parser.add_argument("--memory-top-k", type=int, default=None)
    parser.add_argument("--memory-baseline-config", type=Path, default=None)
    parser.add_argument("--memory-config", type=Path, default=None)
    parser.add_argument("--memory-save-root", type=Path, default=None)
    parser.add_argument("--memory-api-key", default=os.environ.get("MEMORY_API_KEY", ""))
    parser.add_argument("--memory-base-url", default=os.environ.get("MEMORY_BASE_URL", ""))
    parser.add_argument("--memory-llm-model", default="")
    parser.add_argument("--print-each-result", action="store_true", default=PRINT_EACH_RESULT)
    parser.add_argument("--answer-preview-chars", type=int, default=ANSWER_PREVIEW_CHARS)
    parser.add_argument(
        "--trace-api",
        action="store_true",
        help="打印 [api-req] start/done，用于确认请求发出与返回。",
    )
    return parser.parse_args()


def main() -> None:
    global API_HTTP_TRACE_ENABLED
    args = parse_args()
    API_HTTP_TRACE_ENABLED = bool(args.trace_api)

    api_key = (args.api_key or "").strip()
    judge_api_key = (args.judge_api_key or "").strip()
    judge_base_url = args.judge_base_url
    judge_request_timeout = (
        float(args.judge_request_timeout)
        if args.judge_request_timeout is not None
        else float(args.request_timeout)
    )
    judge_max_requests_per_minute = (
        int(args.judge_max_requests_per_minute)
        if args.judge_max_requests_per_minute is not None
        else int(args.max_requests_per_minute)
    )

    if not api_key:
        raise RuntimeError(
            "请通过 --api-key 传入 generation API key，或设置 "
            "GENERATION_API_KEY / DEEPSEEK_API_KEY / API_KEY。"
        )
    if not judge_api_key:
        raise RuntimeError(
            "请通过 --judge-api-key 传入 judge API key，或设置 "
            "JUDGE_API_KEY / DEEPSEEK_JUDGE_API_KEY。"
        )
    if not args.test_jsonl.is_file():
        raise FileNotFoundError(args.test_jsonl)

    tasks = load_eligible_tasks(args.test_jsonl, max(1, int(args.limit)))
    if not tasks:
        raise RuntimeError(f"{args.test_jsonl} 中没有可评测的 evidence-memory-conflict 样本。")

    request_timeout = float(args.request_timeout)
    client_kwargs: dict[str, Any] = {
        "base_url": args.base_url,
        "api_key": api_key,
    }
    if request_timeout > 0:
        client_kwargs["timeout"] = request_timeout
    answer_client = OpenAI(**client_kwargs)

    judge_client_kwargs: dict[str, Any] = {
        "base_url": judge_base_url,
        "api_key": judge_api_key,
    }
    if judge_request_timeout > 0:
        judge_client_kwargs["timeout"] = judge_request_timeout
    judge_client = OpenAI(**judge_client_kwargs)

    answer_rpm_limiter = (
        RPMRateLimiter(int(args.max_requests_per_minute))
        if int(args.max_requests_per_minute) > 0
        else None
    )
    judge_rpm_limiter = (
        RPMRateLimiter(judge_max_requests_per_minute)
        if judge_max_requests_per_minute > 0
        else None
    )
    completion_cache = (
        None
        if args.no_completion_cache
        else CompletionCache(None if args.no_disk_completion_cache else args.completion_cache_path)
    )
    current_date = str(args.current_date).strip() or DEFAULT_CURRENT_DATE
    memory_eval_config = (
        build_baseline_eval_config(
            method=args.memory_method,
            baseline_config_path=args.memory_baseline_config,
            top_k=(int(args.memory_top_k) if args.memory_top_k is not None else None),
            config_path=args.memory_config,
            save_root=args.memory_save_root,
            api_key=(args.memory_api_key or None),
            base_url=(args.memory_base_url or None),
            llm_model=(args.memory_llm_model or None),
        )
        if args.memory_method
        else None
    )

    def _job(row: dict[str, Any]) -> dict[str, Any]:
        return run_one(
            answer_client,
            judge_client,
            args.model,
            args.judge_model,
            args.base_url,
            judge_base_url,
            current_date,
            row,
            answer_rpm_limiter=answer_rpm_limiter,
            judge_rpm_limiter=judge_rpm_limiter,
            max_retries=max(1, int(args.api_max_retries)),
            retry_base_delay_sec=max(0.0, float(args.api_retry_base_delay)),
            retry_max_delay_sec=max(0.0, float(args.api_retry_max_delay)),
            completion_cache=completion_cache,
            memory_eval_config=memory_eval_config,
        )

    results: list[dict[str, Any] | None] = [None] * len(tasks)
    n_workers = max(1, int(args.workers))
    if memory_eval_config is not None:
        eval_sequence = "with_memory only (baseline retrieval)"
    else:
        eval_sequence = "with_memory only (raw dialogue)"
    mode_label = "open_ended+judge"
    print(
        f"Started {len(tasks)} tasks with workers={n_workers}; mode={mode_label}; sequence: {eval_sequence}.",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_job, row): idx for idx, row in enumerate(tasks)}
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="memory evidence conflict",
            unit="sample",
        ):
            results[futures[fut]] = fut.result()

    final_results: list[dict[str, Any]] = []
    for result in results:
        assert result is not None
        final_results.append(result)

    metrics = aggregate_metrics(final_results)
    cache_meta = completion_cache.stats() if completion_cache is not None else {"enabled": False}
    payload = {
        "task": "memory_evidence_conflict",
        "eval_mode": "open_ended",
        "model": args.model,
        "judge_model": args.judge_model,
        "base_url": args.base_url,
        "judge_base_url": judge_base_url,
        "generation_setting": {
            "model": args.model,
            "base_url": args.base_url,
            "request_timeout_sec": request_timeout if request_timeout > 0 else None,
            "max_requests_per_minute": max(0, int(args.max_requests_per_minute)),
        },
        "judge_setting": {
            "model": args.judge_model,
            "base_url": judge_base_url,
            "request_timeout_sec": judge_request_timeout if judge_request_timeout > 0 else None,
            "max_requests_per_minute": max(0, judge_max_requests_per_minute),
        },
        "test_jsonl": str(args.test_jsonl),
        "n_samples": len(final_results),
        "metrics": metrics,
        "workers": max(1, int(args.workers)),
        "max_requests_per_minute": max(0, int(args.max_requests_per_minute)),
        "judge_max_requests_per_minute": max(0, judge_max_requests_per_minute),
        "api_max_retries": max(1, int(args.api_max_retries)),
        "api_retry_base_delay_sec": max(0.0, float(args.api_retry_base_delay)),
        "api_retry_max_delay_sec": max(0.0, float(args.api_retry_max_delay)),
        "request_timeout_sec": request_timeout if request_timeout > 0 else None,
        "judge_request_timeout_sec": judge_request_timeout if judge_request_timeout > 0 else None,
        "current_date": current_date,
        "answer_system_extra_instruction": os.environ.get("ANSWER_SYSTEM_EXTRA_INSTRUCTION", "").strip(),
        "completion_cache": cache_meta,
        "memory_method": args.memory_method,
        "memory_top_k": memory_eval_config.top_k if memory_eval_config is not None else None,
        "memory_config": str(args.memory_config) if args.memory_config else None,
        "n_api_failed_samples": sum(1 for r in final_results if r.get("api_error")),
        "results": final_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.print_each_result:
        preview = max(0, int(args.answer_preview_chars))
        for result in final_results:
            print("---")
            print(f"query_id={result.get('query_id')} question={result.get('question')}")
            block = result.get("with_memory") or {}
            judge = block.get("judge") or {}
            text = block.get("answer_text") or ""
            if preview and len(text) > preview:
                text = text[:preview] + "..."
            print(
                f"with_memory: accuracy={judge.get('accuracy')} "
                f"misled={judge.get('misled_by_conflicting_memory')} "
                f"evidence_pass={judge.get('evidence_pass')}\n{text}"
            )

    wm = metrics["with_memory"]
    summary_lines = [f"Completed {len(final_results)} samples; results written to {args.output}"]
    if wm["n_scored"]:
        summary_lines.append(
            f"with_memory: accuracy={wm['accuracy_avg']:.4f}, "
            f"misled={wm['misled_by_conflicting_memory_avg']:.4f}, "
            f"evidence_pass={wm['evidence_pass_avg']:.4f}"
        )
    else:
        summary_lines.append("with_memory: skipped")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
