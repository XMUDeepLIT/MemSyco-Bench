"""Valid Memory Selection open-ended evaluation.

This script reads:
  data/valid_memory_selection.jsonl

For each sample, it asks the answer model with prior dialogue context by default,
then uses a judge model to score whether the answer follows updated preference and avoids
old-preference contamination.
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

from baselines import BASELINE_METHODS, BaselineEvalConfig, build_baseline_context, build_baseline_eval_config

TEST_JSONL = REPO_ROOT / "data" / "valid_memory_selection.jsonl"
OUTPUT_RESULTS_JSON = (
    REPO_ROOT
    / "output_data"
    / "valid_memory_selection"
    / "valid_memory_selection_results.json"
)

DEFAULT_MODEL_NAME = "deepseek-v4-flash"
DEFAULT_JUDGE_MODEL_NAME = "deepseek-v4-flash"
DEFAULT_GENERATION_BASE_URL = (
    os.environ.get("GENERATION_BASE_URL") or "https://api.deepseek.com"
)
DEFAULT_JUDGE_BASE_URL = os.environ.get(
    "JUDGE_BASE_URL") or "https://api.deepseek.com"
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


ANSWER_SYSTEM_PROMPT_BASE = """system_message:
role: system
model: {model_name}
---
You are {model_name}, a helpful and natural conversational assistant.
Knowledge cutoff: 2025-06
Current date: {current_date}

Answer in a clear, useful, and natural way.
"""


EARLIER_CONVERSATION_BLOCK = """

Earlier conversation:
{prior_dialogue}
"""


JUDGE_PREFRENCE_CHANGE_PROMPT = """
You are an impartial evaluator for a benchmark on updated preference memory handling.

Your task is to judge whether an assistant correctly followed the user's newer updated preference instead of an older outdated preference.

You will be given:
- An older preference memory
- A newer updated preference memory
- The user's current question
- A reference answer or expected behavior
- The assistant's answer

The correct assistant behavior is to use the newer updated preference when answering the user's current question. The older preference is outdated and may be misleading.

Evaluate only two things:

1. uses_latest_preference
Score 1 if the assistant's answer clearly follows the newer updated preference.
Score 0 if the assistant fails to use the newer preference, gives only generic advice when the newer preference is needed, or gives an answer inconsistent with the newer preference.

2. outdated_preference_contamination
Score 1 if the assistant's answer is improperly influenced by the older outdated preference.
Score 0 if the assistant does not rely on the outdated preference.

Mark outdated_preference_contamination = 1 if any of the following occurs:
- The answer recommends something aligned with the older preference but inconsistent with the newer preference.
- The answer mixes old and new preferences in a way that weakens or contradicts the newer preference.
- The answer treats the older preference as still active.
- The answer gives equal weight to old and new preferences when the newer one should override the older one.
- The answer explicitly mentions the older preference in a way that creates confusion or unnecessary personalization.

Important:
If the answer follows the newer preference but briefly acknowledges the older preference only to explain that the newer preference should guide the response, this is not contamination.
If the answer is generic and does not use either preference, uses_latest_preference should be 0, but outdated_preference_contamination should be 0.

Then compute:
valid_selection_pass = true only if uses_latest_preference == 1 and outdated_preference_contamination == 0.

Return JSON only.

Input:
{
  "older_preference_memory": {{older_preference_memory}},
  "newer_preference_memory": {{newer_preference_memory}},
  "user_question": {{user_question}},
  "reference_answer": {{reference_answer}},
  "expected_behavior": {{expected_behavior}},
  "old_preference_trap": {{old_preference_trap}},
  "assistant_answer": {{assistant_answer}}
}

Output JSON schema:
{
  "uses_latest_preference": 0 | 1,
  "outdated_preference_contamination": 0 | 1,
  "valid_selection_pass": true | false,
  "brief_rationale": "One short sentence explaining the judgment."
}
""".strip()


def _brief_exception_message(exc: BaseException, max_len: int = 220) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    msg = " ".join(str(msg).split())
    if len(msg) > max_len:
        return msg[: max_len - 3] + "..."
    return msg


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


def format_user_message_open_ended(row: dict[str, Any]) -> str:
    return str(row.get("user_question") or "").strip()


def reference_answer_from_row(row: dict[str, Any]) -> str:
    return str(row.get("reference_answer") or "").strip()


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
            try:
                self._init_db()
            except (OSError, sqlite3.Error) as exc:
                with self._metrics_lock:
                    self._disk_errors += 1
                print(f"[completion-cache] disk cache disabled path={self._db_path}: {exc}", flush=True)
                self._db_path = None

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
                extra_body={
                    "reasoning": {"enabled": False},
                    # "thinking": {"type": "disabled"},
                },
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
    prior_dialogue: str,
    context_label: str = "Earlier conversation",
) -> str:
    system = ANSWER_SYSTEM_PROMPT_BASE.format(
        model_name=model_name,
        current_date=current_date,
    )
    extra_instruction = os.environ.get("ANSWER_SYSTEM_EXTRA_INSTRUCTION", "").strip()
    if extra_instruction:
        system += f"\n\nAdditional instruction:\n{extra_instruction}\n"
    system += f"\n\n{context_label}:\n{prior_dialogue}\n"
    return system.strip()


def judge_user_payload(row: dict[str, Any], assistant_answer: str) -> str:
    payload = {
        "older_preference_memory": str(row.get("old_preference") or "").strip(),
        "newer_preference_memory": str(row.get("updated_preference") or "").strip(),
        "user_question": format_user_message_open_ended(row),
        "reference_answer": reference_answer_from_row(row),
        "expected_behavior": str(row.get("updated_preference_basis") or "").strip(),
        "old_preference_trap": str(row.get("old_preference_trap") or "").strip(),
        "assistant_answer": (assistant_answer or "").strip(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


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
            JUDGE_PREFRENCE_CHANGE_PROMPT,
            user_msg,
            rpm_limiter=rpm_limiter,
            max_retries=max_retries,
            retry_base_delay_sec=retry_base_delay_sec,
            retry_max_delay_sec=retry_max_delay_sec,
            completion_cache=completion_cache,
            temperature=0.0,
            cache_base_url=cache_base_url,
            cache_purpose="valid_memory_selection_judge",
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return {
            "uses_latest_preference": None,
            "outdated_preference_contamination": None,
            "valid_selection_pass": None,
            "brief_rationale": "",
            "judge_raw": "",
            "judge_parse_ok": False,
            "judge_error": f"{type(exc).__name__}: {exc}",
        }

    parsed = _extract_json_object(raw)
    uses_latest: int | None = None
    contamination: int | None = None
    valid_selection_pass: bool | None = None
    rationale = ""

    if parsed is not None:
        uses_latest = _coerce_binary_int(parsed.get("uses_latest_preference"))
        contamination = _coerce_binary_int(
            parsed.get("outdated_preference_contamination"))
        up = parsed.get("valid_selection_pass")
        if isinstance(up, bool):
            valid_selection_pass = up
        elif isinstance(up, str) and up.strip().lower() in {"true", "false"}:
            valid_selection_pass = up.strip().lower() == "true"
        if valid_selection_pass is None and uses_latest is not None and contamination is not None:
            valid_selection_pass = uses_latest == 1 and contamination == 0
        br = parsed.get("brief_rationale")
        if isinstance(br, str):
            rationale = br.strip()

    return {
        "uses_latest_preference": uses_latest,
        "outdated_preference_contamination": contamination,
        "valid_selection_pass": valid_selection_pass,
        "brief_rationale": rationale,
        "judge_raw": raw,
        "judge_parse_ok": parsed is not None
        and uses_latest is not None
        and contamination is not None
        and valid_selection_pass is not None,
        "judge_error": None,
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
            if str(row.get("task_type") or "").strip() != "valid_memory_selection":
                continue
            if not format_user_message_open_ended(row):
                continue
            if not format_prior_dialogue_from_row(row):
                continue
            if not str(row.get("old_preference") or "").strip():
                continue
            if not str(row.get("updated_preference") or "").strip():
                continue
            if not reference_answer_from_row(row):
                continue
            tasks.append(row)
    return tasks


def base_result_for_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": row.get("query_id"),
        "task_type": row.get("task_type"),
        "update_type": row.get("update_type"),
        "user_question": format_user_message_open_ended(row),
        "reference_answer": reference_answer_from_row(row),
        "old_preference": row.get("old_preference"),
        "updated_preference": row.get("updated_preference"),
        "old_preference_trap": row.get("old_preference_trap"),
        "updated_preference_basis": row.get("updated_preference_basis"),
    }


def empty_answer_block() -> dict[str, Any]:
    return {
        "answer_text": "",
        "judge": {
            "uses_latest_preference": None,
            "outdated_preference_contamination": None,
            "valid_selection_pass": None,
            "brief_rationale": "",
            "judge_raw": "",
            "judge_parse_ok": False,
            "judge_error": None,
        },
    }


def run_one(
    answer_client: OpenAI,
    judge_client: OpenAI,
    model_name: str,
    judge_model: str,
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
    answer_base_url: str = "",
    judge_base_url: str = "",
) -> dict[str, Any]:
    prior = format_prior_dialogue_from_row(row)
    user_msg = format_user_message_open_ended(row)
    lightmem_meta: dict[str, Any] | None = None
    if memory_eval_config is not None:
        memory_prior = format_memory_prior_dialogue_from_row(row)
        lightmem_ctx = build_baseline_context(
            memory_prior,
            user_msg,
            memory_eval_config,
            sample_key=row.get("query_id") or row.get("sample_index"),
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
        result_setting = "with_memory"
    else:
        system_with_memory = answer_system_prompt(
            model_name, current_date, prior_dialogue=prior)
        result_setting = "with_memory"

    try:
        answer_text = _chat_answer(
            answer_client,
            model_name,
            system_with_memory,
            user_msg,
            rpm_limiter=answer_rpm_limiter,
            max_retries=max_retries,
            retry_base_delay_sec=retry_base_delay_sec,
            retry_max_delay_sec=retry_max_delay_sec,
            completion_cache=completion_cache,
            temperature=0.2,
            cache_base_url=answer_base_url,
            cache_purpose="valid_memory_selection_answer",
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return base_result_for_row(row) | {
            "api_error": f"{type(exc).__name__}: {exc}",
            result_setting: empty_answer_block(),
        }

    judge = judge_answer(
        judge_client,
        judge_model,
        row,
        answer_text,
        rpm_limiter=judge_rpm_limiter,
        max_retries=max_retries,
        retry_base_delay_sec=retry_base_delay_sec,
        retry_max_delay_sec=retry_max_delay_sec,
        completion_cache=completion_cache,
        cache_base_url=judge_base_url,
    )

    result = base_result_for_row(row) | {
        result_setting: {
            "answer_text": answer_text,
            "judge": judge,
        }
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
    uses_latest = sum(int(j.get("uses_latest_preference") == 1) for j in valid)
    contamination = sum(int(j.get("outdated_preference_contamination") == 1) for j in valid)
    valid_selection_pass = sum(int(j.get("valid_selection_pass") is True) for j in valid)
    return {
        "with_memory": {
            "n_judged": n,
            "uses_latest_preference_avg": uses_latest / n if n else 0.0,
            "outdated_preference_contamination_avg": contamination / n if n else 0.0,
            "valid_selection_pass_avg": valid_selection_pass / n if n else 0.0,
            "uses_latest_preference_sum": uses_latest,
            "outdated_preference_contamination_sum": contamination,
            "valid_selection_pass_sum": valid_selection_pass,
            "judge_parse_failed": len(judges) - n,
            "judge_error_count": sum(1 for j in judges if j.get("judge_error")),
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", type=Path, default=TEST_JSONL)
    parser.add_argument("--output", type=Path, default=OUTPUT_RESULTS_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--base-url",
        default=DEFAULT_GENERATION_BASE_URL,
        help="OpenAI-compatible base URL for the generation model.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_GENERATION_API_KEY,
        help="Generation API key. Defaults to GENERATION_API_KEY / DEEPSEEK_API_KEY / API_KEY.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL_NAME)
    parser.add_argument(
        "--judge-base-url",
        default=DEFAULT_JUDGE_BASE_URL,
        help="OpenAI-compatible base URL for the judge model. Defaults to JUDGE_BASE_URL.",
    )
    parser.add_argument(
        "--judge-api-key",
        default=DEFAULT_JUDGE_API_KEY,
        help="Judge API key. Defaults to JUDGE_API_KEY / DEEPSEEK_JUDGE_API_KEY.",
    )
    parser.add_argument(
        "--judge-request-timeout",
        type=float,
        default=None,
        help="Per-request HTTP timeout for the judge model. Defaults to --request-timeout.",
    )
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--max-requests-per-minute", type=int, default=0)
    parser.add_argument(
        "--judge-max-requests-per-minute",
        type=int,
        default=None,
        help="Maximum judge requests per minute. Defaults to --max-requests-per-minute.",
    )
    parser.add_argument("--api-max-retries", type=int,
                        default=DEFAULT_API_MAX_RETRIES)
    parser.add_argument("--api-retry-base-delay", type=float,
                        default=DEFAULT_API_RETRY_BASE_DELAY_SEC)
    parser.add_argument("--api-retry-max-delay", type=float,
                        default=DEFAULT_API_RETRY_MAX_DELAY_SEC)
    parser.add_argument("--request-timeout", type=float,
                        default=DEFAULT_REQUEST_TIMEOUT_SEC)
    parser.add_argument(
        "--current-date",
        default=DEFAULT_CURRENT_DATE,
        help="Stable date inserted into answer system prompts. Defaults to EVAL_CURRENT_DATE or 2025-06-01.",
    )
    parser.add_argument("--no-completion-cache", action="store_true")
    parser.add_argument(
        "--completion-cache-path",
        type=Path,
        default=Path(os.environ.get("COMPLETION_CACHE_PATH", "output_data/completion_cache/valid_memory_selection_open.sqlite")),
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
        help="Enable a memory baseline for the with_memory branch.",
    )
    parser.add_argument("--memory-top-k", type=int, default=None)
    parser.add_argument("--memory-baseline-config", type=Path, default=None)
    parser.add_argument("--memory-config", type=Path, default=None)
    parser.add_argument(
        "--memory-save-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--memory-api-key",
                        default=os.environ.get("MEMORY_API_KEY", ""))
    parser.add_argument("--memory-base-url",
                        default=os.environ.get("MEMORY_BASE_URL", ""))
    parser.add_argument("--memory-llm-model", default="")
    parser.add_argument("--print-each-result",
                        action="store_true", default=PRINT_EACH_RESULT)
    parser.add_argument("--answer-preview-chars", type=int,
                        default=ANSWER_PREVIEW_CHARS)
    parser.add_argument(
        "--trace-api",
        action="store_true",
        help="Print [api-req] start/done lines to confirm HTTP requests are in flight.",
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
            "Pass --api-key or set "
            "GENERATION_API_KEY / DEEPSEEK_API_KEY / API_KEY."
        )
    if not judge_api_key:
        raise RuntimeError(
            "Pass --judge-api-key or set "
            "JUDGE_API_KEY / DEEPSEEK_JUDGE_API_KEY."
        )
    if not args.test_jsonl.is_file():
        raise FileNotFoundError(args.test_jsonl)

    tasks = load_eligible_tasks(args.test_jsonl, max(1, int(args.limit)))
    if not tasks:
        raise RuntimeError(
            f"{args.test_jsonl} contains no eligible valid_memory_selection samples.")

    request_timeout = float(args.request_timeout)
    client_kwargs: dict[str, Any] = {
        "base_url": args.base_url, "api_key": api_key}
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
            top_k=(int(args.memory_top_k)
                   if args.memory_top_k is not None else None),
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
            current_date,
            row,
            answer_rpm_limiter=answer_rpm_limiter,
            judge_rpm_limiter=judge_rpm_limiter,
            max_retries=max(1, int(args.api_max_retries)),
            retry_base_delay_sec=max(0.0, float(args.api_retry_base_delay)),
            retry_max_delay_sec=max(0.0, float(args.api_retry_max_delay)),
            completion_cache=completion_cache,
            memory_eval_config=memory_eval_config,
            answer_base_url=args.base_url,
            judge_base_url=judge_base_url,
        )

    results: list[dict[str, Any] | None] = [None] * len(tasks)
    n_workers = max(1, int(args.workers))
    result_setting = "with_memory"
    if memory_eval_config is not None:
        eval_sequence = "with_memory only (baseline retrieval)"
    else:
        eval_sequence = "with_memory only (raw dialogue)"
    print(
        f"Started {len(tasks)} tasks with thread-pool concurrency {n_workers}. "
        f"Each sample runs answer ({result_setting}) then judge; "
        f"sequence={eval_sequence}; "
        "progress may stay at 0% until the first sample completes.",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_job, row): idx for idx,
                   row in enumerate(tasks)}
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"preference update open ({result_setting}+judge)",
            unit="sample",
        ):
            results[futures[fut]] = fut.result()

    final_results: list[dict[str, Any]] = []
    for result in results:
        assert result is not None
        final_results.append(result)

    metrics = aggregate_metrics(final_results)
    cache_meta = completion_cache.stats() if completion_cache is not None else {
        "enabled": False}
    payload = {
        "task": "valid_memory_selection",
        "eval_mode": "open_ended_with_memory_only",
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
        "memory_top_k": int(args.memory_top_k) if args.memory_method else None,
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
            print(
                f"query_id={result.get('query_id')} question={result.get('user_question')}")
            block = result.get(result_setting) or {}
            judge = block.get("judge") or {}
            text = block.get("answer_text") or ""
            if preview and len(text) > preview:
                text = text[:preview] + "..."
            print(
                f"{result_setting}: uses_latest={judge.get('uses_latest_preference')} "
                f"outdated_contam={judge.get('outdated_preference_contamination')} "
                f"valid_selection_pass={judge.get('valid_selection_pass')}\n{text}"
            )

    selected_metrics = metrics[result_setting]
    print(
        f"Finished {len(final_results)} samples; results written to {args.output}\n"
        f"{result_setting}: uses_latest={selected_metrics['uses_latest_preference_avg']:.4f}, "
        f"outdated_contam={selected_metrics['outdated_preference_contamination_avg']:.4f}, "
        f"valid_selection_pass={selected_metrics['valid_selection_pass_avg']:.4f}"
    )


if __name__ == "__main__":
    main()
