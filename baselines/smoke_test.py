"""End-to-end smoke tests for memory-baseline adapters.

These are lightweight, real (non-mock) checks that prove an adapter can build a
``BaselineContext`` end-to-end against the live services it depends on. They are
meant to be run manually before trusting a full evaluation run.

Usage:

    # default: run every smoke test that has its services available
    python -m baselines.smoke_test

    # a single method
    python -m baselines.smoke_test --method MemoryBank

Exit codes:
    0  every selected smoke test passed (or was skipped because its service was
       intentionally unavailable and --strict was not set)
    1  at least one selected smoke test failed
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines import (  # noqa: E402
    BaselineContext,
    build_baseline_context,
    build_baseline_eval_config,
)


SUPPORTED_METHODS = ("MemoryBank",)

# A tiny but information-rich dialogue so retrieval has something to find.
PRIOR_DIALOGUE = (
    "User: Hi! I'm Mia, a backend engineer based in Berlin.\n"
    "Assistant: Nice to meet you, Mia. What are you working on?\n"
    "User: Mostly Go services. I'm vegetarian and allergic to peanuts.\n"
    "Assistant: Got it, I'll remember that.\n"
    "User: I also prefer dark-themed UIs and I run every morning.\n"
    "Assistant: Noted - dark themes and morning runs.\n"
)
USER_QUESTION = "What dietary restrictions does the user have?"


class SmokeResult:
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"

    def __init__(self, method: str, status: str, detail: str) -> None:
        self.method = method
        self.status = status
        self.detail = detail

    def line(self) -> str:
        return f"[{self.status:4}] {self.method}: {self.detail}"


def _check_context(method: str, ctx: BaselineContext) -> SmokeResult:
    if not isinstance(ctx, BaselineContext):
        return SmokeResult(method, SmokeResult.FAIL, f"adapter returned {type(ctx)!r}, not BaselineContext")
    text = (ctx.context_text or "").strip()
    n_mem = len(ctx.retrieved_memories or [])
    if not text:
        return SmokeResult(method, SmokeResult.FAIL, "empty context_text")
    return SmokeResult(
        method,
        SmokeResult.PASS,
        f"context_text={len(text)} chars, retrieved_memories={n_mem}, user_id={ctx.user_id!r}",
    )


def smoke_memorybank(args: argparse.Namespace) -> SmokeResult:
    method = "MemoryBank"
    eval_config = build_baseline_eval_config(
        method=method,
        top_k=args.top_k,
        api_key=args.api_key or None,
        base_url=args.base_url or None,
    )
    try:
        ctx = build_baseline_context(PRIOR_DIALOGUE, USER_QUESTION, eval_config)
    except Exception as exc:  # noqa: BLE001
        return SmokeResult(
            method,
            SmokeResult.FAIL,
            f"build_baseline_context raised {type(exc).__name__}: {exc}. "
            "Check MEMORY_EMBEDDING_BASE_URL / MEMORY_EMBEDDING_MODEL (and the memory LLM "
            "credentials unless MEMORYBANK_DISABLE_SUMMARY=1).",
        )
    return _check_context(method, ctx)


_RUNNERS = {
    "MemoryBank": smoke_memorybank,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--method",
        action="append",
        choices=SUPPORTED_METHODS,
        help="Method to smoke-test (repeatable). Defaults to all supported methods.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--base-url", default="", help="Override the generic memory base_url.")
    parser.add_argument("--api-key", default="", help="Override the generic memory api_key.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reserved for future smoke tests that may skip when services are unavailable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    methods = args.method or list(SUPPORTED_METHODS)

    print("=" * 60)
    print("baselines smoke test")
    print(f"methods: {', '.join(methods)}")
    print("=" * 60)

    results: list[SmokeResult] = []
    for method in methods:
        print(f"\n--- {method} ---", flush=True)
        try:
            result = _RUNNERS[method](args)
        except Exception as exc:  # noqa: BLE001
            result = SmokeResult(method, SmokeResult.FAIL, f"unexpected error: {type(exc).__name__}: {exc}")
        print(result.line(), flush=True)
        results.append(result)

    print("\n" + "=" * 60)
    print("summary")
    for result in results:
        print("  " + result.line())
    print("=" * 60)

    return 1 if any(r.status == SmokeResult.FAIL for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
