# Source: https://github.com/andrewzhao06/structured-memory/blob/6b507608f1c0f6ebe523e040316ce8a7b160bfcd/src/benchmark/routing.py
"""Domain-routing baseline: classify the query into domains, inject only those partitions.

This module holds the *pure* routing logic (no API calls) so it is fully unit-testable:

  - the 11 canonical domains and the explicit PersistBench(9) -> domain(11) mapping
  - parsing a classifier's JSON domain list
  - selecting domains under each routing config (top-k / multi-label / +personal)
  - filtering a cached partition dict to the selected domains, reusing the exact
    fixed-partition rendering (work_planner._normalize_memories) downstream
  - the oracle helper: which domains a ground-truth memory should live in

The router differs from RAG (which scores individual memories by embedding similarity
and thresholds them) and from Fixed Partitions (which injects *all* domains): routing
operates at the level of whole domains and keeps every memory inside a selected domain.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

# The 11 fixed domains. Order here is the canonical render order (matches the
# insertion order used by the memory classifier / cached partition files).
CANONICAL_DOMAINS: tuple[str, ...] = (
    "personal", "education", "employment", "finance", "housing",
    "legal", "health", "schedule", "identity", "social", "romantic",
)
_DOMAIN_SET = frozenset(CANONICAL_DOMAINS)

# Taxonomy descriptions, verbatim from the Appendix E memory classifier, so the
# query-side and memory-side taxonomies are aligned. Used by the offline keyword
# classifier and mirrored in prompts/routing/query_domain_classifier.txt.
DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "health": "physical or mental health, medical conditions, treatments, medications, fitness, therapy",
    "identity": "core personal identity traits such as nationality, religion, gender identity, values, beliefs",
    "social": "non-romantic relationships and interactions with friends, family, acquaintances, or colleagues",
    "romantic": "intimate or romantic relationships including dating, partners, marriage, attraction, breakups",
    "personal": "hobbies, preferences, lifestyle choices, personality traits, interests",
    "education": "schooling, degrees, courses, academic history, tutoring, learning experiences",
    "employment": "jobs, work history, workplace experiences, colleagues, professional skills",
    "finance": "money, savings, income, expenses, debt, investments, banking, taxes",
    "housing": "home, residence, living situation, roommates, neighbors, rent, mortgage",
    "legal": "legal issues, contracts, court matters, rights, criminal record, official documents",
    "schedule": "appointments, routines, recurring events, time-based plans, daily habits",
}

# Explicit, reviewed mapping from PersistBench's 9 sample-level memory-domain labels
# to our 11 domains. Non-fuzzy and total: every PersistBench label maps to >= 1 domain.
# housing/schedule have no PersistBench source label; "Financial and Legal Matters"
# maps to BOTH finance and legal.
PERSISTBENCH_TO_DOMAINS: dict[str, tuple[str, ...]] = {
    "Health and Medical Information": ("health",),
    "Intimate and Romantic Relationships": ("romantic",),
    "Social and Relational Information": ("social",),
    "Professional and Work Life": ("employment",),
    "Educational and Formative Experiences": ("education",),
    "Self-Concept and Identity": ("identity",),
    "Personal Beliefs (Political, Religious, and Social)": ("identity",),
    "Private Thoughts and Journals": ("personal",),
    "Financial and Legal Matters": ("finance", "legal"),
}

# Placeholder memory-domain values that are NOT real domains (used by the
# beneficial and sycophancy subsets). map_persistbench_domain rejects these.
PLACEHOLDER_MEMORY_DOMAINS = frozenset({"positives", "sycophancy"})

# Explicit fallback when the classifier returns nothing usable. Deliberately a
# SINGLE domain (the broad catch-all), never all domains — falling back to "all"
# would make routing indistinguishable from Fixed Partitions and silently corrupt
# the comparison.
FALLBACK_DOMAINS: tuple[str, ...] = ("personal",)


def parse_domain_list(text: str) -> list[str]:
    """Parse a classifier response into a list of domain strings.

    Accepts a raw JSON list, or a JSON list embedded in surrounding prose/code
    fences. Returns [] on any malformed / non-list / non-string content — the
    caller then falls back explicitly via select_domains (never to "all").
    """
    if not isinstance(text, str):
        return []
    candidates: list[str] = []

    def _coerce(obj) -> list[str] | None:
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return list(obj)
        return None

    # 1) whole string is JSON
    try:
        coerced = _coerce(json.loads(text))
        if coerced is not None:
            return coerced
    except (json.JSONDecodeError, ValueError):
        pass

    # 2) first bracketed [...] block anywhere in the text
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            coerced = _coerce(json.loads(match.group(0)))
            if coerced is not None:
                return coerced
        except (json.JSONDecodeError, ValueError):
            pass
    return candidates


def _clean_ranked(ranked: Iterable[str]) -> list[str]:
    """Lower-case, validate against the taxonomy, drop invalids, dedupe by rank."""
    seen: set[str] = set()
    out: list[str] = []
    for d in ranked:
        if not isinstance(d, str):
            continue
        dd = d.strip().lower()
        if dd in _DOMAIN_SET and dd not in seen:
            seen.add(dd)
            out.append(dd)
    return out


def select_domains(
    ranked: Iterable[str],
    *,
    mode: str,
    k: int | None = None,
    always_include_personal: bool = False,
) -> list[str]:
    """Select the routed domains from a classifier's ranked domain list.

    mode="topk"       -> first ``k`` valid domains (fewer than k if the classifier
                         proposes fewer; invalid names are dropped first).
    mode="multilabel" -> every valid domain the classifier proposed.

    always_include_personal appends "personal" if absent (the +personal variant).
    If nothing valid remains, returns FALLBACK_DOMAINS (a single domain) — NEVER
    all domains. Returned in rank order; render order is applied later by
    build_routed_memories.
    """
    valid = _clean_ranked(ranked)

    if mode == "topk":
        if k is None or k < 1:
            raise ValueError(f"topk mode requires k >= 1, got {k!r}")
        chosen = valid[:k]
    elif mode == "multilabel":
        chosen = list(valid)
    else:
        raise ValueError(f"Unknown routing mode {mode!r} (expected 'topk' or 'multilabel')")

    if always_include_personal and "personal" not in chosen:
        chosen = chosen + ["personal"]

    if not chosen:
        return list(FALLBACK_DOMAINS)
    return chosen


def build_routed_memories(
    partition: dict[str, list[str]],
    selected: Iterable[str],
) -> dict[str, list[str]]:
    """Filter a cached partition dict to the selected domains.

    Preserves the partition's own key order (canonical render order) so the
    resulting block is byte-identical to Fixed Partitions with the unselected
    domains removed. Memories are copied verbatim — never mutated or reordered
    within a domain.
    """
    sel = {s.strip().lower() for s in selected}
    return {cat: list(mems) for cat, mems in partition.items() if cat in sel}


def flatten_partition(partition: dict[str, list[str]]) -> list[str]:
    """Flatten a full partition dict into the complete flat memory list.

    Used for ``full_memories`` so the judge scores against the complete pool
    (judgment.py prefers full_memories when present), exactly like RAG.
    """
    return [mem for mems in partition.values() for mem in mems]


def count_memories(partition: dict[str, list[str]]) -> int:
    """Number of memories retained by a (possibly filtered) partition dict."""
    return sum(len(mems) for mems in partition.values())


def map_persistbench_domain(label: str) -> list[str]:
    """Map a PersistBench 9-category memory-domain label to our domain(s).

    Raises KeyError loudly on any unmapped or placeholder label so a new/unseen
    label can never be silently fuzzy-matched or dropped.
    """
    if label in PLACEHOLDER_MEMORY_DOMAINS:
        raise KeyError(
            f"{label!r} is a placeholder memory_domain (beneficial/sycophancy subset), "
            f"not a real PersistBench domain; it has no taxonomy mapping."
        )
    if label not in PERSISTBENCH_TO_DOMAINS:
        raise KeyError(
            f"Unmapped PersistBench memory-domain label: {label!r}. "
            f"Add it to routing.PERSISTBENCH_TO_DOMAINS (no fuzzy matching)."
        )
    return list(PERSISTBENCH_TO_DOMAINS[label])
