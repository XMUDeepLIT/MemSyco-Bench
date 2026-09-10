"""Mem0 + Dynamic Structured Memory (custom partition + query routing).

MemZero still constructs and stores memories. This module reorganizes the
candidate pool (MemZero ``get_all`` when available, else the retrieved set)
and keeps every memory in the query-selected domains.

Upstream Structured Memory pieces live beside this file (arXiv:2608.08300):
``structured_memory_routing.py``, ``query_domain_classifier.txt``, and the
custom-category partition prompt/validation copied below.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ..base import BaselineEvalConfig
from .llm import chat_completion
from .structured_memory_routing import (
    CANONICAL_DOMAINS,
    FALLBACK_DOMAINS,
    build_routed_memories,
    parse_domain_list,
    select_domains,
)

QUERY_CLASSIFIER_PROMPT_FILE = Path(__file__).resolve().parent / "query_domain_classifier.txt"

# Copied from partition_memories_custom_categories.py
DEFAULT_CATEGORIES = [
    "personal",
    "education",
    "employment",
    "finance",
    "housing",
    "legal",
    "health",
    "schedule",
    "identity",
    "social",
    "romantic",
]
CATEGORIES = DEFAULT_CATEGORIES
_CUSTOM_CAT_MAX_NEW = 2
_CUSTOM_CAT_MIN_LEN = 3
_CUSTOM_CAT_MAX_LEN = 12
_CUSTOM_CAT_RE = re.compile(r"^[a-z][a-z_]{1,}[a-z]$")
PARTITION_TEMPERATURE = 0.7
MAX_RETRIES = 5

# Copied verbatim from partition_memories_custom_categories.py (SYSTEM_PROMPT).
SYSTEM_PROMPT = """\
You are a memory classifier. Your task is to sort a list of personal memories
into exactly one category each.

Predefined categories — read the descriptions carefully before classifying:

personal     – the broadest catch-all for individual life outside structured domains:
              hobbies, sports, games, cooking, travel, leisure,
              entertainment, arts & crafts, music, reading, fashion, technology
              interests, outdoor activities, pets, gardening, philosophy, personal
              reflections, lifestyle choices, personality traits, values, opinions,
              volunteering, and any other interest or pastime.
health       – physical or mental health, medical conditions, treatments,
              medications, fitness goals, therapy, disabilities, diet for health.
identity     – core self-concept: nationality, ethnicity, religion, spiritual
              practice, gender identity, sexuality, political ideology, deeply
              held beliefs that define who the person is.
social       – relationships and interactions with any other person who is NOT
              a romantic partner: family (parents, siblings, children, extended
              family), friends, neighbours, acquaintances, colleagues (socially).
romantic     – intimate or romantic relationships: dating, partners, marriage,
              attraction, breakups, divorce, jealousy, affection.
education    – schooling, degrees, courses, academic history, tutoring, exams,
              certifications, formal or informal learning experiences.
employment   – jobs, work history, career, workplace dynamics, colleagues
              (professionally), professional skills, freelance/business ventures.
finance      – money, savings, income, expenses, debt, investments, banking, taxes,
              insurance, financial goals.
housing      – home, residence, living situation, roommates, neighbours (housing),
              rent, mortgage, moving, home maintenance.
legal        – legal issues, contracts, court matters, rights, criminal record,
              official government documents, immigration status.
schedule     – appointments, routines, recurring events, time-based plans,
              deadlines, daily habits, reminders.

When to create a custom category:
A custom category is justified ONLY when ALL of the following are true:
 1. Multiple memories in this batch form a coherent, substantial life domain.
 2. That domain is genuinely absent from every predefined category above, or any new category already created.
 3. The domain cannot reasonably be called a sub-topic of one of the default or newly introduced categories.

Do NOT create custom categories for: lifestyle, leisure, entertainment,
sports, cooking, fashion, technology, gardening, garden, transport,
transportation, vehicles, arts, music, philosophy, history, language, community,
activism, environment, research, productivity, creative_work, interest, pastime, preference,
spiritual_practice, volunteer, volunteering, or any near-synonym of an existing category.
All of these belong in another predefined category.

If you do create a custom category:
 • Lowercase letters and underscores only, 3–15 characters.
 • Choose a single canonical name — do NOT create variants of the same concept
   (e.g. pick "travel" not both "travel" and "trips", or "allirgies" and "diet").
 • Create at most 2 custom categories per response.
 • Only include the key if it has at least one memory in it.

Rules:
1. Each memory must appear in exactly one category.
2. Do not drop or duplicate memories.
3. If a memory fits multiple categories, choose the most specific predefined one.
4. All 11 predefined keys must always be present (use [] if empty).
5. Custom category keys appear after the predefined ones.
6. Do not modify the memory text.

Return ONLY a single-line JSON object. Example with one justified custom category:

{"health": [...], "identity": [...], "social": [...], "romantic": [...], "personal": [...], "education": [...], "employment": [...], "finance": [...], "housing": [...], "legal": [...], "schedule": [...], "travel": [...]}

Omit any custom-category key if it has no memories.
"""

ROUTING_CONFIGS: dict[str, dict[str, Any]] = {
    "top1": {"mode": "topk", "k": 1, "always_include_personal": False},
    "top2": {"mode": "topk", "k": 2, "always_include_personal": False},
    "top3": {"mode": "topk", "k": 3, "always_include_personal": False},
    "multilabel": {"mode": "multilabel", "k": None, "always_include_personal": False},
    "multilabel_personal": {"mode": "multilabel", "k": None, "always_include_personal": True},
}


def extract_json_from_response(content: str) -> dict[str, Any]:
    """Copied from structured-memory ``src/benchmark/utils.py``."""
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract valid JSON from response. Content: {content[:500]}"
    )


def _is_valid_custom_category(name: str) -> bool:
    """Return True if name is an acceptable model-created category label."""
    return (
        name not in DEFAULT_CATEGORIES
        and _CUSTOM_CAT_MIN_LEN <= len(name) <= _CUSTOM_CAT_MAX_LEN
        and _CUSTOM_CAT_RE.match(name) is not None
    )


def _canonicalize_custom_name(name: str) -> str:
    """Normalize morphological variants to a canonical root for deduplication."""
    if name.endswith("ing"):
        root = name[:-3]
        if len(root) >= 4:
            return root
    if name.endswith("tion") and len(name) > 7:
        return name[:-4]
    if name.endswith("s") and len(name) > 5:
        return name[:-1]
    return name


def _validate_partition(memories: list[str], raw: dict) -> dict[str, list[str]]:
    """Ensure every input memory appears exactly once in the result."""
    result: dict[str, list[str]] = {cat: [] for cat in DEFAULT_CATEGORIES}
    placed: set[str] = set()

    for cat in DEFAULT_CATEGORIES:
        for mem in raw.get(cat, []):
            if mem in memories and mem not in placed:
                result[cat].append(mem)
                placed.add(mem)

    custom_accepted = 0
    seen_roots: dict[str, str] = {}
    for cat, items in raw.items():
        if custom_accepted >= _CUSTOM_CAT_MAX_NEW:
            break
        if not _is_valid_custom_category(cat):
            continue
        root = _canonicalize_custom_name(cat)
        valid_items = [m for m in items if m in memories and m not in placed]
        if not valid_items:
            continue
        if root in seen_roots:
            canonical = seen_roots[root]
            result[canonical].extend(valid_items)
        else:
            result[cat] = valid_items
            seen_roots[root] = cat
            custom_accepted += 1
        for mem in valid_items:
            placed.add(mem)

    for mem in memories:
        if mem not in placed:
            result["personal"].append(mem)

    return result


def _personal_fallback(memories: list[str]) -> dict[str, list[str]]:
    fallback = {cat: [] for cat in CATEGORIES}
    fallback["personal"] = list(memories)
    return fallback


def _memory_text(mem: dict[str, Any]) -> str:
    return str(mem.get("used_content") or mem.get("content") or "").strip()


def _max_memories(eval_config: BaselineEvalConfig) -> int:
    raw = (eval_config.options or {}).get("max_memories", 40)
    return max(1, int(raw))


def _routing_config(eval_config: BaselineEvalConfig) -> dict[str, Any]:
    name = str((eval_config.options or {}).get("route_mode") or "multilabel").strip().lower()
    if name not in ROUTING_CONFIGS:
        raise ValueError(
            f"Unknown control.route_mode={name!r}. "
            f"Expected one of: {', '.join(ROUTING_CONFIGS)}"
        )
    return ROUTING_CONFIGS[name]


def _query_classifier_prompt(query: str, extra_domains: list[str]) -> str:
    template = QUERY_CLASSIFIER_PROMPT_FILE.read_text(encoding="utf-8")
    prompt = template.replace("{query}", query)
    if not extra_domains:
        return prompt
    extra_lines = "\n".join(
        f"{name}       – sample-specific custom partition from memory classification"
        for name in extra_domains
    )
    needle = "Rules:\n1. Choose every domain"
    addition = (
        "Additional sample-specific categories created while partitioning these memories "
        "(use the exact names if relevant):\n"
        f"{extra_lines}\n\n"
    )
    if needle in prompt:
        return prompt.replace(needle, addition + needle, 1)
    return prompt + "\n" + addition


def _candidate_pool(
    memories: list[dict[str, Any]],
    layer: Any | None,
    limit: int,
) -> list[dict[str, Any]]:
    if layer is None:
        return memories[:limit]
    try:
        mem_layer = getattr(layer, "memory_layer", None)
        user_id = getattr(getattr(layer, "config", None), "user_id", None)
        if mem_layer is None or not user_id:
            return memories[:limit]
        existing = mem_layer.get_all(user_id=user_id)
        if isinstance(existing, dict):
            results = existing.get("results") or existing.get("memories") or existing.get("data") or []
        elif isinstance(existing, list):
            results = existing
        else:
            results = []
        pool: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            content = str(item.get("memory") or item.get("content") or "").strip()
            if not content:
                continue
            pool.append(
                {
                    "content": content,
                    "used_content": content,
                    "metadata": {k: v for k, v in item.items() if k != "memory"},
                }
            )
            if len(pool) >= limit:
                break
        return pool or memories[:limit]
    except Exception as exc:
        print(f"[DynPartition] get_all failed, using retrieved memories: {exc}", flush=True)
        return memories[:limit]


def _classify_partition(eval_config: BaselineEvalConfig, memories: list[str]) -> dict[str, list[str]]:
    if not memories:
        return {cat: [] for cat in CATEGORIES}
    user_message = json.dumps(memories, ensure_ascii=False)
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            content = chat_completion(
                eval_config,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=PARTITION_TEMPERATURE,
            )
            raw = extract_json_from_response(content)
            if not isinstance(raw, dict):
                raise ValueError(f"partition JSON must be an object, got {type(raw).__name__}")
            return _validate_partition(memories, raw)
        except Exception as exc:
            last_exc = exc
            if attempt == MAX_RETRIES - 1:
                print(
                    f"[DynPartition] partition failed after {MAX_RETRIES} attempts: {exc}; "
                    "falling back to personal",
                    flush=True,
                )
                return _personal_fallback(memories)
            time.sleep(2 ** attempt)
    print(f"[DynPartition] partition failed: {last_exc}; falling back to personal", flush=True)
    return _personal_fallback(memories)


def _custom_category_names(partition: dict[str, list[str]]) -> list[str]:
    names: list[str] = []
    for cat, items in partition.items():
        if cat in DEFAULT_CATEGORIES or not items:
            continue
        if cat not in names:
            names.append(cat)
    return names


def _select_with_custom(
    ranked: list[str],
    *,
    cfg: dict[str, Any],
    custom_names: list[str],
) -> list[str]:
    custom_set = {name.strip().lower() for name in custom_names}
    canonical_set = {str(name).strip().lower() for name in CANONICAL_DOMAINS}
    custom_hits: list[str] = []
    seen: set[str] = set()
    has_canonical = False
    for name in ranked:
        if not isinstance(name, str):
            continue
        key = name.strip().lower()
        if key in canonical_set:
            has_canonical = True
        if key in custom_set and key not in seen:
            custom_hits.append(key)
            seen.add(key)

    if has_canonical:
        selected = select_domains(
            ranked,
            mode=cfg["mode"],
            k=cfg["k"],
            always_include_personal=cfg["always_include_personal"],
        )
    else:
        selected = []
        if cfg["always_include_personal"]:
            selected = ["personal"]

    for name in custom_hits:
        if name not in selected:
            selected.append(name)
    if not selected:
        return list(FALLBACK_DOMAINS)
    return selected


def _route_domains(
    eval_config: BaselineEvalConfig,
    query: str,
    custom_names: list[str],
) -> list[str]:
    cfg = _routing_config(eval_config)
    user = _query_classifier_prompt(query, custom_names)
    ranked: list[str] = []
    for attempt in range(MAX_RETRIES):
        try:
            content = chat_completion(
                eval_config,
                messages=[{"role": "user", "content": user}],
                temperature=0,
            )
            ranked = parse_domain_list(content)
            break
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                print(
                    f"[DynPartition] routing failed after {MAX_RETRIES} attempts: {exc}; "
                    f"empty -> fallback {list(FALLBACK_DOMAINS)}",
                    flush=True,
                )
                ranked = []
                break
            time.sleep(2 ** attempt)
    return _select_with_custom(ranked, cfg=cfg, custom_names=custom_names)


def _take_original(text: str, unused: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    bucket = unused.get(text)
    if bucket:
        return bucket.pop(0)
    return {"content": text, "used_content": text, "metadata": {}}


def filter_memories(
    query: str,
    memories: list[dict[str, Any]],
    eval_config: BaselineEvalConfig,
    *,
    layer: Any | None = None,
) -> list[dict[str, Any]]:
    if not memories and layer is None:
        return []
    pool = _candidate_pool(memories, layer, _max_memories(eval_config))
    texts = [_memory_text(mem) for mem in pool]
    texts = [text for text in texts if text]
    if not texts:
        return []

    partition = _classify_partition(eval_config, texts)
    custom_names = _custom_category_names(partition)
    selected = _route_domains(eval_config, query, custom_names)
    routed = build_routed_memories(partition, selected)
    # Official routing also keeps custom keys if they were selected.
    for name in selected:
        if name in partition and name not in routed:
            routed[name] = list(partition[name])

    unused: dict[str, list[dict[str, Any]]] = {}
    for mem in pool:
        unused.setdefault(_memory_text(mem), []).append(mem)

    kept: list[dict[str, Any]] = []
    for domain, items in routed.items():
        for text in items:
            item = dict(_take_original(text, unused))
            meta = dict(item.get("metadata") or {})
            meta["control"] = {
                "method": "MemZero+DynPartition",
                "partition": domain,
                "routed_domains": selected,
            }
            item["metadata"] = meta
            kept.append(item)
    return kept
