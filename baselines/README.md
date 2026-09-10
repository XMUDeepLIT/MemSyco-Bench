# Baselines

This directory is the first-class integration layer for memory-related evaluation
settings in MemSyco-Bench.

## Nine peer evaluation settings

| Setting | Module | Vendor code |
|---------|--------|-------------|
| `NoMemory` | evaluation runner | — |
| `RawDialogue` | evaluation runner | — |
| `MemZero` | `memzero.py` | `toolkit/vendor/` |
| `A-MEM` | `amem.py` | `toolkit/vendor/` |
| `NaiveRAG` | `naive_rag.py` | `toolkit/vendor/` |
| `LightMem` | `lightmem.py` | `lightmem/vendor/` |
| `MemoryBank` | `memorybank.py` | `memorybank/vendor/` |
| `MemGPT` | `memgpt.py` | — |
| `Supermemory` | `supermemory.py` | — |

## Public API

```python
from baselines import BASELINE_METHODS, build_baseline_context, build_baseline_eval_config
```

Each memory baseline has a default evaluation config in `configs/<Method>.json`.
Override at runtime with `--memory-baseline-config` or `--memory-config`.

## Extra MemZero control baselines

These are **not** in the default nine. Construction and the on-disk store stay
`MemZero`; a control runs after retrieval. Pass them explicitly:

```bash
./scripts/fetch_memgate_checkpoint.sh   # only needed for MemZero+MemGate
./scripts/run_benchmark.sh \
  --methods MemZero+SelfReCheck,MemZero+MemGate,MemZero+DynPartition \
  --tasks objective_fact_judgment \
  --limit 5
```

| Setting | Control | Notes |
|---------|---------|-------|
| `MemZero+SelfReCheck` | OP-Bench Self-ReCheck (arXiv:2601.13722) | One batched LLM keep/drop call per sample. Fail-open (keep all on parse error). |
| `MemZero+MemGate` | MemGate (arXiv:2606.06054) | Inference in `controls/memgate_big.py`. Settings live in `configs/MemZero+MemGate.json` (`control.device`, `control.threshold`, `control.checkpoint`). Encoder is 384-d `all-MiniLM-L6-v2`, not MemZero `bge-m3`. |
| `MemZero+DynPartition` | Structured Memory custom partition + query routing (arXiv:2608.08300) | `controls/dyn_partition.py` plus `structured_memory_routing.py` and `query_domain_classifier.txt`. Settings live in `configs/MemZero+DynPartition.json` (`control.max_memories`, `control.route_mode`: `multilabel` default; also `top1`/`top2`/`top3`/`multilabel_personal`). Empty classifier output falls back to `personal`, not all domains. |

Output files use distinct slugs (`memzero_self_recheck`, `memzero_memgate`,
`memzero_dyn_partition`). The memory save root and outer lock still use
`memzero`, so they can reuse an existing MemZero store and do not run
construction in parallel with `MemZero`.

Control LLM calls use `MEMORY_API_KEY` / `MEMORY_BASE_URL` / `MEMORY_LLM_MODEL`
with thinking disabled. MemGate needs `./scripts/fetch_memgate_checkpoint.sh` and
`torch` + `sentence-transformers` (already in `requirements-memory-baselines.txt`).

These three methods do not create a separate memory store. They reuse
`output_data/memory_stores/<task>/memzero/` and write their own result files
under `output_data/runs/<task>/`. Extra local files: MemGate's
`output_data/checkpoints/Memgate.pt` plus a HuggingFace MiniLM cache on first
load. Self-ReCheck and DynPartition only add extra memory-side LLM calls;
retrieved rows in the result JSON may include `metadata.control`.

## Layout

```text
baselines/
  toolkit/vendor/      shared MemZero / A-MEM / NaiveRAG toolkit code
  lightmem/vendor/     vendored native lightmem package (pip editable install)
  memorybank/vendor/   vendored MemoryBank-SiliconFriend prompts/helpers
  controls/            extra MemZero post-retrieval filters
  configs/             per-method evaluation defaults
```
