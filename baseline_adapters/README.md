# Baseline Adapters

This directory is the **first-class integration layer** for all memory-related evaluation
settings in MemSyco-Bench. The benchmark runner talks only to adapters here—not to `methods/`
directly.

## Nine peer evaluation settings

| Setting | Adapter | Notes |
|---------|---------|-------|
| `NoMemory` | evaluation runner | no prior dialogue injected |
| `RawDialogue` | evaluation runner | full relevant dialogue as context |
| `MemZero` | `memzero.py` | toolkit layer via vendored `memory_toolkits` |
| `A-MEM` | `amem.py` | toolkit layer via vendored `memory_toolkits` |
| `NaiveRAG` | `naive_rag.py` | toolkit layer via vendored `memory_toolkits` |
| `LightMem` | `lightmem.py` | native LightMem memory system |
| `MemoryBank` | `memorybank.py` | in-repo re-implementation |
| `MemGPT` | `memgpt.py` | in-repo re-implementation |
| `Supermemory` | `supermemory.py` | in-repo re-implementation |

Each memory baseline also has a default evaluation config in `configs/<Method>.json`.
Override at runtime with `--memory-baseline-config` or `--memory-config`.

## Public API

Benchmark scripts should call only:

```python
from baseline_adapters import BASELINE_METHODS, build_baseline_context, build_baseline_eval_config
```

Each adapter receives:

```text
prior_dialogue + user_question + BaselineEvalConfig
```

and returns a `BaselineContext` with the text injected into the answer-model prompt plus
structured retrieved memories for the result JSON.

Command-line overrides such as `--memory-top-k`, `--memory-api-key`, and `--memory-base-url`
are applied on top of the per-method JSON defaults.

## Relationship to `methods/`

Four baselines (`MemZero`, `A-MEM`, `NaiveRAG`, `LightMem`) depend on vendored upstream code
under `methods/LightMem/`. That tree is a **dependency checkout**, not the benchmark core.
See [`methods/README.md`](../methods/README.md) for details.
