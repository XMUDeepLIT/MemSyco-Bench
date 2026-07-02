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

## Layout

```text
baselines/
  toolkit/vendor/      shared MemZero / A-MEM / NaiveRAG toolkit code
  lightmem/vendor/     vendored native lightmem package (pip editable install)
  memorybank/vendor/   vendored MemoryBank-SiliconFriend prompts/helpers
  configs/             per-method evaluation defaults
```
