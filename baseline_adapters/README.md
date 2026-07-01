# Baseline Adapters

This directory is the evaluation-side integration layer for memory baselines.

- `methods/` keeps only the vendored source code needed by active baseline adapters.
- `baseline_adapters/` exposes a small common interface used by the benchmark scripts.
- Active adapters: `lightmem.py`, `naive_rag.py`, `amem.py`, `memzero.py`, `memorybank.py`,
  `supermemory.py`, and `memgpt.py`.
- Each baseline also gets an evaluation config in `baseline_adapters/configs/<Method>.json`.
- Method-specific default configs stay with their source repository when available, for example
  `methods/LightMem/src/lightmem/memory_toolkits/configs/*.json`.
- A benchmark run can override the evaluation config with `--memory-baseline-config`.
- A benchmark run can override a method's native config with `--memory-config`.

Supported memory methods:

`MemZero`, `NaiveRAG`, `A-MEM`, `LightMem`, `MemoryBank`, `Supermemory`, `MemGPT`

The benchmark scripts should call only:

```python
from baseline_adapters import BASELINE_METHODS, build_baseline_context, build_baseline_eval_config
```

The benchmark scripts build a `BaselineEvalConfig` from the per-method JSON first, then apply
command-line overrides such as `--memory-top-k`, `--memory-config`, `--memory-api-key`, and
`--memory-base-url`.

Each adapter receives:

```text
prior_dialogue + user_question + BaselineEvalConfig
```

and returns a `BaselineContext` containing the text to inject into the answer model prompt plus
the structured retrieved memories to save in the result JSON.
