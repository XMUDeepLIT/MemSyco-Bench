# Baseline Adapters

This directory is the evaluation-side integration layer for memory baselines.

- `methods/` keeps only the vendored source code needed by active baseline adapters.
- `baseline_adapters/` exposes a small common interface used by the benchmark scripts.
- Each baseline gets its own adapter module, for example `lightmem.py`, `full_context.py`,
  `naive_rag.py`, `amem.py`, `memzero.py`, `langmem.py`, `letta.py`, `zep_cloud.py`, and
  `zep_legacy.py`.
- Each baseline also gets an evaluation config in `baseline_adapters/configs/<Method>.json`.
- Method-specific default configs stay with their source repository when available, for example
  `methods/LightMem/src/lightmem/memory_toolkits/configs/*.json`.
- A benchmark run can override the evaluation config with `--memory-baseline-config`.
- A benchmark run can override a method's native config with `--memory-config`.

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

## Letta adapter

- `Letta` (formerly MemGPT) talks to a running Letta server over its REST API (no extra
  Python dependency), mirroring the Zep adapters.
- Start the server first and ensure it can reach an LLM + embedding provider:

  ```bash
  pip install letta
  letta server   # serves on http://localhost:8283
  ```

- Point the adapter at the server with `--memory-base-url` or `LETTA_BASE_URL` (default
  `http://localhost:8283`). Pass an access token with `--memory-api-key`,
  `LETTA_API_KEY`, or `LETTA_SERVER_PASSWORD` only when the server enforces one.
- Choose the agent model/embedding handles with `LETTA_MODEL` (default
  `openai/gpt-4o-mini`) and `LETTA_EMBEDDING_MODEL` (default
  `openai/text-embedding-3-small`).
- `LETTA_INGEST_MODE` controls how prior dialogue enters memory: `messages` (default,
  feeds the transcript to the agent so its own LLM loop edits core/archival memory) or
  `archival` (cheap RAG-style direct passage inserts). Retrieval combines the agent's core
  memory blocks with a top-k archival-memory semantic search for the question.

Example:

```bash
python evaluation/run_task.py objective_fact_judgment --memory-method Letta
```

## Zep adapters

- `ZepCloud` uses the official `zep-cloud` Python SDK. Pass the key with `--memory-api-key`
  or `ZEP_API_KEY`; pass `--memory-base-url` only when targeting a non-default Zep Cloud
  endpoint.
- `ZepLegacy` targets the deprecated Community Edition service. Deploy a Zep CE
  server yourself (see https://github.com/getzep/zep), set a non-empty `api_secret`,
  and pass that same value with `--memory-api-key` or `ZEP_API_SECRET`. The default
  API URL is `http://localhost:8000/api/v2`.

Example:

```bash
python evaluation/run_task.py objective_fact_judgment --memory-method ZepCloud
python evaluation/run_task.py objective_fact_judgment --memory-method ZepLegacy
```
