# Evaluation

Use the unified runner for all five benchmark tasks:

```bash
python evaluation/run_task.py TASK [--optimized] [TASK OPTIONS]
```

Canonical task names:

```text
objective_fact_judgment
contextual_scope_control
memory_evidence_conflict
valid_memory_selection
personalized_memory_use
```

Examples:

```bash
python evaluation/run_task.py personalized_memory_use --limit 5
python evaluation/run_task.py memory_evidence_conflict --optimized --memory-method MemZero --limit 5
python evaluation/run_task.py objective_fact_judgment --help
```

`--optimized` enables disk-backed memory reuse and forces one worker per task-method job. The repository-level `scripts/eval_baseline_opt_v2_short.sh` driver uses this mode automatically when running task and method matrices.

The `task_*.py` modules contain task-specific prompts, scoring, and result aggregation. They are implementation modules behind `run_task.py`; new users generally do not need to invoke them directly.

Validate the complete task matrix without sending API requests:

```bash
./scripts/eval_baseline_opt_v2_short.sh --dry-run --methods RawDialogue --limit 1
```
