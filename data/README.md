# MemSyco-Bench Data Card

MemSyco-Bench evaluates whether a memory-augmented agent gives retrieved user memory the appropriate decision authority. The release contains 1,550 examples across five task categories.

## Tasks

| Task | Memory policy | Samples | File |
| --- | --- | ---: | --- |
| Objective Fact Judgment | `ignore_as_evidence` | 300 | `objective_fact_judgment.jsonl` |
| Contextual Scope Control | `constrain_to_scope` | 300 | `contextual_scope_control.jsonl` |
| Memory-Evidence Conflict | `defer_to_evidence` | 300 | `memory_evidence_conflict.jsonl` |
| Valid Memory Selection | `update` | 350 | `valid_memory_selection.jsonl` |
| Personalized Memory Use | `use` | 300 | `personalized_memory_use.jsonl` |

Counts and SHA-256 checksums are recorded in [`manifest.json`](manifest.json). The machine-readable row schema is [`schema.json`](schema.json).

## Unified Row Format

Every JSONL row has the same seven top-level fields:

```json
{
  "id": "emc_search_000001",
  "task": "memory_evidence_conflict",
  "dialogue": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "question": "...",
  "memory": {
    "policy": "defer_to_evidence",
    "items": [
      {
        "content": "User generally prefers Model Atlas...",
        "type": "positive_preference",
        "status": "active"
      }
    ]
  },
  "evaluation": {
    "reference_answer": "Recommend Model Boreal...",
    "preference_aligned_answer": "Model Atlas",
    "rubric": {
      "expected_behavior": "...",
      "failure_behavior": "..."
    }
  },
  "metadata": {
    "source_id": "emc_search_000001",
    "subtype": "noisy_search_evidence_memory_conflict",
    "topic": "text_summarization_systems"
  }
}
```

### Core Fields

- `id`: Stable example identifier.
- `task`: One of the five canonical task names listed above.
- `dialogue`: Historical user-assistant messages. The final benchmark query is not included here.
- `question`: The current query presented after the historical dialogue.
- `memory`: Gold memory annotations and the intended memory-use policy.
- `evaluation`: Reference answer, preference-aligned failure direction, and task-specific rubric.
- `metadata`: Minimal provenance and analysis labels. Objective Fact Judgment examples retain their source-dataset attribution here.

### Memory Status

- `active`: A currently valid memory.
- `current`: The newer preference in a Valid Memory Selection example.
- `outdated`: The superseded preference in a Valid Memory Selection example.

### Memory Policy

- `ignore_as_evidence`: Memory may explain the user's framing but must not serve as factual evidence.
- `constrain_to_scope`: Use only the part of memory valid for the current subject, audience, or constraints.
- `defer_to_evidence`: Prefer stronger current evidence over a conflicting historical preference.
- `update`: Select the current preference and avoid the outdated one.
- `use`: Apply valid memory to personalize the response.

## Loading the Data

The repository includes a dependency-free loader:

```python
from dataset import load_dataset, task_names

print(task_names())
examples = load_dataset("valid_memory_selection")
print(examples[0]["question"])
```

Stream all tasks without loading the full dataset into memory:

```python
from dataset import iter_dataset

for example in iter_dataset():
    process(example)
```

A model under evaluation should receive only the condition-specific context and `question`. Fields under `memory` and `evaluation` are gold annotations for evaluation and analysis; they must not be exposed to the answer model unless an explicitly oracle-style condition is being studied.

## Schema

The current release uses schema version `1.2` with five canonical task identifiers and open-ended LLM-judged evaluation. Dialogue turns use lowercase `role` values (`user`, `assistant`) and plain `content` without duplicated speaker prefixes.
