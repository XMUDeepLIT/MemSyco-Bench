# MemSyco-Bench Data Card

MemSyco-Bench evaluates whether a memory-augmented agent gives retrieved user memory the appropriate decision authority. The release contains 1,550 examples across five task categories.

## Tasks

| Task | Memory policy | Samples | File |
| --- | --- | ---: | --- |
| Objective Fact Judgment | `ignore_as_evidence` | 300 | `objective_fact_judgment.jsonl` |
| Contextual Scope Limits | `constrain_to_scope` | 300 | `contextual_scope_limits.jsonl` |
| Preference-Fact Conflict | `defer_to_evidence` | 300 | `preference_fact_conflict.jsonl` |
| Preference Change | `update` | 350 | `preference_change.jsonl` |
| Personalized Recommendation | `use` | 300 | `personalized_recommendation.jsonl` |

Counts and SHA-256 checksums are recorded in [`manifest.json`](manifest.json). The machine-readable row schema is [`schema.json`](schema.json).

## Unified Row Format

Every JSONL row has the same seven top-level fields:

```json
{
  "id": "emc_search_000001",
  "task": "preference_fact_conflict",
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
- `current`: The newer preference in a Preference Change example.
- `outdated`: The superseded preference in a Preference Change example.

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
examples = load_dataset("preference_change")
print(examples[0]["question"])
```

Stream all tasks without loading the full dataset into memory:

```python
from dataset import iter_dataset

for example in iter_dataset():
    process(example)
```

A model under evaluation should receive only the condition-specific context and `question`. Fields under `memory` and `evaluation` are gold annotations for evaluation and analysis; they must not be exposed to the answer model unless an explicitly oracle-style condition is being studied.

## Release Notes

Version `1.1` removes legacy multiple-choice fields (`evaluation.options`, `evaluation.gold_option`). All tasks are evaluated as open-ended questions with LLM judges.

Version `1.0` replaces the previous heterogeneous release layout. Construction-only fields such as dialogue spans, validation flags, duplicated source text, and session-generation metadata were removed. Dialogue turns now use standard lowercase `role` values and content without duplicated `User:` or `Assistant:` prefixes.
