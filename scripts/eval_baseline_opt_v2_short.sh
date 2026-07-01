#!/usr/bin/env bash
# Run optimized memory-baseline evaluations for recommendation,
# preference-update, noisy evidence-memory-conflict, memory-scope,
# and consensus-judgment tasks.
#
# Examples:
#   ./scripts/eval_baseline_opt_v2_short.sh
#   ./scripts/eval_baseline_opt_v2_short.sh --limit 5 --trace-api
#   ./scripts/eval_baseline_opt_v2_short.sh --tasks objective_fact_judgment --methods RawDialogue,MemZero --limit 50
#   ./scripts/eval_baseline_opt_v2_short.sh --limit 0
#
# Limit:
#   --limit 20 runs 20 samples per task/method.
#   --limit 0 omits --limit and lets each Python script use its own default.
#
# Environment expected by the evaluation scripts:
#   GENERATION_API_KEY / DEEPSEEK_API_KEY / API_KEY
#   JUDGE_API_KEY / DEEPSEEK_JUDGE_API_KEY
#
# Environment commonly used by baselines:
#   MEMORY_API_KEY
#   MEMORY_BASE_URL
#   MEMORY_LLM_MODEL
#   MEMORY_EMBEDDING_MODEL
#   MEMORY_EMBEDDING_DIMS
#   MEMORY_EMBEDDING_API_KEY
#   MEMORY_EMBEDDING_BASE_URL
#
# Environment used by the MemGPT baseline (a from-scratch, lightweight re-implementation
# of the MemGPT memory mechanism; no vendored letta platform, no Docker, no extra deps beyond
# openai + numpy). An LLM agent self-manages a two-tier memory via tool calls: it edits core
# memory blocks (in-context) and inserts/searches an archival vector store. Memory is built
# once per dialogue and reused across questions (cached via a marker file), matching the other
# baselines' flow. It reuses the shared MEMORY_* vars exactly like MemoryBank / Supermemory:
#   MEMORY_API_KEY / MEMORY_BASE_URL / MEMORY_LLM_MODEL          ingest agent LLM
#   MEMORY_EMBEDDING_API_KEY / MEMORY_EMBEDDING_BASE_URL         archival embeddings
#   MEMORY_EMBEDDING_MODEL / MEMORY_EMBEDDING_DIMS               embedding model + dim
#   MEMGPT_MAX_STEPS            max agent tool-calling steps per ingest batch (default 12)
#   MEMGPT_INGEST_BATCH_SIZE    dialogue turns per ingest batch (default 6)
#   MEMGPT_LANGUAGE             en (default) | cn
# Note: this is a faithful re-implementation of the MemGPT *mechanism*, not official Letta.
#
# Environment used by the MemoryBank baseline (reuses the MEMORY_* vars above;
# LLM summaries use MEMORY_*, retrieval uses MEMORY_EMBEDDING_*):
#   MEMORYBANK_LANGUAGE        en (default) | cn
#   MEMORYBANK_DISABLE_SUMMARY 1 to skip the LLM summary/personality step (retrieval only)
#
# Environment used by the Supermemory baseline (LOCAL re-implementation; no official
# supermemory API/SDK needed). It faithfully reproduces Supermemory's mechanism on top of
# this repo's infra, reusing the MEMORY_* vars exactly like MemoryBank / A-MEM: an LLM
# extracts atomic facts about the user and splits them into a
# static + dynamic profile; the bge-m3 embedding service (MEMORY_EMBEDDING_*) indexes them
# for top-k semantic retrieval. Memory is built once per dialogue and cached on disk.
#   MEMORY_LLM_MODEL           extraction LLM (default deepseek-v4-flash)
#   MEMORY_EMBEDDING_MODEL     retrieval embedding model (e.g. baai/bge-m3)
#   MEMORY_EMBEDDING_DIMS      embedding dimension (e.g. 1024)
#   MEMORY_EMBEDDING_API_KEY   embedding service key
#   MEMORY_EMBEDDING_BASE_URL  embedding service base url (OpenAI-compatible bge-m3)
#   SUPERMEMORY_LANGUAGE       en (default) | cn  (fact-extraction language)
#   SUPERMEMORY_RETRIEVAL_MODE profile (default, static+dynamic profile + relevant memories)
#                              | search (top-k semantic retrieval only)
#   SUPERMEMORY_MAX_MEMORIES   max facts to extract per dialogue (default 40)

set -euo pipefail

TASKS=("personalized_recommendation" "preference_change" "preference_fact_conflict" "contextual_scope_limits" "objective_fact_judgment")
METHODS=("NoMemory" "RawDialogue" "MemZero" "A-MEM" "LightMem" "MemoryBank" "NaiveRAG" "MemGPT" "Supermemory")
LIMIT=20
WORKERS=1
OUTER_WORKERS=1
MEMORY_TOP_K=10
OUTPUT_ROOT="output_data/baseline_opt_v2_runs_short_extra_instruction"
MEMORY_SAVE_ROOT_BASE="output_data/baseline_opt_v2_memory_runs"
MODELS=("")
BASE_URL=""
ARG_API_KEY=""
JUDGE_MODEL=""
ARG_JUDGE_BASE_URL=""
ARG_JUDGE_API_KEY=""
REQUEST_TIMEOUT=60
JUDGE_REQUEST_TIMEOUT=60
API_MAX_RETRIES=1
CURRENT_DATE="${EVAL_CURRENT_DATE-}"
COMPLETION_CACHE_PATH="output_data/completion_cache/baseline_opt_v2_completions.sqlite"
COMPLETION_CACHE_PATH_SET=0
NO_COMPLETION_CACHE=0
NO_DISK_COMPLETION_CACHE=0
NO_QUESTION_FILTER=1
TRACE_API=0
CONTINUE_ON_ERROR=0
DRY_RUN=0
SCHEDULE_BY="model"
ANSWER_SYSTEM_EXTRA_INSTRUCTION="${ANSWER_SYSTEM_EXTRA_INSTRUCTION-}"

usage() {
  sed -n '2,26p' "$0"
  cat <<'EOF'

Options:
  --tasks personalized_recommendation,preference_change,preference_fact_conflict,contextual_scope_limits,objective_fact_judgment
  --methods NoMemory,RawDialogue,MemZero,A-MEM,LightMem,MemoryBank,NaiveRAG,MemGPT,Supermemory
  --limit N
  --workers N  Accepted for compatibility; optimized evaluators force workers=1.
  --outer-workers N  Run up to N task/method/model combinations concurrently.
                     For disk-backed memory baselines, at most one model runs per
                     task/method at a time so a shared memory store is never read
                     by multiple processes. A file lock under --memory-save-root-base
                     also blocks concurrent runs from other shell invocations.
  --schedule-by model|task
                     Job submission order (default: model).
                     model: iterate model -> task -> method so all tasks for one
                     model can run in parallel and fill --outer-workers.
                     task: iterate task -> method -> model (legacy order).
  --memory-top-k N
  --output-root PATH
  --memory-save-root-base PATH
  --model NAME[,NAME...]
  --models NAME[,NAME...]
  --base-url URL
  --api-key KEY
  --judge-model NAME
  --judge-base-url URL
  --judge-api-key KEY
  --request-timeout N
  --judge-request-timeout N
  --api-max-retries N
  --current-date YYYY-MM-DD
  --completion-cache-path PATH
  --no-completion-cache
  --no-disk-completion-cache
  --no-question-filter
  --trace-api
  --answer-system-extra-instruction TEXT
  --continue-on-error
  --dry-run            Print evaluator commands without running them.
  -h, --help
EOF
}

split_csv() {
  local value="$1"
  local -n out_ref="$2"
  local old_ifs="$IFS"
  IFS=',' read -r -a out_ref <<< "$value"
  IFS="$old_ifs"
}

require_value() {
  local opt="$1"
  local value="${2-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    echo "Missing value for $opt" >&2
    exit 2
  fi
}

method_slug() {
  case "$1" in
    NoMemory) echo "no_memory" ;;
    RawDialogue) echo "raw_dialogue" ;;
    MemZero) echo "memzero" ;;
    NaiveRAG) echo "naive_rag" ;;
    A-MEM) echo "amem" ;;
    LightMem) echo "lightmem" ;;
    MemoryBank) echo "memorybank" ;;
    MemGPT) echo "memgpt" ;;
    Supermemory) echo "supermemory" ;;
    *)
      echo "$1" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
      ;;
  esac
}

is_no_memory_method() {
  [[ "$1" == "NoMemory" ]]
}

supports_no_memory_task() {
  [[ "$1" == "objective_fact_judgment" || "$1" == "consensus_judgment" ]]
}

is_raw_dialogue_method() {
  [[ "$1" == "RawDialogue" ]]
}

is_disk_memory_method() {
  ! is_no_memory_method "$1" && ! is_raw_dialogue_method "$1"
}

memory_store_slot_key() {
  local task_slug="$1"
  local method="$2"
  if ! is_disk_memory_method "$method"; then
    return 0
  fi
  printf '%s_%s' "$task_slug" "$(method_slug "$method")"
}

raw_dialogue_needs_with_memory_only_flag() {
  case "$1" in
    objective_fact_judgment|consensus_judgment)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

model_slug() {
  local value="$1"
  if [[ -z "$value" ]]; then
    echo "default_model"
    return
  fi
  echo "$value" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

task_spec() {
  case "$1" in
    personalized_recommendation|recommend)
      TASK_SLUG="recommend"
      SCRIPT_PATH="evaluation/run_task.py"
      TEST_JSONL="data/personalized_recommendation.jsonl"
      OUTPUT_SUFFIX="recommend_question_open_results_final.json"
      SUPPORTS_NO_QUESTION_FILTER=0
      ;;
    preference_change|recommend_change)
      TASK_SLUG="recommend_change"
      SCRIPT_PATH="evaluation/run_task.py"
      TEST_JSONL="data/preference_change.jsonl"
      OUTPUT_SUFFIX="preference_update_open_eval_result_final.json"
      SUPPORTS_NO_QUESTION_FILTER=0
      ;;
    preference_fact_conflict|evidence_memory_conflict_noisy)
      TASK_SLUG="evidence_memory_conflict_noisy"
      SCRIPT_PATH="evaluation/run_task.py"
      TEST_JSONL="data/preference_fact_conflict.jsonl"
      OUTPUT_SUFFIX="evidence_memory_conflict_noisy_results_final.json"
      SUPPORTS_NO_QUESTION_FILTER=0
      ;;
    contextual_scope_limits|memory_scope_overgeneralization)
      TASK_SLUG="memory_scope_overgeneralization"
      SCRIPT_PATH="evaluation/run_task.py"
      TEST_JSONL="data/contextual_scope_limits.jsonl"
      OUTPUT_SUFFIX="memory_scope_overgeneralization_v3_results_final.json"
      SUPPORTS_NO_QUESTION_FILTER=0
      ;;
    objective_fact_judgment|consensus_judgment)
      TASK_SLUG="consensus_judgment"
      SCRIPT_PATH="evaluation/run_task.py"
      TEST_JSONL="data/objective_fact_judgment.jsonl"
      OUTPUT_SUFFIX="objective_consensus_judgment_results_final.json"
      SUPPORTS_NO_QUESTION_FILTER=0
      ;;
    *)
      echo "Unsupported task '$1'. Use personalized_recommendation, preference_change, preference_fact_conflict, contextual_scope_limits, objective_fact_judgment." >&2
      exit 2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)
      require_value "$1" "${2-}"
      split_csv "$2" TASKS
      shift 2
      ;;
    --methods)
      require_value "$1" "${2-}"
      split_csv "$2" METHODS
      shift 2
      ;;
    --limit)
      require_value "$1" "${2-}"
      LIMIT="$2"
      shift 2
      ;;
    --workers)
      require_value "$1" "${2-}"
      WORKERS="$2"
      if [[ "$WORKERS" != "1" ]]; then
        echo "WARNING: optimized evaluators force --workers 1 for disk memory reuse; ignoring --workers $WORKERS" >&2
        WORKERS=1
      fi
      shift 2
      ;;
    --outer-workers)
      require_value "$1" "${2-}"
      OUTER_WORKERS="$2"
      shift 2
      ;;
    --memory-top-k)
      require_value "$1" "${2-}"
      MEMORY_TOP_K="$2"
      shift 2
      ;;
    --output-root)
      require_value "$1" "${2-}"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --memory-save-root-base)
      require_value "$1" "${2-}"
      MEMORY_SAVE_ROOT_BASE="$2"
      shift 2
      ;;
    --model|--models)
      require_value "$1" "${2-}"
      split_csv "$2" MODELS
      shift 2
      ;;
    --base-url)
      require_value "$1" "${2-}"
      BASE_URL="$2"
      shift 2
      ;;
    --api-key)
      require_value "$1" "${2-}"
      ARG_API_KEY="$2"
      shift 2
      ;;
    --judge-model)
      require_value "$1" "${2-}"
      JUDGE_MODEL="$2"
      shift 2
      ;;
    --judge-base-url)
      require_value "$1" "${2-}"
      ARG_JUDGE_BASE_URL="$2"
      shift 2
      ;;
    --judge-api-key)
      require_value "$1" "${2-}"
      ARG_JUDGE_API_KEY="$2"
      shift 2
      ;;
    --request-timeout)
      require_value "$1" "${2-}"
      REQUEST_TIMEOUT="$2"
      shift 2
      ;;
    --judge-request-timeout)
      require_value "$1" "${2-}"
      JUDGE_REQUEST_TIMEOUT="$2"
      shift 2
      ;;
    --api-max-retries)
      require_value "$1" "${2-}"
      API_MAX_RETRIES="$2"
      shift 2
      ;;
    --current-date)
      require_value "$1" "${2-}"
      CURRENT_DATE="$2"
      shift 2
      ;;
    --completion-cache-path)
      require_value "$1" "${2-}"
      COMPLETION_CACHE_PATH="$2"
      COMPLETION_CACHE_PATH_SET=1
      shift 2
      ;;
    --no-completion-cache)
      NO_COMPLETION_CACHE=1
      shift
      ;;
    --no-disk-completion-cache)
      NO_DISK_COMPLETION_CACHE=1
      shift
      ;;
    --no-question-filter)
      NO_QUESTION_FILTER=1
      shift
      ;;
    --trace-api)
      TRACE_API=1
      shift
      ;;
    --answer-system-extra-instruction)
      require_value "$1" "${2-}"
      ANSWER_SYSTEM_EXTRA_INSTRUCTION="$2"
      shift 2
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --schedule-by)
      require_value "$1" "${2-}"
      SCHEDULE_BY="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OLD_PYTHONPATH="${PYTHONPATH-}"
if [[ -n "$OLD_PYTHONPATH" ]]; then
  export PYTHONPATH="$REPO_ROOT:$OLD_PYTHONPATH"
else
  export PYTHONPATH="$REPO_ROOT"
fi

restore_pythonpath() {
  if [[ -n "$OLD_PYTHONPATH" ]]; then
    export PYTHONPATH="$OLD_PYTHONPATH"
  else
    unset PYTHONPATH
  fi
}
trap restore_pythonpath EXIT

export ANSWER_SYSTEM_EXTRA_INSTRUCTION

mkdir -p "$OUTPUT_ROOT" "$MEMORY_SAVE_ROOT_BASE"

if ! [[ "$OUTER_WORKERS" =~ ^[0-9]+$ ]] || (( OUTER_WORKERS < 1 )); then
  echo "--outer-workers must be a positive integer, got '$OUTER_WORKERS'" >&2
  exit 2
fi

case "$SCHEDULE_BY" in
  model|task) ;;
  *)
    echo "--schedule-by must be 'model' or 'task', got '$SCHEDULE_BY'" >&2
    exit 2
    ;;
esac

running_jobs=0
failed_jobs=0
declare -A MEMORY_STORE_SLOTS_ACTIVE=()
declare -A JOB_MEMORY_STORE_KEY=()

kill_all_background_jobs() {
  local pid
  for pid in "${!JOB_MEMORY_STORE_KEY[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}

cleanup_on_signal() {
  echo "Stopping background eval jobs..." >&2
  kill_all_background_jobs
  while (( running_jobs > 0 )); do
    wait_for_one_job
  done
  exit 130
}

trap cleanup_on_signal INT TERM

release_job_resources() {
  local finished_pid="$1"
  local slot_key="${JOB_MEMORY_STORE_KEY[$finished_pid]-}"

  unset "JOB_MEMORY_STORE_KEY[$finished_pid]"
  if [[ -n "$slot_key" ]]; then
    unset "MEMORY_STORE_SLOTS_ACTIVE[$slot_key]"
  fi
}

wait_for_one_job() {
  local finished_pid="" exit_code=0
  set +e
  if (( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )); then
    wait -n -p finished_pid
    exit_code=$?
  else
    wait -n
    exit_code=$?
    for finished_pid in "${!JOB_MEMORY_STORE_KEY[@]}"; do
      if ! kill -0 "$finished_pid" 2>/dev/null; then
        wait "$finished_pid" 2>/dev/null || true
        break
      fi
    done
  fi
  set -e

  if [[ -n "$finished_pid" ]]; then
    release_job_resources "$finished_pid"
  fi
  if (( exit_code != 0 )); then
    failed_jobs=$((failed_jobs + 1))
    if (( CONTINUE_ON_ERROR == 0 )); then
      echo "A parallel evaluation job failed with exit code $exit_code; waiting for already-started jobs to finish." >&2
    fi
  fi
  running_jobs=$((running_jobs - 1))
}

wait_for_job_slot() {
  while (( running_jobs >= OUTER_WORKERS )); do
    wait_for_one_job
  done
}

wait_for_memory_store_slot() {
  local slot_key="$1"
  [[ -z "$slot_key" ]] && return 0

  while [[ -n "${MEMORY_STORE_SLOTS_ACTIVE[$slot_key]+x}" ]]; do
    if (( running_jobs <= 0 )); then
      echo "Internal error: memory store slot '$slot_key' is active but no jobs are running." >&2
      exit 1
    fi
    echo "Waiting for memory store slot: $slot_key (one model per task/method)" >&2
    wait_for_one_job
  done
}

wait_for_all_jobs() {
  while (( running_jobs > 0 )); do
    wait_for_one_job
  done
}

job_completion_cache_path() {
  local task_slug="$1"
  local method_slug="$2"
  local model_slug="$3"

  if (( OUTER_WORKERS <= 1 || COMPLETION_CACHE_PATH_SET == 1 )); then
    echo "$COMPLETION_CACHE_PATH"
    return
  fi

  local cache_dir cache_file cache_stem cache_ext
  cache_dir="$(dirname "$COMPLETION_CACHE_PATH")"
  cache_file="$(basename "$COMPLETION_CACHE_PATH")"
  cache_ext=""
  cache_stem="$cache_file"
  if [[ "$cache_file" == *.* ]]; then
    cache_ext=".${cache_file##*.}"
    cache_stem="${cache_file%.*}"
  fi
  echo "$cache_dir/${cache_stem}_${task_slug}_${method_slug}_${model_slug}${cache_ext}"
}

run_eval_job() {
  local task_name="$1"
  local task_slug="$2"
  local script_path="$3"
  local test_jsonl="$4"
  local output_suffix="$5"
  local supports_no_question_filter="$6"
  local method="$7"
  local model="$8"

  local method_slug model_slug task_output_dir output_path memory_save_root job_cache_path
  local lock_dir=""
  method_slug="$(method_slug "$method")"
  model_slug="$(model_slug "$model")"
  task_output_dir="$OUTPUT_ROOT/$task_slug"
  output_path="$task_output_dir/${method_slug}_${model_slug}_$output_suffix"
  memory_save_root="$MEMORY_SAVE_ROOT_BASE/$task_slug/$method_slug"
  job_cache_path="$(job_completion_cache_path "$task_slug" "$method_slug" "$model_slug")"

  mkdir -p "$task_output_dir"
  if is_disk_memory_method "$method"; then
    mkdir -p "$memory_save_root"
    mkdir -p "$MEMORY_SAVE_ROOT_BASE/.outer_locks"
    lock_dir="$MEMORY_SAVE_ROOT_BASE/.outer_locks/${task_slug}_${method_slug}.lock"
  fi
  if [[ -n "$job_cache_path" ]]; then
    mkdir -p "$(dirname "$job_cache_path")"
  fi

  echo "============================================================"
  echo "Task:             $task_slug"
  echo "Baseline:         $method"
  echo "Model:            ${model:-script default}"
  echo "Script:           $script_path"
  echo "Test jsonl:       $test_jsonl"
  echo "Output:           $output_path"
  if is_no_memory_method "$method"; then
    echo "Context mode:     no prior dialogue"
  elif is_raw_dialogue_method "$method"; then
    echo "Context mode:     raw prior dialogue"
  else
    echo "Memory save root: $memory_save_root"
  fi
  if (( LIMIT > 0 )); then
    echo "Limit:            $LIMIT"
  else
    echo "Limit:            script default"
  fi
  echo "Workers:          $WORKERS"
  echo "Outer workers:    $OUTER_WORKERS"
  if [[ -n "$CURRENT_DATE" ]]; then
    echo "Current date:     $CURRENT_DATE"
  else
    echo "Current date:     script default"
  fi
  if [[ -n "$ANSWER_SYSTEM_EXTRA_INSTRUCTION" ]]; then
    echo "Answer prompt extra instruction: $ANSWER_SYSTEM_EXTRA_INSTRUCTION"
  fi
  if (( NO_COMPLETION_CACHE == 0 && NO_DISK_COMPLETION_CACHE == 0 )) && [[ -n "$job_cache_path" ]]; then
    echo "Completion cache: $job_cache_path"
  fi
  echo "============================================================"

  local -a cmd_args=(
    -B
    "$script_path"
    "$task_name"
    --optimized
    --test-jsonl "$test_jsonl"
    --output "$output_path"
    --workers "$WORKERS"
    --request-timeout "$REQUEST_TIMEOUT"
    --judge-request-timeout "$JUDGE_REQUEST_TIMEOUT"
    --api-max-retries "$API_MAX_RETRIES"
  )

  if is_no_memory_method "$method"; then
    cmd_args+=(--no-memory-only)
  elif is_raw_dialogue_method "$method"; then
    if raw_dialogue_needs_with_memory_only_flag "$task_name"; then
      cmd_args+=(--with-memory-only)
    fi
  else
    cmd_args+=(
      --memory-method "$method"
      --memory-save-root "$memory_save_root"
      --memory-top-k "$MEMORY_TOP_K"
    )
  fi

  if (( LIMIT > 0 )); then
    cmd_args+=(--limit "$LIMIT")
  fi
  if [[ -n "$model" ]]; then
    cmd_args+=(--model "$model")
  fi
  if [[ -n "$BASE_URL" ]]; then
    cmd_args+=(--base-url "$BASE_URL")
  fi
  if [[ -n "$ARG_API_KEY" ]]; then
    cmd_args+=(--api-key "$ARG_API_KEY")
  fi
  if [[ -n "$JUDGE_MODEL" ]]; then
    cmd_args+=(--judge-model "$JUDGE_MODEL")
  fi
  if [[ -n "$ARG_JUDGE_BASE_URL" ]]; then
    cmd_args+=(--judge-base-url "$ARG_JUDGE_BASE_URL")
  fi
  if [[ -n "$ARG_JUDGE_API_KEY" ]]; then
    cmd_args+=(--judge-api-key "$ARG_JUDGE_API_KEY")
  fi
  if [[ -n "$CURRENT_DATE" ]]; then
    cmd_args+=(--current-date "$CURRENT_DATE")
  fi
  if (( NO_COMPLETION_CACHE == 1 )); then
    cmd_args+=(--no-completion-cache)
  fi
  if [[ -n "$job_cache_path" ]]; then
    cmd_args+=(--completion-cache-path "$job_cache_path")
  fi
  if (( NO_DISK_COMPLETION_CACHE == 1 )); then
    cmd_args+=(--no-disk-completion-cache)
  fi
  if (( NO_QUESTION_FILTER == 1 && supports_no_question_filter == 1 )); then
    cmd_args+=(--no-question-filter)
  fi
  if (( TRACE_API == 1 )); then
    cmd_args+=(--trace-api)
  fi

  if (( DRY_RUN == 1 )); then
    printf 'Command: python'
    printf ' %q' "${cmd_args[@]}"
    printf '\n'
    return 0
  fi

  if [[ -n "$lock_dir" ]]; then
    while ! mkdir "$lock_dir" 2>/dev/null; do
      echo "Waiting for memory-store lock: $lock_dir" >&2
      sleep 5
    done
  fi

  local exit_code
  set +e
  python "${cmd_args[@]}"
  exit_code=$?
  set -e

  if [[ -n "$lock_dir" ]]; then
    rmdir "$lock_dir" 2>/dev/null || true
  fi

  if (( exit_code != 0 )); then
    echo "Task $task_slug, baseline $method, model ${model:-script default} failed with exit code $exit_code" >&2
    return "$exit_code"
  fi

  echo "Finished task=$task_slug, baseline=$method, model=${model:-script default}"
  echo ""
}

validate_task_inputs() {
  local task_name="$1"
  task_spec "$task_name"

  if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "Evaluation script not found: $SCRIPT_PATH" >&2
    exit 1
  fi
  if [[ ! -f "$TEST_JSONL" ]]; then
    echo "Test jsonl not found: $TEST_JSONL" >&2
    exit 1
  fi
}

schedule_eval_job() {
  local task_name="$1"
  local method="$2"
  local model="$3"

  task_spec "$task_name"

  if is_no_memory_method "$method" && ! supports_no_memory_task "$task_name"; then
    echo "Skipping NoMemory for task=$task_name: no-memory is only meaningful for consensus_judgment." >&2
    return 0
  fi

  local memory_slot_key
  memory_slot_key="$(memory_store_slot_key "$TASK_SLUG" "$method")"

  wait_for_job_slot
  wait_for_memory_store_slot "$memory_slot_key"
  if [[ -n "$memory_slot_key" ]]; then
    MEMORY_STORE_SLOTS_ACTIVE[$memory_slot_key]=1
  fi

  run_eval_job \
    "$task_name" \
    "$TASK_SLUG" \
    "$SCRIPT_PATH" \
    "$TEST_JSONL" \
    "$OUTPUT_SUFFIX" \
    "$SUPPORTS_NO_QUESTION_FILTER" \
    "$method" \
    "$model" &
  local job_pid=$!
  JOB_MEMORY_STORE_KEY[$job_pid]="$memory_slot_key"
  running_jobs=$((running_jobs + 1))
}

for task_name in "${TASKS[@]}"; do
  validate_task_inputs "$task_name"
done

if [[ "$SCHEDULE_BY" == "model" ]]; then
  for model in "${MODELS[@]}"; do
    for task_name in "${TASKS[@]}"; do
      for method in "${METHODS[@]}"; do
        schedule_eval_job "$task_name" "$method" "$model"
      done
    done
  done
else
  for task_name in "${TASKS[@]}"; do
    for method in "${METHODS[@]}"; do
      for model in "${MODELS[@]}"; do
        schedule_eval_job "$task_name" "$method" "$model"
      done
    done
  done
fi

wait_for_all_jobs

if (( failed_jobs > 0 )); then
  echo "$failed_jobs evaluation job(s) failed." >&2
  exit 1
fi

echo "All requested evaluations finished."
