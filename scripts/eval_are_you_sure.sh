#!/usr/bin/env bash
# Are-you-sure probe driver.
#
# Reuses first-turn answers from the existing baseline_opt_v2 result JSONs and
# only sends the follow-up "Are you sure?" turn + a re-judge. See
# evaluation/run_stability_probe.py for the reconstruction logic.
#
# Examples:
#   ./scripts/eval_are_you_sure.sh
#   ./scripts/eval_are_you_sure.sh --limit 20 --workers 4
#   ./scripts/eval_are_you_sure.sh --outer-workers 5            # 5 task/method jobs in parallel
#   ./scripts/eval_are_you_sure.sh --tasks recommend,consensus_judgment --methods RawDialogue,A-MEM --limit 50
#
# Required environment (the script does NOT set these):
#   GENERATION_API_KEY          answer-model API key (also used for the re-judge
#                               call when JUDGE_API_KEY is unset)
# Optional:
#   GENERATION_BASE_URL         overrides the follow-up answer endpoint
#                               (default: https://api.deepseek.com)
#   JUDGE_BASE_URL              overrides the re-judge endpoint
#                               (default: https://api.deepseek.com)
#   JUDGE_API_KEY               separate judge key (defaults to GENERATION_API_KEY)
# The follow-up model / judge_model / current_date / extra-instruction are read
# from each input result JSON, so the follow-up matches the original first turn.
#
# Requests go directly to the configured OpenAI-compatible endpoints.
#
# Parallelism:
#   --outer-workers N runs up to N (task, method) jobs concurrently. Each job
#   has its own input result JSON, output file, and completion cache, so they
#   are fully independent (no shared memory store). Total concurrent API calls
#   ~= outer-workers * workers; tune to your rate limits.
#
# Notes on data availability (output_data/baseline_opt_v2_runs_short):
#   - Both models have full clean (no extra-instruction) coverage for the 7
#     default methods x 5 tasks: DeepSeek-V4-Flash and Qwen3-8B.
#   - Default methods mirror the mem_reminder table: Full Dialog (RawDialogue),
#     Mem0 (MemZero), A-Mem (A-MEM), LightMem (LightMemFull), MemGPT
#     (MemGPTMinimal), MemoryBank, SuperMemory. NoMemory and NaiveRAG are
#     excluded by default (add via --methods if needed).
#   - Models: Qwen3-8B (qwen/qwen3-8b) and DeepSeek-V4-Flash (deepseek-v4-flash).

set -euo pipefail

TASKS=("recommend" "recommend_change" "evidence_memory_conflict_noisy" "memory_scope_overgeneralization" "consensus_judgment")
METHODS=("RawDialogue" "MemZero" "A-MEM" "LightMemFull" "MemGPTMinimal" "MemoryBank" "Supermemory")
LIMIT=350
WORKERS=4
OUTER_WORKERS=5
# Each entry is "api_model::discovery_model". The api_model is the name sent to
# the follow-up API; the discovery_model is the slug used to find the existing
# input result JSON (<method>_<discovery_model>_<suffix>). They differ for
# openrouter-served deepseek (api "deepseek/deepseek-v4-flash" vs file slug
# "deepseek-v4-flash"). If no "::", discovery_model = api_model.
MODELS=("deepseek-v4-flash::deepseek-v4-flash")
FOLLOWUP_QUESTION="Are you sure?"
# Endpoints for the follow-up answer + re-judge calls. Generation defaults to
# openrouter (which serves both deepseek-v4-flash and qwen/qwen3-8b), judge
# defaults to the deepseek official endpoint. Override with --base-url /
# --judge-base-url or the GENERATION_BASE_URL / JUDGE_BASE_URL env vars.
BASE_URL="${GENERATION_BASE_URL:-https://api.deepseek.com}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.deepseek.com}"
EXTRA_ARGS=()

usage() {
  sed -n '2,40p' "$0"
  cat <<'EOF'

Options:
  --tasks t1,t2,...        default: all 5 tasks
  --methods m1,m2,...      default: RawDialogue,MemZero,A-MEM,LightMemFull,MemGPTMinimal,MemoryBank,Supermemory
  --models m1,m2,...       default: deepseek/deepseek-v4-flash::deepseek-v4-flash,qwen/qwen3-8b::qwen/qwen3-8b
                           (each entry "api_model::discovery_model"; follow-up model + input-file slug)
  --model NAME             shorthand for --models NAME (single model, api=discovery)
  --base-url URL           default: https://openrouter.ai/api/v1 (follow-up answer endpoint)
  --judge-base-url URL     default: https://api.deepseek.com (re-judge endpoint)
  --limit N                default: 20
  --workers N              default: 4 (inner threads per job)
  --outer-workers N        default: 5 (concurrent task/method/model jobs)
  --followup-question TEXT default: "Are you sure?"
  Any extra args (e.g. --trace-api --no-completion-cache) are forwarded to the python driver.
EOF
}

split_csv() {
  local value="$1"
  local -n out_ref="$2"
  local old_ifs="$IFS"
  IFS=',' read -r -a out_ref <<< "$value"
  IFS="$old_ifs"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks) split_csv "$2" TASKS; shift 2 ;;
    --methods) split_csv "$2" METHODS; shift 2 ;;
    --models) split_csv "$2" MODELS; shift 2 ;;
    --model) MODELS=("$2"); shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --judge-base-url) JUDGE_BASE_URL="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --outer-workers) OUTER_WORKERS="$2"; shift 2 ;;
    --followup-question) FOLLOWUP_QUESTION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if ! [[ "$OUTER_WORKERS" =~ ^[0-9]+$ ]] || (( OUTER_WORKERS < 1 )); then
  echo "--outer-workers must be a positive integer, got '$OUTER_WORKERS'" >&2
  exit 2
fi
if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || (( WORKERS < 1 )); then
  echo "--workers must be a positive integer, got '$WORKERS'" >&2
  exit 2
fi


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH-}"

LOG_DIR="output_data/are_you_sure_runs_short/.logs"
mkdir -p "$LOG_DIR"

running_jobs=0
failed_jobs=0
declare -A JOB_LABEL=()
declare -A JOB_LOG=()

wait_for_one_job() {
  local finished_pid="" exit_code=0
  set +e
  if (( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )); then
    wait -n -p finished_pid
    exit_code=$?
  else
    wait -n
    exit_code=$?
    for finished_pid in "${!JOB_LABEL[@]}"; do
      if ! kill -0 "$finished_pid" 2>/dev/null; then
        wait "$finished_pid" 2>/dev/null || true
        break
      fi
    done
  fi
  set -e
  running_jobs=$((running_jobs - 1))
  if [[ -n "$finished_pid" && -n "${JOB_LABEL[$finished_pid]:-}" ]]; then
    if (( exit_code == 0 )); then
      echo "[are-you-sure] ok: ${JOB_LABEL[$finished_pid]}"
    else
      echo "[are-you-sure] FAILED: ${JOB_LABEL[$finished_pid]} (exit=$exit_code, log: ${JOB_LOG[$finished_pid]})" >&2
      failed_jobs=$((failed_jobs + 1))
    fi
    unset 'JOB_LABEL[$finished_pid]'
    unset 'JOB_LOG[$finished_pid]'
  elif (( exit_code != 0 )); then
    failed_jobs=$((failed_jobs + 1))
  fi
}

wait_for_job_slot() {
  while (( running_jobs >= OUTER_WORKERS )); do
    wait_for_one_job
  done
}

wait_for_all_jobs() {
  while (( running_jobs > 0 )); do
    wait_for_one_job
  done
}

trap 'echo "[are-you-sure] interrupted, waiting for started jobs to finish..." >&2; wait_for_all_jobs; exit 130' INT TERM

schedule_job() {
  local task="$1" method="$2" model="$3" input_model="$4"
  wait_for_job_slot
  local model_slug="${input_model//[\/-]/_}"
  local label="$task/$method/$model"
  local log="$LOG_DIR/${task}__${method}__${model_slug}.log"
  (
    set +e
    echo "============================================================"
    echo "[are-you-sure] task=$task method=$method model=$model input_model=$input_model limit=$LIMIT outer=$OUTER_WORKERS inner=$WORKERS"
    echo "start: $(date -Iseconds)"
    echo "============================================================"
    python -B evaluation/run_stability_probe.py \
        --task "$task" \
        --method "$method" \
        --model "$model" \
        --input-model "$input_model" \
        --base-url "$BASE_URL" \
        --judge-base-url "$JUDGE_BASE_URL" \
        --limit "$LIMIT" \
        --workers "$WORKERS" \
        --followup-question "$FOLLOWUP_QUESTION" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    local rc=$?
    echo "end: $(date -Iseconds) exit=$rc"
    exit "$rc"
  ) >"$log" 2>&1 &
  local job_pid=$!
  JOB_LABEL[$job_pid]="$label"
  JOB_LOG[$job_pid]="$log"
  running_jobs=$((running_jobs + 1))
  echo "[are-you-sure] launched: $label (pid=$job_pid, log: $log)"
}

# Serial mode: run the job in the foreground so tqdm streams live to the
# terminal (no redirect, no log file).
run_job_foreground() {
  local task="$1" method="$2" model="$3" input_model="$4" rc=0
  echo "============================================================"
  echo "[are-you-sure] task=$task method=$method model=$model input_model=$input_model limit=$LIMIT"
  echo "============================================================"
  python -B evaluation/run_stability_probe.py \
      --task "$task" \
      --method "$method" \
      --model "$model" \
      --input-model "$input_model" \
      --base-url "$BASE_URL" \
      --judge-base-url "$JUDGE_BASE_URL" \
      --limit "$LIMIT" \
      --workers "$WORKERS" \
      --followup-question "$FOLLOWUP_QUESTION" \
      "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" || rc=$?
  if (( rc == 0 )); then
    echo "[are-you-sure] ok: $task/$method/$model"
  else
    echo "[are-you-sure] FAILED: $task/$method/$model (exit=$rc)" >&2
    failed_jobs=$((failed_jobs + 1))
  fi
}

# Parallel mode: background heartbeat that tails each job log's last tqdm line
# so progress is visible even though per-job output is redirected to files.
parallel_progress_heartbeat() {
  while true; do
    sleep 5
    local lines=()
    for log in "$LOG_DIR"/*.log; do
      [[ -f "$log" ]] || continue
      local last
      last="$(tr '\r' '\n' < "$log" 2>/dev/null | grep -E '/[0-9]|it/s|s/it' | tail -n 1 || true)"
      [[ -n "$last" ]] && lines+=("$(basename "$log" .log | tr '_' '/'): $last")
    done
    if ((${#lines[@]} > 0)); then
      printf '\n[progress %s]\n' "$(date +%H:%M:%S)"
      printf '  %s\n' "${lines[@]}"
    fi
  done
}

if (( OUTER_WORKERS == 1 )); then
  for entry in "${MODELS[@]}"; do
    model="${entry%%::*}"
    input_model="${entry##*::}"
    for task in "${TASKS[@]}"; do
      for method in "${METHODS[@]}"; do
        run_job_foreground "$task" "$method" "$model" "$input_model"
      done
    done
  done
else
  parallel_progress_heartbeat &
  heartbeat_pid=$!
  for entry in "${MODELS[@]}"; do
    model="${entry%%::*}"
    input_model="${entry##*::}"
    for task in "${TASKS[@]}"; do
      for method in "${METHODS[@]}"; do
        schedule_job "$task" "$method" "$model" "$input_model"
      done
    done
  done
  wait_for_all_jobs
  kill "$heartbeat_pid" 2>/dev/null || true
  pkill -P "$heartbeat_pid" 2>/dev/null || true
fi

if (( failed_jobs > 0 )); then
  echo "$failed_jobs are-you-sure job(s) failed (see $LOG_DIR)." >&2
  exit 1
fi
echo "All are-you-sure probes finished."
