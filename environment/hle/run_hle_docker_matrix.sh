#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_SCRIPT="${ROOT_DIR}/environment/hle/run_hle_docker_smoke.sh"

SOLVER_MODEL="${EMMA_OPENAI_MODEL:-${MEMRL_OPENAI_MODEL:-gpt-4o-mini}}"
EMBEDDING_MODEL="${EMMA_EMBEDDING_MODEL:-${MEMRL_EMBEDDING_MODEL:-text-embedding-3-large}}"
EPISODES="${EMMA_EPISODES:-${MEMRL_EPISODES:-3}}"
START_INDEX="${EMMA_START_INDEX:-${MEMRL_START_INDEX:-0}}"
DATASET_NAME="${EMMA_HLE_DATASET_NAME:-${MEMRL_HLE_DATASET_NAME:-koiwave/hle-short}}"
JUDGE_MODEL="${EMMA_HLE_JUDGE_MODEL:-${MEMRL_HLE_JUDGE_MODEL:-gpt-4o-2024-08-06}}"

run_case() {
  local condition="$1"
  local judge_mode="$2"
  local results_name="$3"

  EMMA_CONDITION="${condition}" \
  EMMA_EPISODES="${EPISODES}" \
  EMMA_START_INDEX="${START_INDEX}" \
  EMMA_OPENAI_MODEL="${SOLVER_MODEL}" \
  EMMA_EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
  EMMA_HLE_DATASET_NAME="${DATASET_NAME}" \
  EMMA_HLE_JUDGE_MODE="${judge_mode}" \
  EMMA_HLE_JUDGE_MODEL="${JUDGE_MODEL}" \
  EMMA_RESULTS_NAME="${results_name}" \
  "${SMOKE_SCRIPT}"
}

echo "[matrix] solver_model=${SOLVER_MODEL} judge_model=${JUDGE_MODEL} episodes=${EPISODES}"

run_case "full" "local_exact_match" "matrix_full_local_${EPISODES}"
run_case "full" "llm_judge" "matrix_full_llmjudge_${EPISODES}"
run_case "no_memory" "local_exact_match" "matrix_nomemory_local_${EPISODES}"
run_case "no_memory" "llm_judge" "matrix_nomemory_llmjudge_${EPISODES}"

echo "[done] matrix complete under environment/hle/results/"
