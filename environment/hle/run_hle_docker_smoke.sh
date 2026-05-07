#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_ROOT="${ROOT_DIR}/environment/hle/results"

API_KEY="${EMMA_OPENAI_API_KEY:-${MEMRL_OPENAI_API_KEY:-}}"
EMBEDDING_KEY="${EMMA_EMBEDDING_API_KEY:-${MEMRL_EMBEDDING_API_KEY:-${API_KEY}}}"
BASE_URL="${EMMA_OPENAI_BASE_URL:-${MEMRL_OPENAI_BASE_URL:-https://api.openai.com/v1}}"
EMBEDDING_BASE_URL="${EMMA_EMBEDDING_BASE_URL:-${MEMRL_EMBEDDING_BASE_URL:-${BASE_URL}}}"
SOLVER_MODEL="${EMMA_OPENAI_MODEL:-${MEMRL_OPENAI_MODEL:-gpt-4o-mini}}"
PRIMARY_MODEL="${EMMA_ROUTING_PRIMARY_MODEL:-${MEMRL_ROUTING_PRIMARY_MODEL:-${SOLVER_MODEL}}}"
SECONDARY_MODEL="${EMMA_ROUTING_SECONDARY_MODEL:-${MEMRL_ROUTING_SECONDARY_MODEL:-}}"
SECONDARY_PROTOCOL="${EMMA_ROUTING_SECONDARY_PROTOCOL:-${MEMRL_ROUTING_SECONDARY_PROTOCOL:-}}"
ROUTING_ENABLED="${EMMA_ROUTING_ENABLED:-${MEMRL_ROUTING_ENABLED:-0}}"
ROUTING_TRIGGER_MODE="${EMMA_ROUTING_TRIGGER_MODE:-${MEMRL_ROUTING_TRIGGER_MODE:-disabled}}"
EMBEDDING_MODEL="${EMMA_EMBEDDING_MODEL:-${MEMRL_EMBEDDING_MODEL:-text-embedding-3-large}}"
JUDGE_MODE="${EMMA_HLE_JUDGE_MODE:-${MEMRL_HLE_JUDGE_MODE:-local_exact_match}}"
JUDGE_MODEL="${EMMA_HLE_JUDGE_MODEL:-${MEMRL_HLE_JUDGE_MODEL:-gpt-4o-2024-08-06}}"
DATASET_NAME="${EMMA_HLE_DATASET_NAME:-${MEMRL_HLE_DATASET_NAME:-koiwave/hle-short}}"
CONDITION="${EMMA_CONDITION:-${MEMRL_CONDITION:-full}}"
EPISODES="${EMMA_EPISODES:-${MEMRL_EPISODES:-1}}"
START_INDEX="${EMMA_START_INDEX:-${MEMRL_START_INDEX:-0}}"
IMAGE_NAME="${EMMA_HLE_DOCKER_IMAGE:-${MEMRL_HLE_DOCKER_IMAGE:-texlive/texlive:latest}}"
RESULTS_NAME="${EMMA_RESULTS_NAME:-${MEMRL_RESULTS_NAME:-smoke_${SOLVER_MODEL}_${JUDGE_MODE}}}"
RESULTS_DIR="environment/hle/results/${RESULTS_NAME}"

require_nonempty() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    echo "[error] missing required env: ${name}" >&2
    exit 1
  fi
}

model_requires_confirmation() {
  local model
  model="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  [[ "${model}" == *"gemini"* ]] || [[ "${model}" == *"claude"* ]] || [[ "${model}" == *"opus"* ]] || [[ "${model}" == *"sonnet"* ]] || [[ "${model}" == *"gpt-5"* ]] || [[ "${model}" == *"o1"* ]] || [[ "${model}" == *"o3"* ]]
}

guard_solver_model() {
  if model_requires_confirmation "${SOLVER_MODEL}" && [[ "${EMMA_ALLOW_EXPENSIVE_SOLVER:-${MEMRL_ALLOW_EXPENSIVE_SOLVER:-0}}" != "1" ]]; then
    echo "[blocked] solver model '${SOLVER_MODEL}' looks expensive." >&2
    echo "[blocked] re-run with EMMA_ALLOW_EXPENSIVE_SOLVER=1 if you really want it." >&2
    exit 2
  fi
}

require_nonempty "EMMA_OPENAI_API_KEY" "${API_KEY}"
require_nonempty "EMMA_EMBEDDING_API_KEY" "${EMBEDDING_KEY}"
guard_solver_model
if [[ -n "${SECONDARY_MODEL}" ]]; then
  if model_requires_confirmation "${SECONDARY_MODEL}" && [[ "${EMMA_ALLOW_EXPENSIVE_SOLVER:-${MEMRL_ALLOW_EXPENSIVE_SOLVER:-0}}" != "1" ]]; then
    echo "[blocked] secondary solver model '${SECONDARY_MODEL}' looks expensive." >&2
    echo "[blocked] re-run with EMMA_ALLOW_EXPENSIVE_SOLVER=1 if you really want it." >&2
    exit 2
  fi
fi

mkdir -p "${RESULTS_ROOT}"

echo "[run] benchmark=hle condition=${CONDITION} episodes=${EPISODES} start_index=${START_INDEX}"
echo "[run] solver_model=${SOLVER_MODEL} judge_mode=${JUDGE_MODE} judge_model=${JUDGE_MODEL}"
echo "[run] routing_enabled=${ROUTING_ENABLED} primary_model=${PRIMARY_MODEL} secondary_model=${SECONDARY_MODEL:-none} secondary_protocol=${SECONDARY_PROTOCOL:-auto} trigger_mode=${ROUTING_TRIGGER_MODE}"
echo "[run] results_dir=${RESULTS_DIR}"

docker run --rm \
  -e DEBIAN_FRONTEND=noninteractive \
  -e EMMA_OPENAI_API_KEY="${API_KEY}" \
  -e EMMA_EMBEDDING_API_KEY="${EMBEDDING_KEY}" \
  -e EMMA_OPENAI_BASE_URL="${BASE_URL}" \
  -e EMMA_EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL}" \
  -e EMMA_OPENAI_MODEL="${SOLVER_MODEL}" \
  -e EMMA_ROUTING_ENABLED="${ROUTING_ENABLED}" \
  -e EMMA_ROUTING_PRIMARY_MODEL="${PRIMARY_MODEL}" \
  -e EMMA_ROUTING_SECONDARY_MODEL="${SECONDARY_MODEL}" \
  -e EMMA_ROUTING_SECONDARY_PROTOCOL="${SECONDARY_PROTOCOL}" \
  -e EMMA_ROUTING_TRIGGER_MODE="${ROUTING_TRIGGER_MODE}" \
  -e EMMA_EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
  -e EMMA_HLE_DATASET_NAME="${DATASET_NAME}" \
  -e EMMA_HLE_JUDGE_MODE="${JUDGE_MODE}" \
  -e EMMA_HLE_JUDGE_MODEL="${JUDGE_MODEL}" \
  -e MEMRL_OPENAI_API_KEY="${API_KEY}" \
  -e MEMRL_EMBEDDING_API_KEY="${EMBEDDING_KEY}" \
  -e MEMRL_OPENAI_BASE_URL="${BASE_URL}" \
  -e MEMRL_EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL}" \
  -e MEMRL_OPENAI_MODEL="${SOLVER_MODEL}" \
  -e MEMRL_ROUTING_ENABLED="${ROUTING_ENABLED}" \
  -e MEMRL_ROUTING_PRIMARY_MODEL="${PRIMARY_MODEL}" \
  -e MEMRL_ROUTING_SECONDARY_MODEL="${SECONDARY_MODEL}" \
  -e MEMRL_ROUTING_SECONDARY_PROTOCOL="${SECONDARY_PROTOCOL}" \
  -e MEMRL_ROUTING_TRIGGER_MODE="${ROUTING_TRIGGER_MODE}" \
  -e MEMRL_EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
  -e MEMRL_HLE_DATASET_NAME="${DATASET_NAME}" \
  -e MEMRL_HLE_JUDGE_MODE="${JUDGE_MODE}" \
  -e MEMRL_HLE_JUDGE_MODEL="${JUDGE_MODEL}" \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace \
  "${IMAGE_NAME}" \
  bash -lc "apt-get update >/tmp/apt-update.log 2>&1 && \
    apt-get install -y python3-pip >/tmp/apt-install.log 2>&1 && \
    python3 -m pip install --break-system-packages -r environment/hle/requirements.txt >/tmp/pip-install.log 2>&1 && \
    python3 environment/run_memrl_benchmark.py --benchmark hle --condition ${CONDITION} --episodes ${EPISODES} --start-index ${START_INDEX} --results-dir ${RESULTS_DIR}"

echo "[done] ${ROOT_DIR}/${RESULTS_DIR}/hle_${CONDITION}.json"
