#!/usr/bin/env bash
# Launch dual vLLM services: 32B semantic model (:8000) + embedding model (:8001).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"

HOST="${HOST:-0.0.0.0}"
PRIMARY_PORT="${PRIMARY_PORT:-8000}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8001}"
PRIMARY_MODEL="${PRIMARY_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-VL-Embedding-2B}"
PRIMARY_GPU_MEM_UTIL="${PRIMARY_GPU_MEM_UTIL:-0.6}"
EMBEDDING_GPU_MEM_UTIL="${EMBEDDING_GPU_MEM_UTIL:-0.05}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
HEALTHCHECK_ATTEMPTS="${HEALTHCHECK_ATTEMPTS:-120}"
HEALTHCHECK_SLEEP_SEC="${HEALTHCHECK_SLEEP_SEC:-2}"
START_PRIMARY_IF_MISSING="${START_PRIMARY_IF_MISSING:-1}"
EMBEDDING_LAUNCH_MODE="${EMBEDDING_LAUNCH_MODE:-runner}"

mkdir -p "${ROOT}/data"
PRIMARY_LOG="${PRIMARY_LOG:-${ROOT}/data/vllm_32b.log}"
EMBEDDING_LOG="${EMBEDDING_LOG:-${ROOT}/data/vllm_embedding.log}"

wait_for_models() {
  local endpoint="$1"
  local attempts=0
  while (( attempts < HEALTHCHECK_ATTEMPTS )); do
    if curl -fsS "${endpoint}" >/dev/null 2>&1; then
      return 0
    fi
    attempts=$((attempts + 1))
    sleep "${HEALTHCHECK_SLEEP_SEC}"
  done
  return 1
}

PRIMARY_MODELS_URL="http://localhost:${PRIMARY_PORT}/v1/models"
EMBEDDING_MODELS_URL="http://localhost:${EMBEDDING_PORT}/v1/models"

if curl -fsS "${PRIMARY_MODELS_URL}" >/dev/null 2>&1; then
  echo "[ok] Primary semantic endpoint already healthy at ${PRIMARY_MODELS_URL}"
else
  if [[ "${START_PRIMARY_IF_MISSING}" != "1" ]]; then
    echo "[error] Primary endpoint ${PRIMARY_MODELS_URL} is not reachable."
    echo "[hint] Start 32B vLLM first with --gpu-memory-utilization ${PRIMARY_GPU_MEM_UTIL}."
    exit 1
  fi

  echo "[run] Launching primary model ${PRIMARY_MODEL} on :${PRIMARY_PORT} (gpu mem util ${PRIMARY_GPU_MEM_UTIL})"
  vllm serve "${PRIMARY_MODEL}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${PRIMARY_GPU_MEM_UTIL}" \
    --host "${HOST}" \
    --port "${PRIMARY_PORT}" >"${PRIMARY_LOG}" 2>&1 &
  echo "[run] Primary log: ${PRIMARY_LOG}"
fi

if curl -fsS "${EMBEDDING_MODELS_URL}" >/dev/null 2>&1; then
  echo "[ok] Embedding endpoint already healthy at ${EMBEDDING_MODELS_URL}"
else
  echo "[run] Launching embedding model ${EMBEDDING_MODEL} on :${EMBEDDING_PORT} (gpu mem util ${EMBEDDING_GPU_MEM_UTIL})"
  if [[ "${EMBEDDING_LAUNCH_MODE}" == "task" ]]; then
    vllm serve "${EMBEDDING_MODEL}" \
      --task embed \
      --gpu-memory-utilization "${EMBEDDING_GPU_MEM_UTIL}" \
      --host "${HOST}" \
      --port "${EMBEDDING_PORT}" >"${EMBEDDING_LOG}" 2>&1 &
  else
    # vLLM recipes now favor pooling runner for embedding services.
    vllm serve "${EMBEDDING_MODEL}" \
      --runner pooling \
      --gpu-memory-utilization "${EMBEDDING_GPU_MEM_UTIL}" \
      --host "${HOST}" \
      --port "${EMBEDDING_PORT}" >"${EMBEDDING_LOG}" 2>&1 &
  fi
  echo "[run] Embedding log: ${EMBEDDING_LOG}"
fi

echo "[health] Waiting for primary endpoint..."
wait_for_models "${PRIMARY_MODELS_URL}" || {
  echo "[error] Primary endpoint failed health checks: ${PRIMARY_MODELS_URL}"
  exit 1
}

echo "[health] Waiting for embedding endpoint..."
wait_for_models "${EMBEDDING_MODELS_URL}" || {
  echo "[error] Embedding endpoint failed health checks: ${EMBEDDING_MODELS_URL}"
  exit 1
}

echo
echo "[verify] curl ${PRIMARY_MODELS_URL}"
curl -fsS "${PRIMARY_MODELS_URL}"
echo
echo "[verify] curl ${EMBEDDING_MODELS_URL}"
curl -fsS "${EMBEDDING_MODELS_URL}"
echo
echo "[done] Both vLLM endpoints are healthy."
