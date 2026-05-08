#!/usr/bin/env bash
# Run full WarehouseEye pipeline for final 32B prerender.
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

export PYTHONPATH="${PYTHONPATH:-.}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
AMD_BASE_URL="${AMD_URL:-http://127.0.0.1:${PORT}/v1}"
AMD_BASE_URL="${AMD_BASE_URL%/}"
if [[ "${AMD_BASE_URL}" != */v1 ]]; then
  AMD_BASE_URL="${AMD_BASE_URL}/v1"
fi
export AMD_URL="${AMD_BASE_URL}"
export AMD_MODEL="${AMD_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
export DATA_PRERENDER_ROOT="${DATA_PRERENDER_ROOT:-data/prerendered}"

mkdir -p "${DATA_PRERENDER_ROOT}"
STATUS_DB="${DATA_PRERENDER_ROOT}/_pipeline_status.sqlite3"

VLLM_PID=""

wait_for_vllm() {
  local attempts=0
  local max_attempts="${VLLM_HEALTHCHECK_MAX_ATTEMPTS:-180}"
  local sleep_sec="${VLLM_HEALTHCHECK_SLEEP_SEC:-5}"

  echo "[health] Waiting for vLLM at ${AMD_URL}/models"
  while (( attempts < max_attempts )); do
    if curl -fsS "${AMD_URL}/models" >/dev/null 2>&1; then
      echo "[health] vLLM ready."
      return 0
    fi
    attempts=$((attempts + 1))
    sleep "${sleep_sec}"
  done

  echo "[health] ERROR: vLLM did not become healthy in time."
  return 1
}

if [[ "${SKIP_VLLM:-0}" != "1" ]]; then
  echo "[run] Launching 32B vLLM server..."
  "${ROOT}/scripts/amd/serve_qwen3vl_32b.sh" > "${DATA_PRERENDER_ROOT}/vllm_32b.log" 2>&1 &
  VLLM_PID="$!"
  echo "[run] vLLM PID: ${VLLM_PID} (log: ${DATA_PRERENDER_ROOT}/vllm_32b.log)"
fi

wait_for_vllm

VIDEO_1="${ROOT}/data/video.mp4"
VIDEO_2="${ROOT}/data/video2.mp4"

if [[ ! -f "${VIDEO_1}" ]]; then
  echo "[run] ERROR: required video missing: ${VIDEO_1}"
  exit 1
fi

declare -a RAN_VIDEOS=()
declare -a FAILED_VIDEOS=()

run_one() {
  local video_path="$1"
  local video_id
  video_id="$(basename "${video_path}")"
  video_id="${video_id%.*}"

  echo "[run] Starting ${video_id}"
  local started_at
  started_at="$(date +%s)"
  if python "${ROOT}/scripts/run_prerender_pipeline.py" \
    --video-path "${video_path}" \
    --video-id "${video_id}" \
    --data-root "${DATA_PRERENDER_ROOT}" \
    --status-db "${STATUS_DB}"; then
    local ended_at
    ended_at="$(date +%s)"
    local elapsed=$((ended_at - started_at))
    RAN_VIDEOS+=("${video_id}:${elapsed}")
    echo "[run] Completed ${video_id} in ${elapsed}s"
  else
    FAILED_VIDEOS+=("${video_id}")
    echo "[run] Failed ${video_id}"
    return 1
  fi
}

run_one "${VIDEO_1}"

if [[ -f "${VIDEO_2}" ]]; then
  run_one "${VIDEO_2}"
else
  echo "[run] Optional video not found, skipping: ${VIDEO_2}"
fi

echo
echo "========== Final summary =========="
echo "Output root: ${DATA_PRERENDER_ROOT}"
echo "Status DB: ${STATUS_DB}"
echo "Model: ${AMD_MODEL}"
echo "Endpoint: ${AMD_URL}"
for item in "${RAN_VIDEOS[@]}"; do
  video_id="${item%%:*}"
  elapsed="${item##*:}"
  echo "- ${video_id}: ${elapsed}s | artifacts: ${DATA_PRERENDER_ROOT}/${video_id}"
done
if [[ "${#FAILED_VIDEOS[@]}" -gt 0 ]]; then
  echo "Failed videos: ${FAILED_VIDEOS[*]}"
  exit 1
fi
echo "All requested videos completed successfully."

if [[ -n "${VLLM_PID}" ]]; then
  echo
  echo "vLLM is still running (PID ${VLLM_PID})."
  echo "Manual stop command (optional): kill ${VLLM_PID}"
fi
