#!/usr/bin/env bash
# Run on the AMD droplet after setup_amd.sh (same venv).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

VENV_DIR="${VENV_DIR:-${HOME}/.venvs/warehouseeye-amd-vllm}"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
fi

echo "Serving ${MODEL} on ${HOST}:${PORT} (bf16, max_model_len=8192, gpu_memory_utilization=0.5)"
exec vllm serve "${MODEL}" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.5 \
  --host "${HOST}" \
  --port "${PORT}"
