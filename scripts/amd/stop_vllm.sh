#!/usr/bin/env bash
# Best-effort: stop vLLM listening on PORT (default 8000) on the droplet.
set -euo pipefail

PORT="${PORT:-8000}"

if command -v fuser >/dev/null 2>&1; then
  echo "Trying fuser -k on TCP ${PORT}..."
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi

if command -v pkill >/dev/null 2>&1; then
  echo "Trying pkill -f 'vllm serve'..."
  pkill -f "vllm serve" 2>/dev/null || true
fi

echo "Done. If something is still bound, run: ss -lntp | grep ${PORT}"
