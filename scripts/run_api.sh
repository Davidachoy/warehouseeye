#!/usr/bin/env bash
# Run WarehouseEye FastAPI locally (Mac/Linux). Default: http://127.0.0.1:8000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH:-.}"

HOST="${API_HOST:-127.0.0.1}"
PORT="${API_PORT:-8000}"

echo "Starting uvicorn api.main:app on http://${HOST}:${PORT}"
exec uvicorn api.main:app --reload --host "${HOST}" --port "${PORT}"
