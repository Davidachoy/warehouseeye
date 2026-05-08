#!/usr/bin/env bash
# Run WarehouseEye Streamlit frontend locally. Default: http://127.0.0.1:8501
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
export API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
PORT="${FRONTEND_PORT:-8501}"

echo "Starting Streamlit frontend on http://127.0.0.1:${PORT}"
exec streamlit run frontend/app.py --server.port "${PORT}"
