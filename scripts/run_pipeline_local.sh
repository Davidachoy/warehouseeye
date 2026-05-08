#!/usr/bin/env bash
# Wrapper around scripts/test_pipeline_local.py (repo root).
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

VIDEO="${1:-data/video.mp4}"
shift || true

exec python scripts/test_pipeline_local.py "${VIDEO}" \
  --base-dir data \
  --scene-threshold 0.25 \
  --sample-every-sec 0.5 \
  --detector-threshold 0.5 \
  --min-bbox-area 0 \
  --tracker-frame-rate 30.0 \
  --tracker-activation-threshold 0.25 \
  --tracker-lost-track-buffer 30 \
  --tracker-matching-threshold 0.8 \
  "$@"
