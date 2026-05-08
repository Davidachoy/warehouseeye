#!/usr/bin/env bash
# GET /health from a running WarehouseEye API.
set -euo pipefail

BASE="${WAREHOUSEEYE_API_BASE:-http://127.0.0.1:8000}"

if command -v curl >/dev/null 2>&1; then
  curl -sS "${BASE}/health"
else
  python3 - <<PY
import json, sys, urllib.request
u = "${BASE}/health"
with urllib.request.urlopen(u, timeout=5) as r:
    print(r.read().decode())
PY
fi | python3 -m json.tool
