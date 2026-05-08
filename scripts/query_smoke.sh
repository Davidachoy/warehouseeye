#!/usr/bin/env bash
# Smoke POST /query. Set WAREHOUSEEYE_VIDEO_ID to a folder under data/ that has warehouseeye.sqlite3.
set -euo pipefail

BASE="${WAREHOUSEEYE_API_BASE:-http://127.0.0.1:8000}"
VIDEO_ID="${WAREHOUSEEYE_VIDEO_ID:?Set WAREHOUSEEYE_VIDEO_ID (e.g. export WAREHOUSEEYE_VIDEO_ID=video)}"

Q1="${Q1:-how many people are there?}"
Q2="${Q2:-what did the person in the orange vest do?}"

payload() {
  python3 -c 'import json,sys; print(json.dumps({"video_id":sys.argv[1],"question":sys.argv[2]}))' "$1" "$2"
}

echo "=== Query 1 (SQLite-friendly): ${Q1}"
curl -sS -X POST "${BASE}/query" \
  -H "Content-Type: application/json" \
  -d "$(payload "${VIDEO_ID}" "${Q1}")" | python3 -m json.tool

echo ""
echo "=== Query 2: ${Q2}"
curl -sS -X POST "${BASE}/query" \
  -H "Content-Type: application/json" \
  -d "$(payload "${VIDEO_ID}" "${Q2}")" | python3 -m json.tool
