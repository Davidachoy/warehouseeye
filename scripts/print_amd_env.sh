#!/usr/bin/env bash
# Print AMD-related env vars after sourcing .env (no secrets expected; AMD_API_KEY is masked if set).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

mask() {
  local v="$1"
  if [[ -z "${v}" ]]; then
    echo "<empty>"
  elif [[ ${#v} -le 8 ]]; then
    echo "<set>"
  else
    echo "${v:0:4}…${v: -4}"
  fi
}

echo "WAREHOUSEEYE_DATA_ROOT=${WAREHOUSEEYE_DATA_ROOT:-data}"
echo "AMD_URL=${AMD_URL:-<empty>}"
echo "AMD_PROFILE=${AMD_PROFILE:-<empty>}"
echo "AMD_MODEL=${AMD_MODEL:-<unset>}"
echo "AMD_MODEL_DEV=${AMD_MODEL_DEV:-<empty>}"
echo "AMD_MODEL_PROD=${AMD_MODEL_PROD:-<empty>}"
echo "AMD_CONCURRENCY=${AMD_CONCURRENCY:-<unset>}"
echo "AMD_TIMEOUT_SEC=${AMD_TIMEOUT_SEC:-<unset>}"
echo "AMD_MAX_TOKENS=${AMD_MAX_TOKENS:-<unset>}"
if [[ -n "${AMD_API_KEY:-}" ]]; then
  echo "AMD_API_KEY=$(mask "${AMD_API_KEY}")"
else
  echo "AMD_API_KEY=<unset>"
fi
