#!/usr/bin/env bash
# Backup prerendered artifacts before manual instance teardown.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

PRERENDER_DIR="${PRERENDER_DIR:-data/prerendered}"
if [[ ! -d "${PRERENDER_DIR}" ]]; then
  echo "[backup] ERROR: missing directory ${PRERENDER_DIR}"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_NAME="warehouseeye_prerendered_${STAMP}.tar.gz"
ARCHIVE_PATH="${ROOT}/${ARCHIVE_NAME}"

echo "[backup] creating archive ${ARCHIVE_PATH}"
tar czf "${ARCHIVE_PATH}" -C "${ROOT}" "${PRERENDER_DIR}"

echo "[backup] verifying archive contents"
tar -tzf "${ARCHIVE_PATH}" >/dev/null

if command -v shasum >/dev/null 2>&1; then
  ARCHIVE_SHA256="$(shasum -a 256 "${ARCHIVE_PATH}" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  ARCHIVE_SHA256="$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')"
else
  ARCHIVE_SHA256="unavailable (install shasum or sha256sum)"
fi

echo "[backup] sha256: ${ARCHIVE_SHA256}"

if [[ -n "${AWS_ACCESS_KEY_ID:-}" && -n "${AWS_SECRET_ACCESS_KEY:-}" && -n "${AWS_S3_BUCKET:-}" ]]; then
  if command -v aws >/dev/null 2>&1; then
    S3_KEY="${AWS_S3_KEY_PREFIX:-warehouseeye}/${ARCHIVE_NAME}"
    S3_URI="s3://${AWS_S3_BUCKET}/${S3_KEY}"
    echo "[backup] uploading to ${S3_URI}"
    aws s3 cp "${ARCHIVE_PATH}" "${S3_URI}"
    echo "[backup] upload complete"
  else
    echo "[backup] AWS credentials found but aws CLI is unavailable."
  fi
else
  echo "[backup] AWS env vars not fully configured; skipping S3 upload."
fi

echo
echo "Archive ready: ${ARCHIVE_PATH}"
echo "Fallback copy command (scp):"
echo "scp \"${ARCHIVE_PATH}\" user@your-host:/path/to/backups/"

echo
read -r -p "Show irreversible DigitalOcean destroy command? [y/N] " CONFIRM_DESTROY
if [[ "${CONFIRM_DESTROY}" =~ ^[Yy]$ ]]; then
  if [[ -z "${DROPLET_ID:-}" ]]; then
    echo "Set DROPLET_ID in .env before destroying."
    echo "Example: doctl compute droplet delete <DROPLET_ID> --force"
  else
    echo "Run this command ONLY when backup is confirmed:"
    echo "doctl compute droplet delete ${DROPLET_ID} --force"
  fi
  echo "DigitalOcean docs: https://docs.digitalocean.com/reference/doctl/reference/compute/droplet/delete/"
else
  echo "Destroy command not shown."
fi
