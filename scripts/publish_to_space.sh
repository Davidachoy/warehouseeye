#!/usr/bin/env bash
set -euo pipefail

# Publish WarehouseEye HF Space from local repository artifacts.
# Required env:
#   HF_SPACE_REPO=<org_or_user>/warehouseeye
#
# Optional env:
#   LOCAL_REPO_ROOT=/path/to/warehouseeye
#   HF_USERNAME=<hf_username>
#   HF_TOKEN=<hf_token>
#
# Size verification:
#   The script prints an estimated payload size from copied files.
#   Keep this well under 1 GB by excluding frames/, audio/, *.sqlite3, and *.log.

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is required." >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "Error: rsync is required." >&2
  exit 1
fi

if ! command -v du >/dev/null 2>&1; then
  echo "Error: du is required." >&2
  exit 1
fi

if [[ -z "${HF_SPACE_REPO:-}" ]]; then
  echo "Error: set HF_SPACE_REPO (example: your-org/warehouseeye)." >&2
  exit 1
fi

if [[ -n "${LOCAL_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${LOCAL_REPO_ROOT}"
else
  REPO_ROOT="$(git rev-parse --show-toplevel)"
fi

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "Error: LOCAL_REPO_ROOT does not exist: ${REPO_ROOT}" >&2
  exit 1
fi

WORK_ROOT="$(mktemp -d)"
SPACE_CLONE_DIR="${WORK_ROOT}/warehouseeye-space"
SPACE_URL="https://huggingface.co/spaces/${HF_SPACE_REPO}"

if [[ -n "${HF_USERNAME:-}" && -n "${HF_TOKEN:-}" ]]; then
  CLONE_URL="https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE_REPO}"
else
  CLONE_URL="https://huggingface.co/spaces/${HF_SPACE_REPO}"
fi

echo "Cloning ${SPACE_URL} to ${SPACE_CLONE_DIR}..."
git clone "${CLONE_URL}" "${SPACE_CLONE_DIR}"

copy_required_file() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "${src}" ]]; then
    echo "Error: missing required file ${src}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${dst}")"
  cp "${src}" "${dst}"
}

copy_optional_file() {
  local src="$1"
  local dst="$2"
  if [[ -f "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp "${src}" "${dst}"
  fi
}

echo "Copying app source files..."
copy_required_file "${REPO_ROOT}/frontend/app_space.py" "${SPACE_CLONE_DIR}/frontend/app_space.py"
copy_required_file "${REPO_ROOT}/frontend/components/overview_tab.py" "${SPACE_CLONE_DIR}/frontend/components/overview_tab.py"
copy_required_file "${REPO_ROOT}/frontend/components/performance_tab.py" "${SPACE_CLONE_DIR}/frontend/components/performance_tab.py"
copy_required_file "${REPO_ROOT}/frontend/components/space_layout.py" "${SPACE_CLONE_DIR}/frontend/components/space_layout.py"
copy_required_file "${REPO_ROOT}/frontend/components/space_query_tab.py" "${SPACE_CLONE_DIR}/frontend/components/space_query_tab.py"
copy_required_file "${REPO_ROOT}/frontend/services/space_data.py" "${SPACE_CLONE_DIR}/frontend/services/space_data.py"

copy_optional_file "${REPO_ROOT}/frontend/__init__.py" "${SPACE_CLONE_DIR}/frontend/__init__.py"
copy_optional_file "${REPO_ROOT}/frontend/components/__init__.py" "${SPACE_CLONE_DIR}/frontend/components/__init__.py"
copy_optional_file "${REPO_ROOT}/frontend/services/__init__.py" "${SPACE_CLONE_DIR}/frontend/services/__init__.py"

echo "Copying Space metadata files..."
copy_required_file "${REPO_ROOT}/space/README.md" "${SPACE_CLONE_DIR}/README.md"
copy_required_file "${REPO_ROOT}/space/requirements.txt" "${SPACE_CLONE_DIR}/requirements.txt"

echo "Copying pre-rendered payload..."
mkdir -p "${SPACE_CLONE_DIR}/data/prerendered"
for video_dir in "${REPO_ROOT}"/data/prerendered/*; do
  if [[ ! -d "${video_dir}" ]]; then
    continue
  fi

  video_id="$(basename "${video_dir}")"
  if [[ ! -f "${video_dir}/timeline.json" ]]; then
    continue
  fi

  target="${SPACE_CLONE_DIR}/data/prerendered/${video_id}"
  mkdir -p "${target}/videos" "${target}/crops"

  copy_required_file "${video_dir}/timeline.json" "${target}/timeline.json"
  copy_optional_file "${video_dir}/benchmarks.json" "${target}/benchmarks.json"
  copy_optional_file "${video_dir}/videos/input_video.mp4" "${target}/videos/input_video.mp4"

  if [[ -d "${video_dir}/crops" ]]; then
    rsync -a --delete \
      --include='*/' \
      --include='*.jpg' \
      --include='*.jpeg' \
      --include='*.png' \
      --include='*.webp' \
      --exclude='*' \
      "${video_dir}/crops/" "${target}/crops/"
  fi
done

echo "Estimated copied payload size:"
du -sh "${SPACE_CLONE_DIR}/data/prerendered" || true

echo "Creating commit and pushing..."
cd "${SPACE_CLONE_DIR}"
git add -A
if git diff --cached --quiet; then
  echo "No changes to push."
else
  git commit -m "deploy: refresh WarehouseEye demo"
  git push origin HEAD
fi

echo "Space live at: ${SPACE_URL}"
