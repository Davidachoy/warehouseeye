#!/usr/bin/env bash
# Run on the AMD GPU droplet (Ubuntu + ROCm), not on macOS.
set -euo pipefail

info() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33mWARN:\033[0m %s\n' "$*" >&2; }
die() { printf '\n\033[1;31mERR:\033[0m %s\n' "$*" >&2; exit 1; }

info "GPU (rocm-smi)"
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showproductname 2>/dev/null || rocm-smi
else
  die "rocm-smi not found. Use an AMD ROCm image on the droplet."
fi

info "ROCm version"
ROCM_VER=""
if [[ -f /opt/rocm/.info/version ]]; then
  ROCM_VER="$(tr -d '\n' </opt/rocm/.info/version)"
elif [[ -f /opt/rocm/VERSION ]]; then
  ROCM_VER="$(head -1 /opt/rocm/VERSION | tr -d '\n')"
fi
if [[ -n "${ROCM_VER}" ]]; then
  echo "Detected ROCm: ${ROCM_VER}"
  # Loose check: major version digit should be 7+
  if [[ ! "${ROCM_VER}" =~ ^7\. ]]; then
    warn "Expected ROCm >= 7.0 for current vLLM ROCm wheels; found '${ROCM_VER}'."
  fi
else
  warn "Could not read /opt/rocm version file; verify ROCm >= 7.0 manually."
fi

info "Python"
if command -v python3.12 >/dev/null 2>&1; then
  PY="python3.12"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null; then
  PY="python3"
else
  warn "Python 3.12 not found as python3.12. Install 3.12 or use https://astral.sh/uv"
  PY="python3"
fi
echo "Using: $($PY -V 2>&1)"

info "uv (recommended)"
if ! command -v uv >/dev/null 2>&1; then
  warn "uv not installed. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  die "Install uv, then re-run this script."
fi

VENV_DIR="${VENV_DIR:-${HOME}/.venvs/warehouseeye-amd-vllm}"
info "Creating venv at ${VENV_DIR}"
if uv venv --python 3.12 "${VENV_DIR}" 2>/dev/null; then
  :
else
  warn "Could not create venv with Python 3.12; falling back to default python for uv."
  uv venv "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

info "Installing vLLM (ROCm wheels; may take several minutes)"
uv pip install -U pip
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm

info "Torch / CUDA (ROCm) smoke test"
python - <<'PY'
import torch

print("torch:", torch.__version__)
ok = torch.cuda.is_available()
print("torch.cuda.is_available():", ok)
if not ok:
    raise SystemExit("CUDA/ROCm device not visible to PyTorch.")
print("device count:", torch.cuda.device_count())
if torch.cuda.device_count():
    print("device 0:", torch.cuda.get_device_name(0))
PY

info "Summary"
echo "  Venv:  ${VENV_DIR}"
echo "  Activate: source ${VENV_DIR}/bin/activate"
echo "  Next: bash scripts/amd/serve_qwen3vl_8b.sh"
