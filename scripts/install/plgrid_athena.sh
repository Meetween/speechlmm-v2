#!/bin/bash
# Install speechlmm-v2 on PLGrid Athena (CUDA 12.8, cu128 wheels).
# All caches and uv itself live on $SCRATCH — not $HOME (home quota is ~10 GB).
# See README_plgrid-athena.md for full setup.
#
# Usage:
#   cd "$SCRATCH/speechlmm-v2"
#   bash scripts/install/plgrid_athena.sh

set -euo pipefail

: "${SCRATCH:?SCRATCH is not set — run on a PLGrid login node}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

export SPEECHLMM_DIR="${SPEECHLMM_DIR:-${ROOT}}"
export SPEECHLMM_VENV="${SPEECHLMM_VENV:-${SPEECHLMM_DIR}/.venv}"
VENV="${SPEECHLMM_VENV}"

export UV_INSTALL_DIR="${SCRATCH}/.local"
export UV_PYTHON_INSTALL_DIR="${SCRATCH}/.local/share/uv/python"
export UV_CACHE_DIR="${SCRATCH}/.cache/uv"
export UV_TOOL_DIR="${SCRATCH}/.local/share/uv/tools"
export HF_HOME="${SCRATCH}/.cache/huggingface"
export PATH="${UV_INSTALL_DIR}:${PATH}"

mkdir -p "${UV_INSTALL_DIR}" "${UV_CACHE_DIR}" "${HF_HOME}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv to ${UV_INSTALL_DIR} ..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${UV_INSTALL_DIR}" sh
fi

if [[ ! -d "${VENV}" ]]; then
    uv venv "${VENV}" --python=3.12
fi
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
uv pip install -e ".[deepspeed,logging]" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match

echo "Done."
echo "  venv:  source ${VENV}/bin/activate"
echo "  uv:    ${UV_INSTALL_DIR}/uv"
echo "  cache: ${UV_CACHE_DIR}"
