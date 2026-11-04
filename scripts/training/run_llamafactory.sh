#!/bin/bash
# Train with llamafactory.
# Expects CONFIG_PATH as first argument and environment variables from env.sh.
# Works both inside a container and on bare metal.

set -euo pipefail

CONFIG_PATH=$1

# Derive project root from this script's location (scripts/training/run_llamafactory.sh)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
LF_DIR="${PROJECT_DIR}/llamafactory"

# Detect GPUs
NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
err=$?
if [ $? -ne 0 ]; then echo "Command Failure" ; fi
if [ "${NPROC_PER_NODE}" -lt 1 ]; then echo "No CUDA devices visible; aborting." ; exit 1 ; fi

echo "Using NPROC_PER_NODE=${NPROC_PER_NODE}"
export HF_HOME="${SHARED_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

# Logging directories
export WANDB_DIR="${SCRATCH_DIR}/logging/wandb/${RUN_NAME}"
export TENSORBOARD_LOGGING_DIR="${SCRATCH_DIR}/logging/tensorboard/${RUN_NAME}"

# Debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCHELASTIC_ERROR_FILE="${SCRATCH_DIR}/debugging/logs/torchrun_error.json"

# transformers 5.2.0 fixes Qwen3-Omni config bug but exceeds llamafactory's strict <=5.0.0 bound
export DISABLE_VERSION_CHECK=1

if [ "${NPROC_PER_NODE}" -lt 1 ]; then
  echo "No CUDA devices visible; aborting."
  exit 1
fi

# Activate environment (optional, e.g. for bare metal runs)
if [ -n "${TRAIN_VENV_PATH:-}" ]; then
  source "${TRAIN_VENV_PATH}/bin/activate"
fi
# Wandb login (requires WANDB_API_KEY to be set in the environment)
wandb login
wandb offline

# Run training from the llamafactory directory so relative paths in the YAML resolve correctly
cd "${LF_DIR}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE="${NPROC_PER_NODE}" llamafactory-cli train "${CONFIG_PATH}"

echo "=== Training Complete ==="
outDir=$(grep ^output_dir "${CONFIG_PATH}" | cut -d' ' -f2)
echo "Output: ${outDir}"
