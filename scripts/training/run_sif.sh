#!/usr/bin/env bash
# Run llamafactory training inside a Singularity container.

uname -a

singularity exec \
    --nv \
    -B ${HOST_SHARED_DIR}:${SHARED_DIR} \
    -B ${HOST_SCRATCH_DIR}:${SCRATCH_DIR} \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env WANDB_MODE="${WANDB_MODE}" \
    --env WANDB_PROJECT="${WANDB_PROJECT}" \
    --env WANDB_LOG_MODEL="${WANDB_LOG_MODEL}" \
    --env RUN_NAME="${RUN_NAME}" \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    --env CUDA_HOME="${CUDA_HOME}" \
    --env SHARED_DIR="${SHARED_DIR}" \
    --env SCRATCH_DIR="${SCRATCH_DIR}" \
    ${SIF} \
    scripts/training/run_llamafactory.sh ${CONFIG}
