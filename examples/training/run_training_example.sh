# Environment variables for training speechlmm-v2 with llamafactory.

# Training config and run name
export CONFIG=/path/to/your_training_config.yaml
export RUN_NAME=test_train

# Paths as seen by run_llamafactory.sh
# Inside a container these are the mount targets; on bare metal set them to the host paths.
export SHARED_DIR=/external_data/speechlmm-v2
export SCRATCH_DIR=/scratch_data/speechlmm-v2

# CUDA
export CUDA_HOME=/usr/local/cuda-12.6

# Wandb
export WANDB_MODE=offline
export WANDB_PROJECT="speechlmm-v2-training_step1"
export WANDB_LOG_MODEL="false"

# GPUs
export CUDA_VISIBLE_DEVICES=5,6

###################### (Optional) Singularity parameters (for run_sif.sh) ######################
# Singularity image
export SIF=/path/to/speechlmm_v2_train.sif

# Bind mounts (host path -> container path)
export HOST_SHARED_DIR=/path/to/speechlmm_v2-shared
export HOST_SCRATCH_DIR=/path/to/speechlmm_v2-scratch


bash scripts/training/run_llamafactory.sh ${CONFIG}

# If using Singularity, run the following command instead (make sure to set the SIF and bind mounts above):
# bash scripts/training/run_sif.sh