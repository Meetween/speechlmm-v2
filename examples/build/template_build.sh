#!/bin/bash

set -euo pipefail

# Environment variables for building Docker images and singularity images (.sif) for training of speechlmm-v2.
export BASE_IMAGE_BASENAME=meetween_base_cuda12-8
export BASE_IMAGE_VERSION="0.1.2"

export TRAIN_IMAGE_BASENAME=meetween_speechlmm_v2_cuda12-8
export TRAIN_IMAGE_VERSION="0.1.9"

export TRAIN_SIF_DIR="PATH_TO_OUTPUT_SIF_DIR"

# Build the Docker image for training of speechlmm-v2.
source scripts/build/env.sh
bash scripts/build/build_base.sh
bash scripts/build/build_train.sh
bash scripts/build/build_sif.sh
