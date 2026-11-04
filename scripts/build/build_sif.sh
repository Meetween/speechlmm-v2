#!/bin/bash

set -euo pipefail

if [ -z "${TRAIN_IMAGE_NAME+x}" ]; then
  echo "Variable TRAIN_IMAGE_NAME undefined"
  exit 1
elif [ -z "$TRAIN_IMAGE_NAME" ]; then
  echo "Variable TRAIN_IMAGE_NAME defined but empty"
  exit 1
fi

if [ -z "${TRAIN_IMAGE_VERSION+x}" ]; then
  echo "Variable TRAIN_IMAGE_VERSION undefined"
  exit 1
elif [ -z "$TRAIN_IMAGE_VERSION" ]; then
  echo "Variable TRAIN_IMAGE_VERSION defined but empty"
  exit 1
fi

if [ -z "${TRAIN_SIF_PATH+x}" ]; then
  echo "Variable TRAIN_SIF_PATH undefined"
  exit 1
elif [ -z "$TRAIN_SIF_PATH" ]; then
  echo "Variable TRAIN_SIF_PATH defined but empty"
  exit 1
fi

# create singularity image (.sif); it assumes to have local repo for docker images
docker tag ${TRAIN_IMAGE_NAME}:${TRAIN_IMAGE_VERSION} localhost:5000/${TRAIN_IMAGE_NAME}:${TRAIN_IMAGE_VERSION}
docker push localhost:5000/${TRAIN_IMAGE_NAME}:${TRAIN_IMAGE_VERSION}
singularity --version
singularity pull --no-https --name ${TRAIN_SIF_PATH} docker://localhost:5000/${TRAIN_IMAGE_NAME}:${TRAIN_IMAGE_VERSION}
