#!/bin/bash

set -euo pipefail

if [ -z "${BASE_IMAGE_NAME+x}" ]; then
  echo "Variable BASE_IMAGE_NAME undefined"
  exit 1
elif [ -z "$BASE_IMAGE_NAME" ]; then
  echo "Variable BASE_IMAGE_NAME defined but empty"
  exit 1
fi

if [ -z "${BASE_IMAGE_VERSION+x}" ]; then
  echo "Variable BASE_IMAGE_VERSION undefined"
  exit 1
elif [ -z "$BASE_IMAGE_VERSION" ]; then
  echo "Variable BASE_IMAGE_VERSION defined but empty"
  exit 1
fi

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

docker build . -f docker/train.Dockerfile -t ${TRAIN_IMAGE_NAME}:${TRAIN_IMAGE_VERSION} \
    --build-arg BASE_IMAGE_NAME=${BASE_IMAGE_NAME} \
    --build-arg BASE_IMAGE_VERSION=${BASE_IMAGE_VERSION}
