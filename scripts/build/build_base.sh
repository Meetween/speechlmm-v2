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

docker build . -f docker/base.Dockerfile  -t $BASE_IMAGE_NAME:$BASE_IMAGE_VERSION
