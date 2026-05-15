#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-}"

if [[ -z "${CONTAINER_RUNTIME}" ]]; then
  if command -v podman >/dev/null 2>&1; then
    CONTAINER_RUNTIME="podman"
  elif command -v docker >/dev/null 2>&1; then
    CONTAINER_RUNTIME="docker"
  else
    echo "Nenhum runtime encontrado. Instale podman ou docker." >&2
    exit 1
  fi
fi

if [[ -z "${IMAGE_NAME}" ]]; then
  if [[ "${CONTAINER_RUNTIME}" == "podman" ]]; then
    IMAGE_NAME="localhost/linktransformer-embeddings"
  else
    IMAGE_NAME="linktransformer-embeddings"
  fi
fi

VOLUME_MOUNT="${REPO_ROOT}/data:/data"
if [[ "${CONTAINER_RUNTIME}" == "podman" ]]; then
  VOLUME_MOUNT="${VOLUME_MOUNT}:Z"
fi

"${CONTAINER_RUNTIME}" build -f "${REPO_ROOT}/Containerfile.embeddings" -t "${IMAGE_NAME}" "${REPO_ROOT}"

"${CONTAINER_RUNTIME}" run --rm \
  -v "${VOLUME_MOUNT}" \
  "${IMAGE_NAME}" \
  --base-path /data/base.csv \
  --query-path /data/query.csv \
  --output-dir /data \
  "$@"
