#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-}"
REQ_HASH="$(sha256sum "${REPO_ROOT}/requirements-embeddings.txt" | awk '{print substr($1,1,16)}')"
BASE_IMAGE_NAME="${BASE_IMAGE_NAME:-}"

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

if [[ -z "${BASE_IMAGE_NAME}" ]]; then
  if [[ "${CONTAINER_RUNTIME}" == "podman" ]]; then
    BASE_IMAGE_NAME="localhost/linktransformer-embeddings-base:${REQ_HASH}"
  else
    BASE_IMAGE_NAME="linktransformer-embeddings-base:${REQ_HASH}"
  fi
fi

VOLUME_MOUNT="${REPO_ROOT}/data:/data"
if [[ "${CONTAINER_RUNTIME}" == "podman" ]]; then
  VOLUME_MOUNT="${VOLUME_MOUNT}:Z"
fi

if ! "${CONTAINER_RUNTIME}" image exists "${BASE_IMAGE_NAME}"; then
  "${CONTAINER_RUNTIME}" build \
    --layers \
    -f "${REPO_ROOT}/Containerfile.embeddings.base" \
    -t "${BASE_IMAGE_NAME}" \
    "${REPO_ROOT}"
else
  echo "Reutilizando imagem base de dependencias: ${BASE_IMAGE_NAME}"
fi

"${CONTAINER_RUNTIME}" build \
  --layers \
  --build-arg "BASE_IMAGE=${BASE_IMAGE_NAME}" \
  -f "${REPO_ROOT}/Containerfile.embeddings" \
  -t "${IMAGE_NAME}" \
  "${REPO_ROOT}"

"${CONTAINER_RUNTIME}" run --rm \
  -v "${VOLUME_MOUNT}" \
  "${IMAGE_NAME}" \
  --base-path /data/base.csv \
  --query-path /data/query.csv \
  --output-dir /data \
  "$@"
