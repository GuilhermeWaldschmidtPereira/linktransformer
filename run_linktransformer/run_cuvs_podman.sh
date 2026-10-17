#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/.." && pwd)"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-}"
IMAGE_NAME="${IMAGE_NAME:-}"
HOST_DATA_DIR="${HOST_DATA_DIR:-${REPO_ROOT}/data}"
HOST_RESULTS_DIR="${HOST_RESULTS_DIR:-${REPO_ROOT}/resultados_cuvs}"

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
    IMAGE_NAME="localhost/linktransformer-cuvs"
  else
    IMAGE_NAME="linktransformer-cuvs"
  fi
fi

GPU_ARGS=()
if [[ "${CONTAINER_RUNTIME}" == "podman" ]]; then
  GPU_ARGS=(--device nvidia.com/gpu=all)
else
  GPU_ARGS=(--gpus all)
fi

mkdir -p "${HOST_RESULTS_DIR}"
VOLUME_ARGS=(-v "${HOST_DATA_DIR}:/data" -v "${HOST_RESULTS_DIR}:/results")
if [[ "${CONTAINER_RUNTIME}" == "podman" ]]; then
  VOLUME_ARGS=(-v "${HOST_DATA_DIR}:/data:Z" -v "${HOST_RESULTS_DIR}:/results:Z")
fi

ENV_ARGS=()
for variable in CUVS_BUILD_ALGO CUVS_GRAPH_DEGREE CUVS_INTERMEDIATE_GRAPH_DEGREE CUVS_ITOPK_SIZE NUM_EXECUCOES_BUSCA; do
  if [[ -n "${!variable:-}" ]]; then
    ENV_ARGS+=(-e "${variable}=${!variable}")
  fi
done

"${CONTAINER_RUNTIME}" build \
  -f "${REPO_ROOT}/Containerfile.cuvs" \
  -t "${IMAGE_NAME}" \
  "${REPO_ROOT}"

"${CONTAINER_RUNTIME}" run --rm \
  "${GPU_ARGS[@]}" \
  "${VOLUME_ARGS[@]}" \
  -e LINKTRANSFORMER_DATA_DIR=/data \
  -e LINKTRANSFORMER_RESULTS_DIR=/results \
  "${ENV_ARGS[@]}" \
  "${IMAGE_NAME}" \
  "$@"