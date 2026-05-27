#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_LINKTRANSFORMER="${LINKTRANSFORMER_IMAGE:-localhost/projeto-mestrado-linktransformer:latest}"
IMAGE_SCANN="${SCANN_IMAGE:-localhost/projeto-mestrado-scann:latest}"

CONTAINERFILE_LINKTRANSFORMER="${PROJECT_ROOT}/Containerfile.linktransformer"
CONTAINERFILE_SCANN="${PROJECT_ROOT}/Containerfile.scann"

if ! command -v podman >/dev/null 2>&1; then
  echo "Erro: podman não está instalado ou não está disponível no PATH." >&2
  exit 1
fi

build_image() {
  local image_name="$1"
  local containerfile="$2"

  if [[ ! -f "$containerfile" ]]; then
    echo "Erro: Containerfile não encontrado: $containerfile" >&2
    exit 1
  fi

  echo ">>> Build da imagem ${image_name} usando $(basename "$containerfile")"
  podman build \
    --layers \
    -f "$containerfile" \
    -t "$image_name" \
    "$PROJECT_ROOT"
}

build_image "${IMAGE_LINKTRANSFORMER}" "${CONTAINERFILE_LINKTRANSFORMER}"
build_image "${IMAGE_SCANN}" "${CONTAINERFILE_SCANN}"

echo ">>> Build concluído."
