#!/usr/bin/env bash
set -e  # se qualquer comando falhar, o script para

# Create results file
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${LINKTRANSFORMER_IMAGE:-localhost/projeto-mestrado-linktransformer:latest}"

if ! command -v podman >/dev/null 2>&1; then
  echo "Erro: podman não está instalado ou não está disponível no PATH." >&2
  exit 1
fi

# Criar o resultados.csv no mesmo diretório do script
rm -f "$SCRIPT_DIR/resultados.csv"
touch "$SCRIPT_DIR/resultados.csv"

echo "metodo,modelo_embedding,index_time,search_time,total_time,num_rows_df1,num_rows_df2,k,mem_used_indexation_MB,avg_mem_used_search_MB,matches" > "$SCRIPT_DIR/resultados.csv"

########################################
# Podman - LinkTransformer sem ScaNN
########################################
if ! podman image exists "${IMAGE_NAME}"; then
  echo ">>> Imagem ${IMAGE_NAME} não encontrada. Construindo com Podman..."
  podman build \
    --layers \
    -f "${PROJECT_ROOT}/Containerfile.linktransformer" \
    -t "${IMAGE_NAME}" \
    "${PROJECT_ROOT}"
fi

echo ">>> Rodando LinkTransformer sem ScaNN via Podman..."
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_NAME}" \
  python /workspace/run_linktransformer/main_linktransformer.py

echo ">>> Rodando NMSLIB via Podman..."
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_NAME}" \
  python /workspace/run_linktransformer/main_nmslib_runner.py

echo ">>> Rodando HNSW Julia via Podman..."
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_NAME}" \
  python-jl /workspace/run_linktransformer/main_hnsw_julia.py

echo ">>> FIM dos experimentos LinkTransformer sem ScaNN."
