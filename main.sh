#!/usr/bin/env bash
set -e  # se qualquer comando falhar, o script para

# Create results file
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_LINKTRANSFORMER="${LINKTRANSFORMER_IMAGE:-localhost/projeto-mestrado-linktransformer:latest}"
IMAGE_SCANN="${SCANN_IMAGE:-localhost/projeto-mestrado-scann:latest}"

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

echo ">>> Rodando LinkTransformer sem ScaNN via Podman..."
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_LINKTRANSFORMER}" \
  python /workspace/run_linktransformer/main_linktransformer.py

echo ">>> Rodando NMSLIB via Podman..."
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_LINKTRANSFORMER}" \
  python /workspace/run_linktransformer/main_nmslib_runner.py

echo ">>> Rodando HNSW Julia via Podman..."
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_LINKTRANSFORMER}" \
  python-jl /workspace/run_linktransformer/main_hnsw_julia.py

echo ">>> Rodando ScaNN via Podman (imagem dedicada)..."
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_SCANN}" \
  python /workspace/run_linktransformer/main_scann.py


echo ">>> FIM de todos os experimentos."
