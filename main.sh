#!/usr/bin/env bash
set -e  # se qualquer comando falhar, o script para

# Create results file
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_LINKTRANSFORMER="${LINKTRANSFORMER_IMAGE:-localhost/projeto-mestrado-linktransformer:latest}"
IMAGE_SCANN="${SCANN_IMAGE:-localhost/projeto-mestrado-scann:latest}"
CONTAINER_DATA_DIR="${LINKTRANSFORMER_DATA_DIR:-/workspace/data}"
HOST_RESULTS_DIR_INPUT="${LINKTRANSFORMER_RESULTS_DIR:-}"

if [[ -n "${HOST_RESULTS_DIR_INPUT}" ]]; then
  case "${HOST_RESULTS_DIR_INPUT}" in
    /workspace/*)
      REL_RESULTS_DIR="${HOST_RESULTS_DIR_INPUT#/workspace/}"
      RESULTS_DIR="${PROJECT_ROOT}/${REL_RESULTS_DIR}"
      CONTAINER_RESULTS_DIR="${HOST_RESULTS_DIR_INPUT}"
      ;;
    /*)
      RESULTS_DIR="${HOST_RESULTS_DIR_INPUT}"
      case "${RESULTS_DIR}" in
        "${PROJECT_ROOT}"/*)
          REL_RESULTS_DIR="${RESULTS_DIR#"${PROJECT_ROOT}/"}"
          CONTAINER_RESULTS_DIR="/workspace/${REL_RESULTS_DIR}"
          ;;
        *)
          echo "Erro: LINKTRANSFORMER_RESULTS_DIR absoluto precisa estar dentro de ${PROJECT_ROOT}." >&2
          echo "Use um caminho relativo, /workspace/... ou um caminho absoluto dentro do projeto." >&2
          exit 1
          ;;
      esac
      ;;
    *)
      RESULTS_DIR="${PROJECT_ROOT}/${HOST_RESULTS_DIR_INPUT}"
      CONTAINER_RESULTS_DIR="/workspace/${HOST_RESULTS_DIR_INPUT}"
      ;;
  esac
else
  RESULTS_DIR_NAME="resultados_$(date +%d%m%Y%H%M%S)"
  RESULTS_DIR="$SCRIPT_DIR/$RESULTS_DIR_NAME"
  CONTAINER_RESULTS_DIR="/workspace/$RESULTS_DIR_NAME"
fi

if ! command -v podman >/dev/null 2>&1; then
  echo "Erro: podman não está instalado ou não está disponível no PATH." >&2
  exit 1
fi

# Criar uma pasta de resultados por execução para manter rastreabilidade.
mkdir -p "$RESULTS_DIR"
touch "$RESULTS_DIR/resultados.csv"

echo ">>> Resultados desta execução serão salvos em: $RESULTS_DIR"
echo ">>> Embeddings serão lidos em: $CONTAINER_DATA_DIR"
echo "metodo,modelo_embedding,index_time,search_time,total_time,num_rows_df1,num_rows_df2,k,mem_used_indexation_MB,avg_mem_used_search_MB,matches" > "$RESULTS_DIR/resultados.csv"

########################################
# Podman - LinkTransformer sem ScaNN
########################################

echo ">>> Rodando LinkTransformer sem ScaNN via Podman..."
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -e LINKTRANSFORMER_RESULTS_DIR="$CONTAINER_RESULTS_DIR" \
  -e LINKTRANSFORMER_DATA_DIR="$CONTAINER_DATA_DIR" \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_LINKTRANSFORMER}" \
  python /workspace/run_linktransformer/main_linktransformer.py "$@"

echo ">>> Rodando NMSLIB via Podman..."
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -e LINKTRANSFORMER_RESULTS_DIR="$CONTAINER_RESULTS_DIR" \
  -e LINKTRANSFORMER_DATA_DIR="$CONTAINER_DATA_DIR" \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_LINKTRANSFORMER}" \
  python /workspace/run_linktransformer/main_nmslib_runner.py "$@"

echo ">>> Rodando HNSW Julia via Podman..."
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -e LINKTRANSFORMER_RESULTS_DIR="$CONTAINER_RESULTS_DIR" \
  -e LINKTRANSFORMER_DATA_DIR="$CONTAINER_DATA_DIR" \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_LINKTRANSFORMER}" \
  python-jl /workspace/run_linktransformer/main_hnsw_julia.py "$@"

echo ">>> Rodando ScaNN via Podman (imagem dedicada)..."
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -e LINKTRANSFORMER_RESULTS_DIR="$CONTAINER_RESULTS_DIR" \
  -e LINKTRANSFORMER_DATA_DIR="$CONTAINER_DATA_DIR" \
  -e SCANN_BUILDER_MODE \
  -e SCANN_NUM_EXECUCOES \
  -e SCANN_QUERY_BATCH_SIZE \
  -e SCANN_NUM_LEAVES \
  -e SCANN_NUM_LEAVES_TO_SEARCH \
  -e SCANN_TRAINING_SAMPLE_SIZE \
  -e SCANN_DIMENSIONS_PER_BLOCK \
  -e SCANN_AH_THRESHOLD \
  -e SCANN_REORDER_K \
  -e SCANN_LEAVES_TO_SEARCH \
  -e SCANN_PRE_REORDER_NUM_NEIGHBORS \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_SCANN}" \
  python /workspace/run_linktransformer/main_scann.py "$@"


echo ">>> FIM de todos os experimentos."
