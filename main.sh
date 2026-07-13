#!/usr/bin/env bash
set -uo pipefail

# Create results file
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_LINKTRANSFORMER="${LINKTRANSFORMER_IMAGE:-localhost/projeto-mestrado-linktransformer:latest}"
IMAGE_SCANN="${SCANN_IMAGE:-localhost/projeto-mestrado-scann:latest}"
CONTAINER_DATA_DIR="${LINKTRANSFORMER_DATA_DIR:-/workspace/data}"
HOST_RESULTS_DIR_INPUT="${LINKTRANSFORMER_RESULTS_DIR:-}"
RESULTS_LAYOUT="${LINKTRANSFORMER_RESULTS_LAYOUT:-nested}"
GLOBAL_CHUNKED="${LINKTRANSFORMER_GLOBAL_CHUNKED:-1}"
GLOBAL_BASE_CHUNK_SIZE="${LINKTRANSFORMER_GLOBAL_BASE_CHUNK_SIZE:-50000}"
GLOBAL_QUERY_BATCH_SIZE="${LINKTRANSFORMER_GLOBAL_QUERY_BATCH_SIZE:-10000}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_BASE_DIR=""
CONTAINER_RESULTS_BASE_DIR=""
CREATE_LATEST_POINTER=0

if [[ -n "${HOST_RESULTS_DIR_INPUT}" ]]; then
  case "${HOST_RESULTS_DIR_INPUT}" in
    /workspace/*)
      REL_RESULTS_DIR="${HOST_RESULTS_DIR_INPUT#/workspace/}"
      RESULTS_BASE_DIR="${PROJECT_ROOT}/${REL_RESULTS_DIR}"
      CONTAINER_RESULTS_BASE_DIR="${HOST_RESULTS_DIR_INPUT}"
      ;;
    /*)
      RESULTS_BASE_DIR="${HOST_RESULTS_DIR_INPUT}"
      case "${RESULTS_BASE_DIR}" in
        "${PROJECT_ROOT}"/*)
          REL_RESULTS_DIR="${RESULTS_BASE_DIR#"${PROJECT_ROOT}/"}"
          CONTAINER_RESULTS_BASE_DIR="/workspace/${REL_RESULTS_DIR}"
          ;;
        *)
          echo "Erro: LINKTRANSFORMER_RESULTS_DIR absoluto precisa estar dentro de ${PROJECT_ROOT}." >&2
          echo "Use um caminho relativo, /workspace/... ou um caminho absoluto dentro do projeto." >&2
          exit 1
          ;;
      esac
      ;;
    *)
      RESULTS_BASE_DIR="${PROJECT_ROOT}/${HOST_RESULTS_DIR_INPUT}"
      CONTAINER_RESULTS_BASE_DIR="/workspace/${HOST_RESULTS_DIR_INPUT}"
      ;;
  esac

  if [[ "${RESULTS_LAYOUT}" == "flat" ]]; then
    RESULTS_DIR="${RESULTS_BASE_DIR}"
    CONTAINER_RESULTS_DIR="${CONTAINER_RESULTS_BASE_DIR}"
  else
    RESULTS_DIR="${RESULTS_BASE_DIR}/run_${RUN_TIMESTAMP}"
    CONTAINER_RESULTS_DIR="${CONTAINER_RESULTS_BASE_DIR}/run_${RUN_TIMESTAMP}"
    CREATE_LATEST_POINTER=1
  fi
else
  RESULTS_DIR_NAME="resultados_$(date +%d%m%Y%H%M%S)"
  RESULTS_DIR="$SCRIPT_DIR/$RESULTS_DIR_NAME"
  CONTAINER_RESULTS_DIR="/workspace/$RESULTS_DIR_NAME"
fi

if ! command -v podman >/dev/null 2>&1; then
  echo "Erro: podman não está instalado ou não está disponível no PATH." >&2
  exit 1
fi

COMMON_PODMAN_ARGS=(
  --rm
  --userns=keep-id
  --user "$(id -u):$(id -g)"
  -e LINKTRANSFORMER_RESULTS_DIR="$CONTAINER_RESULTS_DIR"
  -e LINKTRANSFORMER_DATA_DIR="$CONTAINER_DATA_DIR"
  -e LINKTRANSFORMER_GLOBAL_CHUNKED="$GLOBAL_CHUNKED"
  -e LINKTRANSFORMER_GLOBAL_BASE_CHUNK_SIZE="$GLOBAL_BASE_CHUNK_SIZE"
  -e LINKTRANSFORMER_GLOBAL_QUERY_BATCH_SIZE="$GLOBAL_QUERY_BATCH_SIZE"
  -e LINKTRANSFORMER_BASE_CSV
  -e LINKTRANSFORMER_QUERY_CSV
  -v "${PROJECT_ROOT}:/workspace:Z"
  -w /workspace
)

SCANN_EXTRA_ENV_ARGS=(
  -e SCANN_BUILDER_MODE
  -e SCANN_NUM_EXECUCOES
  -e SCANN_QUERY_BATCH_SIZE
  -e SCANN_NUM_LEAVES
  -e SCANN_NUM_LEAVES_TO_SEARCH
  -e SCANN_TRAINING_SAMPLE_SIZE
  -e SCANN_DIMENSIONS_PER_BLOCK
  -e SCANN_AH_THRESHOLD
  -e SCANN_REORDER_K
  -e SCANN_LEAVES_TO_SEARCH
  -e SCANN_PRE_REORDER_NUM_NEIGHBORS
)

FAILED_METHODS=()
METHOD_STATUS_PATH=""

record_method_status() {
  local label="$1"
  local status_name="$2"
  local status_code="$3"
  local started_at="$4"
  local finished_at="$5"
  local elapsed_seconds="$6"

  printf '%s,%s,%s,%s,%s,%s\n' \
    "$label" \
    "$status_name" \
    "$status_code" \
    "$started_at" \
    "$finished_at" \
    "$elapsed_seconds" \
    >> "$METHOD_STATUS_PATH"
}

run_method() {
  local label="$1"
  shift
  local started_at
  local finished_at
  local start_epoch
  local end_epoch
  local elapsed_seconds

  echo ">>> Rodando ${label} via Podman..."
  started_at="$(date --iso-8601=seconds)"
  start_epoch="$(date +%s)"
  if "$@"; then
    finished_at="$(date --iso-8601=seconds)"
    end_epoch="$(date +%s)"
    elapsed_seconds="$((end_epoch - start_epoch))"
    record_method_status "$label" "success" "0" "$started_at" "$finished_at" "$elapsed_seconds"
    echo ">>> ${label} finalizado com sucesso."
  else
    local status=$?
    finished_at="$(date --iso-8601=seconds)"
    end_epoch="$(date +%s)"
    elapsed_seconds="$((end_epoch - start_epoch))"
    record_method_status "$label" "failure" "$status" "$started_at" "$finished_at" "$elapsed_seconds"
    echo ">>> ${label} falhou com status ${status}. Continuando para o próximo método..." >&2
    FAILED_METHODS+=("${label}:${status}")
  fi
}

# Criar uma pasta de resultados por execução para manter rastreabilidade.
if [[ -n "${RESULTS_BASE_DIR}" ]]; then
  mkdir -p "$RESULTS_BASE_DIR"
fi
mkdir -p "$RESULTS_DIR"
touch "$RESULTS_DIR/resultados.csv"

if [[ "${CREATE_LATEST_POINTER}" == "1" ]]; then
  ln -sfn "$(basename "$RESULTS_DIR")" "$RESULTS_BASE_DIR/latest"
  printf '%s\n' "$RESULTS_DIR" > "$RESULTS_BASE_DIR/LATEST_RUN.txt"
fi

METHOD_STATUS_PATH="$RESULTS_DIR/method_status.csv"
printf 'method,status,exit_code,started_at,finished_at,elapsed_seconds\n' > "$METHOD_STATUS_PATH"
cat > "$RESULTS_DIR/run_info.txt" <<EOF
run_timestamp=${RUN_TIMESTAMP}
results_dir=${RESULTS_DIR}
container_results_dir=${CONTAINER_RESULTS_DIR}
results_layout=${RESULTS_LAYOUT}
data_dir=${CONTAINER_DATA_DIR}
global_chunked=${GLOBAL_CHUNKED}
global_base_chunk_size=${GLOBAL_BASE_CHUNK_SIZE}
global_query_batch_size=${GLOBAL_QUERY_BATCH_SIZE}
base_csv=${LINKTRANSFORMER_BASE_CSV:-}
query_csv=${LINKTRANSFORMER_QUERY_CSV:-}
image_linktransformer=${IMAGE_LINKTRANSFORMER}
image_scann=${IMAGE_SCANN}
args=$*
EOF

echo ">>> Resultados desta execução serão salvos em: $RESULTS_DIR"
echo ">>> Embeddings serão lidos em: $CONTAINER_DATA_DIR"
echo ">>> Chunking global: enabled=${GLOBAL_CHUNKED} | base_chunk_size=${GLOBAL_BASE_CHUNK_SIZE} | query_batch_size=${GLOBAL_QUERY_BATCH_SIZE}"
if [[ "${CREATE_LATEST_POINTER}" == "1" ]]; then
  echo ">>> Atalho para a última execução: ${RESULTS_BASE_DIR}/latest"
fi
echo "metodo,modelo_embedding,index_time,search_time,total_time,num_rows_df1,num_rows_df2,k,mem_used_indexation_MB,avg_mem_used_search_MB,matches" > "$RESULTS_DIR/resultados.csv"

########################################
# Podman - LinkTransformer sem ScaNN
########################################

run_method "LinkTransformer sem ScaNN" \
  podman run "${COMMON_PODMAN_ARGS[@]}" \
  "${IMAGE_LINKTRANSFORMER}" \
  python /workspace/run_linktransformer/main_linktransformer.py "$@"

run_method "NMSLIB" \
  podman run "${COMMON_PODMAN_ARGS[@]}" \
  "${IMAGE_LINKTRANSFORMER}" \
  python /workspace/run_linktransformer/main_nmslib_runner.py "$@"

run_method "HNSW Julia" \
  podman run "${COMMON_PODMAN_ARGS[@]}" \
  "${IMAGE_LINKTRANSFORMER}" \
  python-jl /workspace/run_linktransformer/main_hnsw_julia.py "$@"

run_method "ScaNN" \
  podman run "${COMMON_PODMAN_ARGS[@]}" "${SCANN_EXTRA_ENV_ARGS[@]}" \
  "${IMAGE_SCANN}" \
  python /workspace/run_linktransformer/main_scann.py "$@"

if [[ ${#FAILED_METHODS[@]} -gt 0 ]]; then
  echo ">>> Resumo de falhas nesta execução:"
  for failure in "${FAILED_METHODS[@]}"; do
    echo "    - ${failure%%:*} (status ${failure##*:})"
  done
  printf 'overall_status=failure\n' >> "$RESULTS_DIR/run_info.txt"
  exit 1
fi

printf 'overall_status=success\n' >> "$RESULTS_DIR/run_info.txt"
echo ">>> FIM de todos os experimentos."
