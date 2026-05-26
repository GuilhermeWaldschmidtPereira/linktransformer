#!/usr/bin/env bash
set -euo pipefail

LT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LT_RESULTS_FILE="${LT_ROOT_DIR}/resultados.csv"
LT_RESULTS_HEADER="metodo,modelo_embedding,index_time,search_time,total_time,num_rows_df1,num_rows_df2,k,mem_used_indexation_MB,avg_mem_used_search_MB,matches"
LT_IMAGE_LINKTRANSFORMER="${LINKTRANSFORMER_IMAGE:-localhost/projeto-mestrado-linktransformer:latest}"
LT_IMAGE_SCANN="${SCANN_IMAGE:-localhost/projeto-mestrado-scann:latest}"


lt_require_podman() {
  if ! command -v podman >/dev/null 2>&1; then
    echo "Erro: podman não está instalado ou não está disponível no PATH." >&2
    exit 1
  fi
}


lt_prepare_results_file() {
  if [ "${LINKTRANSFORMER_SKIP_RESULTS_INIT:-0}" = "1" ]; then
    if [ ! -f "${LT_RESULTS_FILE}" ]; then
      echo "${LT_RESULTS_HEADER}" > "${LT_RESULTS_FILE}"
    fi
    return
  fi

  rm -f "${LT_RESULTS_FILE}"
  echo "${LT_RESULTS_HEADER}" > "${LT_RESULTS_FILE}"
}


lt_collect_optional_env_args() {
  local -n env_args_ref=$1
  shift

  local env_name
  for env_name in "$@"; do
    if [ -n "${!env_name:-}" ]; then
      env_args_ref+=(-e "${env_name}=${!env_name}")
    fi
  done
}


lt_run_linktransformer_container() {
  local env_args=()

  lt_collect_optional_env_args \
    env_args \
    LINKTRANSFORMER_METHODS \
    LINKTRANSFORMER_BASE_CSV \
    LINKTRANSFORMER_QUERY_CSV

  podman run --rm \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    "${env_args[@]}" \
    -v "${LT_ROOT_DIR}:/workspace:Z" \
    -w /workspace \
    "${LT_IMAGE_LINKTRANSFORMER}" \
    "$@"
}


lt_run_scann_container() {
  local env_args=()

  lt_collect_optional_env_args \
    env_args \
    LINKTRANSFORMER_BASE_CSV \
    LINKTRANSFORMER_QUERY_CSV \
    SCANN_BUILDER_MODE \
    SCANN_NUM_EXECUCOES \
    SCANN_QUERY_BATCH_SIZE \
    SCANN_NUM_LEAVES \
    SCANN_NUM_LEAVES_TO_SEARCH \
    SCANN_TRAINING_SAMPLE_SIZE \
    SCANN_DIMENSIONS_PER_BLOCK \
    SCANN_AH_THRESHOLD \
    SCANN_REORDER_K \
    SCANN_LEAVES_TO_SEARCH \
    SCANN_PRE_REORDER_NUM_NEIGHBORS

  podman run --rm \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    "${env_args[@]}" \
    -v "${LT_ROOT_DIR}:/workspace:Z" \
    -w /workspace \
    "${LT_IMAGE_SCANN}" \
    "$@"
}
