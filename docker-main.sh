#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/docker-main.log"

mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

rm -f "$SCRIPT_DIR/resultados.csv"
touch "$SCRIPT_DIR/resultados.csv"

if command -v id >/dev/null 2>&1; then
    CURRENT_USER="$(id -un)"
    CURRENT_GROUP="$(id -gn)"
    chown "${CURRENT_USER}:${CURRENT_GROUP}" "$SCRIPT_DIR/resultados.csv" 2>/dev/null || true
fi

echo "metodo,modelo_embedding,index_time,search_time,total_time,num_rows_df1,num_rows_df2,k,mem_used_indexation_MB,avg_mem_used_search_MB,matches" > "$SCRIPT_DIR/resultados.csv"

if command -v python-jl >/dev/null 2>&1; then
    PYTHON_RUNNER="python-jl"
elif command -v python >/dev/null 2>&1; then
    PYTHON_RUNNER="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_RUNNER="python3"
else
    echo "Nenhum interpretador Python encontrado no PATH."
    exit 1
fi

echo "Log file: ${LOG_FILE}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Python runner: ${PYTHON_RUNNER}"
"${PYTHON_RUNNER}" "${SCRIPT_DIR}/run_linktransformer/main2.py"
