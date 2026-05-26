#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/main_common.sh"

lt_require_podman
lt_prepare_results_file

scripts=(
  main_baseline.sh
  main_svs.sh
  main_nmslib.sh
  main_hnsw_julia.sh
  main_scann.sh
)

for script_name in "${scripts[@]}"; do
  echo ">>> Executando ${script_name}..."
  LINKTRANSFORMER_SKIP_RESULTS_INIT=1 "${LT_ROOT_DIR}/${script_name}"
done

echo ">>> FIM de todos os experimentos."
