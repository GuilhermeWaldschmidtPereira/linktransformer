#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/main_common.sh"

lt_require_podman
lt_prepare_results_file

echo ">>> Rodando SVS via Podman..."
(
  export LINKTRANSFORMER_METHODS=svs
  lt_run_linktransformer_container \
    python /workspace/run_linktransformer/main_linktransformer.py
)

echo ">>> FIM do experimento SVS."
