#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/main_common.sh"

lt_require_podman
lt_prepare_results_file

echo ">>> Rodando ScaNN via Podman..."
lt_run_scann_container \
  python /workspace/run_linktransformer/main_scann.py

echo ">>> FIM do experimento ScaNN."
