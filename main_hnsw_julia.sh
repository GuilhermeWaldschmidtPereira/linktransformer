#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/main_common.sh"

lt_require_podman
lt_prepare_results_file

echo ">>> Rodando HNSW Julia via Podman..."
lt_run_linktransformer_container \
  python-jl /workspace/run_linktransformer/main_hnsw_julia.py

echo ">>> FIM do experimento HNSW Julia."
