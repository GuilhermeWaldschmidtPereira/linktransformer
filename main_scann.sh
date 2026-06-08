#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_SCANN="$HOME/venv_scann"

echo "Venv ScaNN: ${VENV_SCANN}"
echo ">>> Ativando venv_scann..."
source "${VENV_SCANN}/bin/activate"

echo ">>> Rodando ScaNN..."
python "${PROJECT_ROOT}/run_linktransformer/main_scann.py" "$@"

echo ">>> FIM do experimento ScaNN."
