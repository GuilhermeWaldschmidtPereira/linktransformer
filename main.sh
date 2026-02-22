#!/usr/bin/env bash
set -e  # se qualquer comando falhar, o script para

# Create results file
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Criar o resultados.csv no mesmo diretório do script
sudo rm -f "$SCRIPT_DIR/resultados.csv"
sudo touch "$SCRIPT_DIR/resultados.csv"

sudo chown gpereira:users "$SCRIPT_DIR/resultados.csv"

echo "metodo,index_time,search_time,total_time,num_rows_df1,num_rows_df2,k" > "$SCRIPT_DIR/resultados.csv"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_SCANN="$HOME/venv_scann"

echo "Venv ScaNN:  ${VENV_SCANN}"


########################################
# 1) venv geral - 4 algoritmos
########################################
(

    # Execute Python script
    python-jl "${PROJECT_ROOT}/run_linktransformer/main2.py" 

)

########################################
# 2) venv_scann - algoritmo ScaNN
########################################
(
  echo ">>> Ativando venv_scann..."
  source "${VENV_SCANN}/bin/activate"

  echo ">>> Rodando ScaNN..."
  python "${PROJECT_ROOT}/run_linktransformer/main_scann.py"
)

echo ">>> FIM de todos os experimentos."