#!/bin/bash

# Create results file
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Criar o resultados.csv no mesmo diretório do script
sudo rm -f "$SCRIPT_DIR/resultados.csv"
sudo touch "$SCRIPT_DIR/resultados.csv"

sudo chown gpereira:users "$SCRIPT_DIR/resultados.csv"

echo "metodo,index_time,search_time,total_time,num_rows_df1,num_rows_df2,k" > "$SCRIPT_DIR/resultados.csv"


# Execute Python script
python-jl my_npy_demo/run_with_npy.py

# Execute Julia script
python-jl hnsw_julia/run_linktransformer_hnsw_julia.py

python-jl run_linktransformer/main.py

python-jl run_linktransformer/main_nmslib.py

# python-jl run_linktransformer/main_scann.py