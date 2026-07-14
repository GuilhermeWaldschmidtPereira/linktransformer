nohup bash -c '
set -e

for size in 10k 25k 100k 500k; do
  DATA_DIR="/home/guilherme_pereira/projeto_mestrado/linktransformer/experimento_ruido/${size}"

  echo ">>> Gerando embedding geral para ${size}"

  HOST_DATA_DIR="${DATA_DIR}" ./run_linktransformer/run_embeddings_partitioned_podman.sh \
    --mode query \
    --right-on uf municipio logradouro numero complemento localidade setor_censitario \
    --merge-partitions

  echo ">>> Finalizado ${size}"
done
' > /home/guilherme_pereira/projeto_mestrado/linktransformer/logs_nohup/embeddings_query_ruido_all_podman.log 2>&1 &



nohup bash -lc '
for SIZE in 10k 25k 100k 500k; do
echo "=== Iniciando ${SIZE} em $(date) ==="
LINKTRANSFORMER_DATA_DIR="experimento_ruido/${SIZE}/exec_match" \
LINKTRANSFORMER_RESULTS_DIR="resultados_ruido/${SIZE}" \
./main.sh --scope geral --model sentence-transformers/all-mpnet-base-v2 \

> "logs_nohup/main_${SIZE}_todos_metodos.log" 2>&1
echo "=== Finalizando ${SIZE} em $(date) ==="
done
' > logs_nohup/main_todos_tamanhos_runner.log 2>&1 &


mkdir -p logs_nohup && nohup bash -lc 'for SIZE in 10k 25k 100k 500k; do echo "=== Iniciando ${SIZE} em $(date) ==="; LINKTRANSFORMER_DATA_DIR="experimento_ruido/${SIZE}/exec_match" LINKTRANSFORMER_RESULTS_DIR="resultados_ruido/${SIZE}" ./main.sh --scope geral --model sentence-transformers/all-mpnet-base-v2 > "logs_nohup/main_${SIZE}_todos_metodos.log" 2>&1; STATUS=$?; echo "=== Finalizando ${SIZE} em $(date), status ${STATUS} ==="; done' > logs_nohup/main_todos_tamanhos_runner.log 2>&1 &
