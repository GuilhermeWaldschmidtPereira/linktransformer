#!/usr/bin/env bash
set -u

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

REPETICOES="${REPETICOES:-1}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
USER_ID="$(id -u)"
GROUP_ID="$(id -g)"
STATUS_LOG="$ROOT/logs_nohup/all_indexers_status.log"

mkdir -p "$ROOT/logs_nohup" "$ROOT/resultados_comparacao"

run_one() {
    local model="$1"
    local indexer="$2"
    local short="${model##*/}"
    local image
    local results="/workspace/resultados_comparacao/$indexer/$short"
    local log
    local -a gpu_args=()
    local -a runtime_env=()

    case "$indexer" in
        faiss)
            image="localhost/projeto-mestrado-faiss-gpu:latest"
            gpu_args=(--device nvidia.com/gpu=1)
            runtime_env=(-e FAISS_USE_GPU=1 -e FAISS_GPU_DEVICE=0)
            ;;
        jasper)
            image="localhost/projeto-mestrado-jasper:latest"
            gpu_args=(--device nvidia.com/gpu=1)
            ;;
        ggnn)
            image="localhost/projeto-mestrado-ggnn:latest"
            gpu_args=(--device nvidia.com/gpu=1)
            ;;
        cagra)
            image="localhost/projeto-mestrado-cagra:latest"
            gpu_args=(--device nvidia.com/gpu=1)
            ;;
        *)
            printf '%s\tERRO(indexador_desconhecido)\t%s\t%s\t%s\n' \
                "$(date '+%F %T')" "$indexer" "$model" "$REPETICOES" \
                >> "$STATUS_LOG"
            return 2
            ;;
    esac

    log="$ROOT/logs_nohup/${indexer}_${short}_${RUN_ID}_repeticoes-${REPETICOES}.log"
    mkdir -p "$ROOT/resultados_comparacao/$indexer/$short"

    printf '%s\tINICIO\t%s\t%s\t%s repeticoes\n' \
        "$(date '+%F %T')" "$indexer" "$model" "$REPETICOES" \
        >> "$STATUS_LOG"

    if podman run --rm \
        --userns=keep-id \
        --user "$USER_ID:$GROUP_ID" \
        "${gpu_args[@]}" \
        "${runtime_env[@]}" \
        -e LINKTRANSFORMER_DATA_DIR=/workspace/dados \
        -e "LINKTRANSFORMER_RESULTS_DIR=$results" \
        -v "$ROOT:/workspace:Z" \
        -w /workspace \
        "$image" \
        python /workspace/run_linktransformer/main_linktransformer.py \
            --scope geral --indexer "$indexer" --model "$model" --k 1 \
            --repetitions "$REPETICOES" \
        > "$log" 2>&1
    then
        printf '%s\tOK\t%s\t%s\t%s repeticoes\n' \
            "$(date '+%F %T')" "$indexer" "$model" "$REPETICOES" \
            >> "$STATUS_LOG"
    else
        local status=$?
        printf '%s\tERRO(%s)\t%s\t%s\t%s repeticoes\n' \
            "$(date '+%F %T')" "$status" "$indexer" "$model" "$REPETICOES" \
            >> "$STATUS_LOG"
    fi
}

models=(
    "sentence-transformers/all-MiniLM-L6-v2"
    "sentence-transformers/all-mpnet-base-v2"
)
indexers=(faiss jasper ggnn cagra)

for model in "${models[@]}"; do
    for indexer in "${indexers[@]}"; do
        run_one "$model" "$indexer"
    done
done
