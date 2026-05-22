### Inference and Linkage script
### We want to link dfs together using embeddings

import os
import time
from typing import Union, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
import psutil

from linktransformer.utils import *

PATH_RESULTADOS = os.path.join(os.path.dirname(__file__), "resultados")

if not os.path.exists(PATH_RESULTADOS):
    os.makedirs(PATH_RESULTADOS)

NAME_DF_resultados_scann = "resultados_scann.csv"
PATH_resultados_scann = os.path.join(PATH_RESULTADOS, NAME_DF_resultados_scann)

df_geral = pd.DataFrame(columns=[
    "execucao",
    "tempo_busca",
    "memoria_usada_busca_MB",
    "modelo_index",
    "modelo_embedding",
])

df_geral.to_csv(PATH_resultados_scann, index=False)


def get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def get_scann_builder_mode() -> str:
    return os.environ.get("SCANN_BUILDER_MODE", "brute_force").strip().lower()


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if not embeddings.flags.c_contiguous:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    embeddings /= norms
    return embeddings


def resolve_query_batch_size(
    requested_batch_size: int,
    num_queries: int,
    num_base_vectors: int,
    builder_mode: str,
) -> int:
    if requested_batch_size > 0:
        return min(requested_batch_size, num_queries)

    if builder_mode != "brute_force":
        return 0

    score_budget_mb = get_env_int("SCANN_MAX_SCORE_MATRIX_MB", 256)
    bytes_per_score = np.dtype(np.float32).itemsize
    bytes_per_query = max(1, num_base_vectors * bytes_per_score)
    auto_batch_size = max(1, (score_budget_mb * 1024 ** 2) // bytes_per_query)
    return min(auto_batch_size, num_queries)


def build_scann_searcher(embeddings1: np.ndarray, k: int):
    import scann

    builder_mode = get_scann_builder_mode()
    print(f">>> [ScaNN] Modo do builder: {builder_mode}", flush=True)

    base_builder = scann.scann_ops_pybind.builder(
        embeddings1,
        k,
        "dot_product",
    )

    if builder_mode == "brute_force":
        print(">>> [ScaNN] Usando score_brute_force() (comportamento atual/padrão).", flush=True)
        return base_builder.score_brute_force()

    if builder_mode == "tree_ah":
        num_leaves = get_env_int("SCANN_NUM_LEAVES", 2000)
        num_leaves_to_search = get_env_int("SCANN_NUM_LEAVES_TO_SEARCH", 100)
        training_sample_size = get_env_int("SCANN_TRAINING_SAMPLE_SIZE", 250000)
        dimensions_per_block = get_env_int("SCANN_DIMENSIONS_PER_BLOCK", 2)
        aq_threshold = get_env_float("SCANN_AH_THRESHOLD", 0.2)
        reorder_k = get_env_int("SCANN_REORDER_K", 100)

        print(
            ">>> [ScaNN] Usando tree + score_ah + reorder "
            f"| num_leaves={num_leaves} "
            f"| num_leaves_to_search={num_leaves_to_search} "
            f"| training_sample_size={training_sample_size} "
            f"| dimensions_per_block={dimensions_per_block} "
            f"| anisotropic_quantization_threshold={aq_threshold} "
            f"| reorder_k={reorder_k}",
            flush=True,
        )

        return (
            base_builder
            .tree(
                num_leaves=num_leaves,
                num_leaves_to_search=num_leaves_to_search,
                training_sample_size=training_sample_size,
            )
            .score_ah(
                dimensions_per_block,
                anisotropic_quantization_threshold=aq_threshold,
            )
            .reorder(reorder_k)
        )

    raise ValueError(
        "SCANN_BUILDER_MODE inválido. Use 'brute_force' ou 'tree_ah'."
    )

def merge_knn_scann(
    k, 
    df1,
    df2, 
    suffixes, 
    model,
) -> DataFrame:
    # ================================
    safe_model = str(model).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")

    PATH_RESULTADOS_scann = f"resultados/scann/{safe_model}"
    embeddings_base_path = f"data/embeddings_base_{safe_model}.npy"
    embeddings_query_path = f"data/embeddings_query_{safe_model}.npy"

    process = psutil.Process(os.getpid())
    mem_process_before_load = process.memory_info().rss / (1024 ** 2)

    print(f">>> [ScaNN] Carregando embeddings do modelo: {model}", flush=True)
    print(f">>> [ScaNN] Arquivo embeddings base: {embeddings_base_path}", flush=True)
    print(f">>> [ScaNN] Arquivo embeddings query: {embeddings_query_path}", flush=True)
    print(
        f">>> [ScaNN] Memória do processo antes do load: {mem_process_before_load:.2f} MB",
        flush=True,
    )

    embeddings1 = np.load(embeddings_base_path)
    print(
        f">>> [ScaNN] Embeddings base carregados | shape={embeddings1.shape} | dtype={embeddings1.dtype}",
        flush=True,
    )

    embeddings2 = np.load(embeddings_query_path)
    print(
        f">>> [ScaNN] Embeddings query carregados | shape={embeddings2.shape} | dtype={embeddings2.dtype}",
        flush=True,
    )
    print(
        f">>> [ScaNN] Embeddings carregados | base={embeddings1.shape} | query={embeddings2.shape}",
        flush=True,
    )

    if not df1.index.equals(pd.RangeIndex(len(df1))):
        df1 = df1.reset_index(drop=True)
    if not df2.index.equals(pd.RangeIndex(len(df2))):
        df2 = df2.reset_index(drop=True)

    print(">>> [ScaNN] Normalizando embeddings...", flush=True)

    embeddings1 = normalize_embeddings(embeddings1)
    embeddings2 = normalize_embeddings(embeddings2)
    print(
        f">>> [ScaNN] Normalização concluída | base_dtype={embeddings1.dtype} | "
        f"query_dtype={embeddings2.dtype}",
        flush=True,
    )

    mem_before = process.memory_info().rss / (1024 ** 2)

    start_index_time = time.time()
    print(
        f">>> [ScaNN] Iniciando criação do builder | k={k} | métrica=dot_product",
        flush=True,
    )

    builder = build_scann_searcher(embeddings1, k)
    print(">>> [ScaNN] Builder criado. Iniciando build do searcher...", flush=True)

    searcher = builder.build()
    del builder
    print(">>> [ScaNN] Searcher construído com sucesso.", flush=True)

    index_time = time.time() - start_index_time

    mem_after = process.memory_info().rss / (1024 ** 2)
    mem_used_create_index = mem_after - mem_before

    print(
        f">>> [ScaNN] Indexação concluída: {index_time:.4f}s | Memória: {mem_used_create_index:.2f} MB",
        flush=True,
    )

    # ================================
    # 4) BUSCA KNN
    # ================================
    num_execucoes = get_env_int("SCANN_NUM_EXECUCOES", 5)
    builder_mode = get_scann_builder_mode()
    requested_query_batch_size = get_env_int("SCANN_QUERY_BATCH_SIZE", 0)
    query_batch_size = resolve_query_batch_size(
        requested_query_batch_size,
        num_queries=len(embeddings2),
        num_base_vectors=len(embeddings1),
        builder_mode=builder_mode,
    )
    leaves_to_search_override = os.environ.get("SCANN_LEAVES_TO_SEARCH")
    pre_reorder_override = os.environ.get("SCANN_PRE_REORDER_NUM_NEIGHBORS")

    search_kwargs = {"final_num_neighbors": k}
    if leaves_to_search_override not in (None, ""):
        search_kwargs["leaves_to_search"] = int(leaves_to_search_override)
    if pre_reorder_override not in (None, ""):
        search_kwargs["pre_reorder_num_neighbors"] = int(pre_reorder_override)

    full_score_matrix_gb = (
        len(embeddings1) * len(embeddings2) * np.dtype(np.float32).itemsize
    ) / (1024 ** 3)
    if requested_query_batch_size <= 0 and query_batch_size > 0:
        chunk_score_matrix_mb = (
            len(embeddings1) * query_batch_size * np.dtype(np.float32).itemsize
        ) / (1024 ** 2)
        print(
            f">>> [ScaNN] Busca total exigiria ~{full_score_matrix_gb:.2f} GB de scores em float32. "
            f"Aplicando chunk automático de {query_batch_size} queries "
            f"(~{chunk_score_matrix_mb:.2f} MB por bloco).",
            flush=True,
        )

    print(
        f">>> [ScaNN] Configuração de busca | execuções={num_execucoes} "
        f"| builder_mode={builder_mode} "
        f"| query_batch_size={query_batch_size if query_batch_size > 0 else 'all'} "
        f"| search_kwargs={search_kwargs}",
        flush=True,
    )
    soma_tempo_busca = 0.0
    soma_memoria = 0.0

    dict_ = {
        "execucao": [],
        "tempo_busca": [],
        "memoria_usada_busca_MB": [],
    }

    print(f">>> [ScaNN] Iniciando busca batelada | execuções={num_execucoes}", flush=True)
    for i in range(num_execucoes):
        print(f">>> [ScaNN] Execução de busca {i + 1}/{num_execucoes} iniciada...", flush=True)

        mem_before = process.memory_info().rss / (1024 ** 2)
        start_search_time = time.time()

        if query_batch_size > 0:
            print(
                f">>> [ScaNN] Processando queries em chunks de {query_batch_size}...",
                flush=True,
            )
            I = None
            D = None

            for start_idx in range(0, len(embeddings2), query_batch_size):
                end_idx = min(start_idx + query_batch_size, len(embeddings2))
                print(
                    f">>> [ScaNN] Chunk de queries: {start_idx}:{end_idx}",
                    flush=True,
                )
                chunk_I, chunk_D = searcher.search_batched(
                    embeddings2[start_idx:end_idx],
                    **search_kwargs,
                )
                if I is None:
                    I = np.empty((len(embeddings2), chunk_I.shape[1]), dtype=chunk_I.dtype)
                    D = np.empty((len(embeddings2), chunk_D.shape[1]), dtype=chunk_D.dtype)
                I[start_idx:end_idx] = chunk_I
                D[start_idx:end_idx] = chunk_D
        else:
            I, D = searcher.search_batched(
                embeddings2,
                **search_kwargs,
            )

        search_time = time.time() - start_search_time
        mem_after = process.memory_info().rss / (1024 ** 2)

        mem_used_search = mem_after - mem_before

        soma_tempo_busca += search_time
        soma_memoria += mem_used_search

        dict_["execucao"].append(i + 1)
        dict_["tempo_busca"].append(search_time)
        dict_["memoria_usada_busca_MB"].append(mem_used_search)
        print(
            f">>> [ScaNN] Execução de busca {i + 1}/{num_execucoes} concluída | "
            f"tempo={search_time:.4f}s | memória={mem_used_search:.2f} MB",
            flush=True,
        )

    avg_search_time = soma_tempo_busca / num_execucoes
    avg_mem_used_search = soma_memoria / num_execucoes

    print(f">>> [ScaNN] Busca média: {avg_search_time:.4f}s", flush=True)

    # ================================
    # 5) LOG DETALHADO GLOBAL
    # ================================
    print(f">>> [ScaNN] Salvando log detalhado global em {PATH_resultados_scann}", flush=True)
    df_tempos_busca_scann = pd.DataFrame(dict_)
    df_tempos_busca_scann["modelo_index"] = "scann"
    df_tempos_busca_scann["modelo_embedding"] = str(model)

    df_aux = pd.read_csv(PATH_resultados_scann)
    df_aux = pd.concat([df_aux, df_tempos_busca_scann], ignore_index=True)
    df_aux.to_csv(PATH_resultados_scann, index=False)

    # ================================
    # 6) MERGE RESULTADO
    # ================================
    df1_expanded = df2.loc[np.repeat(df2.index.values, k)].reset_index(drop=True)
    df2_expanded = df1.iloc[I.flatten()].reset_index(drop=True)

    df_lm_matched = df1_expanded.merge(
        df2_expanded,
        left_index=True,
        right_index=True,
        how="inner",
        suffixes=suffixes,
    )

    print(">>> [ScaNN] Merge concluído. Salvando matched_scann.csv...", flush=True)
    df_lm_matched.to_csv(os.path.join(PATH_RESULTADOS, f"matched_scann.csv"), index=False)

    # ================================
    # 7) SALVAR RESULTADOS INDIVIDUAIS
    # ================================
    if not os.path.exists(PATH_RESULTADOS_scann):
        os.makedirs(PATH_RESULTADOS_scann)

    print(
        f">>> [ScaNN] Salvando resultados individuais em {PATH_RESULTADOS_scann}",
        flush=True,
    )
    df_tempos_busca_scann.to_csv(
        os.path.join(PATH_RESULTADOS_scann, "csv_final_tempos_buscas.csv"),
        index=False
    )

    # ================================
    # 8) SALVAR MÉDIAS (results.csv)
    # ================================
    total_time = index_time + avg_search_time
    matches = (df_lm_matched["setor_censitario_x"] == df_lm_matched["setor_censitario_y"]).sum()

    results_data = {
        "metodo": ["scann"],
        "modelo_embedding": [str(model)],
        "index_time": [index_time],
        "search_time": [avg_search_time],
        "total_time": [total_time],
        "num_rows_df1": [len(df1)],
        "num_rows_df2": [len(df2)],
        "k": [k],
        "mem_used_indexation_MB": [mem_used_create_index],
        "avg_mem_used_search_MB": [avg_mem_used_search],
        "matches": [matches],
    }

    results_df = pd.DataFrame(results_data)

    results_file = "resultados.csv"
    print(f">>> [ScaNN] Atualizando resumo final em {results_file}", flush=True)

    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(results_file, mode="w", header=True, index=False)

    print(">>> [ScaNN] Processamento do modelo concluído com sucesso.", flush=True)
    return True
