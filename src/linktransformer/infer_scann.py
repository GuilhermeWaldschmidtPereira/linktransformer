### Inference and Linkage script
### We want to link dfs together using embeddings

import os
import time
from typing import Union, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from linktransformer.utils import *
import scann
import psutil


def merge_knn_scann(
    k: int,
    df1: DataFrame,
    df2: DataFrame,
    suffixes: Tuple[str, str]
) -> DataFrame:
    """
    Versão ScaNN no mesmo padrão da merge_knn_hnsw_julia:

    - Lê embeddings pré-computados de ../data/embeddings_base.npy e ../data/embeddings_query.npy
    - Constrói o índice ScaNN sobre a base (embeddings1)
    - Faz busca k-NN para embeddings2
    - Faz merge fuzzy df1 x df2
    - Salva tempos em resultados.csv com metodo = "scann_knn"
    """

    # ================================
    #     CARREGAR EMBEDDINGS
    # ================================
    embeddings1 = np.load("../data/embeddings_base.npy")   # df1
    embeddings2 = np.load("../data/embeddings_query.npy")  # df2

    # Normalizar (ScaNN com dot_product ~ cosseno)
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # ================================
    #     INDEXAÇÃO (ScaNN)
    # ================================
    start_index_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB

    n_points = embeddings1.shape[0]

    # parâmetros "base" pensados para bases grandes
    base_num_leaves = 2000
    base_num_leaves_to_search = 100

    # adapta ao tamanho real
    num_leaves = max(min(base_num_leaves, n_points), 1)
    num_leaves_to_search = min(base_num_leaves_to_search, num_leaves)

    builder = scann.scann_ops_pybind.builder(
        embeddings1,      # base indexada
        k,                # número de vizinhos
        "dot_product",    # métrica (com vetores normalizados)
    )

    usa_tree = n_points > 1
    usa_ah = n_points >= 64  # só ativa asymmetric hashing em bases razoavelmente grandes

    if usa_tree:
        builder = builder.tree(
            num_leaves=num_leaves,
            num_leaves_to_search=num_leaves_to_search,
        )

    # EXATAMENTE 1 entre score_ah e score_brute_force
    if usa_ah:
        builder = builder.score_ah(
            2,
            anisotropic_quantization_threshold=0.2,
        ).reorder(100)
    else:
        # brute force + reorder (reorder aqui é redundante, mas mantém a assinatura)
        builder = builder.score_brute_force().reorder(100)

    searcher = builder.build()

    mem_after = process.memory_info().rss / (1024 ** 2)  # MB
    mem_used_create_index = mem_after - mem_before
    index_time = time.time() - start_index_time

    print(
        f"[ScaNN] índice criado em {index_time:.6f} s | "
        f"n_points={n_points}, num_leaves={num_leaves}, "
        f"num_leaves_to_search={num_leaves_to_search}, "
        f"usa_tree={usa_tree}, usa_AH={usa_ah}"
    )
    print(f"Memória utilizada na indexação (ScaNN): {mem_used_create_index:.2f} MB")

    # ================================
    #     BUSCA KNN
    # ================================
    num_execucoes = 100
    soma_tempo_busca = 0.0
    soma_memoria_usada = 0.0
    I = None
    D = None

    for i in range(num_execucoes):
        start_search_time = time.time()
        mem_before = process.memory_info().rss / (1024 ** 2)  # MB

        # search_batched retorna (neighbors, scores)
        I, D = searcher.search_batched(
            embeddings2,
            final_num_neighbors=k,
        )

        mem_after = process.memory_info().rss / (1024 ** 2)  # MB
        mem_used_search = mem_after - mem_before
        soma_memoria_usada += mem_used_search

        search_time = time.time() - start_search_time
        soma_tempo_busca += search_time

    avg_search_time = soma_tempo_busca / num_execucoes
    avg_mem_used_search = soma_memoria_usada / num_execucoes

    print(
        f"[ScaNN] tempo médio de busca em {num_execucoes} execuções: "
        f"{avg_search_time:.6f} s"
    )
    print(
        f"Memória média utilizada na busca (ScaNN): "
        f"{avg_mem_used_search:.4f} MB"
    )

    # ================================
    #     MERGE FUZZY
    # ================================
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    df1_expanded = df1.loc[np.repeat(df1.index.values, k)].reset_index(drop=True)
    df2_expanded = df2.iloc[I.flatten()].reset_index(drop=True)

    df_lm_matched = df1_expanded.merge(
        df2_expanded,
        left_index=True,
        right_index=True,
        how="inner",
        suffixes=suffixes,
    )

    # ScaNN retorna "scores" (dot-product similarity)
    df_lm_matched["score"] = D.flatten()

    print(
        f"LM matched (ScaNN) - left: {None}{suffixes[0]}, "
        f"right: {None}{suffixes[1]}"
    )

    # ================================
    #   SALVAR RESULTADOS DE TEMPO
    # ================================
    results_file = "resultados.csv"
    total_time = index_time + avg_search_time

    results_data = {
        "metodo": ["scann_knn"],
        "index_time": [index_time],
        "search_time": [avg_search_time],
        "total_time": [total_time],
        "num_rows_df1": [len(df1)],
        "num_rows_df2": [len(df2)],
        "k": [k],
        "mem_used_indexation_MB": [mem_used_create_index],
        "avg_mem_used_search_MB": [avg_mem_used_search],
    }
    results_df = pd.DataFrame(results_data)

    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(results_file, mode="w", header=True, index=False)

    print(f"Resultados (ScaNN) salvos em {results_file}")

    return df_lm_matched
