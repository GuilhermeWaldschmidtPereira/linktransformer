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
    safe_model = str(model).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")

    embeddings1 = np.load(f"data/embeddings_base_{safe_model}.npy")
    embeddings2 = np.load(f"data/embeddings_query_{safe_model}.npy")

    df1 = df1.copy().reset_index(drop=True)
    df2 = df2.copy().reset_index(drop=True)

    df1["id_lt"] = np.arange(len(df1))
    df2["id_lt"] = np.arange(len(df2))


    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # ================================
    # 3) INDEXAÇÃO (ScaNN)
    # ================================
    import scann

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)

    start_index_time = time.time()

    builder = scann.scann_ops_pybind.builder(
        embeddings1,
        k,
        "dot_product",
    ).score_brute_force()

    searcher = builder.build()

    index_time = time.time() - start_index_time

    mem_after = process.memory_info().rss / (1024 ** 2)
    mem_used_create_index = mem_after - mem_before

    print(f"[ScaNN] Indexação: {index_time:.4f}s | Memória: {mem_used_create_index:.2f} MB")

    # ================================
    # 4) BUSCA KNN (100 execuções)
    # ================================
    num_execucoes = 100
    soma_tempo_busca = 0.0
    soma_memoria = 0.0

    dict_ = {
        "execucao": [],
        "tempo_busca": [],
        "memoria_usada_busca_MB": [],
    }

    for i in range(num_execucoes):

        mem_before = process.memory_info().rss / (1024 ** 2)
        start_search_time = time.time()

        I, D = searcher.search_batched(
            embeddings2,
            final_num_neighbors=k,
        )

        search_time = time.time() - start_search_time
        mem_after = process.memory_info().rss / (1024 ** 2)

        mem_used_search = mem_after - mem_before

        soma_tempo_busca += search_time
        soma_memoria += mem_used_search

        dict_["execucao"].append(i + 1)
        dict_["tempo_busca"].append(search_time)
        dict_["memoria_usada_busca_MB"].append(mem_used_search)

    avg_search_time = soma_tempo_busca / num_execucoes
    avg_mem_used_search = soma_memoria / num_execucoes

    print(f"[ScaNN] Busca média: {avg_search_time:.4f}s")

    # ================================
    # 5) LOG DETALHADO GLOBAL
    # ================================
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

    df_lm_matched.to_csv(os.path.join(PATH_RESULTADOS, f"matched_scann.csv"), index=False)

    # ================================
    # 7) SALVAR RESULTADOS INDIVIDUAIS
    # ================================
    safe_model = str(model).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")

    PATH_RESULTADOS_scann = f"resultados/scann/{safe_model}"

    if not os.path.exists(PATH_RESULTADOS_scann):
        os.makedirs(PATH_RESULTADOS_scann)

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

    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(results_file, mode="w", header=True, index=False)

    return True