###Inference and Linkage script
###We want to link dfs together using embeddings
import os
import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple
from pandas import DataFrame

from linktransformer.cluster_fns import cluster
from linktransformer.utils import *
from sklearn.metrics.pairwise import cosine_similarity
import time

def merge_knn_scann(
    df1: DataFrame,
    df2: DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    model: Union[str, LinkTransformer] = "all-MiniLM-L6-v2",
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    k: int = 1,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    batch_size: int = 128,
    openai_key: Optional[str] = None,
    drop_sim_threshold: float = None,
) -> DataFrame:
    """
    Merge two dataframes using language model embeddings and ScaNN as k-NN index.

    - df2 vira a base indexada (database).
    - df1 gera as queries.
    - ScaNN usa dot_product em embeddings normalizados.
    - Tempos de index/search são salvos em resultados.csv.
    """

    # -------------------------
    # 1) Preparar colunas e IDs
    # -------------------------
    if on is None:
        on = list(set(df1.columns).intersection(set(df2.columns)))

    if left_on is None:
        left_on = on
    if right_on is None:
        right_on = on

    on = None

    df1 = df1.copy()
    df2 = df2.copy()

    if "id_lt" in df1.columns:
        raise ValueError("Column id_lt already exists in df1, please rename it to proceed")
    if "id_lt" in df2.columns:
        raise ValueError("Column id_lt already exists in df2, please rename it to proceed")

    df1.loc[:, "id_lt"] = np.arange(len(df1))
    df2.loc[:, "id_lt"] = np.arange(len(df2))

    if isinstance(right_on, list):
        strings_right = serialize_columns(df2, right_on, model=model)
    if isinstance(left_on, list):
        strings_left = serialize_columns(df1, left_on, model=model)
    else:
        strings_left = df1[left_on].tolist()
        strings_right = df2[right_on].tolist()

    # -------------------------
    # 2) Carregar modelo e embeddings
    # -------------------------
    if isinstance(model, str):
        if openai_key is None:
            model = load_model(model)

    embeddings1 = infer_embeddings(
        strings_left, model,
        batch_size=batch_size,
        openai_key=openai_key,
        return_numpy=True
    )
    embeddings2 = infer_embeddings(
        strings_right, model,
        batch_size=batch_size,
        openai_key=openai_key,
        return_numpy=True
    )

    if len(embeddings1.shape) == 1:
        embeddings1 = np.expand_dims(embeddings1, axis=0)
    if len(embeddings2.shape) == 1:
        embeddings2 = np.expand_dims(embeddings2, axis=0)

    # Normaliza embeddings -> ScaNN com "dot_product" vira cosine-like
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # -------------------------
    # 3) Construir índice ScaNN
    # -------------------------
    import scann

    start_index_time = time.time()

    # Aqui você pode ajustar hiperparâmetros conforme o tamanho da base.
    # Para bases pequenas, isso é meio overkill, mas mantém a mesma forma.
    searcher = (
        scann.scann_ops_pybind.builder(
            embeddings2,          # base indexada (df2)
            k,                    # número de vizinhos
            "dot_product",        # métrica
        )
        .tree(
            num_leaves=2000,
            num_leaves_to_search=100,
        )
        .score_ah(
            2, anisotropic_quantization_threshold=0.2,
        )
        .reorder(100)
        .build()
    )

    index_time = time.time() - start_index_time
    print(f"Tempo de criação do índice (ScaNN): {index_time:.6f} segundos")

    # -------------------------
    # 4) Buscar k-NN com ScaNN
    # -------------------------
    num_execucoes = 3
    soma_tempo_busca = 0.0
    I = None
    D = None

    for i in range(num_execucoes):
        start_search_time = time.time()
        # search_batched retorna (neighbors, distances/scores)
        I, D = searcher.search_batched(
            embeddings1,
            final_num_neighbors=k,
        )
        search_time = time.time() - start_search_time
        soma_tempo_busca += search_time

    avg_search_time = soma_tempo_busca / num_execucoes
    print(f"Tempo médio de busca (ScaNN) em {num_execucoes} execuções: {avg_search_time:.6f} segundos")

    # D já são scores de similaridade (dot product) – maior = mais similar.
    # shape: (nq, k)
    # I: índices em df2

    # -------------------------
    # 5) Merge fuzzy com os DataFrames
    # -------------------------
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

    df_lm_matched["score"] = D.flatten()  # dot-product similarity

    if drop_sim_threshold is not None:
        df_lm_matched = df_lm_matched[df_lm_matched["score"] >= drop_sim_threshold]
        print(f"Dropped rows with similarity below {drop_sim_threshold}")

    print(
        f"LM matched on key columns - left: {left_on}{suffixes[0]}, "
        f"right: {right_on}{suffixes[1]}"
    )

    # -------------------------
    # 6) Salvar tempos em resultados.csv
    # -------------------------
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
    }
    results_df = pd.DataFrame(results_data)

    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(results_file, mode="w", header=True, index=False)

    print(f"Results (ScaNN) saved to {results_file}")

    return df_lm_matched
