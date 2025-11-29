###Inference and Linkage script
###We want to link dfs together using embeddings
import json
import os
import warnings
import faiss
import numpy as np
import pandas as pd
import svs
from typing import Union, List, Optional, Tuple,Dict, Any
from pandas import DataFrame
from julia import Julia, Main

import psutil
import os
from linktransformer.cluster_fns import cluster
# from linktransformer.utils import serialize_columns, infer_embeddings, load_model, load_clf, cosine_similarity_corresponding_pairs, tokenize_data_for_inference, predict_rows_with_openai
from linktransformer.utils import *
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from transformers import TrainingArguments, Trainer
from linktransformer.main_svs import VamanaIndexer
import time
import nmslib

def merge_knn(k, df1,df2, suffixes) -> DataFrame:
    # ================================
    #     INDEXAÇÃO (FAISS)
    # ================================
    # Medir tempo de criação do índice + add

    embeddings1 = np.load("../data/embeddings_base.npy")
    embeddings2 = np.load("../data/embeddings_query.npy")

    start_index_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB
    
    index = faiss.IndexFlatIP(embeddings1.shape[1])
    print("Adding embeddings to index")
    index.add(embeddings1)
    
    mem_after = process.memory_info().rss / (1024 ** 2)  # MB
    mem_used_create_index = mem_after - mem_before
    
    index_time = time.time() - start_index_time
    print(f"Memória utilizada na indexação: {mem_used_create_index:.2f} MB")
    print(f"Tempo de indexação (FAISS): {index_time:.4f} segundos")

    # ================================
    #     BUSCA KNN (FAISS)
    # ================================
    num_execucoes = 100
    soma_tempo_busca = 0.0
    D = None
    I = None
    soma_qtde_mem = 0.0

    print("Searching index")
    for i in range(num_execucoes):
        start_search_time = time.time()
        mem_before = process.memory_info().rss / (1024 ** 2)  # MB
        D, I = index.search(embeddings2, k)
        mem_after = process.memory_info().rss / (1024 ** 2)  # MB
        mem_used_search = mem_after - mem_before
        search_time = time.time() - start_search_time
        soma_tempo_busca += search_time

    avg_search_time = soma_tempo_busca / num_execucoes
    avg_mem_used_search = mem_used_search / num_execucoes
    print(f"Tempo médio de busca (FAISS) em {num_execucoes} execuções: {avg_search_time:.4f} segundos")

    ## Check nearest neighbor of the first text in df1 as a test
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    ## Fuzzily merge the dfs based on the faiss index queries
    ### Each I sublst is a list of k nearest neighbors for each row in df1 in terms of indices of df2
    ### We need to expand the rows of df1 and df2 to match the number of rows in df1
    ### We also need to expand the scores to match the number of rows in df1

    ### First, expand the rows of df1
    df1_expanded = df1.loc[np.repeat(df1.index.values, k)].reset_index(drop=True)
    ### Now, expand the rows of df2
    df2_expanded = df2.iloc[I.flatten()].reset_index(drop=True)

    ### Now, merge the expanded dfs
    df_lm_matched = df1_expanded.merge(
        df2_expanded,
        left_index=True,
        right_index=True,
        how="inner",
        suffixes=suffixes,
    )

    ### Add score column
    df_lm_matched["score"] = D.flatten()

    if None is not None:
        df_lm_matched = df_lm_matched[df_lm_matched["score"] >= None]
        print(f"Dropped rows with similarity below {None}")

    print(
        f"LM matched on key columns - left: {None}{suffixes[0]}, "
        f"right: {None}{suffixes[1]}"
    )

    # ================================
    #   SALVAR RESULTADOS DE TEMPO
    # ================================
    results_file = "resultados.csv"
    total_time = index_time + avg_search_time

    results_data = {
        "metodo": ["faiss_knn"],
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

    print(f"Resultados (FAISS) salvos em {results_file}")

    return df_lm_matched

def merge_knn2(k, df1, df2, suffixes) -> DataFrame:
    """
    Versão SVS/Vamana no mesmo padrão da merge_knn (FAISS):

    - Lê embeddings pré-computados de ../data/embeddings_base.npy e ../data/embeddings_query.npy
    - Constrói índice SVS sobre a base (query) e busca k-NN para a outra
    - Faz o merge fuzzy df1 x df2
    - Salva tempos em resultados.csv com metodo = "svs_knn"
    """

    # ================================
    #     CARREGAR EMBEDDINGS
    # ================================
    embeddings1 = np.load("../data/embeddings_base.npy")   # mesmos arquivos da merge_knn
    embeddings2 = np.load("../data/embeddings_query.npy")

    # ================================
    #     INDEXAÇÃO (SVS / Vamana)
    # ================================
    class_svs = VamanaIndexer()

    # Aqui vamos seguir a mesma lógica da FAISS:
    #   - df2 é a base indexada (candidatos)
    #   - df1 gera as queries
    # Logo, indexamos embeddings2 (df2) e consultamos com embeddings1 (df1)
    start_index_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB
    
    index = class_svs.build(
        base_embeddings=embeddings1,        # base indexada (df2)
        reduced_dims=128,                   # projeção para 128D
        graph_max_degree=64,                # M (grau máximo do grafo)
        window_size=128,                    # janela para construção
        distance=svs.DistanceType.L2,       # métrica L2
        num_threads=4,                      # paralelismo
        primary_kind=svs.LeanVecKind.lvq4,
        secondary_kind=svs.LeanVecKind.lvq8,
    )
    
    mem_after = process.memory_info().rss / (1024 ** 2)  # MB
    mem_used_create_index = mem_after - mem_before
    print(f"Memória utilizada na indexação (SVS): {mem_used_create_index:.2f} MB")
    index_time = time.time() - start_index_time
    print(f"Tempo de indexação (SVS): {index_time:.4f} segundos")

    # ================================
    #     BUSCA KNN (SVS)
    # ================================
    num_execucoes = 100
    soma_tempo_busca = 0.0
    soma_qtde_mem = 0.0
    I = None
    D = None

    print("Searching SVS index")
    for i in range(num_execucoes):
        start_search_time = time.time()
        # consultas: embeddings1 (df1) procurando em embeddings2 (df2)
        mem_before = process.memory_info().rss / (1024 ** 2)  # MB
        I, D = index.search(embeddings2, k)
        mem_after = process.memory_info().rss / (1024 ** 2)  # MB
        mem_used_search = mem_after - mem_before
        soma_qtde_mem += mem_used_search
        search_time = time.time() - start_search_time
        soma_tempo_busca += search_time

    avg_search_time = soma_tempo_busca / num_execucoes
    avg_mem_used_search = soma_qtde_mem / num_execucoes

    # ================================
    #     MERGE FUZZY
    # ================================
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # expandir df1 e df2 como na merge_knn
    df1_expanded = df1.loc[np.repeat(df1.index.values, k)].reset_index(drop=True)
    df2_expanded = df2.iloc[I.flatten()].reset_index(drop=True)

    df_lm_matched = df1_expanded.merge(
        df2_expanded,
        left_index=True,
        right_index=True,
        how="inner",
        suffixes=suffixes,
    )

    # D aqui costuma ser distância (L2); se quiser pode transformar em similaridade
    # por agora seguimos o padrão e usamos D "cru" como score:
    df_lm_matched["score"] = D.flatten()

    # (mantendo o padrão da merge_knn: sem threshold explícito)
    # if None is not None:
    #     df_lm_matched = df_lm_matched[df_lm_matched["score"] >= None]

    print(
        f"LM matched (SVS) - left: {None}{suffixes[0]}, "
        f"right: {None}{suffixes[1]}"
    )

    # ================================
    #   SALVAR RESULTADOS DE TEMPO
    # ================================
    results_file = "resultados.csv"
    total_time = index_time + avg_search_time

    results_data = {
        "metodo": ["svs_knn"],
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

    print(f"Resultados (SVS) salvos em {results_file}")

    return df_lm_matched

def merge_knn_hnsw_julia(k, df1, df2, suffixes) -> DataFrame:
    """
    Versão HNSW (Julia) no mesmo padrão da merge_knn (FAISS):

    - Lê embeddings pré-computados de ../data/embeddings_base.npy e ../data/embeddings_query.npy
    - Constrói o índice HNSW em Julia sobre a base (df2 / embeddings_query)
    - Faz busca k-NN para df1
    - Faz merge fuzzy df1 x df2
    - Salva tempos em resultados.csv com metodo = "hnsw_julia"
    """
    # ================================
    #     CARREGAR EMBEDDINGS
    # ================================
    embeddings1 = np.load("../data/embeddings_base.npy")   # df1
    embeddings2 = np.load("../data/embeddings_query.npy")  # df2

    # Normalizar (Julia HNSW geralmente trabalha com L2/cosseno)
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # ================================
    #     INDEXAÇÃO (HNSW em Julia)
    # ================================
    # Importante: ajustar caminho conforme a estrutura do projeto
    Main.include("../hnsw_julia/hnsw_wrapper.jl")

    # Indexar a BASE = df2 / embeddings2, igual ao padrão FAISS/SVS/NMSLIB
    start_index_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB
    
    hnsw = Main.build_hnsw(embeddings1)
    
    mem_after = process.memory_info().rss / (1024 ** 2)  # MB
    mem_used_create_index = mem_after - mem_before
    print(f"Memória utilizada na indexação (HNSW Julia): {mem_used_create_index:.2f} MB")
    index_time = time.time() - start_index_time
    print(f"Tempo de criação do índice (HNSW Julia): {index_time:.4f} segundos")

    # ================================
    #     BUSCA KNN
    # ================================
    soma_tempo_busca = 0.0
    num_execucoes = 100
    I = None
    D = None
    soma_memoria_usada = 0.0

    for i in range(num_execucoes):
        start_search_time = time.time()
        # search_hnsw(hnsw, queries, k)  -> (I, D, tempo_busca)
        mem_before = process.memory_info().rss / (1024 ** 2)  # MB
        I, D, tempo_busca = Main.search_hnsw(hnsw, embeddings2)
        mem_after = process.memory_info().rss / (1024 ** 2)  # MB
        mem_used_search = mem_after - mem_before
        soma_memoria_usada += mem_used_search
        search_time = time.time() - start_search_time
        soma_tempo_busca += tempo_busca  # ou search_time, se preferir consistência
    # Julia costuma retornar índices 1-based
    I = I - 1

    avg_search_time = soma_tempo_busca / num_execucoes
    avg_mem_used_search = soma_memoria_usada / num_execucoes
    print(f"Tempo médio de busca (HNSW Julia) em {num_execucoes} execuções: {avg_search_time:.4f} segundos")

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

    df_lm_matched["score"] = D.flatten()

    print(
        f"LM matched (HNSW Julia) - left: {None}{suffixes[0]}, "
        f"right: {None}{suffixes[1]}"
    )

    # ================================
    #   SALVAR RESULTADOS DE TEMPO
    # ================================
    results_file = "resultados.csv"
    total_time = index_time + avg_search_time

    results_data = {
        "metodo": ["hnsw_julia"],
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

    print(f"Resultados (HNSW Julia) salvos em {results_file}")

    return df_lm_matched

def merge_knn_nmslib(k, df1, df2, suffixes) -> DataFrame:
    """
    Versão NMSLIB (HNSW) no mesmo padrão da merge_knn (FAISS):

    - Lê embeddings pré-computados de ../data/embeddings_base.npy e ../data/embeddings_query.npy
    - Constrói índice HNSW (nmslib) sobre df2 / embeddings_query
    - Busca k-NN para df1
    - Faz merge fuzzy df1 x df2
    - Salva tempos em resultados.csv com metodo = "nmslib_hnsw"
    """

    # ================================
    #     CARREGAR EMBEDDINGS
    # ================================
    embeddings1 = np.load("../data/embeddings_base.npy")   # df1
    embeddings2 = np.load("../data/embeddings_query.npy")  # df2

    # Normalizar para cosinesimil
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # ================================
    #     INDEXAÇÃO (NMSLIB / HNSW)
    # ================================
    start_index_time = time.time()

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB
    
    index = nmslib.init(
        space="cosinesimil",
        method="hnsw"
    )
    index.addDataPointBatch(embeddings1)  # base indexada = df2

    index.createIndex(
        {
            "M": 16,
            "efConstruction": 200,
        },
        print_progress=False,
    )
    
    mem_after = process.memory_info().rss / (1024 ** 2)  # MB
    mem_used_create_index = mem_after - mem_before
    print(f"Memória utilizada na indexação (NMSLIB): {mem_used_create_index:.2f} MB")

    index_time = time.time() - start_index_time
    print(f"Tempo de criação do índice (NMSLIB HNSW): {index_time:.4f} segundos")

    index.setQueryTimeParams({"efSearch": 50})

    # ================================
    #     BUSCA KNN (NMSLIB)
    # ================================
    num_execucoes = 100
    soma_tempo_busca = 0.0
    neighbors = None
    distances = None
    soma_qtde_mem = 0.0

    for i in range(num_execucoes):
        start_search_time = time.time()
        mem_before = process.memory_info().rss / (1024 ** 2)  # MB
        res = index.knnQueryBatch(embeddings2, k=k)  # queries = df1
        mem_after = process.memory_info().rss / (1024 ** 2)  # MB
        mem_used_search = mem_after - mem_before
        soma_qtde_mem += mem_used_search
        search_time = time.time() - start_search_time
        soma_tempo_busca += search_time
        neighbors, distances = zip(*res)

    avg_search_time = soma_tempo_busca / num_execucoes
    avg_mem_used_search = soma_qtde_mem / num_execucoes
    print(f"Tempo médio de busca (NMSLIB HNSW) em {num_execucoes} execuções: {avg_search_time:.4f} segundos")

    I = np.vstack(neighbors)       # indices em df2
    D_dist = np.vstack(distances)  # distâncias

    # Converter distâncias em similaridade (maior = melhor)
    score_sim = 1.0 - D_dist

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

    df_lm_matched["score"] = score_sim.flatten()

    print(
        f"LM matched (NMSLIB HNSW) - left: {None}{suffixes[0]}, "
        f"right: {None}{suffixes[1]}"
    )

    # ================================
    #   SALVAR RESULTADOS DE TEMPO
    # ================================
    results_file = "resultados.csv"
    total_time = index_time + avg_search_time

    results_data = {
        "metodo": ["nmslib_hnsw"],
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

    print(f"Resultados (NMSLIB HNSW) salvos em {results_file}")

    return df_lm_matched

def merge_knn_scann(k, df1, df2, suffixes) -> DataFrame:
    """
    Versão ScaNN no mesmo padrão da merge_knn (FAISS):

    - Lê embeddings pré-computados de ../data/embeddings_base.npy e ../data/embeddings_query.npy
    - Constrói índice ScaNN sobre df2 / embeddings_query
    - Faz busca k-NN para df1
    - Faz merge fuzzy df1 x df2
    - Salva tempos em resultados.csv com metodo = "scann_knn"
    """
    import scann

    # ================================
    #     CARREGAR EMBEDDINGS
    # ================================
    embeddings1 = np.load("../data/embeddings_base.npy")   # df1
    embeddings2 = np.load("../data/embeddings_query.npy")  # df2

    # Normalizar (ScaNN + dot_product ≈ cosine)
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # ================================
    #     INDEXAÇÃO (ScaNN)
    # ================================
    start_index_time = time.time()

    # Config simples e robusta: brute-force score
    searcher = (
        scann.scann_ops_pybind.builder(
            embeddings2,          # base indexada = df2
            k,
            "dot_product",
        )
        .score_brute_force()     # evita problema de clusters/treino com bases pequenas
        .build()
    )

    index_time = time.time() - start_index_time
    print(f"Tempo de criação do índice (ScaNN): {index_time:.4f} segundos")

    # ================================
    #     BUSCA KNN (ScaNN)
    # ================================
    num_execucoes = 100
    soma_tempo_busca = 0.0
    I = None
    D = None

    for i in range(num_execucoes):
        start_search_time = time.time()
        I, D = searcher.search_batched(
            embeddings1,               # queries = df1
            final_num_neighbors=k,
        )
        search_time = time.time() - start_search_time
        soma_tempo_busca += search_time

    avg_search_time = soma_tempo_busca / num_execucoes
    print(f"Tempo médio de busca (ScaNN) em {num_execucoes} execuções: {avg_search_time:.4f} segundos")

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

    df_lm_matched["score"] = D.flatten()  # dot-product similarity

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
    }
    results_df = pd.DataFrame(results_data)

    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(results_file, mode="w", header=True, index=False)

    print(f"Resultados (ScaNN) salvos em {results_file}")

    return df_lm_matched
