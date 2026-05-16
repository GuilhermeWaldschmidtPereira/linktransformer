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

PATH_RESULTADOS = os.path.join(os.path.dirname(__file__), "resultados")

NUM_EXECUCOES_BUSCA = max(1, int(os.environ.get("LT_NUM_EXECUCOES_BUSCA", "5")))

if not os.path.exists(PATH_RESULTADOS):
    os.makedirs(PATH_RESULTADOS)

NAME_DF_RESULTADOS_GERAL = "resultados_geral.csv"
PATH_RESULTADOS_GERAL = os.path.join(PATH_RESULTADOS, NAME_DF_RESULTADOS_GERAL)

df_geral = pd.DataFrame(columns=[
    "execucao",
    "tempo_busca",
    "memoria_usada_busca_MB",
    "modelo_index",
    "modelo_embedding",
])

df_geral.to_csv(PATH_RESULTADOS_GERAL, index=False)

def merge_knn(k, df1,df2, suffixes, model) -> DataFrame:
    # ================================
    #     INDEXAÇÃO (FAISS)
    # ================================
    # Medir tempo de criação do índice + add
    safe_model = str(model).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")

    print(f">>> [FAISS] Carregando embeddings do modelo: {model}", flush=True)
    embeddings1 = np.load(f"data/embeddings_base_{safe_model}.npy")
    embeddings2 = np.load(f"data/embeddings_query_{safe_model}.npy")
    print(
        f">>> [FAISS] Embeddings carregados | base={embeddings1.shape} | query={embeddings2.shape}",
        flush=True,
    )

    start_index_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB
    
    print(">>> [FAISS] Criando índice IndexFlatIP...", flush=True)
    index = faiss.IndexFlatIP(embeddings1.shape[1])
    print(">>> [FAISS] Adicionando embeddings base ao índice...", flush=True)
    index.add(embeddings1)
    
    mem_after = process.memory_info().rss / (1024 ** 2)  # MB
    mem_used_create_index = mem_after - mem_before
    
    index_time = time.time() - start_index_time
    print(f"Memória utilizada na indexação: {mem_used_create_index:.2f} MB")
    print(f"Tempo de indexação (FAISS): {index_time:.4f} segundos")

    # ================================
    #     BUSCA KNN (FAISS)
    # ================================
    num_execucoes = NUM_EXECUCOES_BUSCA
    soma_tempo_busca = 0.0
    D = None
    I = None
    soma_qtde_mem = 0.0

    dict_ = {
        "execucao": [],
        "tempo_busca": [],
        "memoria_usada_busca_MB": [],
    }

    print(">>> [FAISS] Iniciando busca KNN...", flush=True)
    for i in range(num_execucoes):
        start_search_time = time.time()
        mem_before = process.memory_info().rss / (1024 ** 2)  # MB
        D, I = index.search(embeddings2, k)
        mem_after = process.memory_info().rss / (1024 ** 2)  # MB
        mem_used_search = mem_after - mem_before
        search_time = time.time() - start_search_time
        soma_tempo_busca += search_time

        dict_["execucao"].append(i+1)
        dict_["tempo_busca"].append(search_time)
        dict_["memoria_usada_busca_MB"].append(mem_used_search)


    avg_search_time = soma_tempo_busca / num_execucoes
    avg_mem_used_search = mem_used_search / num_execucoes
    print(f"Tempo médio de busca (FAISS) em {num_execucoes} execuções: {avg_search_time:.4f} segundos")

    df_tempos_busca_faiss_baseline = pd.DataFrame(dict_)
    df_tempos_busca_faiss_baseline['modelo_index'] = 'faiss_baseline'

    df_tempos_busca_faiss_baseline['modelo_embedding'] = model

    df_aux = pd.read_csv(PATH_RESULTADOS_GERAL)
    df_aux = pd.concat([df_aux, df_tempos_busca_faiss_baseline], ignore_index=True)
    df_aux.to_csv(PATH_RESULTADOS_GERAL, index=False)


    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # expandir df1 e df2 como na merge_knn
    df1_expanded = df2.loc[np.repeat(df2.index.values, k)].reset_index(drop=True)
    df2_expanded = df1.iloc[I.flatten()].reset_index(drop=True)    

    df_lm_matched = df1_expanded.merge(
        df2_expanded,
        left_index=True,
        right_index=True,
        how="inner",
        suffixes=suffixes,
    )

    print(
        f"LM matched on key columns - left: {None}{suffixes[0]}, "
        f"right: {None}{suffixes[1]}"
    )

    safe_model = str(model).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")

    PATH_RESULTADOS_baseline = f"resultados/baseline/{safe_model}"

    # ================================
    #   SALVAR RESULTADOS DE TEMPO
    # ================================
    if not os.path.exists(PATH_RESULTADOS_baseline):
        os.makedirs(PATH_RESULTADOS_baseline)

    RESULTADO_DE_TEMPO = f"csv_final_tempos_buscas.csv"
    df_tempos_busca_faiss_baseline.to_csv(os.path.join(PATH_RESULTADOS_baseline, RESULTADO_DE_TEMPO), index=False)

    # ================================
    #   SALVAR MÉDIAS DOS RESULTADOS
    # ================================
    results_file = "resultados.csv"
    total_time = index_time + avg_search_time
    df_lm_matched.to_csv('teste.csv')
    matches = (df_lm_matched["setor_esperado_x"] == df_lm_matched["setor_esperado_y"]).sum()
    results_data = {
        "metodo": ["baseline"],
        "modelo_embedding": [model],
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

    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(results_file, mode="w", header=True, index=False)

    return True

def merge_knn2(k, df1, df2, suffixes, model) -> DataFrame:
    """
    Versão SVS/Vamana no mesmo padrão da merge_knn (FAISS):

    - Lê embeddings pré-computados de data/embeddings_base.npy e data/embeddings_query.npy
    - Constrói índice SVS sobre a base (query) e busca k-NN para a outra
    - Faz o merge fuzzy df1 x df2
    - Salva tempos em resultados.csv com metodo = "svs_knn"
    """

    # ================================
    #     CARREGAR EMBEDDINGS
    # ================================
    safe_model = str(model).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")

    print(f">>> [SVS] Carregando embeddings do modelo: {model}", flush=True)
    embeddings1 = np.load(f"data/embeddings_base_{safe_model}.npy")
    embeddings2 = np.load(f"data/embeddings_query_{safe_model}.npy")
    print(
        f">>> [SVS] Embeddings carregados | base={embeddings1.shape} | query={embeddings2.shape}",
        flush=True,
    )

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
    
    print(">>> [SVS] Construindo índice Vamana...", flush=True)
    index = class_svs.build(
        base_embeddings=embeddings1,        # base indexada (df2)
        reduced_dims=128,                   # projeção para 128D
        graph_max_degree=64,                # M (grau máximo do grafo)
        window_size=128,                    # janela para construção
        distance="L2",                      # métrica L2
        num_threads=4,                      # paralelismo
        primary_kind="lvq4",
        secondary_kind="lvq8",
    )

    
    mem_after = process.memory_info().rss / (1024 ** 2)  # MB
    mem_used_create_index = mem_after - mem_before
    print(f"Memória utilizada na indexação (SVS): {mem_used_create_index:.2f} MB")
    index_time = time.time() - start_index_time
    print(f"Tempo de indexação (SVS): {index_time:.4f} segundos")

    # ================================
    #     BUSCA KNN (SVS)
    # ================================
    num_execucoes = NUM_EXECUCOES_BUSCA
    soma_tempo_busca = 0.0
    soma_qtde_mem = 0.0
    I = None
    D = None

    dict_ = {
        "execucao": [],
        "tempo_busca": [],
        "memoria_usada_busca_MB": [],
    }


    print("Searching SVS index")
    print(">>> [SVS] Iniciando busca KNN...", flush=True)
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

        dict_["execucao"].append(i+1)
        dict_["tempo_busca"].append(search_time)
        dict_["memoria_usada_busca_MB"].append(mem_used_search)

    avg_search_time = soma_tempo_busca / num_execucoes
    avg_mem_used_search = soma_qtde_mem / num_execucoes

    df_tempos_busca_svs = pd.DataFrame(dict_)
    df_tempos_busca_svs['modelo_index'] = 'svs'
    df_tempos_busca_svs['modelo_embedding'] = model

    df_aux = pd.read_csv(PATH_RESULTADOS_GERAL)
    df_aux = pd.concat([df_aux, df_tempos_busca_svs], ignore_index=True)
    df_aux.to_csv(PATH_RESULTADOS_GERAL, index=False)

    # ================================
    #     MERGE FUZZY
    # ================================
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # expandir df1 e df2 como na merge_knn
    df1_expanded = df2.loc[np.repeat(df2.index.values, k)].reset_index(drop=True)
    df2_expanded = df1.iloc[I.flatten()].reset_index(drop=True)    

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

    df_lm_matched.to_csv(f"df_lm_matched_svs.csv", index=False)

    print(
        f"LM matched (SVS) - left: {None}{suffixes[0]}, "
        f"right: {None}{suffixes[1]}"
    )

    safe_model = str(model).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")

    PATH_RESULTADOS_SVS = f"resultados/svs/{safe_model}"

    # ================================
    #   SALVAR RESULTADOS DE TEMPO
    # ================================
    if not os.path.exists(PATH_RESULTADOS_SVS):
        os.makedirs(PATH_RESULTADOS_SVS)

    RESULTADO_DE_TEMPO = f"csv_final_tempos_buscas.csv"
    df_tempos_busca_svs.to_csv(os.path.join(PATH_RESULTADOS_SVS, RESULTADO_DE_TEMPO), index=False)

    # ================================
    #   SALVAR MÉDIAS DOS RESULTADOS
    # ================================
    results_file = "resultados.csv"
    total_time = index_time + avg_search_time
    matches = (df_lm_matched["setor_esperado_x"] == df_lm_matched["setor_esperado_y"]).sum()
    results_data = {
        "metodo": ["svs"],
        "modelo_embedding": [model],
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

    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(results_file, mode="w", header=True, index=False)

    return True

def merge_knn_hnsw_julia(k, df1, df2, suffixes, model) -> DataFrame:
    """
    Versão HNSW (Julia) no mesmo padrão da merge_knn (FAISS):

    - Lê embeddings pré-computados de data/embeddings_base.npy e data/embeddings_query.npy
    - Constrói o índice HNSW em Julia sobre a base (df2 / embeddings_query)
    - Faz busca k-NN para df1
    - Faz merge fuzzy df1 x df2
    - Salva tempos em resultados.csv com metodo = "hnsw_julia"
    """
    # ================================
    #     CARREGAR EMBEDDINGS
    # ================================
    safe_model = str(model).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")

    embeddings1 = np.load(f"data/embeddings_base_{safe_model}.npy")
    embeddings2 = np.load(f"data/embeddings_query_{safe_model}.npy")

    # Normalizar (Julia HNSW geralmente trabalha com L2/cosseno)
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # ================================
    #     INDEXAÇÃO (HNSW em Julia)
    # ================================
    # Importante: ajustar caminho conforme a estrutura do projeto
    from julia import Main

    Main.include("hnsw_julia/hnsw_wrapper.jl")

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
    num_execucoes = NUM_EXECUCOES_BUSCA
    I = None
    D = None
    soma_memoria_usada = 0.0
    
    dict_ = {
        "execucao": [],
        "tempo_busca": [],
        "memoria_usada_busca_MB": [],
    }

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

        dict_["execucao"].append(i+1)
        dict_["tempo_busca"].append(tempo_busca)
        dict_["memoria_usada_busca_MB"].append(mem_used_search)

    df_tempos_busca_hnsw_julia = pd.DataFrame(dict_)
    df_tempos_busca_hnsw_julia['modelo_index'] = 'hnsw_julia'
    df_tempos_busca_hnsw_julia['modelo_embedding'] = model

    df_aux = pd.read_csv(PATH_RESULTADOS_GERAL)
    df_aux = pd.concat([df_aux, df_tempos_busca_hnsw_julia], ignore_index=True)
    df_aux.to_csv(PATH_RESULTADOS_GERAL, index=False)

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

    # expandir df1 e df2 como na merge_knn
    df1_expanded = df2.loc[np.repeat(df2.index.values, k)].reset_index(drop=True)
    df2_expanded = df1.iloc[I.flatten()].reset_index(drop=True)    

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

    safe_model = str(model).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")

    PATH_RESULTADOS_hnsw_julia = f"resultados/hnsw_julia/{safe_model}"

    # ================================
    #   SALVAR RESULTADOS DE TEMPO
    # ================================
    if not os.path.exists(PATH_RESULTADOS_hnsw_julia):
        os.makedirs(PATH_RESULTADOS_hnsw_julia)

    RESULTADO_DE_TEMPO = f"csv_final_tempos_buscas.csv"
    df_tempos_busca_hnsw_julia.to_csv(os.path.join(PATH_RESULTADOS_hnsw_julia, RESULTADO_DE_TEMPO), index=False)

    # ================================
    #   SALVAR MÉDIAS DOS RESULTADOS
    # ================================
    results_file = "resultados.csv"
    total_time = index_time + avg_search_time
    matches = (df_lm_matched["setor_esperado_x"] == df_lm_matched["setor_esperado_y"]).sum()
    results_data = {
        "metodo": ["hnsw_julia"],
        "modelo_embedding": [model],
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

    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(results_file, mode="w", header=True, index=False)

    return True

def merge_knn_nmslib(k, df1, df2, suffixes, model) -> DataFrame:
    """
    Versão NMSLIB (HNSW) no mesmo padrão da merge_knn (FAISS):

    - Lê embeddings pré-computados de data/embeddings_base.npy e data/embeddings_query.npy
    - Constrói índice HNSW (nmslib) sobre df2 / embeddings_query
    - Busca k-NN para df1
    - Faz merge fuzzy df1 x df2
    - Salva tempos em resultados.csv com metodo = "nmslib_hnsw"
    """
    import nmslib

    # ================================
    #     CARREGAR EMBEDDINGS
    # ================================
    safe_model = str(model).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")

    embeddings1 = np.load(f"data/embeddings_base_{safe_model}.npy")
    embeddings2 = np.load(f"data/embeddings_query_{safe_model}.npy")

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
            "M": 48,
            "efConstruction": 600,
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
    num_execucoes = NUM_EXECUCOES_BUSCA
    soma_tempo_busca = 0.0
    neighbors = None
    distances = None
    soma_qtde_mem = 0.0

    dict_ = {
        "execucao": [],
        "tempo_busca": [],
        "memoria_usada_busca_MB": [],
    }

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

        dict_["execucao"].append(i+1)
        dict_["tempo_busca"].append(search_time)
        dict_["memoria_usada_busca_MB"].append(mem_used_search)


    avg_search_time = soma_tempo_busca / num_execucoes
    avg_mem_used_search = soma_qtde_mem / num_execucoes
    print(f"Tempo médio de busca (NMSLIB HNSW) em {num_execucoes} execuções: {avg_search_time:.4f} segundos")

    df_tempos_busca_nmslib_hnsw = pd.DataFrame(dict_)
    df_tempos_busca_nmslib_hnsw['modelo_index'] = 'nmslib_hnsw'
    df_tempos_busca_nmslib_hnsw['modelo_embedding'] = model

    df_aux = pd.read_csv(PATH_RESULTADOS_GERAL)
    df_aux = pd.concat([df_aux, df_tempos_busca_nmslib_hnsw], ignore_index=True)
    df_aux.to_csv(PATH_RESULTADOS_GERAL, index=False)


    I = np.vstack(neighbors)       # indices em df2
    D_dist = np.vstack(distances)  # distâncias

    # Converter distâncias em similaridade (maior = melhor)
    score_sim = 1.0 - D_dist

    # ================================
    #     MERGE FUZZY
    # ================================
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # expandir df1 e df2 como na merge_knn
    df1_expanded = df2.loc[np.repeat(df2.index.values, k)].reset_index(drop=True)
    df2_expanded = df1.iloc[I.flatten()].reset_index(drop=True)    

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

    safe_model = str(model).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")

    PATH_RESULTADOS_NMSLIB = f"resultados/NMSLIB/{safe_model}"

    # ================================
    #   SALVAR RESULTADOS DE TEMPO
    # ================================
    if not os.path.exists(PATH_RESULTADOS_NMSLIB):
        os.makedirs(PATH_RESULTADOS_NMSLIB)

    RESULTADO_DE_TEMPO = f"csv_final_tempos_buscas.csv"
    df_tempos_busca_nmslib_hnsw.to_csv(os.path.join(PATH_RESULTADOS_NMSLIB, RESULTADO_DE_TEMPO), index=False)

    # ================================
    #   SALVAR MÉDIAS DOS RESULTADOS
    # ================================
    results_file = "resultados.csv"
    total_time = index_time + avg_search_time
    matches = (df_lm_matched["setor_esperado_x"] == df_lm_matched["setor_esperado_y"]).sum()
    results_data = {
        "metodo": ["NMSLIB"],
        "modelo_embedding": [model],
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

    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(results_file, mode="w", header=True, index=False)

    return True

def merge_knn_scann(k, df1, df2, suffixes) -> DataFrame:
    """
    Versão ScaNN no mesmo padrão da merge_knn (FAISS):

    - Lê embeddings pré-computados de data/embeddings_base.npy e data/embeddings_query.npy
    - Constrói índice ScaNN sobre df2 / embeddings_query
    - Faz busca k-NN para df1
    - Faz merge fuzzy df1 x df2
    - Salva tempos em resultados.csv com metodo = "scann_knn"
    """
    import scann

    # ================================
    #     CARREGAR EMBEDDINGS
    # ================================
    embeddings1 = np.load("data/embeddings_base.npy")   # df1
    embeddings2 = np.load("data/embeddings_query.npy")  # df2

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
    num_execucoes = NUM_EXECUCOES_BUSCA
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
