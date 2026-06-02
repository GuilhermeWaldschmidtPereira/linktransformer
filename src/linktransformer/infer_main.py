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

def get_data_dir_candidates() -> List[str]:
    candidates = []
    env_data_dir = os.environ.get("LINKTRANSFORMER_DATA_DIR")
    if env_data_dir:
        candidates.append(os.path.abspath(env_data_dir))
    candidates.extend([
        os.path.abspath("data"),
        os.path.abspath(os.path.join("linktransformer", "data")),
    ])
    return list(dict.fromkeys(candidates))


DATA_DIR_CANDIDATES = get_data_dir_candidates()


def build_results_dir() -> str:
    results_dir = os.environ.get("LINKTRANSFORMER_RESULTS_DIR")
    if results_dir:
        return os.path.abspath(results_dir)
    return os.path.abspath(f"resultados_{time.strftime('%d%m%Y%H%M%S')}")


PATH_RESULTADOS = build_results_dir()

NUM_EXECUCOES_BUSCA = 1

if not os.path.exists(PATH_RESULTADOS):
    os.makedirs(PATH_RESULTADOS)

NAME_DF_RESULTADOS_GERAL = "resultados_geral.csv"
PATH_RESULTADOS_GERAL = os.path.join(PATH_RESULTADOS, NAME_DF_RESULTADOS_GERAL)
PATH_RESULTADOS_POR_MUNICIPIO_DIR = os.path.join(PATH_RESULTADOS, "resultados_por_municipio")
PATH_RESULTADOS_POR_MUNICIPIO = os.path.join(PATH_RESULTADOS, "resultados_por_municipio.csv")

RESULTADOS_POR_MUNICIPIO_COLUMNS = [
    "metodo",
    "modelo_embedding",
    "id_municipio",
    "index_time",
    "search_time",
    "total_time",
    "num_rows_df1",
    "num_rows_df2",
    "quantidade_enderecos_por_municipio",
    "quantidade_enderecos_buscados_municipio",
    "quantidade_acertos_municipio",
    "k",
    "k_efetivo",
    "mem_used_indexation_MB",
    "avg_mem_used_search_MB",
    "matches",
]

df_geral = pd.DataFrame(columns=[
    "execucao",
    "tempo_busca",
    "memoria_usada_busca_MB",
    "modelo_index",
    "modelo_embedding",
])

df_geral.to_csv(PATH_RESULTADOS_GERAL, index=False)


def safe_model_name(model) -> str:
    safe_model = str(model).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")
    return safe_model


def sanitize_municipality_id(municipality_id: object) -> str:
    if pd.notna(municipality_id) and isinstance(municipality_id, (int, float, np.integer, np.floating)):
        municipality_float = float(municipality_id)
        if municipality_float.is_integer():
            return str(int(municipality_float))

    value = str(municipality_id).strip()
    if not value:
        return "vazio"

    try:
        municipality_float = float(value)
        if municipality_float.is_integer():
            return str(int(municipality_float))
    except ValueError:
        pass

    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def get_flat_embeddings_path(side: str, safe_model: str) -> str:
    filename = f"embeddings_{side}_{safe_model}.npy"
    for data_dir in DATA_DIR_CANDIDATES:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return path
    return os.path.join(DATA_DIR_CANDIDATES[0], filename)


def get_partitioned_embeddings_dir(side: str, safe_model: str) -> str:
    relative_path = os.path.join(side, safe_model)
    for data_dir in DATA_DIR_CANDIDATES:
        path = os.path.join(data_dir, relative_path)
        if os.path.isdir(path):
            return path
    return os.path.join(DATA_DIR_CANDIDATES[0], relative_path)


def get_partitioned_embeddings_path(side: str, safe_model: str, municipio_id: object) -> str:
    municipio_safe = sanitize_municipality_id(municipio_id)
    return os.path.join(
        get_partitioned_embeddings_dir(side, safe_model),
        f"embedding_{side}_{municipio_safe}.npy",
    )


def has_partitioned_embeddings(safe_model: str) -> bool:
    return all(
        os.path.isdir(get_partitioned_embeddings_dir(side, safe_model))
        for side in ("base", "query")
    )


def load_flat_embeddings(safe_model: str) -> Tuple[np.ndarray, np.ndarray]:
    embeddings_base_path = get_flat_embeddings_path("base", safe_model)
    embeddings_query_path = get_flat_embeddings_path("query", safe_model)
    return np.load(embeddings_base_path), np.load(embeddings_query_path)


def load_partitioned_embeddings(
    side: str,
    safe_model: str,
    municipio_id: object,
    expected_rows: int,
) -> np.ndarray:
    path = get_partitioned_embeddings_path(side, safe_model, municipio_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Embedding particionado não encontrado para {side}, "
            f"modelo={safe_model}, municipio={municipio_id}: {path}"
        )

    embeddings = np.load(path)
    if len(embeddings) != expected_rows:
        raise ValueError(
            f"Quantidade de embeddings incompatível em {path}: "
            f"esperado={expected_rows}, encontrado={len(embeddings)}"
        )
    return embeddings


def get_municipio_column(df1: DataFrame, df2: DataFrame) -> str:
    for column in ("id_municipio", "municipio"):
        if column in df1.columns and column in df2.columns:
            return column
    raise ValueError(
        "Não encontrei uma coluna de município comum. "
        "Use 'id_municipio' ou 'municipio' em base e query."
    )


def append_csv(path: str, df: DataFrame) -> None:
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def append_resultados_por_municipio(df_resultados: DataFrame, metodo: str) -> None:
    if df_resultados.empty:
        return
    os.makedirs(PATH_RESULTADOS_POR_MUNICIPIO_DIR, exist_ok=True)
    df_resultados = df_resultados.reindex(columns=RESULTADOS_POR_MUNICIPIO_COLUMNS)
    append_csv(PATH_RESULTADOS_POR_MUNICIPIO, df_resultados)
    append_csv(os.path.join(PATH_RESULTADOS_POR_MUNICIPIO_DIR, f"{metodo}.csv"), df_resultados)


def build_matches_por_municipio(
    df_base_mun: DataFrame,
    df_query_mun: DataFrame,
    I: np.ndarray,
    k_efetivo: int,
    suffixes: Tuple[str, str],
    scores: Optional[np.ndarray] = None,
) -> DataFrame:
    df_query_expanded = df_query_mun.loc[
        np.repeat(df_query_mun.index.values, k_efetivo)
    ].reset_index(drop=True)
    df_base_expanded = df_base_mun.iloc[I.flatten()].reset_index(drop=True)

    df_lm_matched = df_query_expanded.merge(
        df_base_expanded,
        left_index=True,
        right_index=True,
        how="inner",
        suffixes=suffixes,
    )

    if scores is not None:
        df_lm_matched["score"] = scores.flatten()

    return df_lm_matched


def count_setor_matches(df_lm_matched: DataFrame) -> int:
    if "setor_censitario_x" in df_lm_matched.columns and "setor_censitario_y" in df_lm_matched.columns:
        return int((df_lm_matched["setor_censitario_x"] == df_lm_matched["setor_censitario_y"]).sum())
    return 0

def merge_knn(k, df1,df2, suffixes, model) -> DataFrame:
    # ================================
    #     INDEXAÇÃO (FAISS)
    # ================================
    # Medir tempo de criação do índice + add
    safe_model = safe_model_name(model)

    use_partitioned_embeddings = has_partitioned_embeddings(safe_model)
    embeddings1 = None
    embeddings2 = None
    if use_partitioned_embeddings:
        print(
            f">>> [FAISS] Usando embeddings particionados em "
            f"{get_partitioned_embeddings_dir('base', safe_model)} e "
            f"{get_partitioned_embeddings_dir('query', safe_model)}",
            flush=True,
        )
    else:
        print(f">>> [FAISS] Carregando embeddings do modelo: {model}", flush=True)
        embeddings1, embeddings2 = load_flat_embeddings(safe_model)
        print(
            f">>> [FAISS] Embeddings carregados | base={embeddings1.shape} | query={embeddings2.shape}",
            flush=True,
        )

    df1 = df1.copy().reset_index(drop=True)
    df2 = df2.copy().reset_index(drop=True)
    municipio_col = get_municipio_column(df1, df2)
    process = psutil.Process(os.getpid())
    num_execucoes = NUM_EXECUCOES_BUSCA

    dict_ = {
        "execucao": [],
        "tempo_busca": [],
        "memoria_usada_busca_MB": [],
    }
    resultados_municipio = []
    matched_parts = []

    print(">>> [FAISS] Iniciando busca por município...", flush=True)
    for id_municipio in df2[municipio_col].dropna().drop_duplicates().tolist():
        base_idx = df1.index[df1[municipio_col] == id_municipio].to_numpy()
        query_idx = df2.index[df2[municipio_col] == id_municipio].to_numpy()

        if len(base_idx) == 0 or len(query_idx) == 0:
            continue

        k_efetivo = min(k, len(base_idx))
        if use_partitioned_embeddings:
            embeddings_base_mun = load_partitioned_embeddings("base", safe_model, id_municipio, len(base_idx))
            embeddings_query_mun = load_partitioned_embeddings("query", safe_model, id_municipio, len(query_idx))
        else:
            embeddings_base_mun = embeddings1[base_idx]
            embeddings_query_mun = embeddings2[query_idx]

        start_index_time = time.time()
        mem_before = process.memory_info().rss / (1024 ** 2)
        index = faiss.IndexFlatIP(embeddings_base_mun.shape[1])
        index.add(embeddings_base_mun)
        mem_after = process.memory_info().rss / (1024 ** 2)
        mem_used_create_index = mem_after - mem_before
        index_time = time.time() - start_index_time

        soma_tempo_busca = 0.0
        soma_memoria_busca = 0.0
        D = None
        I = None
        for i in range(num_execucoes):
            start_search_time = time.time()
            mem_before = process.memory_info().rss / (1024 ** 2)
            D, I = index.search(embeddings_query_mun, k_efetivo)
            mem_after = process.memory_info().rss / (1024 ** 2)
            mem_used_search = mem_after - mem_before
            search_time = time.time() - start_search_time
            soma_tempo_busca += search_time
            soma_memoria_busca += mem_used_search

            dict_["execucao"].append(i + 1)
            dict_["tempo_busca"].append(search_time)
            dict_["memoria_usada_busca_MB"].append(mem_used_search)

        avg_search_time = soma_tempo_busca / num_execucoes
        avg_mem_used_search = soma_memoria_busca / num_execucoes

        df_lm_matched_mun = build_matches_por_municipio(
            df1.iloc[base_idx].reset_index(drop=True),
            df2.iloc[query_idx].reset_index(drop=True),
            I,
            k_efetivo,
            suffixes,
            D,
        )
        matched_parts.append(df_lm_matched_mun)
        matches = count_setor_matches(df_lm_matched_mun)

        resultados_municipio.append({
            "metodo": "baseline",
            "modelo_embedding": model,
            "id_municipio": id_municipio,
            "index_time": index_time,
            "search_time": avg_search_time,
            "total_time": index_time + avg_search_time,
            "num_rows_df1": len(base_idx),
            "num_rows_df2": len(query_idx),
            "quantidade_enderecos_por_municipio": len(base_idx),
            "quantidade_enderecos_buscados_municipio": len(query_idx),
            "quantidade_acertos_municipio": matches,
            "k": k,
            "k_efetivo": k_efetivo,
            "mem_used_indexation_MB": mem_used_create_index,
            "avg_mem_used_search_MB": avg_mem_used_search,
            "matches": matches,
        })

    df_lm_matched = pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame()
    df_resultados_municipio = pd.DataFrame(resultados_municipio)
    append_resultados_por_municipio(df_resultados_municipio, "baseline")

    total_index_time = df_resultados_municipio["index_time"].sum() if not df_resultados_municipio.empty else 0.0
    total_search_time = df_resultados_municipio["search_time"].sum() if not df_resultados_municipio.empty else 0.0
    avg_mem_used_search = df_resultados_municipio["avg_mem_used_search_MB"].mean() if not df_resultados_municipio.empty else 0.0
    mem_used_create_index = df_resultados_municipio["mem_used_indexation_MB"].sum() if not df_resultados_municipio.empty else 0.0

    print(
        f"LM matched on key columns - left: {None}{suffixes[0]}, "
        f"right: {None}{suffixes[1]}"
    )

    PATH_RESULTADOS_baseline = os.path.join(PATH_RESULTADOS, "baseline", safe_model)

    # ================================
    #   SALVAR RESULTADOS DE TEMPO
    # ================================
    if not os.path.exists(PATH_RESULTADOS_baseline):
        os.makedirs(PATH_RESULTADOS_baseline)

    RESULTADO_DE_TEMPO = f"csv_final_tempos_buscas.csv"
    df_tempos_busca_faiss_baseline = pd.DataFrame(dict_)
    df_tempos_busca_faiss_baseline['modelo_index'] = 'faiss_baseline'
    df_tempos_busca_faiss_baseline['modelo_embedding'] = model

    df_aux = pd.read_csv(PATH_RESULTADOS_GERAL)
    df_aux = pd.concat([df_aux, df_tempos_busca_faiss_baseline], ignore_index=True)
    df_aux.to_csv(PATH_RESULTADOS_GERAL, index=False)

    df_tempos_busca_faiss_baseline.to_csv(os.path.join(PATH_RESULTADOS_baseline, RESULTADO_DE_TEMPO), index=False)

    # ================================
    #   SALVAR MÉDIAS DOS RESULTADOS
    # ================================
    results_file = os.path.join(PATH_RESULTADOS, "resultados.csv")
    total_time = total_index_time + total_search_time
    df_lm_matched.to_csv(os.path.join(PATH_RESULTADOS, "teste.csv"), index=False)
    matches = count_setor_matches(df_lm_matched)
    results_data = {
        "metodo": ["baseline"],
        "modelo_embedding": [model],
        "index_time": [total_index_time],
        "search_time": [total_search_time],
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
    safe_model = safe_model_name(model)

    use_partitioned_embeddings = has_partitioned_embeddings(safe_model)
    embeddings1 = None
    embeddings2 = None
    if use_partitioned_embeddings:
        print(
            f">>> [SVS] Usando embeddings particionados em "
            f"{get_partitioned_embeddings_dir('base', safe_model)} e "
            f"{get_partitioned_embeddings_dir('query', safe_model)}",
            flush=True,
        )
    else:
        print(f">>> [SVS] Carregando embeddings do modelo: {model}", flush=True)
        embeddings1, embeddings2 = load_flat_embeddings(safe_model)
        print(
            f">>> [SVS] Embeddings carregados | base={embeddings1.shape} | query={embeddings2.shape}",
            flush=True,
        )

    df1 = df1.copy().reset_index(drop=True)
    df2 = df2.copy().reset_index(drop=True)
    municipio_col = get_municipio_column(df1, df2)
    class_svs = VamanaIndexer()
    process = psutil.Process(os.getpid())
    num_execucoes = NUM_EXECUCOES_BUSCA

    dict_ = {
        "execucao": [],
        "tempo_busca": [],
        "memoria_usada_busca_MB": [],
    }
    resultados_municipio = []
    matched_parts = []

    print(">>> [SVS] Iniciando busca por município...", flush=True)
    for id_municipio in df2[municipio_col].dropna().drop_duplicates().tolist():
        base_idx = df1.index[df1[municipio_col] == id_municipio].to_numpy()
        query_idx = df2.index[df2[municipio_col] == id_municipio].to_numpy()

        if len(base_idx) == 0 or len(query_idx) == 0:
            continue

        k_efetivo = min(k, len(base_idx))
        if use_partitioned_embeddings:
            embeddings_base_mun = load_partitioned_embeddings("base", safe_model, id_municipio, len(base_idx))
            embeddings_query_mun = load_partitioned_embeddings("query", safe_model, id_municipio, len(query_idx))
        else:
            embeddings_base_mun = embeddings1[base_idx]
            embeddings_query_mun = embeddings2[query_idx]

        start_index_time = time.time()
        mem_before = process.memory_info().rss / (1024 ** 2)
        index = class_svs.build(
            base_embeddings=embeddings_base_mun,
            reduced_dims=128,
            graph_max_degree=64,
            window_size=128,
            distance="L2",
            num_threads=4,
            primary_kind="lvq4",
            secondary_kind="lvq8",
        )
        mem_after = process.memory_info().rss / (1024 ** 2)
        mem_used_create_index = mem_after - mem_before
        index_time = time.time() - start_index_time

        soma_tempo_busca = 0.0
        soma_memoria_busca = 0.0
        I = None
        D = None
        for i in range(num_execucoes):
            start_search_time = time.time()
            mem_before = process.memory_info().rss / (1024 ** 2)
            I, D = index.search(embeddings_query_mun, k_efetivo)
            mem_after = process.memory_info().rss / (1024 ** 2)
            mem_used_search = mem_after - mem_before
            search_time = time.time() - start_search_time
            soma_tempo_busca += search_time
            soma_memoria_busca += mem_used_search

            dict_["execucao"].append(i + 1)
            dict_["tempo_busca"].append(search_time)
            dict_["memoria_usada_busca_MB"].append(mem_used_search)

        avg_search_time = soma_tempo_busca / num_execucoes
        avg_mem_used_search = soma_memoria_busca / num_execucoes

        df_lm_matched_mun = build_matches_por_municipio(
            df1.iloc[base_idx].reset_index(drop=True),
            df2.iloc[query_idx].reset_index(drop=True),
            I,
            k_efetivo,
            suffixes,
            D,
        )
        matched_parts.append(df_lm_matched_mun)
        matches = count_setor_matches(df_lm_matched_mun)

        resultados_municipio.append({
            "metodo": "svs",
            "modelo_embedding": model,
            "id_municipio": id_municipio,
            "index_time": index_time,
            "search_time": avg_search_time,
            "total_time": index_time + avg_search_time,
            "num_rows_df1": len(base_idx),
            "num_rows_df2": len(query_idx),
            "quantidade_enderecos_por_municipio": len(base_idx),
            "quantidade_enderecos_buscados_municipio": len(query_idx),
            "quantidade_acertos_municipio": matches,
            "k": k,
            "k_efetivo": k_efetivo,
            "mem_used_indexation_MB": mem_used_create_index,
            "avg_mem_used_search_MB": avg_mem_used_search,
            "matches": matches,
        })

    df_lm_matched = pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame()
    df_resultados_municipio = pd.DataFrame(resultados_municipio)
    append_resultados_por_municipio(df_resultados_municipio, "svs")

    total_index_time = df_resultados_municipio["index_time"].sum() if not df_resultados_municipio.empty else 0.0
    total_search_time = df_resultados_municipio["search_time"].sum() if not df_resultados_municipio.empty else 0.0
    avg_mem_used_search = df_resultados_municipio["avg_mem_used_search_MB"].mean() if not df_resultados_municipio.empty else 0.0
    mem_used_create_index = df_resultados_municipio["mem_used_indexation_MB"].sum() if not df_resultados_municipio.empty else 0.0

    df_tempos_busca_svs = pd.DataFrame(dict_)
    df_tempos_busca_svs['modelo_index'] = 'svs'
    df_tempos_busca_svs['modelo_embedding'] = model

    df_aux = pd.read_csv(PATH_RESULTADOS_GERAL)
    df_aux = pd.concat([df_aux, df_tempos_busca_svs], ignore_index=True)
    df_aux.to_csv(PATH_RESULTADOS_GERAL, index=False)

    df_lm_matched.to_csv(os.path.join(PATH_RESULTADOS, "df_lm_matched_svs.csv"), index=False)

    print(
        f"LM matched (SVS) - left: {None}{suffixes[0]}, "
        f"right: {None}{suffixes[1]}"
    )

    PATH_RESULTADOS_SVS = os.path.join(PATH_RESULTADOS, "svs", safe_model)

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
    results_file = os.path.join(PATH_RESULTADOS, "resultados.csv")
    total_time = total_index_time + total_search_time
    matches = count_setor_matches(df_lm_matched)
    results_data = {
        "metodo": ["svs"],
        "modelo_embedding": [model],
        "index_time": [total_index_time],
        "search_time": [total_search_time],
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
    safe_model = safe_model_name(model)

    use_partitioned_embeddings = has_partitioned_embeddings(safe_model)
    embeddings1 = None
    embeddings2 = None
    if use_partitioned_embeddings:
        print(
            f">>> [HNSW Julia] Usando embeddings particionados em "
            f"{get_partitioned_embeddings_dir('base', safe_model)} e "
            f"{get_partitioned_embeddings_dir('query', safe_model)}",
            flush=True,
        )
    else:
        embeddings1, embeddings2 = load_flat_embeddings(safe_model)

        # Normalizar (Julia HNSW geralmente trabalha com L2/cosseno)
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    # ================================
    #     INDEXAÇÃO (HNSW em Julia)
    # ================================
    # Importante: ajustar caminho conforme a estrutura do projeto
    from julia import Main

    Main.include("hnsw_julia/hnsw_wrapper.jl")

    df1 = df1.copy().reset_index(drop=True)
    df2 = df2.copy().reset_index(drop=True)
    municipio_col = get_municipio_column(df1, df2)
    process = psutil.Process(os.getpid())
    num_execucoes = NUM_EXECUCOES_BUSCA

    dict_ = {
        "execucao": [],
        "tempo_busca": [],
        "memoria_usada_busca_MB": [],
    }
    resultados_municipio = []
    matched_parts = []

    print(">>> [HNSW Julia] Iniciando busca por município...", flush=True)
    for id_municipio in df2[municipio_col].dropna().drop_duplicates().tolist():
        base_idx = df1.index[df1[municipio_col] == id_municipio].to_numpy()
        query_idx = df2.index[df2[municipio_col] == id_municipio].to_numpy()

        if len(base_idx) == 0 or len(query_idx) == 0:
            continue

        k_efetivo = min(k, len(base_idx))
        if use_partitioned_embeddings:
            embeddings_base_mun = load_partitioned_embeddings("base", safe_model, id_municipio, len(base_idx))
            embeddings_query_mun = load_partitioned_embeddings("query", safe_model, id_municipio, len(query_idx))
            embeddings_base_mun = embeddings_base_mun / np.linalg.norm(embeddings_base_mun, axis=1, keepdims=True)
            embeddings_query_mun = embeddings_query_mun / np.linalg.norm(embeddings_query_mun, axis=1, keepdims=True)
        else:
            embeddings_base_mun = embeddings1[base_idx]
            embeddings_query_mun = embeddings2[query_idx]

        start_index_time = time.time()
        mem_before = process.memory_info().rss / (1024 ** 2)
        hnsw = Main.build_hnsw(embeddings_base_mun)
        mem_after = process.memory_info().rss / (1024 ** 2)
        mem_used_create_index = mem_after - mem_before
        index_time = time.time() - start_index_time

        soma_tempo_busca = 0.0
        soma_memoria_busca = 0.0
        I = None
        D = None
        for i in range(num_execucoes):
            mem_before = process.memory_info().rss / (1024 ** 2)
            I, D, tempo_busca = Main.search_hnsw(hnsw, embeddings_query_mun, K=k_efetivo)
            mem_after = process.memory_info().rss / (1024 ** 2)
            mem_used_search = mem_after - mem_before
            soma_tempo_busca += tempo_busca
            soma_memoria_busca += mem_used_search

            dict_["execucao"].append(i + 1)
            dict_["tempo_busca"].append(tempo_busca)
            dict_["memoria_usada_busca_MB"].append(mem_used_search)

        I = np.asarray(I) - 1
        D = np.asarray(D)
        avg_search_time = soma_tempo_busca / num_execucoes
        avg_mem_used_search = soma_memoria_busca / num_execucoes

        df_lm_matched_mun = build_matches_por_municipio(
            df1.iloc[base_idx].reset_index(drop=True),
            df2.iloc[query_idx].reset_index(drop=True),
            I,
            k_efetivo,
            suffixes,
            D,
        )
        matched_parts.append(df_lm_matched_mun)
        matches = count_setor_matches(df_lm_matched_mun)

        resultados_municipio.append({
            "metodo": "hnsw_julia",
            "modelo_embedding": model,
            "id_municipio": id_municipio,
            "index_time": index_time,
            "search_time": avg_search_time,
            "total_time": index_time + avg_search_time,
            "num_rows_df1": len(base_idx),
            "num_rows_df2": len(query_idx),
            "quantidade_enderecos_por_municipio": len(base_idx),
            "quantidade_enderecos_buscados_municipio": len(query_idx),
            "quantidade_acertos_municipio": matches,
            "k": k,
            "k_efetivo": k_efetivo,
            "mem_used_indexation_MB": mem_used_create_index,
            "avg_mem_used_search_MB": avg_mem_used_search,
            "matches": matches,
        })

    df_lm_matched = pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame()
    df_resultados_municipio = pd.DataFrame(resultados_municipio)
    append_resultados_por_municipio(df_resultados_municipio, "hnsw_julia")

    total_index_time = df_resultados_municipio["index_time"].sum() if not df_resultados_municipio.empty else 0.0
    total_search_time = df_resultados_municipio["search_time"].sum() if not df_resultados_municipio.empty else 0.0
    avg_mem_used_search = df_resultados_municipio["avg_mem_used_search_MB"].mean() if not df_resultados_municipio.empty else 0.0
    mem_used_create_index = df_resultados_municipio["mem_used_indexation_MB"].sum() if not df_resultados_municipio.empty else 0.0

    df_tempos_busca_hnsw_julia = pd.DataFrame(dict_)
    df_tempos_busca_hnsw_julia['modelo_index'] = 'hnsw_julia'
    df_tempos_busca_hnsw_julia['modelo_embedding'] = model

    df_aux = pd.read_csv(PATH_RESULTADOS_GERAL)
    df_aux = pd.concat([df_aux, df_tempos_busca_hnsw_julia], ignore_index=True)
    df_aux.to_csv(PATH_RESULTADOS_GERAL, index=False)

    print(
        f"LM matched (HNSW Julia) - left: {None}{suffixes[0]}, "
        f"right: {None}{suffixes[1]}"
    )

    PATH_RESULTADOS_hnsw_julia = os.path.join(PATH_RESULTADOS, "hnsw_julia", safe_model)

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
    results_file = os.path.join(PATH_RESULTADOS, "resultados.csv")
    total_time = total_index_time + total_search_time
    matches = count_setor_matches(df_lm_matched)
    results_data = {
        "metodo": ["hnsw_julia"],
        "modelo_embedding": [model],
        "index_time": [total_index_time],
        "search_time": [total_search_time],
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
    safe_model = safe_model_name(model)

    use_partitioned_embeddings = has_partitioned_embeddings(safe_model)
    embeddings1 = None
    embeddings2 = None
    if use_partitioned_embeddings:
        print(
            f">>> [NMSLIB] Usando embeddings particionados em "
            f"{get_partitioned_embeddings_dir('base', safe_model)} e "
            f"{get_partitioned_embeddings_dir('query', safe_model)}",
            flush=True,
        )
    else:
        embeddings1, embeddings2 = load_flat_embeddings(safe_model)

        # Normalizar para cosinesimil
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    df1 = df1.copy().reset_index(drop=True)
    df2 = df2.copy().reset_index(drop=True)
    municipio_col = get_municipio_column(df1, df2)
    process = psutil.Process(os.getpid())
    num_execucoes = NUM_EXECUCOES_BUSCA

    dict_ = {
        "execucao": [],
        "tempo_busca": [],
        "memoria_usada_busca_MB": [],
    }
    resultados_municipio = []
    matched_parts = []

    print(">>> [NMSLIB] Iniciando busca por município...", flush=True)
    for id_municipio in df2[municipio_col].dropna().drop_duplicates().tolist():
        base_idx = df1.index[df1[municipio_col] == id_municipio].to_numpy()
        query_idx = df2.index[df2[municipio_col] == id_municipio].to_numpy()

        if len(base_idx) == 0 or len(query_idx) == 0:
            continue

        k_efetivo = min(k, len(base_idx))
        if use_partitioned_embeddings:
            embeddings_base_mun = load_partitioned_embeddings("base", safe_model, id_municipio, len(base_idx))
            embeddings_query_mun = load_partitioned_embeddings("query", safe_model, id_municipio, len(query_idx))
            embeddings_base_mun = embeddings_base_mun / np.linalg.norm(embeddings_base_mun, axis=1, keepdims=True)
            embeddings_query_mun = embeddings_query_mun / np.linalg.norm(embeddings_query_mun, axis=1, keepdims=True)
        else:
            embeddings_base_mun = embeddings1[base_idx]
            embeddings_query_mun = embeddings2[query_idx]

        start_index_time = time.time()
        mem_before = process.memory_info().rss / (1024 ** 2)
        index = nmslib.init(
            space="cosinesimil",
            method="hnsw"
        )
        index.addDataPointBatch(embeddings_base_mun)
        index.createIndex(
            {
                "M": 48,
                "efConstruction": 600,
            },
            print_progress=False,
        )
        index.setQueryTimeParams({"efSearch": 50})
        mem_after = process.memory_info().rss / (1024 ** 2)
        mem_used_create_index = mem_after - mem_before
        index_time = time.time() - start_index_time

        soma_tempo_busca = 0.0
        soma_memoria_busca = 0.0
        neighbors = None
        distances = None
        for i in range(num_execucoes):
            start_search_time = time.time()
            mem_before = process.memory_info().rss / (1024 ** 2)
            res = index.knnQueryBatch(embeddings_query_mun, k=k_efetivo)
            mem_after = process.memory_info().rss / (1024 ** 2)
            mem_used_search = mem_after - mem_before
            search_time = time.time() - start_search_time
            soma_tempo_busca += search_time
            soma_memoria_busca += mem_used_search
            neighbors, distances = zip(*res)

            dict_["execucao"].append(i + 1)
            dict_["tempo_busca"].append(search_time)
            dict_["memoria_usada_busca_MB"].append(mem_used_search)

        avg_search_time = soma_tempo_busca / num_execucoes
        avg_mem_used_search = soma_memoria_busca / num_execucoes
        I = np.vstack(neighbors)
        D_dist = np.vstack(distances)
        score_sim = 1.0 - D_dist

        df_lm_matched_mun = build_matches_por_municipio(
            df1.iloc[base_idx].reset_index(drop=True),
            df2.iloc[query_idx].reset_index(drop=True),
            I,
            k_efetivo,
            suffixes,
            score_sim,
        )
        matched_parts.append(df_lm_matched_mun)
        matches = count_setor_matches(df_lm_matched_mun)

        resultados_municipio.append({
            "metodo": "NMSLIB",
            "modelo_embedding": model,
            "id_municipio": id_municipio,
            "index_time": index_time,
            "search_time": avg_search_time,
            "total_time": index_time + avg_search_time,
            "num_rows_df1": len(base_idx),
            "num_rows_df2": len(query_idx),
            "quantidade_enderecos_por_municipio": len(base_idx),
            "quantidade_enderecos_buscados_municipio": len(query_idx),
            "quantidade_acertos_municipio": matches,
            "k": k,
            "k_efetivo": k_efetivo,
            "mem_used_indexation_MB": mem_used_create_index,
            "avg_mem_used_search_MB": avg_mem_used_search,
            "matches": matches,
        })

    df_lm_matched = pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame()
    df_resultados_municipio = pd.DataFrame(resultados_municipio)
    append_resultados_por_municipio(df_resultados_municipio, "NMSLIB")

    total_index_time = df_resultados_municipio["index_time"].sum() if not df_resultados_municipio.empty else 0.0
    total_search_time = df_resultados_municipio["search_time"].sum() if not df_resultados_municipio.empty else 0.0
    avg_mem_used_search = df_resultados_municipio["avg_mem_used_search_MB"].mean() if not df_resultados_municipio.empty else 0.0
    mem_used_create_index = df_resultados_municipio["mem_used_indexation_MB"].sum() if not df_resultados_municipio.empty else 0.0

    df_tempos_busca_nmslib_hnsw = pd.DataFrame(dict_)
    df_tempos_busca_nmslib_hnsw['modelo_index'] = 'nmslib_hnsw'
    df_tempos_busca_nmslib_hnsw['modelo_embedding'] = model

    df_aux = pd.read_csv(PATH_RESULTADOS_GERAL)
    df_aux = pd.concat([df_aux, df_tempos_busca_nmslib_hnsw], ignore_index=True)
    df_aux.to_csv(PATH_RESULTADOS_GERAL, index=False)


    print(
        f"LM matched (NMSLIB HNSW) - left: {None}{suffixes[0]}, "
        f"right: {None}{suffixes[1]}"
    )

    PATH_RESULTADOS_NMSLIB = os.path.join(PATH_RESULTADOS, "NMSLIB", safe_model)

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
    results_file = os.path.join(PATH_RESULTADOS, "resultados.csv")
    total_time = total_index_time + total_search_time
    matches = count_setor_matches(df_lm_matched)
    results_data = {
        "metodo": ["NMSLIB"],
        "modelo_embedding": [model],
        "index_time": [total_index_time],
        "search_time": [total_search_time],
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
    results_file = os.path.join(PATH_RESULTADOS, "resultados.csv")
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
