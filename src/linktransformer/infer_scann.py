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

def build_results_dir() -> str:
    results_dir = os.environ.get("LINKTRANSFORMER_RESULTS_DIR")
    if results_dir:
        return os.path.abspath(results_dir)
    return os.path.abspath(f"resultados_{time.strftime('%d%m%Y%H%M%S')}")


PATH_RESULTADOS = build_results_dir()

if not os.path.exists(PATH_RESULTADOS):
    os.makedirs(PATH_RESULTADOS)

NAME_DF_resultados_scann = "resultados_scann.csv"
PATH_resultados_scann = os.path.join(PATH_RESULTADOS, NAME_DF_resultados_scann)
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


def safe_model_name(model) -> str:
    safe_model = str(model).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")
    return safe_model


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


def build_scann_searcher(embeddings1: np.ndarray, k: int):
    import scann

    builder_mode = os.environ.get("SCANN_BUILDER_MODE", "brute_force").strip().lower()
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
    safe_model = safe_model_name(model)

    PATH_RESULTADOS_scann = os.path.join(PATH_RESULTADOS, "scann", safe_model)
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
    embeddings2 = np.load(embeddings_query_path)
    print(
        f">>> [ScaNN] Embeddings carregados | base={embeddings1.shape} | query={embeddings2.shape}",
        flush=True,
    )

    df1 = df1.copy().reset_index(drop=True)
    df2 = df2.copy().reset_index(drop=True)
    municipio_col = get_municipio_column(df1, df2)

    print(">>> [ScaNN] Normalizando embeddings...", flush=True)
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    print(">>> [ScaNN] Normalização concluída.", flush=True)

    num_execucoes = get_env_int("SCANN_NUM_EXECUCOES", 1)
    query_batch_size = get_env_int("SCANN_QUERY_BATCH_SIZE", 0)
    leaves_to_search_override = os.environ.get("SCANN_LEAVES_TO_SEARCH")
    pre_reorder_override = os.environ.get("SCANN_PRE_REORDER_NUM_NEIGHBORS")

    dict_ = {
        "execucao": [],
        "tempo_busca": [],
        "memoria_usada_busca_MB": [],
    }
    resultados_municipio = []
    matched_parts = []

    print(">>> [ScaNN] Iniciando indexação e busca por município...", flush=True)
    for id_municipio in df2[municipio_col].dropna().drop_duplicates().tolist():
        base_idx = df1.index[df1[municipio_col] == id_municipio].to_numpy()
        query_idx = df2.index[df2[municipio_col] == id_municipio].to_numpy()

        if len(base_idx) == 0 or len(query_idx) == 0:
            continue

        k_efetivo = min(k, len(base_idx))
        embeddings_base_mun = embeddings1[base_idx]
        embeddings_query_mun = embeddings2[query_idx]

        mem_before = process.memory_info().rss / (1024 ** 2)
        start_index_time = time.time()
        builder = build_scann_searcher(embeddings_base_mun, k_efetivo)
        searcher = builder.build()
        index_time = time.time() - start_index_time
        mem_after = process.memory_info().rss / (1024 ** 2)
        mem_used_create_index = mem_after - mem_before

        search_kwargs = {"final_num_neighbors": k_efetivo}
        if leaves_to_search_override not in (None, ""):
            search_kwargs["leaves_to_search"] = int(leaves_to_search_override)
        if pre_reorder_override not in (None, ""):
            search_kwargs["pre_reorder_num_neighbors"] = int(pre_reorder_override)

        soma_tempo_busca = 0.0
        soma_memoria = 0.0
        I = None
        D = None
        for i in range(num_execucoes):
            print(
                f">>> [ScaNN] Município {id_municipio} | execução "
                f"{i + 1}/{num_execucoes}",
                flush=True,
            )
            mem_before = process.memory_info().rss / (1024 ** 2)
            start_search_time = time.time()
            if query_batch_size > 0:
                result_indices = []
                result_distances = []
                for start_idx in range(0, len(embeddings_query_mun), query_batch_size):
                    end_idx = min(start_idx + query_batch_size, len(embeddings_query_mun))
                    chunk_I, chunk_D = searcher.search_batched(
                        embeddings_query_mun[start_idx:end_idx],
                        **search_kwargs,
                    )
                    result_indices.append(chunk_I)
                    result_distances.append(chunk_D)
                I = np.vstack(result_indices)
                D = np.vstack(result_distances)
            else:
                I, D = searcher.search_batched(
                    embeddings_query_mun,
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

        avg_search_time = soma_tempo_busca / num_execucoes
        avg_mem_used_search = soma_memoria / num_execucoes

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
            "metodo": "scann",
            "modelo_embedding": str(model),
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
    append_resultados_por_municipio(df_resultados_municipio, "scann")

    total_index_time = df_resultados_municipio["index_time"].sum() if not df_resultados_municipio.empty else 0.0
    total_search_time = df_resultados_municipio["search_time"].sum() if not df_resultados_municipio.empty else 0.0
    avg_mem_used_search = df_resultados_municipio["avg_mem_used_search_MB"].mean() if not df_resultados_municipio.empty else 0.0
    mem_used_create_index = df_resultados_municipio["mem_used_indexation_MB"].sum() if not df_resultados_municipio.empty else 0.0

    print(f">>> [ScaNN] Salvando log detalhado global em {PATH_resultados_scann}", flush=True)
    df_tempos_busca_scann = pd.DataFrame(dict_)
    df_tempos_busca_scann["modelo_index"] = "scann"
    df_tempos_busca_scann["modelo_embedding"] = str(model)

    df_aux = pd.read_csv(PATH_resultados_scann)
    df_aux = pd.concat([df_aux, df_tempos_busca_scann], ignore_index=True)
    df_aux.to_csv(PATH_resultados_scann, index=False)

    print(">>> [ScaNN] Merge concluído. Salvando matched_scann.csv...", flush=True)
    df_lm_matched.to_csv(os.path.join(PATH_RESULTADOS, "matched_scann.csv"), index=False)

    if not os.path.exists(PATH_RESULTADOS_scann):
        os.makedirs(PATH_RESULTADOS_scann)

    print(
        f">>> [ScaNN] Salvando resultados individuais em {PATH_RESULTADOS_scann}",
        flush=True,
    )
    df_tempos_busca_scann.to_csv(
        os.path.join(PATH_RESULTADOS_scann, "csv_final_tempos_buscas.csv"),
        index=False,
    )

    results_data = {
        "metodo": ["scann"],
        "modelo_embedding": [str(model)],
        "index_time": [total_index_time],
        "search_time": [total_search_time],
        "total_time": [total_index_time + total_search_time],
        "num_rows_df1": [len(df1)],
        "num_rows_df2": [len(df2)],
        "k": [k],
        "mem_used_indexation_MB": [mem_used_create_index],
        "avg_mem_used_search_MB": [avg_mem_used_search],
        "matches": [count_setor_matches(df_lm_matched)],
    }

    results_df = pd.DataFrame(results_data)

    results_file = os.path.join(PATH_RESULTADOS, "resultados.csv")
    print(f">>> [ScaNN] Atualizando resumo final em {results_file}", flush=True)

    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(results_file, mode="w", header=True, index=False)

    print(">>> [ScaNN] Processamento do modelo concluído com sucesso.", flush=True)
    return True
