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

from linktransformer.cluster_fns import cluster
from linktransformer.global_chunking import (
    append_data_dir_candidates,
    get_global_base_chunk_size,
    get_global_query_batch_size,
    init_topk_buffers,
    iter_row_ranges,
    merge_topk_buffers,
    release_native_memory,
    should_use_chunked_global_search,
)
# from linktransformer.utils import serialize_columns, infer_embeddings, load_model, load_clf, cosine_similarity_corresponding_pairs, tokenize_data_for_inference, predict_rows_with_openai
from linktransformer.memory_utils import PeakMemoryMonitor
from linktransformer.utils import *
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from transformers import TrainingArguments, Trainer
from linktransformer.main_svs import VamanaIndexer
import time

def get_data_dir_candidates() -> List[str]:
    return append_data_dir_candidates(
        repo_root=os.path.abspath("."),
        env_data_dir=os.environ.get("LINKTRANSFORMER_DATA_DIR"),
    )


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


def load_flat_embeddings(
    safe_model: str,
    mmap_mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    embeddings_base_path = get_flat_embeddings_path("base", safe_model)
    embeddings_query_path = get_flat_embeddings_path("query", safe_model)
    return (
        np.load(embeddings_base_path, mmap_mode=mmap_mode),
        np.load(embeddings_query_path, mmap_mode=mmap_mode),
    )


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


def build_matches_global(
    df_base: DataFrame,
    df_query: DataFrame,
    I: np.ndarray,
    k_efetivo: int,
    suffixes: Tuple[str, str],
    scores: Optional[np.ndarray] = None,
) -> DataFrame:
    df_query_expanded = df_query.loc[
        np.repeat(df_query.index.values, k_efetivo)
    ].reset_index(drop=True)
    df_base_expanded = df_base.iloc[I.flatten()].reset_index(drop=True)

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


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def load_global_embeddings(safe_model: str, normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    embeddings_base, embeddings_query = load_flat_embeddings(safe_model)
    if normalize:
        embeddings_base = normalize_embeddings(embeddings_base)
        embeddings_query = normalize_embeddings(embeddings_query)
    return embeddings_base, embeddings_query


def ensure_zero_based_index(df: DataFrame) -> DataFrame:
    if isinstance(df.index, pd.RangeIndex) and df.index.start == 0 and df.index.step == 1:
        return df
    return df.reset_index(drop=True)


def run_chunked_global_search(
    safe_model: str,
    k: int,
    method_label: str,
    prefer_higher_scores: bool,
    normalize_base: bool,
    normalize_query: bool,
    build_index_fn,
    search_index_fn,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, Dict[str, List[Any]]]:
    embeddings_base, embeddings_query = load_flat_embeddings(safe_model, mmap_mode="r")
    num_rows_base = len(embeddings_base)
    num_rows_query = len(embeddings_query)
    k_efetivo = min(k, num_rows_base)
    num_execucoes = NUM_EXECUCOES_BUSCA
    base_chunk_size = get_global_base_chunk_size(num_rows_base)
    query_batch_size = get_global_query_batch_size(num_rows_query)
    base_ranges = list(iter_row_ranges(num_rows_base, base_chunk_size))
    query_ranges = list(iter_row_ranges(num_rows_query, query_batch_size))

    print(
        f">>> [{method_label}] Usando processamento global em chunks | "
        f"base_chunk_size={base_chunk_size} | query_batch_size={query_batch_size} | "
        f"chunks_base={len(base_ranges)} | lotes_query={len(query_ranges)}",
        flush=True,
    )

    exec_scores = []
    exec_indices = []
    for _ in range(num_execucoes):
        scores, indices = init_topk_buffers(
            num_rows=num_rows_query,
            k=k_efetivo,
            prefer_higher_scores=prefer_higher_scores,
        )
        exec_scores.append(scores)
        exec_indices.append(indices)

    total_index_time = 0.0
    total_search_times = [0.0 for _ in range(num_execucoes)]
    peak_index_memory_mb = 0.0
    peak_search_memory_mb = [0.0 for _ in range(num_execucoes)]

    for chunk_idx, (base_start, base_end) in enumerate(base_ranges, start=1):
        if chunk_idx == 1 or chunk_idx == len(base_ranges) or chunk_idx % 10 == 0:
            print(
                f">>> [{method_label}] Chunk base {chunk_idx}/{len(base_ranges)} | "
                f"linhas {base_start}-{base_end - 1}",
                flush=True,
            )

        base_chunk = np.ascontiguousarray(embeddings_base[base_start:base_end], dtype=np.float32)
        if normalize_base:
            base_chunk = normalize_embeddings(base_chunk)

        k_chunk = min(k_efetivo, len(base_chunk))
        start_index_time = time.time()
        with PeakMemoryMonitor() as memory_monitor:
            index = build_index_fn(base_chunk, k_chunk)
        total_index_time += time.time() - start_index_time
        peak_index_memory_mb = max(peak_index_memory_mb, memory_monitor.peak_delta_mb)

        for exec_idx in range(num_execucoes):
            for query_start, query_end in query_ranges:
                query_batch = np.ascontiguousarray(embeddings_query[query_start:query_end], dtype=np.float32)
                if normalize_query:
                    query_batch = normalize_embeddings(query_batch)

                start_search_time = time.time()
                with PeakMemoryMonitor() as memory_monitor:
                    search_output = search_index_fn(index, query_batch, k_chunk)
                wall_search_time = time.time() - start_search_time

                if len(search_output) == 2:
                    chunk_indices, chunk_scores = search_output
                    measured_search_time = wall_search_time
                else:
                    chunk_indices, chunk_scores, measured_search_time = search_output

                chunk_indices = np.asarray(chunk_indices, dtype=np.int64) + base_start
                chunk_scores = np.asarray(chunk_scores, dtype=np.float32)

                merged_scores, merged_indices = merge_topk_buffers(
                    current_scores=exec_scores[exec_idx][query_start:query_end],
                    current_indices=exec_indices[exec_idx][query_start:query_end],
                    candidate_scores=chunk_scores,
                    candidate_indices=chunk_indices,
                    k=k_efetivo,
                    prefer_higher_scores=prefer_higher_scores,
                )
                exec_scores[exec_idx][query_start:query_end] = merged_scores
                exec_indices[exec_idx][query_start:query_end] = merged_indices

                total_search_times[exec_idx] += measured_search_time
                peak_search_memory_mb[exec_idx] = max(
                    peak_search_memory_mb[exec_idx],
                    memory_monitor.peak_delta_mb,
                )

        del index
        del base_chunk
        release_native_memory()

    detailed_rows = {
        "execucao": list(range(1, num_execucoes + 1)),
        "tempo_busca": total_search_times,
        "memoria_usada_busca_MB": peak_search_memory_mb,
    }

    final_indices = exec_indices[-1]
    final_scores = exec_scores[-1]
    avg_search_time = sum(total_search_times) / num_execucoes
    avg_mem_used_search = sum(peak_search_memory_mb) / num_execucoes
    return (
        final_indices,
        final_scores,
        total_index_time,
        avg_search_time,
        peak_index_memory_mb,
        avg_mem_used_search,
        detailed_rows,
    )


def save_global_outputs(
    metodo: str,
    modelo_embedding: str,
    safe_model: str,
    df_base: DataFrame,
    df_query: DataFrame,
    k: int,
    index_time: float,
    avg_search_time: float,
    mem_used_create_index: float,
    avg_mem_used_search: float,
    detailed_rows: Dict[str, List[Any]],
    df_lm_matched: DataFrame,
    result_subdir: str,
    modelo_index: str,
    matched_filename: str,
) -> None:
    df_tempos = pd.DataFrame(detailed_rows)
    df_tempos["modelo_index"] = modelo_index
    df_tempos["modelo_embedding"] = modelo_embedding

    df_aux = pd.read_csv(PATH_RESULTADOS_GERAL)
    df_aux = pd.concat([df_aux, df_tempos], ignore_index=True)
    df_aux.to_csv(PATH_RESULTADOS_GERAL, index=False)

    path_resultados_metodo = os.path.join(PATH_RESULTADOS, result_subdir, safe_model)
    os.makedirs(path_resultados_metodo, exist_ok=True)
    df_tempos.to_csv(
        os.path.join(path_resultados_metodo, "csv_final_tempos_buscas.csv"),
        index=False,
    )
    df_lm_matched.to_csv(os.path.join(PATH_RESULTADOS, matched_filename), index=False)

    results_file = os.path.join(PATH_RESULTADOS, "resultados.csv")
    results_df = pd.DataFrame({
        "metodo": [metodo],
        "modelo_embedding": [modelo_embedding],
        "index_time": [index_time],
        "search_time": [avg_search_time],
        "total_time": [index_time + avg_search_time],
        "num_rows_df1": [len(df_base)],
        "num_rows_df2": [len(df_query)],
        "k": [k],
        "mem_used_indexation_MB": [mem_used_create_index],
        "avg_mem_used_search_MB": [avg_mem_used_search],
        "matches": [count_setor_matches(df_lm_matched)],
    })

    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results_df.to_csv(results_file, mode="w", header=True, index=False)


def merge_knn_global(k, df1, df2, suffixes, model) -> DataFrame:
    safe_model = safe_model_name(model)
    df1 = ensure_zero_based_index(df1)
    df2 = ensure_zero_based_index(df2)
    k_efetivo = min(k, len(df1))
    if should_use_chunked_global_search(len(df1)):
        def build_index_fn(base_chunk: np.ndarray, _k_chunk: int):
            index = faiss.IndexFlatIP(base_chunk.shape[1])
            index.add(base_chunk)
            return index

        def search_index_fn(index, query_batch: np.ndarray, k_chunk: int):
            return index.search(query_batch, k_chunk)

        I, D, index_time, avg_search_time, mem_used_create_index, avg_mem_used_search, detailed_rows = (
            run_chunked_global_search(
                safe_model=safe_model,
                k=k,
                method_label="FAISS",
                prefer_higher_scores=True,
                normalize_base=False,
                normalize_query=False,
                build_index_fn=build_index_fn,
                search_index_fn=search_index_fn,
            )
        )
    else:
        print(f">>> [FAISS] Carregando embeddings globais do modelo: {model}", flush=True)
        embeddings_base, embeddings_query = load_global_embeddings(safe_model)
        print(
            f">>> [FAISS] Embeddings carregados | base={embeddings_base.shape} | query={embeddings_query.shape}",
            flush=True,
        )

        start_index_time = time.time()
        with PeakMemoryMonitor() as memory_monitor:
            index = faiss.IndexFlatIP(embeddings_base.shape[1])
            index.add(embeddings_base)
        index_time = time.time() - start_index_time
        mem_used_create_index = memory_monitor.peak_delta_mb

        detailed_rows = {"execucao": [], "tempo_busca": [], "memoria_usada_busca_MB": []}
        soma_tempo_busca = 0.0
        soma_memoria_busca = 0.0
        D = None
        I = None
        for i in range(NUM_EXECUCOES_BUSCA):
            start_search_time = time.time()
            with PeakMemoryMonitor() as memory_monitor:
                D, I = index.search(embeddings_query, k_efetivo)
            search_time = time.time() - start_search_time
            mem_used_search = memory_monitor.peak_delta_mb
            soma_tempo_busca += search_time
            soma_memoria_busca += mem_used_search
            detailed_rows["execucao"].append(i + 1)
            detailed_rows["tempo_busca"].append(search_time)
            detailed_rows["memoria_usada_busca_MB"].append(mem_used_search)

        avg_search_time = soma_tempo_busca / NUM_EXECUCOES_BUSCA
        avg_mem_used_search = soma_memoria_busca / NUM_EXECUCOES_BUSCA

    df_lm_matched = build_matches_global(df1, df2, I, k_efetivo, suffixes, D)
    save_global_outputs(
        "baseline",
        model,
        safe_model,
        df1,
        df2,
        k,
        index_time,
        avg_search_time,
        mem_used_create_index,
        avg_mem_used_search,
        detailed_rows,
        df_lm_matched,
        "baseline",
        "faiss_baseline",
        "matched_faiss_global.csv",
    )
    return df_lm_matched


def merge_knn2_global(k, df1, df2, suffixes, model) -> DataFrame:
    safe_model = safe_model_name(model)
    df1 = ensure_zero_based_index(df1)
    df2 = ensure_zero_based_index(df2)
    k_efetivo = min(k, len(df1))
    if should_use_chunked_global_search(len(df1)):
        def build_index_fn(base_chunk: np.ndarray, _k_chunk: int):
            return VamanaIndexer().build(
                base_embeddings=base_chunk,
                reduced_dims=128,
                graph_max_degree=64,
                window_size=128,
                distance="L2",
                num_threads=4,
                primary_kind="lvq4",
                secondary_kind="lvq8",
            )

        def search_index_fn(index, query_batch: np.ndarray, k_chunk: int):
            return index.search(query_batch, k_chunk)

        I, D, index_time, avg_search_time, mem_used_create_index, avg_mem_used_search, detailed_rows = (
            run_chunked_global_search(
                safe_model=safe_model,
                k=k,
                method_label="SVS",
                prefer_higher_scores=False,
                normalize_base=False,
                normalize_query=False,
                build_index_fn=build_index_fn,
                search_index_fn=search_index_fn,
            )
        )
    else:
        print(f">>> [SVS] Carregando embeddings globais do modelo: {model}", flush=True)
        embeddings_base, embeddings_query = load_global_embeddings(safe_model)

        class_svs = VamanaIndexer()
        start_index_time = time.time()
        with PeakMemoryMonitor() as memory_monitor:
            index = class_svs.build(
                base_embeddings=embeddings_base,
                reduced_dims=128,
                graph_max_degree=64,
                window_size=128,
                distance="L2",
                num_threads=4,
                primary_kind="lvq4",
                secondary_kind="lvq8",
            )
        index_time = time.time() - start_index_time
        mem_used_create_index = memory_monitor.peak_delta_mb

        detailed_rows = {"execucao": [], "tempo_busca": [], "memoria_usada_busca_MB": []}
        soma_tempo_busca = 0.0
        soma_memoria_busca = 0.0
        I = None
        D = None
        for i in range(NUM_EXECUCOES_BUSCA):
            start_search_time = time.time()
            with PeakMemoryMonitor() as memory_monitor:
                I, D = index.search(embeddings_query, k_efetivo)
            search_time = time.time() - start_search_time
            mem_used_search = memory_monitor.peak_delta_mb
            soma_tempo_busca += search_time
            soma_memoria_busca += mem_used_search
            detailed_rows["execucao"].append(i + 1)
            detailed_rows["tempo_busca"].append(search_time)
            detailed_rows["memoria_usada_busca_MB"].append(mem_used_search)

        avg_search_time = soma_tempo_busca / NUM_EXECUCOES_BUSCA
        avg_mem_used_search = soma_memoria_busca / NUM_EXECUCOES_BUSCA

    df_lm_matched = build_matches_global(df1, df2, I, k_efetivo, suffixes, D)
    save_global_outputs(
        "svs",
        model,
        safe_model,
        df1,
        df2,
        k,
        index_time,
        avg_search_time,
        mem_used_create_index,
        avg_mem_used_search,
        detailed_rows,
        df_lm_matched,
        "svs",
        "svs",
        "matched_svs_global.csv",
    )
    return df_lm_matched


def merge_knn_hnsw_julia_global(k, df1, df2, suffixes, model) -> DataFrame:
    from julia import Main

    safe_model = safe_model_name(model)
    Main.include("hnsw_julia/hnsw_wrapper.jl")
    df1 = ensure_zero_based_index(df1)
    df2 = ensure_zero_based_index(df2)
    k_efetivo = min(k, len(df1))
    if should_use_chunked_global_search(len(df1)):
        def build_index_fn(base_chunk: np.ndarray, _k_chunk: int):
            return Main.build_hnsw(base_chunk)

        def search_index_fn(index, query_batch: np.ndarray, k_chunk: int):
            indices, distances, tempo_busca = Main.search_hnsw(index, query_batch, K=k_chunk)
            return np.asarray(indices) - 1, np.asarray(distances), tempo_busca

        I, D, index_time, avg_search_time, mem_used_create_index, avg_mem_used_search, detailed_rows = (
            run_chunked_global_search(
                safe_model=safe_model,
                k=k,
                method_label="HNSW Julia",
                prefer_higher_scores=False,
                normalize_base=True,
                normalize_query=True,
                build_index_fn=build_index_fn,
                search_index_fn=search_index_fn,
            )
        )
    else:
        print(f">>> [HNSW Julia] Carregando embeddings globais do modelo: {model}", flush=True)
        embeddings_base, embeddings_query = load_global_embeddings(safe_model, normalize=True)

        start_index_time = time.time()
        with PeakMemoryMonitor() as memory_monitor:
            hnsw = Main.build_hnsw(embeddings_base)
        index_time = time.time() - start_index_time
        mem_used_create_index = memory_monitor.peak_delta_mb

        detailed_rows = {"execucao": [], "tempo_busca": [], "memoria_usada_busca_MB": []}
        soma_tempo_busca = 0.0
        soma_memoria_busca = 0.0
        I = None
        D = None
        for i in range(NUM_EXECUCOES_BUSCA):
            with PeakMemoryMonitor() as memory_monitor:
                I, D, tempo_busca = Main.search_hnsw(hnsw, embeddings_query, K=k_efetivo)
            mem_used_search = memory_monitor.peak_delta_mb
            soma_tempo_busca += tempo_busca
            soma_memoria_busca += mem_used_search
            detailed_rows["execucao"].append(i + 1)
            detailed_rows["tempo_busca"].append(tempo_busca)
            detailed_rows["memoria_usada_busca_MB"].append(mem_used_search)

        I = np.asarray(I) - 1
        D = np.asarray(D)
        avg_search_time = soma_tempo_busca / NUM_EXECUCOES_BUSCA
        avg_mem_used_search = soma_memoria_busca / NUM_EXECUCOES_BUSCA

    df_lm_matched = build_matches_global(df1, df2, I, k_efetivo, suffixes, D)
    save_global_outputs(
        "hnsw_julia",
        model,
        safe_model,
        df1,
        df2,
        k,
        index_time,
        avg_search_time,
        mem_used_create_index,
        avg_mem_used_search,
        detailed_rows,
        df_lm_matched,
        "hnsw_julia",
        "hnsw_julia",
        "matched_hnsw_julia_global.csv",
    )
    return df_lm_matched


def merge_knn_nmslib_global(k, df1, df2, suffixes, model) -> DataFrame:
    import nmslib

    safe_model = safe_model_name(model)
    df1 = ensure_zero_based_index(df1)
    df2 = ensure_zero_based_index(df2)
    k_efetivo = min(k, len(df1))
    if should_use_chunked_global_search(len(df1)):
        def build_index_fn(base_chunk: np.ndarray, _k_chunk: int):
            index = nmslib.init(space="cosinesimil", method="hnsw")
            index.addDataPointBatch(base_chunk)
            index.createIndex({"M": 48, "efConstruction": 600}, print_progress=False)
            index.setQueryTimeParams({"efSearch": 50})
            return index

        def search_index_fn(index, query_batch: np.ndarray, k_chunk: int):
            res = index.knnQueryBatch(query_batch, k=k_chunk)
            neighbors, distances = zip(*res)
            return np.vstack(neighbors), 1.0 - np.vstack(distances)

        I, score_sim, index_time, avg_search_time, mem_used_create_index, avg_mem_used_search, detailed_rows = (
            run_chunked_global_search(
                safe_model=safe_model,
                k=k,
                method_label="NMSLIB",
                prefer_higher_scores=True,
                normalize_base=True,
                normalize_query=True,
                build_index_fn=build_index_fn,
                search_index_fn=search_index_fn,
            )
        )
    else:
        print(f">>> [NMSLIB] Carregando embeddings globais do modelo: {model}", flush=True)
        embeddings_base, embeddings_query = load_global_embeddings(safe_model, normalize=True)

        start_index_time = time.time()
        with PeakMemoryMonitor() as memory_monitor:
            index = nmslib.init(space="cosinesimil", method="hnsw")
            index.addDataPointBatch(embeddings_base)
            index.createIndex({"M": 48, "efConstruction": 600}, print_progress=False)
            index.setQueryTimeParams({"efSearch": 50})
        index_time = time.time() - start_index_time
        mem_used_create_index = memory_monitor.peak_delta_mb

        detailed_rows = {"execucao": [], "tempo_busca": [], "memoria_usada_busca_MB": []}
        soma_tempo_busca = 0.0
        soma_memoria_busca = 0.0
        neighbors = None
        distances = None
        for i in range(NUM_EXECUCOES_BUSCA):
            start_search_time = time.time()
            with PeakMemoryMonitor() as memory_monitor:
                res = index.knnQueryBatch(embeddings_query, k=k_efetivo)
            search_time = time.time() - start_search_time
            mem_used_search = memory_monitor.peak_delta_mb
            soma_tempo_busca += search_time
            soma_memoria_busca += mem_used_search
            neighbors, distances = zip(*res)
            detailed_rows["execucao"].append(i + 1)
            detailed_rows["tempo_busca"].append(search_time)
            detailed_rows["memoria_usada_busca_MB"].append(mem_used_search)

        I = np.vstack(neighbors)
        score_sim = 1.0 - np.vstack(distances)
        avg_search_time = soma_tempo_busca / NUM_EXECUCOES_BUSCA
        avg_mem_used_search = soma_memoria_busca / NUM_EXECUCOES_BUSCA

    df_lm_matched = build_matches_global(df1, df2, I, k_efetivo, suffixes, score_sim)
    save_global_outputs(
        "NMSLIB",
        model,
        safe_model,
        df1,
        df2,
        k,
        index_time,
        avg_search_time,
        mem_used_create_index,
        avg_mem_used_search,
        detailed_rows,
        df_lm_matched,
        "NMSLIB",
        "nmslib_hnsw",
        "matched_nmslib_global.csv",
    )
    return df_lm_matched

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
        with PeakMemoryMonitor() as memory_monitor:
            index = faiss.IndexFlatIP(embeddings_base_mun.shape[1])
            index.add(embeddings_base_mun)
        mem_used_create_index = memory_monitor.peak_delta_mb
        index_time = time.time() - start_index_time

        soma_tempo_busca = 0.0
        soma_memoria_busca = 0.0
        D = None
        I = None
        for i in range(num_execucoes):
            start_search_time = time.time()
            with PeakMemoryMonitor() as memory_monitor:
                D, I = index.search(embeddings_query_mun, k_efetivo)
            mem_used_search = memory_monitor.peak_delta_mb
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
        with PeakMemoryMonitor() as memory_monitor:
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
        mem_used_create_index = memory_monitor.peak_delta_mb
        index_time = time.time() - start_index_time

        soma_tempo_busca = 0.0
        soma_memoria_busca = 0.0
        I = None
        D = None
        for i in range(num_execucoes):
            start_search_time = time.time()
            with PeakMemoryMonitor() as memory_monitor:
                I, D = index.search(embeddings_query_mun, k_efetivo)
            mem_used_search = memory_monitor.peak_delta_mb
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
        with PeakMemoryMonitor() as memory_monitor:
            hnsw = Main.build_hnsw(embeddings_base_mun)
        mem_used_create_index = memory_monitor.peak_delta_mb
        index_time = time.time() - start_index_time

        soma_tempo_busca = 0.0
        soma_memoria_busca = 0.0
        I = None
        D = None
        for i in range(num_execucoes):
            with PeakMemoryMonitor() as memory_monitor:
                I, D, tempo_busca = Main.search_hnsw(hnsw, embeddings_query_mun, K=k_efetivo)
            mem_used_search = memory_monitor.peak_delta_mb
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
        with PeakMemoryMonitor() as memory_monitor:
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
        mem_used_create_index = memory_monitor.peak_delta_mb
        index_time = time.time() - start_index_time

        soma_tempo_busca = 0.0
        soma_memoria_busca = 0.0
        neighbors = None
        distances = None
        for i in range(num_execucoes):
            start_search_time = time.time()
            with PeakMemoryMonitor() as memory_monitor:
                res = index.knnQueryBatch(embeddings_query_mun, k=k_efetivo)
            mem_used_search = memory_monitor.peak_delta_mb
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
