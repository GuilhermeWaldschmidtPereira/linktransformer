#!/usr/bin/env python3
"""
Script wrapper para gerar embeddings particionados.

Lê o CSV da base em partições de 100k linhas, gera embeddings para cada partição,
e salva os resultados em pastas separadas por modelo.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Adiciona o diretório do projeto ao path
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, THIS_DIR)

from run_embeddings import (
    ColumnSpec,
    build_embeddings,
    current_memory_mb,
    elapsed_summary,
    normalize_column_spec,
    resolve_columns,
    resolve_csv_separator,
    resolve_single_side_columns,
    safe_model_name,
)


DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/multilingual-e5-large",
    "neuralmind/bert-large-portuguese-cased",
]

PARTITION_SIZE = 100_000


def parse_args() -> argparse.Namespace:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(
        description="Gera embeddings particionados (a cada 100k linhas) a partir de CSVs.",
    )
    parser.add_argument("--base-path", default=os.path.join(repo_root, "data", "base.csv"))
    parser.add_argument("--query-path", default=os.path.join(repo_root, "data", "query.csv"))
    parser.add_argument("--output-dir", default=os.path.join(repo_root, "data"))
    parser.add_argument("--model", dest="models", action="append")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--partition-size", type=int, default=PARTITION_SIZE)
    parser.add_argument(
        "--mode",
        choices=["both", "base", "query"],
        default="both",
        help="Controla se gera embeddings da base, da query, ou de ambos.",
    )
    parser.add_argument("--on", nargs="+")
    parser.add_argument("--left-on", nargs="+")
    parser.add_argument("--right-on", nargs="+")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument(
        "--csv-encoding",
        default=None,
        help="Encoding do CSV. Se omitido, tenta utf-8, utf-8-sig, latin1 e cp1252.",
    )
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument(
        "--merge-partitions",
        action="store_true",
        help="Se ativado, mescla as partições em um único arquivo ao final.",
    )
    return parser.parse_args()


def get_partition_output_dir(base_output_dir: str, model_name: str) -> str:
    """Retorna o diretório onde as partições serão salvas para um modelo."""
    safe_model = safe_model_name(model_name)
    return os.path.join(base_output_dir, "embeddings_partitions", safe_model)


def get_partition_path(partition_dir: str, partition_idx: int, side: str) -> str:
    """Retorna o caminho do arquivo de partição (base ou query)."""
    return os.path.join(partition_dir, f"partition_{partition_idx:06d}_{side}.npy")


def get_merged_output_path(base_output_dir: str, model_name: str, side: str) -> str:
    """Retorna o caminho do arquivo de embeddings mesclado (compatível com código antigo)."""
    safe_model = safe_model_name(model_name)
    return os.path.join(base_output_dir, f"embeddings_{side}_{safe_model}.npy")


def infer_file_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(f"Formato de arquivo não suportado para {path}. Use .csv, .parquet ou .pq.")


def load_dataframe_columns(path: str, csv_encoding: Optional[str] = None) -> pd.DataFrame:
    """Carrega apenas o cabeçalho/metadata para resolver colunas sem materializar tudo em memória."""
    file_format = infer_file_format(path)
    if file_format == "csv":
        sep = resolve_csv_separator(path)
        if csv_encoding:
            return pd.read_csv(path, encoding=csv_encoding, sep=sep, nrows=0)

        candidate_encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
        last_error: Optional[UnicodeDecodeError] = None
        for candidate in candidate_encodings:
            try:
                return pd.read_csv(path, encoding=candidate, sep=sep, nrows=0)
            except UnicodeDecodeError as exc:
                last_error = exc

        raise UnicodeDecodeError(
            last_error.encoding if last_error else "utf-8",
            last_error.object if last_error else b"",
            last_error.start if last_error else 0,
            last_error.end if last_error else 1,
            f"Nao foi possivel ler o cabeçalho de {path} com os encodings testados.",
        )

    parquet_file = pq.ParquetFile(path)
    return pd.DataFrame(columns=parquet_file.schema.names)


def count_rows(path: str) -> int:
    """Conta linhas sem carregar o dataset inteiro em memória."""
    file_format = infer_file_format(path)
    if file_format == "csv":
        with open(path, "rb") as handle:
            line_count = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))
        return max(line_count - 1, 0)

    parquet_file = pq.ParquetFile(path)
    return parquet_file.metadata.num_rows


def iter_dataframe_partitions(
    path: str,
    partition_size: int,
    csv_encoding: Optional[str] = None,
) -> Iterator[pd.DataFrame]:
    """Itera partições do dataset sem carregar tudo na memória de uma vez."""
    file_format = infer_file_format(path)

    if file_format == "csv":
        sep = resolve_csv_separator(path)
        if csv_encoding:
            yield from pd.read_csv(path, encoding=csv_encoding, sep=sep, chunksize=partition_size)
            return

        candidate_encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
        last_error: Optional[UnicodeDecodeError] = None
        for candidate in candidate_encodings:
            try:
                reader = pd.read_csv(path, encoding=candidate, sep=sep, chunksize=partition_size)
                iterator = iter(reader)
                first_chunk = next(iterator)
            except StopIteration:
                return
            except UnicodeDecodeError as exc:
                last_error = exc
                continue

            yield first_chunk
            yield from iterator
            return

        raise UnicodeDecodeError(
            last_error.encoding if last_error else "utf-8",
            last_error.object if last_error else b"",
            last_error.start if last_error else 0,
            last_error.end if last_error else 1,
            f"Nao foi possivel iterar {path} com os encodings testados.",
        )

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=partition_size):
        yield batch.to_pandas()


def init_merged_memmap(output_path: str, total_rows: int, embedding_dim: int) -> np.memmap:
    """Cria o arquivo final no disco sem concatenar tudo na RAM."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, embedding_dim),
    )


def write_to_merged_memmap(
    merged_memmap: Optional[np.memmap],
    merged_output_path: str,
    total_rows: int,
    offset: int,
    embeddings_partition: np.ndarray,
) -> Tuple[np.memmap, int]:
    """Escreve uma partição no arquivo final mesclado em disco."""
    embeddings_partition = np.asarray(embeddings_partition, dtype=np.float32)
    if merged_memmap is None:
        merged_memmap = init_merged_memmap(
            merged_output_path,
            total_rows=total_rows,
            embedding_dim=embeddings_partition.shape[1],
        )

    rows = embeddings_partition.shape[0]
    merged_memmap[offset:offset + rows] = embeddings_partition
    return merged_memmap, offset + rows


def process_base_partitioned(
    args: argparse.Namespace,
    models: List[str],
    manifest: List[dict],
) -> None:
    """Processa apenas a base em partições."""
    print(f">>> Preparando base para processamento particionado...", flush=True)
    base_columns = resolve_single_side_columns(args.left_on, normalize_column_spec(args.on), "left")

    num_rows = count_rows(args.base_path)
    num_partitions = (num_rows + args.partition_size - 1) // args.partition_size
    print(f">>> Base com {num_rows} linhas | {num_partitions} partições de {args.partition_size} linhas", flush=True)

    for model_name in models:
        print(f"\n>>> Processando modelo: {model_name}", flush=True)
        model_start_time = time.perf_counter()
        model_memory_start_mb = current_memory_mb()

        partition_dir = get_partition_output_dir(args.output_dir, model_name)
        os.makedirs(partition_dir, exist_ok=True)

        merged_memmap = None
        merged_offset = 0
        merged_path = get_merged_output_path(args.output_dir, model_name, "base")

        for partition_idx, df_partition in enumerate(
            iter_dataframe_partitions(
                args.base_path,
                partition_size=args.partition_size,
                csv_encoding=args.csv_encoding,
            )
        ):
            start_idx = partition_idx * args.partition_size
            end_idx = min(start_idx + args.partition_size, num_rows)

            print(
                f"  Partição {partition_idx + 1}/{num_partitions} | "
                f"linhas {start_idx}-{end_idx-1} ({len(df_partition)} registros)",
                flush=True,
            )

            partition_start_time = time.perf_counter()
            embeddings_partition = build_embeddings(
                df=df_partition,
                columns=base_columns,
                model_name=model_name,
                batch_size=args.batch_size,
                openai_key=args.openai_key,
                label=f"base_partition_{partition_idx}",
            )

            partition_path = get_partition_path(partition_dir, partition_idx, "base")
            os.makedirs(os.path.dirname(partition_path), exist_ok=True)
            np.save(partition_path, embeddings_partition.astype(np.float32))

            if args.merge_partitions:
                merged_memmap, merged_offset = write_to_merged_memmap(
                    merged_memmap,
                    merged_output_path=merged_path,
                    total_rows=num_rows,
                    offset=merged_offset,
                    embeddings_partition=embeddings_partition,
                )
                merged_memmap.flush()

            partition_elapsed = time.perf_counter() - partition_start_time
            print(f"    Salvo: {partition_path} | {elapsed_summary(partition_elapsed)}", flush=True)

            del embeddings_partition
            del df_partition
            gc.collect()

        if args.merge_partitions and merged_memmap is not None:
            del merged_memmap
            print(f"  Embeddings mesclados salvos em: {merged_path}", flush=True)

        elapsed_seconds = time.perf_counter() - model_start_time
        model_memory_end_mb = current_memory_mb()
        model_memory_peak_mb = max(model_memory_start_mb, model_memory_end_mb)

        print(f"Tempo total do modelo {model_name}: {elapsed_summary(elapsed_seconds)}", flush=True)
        print(
            f"Memória total do modelo {model_name}: "
            f"início {model_memory_start_mb:.2f} MB | fim {model_memory_end_mb:.2f} MB | "
            f"peak aprox. {model_memory_peak_mb:.2f} MB | delta {model_memory_end_mb - model_memory_start_mb:.2f} MB",
            flush=True,
        )

        manifest_entry = {
            "model": model_name,
            "safe_model_name": safe_model_name(model_name),
            "mode": "base",
            "partition_size": args.partition_size,
            "num_partitions": num_partitions,
            "total_rows": num_rows,
            "partition_dir": partition_dir,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "memory_start_mb": model_memory_start_mb,
            "memory_end_mb": model_memory_end_mb,
            "memory_peak_mb": model_memory_peak_mb,
            "memory_delta_mb": model_memory_end_mb - model_memory_start_mb,
        }

        if args.merge_partitions:
            manifest_entry["merged_path"] = get_merged_output_path(args.output_dir, model_name, "base")

        manifest.append(manifest_entry)


def process_both_partitioned(
    args: argparse.Namespace,
    models: List[str],
    manifest: List[dict],
) -> None:
    """Processa base e query em partições."""
    print(f">>> Preparando base e query para processamento particionado...", flush=True)
    df_base_columns = load_dataframe_columns(args.base_path, csv_encoding=args.csv_encoding)
    df_query_columns = load_dataframe_columns(args.query_path, csv_encoding=args.csv_encoding)
    left_on, right_on = resolve_columns(args, df_base_columns, df_query_columns)

    num_rows_base = count_rows(args.base_path)
    num_partitions_base = (num_rows_base + args.partition_size - 1) // args.partition_size
    num_rows_query = count_rows(args.query_path)
    num_partitions_query = (num_rows_query + args.partition_size - 1) // args.partition_size

    print(
        f">>> Base com {num_rows_base} linhas | {num_partitions_base} partições de {args.partition_size} linhas",
        flush=True,
    )
    print(
        f">>> Query com {num_rows_query} linhas | {num_partitions_query} partições de {args.partition_size} linhas",
        flush=True,
    )

    for model_name in models:
        print(f"\n>>> Processando modelo: {model_name}", flush=True)
        model_start_time = time.perf_counter()
        model_memory_start_mb = current_memory_mb()

        partition_dir = get_partition_output_dir(args.output_dir, model_name)
        os.makedirs(partition_dir, exist_ok=True)

        merged_memmap_base = None
        merged_memmap_query = None
        merged_offset_base = 0
        merged_offset_query = 0
        merged_path_base = get_merged_output_path(args.output_dir, model_name, "base")
        merged_path_query = get_merged_output_path(args.output_dir, model_name, "query")

        # Processa base
        print(f"  Processando base...", flush=True)
        for partition_idx, df_partition in enumerate(
            iter_dataframe_partitions(
                args.base_path,
                partition_size=args.partition_size,
                csv_encoding=args.csv_encoding,
            )
        ):
            start_idx = partition_idx * args.partition_size
            end_idx = min(start_idx + args.partition_size, num_rows_base)

            print(
                f"    Partição base {partition_idx + 1}/{num_partitions_base} | "
                f"linhas {start_idx}-{end_idx-1} ({len(df_partition)} registros)",
                flush=True,
            )

            partition_start_time = time.perf_counter()
            embeddings_partition = build_embeddings(
                df=df_partition,
                columns=left_on,
                model_name=model_name,
                batch_size=args.batch_size,
                openai_key=args.openai_key,
                label=f"base_partition_{partition_idx}",
            )

            partition_path = get_partition_path(partition_dir, partition_idx, "base")
            os.makedirs(os.path.dirname(partition_path), exist_ok=True)
            np.save(partition_path, embeddings_partition.astype(np.float32))

            if args.merge_partitions:
                merged_memmap_base, merged_offset_base = write_to_merged_memmap(
                    merged_memmap_base,
                    merged_output_path=merged_path_base,
                    total_rows=num_rows_base,
                    offset=merged_offset_base,
                    embeddings_partition=embeddings_partition,
                )
                merged_memmap_base.flush()

            partition_elapsed = time.perf_counter() - partition_start_time
            print(f"      Salvo: {partition_path} | {elapsed_summary(partition_elapsed)}", flush=True)

            del embeddings_partition
            del df_partition
            gc.collect()

        # Processa query
        print(f"  Processando query...", flush=True)
        for partition_idx, df_partition in enumerate(
            iter_dataframe_partitions(
                args.query_path,
                partition_size=args.partition_size,
                csv_encoding=args.csv_encoding,
            )
        ):
            start_idx = partition_idx * args.partition_size
            end_idx = min(start_idx + args.partition_size, num_rows_query)

            print(
                f"    Partição query {partition_idx + 1}/{num_partitions_query} | "
                f"linhas {start_idx}-{end_idx-1} ({len(df_partition)} registros)",
                flush=True,
            )

            partition_start_time = time.perf_counter()
            embeddings_partition = build_embeddings(
                df=df_partition,
                columns=right_on,
                model_name=model_name,
                batch_size=args.batch_size,
                openai_key=args.openai_key,
                label=f"query_partition_{partition_idx}",
            )

            partition_path = get_partition_path(partition_dir, partition_idx, "query")
            os.makedirs(os.path.dirname(partition_path), exist_ok=True)
            np.save(partition_path, embeddings_partition.astype(np.float32))

            if args.merge_partitions:
                merged_memmap_query, merged_offset_query = write_to_merged_memmap(
                    merged_memmap_query,
                    merged_output_path=merged_path_query,
                    total_rows=num_rows_query,
                    offset=merged_offset_query,
                    embeddings_partition=embeddings_partition,
                )
                merged_memmap_query.flush()

            partition_elapsed = time.perf_counter() - partition_start_time
            print(f"      Salvo: {partition_path} | {elapsed_summary(partition_elapsed)}", flush=True)

            del embeddings_partition
            del df_partition
            gc.collect()

        if args.merge_partitions and merged_memmap_base is not None:
            del merged_memmap_base
            print(f"  Embeddings da base mesclados salvos em: {merged_path_base}", flush=True)

        if args.merge_partitions and merged_memmap_query is not None:
            del merged_memmap_query
            print(f"  Embeddings da query mesclados salvos em: {merged_path_query}", flush=True)

        elapsed_seconds = time.perf_counter() - model_start_time
        model_memory_end_mb = current_memory_mb()
        model_memory_peak_mb = max(model_memory_start_mb, model_memory_end_mb)

        print(f"Tempo total do modelo {model_name}: {elapsed_summary(elapsed_seconds)}", flush=True)
        print(
            f"Memória total do modelo {model_name}: "
            f"início {model_memory_start_mb:.2f} MB | fim {model_memory_end_mb:.2f} MB | "
            f"peak aprox. {model_memory_peak_mb:.2f} MB | delta {model_memory_end_mb - model_memory_start_mb:.2f} MB",
            flush=True,
        )

        manifest_entry = {
            "model": model_name,
            "safe_model_name": safe_model_name(model_name),
            "mode": "both",
            "partition_size": args.partition_size,
            "num_partitions_base": num_partitions_base,
            "num_partitions_query": num_partitions_query,
            "total_rows_base": num_rows_base,
            "total_rows_query": num_rows_query,
            "partition_dir": partition_dir,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "memory_start_mb": model_memory_start_mb,
            "memory_end_mb": model_memory_end_mb,
            "memory_peak_mb": model_memory_peak_mb,
            "memory_delta_mb": model_memory_end_mb - model_memory_start_mb,
        }

        if args.merge_partitions:
            manifest_entry["merged_path_base"] = get_merged_output_path(args.output_dir, model_name, "base")
            manifest_entry["merged_path_query"] = get_merged_output_path(args.output_dir, model_name, "query")

        manifest.append(manifest_entry)


def main() -> None:
    args = parse_args()
    models = args.models or DEFAULT_MODELS

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = []

    print(f">>> Partições de {args.partition_size:,} linhas", flush=True)
    print(f">>> Output dir: {args.output_dir}", flush=True)
    print(f">>> Modelos: {', '.join(models)}", flush=True)
    print(f">>> Merge partitions: {args.merge_partitions}", flush=True)

    if args.mode == "both":
        process_both_partitioned(args, models, manifest)
    elif args.mode == "base":
        process_base_partitioned(args, models, manifest)
    else:
        raise NotImplementedError("Modo 'query' ainda não implementado para particionamento")

    if args.manifest_path:
        manifest_dir = os.path.dirname(args.manifest_path)
        if manifest_dir:
            os.makedirs(manifest_dir, exist_ok=True)
        with open(args.manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)
        print(f">>> Manifest salvo em: {args.manifest_path}", flush=True)

    print(f"\n>>> Processamento concluído!", flush=True)


if __name__ == "__main__":
    main()
