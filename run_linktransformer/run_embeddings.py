#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import openai
import pandas as pd
import psutil
import transformers
from sentence_transformers import SentenceTransformer


DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/multilingual-e5-large",
    "neuralmind/bert-large-portuguese-cased",
]

ColumnSpec = Union[str, List[str]]
EmbeddingMode = str


def parse_args() -> argparse.Namespace:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(
        description="Gera embeddings a partir de dois CSVs ou Parquets sem instalar o pacote inteiro do linktransformer.",
    )
    parser.add_argument("--base-path", default=os.path.join(repo_root, "data", "base.csv"))
    parser.add_argument("--query-path", default=os.path.join(repo_root, "data", "query.csv"))
    parser.add_argument("--output-dir", default=os.path.join(repo_root, "data"))
    parser.add_argument("--model", dest="models", action="append")
    parser.add_argument("--batch-size", type=int, default=128)
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
    return parser.parse_args()


def safe_model_name(model_name: str) -> str:
    sanitized = model_name.replace(os.sep, "_")
    if os.path.altsep:
        sanitized = sanitized.replace(os.path.altsep, "_")
    return sanitized


def load_csv_with_fallback(path: str, encoding: Optional[str] = None) -> pd.DataFrame:
    if encoding:
        print(f"Lendo CSV com encoding explícito: {encoding}")
        return pd.read_csv(path, encoding=encoding)

    candidate_encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_error: Optional[UnicodeDecodeError] = None

    for candidate in candidate_encodings:
        try:
            print(f"Tentando ler CSV com encoding: {candidate}")
            return pd.read_csv(path, encoding=candidate)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        last_error.encoding if last_error else "utf-8",
        last_error.object if last_error else b"",
        last_error.start if last_error else 0,
        last_error.end if last_error else 1,
        (
            f"Nao foi possivel ler {path} com os encodings testados: "
            f"{', '.join(candidate_encodings)}. Use --csv-encoding para informar um encoding valido."
        ),
    )


def load_dataframe(path: str, csv_encoding: Optional[str] = None) -> pd.DataFrame:
    suffix = os.path.splitext(path)[1].lower()

    if suffix == ".csv":
        return load_csv_with_fallback(path, encoding=csv_encoding)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        f"Formato de arquivo não suportado para {path}. Use .csv, .parquet ou .pq."
    )


def require_input_path(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Não encontrei {label}: {path}")


def ordered_common_columns(df_base: pd.DataFrame, df_query: pd.DataFrame) -> List[str]:
    return [column for column in df_base.columns if column in df_query.columns]


def normalize_column_spec(columns: Optional[List[str]]) -> Optional[ColumnSpec]:
    if columns is None:
        return None
    if len(columns) == 1:
        return columns[0]
    return columns


def resolve_columns(
    args: argparse.Namespace,
    df_base: pd.DataFrame,
    df_query: pd.DataFrame,
) -> Tuple[ColumnSpec, ColumnSpec]:
    shared_columns = normalize_column_spec(args.on)
    left_on = normalize_column_spec(args.left_on)
    right_on = normalize_column_spec(args.right_on)

    if shared_columns is None and left_on is None and right_on is None:
        shared_columns = ordered_common_columns(df_base, df_query)
        if not shared_columns:
            raise ValueError(
                "Nenhuma coluna em comum encontrada. Use --on ou informe --left-on e --right-on."
            )
        print(f"Colunas em comum detectadas para matching: {shared_columns}")

    if left_on is None:
        left_on = shared_columns
    if right_on is None:
        right_on = shared_columns

    if left_on is None or right_on is None:
        raise ValueError("Não foi possível resolver as colunas de entrada para embeddings.")

    return left_on, right_on


def resolve_single_side_columns(
    side_columns: Optional[List[str]],
    fallback_columns: Optional[ColumnSpec],
    label: str,
) -> ColumnSpec:
    resolved_columns = normalize_column_spec(side_columns)
    if resolved_columns is not None:
        return resolved_columns
    if fallback_columns is not None:
        return fallback_columns
    raise ValueError(
        f"Não foi possível resolver as colunas de entrada para {label}. "
        f"Use --on ou informe --{label}-on."
    )


def resolve_sep_token(model_name: str) -> str:
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        return tokenizer.sep_token or "</s>"
    except Exception:
        return "</s>"


def serialize_columns(df: pd.DataFrame, columns: Sequence[str], sep_token: str) -> List[str]:
    return df[list(columns)].apply(lambda row: sep_token.join(row.astype(str)), axis=1).tolist()


def serialize_embedding_input(df: pd.DataFrame, columns: ColumnSpec, model_name: str) -> List[str]:
    if isinstance(columns, list):
        return serialize_columns(df, columns, resolve_sep_token(model_name))
    return df[columns].astype(str).tolist()


def split_openai_batches(strings: List[str], max_chars: int = 5000) -> List[List[str]]:
    batches: List[List[str]] = []
    current_batch: List[str] = []
    current_chars = 0

    for text in strings:
        text_len = len(text)
        if current_batch and current_chars + text_len > max_chars:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(text)
        current_chars += text_len

    if current_batch:
        batches.append(current_batch)

    return batches


def infer_openai_embeddings(strings: List[str], model_name: str, openai_key: str) -> np.ndarray:
    client = openai.OpenAI(api_key=openai_key)
    outputs = []
    for batch in split_openai_batches(strings):
        response = client.embeddings.create(input=batch, model=model_name)
        outputs.append(np.array([item.embedding for item in response.data], dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def infer_sentence_transformer_embeddings(
    strings: List[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(strings, batch_size=batch_size, convert_to_numpy=True)


def ensure_2d_normalized(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def elapsed_summary(elapsed_seconds: float) -> str:
    return f"{elapsed_seconds:.2f}s ({elapsed_seconds / 60.0:.2f} min)"


def current_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def build_embeddings(
    df: pd.DataFrame,
    columns: ColumnSpec,
    model_name: str,
    batch_size: int,
    openai_key: Optional[str],
    label: str,
) -> np.ndarray:
    start_time = time.perf_counter()
    memory_before_mb = current_memory_mb()
    strings = serialize_embedding_input(df, columns, model_name)

    if openai_key:
        print(f"Inferindo embeddings com OpenAI para {label}...")
        embeddings = infer_openai_embeddings(strings, model_name, openai_key)
    else:
        print(f"Inferindo embeddings com SentenceTransformer para {label}...")
        embeddings = infer_sentence_transformer_embeddings(strings, model_name, batch_size)

    embeddings = ensure_2d_normalized(embeddings)
    print(f"embeddings_{label} shape: {embeddings.shape}")
    elapsed_seconds = time.perf_counter() - start_time
    memory_after_mb = current_memory_mb()
    print(f"Tempo para gerar embeddings_{label}: {elapsed_summary(elapsed_seconds)}")
    print(
        f"Memória para embeddings_{label}: "
        f"início {memory_before_mb:.2f} MB | fim {memory_after_mb:.2f} MB | delta {memory_after_mb - memory_before_mb:.2f} MB"
    )
    return embeddings


def build_embedding_pair(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_on: ColumnSpec,
    right_on: ColumnSpec,
    model_name: str,
    batch_size: int,
    openai_key: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    start_time = time.perf_counter()
    memory_before_mb = current_memory_mb()
    strings_left = serialize_embedding_input(df_left, left_on, model_name)
    strings_right = serialize_embedding_input(df_right, right_on, model_name)

    if openai_key:
        print("Inferindo embeddings com OpenAI para df_left...")
        embeddings_left = infer_openai_embeddings(strings_left, model_name, openai_key)
        print("Inferindo embeddings com OpenAI para df_right...")
        embeddings_right = infer_openai_embeddings(strings_right, model_name, openai_key)
    else:
        print("Inferindo embeddings com SentenceTransformer para df_left...")
        embeddings_left = infer_sentence_transformer_embeddings(strings_left, model_name, batch_size)
        print("Inferindo embeddings com SentenceTransformer para df_right...")
        embeddings_right = infer_sentence_transformer_embeddings(strings_right, model_name, batch_size)

    embeddings_left = ensure_2d_normalized(embeddings_left)
    embeddings_right = ensure_2d_normalized(embeddings_right)

    print(f"embeddings_left shape: {embeddings_left.shape}")
    print(f"embeddings_right shape: {embeddings_right.shape}")
    elapsed_seconds = time.perf_counter() - start_time
    memory_after_mb = current_memory_mb()
    print(f"Tempo para gerar embeddings do par: {elapsed_summary(elapsed_seconds)}")
    print(
        f"Memória para embeddings do par: "
        f"início {memory_before_mb:.2f} MB | fim {memory_after_mb:.2f} MB | delta {memory_after_mb - memory_before_mb:.2f} MB"
    )
    return embeddings_left, embeddings_right


def save_embeddings(
    embeddings_left: np.ndarray,
    embeddings_right: np.ndarray,
    left_output_path: str,
    right_output_path: str,
) -> None:
    os.makedirs(os.path.dirname(left_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(right_output_path), exist_ok=True)
    np.save(left_output_path, embeddings_left.astype(np.float32))
    np.save(right_output_path, embeddings_right.astype(np.float32))


def build_manifest_entry(model_name: str, left_path: str, right_path: str) -> dict:
    return {
        "model": model_name,
        "safe_model_name": safe_model_name(model_name),
        "base_embeddings_path": left_path,
        "query_embeddings_path": right_path,
    }


def add_timing_to_manifest(manifest_entry: dict, elapsed_seconds: float) -> dict:
    manifest_entry["elapsed_seconds"] = elapsed_seconds
    manifest_entry["elapsed_minutes"] = elapsed_seconds / 60.0
    return manifest_entry


def add_memory_to_manifest(
    manifest_entry: dict,
    memory_start_mb: float,
    memory_end_mb: float,
    memory_peak_mb: float,
) -> dict:
    manifest_entry["memory_start_mb"] = memory_start_mb
    manifest_entry["memory_end_mb"] = memory_end_mb
    manifest_entry["memory_peak_mb"] = memory_peak_mb
    manifest_entry["memory_delta_mb"] = memory_end_mb - memory_start_mb
    return manifest_entry


def build_single_manifest_entry(model_name: str, output_path: str, mode: EmbeddingMode) -> dict:
    manifest = {
        "model": model_name,
        "safe_model_name": safe_model_name(model_name),
        "mode": mode,
    }
    if mode == "base":
        manifest["base_embeddings_path"] = output_path
    elif mode == "query":
        manifest["query_embeddings_path"] = output_path
    return manifest


def build_output_paths(output_dir: str, model_name: str) -> Tuple[str, str]:
    safe_name = safe_model_name(model_name)
    return (
        os.path.join(output_dir, f"embeddings_base_{safe_name}.npy"),
        os.path.join(output_dir, f"embeddings_query_{safe_name}.npy"),
    )


def main() -> None:
    args = parse_args()
    models = args.models or DEFAULT_MODELS

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = []

    if args.mode == "both":
        require_input_path(args.base_path, "base")
        require_input_path(args.query_path, "query")
        df_base = load_dataframe(args.base_path, csv_encoding=args.csv_encoding)
        df_query = load_dataframe(args.query_path, csv_encoding=args.csv_encoding)
        left_on, right_on = resolve_columns(args, df_base, df_query)

        for model_name in models:
            model_start_time = time.perf_counter()
            model_memory_start_mb = current_memory_mb()
            base_output_path, query_output_path = build_output_paths(args.output_dir, model_name)

            print(f"Gerando embeddings com o modelo: {model_name}")
            embeddings_left, embeddings_right = build_embedding_pair(
                df_left=df_base,
                df_right=df_query,
                left_on=left_on,
                right_on=right_on,
                model_name=model_name,
                batch_size=args.batch_size,
                openai_key=args.openai_key,
            )
            save_embeddings(
                embeddings_left=embeddings_left,
                embeddings_right=embeddings_right,
                left_output_path=base_output_path,
                right_output_path=query_output_path,
            )
            print(f"Embeddings salvos em: {base_output_path} e {query_output_path}")
            elapsed_seconds = time.perf_counter() - model_start_time
            model_memory_end_mb = current_memory_mb()
            model_memory_peak_mb = max(model_memory_start_mb, model_memory_end_mb)
            print(f"Tempo total do modelo {model_name}: {elapsed_summary(elapsed_seconds)}")
            print(
                f"Memória total do modelo {model_name}: "
                f"início {model_memory_start_mb:.2f} MB | fim {model_memory_end_mb:.2f} MB | "
                f"peak aprox. {model_memory_peak_mb:.2f} MB | delta {model_memory_end_mb - model_memory_start_mb:.2f} MB"
            )
            manifest.append(
                add_memory_to_manifest(
                    add_timing_to_manifest(
                        build_manifest_entry(model_name, base_output_path, query_output_path),
                        elapsed_seconds,
                    ),
                    model_memory_start_mb,
                    model_memory_end_mb,
                    model_memory_peak_mb,
                )
            )

    elif args.mode == "base":
        require_input_path(args.base_path, "base")
        df_base = load_dataframe(args.base_path, csv_encoding=args.csv_encoding)
        base_columns = resolve_single_side_columns(args.left_on, normalize_column_spec(args.on), "left")

        for model_name in models:
            model_start_time = time.perf_counter()
            model_memory_start_mb = current_memory_mb()
            base_output_path, _ = build_output_paths(args.output_dir, model_name)

            print(f"Gerando embeddings da base com o modelo: {model_name}")
            embeddings_base = build_embeddings(
                df=df_base,
                columns=base_columns,
                model_name=model_name,
                batch_size=args.batch_size,
                openai_key=args.openai_key,
                label="base",
            )
            os.makedirs(os.path.dirname(base_output_path), exist_ok=True)
            np.save(base_output_path, embeddings_base.astype(np.float32))
            print(f"Embeddings salvos em: {base_output_path}")
            elapsed_seconds = time.perf_counter() - model_start_time
            model_memory_end_mb = current_memory_mb()
            model_memory_peak_mb = max(model_memory_start_mb, model_memory_end_mb)
            print(f"Tempo total do modelo {model_name}: {elapsed_summary(elapsed_seconds)}")
            print(
                f"Memória total do modelo {model_name}: "
                f"início {model_memory_start_mb:.2f} MB | fim {model_memory_end_mb:.2f} MB | "
                f"peak aprox. {model_memory_peak_mb:.2f} MB | delta {model_memory_end_mb - model_memory_start_mb:.2f} MB"
            )
            manifest.append(
                add_memory_to_manifest(
                    add_timing_to_manifest(
                        build_single_manifest_entry(model_name, base_output_path, "base"),
                        elapsed_seconds,
                    ),
                    model_memory_start_mb,
                    model_memory_end_mb,
                    model_memory_peak_mb,
                )
            )

    elif args.mode == "query":
        require_input_path(args.query_path, "query")
        df_query = load_dataframe(args.query_path, csv_encoding=args.csv_encoding)
        query_columns = resolve_single_side_columns(args.right_on, normalize_column_spec(args.on), "right")

        for model_name in models:
            model_start_time = time.perf_counter()
            model_memory_start_mb = current_memory_mb()
            _, query_output_path = build_output_paths(args.output_dir, model_name)

            print(f"Gerando embeddings da query com o modelo: {model_name}")
            embeddings_query = build_embeddings(
                df=df_query,
                columns=query_columns,
                model_name=model_name,
                batch_size=args.batch_size,
                openai_key=args.openai_key,
                label="query",
            )
            os.makedirs(os.path.dirname(query_output_path), exist_ok=True)
            np.save(query_output_path, embeddings_query.astype(np.float32))
            print(f"Embeddings salvos em: {query_output_path}")
            elapsed_seconds = time.perf_counter() - model_start_time
            model_memory_end_mb = current_memory_mb()
            model_memory_peak_mb = max(model_memory_start_mb, model_memory_end_mb)
            print(f"Tempo total do modelo {model_name}: {elapsed_summary(elapsed_seconds)}")
            print(
                f"Memória total do modelo {model_name}: "
                f"início {model_memory_start_mb:.2f} MB | fim {model_memory_end_mb:.2f} MB | "
                f"peak aprox. {model_memory_peak_mb:.2f} MB | delta {model_memory_end_mb - model_memory_start_mb:.2f} MB"
            )
            manifest.append(
                add_memory_to_manifest(
                    add_timing_to_manifest(
                        build_single_manifest_entry(model_name, query_output_path, "query"),
                        elapsed_seconds,
                    ),
                    model_memory_start_mb,
                    model_memory_end_mb,
                    model_memory_peak_mb,
                )
            )

    if args.manifest_path:
        manifest_dir = os.path.dirname(args.manifest_path)
        if manifest_dir:
            os.makedirs(manifest_dir, exist_ok=True)
        with open(args.manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)
        print(f"Manifest salvo em: {args.manifest_path}")


if __name__ == "__main__":
    main()
