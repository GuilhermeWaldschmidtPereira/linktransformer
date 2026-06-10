#!/usr/bin/env python3
import argparse
import json
import os
import threading
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

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
DEFAULT_MUNICIPALITY_COLUMN = "id_municipio"
CSV_SEPARATORS = [",", ";"]

ColumnSpec = Union[str, List[str]]
EmbeddingMode = str
MB = 1024 ** 2


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
    parser.add_argument(
        "--municipality-column",
        default=None,
        help=(
            "Nome da coluna de ID do município usada para agrupar base e query. "
            f"Se omitida, usa {DEFAULT_MUNICIPALITY_COLUMN}."
        ),
    )
    parser.add_argument(
        "--base-municipality-column",
        default=None,
        help="Nome da coluna de ID do município na base. Sobrescreve --municipality-column.",
    )
    parser.add_argument(
        "--query-municipality-column",
        default=None,
        help="Nome da coluna de ID do município na query. Sobrescreve --municipality-column.",
    )
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


def resolve_csv_separator(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as csv_file:
        header = csv_file.readline()

    if header.count(";") > header.count(","):
        return ";"
    return ","


def candidate_csv_separators(path: str) -> List[str]:
    preferred = resolve_csv_separator(path)
    return [preferred] + [separator for separator in CSV_SEPARATORS if separator != preferred]


def dataframe_has_plausible_columns(df: pd.DataFrame) -> bool:
    if len(df.columns) != 1:
        return True

    column_name = str(df.columns[0])
    return not any(separator in column_name for separator in CSV_SEPARATORS)


def load_csv_with_fallback(
    path: str,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    candidate_separators = candidate_csv_separators(path)
    first_implausible: Optional[pd.DataFrame] = None
    if encoding:
        for sep in candidate_separators:
            print(f"Lendo CSV com encoding explícito: {encoding} | separador: {sep}")
            try:
                df = pd.read_csv(path, encoding=encoding, sep=sep)
            except pd.errors.ParserError:
                continue
            if dataframe_has_plausible_columns(df):
                return df
            if first_implausible is None:
                first_implausible = df
        if first_implausible is not None:
            return first_implausible
        return pd.read_csv(path, encoding=encoding, sep=resolve_csv_separator(path))

    candidate_encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_error: Optional[UnicodeDecodeError] = None
    first_implausible = None

    for candidate in candidate_encodings:
        for sep in candidate_separators:
            try:
                print(f"Tentando ler CSV com encoding: {candidate} | separador: {sep}")
                df = pd.read_csv(path, encoding=candidate, sep=sep)
            except UnicodeDecodeError as exc:
                last_error = exc
                break
            except pd.errors.ParserError:
                continue

            if dataframe_has_plausible_columns(df):
                return df
            if first_implausible is None:
                first_implausible = df

    if first_implausible is not None:
        return first_implausible

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


def load_dataframe(
    path: str,
    csv_encoding: Optional[str] = None,
) -> pd.DataFrame:
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
    return process.memory_info().rss / MB


class PeakMemoryMonitor:
    """Samples process RSS while a block runs and reports the peak increase."""

    def __init__(self, interval_seconds: float = 0.001):
        self.process = psutil.Process(os.getpid())
        self.interval_seconds = interval_seconds
        self.start_mb = 0.0
        self.end_mb = 0.0
        self.peak_mb = 0.0
        self.peak_delta_mb = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "PeakMemoryMonitor":
        self.start_mb = current_memory_mb()
        self.peak_mb = self.start_mb
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.end_mb = current_memory_mb()
        self.peak_mb = max(self.peak_mb, self.end_mb)
        self.peak_delta_mb = max(0.0, self.peak_mb - self.start_mb)

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.is_set():
            self.peak_mb = max(self.peak_mb, current_memory_mb())
            self._stop_event.wait(self.interval_seconds)


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

    with PeakMemoryMonitor() as memory_monitor:
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
    memory_after_mb = memory_monitor.end_mb
    print(f"Tempo para gerar embeddings_{label}: {elapsed_summary(elapsed_seconds)}")
    print(
        f"Memória para embeddings_{label}: "
        f"início {memory_before_mb:.2f} MB | fim {memory_after_mb:.2f} MB | "
        f"pico {memory_monitor.peak_mb:.2f} MB | pico_delta {memory_monitor.peak_delta_mb:.2f} MB"
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

    with PeakMemoryMonitor() as memory_monitor:
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
    memory_after_mb = memory_monitor.end_mb
    print(f"Tempo para gerar embeddings do par: {elapsed_summary(elapsed_seconds)}")
    print(
        f"Memória para embeddings do par: "
        f"início {memory_before_mb:.2f} MB | fim {memory_after_mb:.2f} MB | "
        f"pico {memory_monitor.peak_mb:.2f} MB | pico_delta {memory_monitor.peak_delta_mb:.2f} MB"
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
    manifest_entry["memory_delta_mb"] = max(0.0, memory_end_mb - memory_start_mb)
    manifest_entry["memory_net_delta_mb"] = memory_end_mb - memory_start_mb
    manifest_entry["memory_peak_delta_mb"] = max(0.0, memory_peak_mb - memory_start_mb)
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


def sanitize_municipality_id(municipality_id: object) -> str:
    value = str(municipality_id).strip()
    if not value:
        return "vazio"

    sanitized = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized)


def serialize_municipality_id(municipality_id: object) -> Optional[object]:
    if pd.isna(municipality_id):
        return None
    if isinstance(municipality_id, np.generic):
        return municipality_id.item()
    return municipality_id


def resolve_municipality_columns(
    args: argparse.Namespace,
    df_base: Optional[pd.DataFrame] = None,
    df_query: Optional[pd.DataFrame] = None,
) -> Tuple[str, str]:
    shared_column = args.municipality_column or DEFAULT_MUNICIPALITY_COLUMN
    base_column = args.base_municipality_column or shared_column
    query_column = args.query_municipality_column or shared_column

    if df_base is not None and base_column not in df_base.columns:
        raise ValueError(
            f"Coluna de município da base não encontrada: {base_column}. "
            "Use --base-municipality-column ou --municipality-column."
        )
    if df_query is not None and query_column not in df_query.columns:
        raise ValueError(
            f"Coluna de município da query não encontrada: {query_column}. "
            "Use --query-municipality-column ou --municipality-column."
        )

    return base_column, query_column


def get_municipality_output_dir(
    output_dir: str,
    side: EmbeddingMode,
    model_name: str,
    multiple_models: bool,
) -> str:
    side_output_dir = os.path.join(output_dir, side)
    return os.path.join(side_output_dir, safe_model_name(model_name))


def get_municipality_output_path(
    output_dir: str,
    side: EmbeddingMode,
    model_name: str,
    municipality_id: object,
    multiple_models: bool,
) -> str:
    side_output_dir = get_municipality_output_dir(
        output_dir=output_dir,
        side=side,
        model_name=model_name,
        multiple_models=multiple_models,
    )
    municipality_id_safe = sanitize_municipality_id(municipality_id)
    return os.path.join(side_output_dir, f"embedding_{side}_{municipality_id_safe}.npy")


def iter_municipality_groups(
    df: pd.DataFrame,
    municipality_column: str,
) -> Iterator[Tuple[object, pd.DataFrame]]:
    grouped = df.groupby(municipality_column, sort=True, dropna=False)
    for municipality_id, df_municipality in grouped:
        yield municipality_id, df_municipality


def process_side_by_municipality(
    df: pd.DataFrame,
    columns: ColumnSpec,
    municipality_column: str,
    output_dir: str,
    side: EmbeddingMode,
    model_name: str,
    batch_size: int,
    openai_key: Optional[str],
    multiple_models: bool,
) -> List[dict]:
    side_output_dir = get_municipality_output_dir(
        output_dir=output_dir,
        side=side,
        model_name=model_name,
        multiple_models=multiple_models,
    )
    os.makedirs(side_output_dir, exist_ok=True)

    entries = []
    municipality_ids = df[municipality_column].drop_duplicates()
    print(
        f">>> {side}: {len(df)} registros | {len(municipality_ids)} municípios | "
        f"coluna {municipality_column}",
        flush=True,
    )

    for municipality_id, df_municipality in iter_municipality_groups(df, municipality_column):
        output_path = get_municipality_output_path(
            output_dir=output_dir,
            side=side,
            model_name=model_name,
            municipality_id=municipality_id,
            multiple_models=multiple_models,
        )
        print(
            f"  Município {municipality_id} | {len(df_municipality)} registros | "
            f"gerando {side}",
            flush=True,
        )
        embeddings = build_embeddings(
            df=df_municipality,
            columns=columns,
            model_name=model_name,
            batch_size=batch_size,
            openai_key=openai_key,
            label=f"{side}_{municipality_id}",
        )
        np.save(output_path, embeddings.astype(np.float32))
        print(
            f"  Município {municipality_id} | {len(df_municipality)} registros | "
            f"salvo em: {output_path}",
            flush=True,
        )
        entries.append(
            {
                "municipality_id": serialize_municipality_id(municipality_id),
                "num_rows": len(df_municipality),
                "embeddings_path": output_path,
            }
        )
    return entries


def build_municipality_manifest_entry(
    model_name: str,
    mode: EmbeddingMode,
    base_municipality_column: Optional[str] = None,
    query_municipality_column: Optional[str] = None,
    base_files: Optional[List[dict]] = None,
    query_files: Optional[List[dict]] = None,
) -> dict:
    manifest = {
        "model": model_name,
        "safe_model_name": safe_model_name(model_name),
        "mode": mode,
    }
    if base_municipality_column is not None:
        manifest["base_municipality_column"] = base_municipality_column
    if query_municipality_column is not None:
        manifest["query_municipality_column"] = query_municipality_column
    if base_files is not None:
        manifest["base_files"] = base_files
    if query_files is not None:
        manifest["query_files"] = query_files
    return manifest


def main() -> None:
    args = parse_args()
    models = args.models or DEFAULT_MODELS

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = []
    multiple_models = len(models) > 1

    os.makedirs(os.path.join(args.output_dir, "base"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "query"), exist_ok=True)

    if args.mode == "both":
        require_input_path(args.base_path, "base")
        require_input_path(args.query_path, "query")
        df_base = load_dataframe(args.base_path, csv_encoding=args.csv_encoding)
        df_query = load_dataframe(args.query_path, csv_encoding=args.csv_encoding)
        left_on, right_on = resolve_columns(args, df_base, df_query)
        base_municipality_column, query_municipality_column = resolve_municipality_columns(
            args,
            df_base=df_base,
            df_query=df_query,
        )

        for model_name in models:
            model_start_time = time.perf_counter()
            model_memory_start_mb = current_memory_mb()

            print(f"Gerando embeddings com o modelo: {model_name}")
            with PeakMemoryMonitor() as memory_monitor:
                base_files = process_side_by_municipality(
                    df=df_base,
                    columns=left_on,
                    municipality_column=base_municipality_column,
                    output_dir=args.output_dir,
                    side="base",
                    model_name=model_name,
                    batch_size=args.batch_size,
                    openai_key=args.openai_key,
                    multiple_models=multiple_models,
                )
                query_files = process_side_by_municipality(
                    df=df_query,
                    columns=right_on,
                    municipality_column=query_municipality_column,
                    output_dir=args.output_dir,
                    side="query",
                    model_name=model_name,
                    batch_size=args.batch_size,
                    openai_key=args.openai_key,
                    multiple_models=multiple_models,
                )
            elapsed_seconds = time.perf_counter() - model_start_time
            model_memory_end_mb = memory_monitor.end_mb
            model_memory_peak_mb = memory_monitor.peak_mb
            print(f"Tempo total do modelo {model_name}: {elapsed_summary(elapsed_seconds)}")
            print(
                f"Memória total do modelo {model_name}: "
                f"início {model_memory_start_mb:.2f} MB | fim {model_memory_end_mb:.2f} MB | "
                f"pico {model_memory_peak_mb:.2f} MB | pico_delta {memory_monitor.peak_delta_mb:.2f} MB"
            )
            manifest.append(
                add_memory_to_manifest(
                    add_timing_to_manifest(
                        build_municipality_manifest_entry(
                            model_name,
                            "both",
                            base_municipality_column=base_municipality_column,
                            query_municipality_column=query_municipality_column,
                            base_files=base_files,
                            query_files=query_files,
                        ),
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
        base_municipality_column, _ = resolve_municipality_columns(args, df_base=df_base)

        for model_name in models:
            model_start_time = time.perf_counter()
            model_memory_start_mb = current_memory_mb()

            print(f"Gerando embeddings da base com o modelo: {model_name}")
            with PeakMemoryMonitor() as memory_monitor:
                base_files = process_side_by_municipality(
                    df=df_base,
                    columns=base_columns,
                    municipality_column=base_municipality_column,
                    output_dir=args.output_dir,
                    side="base",
                    model_name=model_name,
                    batch_size=args.batch_size,
                    openai_key=args.openai_key,
                    multiple_models=multiple_models,
                )
            elapsed_seconds = time.perf_counter() - model_start_time
            model_memory_end_mb = memory_monitor.end_mb
            model_memory_peak_mb = memory_monitor.peak_mb
            print(f"Tempo total do modelo {model_name}: {elapsed_summary(elapsed_seconds)}")
            print(
                f"Memória total do modelo {model_name}: "
                f"início {model_memory_start_mb:.2f} MB | fim {model_memory_end_mb:.2f} MB | "
                f"pico {model_memory_peak_mb:.2f} MB | pico_delta {memory_monitor.peak_delta_mb:.2f} MB"
            )
            manifest.append(
                add_memory_to_manifest(
                    add_timing_to_manifest(
                        build_municipality_manifest_entry(
                            model_name,
                            "base",
                            base_municipality_column=base_municipality_column,
                            base_files=base_files,
                        ),
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
        _, query_municipality_column = resolve_municipality_columns(args, df_query=df_query)

        for model_name in models:
            model_start_time = time.perf_counter()
            model_memory_start_mb = current_memory_mb()

            print(f"Gerando embeddings da query com o modelo: {model_name}")
            with PeakMemoryMonitor() as memory_monitor:
                query_files = process_side_by_municipality(
                    df=df_query,
                    columns=query_columns,
                    municipality_column=query_municipality_column,
                    output_dir=args.output_dir,
                    side="query",
                    model_name=model_name,
                    batch_size=args.batch_size,
                    openai_key=args.openai_key,
                    multiple_models=multiple_models,
                )
            elapsed_seconds = time.perf_counter() - model_start_time
            model_memory_end_mb = memory_monitor.end_mb
            model_memory_peak_mb = memory_monitor.peak_mb
            print(f"Tempo total do modelo {model_name}: {elapsed_summary(elapsed_seconds)}")
            print(
                f"Memória total do modelo {model_name}: "
                f"início {model_memory_start_mb:.2f} MB | fim {model_memory_end_mb:.2f} MB | "
                f"pico {model_memory_peak_mb:.2f} MB | pico_delta {memory_monitor.peak_delta_mb:.2f} MB"
            )
            manifest.append(
                add_memory_to_manifest(
                    add_timing_to_manifest(
                        build_municipality_manifest_entry(
                            model_name,
                            "query",
                            query_municipality_column=query_municipality_column,
                            query_files=query_files,
                        ),
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
