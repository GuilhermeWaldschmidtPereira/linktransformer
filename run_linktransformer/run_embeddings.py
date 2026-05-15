#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import openai
import pandas as pd
import transformers
from sentence_transformers import SentenceTransformer


DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/multilingual-e5-large",
    "neuralmind/bert-large-portuguese-cased",
]

ColumnSpec = Union[str, List[str]]


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
    parser.add_argument("--on", nargs="+")
    parser.add_argument("--left-on", nargs="+")
    parser.add_argument("--right-on", nargs="+")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--manifest-path", default=None)
    return parser.parse_args()


def safe_model_name(model_name: str) -> str:
    sanitized = model_name.replace(os.sep, "_")
    if os.path.altsep:
        sanitized = sanitized.replace(os.path.altsep, "_")
    return sanitized


def load_dataframe(path: str) -> pd.DataFrame:
    suffix = os.path.splitext(path)[1].lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        f"Formato de arquivo não suportado para {path}. Use .csv, .parquet ou .pq."
    )


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


def build_embedding_pair(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_on: ColumnSpec,
    right_on: ColumnSpec,
    model_name: str,
    batch_size: int,
    openai_key: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
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


def main() -> None:
    args = parse_args()
    models = args.models or DEFAULT_MODELS

    if not os.path.exists(args.base_path):
        raise FileNotFoundError(f"Não encontrei {args.base_path}")
    if not os.path.exists(args.query_path):
        raise FileNotFoundError(f"Não encontrei {args.query_path}")

    df_base = load_dataframe(args.base_path)
    df_query = load_dataframe(args.query_path)
    left_on, right_on = resolve_columns(args, df_base, df_query)

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = []

    for model_name in models:
        safe_name = safe_model_name(model_name)
        base_output_path = os.path.join(args.output_dir, f"embeddings_base_{safe_name}.npy")
        query_output_path = os.path.join(args.output_dir, f"embeddings_query_{safe_name}.npy")

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
        manifest.append(build_manifest_entry(model_name, base_output_path, query_output_path))

    if args.manifest_path:
        manifest_dir = os.path.dirname(args.manifest_path)
        if manifest_dir:
            os.makedirs(manifest_dir, exist_ok=True)
        with open(args.manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)
        print(f"Manifest salvo em: {args.manifest_path}")


if __name__ == "__main__":
    main()
