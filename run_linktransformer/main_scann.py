#!/usr/bin/env python3
import os
import sys
import traceback
from typing import List

import numpy as np
import pandas as pd


THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
DATA_DIR = os.path.join(REPO_ROOT, "data")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from linktransformer.infer_scann import merge_knn_scann  # noqa: E402


MODELOS_A_UTILIZAR: List[str] = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/multilingual-e5-large",
    "neuralmind/bert-large-portuguese-cased",
]


def safe_model_name(modelo: str) -> str:
    safe_model = str(modelo).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")
    return safe_model


def assert_embeddings_exist(modelo: str) -> None:
    safe_model = safe_model_name(modelo)
    paths = [
        os.path.join(DATA_DIR, f"embeddings_base_{safe_model}.npy"),
        os.path.join(DATA_DIR, f"embeddings_query_{safe_model}.npy"),
    ]
    missing_paths = [path for path in paths if not os.path.exists(path)]
    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Embeddings pre-computados não encontrados para o ScaNN. "
            "Execute primeiro o script isolado de embeddings.\n"
            f"{missing}"
        )


def load_input_data():
    base_path = os.environ.get("LINKTRANSFORMER_BASE_CSV", os.path.join(DATA_DIR, "base.csv"))
    query_path = os.environ.get("LINKTRANSFORMER_QUERY_CSV", os.path.join(DATA_DIR, "query.csv"))

    fallback_path = os.path.join(DATA_DIR, "cnefe_layout_setor_esperado.csv")
    if not os.path.exists(base_path) and not os.path.exists(query_path) and os.path.exists(fallback_path):
        print(
            ">>> data/base.csv e data/query.csv não encontrados. "
            f"Usando {fallback_path} como base e query."
        )
        base_path = fallback_path
        query_path = fallback_path
    else:
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Não encontrei {base_path}")
        if not os.path.exists(query_path):
            raise FileNotFoundError(f"Não encontrei {query_path}")

    print(f">>> Lendo CSVs\n    base = {base_path}\n    query = {query_path}", flush=True)

    def read_csv_with_fallback(path):
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f">>> Arquivo {path} ({file_size_mb:.2f} MB)", flush=True)

        for sep in [",", ";"]:
            try:
                print(f">>> Tentando ler {path} com sep='{sep}'...", flush=True)
                df = pd.read_csv(path, sep=sep)
                print(
                    f"    Sucesso! Arquivo lido com sep='{sep}' | linhas={len(df)} | colunas={len(df.columns)}",
                    flush=True,
                )
                return df
            except pd.errors.ParserError as e:
                print(f"    Falha com sep='{sep}': ParserError -> {str(e)[:150]}", flush=True)
            except Exception as e:
                print(f"    Falha com sep='{sep}': {type(e).__name__} -> {str(e)[:150]}", flush=True)
                continue

        raise ValueError(f"Não consegui ler {path} com nenhum separador testado (,;)")

    return read_csv_with_fallback(base_path), read_csv_with_fallback(query_path)


def prepare_dataframes(df_base: pd.DataFrame, df_query: pd.DataFrame):
    df1 = df_base.copy()
    df2 = df_query.copy()

    if "id_lt" in df1.columns:
        raise ValueError("Column id_lt already exists in df_base, renomeie antes de continuar")
    if "id_lt" in df2.columns:
        raise ValueError("Column id_lt already exists in df_query, renomeie antes de continuar")

    df1.loc[:, "id_lt"] = np.arange(len(df1))
    df2.loc[:, "id_lt"] = np.arange(len(df2))
    return df1, df2


def main() -> None:
    print(f">>> Script em execução: {__file__}", flush=True)
    print(f">>> Modelos configurados: {MODELOS_A_UTILIZAR}", flush=True)

    try:
        df_base, df_query = load_input_data()
        suffixes = ("_x", "_y")

        # Mantém o comportamento anterior de range(2) com i > 0, que executava apenas k=1.
        ks = [1]

        for modelo in MODELOS_A_UTILIZAR:
            print(f">>> Iniciando processamento do modelo: {modelo}", flush=True)
            assert_embeddings_exist(modelo)
            df1, df2 = prepare_dataframes(df_base, df_query)

            for k in ks:
                print(f">>> Rodando ScaNN | modelo={modelo} | k={k}", flush=True)
                print(
                    f">>> Tamanhos dos dataframes: df1={len(df1)} linhas | df2={len(df2)} linhas",
                    flush=True,
                )
                merge_knn_scann(k, df1, df2, suffixes, modelo)
    except Exception as exc:
        print(f">>> ERRO em main_scann.py: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc(file=sys.stdout)
        raise


if __name__ == "__main__":
    main()
