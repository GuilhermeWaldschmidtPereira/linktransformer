# my_npy_demo/run_from_source.py

#!/usr/bin/env python3
import os
import sys
from typing import Union, List, Optional


import numpy as np
import pandas as pd


# ======================================================
# 1) Colocar o src/ do repositório no sys.path
#    (para importar diretamente o código do GitHub)
# ======================================================
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Agora podemos importar direto do arquivo infer.py
from linktransformer.infer_scann import merge_knn_scann
modelos_a_utilizar = [
                        "sentence-transformers/all-MiniLM-L6-v2", 
                        "sentence-transformers/all-mpnet-base-v2", 
                        "intfloat/multilingual-e5-large",
                        "neuralmind/bert-large-portuguese-cased"
                    ]

# modelos_a_utilizar = ["sentence-transformers/all-MiniLM-L6-v2"]

def read_csv_with_fallback(path: str) -> pd.DataFrame:
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

def main():
    # ==========================================
    # 2) Ler os CSV de endereços
    # ==========================================
    data_dir = os.path.join(THIS_DIR, "../data")
    base_path = os.path.join(data_dir, "base.csv")
    query_path = os.path.join(data_dir, "query.csv")

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Não encontrei {base_path}")
    if not os.path.exists(query_path):
        raise FileNotFoundError(f"Não encontrei {query_path}")

    embeddings_query_path = os.path.join(data_dir, "embeddings_query.npy")
    embeddings_base_path = os.path.join(data_dir, "embeddings_base.npy")

    if not os.path.exists(embeddings_base_path):
        raise FileNotFoundError(f"Não encontrei {embeddings_base_path}")
    if not os.path.exists(embeddings_query_path):
        raise FileNotFoundError(f"Não encontrei {embeddings_query_path}")

    df_base = read_csv_with_fallback(base_path)
    df_query = read_csv_with_fallback(query_path)

    # -------------------------
    # Configurações
    # -------------------------
    on: Optional[Union[str, List[str]]] = None
    left_on: Optional[Union[str, List[str]]] = None
    right_on: Optional[Union[str, List[str]]] = None
    for modelo in modelos_a_utilizar:
        suffixes = ("_x", "_y")

        # -------------------------
        # 3) Escolher colunas de junção
        # -------------------------
        if on is None:
            on = list(set(df_base.columns).intersection(set(df_query.columns)))
            print(f"Colunas em comum detectadas para matching: {on}")

        if left_on is None:
            left_on = on
        if right_on is None:
            right_on = on

        # não usamos mais "on" diretamente
        on = None

        df1 = df_base.copy()
        df2 = df_query.copy()

        # garantir que não existe id_lt
        if "id_lt" in df1.columns:
            raise ValueError("Column id_lt already exists in df_base, renomeie antes de continuar")
        if "id_lt" in df2.columns:
            raise ValueError("Column id_lt already exists in df_query, renomeie antes de continuar")

        df1.loc[:, "id_lt"] = np.arange(len(df1))
        df2.loc[:, "id_lt"] = np.arange(len(df2))

        k = 2
        for i in range(k):
            if i > 0:
                merge_knn_scann(i, df1, df2, suffixes, modelo)
    

if __name__ == "__main__":
    main()
