#!/usr/bin/env python3
import os
import sys
import time
from typing import Union, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
# ==========================================
# 1) Tornar o src/ importável como pacote
# ==========================================
THIS_DIR = os.path.dirname(__file__)           # pasta deste script (ex.: run_linktransformer)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
data_dir = os.path.join(REPO_ROOT, "data/dedup_task")
base_path = os.path.join(data_dir, "dup_data.csv")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Agora podemos importar do pacote linktransformer
from linktransformer.utils import (
    serialize_columns,
    infer_embeddings,
    load_model,
)

from linktransformer.infer_main import dedup_rows

def main():
    # ==========================================
    # 2) Ler o CSV de endereços
    # ==========================================

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Não encontrei {base_path}")

    df_base = pd.read_csv(base_path)

    print("\nBase (endereços com duplicatas):")
    print(df_base.head())

    # ==========================================
    # 3) Chamar dedup_rows diretamente
    # ==========================================
    deduped = dedup_rows(
        df=df_base,
        on=['nome', 'sobrenome'],
        model="sentence-transformers/all-MiniLM-L6-v2",
        cluster_type="agglomerative",
        cluster_params={
            'threshold': 0.7,
        }
    )

    print("\nDataFrame após deduplicação:")
    deduped.to_csv("deduplicated_output.csv", index=False)

if __name__ == "__main__":
    main()