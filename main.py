### Inference and Linkage script
### We want to link dfs together using embeddings

import os
import time
from typing import Union, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from linktransformer.utils import *

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

df_base = pd.read_csv(base_path)
df_query = pd.read_csv(query_path)


# -------------------------
# 1) Preparar colunas e IDs
# -------------------------
if on is None:
    on = list(set(df1.columns).intersection(set(df2.columns)))

if left_on is None:
    left_on = on
if right_on is None:
    right_on = on

# não usamos mais "on" diretamente
on = None

df1 = df1.copy()
df2 = df2.copy()

if "id_lt" in df1.columns:
    raise ValueError("Column id_lt already exists in df1, please rename it to proceed")
if "id_lt" in df2.columns:
    raise ValueError("Column id_lt already exists in df2, please rename it to proceed")

df1.loc[:, "id_lt"] = np.arange(len(df1))
df2.loc[:, "id_lt"] = np.arange(len(df2))

if isinstance(right_on, list):
    strings_right = serialize_columns(df2, right_on, model=model)
if isinstance(left_on, list):
    strings_left = serialize_columns(df1, left_on, model=model)
else:
    # caso simples: uma única coluna em cada lado
    strings_left = df1[left_on].tolist()
    strings_right = df2[right_on].tolist()

# -------------------------
# 2) Carregar modelo e embeddings
# -------------------------
if isinstance(model, str):
    if openai_key is None:
        model = load_model(model)

embeddings1 = infer_embeddings(
    strings_left,
    model,
    batch_size=batch_size,
    openai_key=openai_key,
    return_numpy=True,
)
embeddings2 = infer_embeddings(
    strings_right,
    model,
    batch_size=batch_size,
    openai_key=openai_key,
    return_numpy=True,
)

if len(embeddings1.shape) == 1:
    embeddings1 = np.expand_dims(embeddings1, axis=0)
if len(embeddings2.shape) == 1:
    embeddings2 = np.expand_dims(embeddings2, axis=0)

# Normaliza embeddings -> ScaNN com "dot_product" ~ cosine similarity
embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
