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

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Agora podemos importar do pacote linktransformer
from linktransformer.utils import (
    serialize_columns,
    infer_embeddings,
    load_model,
)
from linktransformer.infer_main import merge_knn, merge_knn2, merge_knn_hnsw_julia, merge_knn_nmslib

modelos_a_utilizar = [
                        # "sentence-transformers/all-MiniLM-L6-v2", 
                        # "sentence-transformers/all-mpnet-base-v2", 
                        # "intfloat/multilingual-e5-large",
                        "neuralmind/bert-large-portuguese-cased"
                    ]

# modelos_a_utilizar = ["sentence-transformers/all-MiniLM-L6-v2"]

# ==========================================
# 2) Ler os CSV de endereços
# ==========================================
data_dir = os.path.join(THIS_DIR, "../data")
base_path = os.path.join(data_dir, "enderecos_10000.csv")
query_path = os.path.join(data_dir, "enderecos_100_ruido.csv")

if not os.path.exists(base_path):
    raise FileNotFoundError(f"Não encontrei {base_path}")
if not os.path.exists(query_path):
    raise FileNotFoundError(f"Não encontrei {query_path}")

# Verificar se os arquivos de embeddings já existem
embeddings_query_path = os.path.join(data_dir, "embeddings_query.npy")
embeddings_base_path = os.path.join(data_dir, "embeddings_base.npy")

df_base = pd.read_csv(base_path)
df_query = pd.read_csv(query_path)

def main():
    for modelo in modelos_a_utilizar:

        # -------------------------
        # Configurações
        # -------------------------
        on: Optional[Union[str, List[str]]] = None
        left_on: Optional[Union[str, List[str]]] = None
        right_on: Optional[Union[str, List[str]]] = None

        df_base = pd.read_csv(base_path)
        df_query = pd.read_csv(query_path)

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
        model: Union[str, object] = modelo
        suffixes = ("_x", "_y")

        k = 5
        for i in range(k):
            if i > 0:
                merge_knn(i, df1, df2, suffixes, modelo)
                #merge_knn2(i, df1, df2, suffixes, modelo)
                #merge_knn_nmslib(i, df1, df2, suffixes, modelo)
                # merge_knn_hnsw_julia(i, df1, df2, suffixes, modelo)

def build_embedding():

    for modelo in modelos_a_utilizar:
        model: Union[str, object] = modelo
        suffixes = ("_x", "_y")
        batch_size = 128
        openai_key: Optional[str] = None  # se for usar OpenAI, coloque aqui
        on = None
        left_on = None
        right_on = None

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

    
        print(f"Embeddings não encontrados. Serão gerados novamente.")

        # -------------------------
        # 4) Serializar colunas (usa utils do linktransformer)
        # -------------------------
        if isinstance(right_on, list):
            strings_right = serialize_columns(df2, right_on, model=model)
        else:
            # caso simples: uma única coluna em cada lado
            strings_right = df2[right_on].tolist()

        if isinstance(left_on, list):
            strings_left = serialize_columns(df1, left_on, model=model)
        else:
            strings_left = df1[left_on].tolist()

        # -------------------------
        # 5) Carregar modelo e inferir embeddings
        # -------------------------
        if isinstance(model, str): 
            if openai_key is None:
                print(f"Carregando modelo {model} via linktransformer.load_model...")
                model = load_model(model)

        print("Inferindo embeddings para df_query (df1)...")
        embeddings1 = infer_embeddings(
            strings_left,
            model,
            batch_size=batch_size,
            openai_key=openai_key,
            return_numpy=True,
        )

        print("Inferindo embeddings para df_base (df2)...")
        embeddings2 = infer_embeddings(
            strings_right,
            model,
            batch_size=batch_size,
            openai_key=openai_key,
            return_numpy=True,
        )

        # Garantir shape 2D
        if embeddings1.ndim == 1:
            embeddings1 = np.expand_dims(embeddings1, axis=0)
        if embeddings2.ndim == 1:
            embeddings2 = np.expand_dims(embeddings2, axis=0)

        # Normaliza embeddings -> cosine / dot_product-friendly
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

        print(f"embeddings1 shape: {embeddings1.shape}")
        print(f"embeddings2 shape: {embeddings2.shape}")

        # ----------------------------------------
        # 6) (Opcional) salvar embeddings em .npy
        #    para outros scripts (FAISS, SVS, HNSW, ScaNN, etc.)
        # ----------------------------------------
        emb_dir = os.path.join(THIS_DIR, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)

        safe_model = str(modelo).replace(os.sep, "_")
        
        if os.path.altsep:
            safe_model = safe_model.replace(os.path.altsep, "_")


        np.save(os.path.join(f"{data_dir}/embeddings_base_{safe_model}.npy"), embeddings1.astype(np.float32))
        np.save(os.path.join(f"{data_dir}/embeddings_query_{safe_model}.npy"), embeddings2.astype(np.float32))

        print(f"Embeddings salvos em: {emb_dir}")
            
    
if __name__ == "__main__":

    # Gera os embeddings da base de dados (Vai gerar 4 embeddings, um para cada modelo)
    build_embedding()

    # Roda os processamentos já com os embeddings (Vai executar para cada embedding e para cada modelo, vamos utilizar apenas o ScaNN e o Linktransformer)
    main()