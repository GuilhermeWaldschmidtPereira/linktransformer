# my_npy_demo/run_from_source.py

import os
import sys
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

    df_base = pd.read_csv(base_path)
    df_query = pd.read_csv(query_path)

    print("\nBase (endereços corretos):")
    print(df_base.head())
    print("\nQuery (endereços com erros):")
    print(df_query.head())

    # ==========================================
    # 3) Chamar merge_knn diretamente
    #    (sem usar 'import linktransformer as lt')
    # ==========================================
    # ATENÇÃO: assinatura do merge_knn em infer.py (versão atual upstream)
    # merge_knn(df1, df2, merge_type='1:1', on=None, model='all-MiniLM-L6-v2',
    #           left_on=None, right_on=None, k=1, suffixes=('_x', '_y'),
    #           use_gpu=False, batch_size=128, openai_key=None,
    #           drop_sim_threshold=None)
    #
    # Aqui vou deixar merge_type no default ('1:1') e só setar o essencial.
    merged = merge_knn_scann(
        df1=df_base,
        df2=df_query,
        on=None,
        model="sentence-transformers/all-MiniLM-L6-v2",
        k=1,
        # você pode ajustar batch_size se quiser
        batch_size=64,
    )

    print("\nResultado do merge_knn (endereços):")
    # As colunas exatas podem variar dependendo da versão; ajuste se precisar
    cols_to_show = [c for c in merged.columns if any(
        x in c for x in ["id", "name", "score"]
    )]
    print(merged[cols_to_show].head(15))


if __name__ == "__main__":
    main()
