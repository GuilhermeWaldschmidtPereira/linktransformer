import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# CONFIG
# ==========================
CSV_PATH = "resultados.csv"
OUT_DIR = "figs_triplo_padrao"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================
# LOAD
# ==========================
df = pd.read_csv(CSV_PATH)

num_cols = [
    "total_time",
    "mem_used_indexation_MB",
    "matches",
    "k"
]

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["k"] = df["k"].astype(int)

# ==========================
# PADRONIZAÇÃO EMBEDDINGS
# ==========================
df["Embedding_Label"] = df["modelo_embedding"].astype(str).str.lower().apply(
    lambda x:
        "Modelo 1" if "intfloat/multilingual-e5-large" in x else
        "Modelo 2" if "neuralmind/bert-large-portuguese-cased" in x else
        "Modelo 3" if "sentence-transformers/all-minilm-l6-v2" in x else
        "Modelo 4" if "sentence-transformers/all-mpnet-base-v2" in x else
        None
)

df = df[df["Embedding_Label"].notna()]

ordem_modelos = ["Modelo 1", "Modelo 2", "Modelo 3", "Modelo 4"]
metodos = sorted(df["metodo"].unique())

# ==========================
# FUNÇÃO DE BARRAS AGRUPADAS
# ==========================
def grouped_bar(metric, ylabel, titulo_prefixo, nome_arquivo, k_value=4):

    sub = df[df["k"] == k_value]

    x = np.arange(len(ordem_modelos))
    width = 0.18

    plt.figure()

    for i, metodo in enumerate(metodos):
        valores = []
        for modelo in ordem_modelos:
            row = sub[(sub["Embedding_Label"] == modelo) & (sub["metodo"] == metodo)]
            valores.append(float(row.iloc[0][metric]) if len(row) else 0.0)

        plt.bar(x + i*width, valores, width=width, label=metodo)

    plt.xticks(x + width*(len(metodos)-1)/2, ordem_modelos)
    plt.xlabel("Modelo de Embedding")
    plt.ylabel(ylabel)
    plt.title(titulo_prefixo + f" (k={k_value})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, nome_arquivo), dpi=300, bbox_inches="tight")
    plt.close()

# ==========================
# GERAÇÃO DOS TRÊS GRÁFICOS
# ==========================
for k in sorted(df["k"].unique()):
    grouped_bar(
        metric="total_time",
        ylabel="Tempo total (s)",
        titulo_prefixo="(a) Tempo total",
        nome_arquivo=f"fig_a_tempo_k{k}.png",
        k_value=k
    )

    grouped_bar(
        metric="mem_used_indexation_MB",
        ylabel="Memória (MB)",
        titulo_prefixo="(b) Memória",
        nome_arquivo=f"fig_b_memoria_k{k}.png",
        k_value=k
    )

    grouped_bar(
        metric="matches",
        ylabel="Número de acertos",
        titulo_prefixo="(c) Acertos",
        nome_arquivo=f"fig_c_acertos_k{k}.png",
        k_value=k
    )

print("Gráficos gerados em:", OUT_DIR)