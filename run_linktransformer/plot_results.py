import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_results(results_df, output_path):
    """
    Plota gráficos de barras para tempo de indexação e busca a partir do DataFrame de resultados.

    Args:
        results_df (DataFrame): DataFrame contendo os resultados das execuções.
        output_path (str): Caminho para salvar os gráficos gerados.
    """
    sns.set(style="whitegrid")

    # Gráfico de Tempo de Indexação
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="metodo", y="index_time", data=results_df[["metodo", "index_time"]].drop_duplicates())
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    plt.title("Tempo de Indexação por Método")
    plt.ylabel("Tempo de Indexação (segundos)")
    plt.xlabel("Método")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_path}/index_time.png")
    plt.close()

    # Gráfico de Tempo de Busca
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="metodo", y="search_time", data=results_df[["metodo", "search_time"]].drop_duplicates())
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    plt.title("Tempo de Busca por Método")
    plt.ylabel("Tempo de Busca (segundos)")
    plt.xlabel("Método")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_path}/search_time.png")
    plt.close()

    # Gráfico de Uso de Memória
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="metodo", y="avg_mem_used_search_MB", data=results_df[["metodo", "avg_mem_used_search_MB"]].drop_duplicates())
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    plt.title("Uso de Memória por Método")
    plt.ylabel("Uso de Memória (MB)")
    plt.xlabel("Método")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_path}/memory_usage_search.png")
    plt.close()

    # Gráfico de Uso de Memória
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="metodo", y="mem_used_indexation_MB", data=results_df[["metodo", "mem_used_indexation_MB"]].drop_duplicates())
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    plt.title("Uso de Memória por Método")
    plt.ylabel("Uso de Memória (MB)")
    plt.xlabel("Método")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_path}/memory_usage_indexation.png")
    plt.close()

results_df = pd.read_csv("resultados.csv")
plot_results(results_df, output_path="results_plots")