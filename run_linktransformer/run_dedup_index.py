import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
import time

# ==========================================
# Configuração do Ambiente e Imports
# ==========================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
DATA_DIR = os.path.join(REPO_ROOT, "data/dedup_task")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from linktransformer.infer import merge_knn, merge_knn_nmslib       # Wrapper para FAISS

def deduplicate_with_index(
    df: pd.DataFrame,
    on: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    method: str = "scann", 
    k: int = 3,           # k=3 é uma boa heurística (1=self, 2 e 3=candidatos)
    threshold: float = 0.6,
    batch_size: int = 128
) -> pd.DataFrame:
    """
    Executa deduplicação via Index-Based Blocking (ScaNN/FAISS) + Componentes Conexos.
    Retorna o DataFrame limpo e imprime estatísticas para a dissertação.
    """
    
    print(f"\n[INFO] Iniciando Deduplicação | Método: {method.upper()} | Threshold: {threshold}")
    print(f"[INFO] Base Original: {len(df)} registros")
    
    start_time = time.time()

    # 1. Garantir ID único para rastreamento no grafo
    # Usamos o índice resetado para garantir integridade 0..N-1
    df = df.reset_index(drop=True)
    df['temp_id'] = df.index 

    # 2. Self-Join via k-NN (Blocking Aproximado)
    print(f"[STEP 1] Executando busca de vizinhos (k={k})...")
    
    if method.lower() == "scann":
        # ScaNN: Otimizado para produto escalar (MIPS)
        df_pairs = merge_knn_scann(
            df1=df, 
            df2=df, 
            on=on, 
            model=model_name, 
            k=k, 
            suffixes=('_l', '_r'),
            batch_size=batch_size,
            drop_sim_threshold=threshold
        )
    else:
        # FAISS: IndexFlatIP (Produto Interno exato ou HNSW se configurado)
        df_pairs = merge_knn_nmslib(
            df1=df, 
            df2=df, 
            on=on, 
            model=model_name, 
            k=k, 
            suffixes=('_l', '_r'),
            batch_size=batch_size,
            drop_sim_threshold=threshold
        )
        
        df_pairs.to_csv("debug_faiss_pairs.csv", index=False)  # DEBUG
    
    # 3. Construção do Grafo de Similaridade
    # Filtramos auto-loops (nó conectado a ele mesmo)
    print("[STEP 2] Construindo grafo de similaridade e resolvendo entidades...")
    
    # O output do merge_knn geralmente expande as linhas. Precisamos dos IDs.
    # Assumindo que o merge preservou colunas ou índices, mas para garantir,
    # vamos confiar que se df1 e df2 eram o mesmo DF, os índices alinham se usarmos 
    # a lógica de 'left_index' e 'right_index' que existe dentro do merge_knn.
    # Mas como merge_knn retorna um DF novo, precisamos recuperar quem é quem.
    
    # HACK: Se o merge_knn não retornar as colunas 'temp_id_l' e 'temp_id_r' explicitamente,
    # precisamos inferir ou garantir que o merge_knn as traga. 
    # No código atual do linktransformer, ele retorna colunas com sufixos.
    
    col_l = f"temp_id_l"
    col_r = f"temp_id_r"
    
    # Verifica se as colunas existem (ajuste conforme sufixos passados)
    # Se não existirem, usamos index se o merge preservou
    edges = []
    if col_l in df_pairs.columns and col_r in df_pairs.columns:
        # Vetorização para performance (evita iterrows em bases grandes)
        pairs = df_pairs[[col_l, col_r]].values
        # Remove self-loops (col_l == col_r)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        edges = list(map(tuple, pairs))
    else:
        # Fallback: se o merge_knn usar sufixos diferentes
        # Tente identificar colunas de ID
        print("[WARN] Colunas de ID não identificadas automaticamente. Verifique os sufixos.")
        # (Adicione lógica de fallback aqui se necessário)
        pass
    
    print("Edges de similaridade encontrados:", edges)

    # 4. Transitive Closure (Componentes Conexos)
    G = nx.Graph()
    G.add_edges_from(edges)
    
    clusters = list(nx.connected_components(G))
    print(f"[STATS] Clusters de duplicatas encontrados: {len(clusters)}")
    
    print(f"Clusters Formados: {clusters}")
    
    # ---------------------------------------------
    # 5. Canonicalização / Escolha do registro oficial
    # ---------------------------------------------
    ids_to_drop = set()

    for cluster in clusters:
        # cluster com apenas 1 registro NÃO é duplicata
        if len(cluster) == 1:
            continue
        
        # Ordena os IDs para sempre manter o primeiro
        sorted_cluster = sorted(cluster)
        winner = sorted_cluster[0]        # mantém
        losers = sorted_cluster[1:]       # remove
        
        ids_to_drop.update(losers)

    print(f"[STATS] Registros redundantes removidos: {len(ids_to_drop)}")

    # Aplicar dedup
    df_dedup = df[~df['temp_id'].isin(ids_to_drop)].drop(columns=['temp_id']).copy()
    
    return df_dedup


def main():
    if not os.path.exists(DATA_DIR):
        print(f"Diretório {DATA_DIR} não encontrado.")
        return

    # Carregar dados
    input_path = os.path.join(DATA_DIR, "dup_data.csv")
    output_path = os.path.join(DATA_DIR, "dup_data_deduped.csv")
    
    df = pd.read_csv(input_path)
    
    # Executar Dedup
    # Nota: Threshold alto (0.85/0.90) evita falsos positivos em dados textuais ruidosos
    df_clean = deduplicate_with_index(
        df, 
        on=['nome', 'sobrenome'], 
        method='other', 
        k=3,            
        threshold=0.7
    )
    
    # Salvar
    df_clean.to_csv(output_path, index=False)
    print(f"\nArquivo salvo em: {output_path}")
    print("Amostra:")
    print(df_clean[['id', 'nome', 'sobrenome']].head())

if __name__ == "__main__":
    main()