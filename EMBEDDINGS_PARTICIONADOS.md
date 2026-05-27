# Geração de Embeddings Particionados

## Visão Geral

O novo sistema de geração de embeddings processa os dados em **partições de 100k linhas** para melhor gerenciamento de memória e escalabilidade.

### Características

- ✅ Processa CSV em partições automáticas (padrão: 100k linhas)
- ✅ Salva cada partição em arquivo `.npy` separado
- ✅ Organiza partições em pastas por modelo
- ✅ Opção para mesclar partições em um único arquivo
- ✅ Registra informações de timing e memória em JSON manifest
- ✅ Compatível com execução via Podman/Docker

## Estrutura de Saída

```
data/
├── embeddings_partitions/              # Pasta raiz de partições
│   ├── sentence-transformers_all-MiniLM-L6-v2/
│   │   ├── partition_000000_base.npy
│   │   ├── partition_000001_base.npy
│   │   ├── partition_000000_query.npy
│   │   └── partition_000001_query.npy
│   ├── sentence-transformers_all-mpnet-base-v2/
│   │   └── ...
│   └── ...
├── embeddings_base_sentence-transformers_all-MiniLM-L6-v2.npy   # (Se --merge-partitions)
├── embeddings_query_sentence-transformers_all-MiniLM-L6-v2.npy  # (Se --merge-partitions)
└── embeddings_manifest.json              # Metadados da execução
```

## Uso

### 1. Execução Local (Python direto)

```bash
# Processar base com particionamento
python3 run_linktransformer/run_embeddings_partitioned.py \
  --mode base \
  --base-path pasta_aux/base.csv \
  --output-dir pasta_aux \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  --merge-partitions

# Processar base e query
python3 run_linktransformer/run_embeddings_partitioned.py \
  --mode both \
  --base-path pasta_aux/base.csv \
  --query-path pasta_aux/query.csv \
  --output-dir pasta_aux \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  --right-on uf municipio logradouro numero complemento localidade setor_censitario \
  --merge-partitions
```

### 2. Execução com Podman (recomendado)

O comando que você usava anteriormente agora suporta particionamento automaticamente:

```bash
HOST_DATA_DIR="$(pwd)/pasta_aux" nohup ./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  > embeddings_base_pasta_aux.log 2>&1 &
```

### 3. Sem Mesclar Partições (apenas para análise)

Se você não precisa dos arquivos mesclados (útil se só quer testar):

```bash
python3 run_linktransformer/run_embeddings_partitioned.py \
  --mode base \
  --base-path pasta_aux/base.csv \
  --output-dir pasta_aux \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario
  # Sem --merge-partitions
```

Neste caso, os dados estarão apenas em `pasta_aux/embeddings_partitions/`

## Opções de Linha de Comando

| Opção | Padrão | Descrição |
|-------|--------|-----------|
| `--base-path` | `data/base.csv` | Caminho do CSV base |
| `--query-path` | `data/query.csv` | Caminho do CSV query |
| `--output-dir` | `data/` | Diretório onde salvar embeddings |
| `--mode` | `both` | `base`, `query`, ou `both` |
| `--left-on` | - | Colunas da base para embeddings |
| `--right-on` | - | Colunas da query para embeddings |
| `--model` | Todos 4 padrão | Modelo específico (pode repetir) |
| `--batch-size` | 128 | Tamanho do batch para SentenceTransformer |
| `--partition-size` | 100000 | Linhas por partição |
| `--merge-partitions` | False | Mesclar partições em arquivo único |
| `--manifest-path` | - | Onde salvar manifest JSON |

## Manifest JSON

Exemplo de saída em `embeddings_manifest.json`:

```json
[
  {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "safe_model_name": "sentence-transformers_all-MiniLM-L6-v2",
    "mode": "base",
    "partition_size": 100000,
    "num_partitions": 3,
    "total_rows": 250000,
    "partition_dir": "/path/to/data/embeddings_partitions/sentence-transformers_all-MiniLM-L6-v2",
    "merged_path": "/path/to/data/embeddings_base_sentence-transformers_all-MiniLM-L6-v2.npy",
    "elapsed_seconds": 1234.56,
    "elapsed_minutes": 20.58,
    "memory_start_mb": 512.0,
    "memory_end_mb": 2048.0,
    "memory_peak_mb": 2048.0,
    "memory_delta_mb": 1536.0
  }
]
```

## Carregando os Resultados

### Com Partições Mescladas

```python
import numpy as np

# Carregar como antes
embeddings_base = np.load('pasta_aux/embeddings_base_sentence-transformers_all-MiniLM-L6-v2.npy')
print(f"Shape: {embeddings_base.shape}")  # (num_linhas, dimensão)
```

### Com Partições Separadas

```python
import numpy as np
from pathlib import Path

partition_dir = Path('pasta_aux/embeddings_partitions/sentence-transformers_all-MiniLM-L6-v2')

# Listar todas as partições
partitions = sorted(partition_dir.glob('partition_*_base.npy'))

# Carregar uma partição específica
part_0 = np.load(partitions[0])

# Carregar todas e mesclar manualmente
all_parts = [np.load(p) for p in partitions]
embeddings = np.concatenate(all_parts, axis=0)
```

## Migrando do Comando Anterior

Se você estava usando:

```bash
HOST_DATA_DIR="$(pwd)/pasta_aux" nohup ./run_linktransformer/run_embeddings_podman.sh \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  > embeddings_base_pasta_aux.log 2>&1 &
```

**Mude para:**

```bash
HOST_DATA_DIR="$(pwd)/pasta_aux" nohup ./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  > embeddings_base_pasta_aux.log 2>&1 &
```

A única mudança é o nome do script (`run_embeddings_partitioned_podman.sh`).

## Monitorar Progresso

```bash
# Terminal 1: Acompanhar o log
tail -f embeddings_base_pasta_aux.log

# Terminal 2: Ver espaço em disco
watch -n 5 'du -sh pasta_aux/'

# Terminal 3: Ver uso de memória do podman
podman stats
```

## Troubleshooting

### Erro de Memória
Se receber erro de memória durante particionamento, tente:
- Reduzir `--batch-size` (padrão 128 → tente 32 ou 64)
- Reduzir `--partition-size` (padrão 100k → tente 50k)

### Espaço em Disco Insuficiente
Se a pasta crescer muito rápido:
- Não use `--merge-partitions` (economiza 2x espaço)
- Processe um modelo por vez usando `--model <modelo-específico>`
- Remova partições antigas manualmente após validação

### Partições Incompletas
Se o processo for interrompido, você pode:
- Retomar manualmente processando de novo (os arquivos já existentes serão sobrescritos)
- Ou limpar `embeddings_partitions/` e começar do zero
