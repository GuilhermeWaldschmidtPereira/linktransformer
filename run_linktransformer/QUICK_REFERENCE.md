# ⚡ Quick Reference - Embeddings Particionados

## Mudança Rápida

```bash
# ANTES
./run_linktransformer/run_embeddings_podman.sh --mode base --left-on uf municipio ...

# DEPOIS
./run_linktransformer/run_embeddings_partitioned_podman.sh --mode base --left-on uf municipio ...
```

**Só mude o nome do script!** Tudo mais é igual.

---

## Commandos Mais Comuns

### Seu comando atual (atualizado)
```bash
HOST_DATA_DIR="$(pwd)/pasta_aux" nohup ./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  > embeddings_base_pasta_aux.log 2>&1 &
```

### Local sem Podman
```bash
python3 run_linktransformer/run_embeddings_partitioned.py \
  --base-path pasta_aux/base.csv \
  --output-dir pasta_aux \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario
```

### Economizar espaço (sem mesclar)
```bash
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --no-merge-partitions \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario
```

### Baixa memória (partições menores)
```bash
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --partition-size 50000 \
  --batch-size 32 \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario
```

### Um modelo específico
```bash
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario
```

### Mesclar partições depois
```bash
python3 run_linktransformer/merge_partitions.py \
  pasta_aux/embeddings_partitions/modelo_name \
  --side base \
  --verbose
```

### Teste rápido
```bash
./run_linktransformer/test_embeddings_partitioned.sh
```

---

## Estrutura de Saída

```
pasta_aux/
├── embeddings_partitions/
│   ├── sentence-transformers_all-MiniLM-L6-v2/
│   │   ├── partition_000000_base.npy    ← 100k linhas
│   │   ├── partition_000001_base.npy    ← 100k linhas
│   │   └── partition_000002_base.npy    ← resto
│   └── ... (outros modelos)
├── embeddings_base_*.npy                 ← Mesclados (se --merge-partitions)
├── embeddings_query_*.npy                ← Mesclados (se --merge-partitions)
└── embeddings_manifest.json              ← Metadados
```

---

## Carregar Embeddings (Python)

```python
import numpy as np

# Arquivo mesclado (padrão)
embeddings = np.load('pasta_aux/embeddings_base_sentence-transformers_all-MiniLM-L6-v2.npy')
print(embeddings.shape)  # (N, 384)
```

---

## Monitorar Execução

```bash
# Terminal 1: Ver log
tail -f embeddings_base_pasta_aux.log

# Terminal 2: Ver progresso
watch -n 5 'du -sh pasta_aux/ && ls pasta_aux/embeddings_partitions/*/partition_*.npy | wc -l'

# Terminal 3: Ver recursos
podman stats
```

---

## Opções Principais

| Flag | Padrão | O quê |
|------|--------|-------|
| `--partition-size` | 100000 | Linhas por partição |
| `--batch-size` | 128 | Batch para SentenceTransformer |
| `--merge-partitions` | True (Podman) | Mesclar em arquivo único |
| `--mode` | both | base / query / both |
| `--model` | Todos 4 | Modelo específico (repetir para vários) |

---

## Resolução de Problemas

| Problema | Solução |
|----------|---------|
| **Out of Memory** | `--batch-size 32 --partition-size 50000` |
| **Disco Cheio** | `--no-merge-partitions` |
| **Lento** | `--batch-size 256 --partition-size 200000` |
| **Partições Incompletas** | Reexecutar (sobrescreve) |

---

## Arquivos Criados/Alterados

- ✨ `run_embeddings_partitioned.py` - Script principal
- ✨ `run_embeddings_partitioned_podman.sh` - Wrapper Podman
- ✨ `merge_partitions.py` - Mesclar manualmente
- ✨ `test_embeddings_partitioned.sh` - Teste rápido
- 🔄 `Containerfile.embeddings` - Entrypoint alterado

---

## Documentação Completa

- 📖 [`EMBEDDINGS_PARTICIONADOS.md`](../EMBEDDINGS_PARTICIONADOS.md)
- 📖 [`EXEMPLOS_USO_EMBEDDINGS_PARTICIONADOS.md`](./EXEMPLOS_USO_EMBEDDINGS_PARTICIONADOS.md)
- 📖 [`GUIA_INTEGRACAO_EMBEDDINGS_PARTICIONADOS.md`](./GUIA_INTEGRACAO_EMBEDDINGS_PARTICIONADOS.md)
- 📖 [`README_ALTERACOES_EMBEDDINGS.md`](./README_ALTERACOES_EMBEDDINGS.md)

---

**TL;DR:** Mude `run_embeddings_podman.sh` → `run_embeddings_partitioned_podman.sh` e pronto!
