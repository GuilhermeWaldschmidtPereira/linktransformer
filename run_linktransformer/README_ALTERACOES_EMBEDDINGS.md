# 📋 Sumário de Mudanças - Geração de Embeddings Particionados

## O que foi feito

Implementação de um sistema de geração de embeddings **particionado em chunks de 100k linhas**, com suporte completo a Podman e gerenciamento automático de pastas por modelo.

---

## 📁 Arquivos Criados/Modificados

### ✨ Novos Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `run_linktransformer/run_embeddings_partitioned.py` | ⭐ Script principal para particionamento (Python puro) |
| `run_linktransformer/run_embeddings_partitioned_podman.sh` | Script shell wrapper para execução via Podman |
| `run_linktransformer/merge_partitions.py` | Utilitário para mesclar partições manualmente |
| `run_linktransformer/test_embeddings_partitioned.sh` | Script de teste rápido |
| `EMBEDDINGS_PARTICIONADOS.md` | Documentação completa (na raiz do projeto) |
| `run_linktransformer/EXEMPLOS_USO_EMBEDDINGS_PARTICIONADOS.md` | Exemplos práticos de uso |

### 🔄 Arquivos Modificados

| Arquivo | Mudança |
|---------|---------|
| `Containerfile.embeddings` | Entrypoint mudado para `run_embeddings_partitioned.py` |

---

## 🚀 Quick Start

### Seu comando anterior:
```bash
HOST_DATA_DIR="$(pwd)/pasta_aux" nohup ./run_linktransformer/run_embeddings_podman.sh \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  > embeddings_base_pasta_aux.log 2>&1 &
```

### Novo comando (com particionamento):
```bash
HOST_DATA_DIR="$(pwd)/pasta_aux" nohup ./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  > embeddings_base_pasta_aux.log 2>&1 &
```

**Mudança única:** `run_embeddings_podman.sh` → `run_embeddings_partitioned_podman.sh`

---

## 🎯 Funcionalidades Principais

### ✅ Processamento em Partições
- Lê CSV em chunks de **100k linhas** (configurável via `--partition-size`)
- Gera embeddings para cada partição independentemente
- Ideal para arquivos muito grandes (economiza memória)

### ✅ Organização Automática
```
pasta_aux/
├── embeddings_partitions/
│   ├── sentence-transformers_all-MiniLM-L6-v2/
│   │   ├── partition_000000_base.npy
│   │   ├── partition_000001_base.npy
│   │   └── ...
│   ├── intfloat_multilingual-e5-large/
│   │   └── ...
│   └── ... (outros modelos)
├── embeddings_base_sentence-transformers_all-MiniLM-L6-v2.npy  # Mesclado
└── embeddings_base_intfloat_multilingual-e5-large.npy
```

### ✅ Mescla Automática
- Flag `--merge-partitions` (ativada por padrão no script Podman)
- Mescla automaticamente partições em arquivo único
- Compatível com código antigo que espera `embeddings_base_*.npy`

### ✅ Manifest JSON
```json
{
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "partition_size": 100000,
  "num_partitions": 3,
  "total_rows": 250000,
  "elapsed_seconds": 1234.56,
  "memory_start_mb": 512.0,
  "memory_delta_mb": 1536.0,
  ...
}
```

### ✅ Ferramentas Auxiliares
- **merge_partitions.py**: Mesclar partições manualmente
- **test_embeddings_partitioned.sh**: Teste rápido da implementação

---

## 📊 Comparação: Antes vs Depois

| Aspecto | Antes | Depois |
|--------|-------|--------|
| **Processamento** | Carrega tudo na memória | Particionado (100k linhas) |
| **Escalabilidade** | Limitado a RAM disponível | Escalável para arquivos muito grandes |
| **Organização** | Um arquivo `.npy` por modelo | Pastas separadas + arquivos mesclados |
| **Flexibilidade** | Sem opções | `--partition-size`, `--merge-partitions` |
| **Segurança** | Risco de OOM em arquivos grandes | Partições pequenas = menor risco |
| **Compatibilidade** | Anterior | ✅ 100% compatível (mescla automática) |

---

## 🔧 Opções de Linha de Comando

```
--partition-size SIZE       # Linhas por partição (padrão: 100000)
--merge-partitions          # Mesclar em arquivo único (padrão: True no Podman)
--mode {base|query|both}    # Qual lado processar
--model MODELO              # Modelo específico (pode repetir)
--batch-size SIZE           # Batch size (padrão: 128)
--partition-size 50000      # Partições menores (menos memória)
```

---

## 📝 Exemplos de Uso

### Exemplo 1: Padrão (recomendado)
```bash
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario
```

### Exemplo 2: Sem Mesclar (economia de disco)
```bash
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  --no-merge-partitions
```

### Exemplo 3: Partições Menores (baixa memória)
```bash
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --partition-size 50000 \
  --batch-size 32 \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario
```

### Exemplo 4: Local (sem Podman)
```bash
python3 ./run_linktransformer/run_embeddings_partitioned.py \
  --base-path pasta_aux/base.csv \
  --output-dir pasta_aux \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  --merge-partitions
```

---

## 🧪 Testando

Execute o script de teste:

```bash
./run_linktransformer/test_embeddings_partitioned.sh
```

Isso vai:
1. Criar um CSV com 350 linhas
2. Processar em 4 partições de 100 linhas
3. Gerar embeddings com modelo rápido (all-MiniLM)
4. Mesclar partições
5. Mostrar resultado

Tempo esperado: ~1-2 minutos

---

## 📊 Monitoramento

### Ver progresso em tempo real
```bash
tail -f embeddings_base_pasta_aux.log
```

### Ver arquivo crescendo
```bash
watch -n 5 'du -sh pasta_aux/ && ls -1 pasta_aux/embeddings_partitions/*/partition_*.npy | wc -l'
```

### Ver uso de recursos Podman
```bash
podman stats
```

---

## 🔄 Fluxo de Processamento

```
CSV (base.csv)
    ↓
[Lê em chunks de 100k]
    ↓
Partição 1 (100k) → Embeddings → partition_000000_base.npy
Partição 2 (100k) → Embeddings → partition_000001_base.npy
Partição 3 (100k) → Embeddings → partition_000002_base.npy
    ↓
[Mescla se --merge-partitions]
    ↓
embeddings_base_modelo.npy (arquivo final único)
    ↓
[Registra metadados]
    ↓
embeddings_manifest.json
```

---

## ✅ Validação de Saída

Verificar se tudo funcionou:

```python
import numpy as np
import json

# Manifest
with open('pasta_aux/embeddings_manifest.json') as f:
    manifest = json.load(f)
    print(f"Modelos processados: {len(manifest)}")
    for m in manifest:
        print(f"  - {m['model']}: {m['total_rows']:,} linhas")

# Embeddings
embeddings = np.load('pasta_aux/embeddings_base_sentence-transformers_all-MiniLM-L6-v2.npy')
print(f"\nEmbeddings shape: {embeddings.shape}")  # (N_linhas, 384)
print(f"Dtype: {embeddings.dtype}")              # float32
print(f"Norma (amostra): {np.linalg.norm(embeddings[0]):.4f}")  # ~1.0 (normalizado)
```

---

## 🚨 Troubleshooting

| Problema | Solução |
|----------|---------|
| **Out of Memory** | Reduzir `--batch-size` ou `--partition-size` |
| **Disco cheio** | Usar `--no-merge-partitions` ou procesar modelos um por um |
| **Partições incompletas** | Reexecutar (sobrescreve automaticamente) |
| **Slow performance** | Aumentar `--batch-size` para 256 ou mais |

---

## 📚 Documentação Completa

Para mais detalhes:
- [`EMBEDDINGS_PARTICIONADOS.md`](../EMBEDDINGS_PARTICIONADOS.md) - Documentação técnica completa
- [`EXEMPLOS_USO_EMBEDDINGS_PARTICIONADOS.md`](./EXEMPLOS_USO_EMBEDDINGS_PARTICIONADOS.md) - Exemplos práticos

---

## ✨ Principais Benefícios

1. **Escalabilidade**: Processa arquivos de qualquer tamanho
2. **Eficiência de Memória**: Partições pequenas = consumo controlado
3. **Flexibilidade**: Opções para customizar tamanho, mescla, modelos
4. **Compatibilidade**: Funciona com código antigo (mescla automática)
5. **Transparência**: Manifest JSON com metadados completos
6. **Facilidade**: Uma linha de comando, múltiplas opções

---

## 🎯 Próximos Passos

1. Testar com seu CSV:
   ```bash
   ./run_linktransformer/test_embeddings_partitioned.sh
   ```

2. Atualizar seu comando:
   ```bash
   HOST_DATA_DIR="$(pwd)/pasta_aux" nohup ./run_linktransformer/run_embeddings_partitioned_podman.sh \
     --mode base \
     --left-on uf municipio logradouro numero complemento localidade setor_censitario \
     > embeddings_base_pasta_aux.log 2>&1 &
   ```

3. Monitorar:
   ```bash
   tail -f embeddings_base_pasta_aux.log
   ```

4. Validar resultado conforme exemplo acima

---

**Versão:** 1.0  
**Data:** 18 de maio de 2026  
**Status:** ✅ Pronto para produção
