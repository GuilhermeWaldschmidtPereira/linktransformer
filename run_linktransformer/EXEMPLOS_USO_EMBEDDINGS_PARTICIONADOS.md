# Exemplos de Uso - Geração de Embeddings Particionados

## Exemplo 1: Seu Comando Atual (Atualizado)

Se você está usando:

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

Isso processará a base em partições de 100k linhas e mesclará automaticamente ao final.

---

## Exemplo 2: Processamento Sem Mesclar (Economia de Disco)

Se você quer economizar espaço durante o processamento:

```bash
HOST_DATA_DIR="$(pwd)/pasta_aux" nohup ./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  --no-merge-partitions \
  > embeddings_base_pasta_aux.log 2>&1 &
```

Neste caso, você terá:
- `pasta_aux/embeddings_partitions/modelo_name/partition_*.npy` (todos os chunks)
- Sem os arquivos `.npy` mesclados

---

## Exemplo 3: Processamento Local (Sem Podman)

```bash
# Ativar venv se necessário
source ~/venv/bin/activate

# Executar com particionamento
python3 ./run_linktransformer/run_embeddings_partitioned.py \
  --base-path pasta_aux/base.csv \
  --output-dir pasta_aux \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  --merge-partitions \
  --manifest-path pasta_aux/embeddings_manifest.json
```

---

## Exemplo 4: Processamento Customizado

### Aumentar Tamanho de Partição (menos arquivos)

```bash
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  --partition-size 500000  # 500k linhas por partição
```

### Diminuir Tamanho de Partição (menor memória)

```bash
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  --partition-size 50000   # 50k linhas por partição (menor consumo)
```

### Processar Modelo Específico

```bash
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario
```

### Múltiplos Modelos Selecionados

```bash
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --model intfloat/multilingual-e5-large \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario
```

---

## Exemplo 5: Base e Query Juntas

```bash
HOST_DATA_DIR="$(pwd)/pasta_aux" ./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode both \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  --right-on uf municipio logradouro numero complemento localidade setor_censitario
```

---

## Exemplo 6: Mesclar Partições Manualmente Depois

Se você processou sem `--merge-partitions` e quer mesclar depois:

```bash
# Mesclar base
python3 ./run_linktransformer/merge_partitions.py \
  pasta_aux/embeddings_partitions/sentence-transformers_all-MiniLM-L6-v2 \
  --side base \
  --verbose

# Mesclar query
python3 ./run_linktransformer/merge_partitions.py \
  pasta_aux/embeddings_partitions/sentence-transformers_all-MiniLM-L6-v2 \
  --side query \
  --verbose
```

---

## Monitoramento Durante Execução

### 1. Ver log em tempo real

```bash
tail -f embeddings_base_pasta_aux.log
```

### 2. Ver progresso de arquivo

```bash
# Terminal separado: atualizar a cada 5 segundos
watch -n 5 'ls -lh pasta_aux/embeddings_partitions/*/partition_*.npy | wc -l && du -sh pasta_aux/'
```

### 3. Ver uso de recurso do Podman

```bash
podman stats
```

---

## Estrutura de Saída Esperada

Após o processamento com `--merge-partitions`:

```
pasta_aux/
├── base.csv                                    # Input
├── embeddings_partitions/                      # Partições
│   ├── sentence-transformers_all-MiniLM-L6-v2/
│   │   ├── partition_000000_base.npy          # 100k linhas
│   │   ├── partition_000001_base.npy          # 100k linhas
│   │   ├── partition_000002_base.npy          # Resto
│   │   └── ... (outros modelos)
│   └── ... (outros modelos)
├── embeddings_base_sentence-transformers_all-MiniLM-L6-v2.npy   # ✓ Mesclado
├── embeddings_base_sentence-transformers_all-mpnet-base-v2.npy
├── embeddings_base_intfloat_multilingual-e5-large.npy
├── embeddings_base_neuralmind_bert-large-portuguese-cased.npy
└── embeddings_manifest.json                   # Metadados
```

---

## Verificar Resultado

```python
import json
import numpy as np

# 1. Verificar manifest
with open('pasta_aux/embeddings_manifest.json') as f:
    manifest = json.load(f)
    for entry in manifest:
        print(f"Modelo: {entry['model']}")
        print(f"  Partições: {entry.get('num_partitions', 'N/A')}")
        print(f"  Total de linhas: {entry.get('total_rows', 'N/A'):,}")
        print(f"  Tempo: {entry['elapsed_seconds']:.1f}s")
        print()

# 2. Verificar arquivo mesclado
embeddings = np.load('pasta_aux/embeddings_base_sentence-transformers_all-MiniLM-L6-v2.npy')
print(f"Embeddings shape: {embeddings.shape}")
print(f"Dtype: {embeddings.dtype}")
```

---

## Troubleshooting

### Erro: "Nenhuma partição encontrada"

Se você tentar mesclar e receber erro, verifique:

```bash
ls -la pasta_aux/embeddings_partitions/*/partition_*.npy
```

Deve haver arquivos `.npy` para cada partição.

### Erro: "Out of Memory"

Se durante o processamento receber erro de memória:

1. Reduzir batch size:
   ```bash
   ./run_linktransformer/run_embeddings_partitioned_podman.sh \
     --batch-size 32
   ```

2. Reduzir tamanho de partição:
   ```bash
   ./run_linktransformer/run_embeddings_partitioned_podman.sh \
     --partition-size 50000
   ```

3. Processar um modelo por vez:
   ```bash
   ./run_linktransformer/run_embeddings_partitioned_podman.sh \
     --model sentence-transformers/all-MiniLM-L6-v2
   ```

### Espaço em Disco Cheio

Se o disco ficar cheio durante processamento:

1. **Opção A:** Não usar `--merge-partitions`
   - Economiza 2x espaço (não precisa dos arquivos finais + partições)
   - Você mantém apenas as partições

2. **Opção B:** Processar modelos um por vez
   - Libera espaço entre modelos
   - Menos carga na rede/IO

3. **Opção C:** Limpar partições antigas
   ```bash
   rm -rf pasta_aux/embeddings_partitions/
   ```
   Depois mescle novamente com `merge_partitions.py`
