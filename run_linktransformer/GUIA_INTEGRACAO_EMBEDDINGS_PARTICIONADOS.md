# 🔗 Guia de Integração - Embeddings Particionados

## Para Código Existente

Se você tem código que carrega os embeddings da forma antiga, **não precisa alterar nada**!

### Código Antigo (continua funcionando)

```python
import numpy as np

# Isso ainda funciona!
embeddings = np.load('data/embeddings_base_sentence-transformers_all-MiniLM-L6-v2.npy')
print(f"Shape: {embeddings.shape}")
```

**Por quê?** O novo script automaticamente mescla as partições nesses arquivos (com `--merge-partitions`).

---

## Verificar se os Embeddings Foram Particionados

```python
import os
from pathlib import Path

# Verificar se existem partições
partition_dir = Path('data/embeddings_partitions')
if partition_dir.exists():
    print("✓ Embeddings particionados encontrados!")
    
    # Listar modelos processados
    for model_dir in partition_dir.iterdir():
        if model_dir.is_dir():
            partitions = list(model_dir.glob('partition_*.npy'))
            print(f"  {model_dir.name}: {len(partitions)} partições")
else:
    print("ℹ️ Sem partições (processamento antigo ou não executado)")
```

---

## Acessar Partições Específicas

Se você quer trabalhar diretamente com partições (útil para processamento paralelo):

```python
import numpy as np
from pathlib import Path
import json

# Ler manifest para informações
with open('data/embeddings_manifest.json') as f:
    manifest = json.load(f)
    
for entry in manifest:
    if entry['model'] == 'sentence-transformers/all-MiniLM-L6-v2':
        partition_dir = Path(entry['partition_dir'])
        num_partitions = entry['num_partitions']
        
        # Processar cada partição
        for i in range(num_partitions):
            partition_file = partition_dir / f'partition_{i:06d}_base.npy'
            partition_data = np.load(partition_file)
            print(f"Partição {i}: {partition_data.shape}")
```

---

## Atualizar Pipeline Antigo

Se seu pipeline usa o Containerfile antigo, atualize:

### Antes
```dockerfile
ENTRYPOINT ["python3", "run_linktransformer/run_embeddings.py"]
```

### Depois
```dockerfile
ENTRYPOINT ["python3", "run_linktransformer/run_embeddings_partitioned.py"]
```

**Nota:** O novo script aceita os mesmos argumentos, então seu command antigo continua funcionando!

---

## Atualizar Scripts Bash

### Script Antigo
```bash
#!/usr/bin/env bash
./run_linktransformer/run_embeddings_podman.sh \
  --mode base \
  --left-on uf municipio
```

### Script Novo
```bash
#!/usr/bin/env bash
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --left-on uf municipio
```

Tudo o mais permanece igual!

---

## Usar Partições em Produção

### Opção 1: Manter Partições (menos espaço)
```bash
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --left-on ... \
  --no-merge-partitions  # Economiza espaço
```

Seu código:
```python
from pathlib import Path
import numpy as np

def load_all_embeddings(model_name, side='base'):
    """Carrega todas as partições e mescla."""
    partition_dir = Path(f'data/embeddings_partitions/{model_name}')
    partitions = sorted(partition_dir.glob(f'partition_*_{side}.npy'))
    
    arrays = [np.load(p) for p in partitions]
    return np.concatenate(arrays, axis=0)

# Usar
embeddings = load_all_embeddings('sentence-transformers_all-MiniLM-L6-v2')
```

### Opção 2: Usar Partições Diretamente (muito eficiente)
```python
def get_partition_batch(model_name, partition_idx, side='base', batch_size=1000):
    """Carrega uma partição específica e retorna mini-batches."""
    from pathlib import Path
    import numpy as np
    
    partition_file = Path(f'data/embeddings_partitions/{model_name}/partition_{partition_idx:06d}_{side}.npy')
    embeddings = np.load(partition_file)
    
    # Retornar em batches
    for i in range(0, len(embeddings), batch_size):
        yield embeddings[i:i+batch_size]

# Usar
for mini_batch in get_partition_batch('sentence-transformers_all-MiniLM-L6-v2', 0):
    # Processar mini_batch
    pass
```

---

## Gerenciar Espaço em Disco

### Cenário 1: Disk space crítico
```bash
# Procesar apenas um modelo por vez
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --left-on ...

# Mesclar manualmente se precisar do arquivo final
python3 ./run_linktransformer/merge_partitions.py \
  data/embeddings_partitions/sentence-transformers_all-MiniLM-L6-v2 \
  --side base

# Repetir para próximo modelo
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --model sentence-transformers/all-mpnet-base-v2 \
  --left-on ...
```

### Cenário 2: Limpar partições antigas
```bash
# Manter apenas mesclados (remove partições)
rm -rf data/embeddings_partitions/

# Manter apenas partições (remove mesclados)
rm data/embeddings_base_*.npy data/embeddings_query_*.npy
```

---

## Monitorar em Produção

### Health Check Script
```bash
#!/usr/bin/env bash
# check_embeddings.sh

set -e

MANIFEST="data/embeddings_manifest.json"

if [[ ! -f "$MANIFEST" ]]; then
    echo "✗ Manifest não encontrado: $MANIFEST"
    exit 1
fi

echo "✓ Manifest encontrado"

# Verificar cada modelo
python3 << 'PYTHON'
import json
import numpy as np
from pathlib import Path

with open('data/embeddings_manifest.json') as f:
    manifest = json.load(f)

for entry in manifest:
    model = entry['safe_model_name']
    side = entry['mode']
    
    # Verificar partições
    partition_dir = Path(entry['partition_dir'])
    partitions = list(partition_dir.glob('partition_*.npy'))
    print(f"✓ {model}: {len(partitions)} partições")
    
    # Verificar arquivo mesclado se existir
    if 'merged_path' in entry:
        merged_path = entry['merged_path']
        if Path(merged_path).exists():
            emb = np.load(merged_path)
            print(f"  └─ Merged: {emb.shape} {emb.dtype}")
        else:
            print(f"  └─ Merged: FALTA {merged_path}")

print("\n✓ Verificação concluída")
PYTHON
```

Use em cron:
```bash
0 */6 * * * /home/gpereira/projeto_mestrado/run_linktransformer/check_embeddings.sh >> /var/log/embeddings_check.log 2>&1
```

---

## Recuperar de Falhas

### Se o processamento foi interrompido

```bash
# Verificar o que foi processado
ls -lh data/embeddings_partitions/*/partition_*.npy | head -10

# Continuar de onde parou (apenas reexecuta, sobrescreve partições existentes)
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --left-on ...

# Ou mesclar o que foi feito até agora
for model_dir in data/embeddings_partitions/*/; do
    model_name=$(basename "$model_dir")
    python3 ./run_linktransformer/merge_partitions.py \
        "data/embeddings_partitions/$model_name" \
        --side base \
        --verbose
done
```

---

## Comparar Antiga vs Nova Geração

Se você reprocessou com particionamento, validar que os resultados são iguais:

```python
import numpy as np
import json

# Carregar nova versão (mesclada)
embeddings_new = np.load('data/embeddings_base_sentence-transformers_all-MiniLM-L6-v2.npy')

# Validar
print(f"Shape: {embeddings_new.shape}")
print(f"Dtype: {embeddings_new.dtype}")
print(f"Min: {embeddings_new.min():.6f}")
print(f"Max: {embeddings_new.max():.6f}")
print(f"Mean: {embeddings_new.mean():.6f}")

# Verificar normalização
norms = np.linalg.norm(embeddings_new, axis=1)
print(f"Normas (devem ser ~1.0):")
print(f"  Min: {norms.min():.6f}")
print(f"  Max: {norms.max():.6f}")
print(f"  Mean: {norms.mean():.6f}")

# Ler manifest
with open('data/embeddings_manifest.json') as f:
    manifest = json.load(f)
    for entry in manifest:
        if entry['safe_model_name'] == 'sentence-transformers_all-MiniLM-L6-v2':
            print(f"\nMetadados:")
            print(f"  Tempo: {entry['elapsed_seconds']:.1f}s ({entry['elapsed_minutes']:.1f}min)")
            print(f"  Memória: {entry['memory_delta_mb']:.0f} MB")
            print(f"  Partições: {entry.get('num_partitions', 'N/A')}")
```

---

## Otimizações Avançadas

### 1. Processamento Paralelo de Modelos

```bash
#!/usr/bin/env bash
# Processar modelos em paralelo (máx 2 por vez)

models=(
    "sentence-transformers/all-MiniLM-L6-v2"
    "sentence-transformers/all-mpnet-base-v2"
    "intfloat/multilingual-e5-large"
    "neuralmind/bert-large-portuguese-cased"
)

for model in "${models[@]}"; do
    ./run_linktransformer/run_embeddings_partitioned_podman.sh \
        --mode base \
        --model "$model" \
        --left-on ... \
        --merge-partitions &
    
    # Limitar a 2 processos paralelos
    if [[ $(jobs -r -p | wc -l) -ge 2 ]]; then
        wait -n
    fi
done

wait
echo "✓ Todos os modelos processados"
```

### 2. Usar RAM Disk para Partições Temporárias

```bash
# Criar RAM disk (4GB)
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=4G tmpfs /mnt/ramdisk

# Processar com partições em RAM
HOST_DATA_DIR="$PWD/pasta_aux" \
CONTAINER_DATA_DIR="/data" \
./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --left-on ...

# Limpar
sudo umount /mnt/ramdisk
```

---

## FAQ de Integração

**P: Os embeddings antigos ainda funcionam?**  
R: Sim! Se você não tiver `--merge-partitions`, gere manualmente com `merge_partitions.py`

**P: Preciso reprocessar tudo?**  
R: Não é obrigatório, mas recomendado para aproveitar eficiência de memória

**P: Quanto espaço economizo sem `--merge-partitions`?**  
R: ~50% (não precisa manter partições + arquivo final)

**P: Como processar um arquivo MUITO grande?**  
R: Reduzir `--partition-size` de 100k para 50k, 25k, etc.

**P: Posso usar partições em aplicações tempo-real?**  
R: Sim! Use streaming de partições para baixa latência

---

## Próximas Etapas

1. ✅ Testar novo script com `test_embeddings_partitioned.sh`
2. ✅ Atualizar comando Podman existente
3. ✅ Executar primeira vez com `--merge-partitions`
4. ✅ Validar saída com health check
5. ⚪ (Opcional) Otimizar `--partition-size` e `--batch-size` para seu hardware
6. ⚪ (Opcional) Integrar processamento paralelo de modelos

---

**Versão:** 1.0  
**Última atualização:** 18 de maio de 2026
