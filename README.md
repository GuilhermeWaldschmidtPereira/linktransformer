# LinkTransformer: benchmark ANN com embeddings pré-computados

Este repositório compara diferentes estratégias de busca vetorial sobre embeddings gerados a partir de dois arquivos de entrada:

- `data/base.csv`
- `data/query.csv`

O fluxo atual do código é este:

1. gerar os embeddings em `data/`
2. construir as imagens dos benchmarks
3. executar os métodos via `podman`

Hoje, o caminho principal do projeto está concentrado nestes arquivos:

- `run_linktransformer/run_embeddings_partitioned_podman.sh`
- `run_linktransformer/run_embeddings_partitioned.py`
- `run_linktransformer/run_embeddings.py`
- `build.sh`
- `main.sh`
- `run_linktransformer/main_linktransformer.py`
- `run_linktransformer/main_nmslib_runner.py`
- `run_linktransformer/main_hnsw_julia.py`
- `run_linktransformer/main_scann.py`

## Estrutura resumida

```text
.
├── build.sh
├── main.sh
├── Containerfile.linktransformer
├── Containerfile.scann
├── Containerfile.embeddings.base
├── Containerfile.embeddings
├── data/
├── resultados/
├── run_linktransformer/
└── src/
```

## Pré-requisitos

Para o fluxo recomendado:

- `podman`
- acesso a disco suficiente para os arquivos `.npy`
- os CSVs de entrada em `data/`

Para alternativas locais sem container:

- Python 3.10 ou 3.11
- `venv`

## Arquivos de entrada esperados

Por padrão, os scripts usam:

- `data/base.csv`
- `data/query.csv`

No estado atual do repositório:

- `base.csv` é lido com separador `;`
- `query.csv` é lido com separador `,`

Os scripts de embeddings já tratam isso automaticamente.

Exemplo de colunas existentes hoje:

```text
uf, municipio, logradouro, numero, complemento, localidade, setor_censitario
```

Essas colunas podem ser passadas explicitamente nas flags `--left-on`, `--right-on` ou `--on`.

## Fluxo rápido com Podman

Se você quer só o caminho principal, execute nesta ordem:

### 1. Gerar embeddings

O benchmark atual usa estes dois modelos:

- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`

Para gerar apenas os embeddings necessários ao pipeline atual:

```bash
HOST_DATA_DIR="$(pwd)/data" ./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode both \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --model sentence-transformers/all-mpnet-base-v2 \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  --right-on uf municipio logradouro numero complemento localidade setor_censitario \
  --manifest-path /data/embeddings_manifest.json
```

Esse comando:

- usa `data/base.csv` e `data/query.csv`
- grava os `.npy` em `data/`
- grava as partições em `data/embeddings_partitions/`
- grava o manifest em `data/embeddings_manifest.json`
- faz o merge final automaticamente

Se `OPENAI_API_KEY` estiver exportado no seu shell e você quiser usar os modelos locais acima, remova a variável antes de rodar:

```bash
unset OPENAI_API_KEY
```

### 2. Construir as imagens dos benchmarks

```bash
./build.sh
```

Esse script gera:

- `localhost/projeto-mestrado-linktransformer:latest`
- `localhost/projeto-mestrado-scann:latest`

Se quiser nomes diferentes:

```bash
LINKTRANSFORMER_IMAGE=meu-linktransformer:latest \
SCANN_IMAGE=meu-scann:latest \
./build.sh
```

### 3. Rodar o pipeline completo

```bash
./main.sh
```

O `main.sh`:

- recria `resultados.csv` na raiz do projeto
- roda `main_linktransformer.py`
- roda `main_nmslib_runner.py`
- roda `main_hnsw_julia.py`
- roda `main_scann.py`

Por padrão, a indexação roda todos os modelos de embedding configurados. Para rodar apenas um modelo:

```bash
./main.sh --model sentence-transformers/all-MiniLM-L6-v2
```

Para rodar um subconjunto, repita `--model`:

```bash
./main.sh \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --model sentence-transformers/all-mpnet-base-v2
```

Também é aceito o nome sanitizado usado nos arquivos `.npy`, por exemplo `sentence-transformers_all-MiniLM-L6-v2`. Para deixar explícito que quer todos os modelos, use `--model all`.

Por padrão, o pipeline roda no escopo por município, criando um índice por
`id_municipio`/`municipio`. Para rodar a versão geral, com um único índice para
toda a base e consulta sobre toda a query:

```bash
./main.sh --scope geral --model sentence-transformers/all-MiniLM-L6-v2
```

Para rodar explicitamente a versão por município:

```bash
./main.sh --scope municipio --model sentence-transformers/all-MiniLM-L6-v2
```

O modo `--scope geral` precisa dos embeddings consolidados em `data/`, por
exemplo `embeddings_base_<modelo>.npy` e `embeddings_query_<modelo>.npy`.

## Execuções individuais com Podman

Antes de rodar individualmente, garanta que:

- os embeddings já foram gerados
- as imagens já foram construídas com `./build.sh`

Se quiser limpar o consolidado antes de uma rodada manual:

```bash
rm -f resultados.csv
printf 'metodo,modelo_embedding,index_time,search_time,total_time,num_rows_df1,num_rows_df2,k,mem_used_indexation_MB,avg_mem_used_search_MB,matches\n' > resultados.csv
```

Defina estas variáveis uma vez:

```bash
export PROJECT_ROOT="$(pwd)"
export IMAGE_LINKTRANSFORMER="${LINKTRANSFORMER_IMAGE:-localhost/projeto-mestrado-linktransformer:latest}"
export IMAGE_SCANN="${SCANN_IMAGE:-localhost/projeto-mestrado-scann:latest}"
```

### 1. Baseline FAISS + SVS/Vamana

Observação importante: `run_linktransformer/main_linktransformer.py` executa os dois métodos no mesmo comando.

```bash
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_LINKTRANSFORMER}" \
  python /workspace/run_linktransformer/main_linktransformer.py \
    --model sentence-transformers/all-MiniLM-L6-v2
```

### 2. NMSLIB

```bash
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_LINKTRANSFORMER}" \
  python /workspace/run_linktransformer/main_nmslib_runner.py \
    --model sentence-transformers/all-MiniLM-L6-v2
```

### 3. HNSW Julia

```bash
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_LINKTRANSFORMER}" \
  python-jl /workspace/run_linktransformer/main_hnsw_julia.py \
    --model sentence-transformers/all-MiniLM-L6-v2
```

### 4. ScaNN

Modo padrão, sem ajuste fino:

```bash
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_SCANN}" \
  python /workspace/run_linktransformer/main_scann.py \
    --model sentence-transformers/all-MiniLM-L6-v2
```

Exemplo com variáveis de ambiente do ScaNN:

```bash
SCANN_BUILDER_MODE=tree_ah \
SCANN_NUM_EXECUCOES=1 \
SCANN_QUERY_BATCH_SIZE=0 \
SCANN_NUM_LEAVES=2000 \
SCANN_NUM_LEAVES_TO_SEARCH=100 \
SCANN_TRAINING_SAMPLE_SIZE=250000 \
SCANN_DIMENSIONS_PER_BLOCK=2 \
SCANN_AH_THRESHOLD=0.2 \
SCANN_REORDER_K=100 \
podman run --rm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  -e SCANN_BUILDER_MODE \
  -e SCANN_NUM_EXECUCOES \
  -e SCANN_QUERY_BATCH_SIZE \
  -e SCANN_NUM_LEAVES \
  -e SCANN_NUM_LEAVES_TO_SEARCH \
  -e SCANN_TRAINING_SAMPLE_SIZE \
  -e SCANN_DIMENSIONS_PER_BLOCK \
  -e SCANN_AH_THRESHOLD \
  -e SCANN_REORDER_K \
  -v "${PROJECT_ROOT}:/workspace:Z" \
  -w /workspace \
  "${IMAGE_SCANN}" \
  python /workspace/run_linktransformer/main_scann.py
```

## Geração de embeddings

### Comando recomendado com Podman

Para base e query juntas:

```bash
HOST_DATA_DIR="$(pwd)/data" ./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode both \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --model sentence-transformers/all-mpnet-base-v2 \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  --right-on uf municipio logradouro numero complemento localidade setor_censitario \
  --manifest-path /data/embeddings_manifest.json
```

Para gerar só a base:

```bash
HOST_DATA_DIR="$(pwd)/data" ./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --model sentence-transformers/all-mpnet-base-v2 \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  --manifest-path /data/embeddings_manifest.json
```

Para ajustar memória e throughput:

```bash
HOST_DATA_DIR="$(pwd)/data" ./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode both \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --model sentence-transformers/all-mpnet-base-v2 \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  --right-on uf municipio logradouro numero complemento localidade setor_censitario \
  --batch-size 32 \
  --partition-size 50000 \
  --manifest-path /data/embeddings_manifest.json
```

### Query isolada

No código atual, `run_embeddings_partitioned.py` não implementa `--mode query`.

Se você precisa gerar apenas a query, use `run_embeddings.py`.

Exemplo local:

```bash
python3 run_linktransformer/run_embeddings.py \
  --base-path data/base.csv \
  --query-path data/query.csv \
  --output-dir data \
  --mode query \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --model sentence-transformers/all-mpnet-base-v2 \
  --right-on uf municipio logradouro numero complemento localidade setor_censitario \
  --manifest-path data/embeddings_query_manifest.json
```

Se você quiser fazer isso via `podman`, rode manualmente o container de embeddings:

```bash
REQ_HASH="$(sha256sum requirements-embeddings.txt | awk '{print substr($1,1,16)}')"
BASE_IMAGE="localhost/linktransformer-embeddings-base:${REQ_HASH}"
IMAGE_EMBEDDINGS="localhost/linktransformer-embeddings"

podman build --layers -f Containerfile.embeddings.base -t "${BASE_IMAGE}" .
podman build --layers --build-arg BASE_IMAGE="${BASE_IMAGE}" -f Containerfile.embeddings -t "${IMAGE_EMBEDDINGS}" .

podman run --rm \
  -v "$(pwd)/data:/data:Z" \
  --entrypoint python3 \
  "${IMAGE_EMBEDDINGS}" \
  run_linktransformer/run_embeddings.py \
  --base-path /data/base.csv \
  --query-path /data/query.csv \
  --output-dir /data \
  --mode query \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --model sentence-transformers/all-mpnet-base-v2 \
  --right-on uf municipio logradouro numero complemento localidade setor_censitario \
  --manifest-path /data/embeddings_query_manifest.json
```

### Comando local sem Podman

Para rodar embeddings localmente:

```bash
python3 -m venv .venv-embeddings
source .venv-embeddings/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-embeddings.txt
```

Depois:

```bash
python3 run_linktransformer/run_embeddings_partitioned.py \
  --base-path data/base.csv \
  --query-path data/query.csv \
  --output-dir data \
  --mode both \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --model sentence-transformers/all-mpnet-base-v2 \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario \
  --right-on uf municipio logradouro numero complemento localidade setor_censitario \
  --merge-partitions \
  --manifest-path data/embeddings_manifest.json
```

## Arquivos gerados

Depois da etapa de embeddings, você deve encontrar em `data/`:

- `embeddings_base_sentence-transformers_all-MiniLM-L6-v2.npy`
- `embeddings_query_sentence-transformers_all-MiniLM-L6-v2.npy`
- `embeddings_base_sentence-transformers_all-mpnet-base-v2.npy`
- `embeddings_query_sentence-transformers_all-mpnet-base-v2.npy`
- `embeddings_partitions/...`
- `embeddings_manifest.json`

Depois dos benchmarks, você deve encontrar:

- `resultados.csv`
- `resultados/baseline/...`
- `resultados/svs/...`
- `resultados/NMSLIB/...`
- `resultados/hnsw_julia/...`
- `resultados/scann/...`

## Variáveis úteis

- `LINKTRANSFORMER_IMAGE`: nome da imagem principal
- `SCANN_IMAGE`: nome da imagem de ScaNN
- `HOST_DATA_DIR`: diretório host montado como `/data` no container de embeddings
- `LINKTRANSFORMER_BASE_CSV`: override do CSV base nos benchmarks
- `LINKTRANSFORMER_QUERY_CSV`: override do CSV query nos benchmarks
- `SCANN_BUILDER_MODE`: `brute_force` ou `tree_ah`
- `SCANN_NUM_EXECUCOES`: repetições da busca no ScaNN
- `SCANN_QUERY_BATCH_SIZE`: tamanho do lote de queries no ScaNN

## Observações importantes

- O pipeline atual depende de embeddings pré-computados. Os scripts de benchmark falham se os `.npy` não existirem.
- Os scripts de benchmark usam hoje dois modelos configurados em código: `all-MiniLM-L6-v2` e `all-mpnet-base-v2`.
- O wrapper `run_embeddings_partitioned_podman.sh` faz merge automático das partições.
- O fluxo recomendado para container é `run_embeddings_partitioned_podman.sh`. O script `run_embeddings_podman.sh` continua no repositório, mas hoje aponta para a mesma imagem de embeddings particionados.
- O script `run_embeddings_partitioned.py` não implementa `--mode query`.
- Se `OPENAI_API_KEY` estiver definido, os scripts de embeddings tentam usar a API da OpenAI para os modelos informados. Só exporte essa variável quando realmente quiser embeddings via API.
- Os scripts legados `main2.sh`, `run_linktransformer/main.sh` e `main_scann.sh` continuam no repositório, mas o fluxo documentado acima é o alinhado ao código atual.

## Autor

Guilherme Waldschmidt Pereira
