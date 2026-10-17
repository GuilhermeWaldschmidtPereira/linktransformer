# Backend cuVS CAGRA

O runner usa CAGRA do pacote Python oficial cuVS sobre embeddings `.npy` já gerados. O cuVS e o CuPy ficam isolados do ambiente padrão do projeto; esta imagem usa NumPy 2, sem alterar o ambiente baseline.

## Requisitos do servidor

- Linux x86_64 com GPU NVIDIA suportada pelo cuVS. Para builds atuais, a NVIDIA recomenda Ampere ou mais nova; confira modelo e driver com `nvidia-smi` no servidor.
- Driver NVIDIA compatível com CUDA 12.9.2 e acesso à GPU pelo Podman CDI ou Docker NVIDIA Container Toolkit.
- `podman` ou `docker`, além dos CSVs e embeddings em `data/`.

## Execução

No servidor, dentro do checkout do repositório:

```bash
nvidia-smi
bash run_linktransformer/run_cuvs_podman.sh \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --scope geral \
  --k 10
```

Para usar um índice por município, troque `--scope geral` por `--scope municipio`. Se `--model` for omitido, o runner processa todos os modelos configurados. Os dados são lidos de `data/` e os resultados ficam em `resultados_cuvs/`.

## Ajustes CAGRA

Os parâmetros são opcionais e podem ser definidos antes do comando:

```bash
CUVS_BUILD_ALGO=ivf_pq \
CUVS_GRAPH_DEGREE=64 \
CUVS_INTERMEDIATE_GRAPH_DEGREE=128 \
CUVS_ITOPK_SIZE=64 \
bash run_linktransformer/run_cuvs_podman.sh --scope geral --k 10
```

`CUVS_BUILD_ALGO` aceita `ivf_pq` ou `nn_descent`. Os graus são limitados ao tamanho de cada base municipal. O runner grava os arquivos de correspondência e tempos no mesmo formato de resultados dos outros backends.

O container instala `cuvs-cu12==26.10.0`, `cupy-cuda12x` e NumPy 2; não instale esses pacotes no ambiente padrão nem use a imagem CUDA 11.1 do cuHNSW para este backend.