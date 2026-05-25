# Pipeline de Geocodificação com Indexação ANN

Este repositório contém o pipeline experimental utilizado para avaliação
de diferentes métodos de indexação para busca aproximada de vizinhos
(*Approximate Nearest Neighbors -- ANN*) aplicados a tarefas de
*matching* textual em cenários de geocodificação.

O pipeline integra diferentes algoritmos de indexação, permitindo a
execução automatizada de experimentos com diferentes modelos de
*embeddings* e diferentes estruturas de busca vetorial.

------------------------------------------------------------------------

# Estrutura do Projeto

    .
    ├── main.sh
    ├── requirements.txt
    ├── requirements_scann.txt
    ├── run_linktransformer/
    ├── src/
    ├── data/
    └── resultados/

Descrição dos principais componentes:

  Diretório                  Descrição
  -------------------------- ------------------------------------------------------
  `main.sh`                  Script principal responsável por executar o pipeline
  `requirements.txt`         Dependências do ambiente principal
  `requirements_scann.txt`   Dependências específicas do método ScaNN
  `run_linktransformer`      Scripts de execução do processo de matching
  `src`                      Implementação dos métodos de indexação
  `data`                     Bases de dados utilizadas nos experimentos
  `resultados`               Saída gerada pelos experimentos

------------------------------------------------------------------------

# Pré-requisitos

Antes de executar o pipeline, é necessário possuir os seguintes
softwares instalados:

-   Python 3.11.2
-   pip
-   venv

Em sistemas Linux (Ubuntu/Debian):

``` bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

Verifique a versão instalada:

``` bash
python3.11 --version
```


## Preparação dos Dados

Para a execução dos experimentos de *matching* textual, os campos de endereço foram concatenados em uma única string.

Originalmente, os dados estavam estruturados em múltiplos campos:

```
rua | bairro | numero
```

Exemplo da estrutura original:

```
rua: Av. Brasil
bairro: Centro
numero: 123
```

Após a concatenação, o endereço passa a ser representado como:

```
Av. Brasil Centro 123
```

Essa representação foi utilizada como entrada para a geração dos *embeddings* textuais utilizados no processo de *matching* entre as bases.

As bases de dados utilizadas nos experimentos devem estar armazenadas em arquivos `.csv` no diretório:

```
/data
```

O pipeline espera encontrar **dois arquivos específicos** nesse diretório:

```
enderecos_base.csv
enderecos_query.csv
```

onde:

- **enderecos_base.csv**: contém a base de registros utilizada para indexação.
- **enderecos_query.csv**: contém os registros utilizados como consultas no processo de busca vetorial.

------------------------------------------------------------------------

# Criação dos Ambientes Virtuais

Para a execução do pipeline são necessários **dois ambientes virtuais**,
pois alguns métodos de indexação utilizam as mesmas bibliotecas, porém
em **versões diferentes**.

Todos os comandos devem ser executados **na raiz do projeto**.

## Criar ambiente virtual geral

``` bash
python3.11 -m venv venv
```

## Criar ambiente virtual para ScaNN

``` bash
python3.11 -m venv venv_scann
```

------------------------------------------------------------------------

# Instalação das Dependências

Cada ambiente virtual deve ser ativado separadamente para instalar as
dependências correspondentes.

## Ambiente principal (venv)

Ativar ambiente:

``` bash
source venv/bin/activate
```

Atualizar pip:

``` bash
python -m pip install --upgrade pip
```

Instalar dependências:

``` bash
pip install -r requirements.txt
```

Desativar ambiente:

``` bash
deactivate
```

Observação: a execução principal atual roda via Podman. Esse ambiente
virtual continua útil para desenvolvimento local, testes manuais e uso
fora do container.

------------------------------------------------------------------------

## Ambiente ScaNN (venv_scann)

Ativar ambiente:

``` bash
source venv_scann/bin/activate
```

Atualizar pip:

``` bash
python -m pip install --upgrade pip
```

Instalar dependências:

``` bash
pip install -r requirements_scann.txt
```

Desativar ambiente:

``` bash
deactivate
```

------------------------------------------------------------------------

# Permissão de Execução

Antes de executar o pipeline, conceda permissão de execução ao script
principal:

``` bash
chmod +x main.sh
chmod +x main_scann.sh
```

------------------------------------------------------------------------

# Execução do LinkTransformer sem ScaNN

Após instalar o Podman, execute o pipeline principal com:

``` bash
./main.sh
```

Esse comando constrói automaticamente a imagem
`localhost/projeto-mestrado-linktransformer:latest`, caso ela ainda não
exista, e executa somente os métodos do LinkTransformer que não dependem
do ScaNN. O diretório do projeto é montado em `/workspace`, então os
arquivos de entrada, embeddings e resultados continuam sendo lidos e
gravados no workspace local.

Para forçar outro nome de imagem:

``` bash
LINKTRANSFORMER_IMAGE=meu-linktransformer:latest ./main.sh
```

Para executar apenas o ScaNN, use:

``` bash
./main_scann.sh
```

O script `main.sh` executará automaticamente os experimentos definidos
para o ambiente principal, sem ativar o ambiente `venv_scann`.

------------------------------------------------------------------------

# Saída dos Experimentos

Os resultados gerados pelo pipeline incluem métricas como:

-   tempo de indexação
-   tempo de busca
-   tempo total de execução
-   uso de memória
-   número de matches encontrados

Essas informações são armazenadas no diretório:

    resultados/

Os arquivos gerados podem ser utilizados posteriormente para análise
estatística ou geração de gráficos comparativos.

------------------------------------------------------------------------

# Observações

-   Todos os comandos devem ser executados na raiz do projeto.
-   O uso de dois ambientes virtuais evita conflitos entre versões de
    bibliotecas utilizadas por diferentes métodos de indexação.
-   Caso ocorram problemas com o pip, utilize:

``` bash
python -m pip install -r requirements.txt
```

ou

``` bash
python -m pip install -r requirements_scann.txt
```

------------------------------------------------------------------------

# Autor

Guilherme Waldschmidt Pereira
