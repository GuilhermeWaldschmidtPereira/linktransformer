FROM python:3.11-slim

# 1. Dependências de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Instalar Julia 1.11.1
ENV JULIA_VERSION=1.11.1

RUN curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-$JULIA_VERSION-linux-x86_64.tar.gz -o julia.tar.gz \
    && tar -xzf julia.tar.gz -C /opt \
    && rm julia.tar.gz \
    && ln -s /opt/julia-$JULIA_VERSION/bin/julia /usr/local/bin/julia

# 3. Diretório de trabalho
WORKDIR /usr/src/app

# 4. Instalar dependências Python (inclui o pacote 'julia', que fornece o python-jl)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Inicializar o PyJulia (gera o python-jl pronto pra uso)
RUN python -m julia.install

# 6. Copiar o código
COPY . .

# 7. Comando padrão: roda o script integrando Python + Julia
CMD ["python-jl", "run_linktransformer/main2.py"]
