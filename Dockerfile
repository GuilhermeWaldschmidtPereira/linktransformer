FROM python:3.11.2-slim

# 1) Dependências de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg build-essential \
    libstdc++6 libgcc-s1 libgomp1 libatomic1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) Instalar Julia
ENV JULIA_VERSION=1.11.1
RUN curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-${JULIA_VERSION}-linux-x86_64.tar.gz -o julia.tar.gz \
    && tar -xzf julia.tar.gz -C /opt \
    && rm julia.tar.gz \
    && ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia

# (Opcional, mas útil) evitar precompilação “sumir” entre builds
ENV JULIA_DEPOT_PATH=/usr/local/julia_depot

WORKDIR /usr/src/app

# 3) Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Inicializar PyJulia (se você usa o pacote `julia`)
# Se não usa PyJulia, pode remover essa linha.
RUN python -m julia.install

# 5) Copiar código
COPY . .

CMD ["python", "run_linktransformer/main_scann.py"]
