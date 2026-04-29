FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
    wget \
    ca-certificates \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# ── Julia ──────────────────────────────────────────────────────────────
ARG JULIA_VERSION=1.10.4
RUN wget -q "https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
    && tar -xzf "julia-${JULIA_VERSION}-linux-x86_64.tar.gz" -C /opt \
    && mv "/opt/julia-${JULIA_VERSION}" /opt/julia \
    && rm "julia-${JULIA_VERSION}-linux-x86_64.tar.gz"
ENV PATH="/opt/julia/bin:$PATH"

# Limpar o flag de executable-stack nas .so do Julia (restrição de seccomp do Docker)
RUN find /opt/julia -name "*.so*" -type f \
    -exec patchelf --clear-execstack {} \; 2>/dev/null || true

# ── Pré-instalar pacotes Julia necessários ─────────────────────────────
RUN julia -e 'using Pkg; Pkg.add.(["HNSW", "Distances", "NPZ"]); Pkg.precompile()'

# ── Python requirements ─────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt

RUN grep -v '^hdbscan==' /tmp/requirements.txt > /tmp/requirements.docker.txt

RUN python -m pip install --upgrade pip "setuptools<81" wheel && \
    python -m pip install "Cython<3" "numpy==1.22.4" && \
    python -m pip install --no-build-isolation -r /tmp/requirements.docker.txt

# ── nmslib e PyJulia ────────────────────────────────────────────────────
RUN python -m pip install nmslib julia

# Configura PyJulia (instala PyCall.jl dentro do Julia)
RUN python -c "import julia; julia.install()"

COPY . /app

RUN chmod +x /app/docker-main.sh

CMD ["bash", "./docker-main.sh"]
