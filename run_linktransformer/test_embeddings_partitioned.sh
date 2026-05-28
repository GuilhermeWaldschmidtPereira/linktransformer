#!/usr/bin/env bash
# Script de teste rápido para a geração de embeddings particionados
# Este script cria um CSV de teste pequeno e gera embeddings particionados

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="${REPO_ROOT}/test_embeddings_partitioned"
PYTHON_CMD="python3"

# Verificar se estamos em venv
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_CMD="${VIRTUAL_ENV}/bin/python3"
    echo ">>> Usando venv: ${VIRTUAL_ENV}"
fi

echo ">>> Criando diretório de teste: ${TEST_DIR}"
mkdir -p "${TEST_DIR}"

echo ">>> Gerando CSV de teste com 350 linhas (3 partições de 100 linhas)..."
cat > "${TEST_DIR}/base_test.csv" << 'EOF'
id_municipio,uf,municipio,logradouro,numero,complemento,localidade,setor_censitario
3550308,SP,São Paulo,Rua A,100,,Centro,001
3550308,SP,São Paulo,Rua B,200,,Vila,002
3106200,MG,Belo Horizonte,Avenida C,300,,Zona,003
3304557,RJ,Rio de Janeiro,Rua D,400,,Praia,004
2927408,BA,Salvador,Rua E,500,,Barra,005
EOF

# Repetir para ter 350 linhas
echo ">>> Expandindo CSV de teste..."
for i in {1..69}; do
    sed '2,$s/.*//' "${TEST_DIR}/base_test.csv" | tail -n +2 | sed "s/^/SP,São Paulo,Rua $(printf 'Z%.0s' $(seq 1 $i)),$(($i*100)),,Teste,00$(($i % 10))/" >> "${TEST_DIR}/base_test.csv" 2>/dev/null || true
done

# Método simples e confiável: criar arquivo com exatamente 350 linhas
python3 << 'PYTHON_SCRIPT'
import csv

data = [
    ["id_municipio", "uf", "municipio", "logradouro", "numero", "complemento", "localidade", "setor_censitario"],
]

# Gerar 350 linhas de dados
for i in range(350):
    data.append([
        "3550308" if i < 200 else "3304557",
        "SP", 
        "São Paulo",
        f"Rua Test {i}",
        str(1000 + i),
        f"Apt {i}",
        "Centro",
        f"{i % 100:03d}"
    ])

with open("/tmp/test_embeddings.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerows(data)

print(f">>> CSV criado com {len(data)-1} linhas")
PYTHON_SCRIPT

echo ">>> Executando geração de embeddings particionados..."
echo ">>> Usando apenas 1 modelo para teste rápido..."

${PYTHON_CMD} "${REPO_ROOT}/run_linktransformer/run_embeddings_partitioned.py" \
    --base-path /tmp/test_embeddings.csv \
    --output-dir "${TEST_DIR}" \
    --mode base \
    --model "sentence-transformers/all-MiniLM-L6-v2" \
    --batch-size 8 \
    --partition-size 100 \
    --left-on uf municipio logradouro numero \
    --merge-partitions \
    --manifest-path "${TEST_DIR}/embeddings_manifest.json"

echo ""
echo ">>> ✓ Teste concluído com sucesso!"
echo ""
echo ">>> Estrutura de saída:"
find "${TEST_DIR}" -type f -name "*.npy" -o -name "*.json" | sort | sed 's|^|  |'

echo ""
echo ">>> Tamanho dos arquivos:"
du -h "${TEST_DIR}"/* 2>/dev/null | sed 's|^|  |'

echo ""
echo ">>> Manifest:"
cat "${TEST_DIR}/embeddings_manifest.json" | python3 -m json.tool | head -30

echo ""
echo ">>> Limpeza de teste (comentado):"
echo "#   rm -rf ${TEST_DIR}"
echo "#   rm -f /tmp/test_embeddings.csv"
