#!/usr/bin/env python3
"""Subamostra um arquivo .npy

Cria um novo arquivo .npy contendo apenas N registros do primeiro eixo
do arquivo original. Por padrão grava 200000 registros. Suporta amostragem
aleatória sem reposição com `--random`.

Exemplos:
  python3 scripts/subsample_npy.py ../dados_nordeste/embeddings_base_sentence-transformers_all-MiniLM-L6-v2_copia.npy embeddings_base_sentence-transformers_all-MiniLM-L6-v2.npy
  python3 scripts/subsample_npy.py ../dados_nordeste/embeddings_base_sentence-transformers_all-MiniLM-L6-v2_copia.npy embeddings_base_sentence-transformers_all-mpnet-base-v2_copia.npy
  python3 scripts/subsample_npy.py ../dados_nordeste/embeddings_base.npy out.npy --n 200000 --random --seed 42
"""
from __future__ import annotations
import argparse
import sys
import os
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Subamostra um arquivo .npy")
    p.add_argument("input", help="Caminho para o arquivo .npy de entrada")
    p.add_argument("output", help="Caminho do .npy de saída")
    p.add_argument("--n", type=int, default=200_000, help="Número de registros desejados (padrão: 200000)")
    p.add_argument("--random", action="store_true", help="Amostragem aleatória sem reposição (por padrão pega os primeiros N)")
    p.add_argument("--seed", type=int, default=None, help="Semente RNG (apenas quando --random)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inp = args.input
    out = args.output
    n = args.n

    if not os.path.isfile(inp):
        print(f"Arquivo de entrada não encontrado: {inp}", file=sys.stderr)
        return 2

    # Carrega usando mmap para evitar carregar tudo na memória quando possível
    try:
        arr = np.load(inp, mmap_mode='r')
    except Exception as e:
        print(f"Erro ao carregar {inp}: {e}", file=sys.stderr)
        return 3

    if arr.ndim == 0:
        # Escalar — nada para subamostrar
        print("Arquivo .npy não contém um array com primeiro eixo (0-d). Copiando arquivo inteiro.")
        data = np.array(arr)
        np.save(out, data)
        print(f"Gravado {out} ({data.shape})")
        return 0

    total = arr.shape[0]
    if n >= total:
        # se solicitado mais do que existe, copia tudo (usaremos copia de arquivo para preservar exatamente)
        try:
            # tentar copiar diretamente o arquivo .npy para manter formato exatamente igual
            import shutil

            shutil.copyfile(inp, out)
            print(f"Solicitado {n} registros, mas o arquivo tem {total}. Copiado arquivo inteiro para {out}.")
            return 0
        except Exception:
            # fallback: salvar carregando em memória
            np.save(out, np.array(arr))
            print(f"Fallback: salvou todo o array para {out}.")
            return 0

    if args.random:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(total, size=n, replace=False)
        # ordenar índices para manter alguma localidade ao ler via mmap
        idx.sort()
        try:
            sample = arr[idx]
        except Exception:
            # caso arr seja memmap e seleção fancy falhe, materializar em memória por blocos
            sample = _gather_by_indices_memmap(arr, idx)
    else:
        # fatia simples: preferível para velocidade e IO
        sample = arr[:n]

    # grava saída
    try:
        np.save(out, sample)
        print(f"Gravado {out} com {n} registros (de {total})")
    except Exception as e:
        print(f"Erro ao salvar {out}: {e}", file=sys.stderr)
        return 4

    return 0


def _gather_by_indices_memmap(arr, indices):
    """Ler memmap por blocos usando índices ordenados e concatenar.

    Útil quando fancy-indexing não é suportado diretamente pela instância memmap.
    """
    parts = []
    i = 0
    L = len(indices)
    while i < L:
        j = i + 1
        while j < L and indices[j] == indices[j - 1] + 1:
            j += 1
        s = indices[i]
        e = indices[j - 1] + 1
        parts.append(np.array(arr[s:e]))
        i = j
    return np.concatenate(parts, axis=0)


if __name__ == "__main__":
    raise SystemExit(main())
