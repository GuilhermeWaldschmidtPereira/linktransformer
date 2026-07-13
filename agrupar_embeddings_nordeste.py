#!/usr/bin/env python3
"""
Agrupa arquivos .npy de municipios do Nordeste em um unico embedding_base.npy.

O filtro considera arquivos cujo nome contenha:
  - uma UF do Nordeste como token: AL, BA, CE, MA, PB, PE, PI, RN, SE; ou
  - um codigo IBGE municipal de 7 digitos iniciado por um codigo de UF do Nordeste.

Exemplos:
  python agrupar_embeddings_nordeste.py --input-dir /dados/embeddings
  python agrupar_embeddings_nordeste.py --input-dir /dados/embeddings --output /dados/embedding_base.npy
  python agrupar_embeddings_nordeste.py --input-dir /dados/embeddings --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


NORDESTE_UFS = {"AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"}
NORDESTE_IBGE_PREFIXES = {"21", "22", "23", "24", "25", "26", "27", "28", "29"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatena arquivos .npy do Nordeste em embedding_base.npy."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Pasta que contem os arquivos .npy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("embedding_base.npy"),
        help="Arquivo .npy final. Padrao: embedding_base.npy no diretorio atual.",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=0,
        help="Eixo de concatenacao. Para embeddings, normalmente 0. Padrao: 0.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Busca apenas na pasta informada, sem percorrer subpastas.",
    )
    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help=(
            "Inclui tambem arquivos cujo nome nao permite inferir UF/codigo IBGE. "
            "Use apenas se a pasta ja tiver somente municipios do Nordeste."
        ),
    )
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Permite carregar .npy com objetos Python. Desativado por seguranca.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra quais arquivos seriam usados e nao gera o arquivo final.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV opcional com o resumo dos arquivos incluidos.",
    )
    return parser.parse_args()


def is_northeast_file(path: Path) -> tuple[bool, str]:
    name = path.stem.upper()

    uf_tokens = set(re.findall(r"(?<![A-Z0-9])([A-Z]{2})(?![A-Z0-9])", name))
    matched_ufs = sorted(uf_tokens & NORDESTE_UFS)
    if matched_ufs:
        return True, f"uf={matched_ufs[0]}"

    ibge_codes = re.findall(r"(?<!\d)(\d{7})(?!\d)", name)
    for code in ibge_codes:
        if code[:2] in NORDESTE_IBGE_PREFIXES:
            return True, f"ibge={code}"

    return False, "sem_uf_ou_codigo_ibge_do_nordeste_no_nome"


def iter_npy_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.npy" if recursive else "*.npy"
    return sorted(p for p in input_dir.glob(pattern) if p.is_file())


def shape_after_concat(shapes: list[tuple[int, ...]], axis: int) -> tuple[int, ...]:
    base = list(shapes[0])
    if axis < 0:
        axis += len(base)
    if axis < 0 or axis >= len(base):
        raise ValueError(f"axis={axis} invalido para arrays com {len(base)} dimensoes")

    total = 0
    for shape in shapes:
        if len(shape) != len(base):
            raise ValueError(f"Dimensoes diferentes encontradas: {tuple(base)} e {shape}")
        for idx, (expected, actual) in enumerate(zip(base, shape)):
            if idx != axis and expected != actual:
                raise ValueError(
                    "Shapes incompativeis para concatenacao: "
                    f"{tuple(base)} e {shape} diferem no eixo {idx}"
                )
        total += shape[axis]

    base[axis] = total
    return tuple(base)


def write_manifest(rows: list[dict[str, object]], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "shape", "dtype", "motivo_inclusao"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output = args.output.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERRO: pasta nao encontrada: {input_dir}", file=sys.stderr)
        return 2

    all_files = iter_npy_files(input_dir, recursive=not args.no_recursive)
    selected: list[tuple[Path, str]] = []
    skipped: list[tuple[Path, str]] = []

    for path in all_files:
        if path.resolve() == output:
            continue
        ok, reason = is_northeast_file(path)
        if ok or args.include_unmatched:
            selected.append((path, reason if ok else "incluido_por_include_unmatched"))
        else:
            skipped.append((path, reason))

    if not selected:
        print("Nenhum .npy do Nordeste foi encontrado.", file=sys.stderr)
        print(
            "Dica: se a pasta ja contem apenas Nordeste, rode com --include-unmatched.",
            file=sys.stderr,
        )
        return 1

    arrays_info: list[dict[str, object]] = []
    shapes: list[tuple[int, ...]] = []
    dtype: np.dtype | None = None

    print(f"Arquivos .npy encontrados: {len(all_files)}")
    print(f"Arquivos selecionados: {len(selected)}")
    print(f"Arquivos ignorados: {len(skipped)}")

    for path, reason in selected:
        arr = np.load(path, mmap_mode="r", allow_pickle=args.allow_pickle)
        if dtype is None:
            dtype = arr.dtype
        elif arr.dtype != dtype:
            raise ValueError(f"Dtypes diferentes: {path} tem {arr.dtype}, esperado {dtype}")

        shapes.append(tuple(arr.shape))
        arrays_info.append(
            {
                "path": str(path),
                "shape": "x".join(str(dim) for dim in arr.shape),
                "dtype": str(arr.dtype),
                "motivo_inclusao": reason,
            }
        )

    final_shape = shape_after_concat(shapes, args.axis)
    print(f"Shape final: {final_shape}")
    print(f"Dtype final: {dtype}")
    print(f"Saida: {output}")

    if args.manifest:
        write_manifest(arrays_info, args.manifest.resolve())
        print(f"Manifest salvo em: {args.manifest.resolve()}")

    if args.dry_run:
        print("Dry-run concluido; nenhum arquivo foi gerado.")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    out = open_memmap(output, mode="w+", dtype=dtype, shape=final_shape)

    cursor = 0
    axis = args.axis if args.axis >= 0 else args.axis + len(final_shape)
    for path, _reason in selected:
        arr = np.load(path, mmap_mode="r", allow_pickle=args.allow_pickle)
        width = arr.shape[axis]
        slices = [slice(None)] * len(final_shape)
        slices[axis] = slice(cursor, cursor + width)
        out[tuple(slices)] = arr
        cursor += width
        print(f"Adicionado: {path} -> posicao {cursor}")

    out.flush()
    print("Concluido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

