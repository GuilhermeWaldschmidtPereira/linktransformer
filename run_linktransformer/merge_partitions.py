#!/usr/bin/env python3
"""
Utilitário para mesclar partições de embeddings em um único arquivo.

Útil quando a geração foi feita sem --merge-partitions e você quer consolidar depois.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mescla partições de embeddings em um único arquivo .npy",
    )
    parser.add_argument(
        "partition_dir",
        help="Diretório contendo as partições (ex: data/embeddings_partitions/modelo_name)",
    )
    parser.add_argument(
        "--side",
        choices=["base", "query"],
        required=True,
        help="Lado a mesclar: base ou query",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Caminho do arquivo de saída (padrão: embeddings_{side}_{model_name}.npy no diretório pai)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Mostrar informações detalhadas",
    )
    parser.add_argument(
        "--delete-partitions",
        action="store_true",
        help="Remove cada partição assim que ela for escrita no arquivo mesclado.",
    )
    return parser.parse_args()


def find_partitions(partition_dir: Path, side: str) -> List[Path]:
    """Encontra todas as partições para um lado (base/query) ordenadas por índice."""
    pattern = f"partition_*_{side}.npy"
    partitions = sorted(partition_dir.glob(pattern))

    if not partitions:
        raise FileNotFoundError(f"Nenhuma partição encontrada em {partition_dir} com padrão {pattern}")

    return partitions


def extract_partition_index(filename: str) -> int:
    """Extrai o índice da partição do nome do arquivo."""
    # partition_000000_base.npy -> 000000
    try:
        return int(filename.split("_")[1])
    except (IndexError, ValueError):
        return 0


def merge_partitions(
    partition_dir: Path,
    side: str,
    output_path: str,
    verbose: bool = False,
    delete_partitions: bool = False,
) -> Tuple[int, int]:
    """
    Mescla as partições em um único arquivo.

    Retorna (num_partitions, total_shapes).
    """
    partitions = find_partitions(partition_dir, side)

    if verbose:
        print(f">>> Encontradas {len(partitions)} partições para '{side}'")
        print(f">>> Primeira: {partitions[0].name}")
        print(f">>> Última: {partitions[-1].name}")

    merged_memmap: Optional[np.memmap] = None
    write_offset = 0
    total_rows = 0

    for i, partition_file in enumerate(partitions):
        if verbose:
            print(f"  [{i + 1}/{len(partitions)}] Carregando {partition_file.name}...", end="", flush=True)

        arr = np.load(partition_file, mmap_mode="r")
        rows = arr.shape[0]

        if merged_memmap is None:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            total_rows = sum(np.load(path, mmap_mode="r").shape[0] for path in partitions)
            merged_memmap = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=np.float32,
                shape=(total_rows, arr.shape[1]),
            )

        merged_memmap[write_offset:write_offset + rows] = arr
        merged_memmap.flush()
        write_offset += rows

        if verbose:
            print(f" shape={arr.shape}")

        del arr

        if delete_partitions:
            partition_file.unlink()
            if verbose:
                print(f"    Removida após merge: {partition_file}")

    if merged_memmap is not None:
        merged_memmap.flush()
        merged_shape = merged_memmap.shape
        del merged_memmap
    else:
        merged_shape = (0, 0)

    if verbose:
        print(f">>> Array mesclado: shape={merged_shape}, dtype=float32")
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f">>> Salvo em: {output_path}")
        print(f">>> Tamanho: {file_size_mb:.2f} MB")

    return len(partitions), merged_shape[0]


def main() -> None:
    args = parse_args()

    partition_dir = Path(args.partition_dir).resolve()
    if not partition_dir.is_dir():
        print(f"Erro: {partition_dir} não é um diretório válido", file=sys.stderr)
        sys.exit(1)

    # Determinar caminho de saída
    if args.output:
        output_path = args.output
    else:
        # Padrão: embeddings_{side}_{model_name}.npy no diretório pai
        model_name = partition_dir.name
        output_path = str(partition_dir.parent / f"embeddings_{args.side}_{model_name}.npy")

    if args.verbose:
        print(f">>> Partição dir: {partition_dir}")
        print(f">>> Lado: {args.side}")
        print(f">>> Saída: {output_path}")
        print()

    # Mesclar
    num_partitions, total_rows = merge_partitions(
        partition_dir,
        args.side,
        output_path,
        verbose=args.verbose,
        delete_partitions=args.delete_partitions,
    )

    print(
        f"✓ Sucesso! Mescladas {num_partitions} partições com {total_rows:,} linhas em {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
