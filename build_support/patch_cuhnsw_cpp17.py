#!/usr/bin/env python3
"""Patch CUHNSW 0.0.8's CUDA extension to compile consistently as C++17."""

from pathlib import Path
import sys


def replace_once(path: Path, old: str, new: str) -> None:
    content = path.read_text()
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r} in {path}, found {count}")
    path.write_text(content.replace(old, new, 1))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_cuhnsw_cpp17.py CUHNSW_SOURCE_DIR")

    source_dir = Path(sys.argv[1])
    setup_py = source_dir / "setup.py"
    cuda_setup_py = source_dir / "cuda_setup.py"

    replace_once(setup_py, "-std=c++14", "-std=c++17")

    cuda_setup = cuda_setup_py.read_text()
    standard_flag_count = cuda_setup.count("-std=c++14")
    if standard_flag_count != 2:
        raise RuntimeError(
            "Expected two C++14 flags in CUHNSW 0.0.8 cuda_setup.py, "
            f"found {standard_flag_count}"
        )
    cuda_setup = cuda_setup.replace("-std=c++14", "-std=c++17")
    old_nvcc_flags = 'post_args = [f"-arch=sm_{arch}"] + \\'
    new_nvcc_flags = 'post_args = ["-std=c++17", f"-arch=sm_{arch}"] + \\'
    nvcc_flag_count = cuda_setup.count(old_nvcc_flags)
    if nvcc_flag_count != 1:
        raise RuntimeError(
            "Expected one NVCC post_args declaration in cuda_setup.py, "
            f"found {nvcc_flag_count}"
        )
    cuda_setup = cuda_setup.replace(old_nvcc_flags, new_nvcc_flags, 1)
    cuda_setup_py.write_text(cuda_setup)


if __name__ == "__main__":
    main()
