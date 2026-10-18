#!/usr/bin/env python3
"""Set Jasper's Python FFI target to the server's Ampere SM 8.6 architecture."""

from pathlib import Path
import re
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: patch_jasper_cuda_arch.py CMakeLists.txt CUDA_ARCH")

    cmake_file = Path(sys.argv[1])
    arch = sys.argv[2]
    if not arch.isdigit():
        raise SystemExit(f"CUDA_ARCH must be numeric, got {arch!r}")

    original = cmake_file.read_text()
    pattern = re.compile(
        r'(set_target_properties\(jasper_ffi\s+PROPERTIES\s+'
        r'CUDA_ARCHITECTURES\s+)"[^"]+"',
        re.MULTILINE,
    )
    updated, replacements = pattern.subn(r'\g<1>"' + arch + '"', original)
    if replacements != 1:
        raise SystemExit(
            "Expected exactly one CUDA_ARCHITECTURES property for jasper_ffi; "
            f"found {replacements} in {cmake_file}"
        )
    cmake_file.write_text(updated)


if __name__ == "__main__":
    main()
