#!/usr/bin/env python3
"""Patch Jasper's FFI architecture and pin compatible Gallatin sources."""

from pathlib import Path
import re
import sys


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: patch_jasper_cuda_arch.py CMakeLists.txt CUDA_ARCH GALLATIN_COMMIT"
        )

    cmake_file = Path(sys.argv[1])
    arch = sys.argv[2]
    gallatin_commit = sys.argv[3]
    if not arch.isdigit():
        raise SystemExit(f"CUDA_ARCH must be numeric, got {arch!r}")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", gallatin_commit):
        raise SystemExit(f"GALLATIN_COMMIT must be a full Git SHA, got {gallatin_commit!r}")

    original = cmake_file.read_text()
    arch_pattern = re.compile(
        r'(set_target_properties\(jasper_ffi\s+PROPERTIES\s+'
        r'CUDA_ARCHITECTURES\s+)"[^"]+"',
        re.MULTILINE,
    )
    updated, arch_replacements = arch_pattern.subn(
        r'\g<1>"' + arch + '"', original
    )
    if arch_replacements != 1:
        raise SystemExit(
            "Expected exactly one CUDA_ARCHITECTURES property for jasper_ffi; "
            f"found {arch_replacements} in {cmake_file}"
        )

    gallatin_pattern = re.compile(
        r'(CPMAddPackage\(\s*NAME\s+gallatin\s+'
        r'GITHUB_REPOSITORY\s+saltsystemslab/gallatin\s+GIT_TAG\s+)\S+',
        re.MULTILINE,
    )
    updated, gallatin_replacements = gallatin_pattern.subn(
        r'\g<1>' + gallatin_commit, updated
    )
    if gallatin_replacements != 1:
        raise SystemExit(
            "Expected exactly one Gallatin dependency declaration; "
            f"found {gallatin_replacements} in {cmake_file}"
        )
    cmake_file.write_text(updated)


if __name__ == "__main__":
    main()
