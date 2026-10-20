#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT="c0804a5f46fd603794f4ce6205956a35ca46890e"
CUDA_ARCH="${PATHWEAVER_CUDA_ARCH:-86}"
JOBS="${PATHWEAVER_BUILD_JOBS:-2}"
OUTPUT="${PATHWEAVER_EXTENSION_DIR:-$ROOT/.pathweaver/extensions}"
if [[ -n "${PATHWEAVER_BUILD_DIR:-}" ]]; then
    BUILD_ROOT="$PATHWEAVER_BUILD_DIR"
    mkdir -p "$BUILD_ROOT"
else
    BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pathweaver-build.XXXXXX")"
fi
cleanup() {
    local status=$?
    trap - EXIT
    if (( status == 0 )); then
        rm -rf "$BUILD_ROOT"
    else
        printf 'Build failed; preserving temporary files in %s for resuming.\n' "$BUILD_ROOT" >&2
    fi
    exit "$status"
}
trap cleanup EXIT

if [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
    echo "CUDA toolkit compiler not found at /usr/local/cuda/bin/nvcc" >&2
    exit 1
fi
if [[ ! -f "$ROOT/build_support/patch_pathweaver.py" ]]; then
    echo "Run this script from the LinkTransformer workspace." >&2
    exit 1
fi

mkdir -p "$OUTPUT"
rm -f "$OUTPUT"/pathweaver*.so "$OUTPUT"/cpu_generate_sign_bit*.so
if [[ ! -d "$BUILD_ROOT/upstream/.git" ]]; then
    git init -q "$BUILD_ROOT/upstream"
    git -C "$BUILD_ROOT/upstream" remote add origin https://github.com/AIS-SNU/PathWeaver.git
else
    git -C "$BUILD_ROOT/upstream" remote set-url origin https://github.com/AIS-SNU/PathWeaver.git
fi
git -C "$BUILD_ROOT/upstream" fetch --depth 1 origin "$COMMIT"
git -C "$BUILD_ROOT/upstream" checkout -q --detach FETCH_HEAD
python "$ROOT/build_support/patch_pathweaver.py" "$BUILD_ROOT/upstream" --cuda-arch "$CUDA_ARCH"

# Build tools and intermediate files live in the temporary container layer;
# --target also lets the build run as the host user without changing /opt/venv.
TOOL_PACKAGES="$BUILD_ROOT/python-packages"
python -m pip install --disable-pip-version-check --no-cache-dir \
    --target "$TOOL_PACKAGES" \
    "cmake==3.31.6" "pybind11==2.13.6"
export PYTHONPATH="$TOOL_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$TOOL_PACKAGES/cmake/data/bin:$PATH"
export CUDACXX=/usr/local/cuda/bin/nvcc
export CMAKE_CUDA_COMPILER="$CUDACXX"
export CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
export MAX_JOBS="$JOBS"

SOURCE="$BUILD_ROOT/upstream/pathweaver"
TORCH_CMAKE_PREFIX="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
PYBIND11_CMAKE_DIR="$(python -m pybind11 --cmakedir)"
if [[ ! -f "$OUTPUT/libraft.so" ]] || ! compgen -G "$OUTPUT/cagra_wrapper*.so" >/dev/null; then
    cmake -S "$SOURCE/csrc/cagra_graph" -B "$BUILD_ROOT/cagra" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DCMAKE_CUDA_COMPILER="$CUDACXX" \
        -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" \
        -Dpybind11_DIR="$PYBIND11_CMAKE_DIR" \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$OUTPUT" \
        -DRAFT_NVTX=OFF
    cmake --build "$BUILD_ROOT/cagra" --parallel "$JOBS"
else
    echo "Reusing existing RAFT and CAGRA extensions in $OUTPUT"
fi

(
    cd "$SOURCE"
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCH:0:1}.${CUDA_ARCH:1}" \
        python setup.py build_ext \
            --build-lib "$OUTPUT" --build-temp "$BUILD_ROOT/torch"
)

for module in pathweaver cagra_wrapper cpu_generate_sign_bit; do
    if ! compgen -G "$OUTPUT/${module}*.so" >/dev/null; then
        echo "Missing compiled PathWeaver module: $module" >&2
        exit 1
    fi
done
printf 'PathWeaver extensions ready in %s (commit %s, sm_%s).\n' "$OUTPUT" "$COMMIT" "$CUDA_ARCH"
