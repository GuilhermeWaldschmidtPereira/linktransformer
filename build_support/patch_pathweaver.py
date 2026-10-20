#!/usr/bin/env python3
"""Apply the small compatibility fixes needed by this LinkTransformer adapter."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str, description: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Could not apply PathWeaver compatibility patch: {description}")
    path.write_text(text.replace(old, new, 1))


def add_384_kernel(path: Path) -> None:
    text = path.read_text()
    if "case 384:" in text:
        return
    start = text.find("        case 256:")
    end = text.find("        case 96:", start)
    if start < 0 or end < 0:
        raise RuntimeError("Could not find PathWeaver's 256D kernel dispatch to extend it to 384D")
    block = text[start:end]
    if "search_kernel<256," not in block:
        raise RuntimeError("PathWeaver's 256D kernel block has an unexpected format")
    block = block.replace("case 256:", "case 384:", 1)
    block = block.replace("nytimes-256-inner", "LinkTransformer-384-inner", 1)
    block = block.replace("search_kernel<256,", "search_kernel<384,")
    path.write_text(text[:end] + block + text[end:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--cuda-arch", default="86")
    args = parser.parse_args()

    source = args.source.resolve()
    pathweaver_dir = source / "pathweaver"
    add_384_kernel(pathweaver_dir / "csrc" / "pathweaver.cu")

    wrapper = pathweaver_dir / "csrc" / "cagra_graph" / "cagra_wrapper.cu"
    replace_once(
        wrapper,
        "return py::array_t<uint32_t>({graph_rows, graph_cols}, h_graph.data());",
        """cudaStreamSynchronize(dev_resources.get_stream());
    py::array_t<uint32_t> graph_array({graph_rows, graph_cols});
    std::memcpy(graph_array.mutable_data(), h_graph.data(), h_graph.size() * sizeof(uint32_t));
    return graph_array;""",
        "finish the graph copy and return an owned NumPy array",
    )
    replace_once(
        wrapper,
        "CagraWrapper() : dev_resources() {\n        rmm::mr::set_current_device_resource(&pool_mr);\n    }",
        "CagraWrapper() : dev_resources(), previous_mr(rmm::mr::get_current_device_resource()) {\n        rmm::mr::set_current_device_resource(&pool_mr);\n    }\n\n    ~CagraWrapper() {\n        rmm::mr::set_current_device_resource(previous_mr);\n    }",
        "restore the previous RMM allocator when the CAGRA wrapper is destroyed",
    )
    replace_once(
        wrapper,
        "    py::array_t<uint32_t> result_neighbors;",
        "    rmm::mr::device_memory_resource* previous_mr;\n\n    py::array_t<uint32_t> result_neighbors;",
        "store the prior RMM allocator for RAII restoration",
    )
    text = wrapper.read_text()
    if "#include <cstring>" not in text:
        text = text.replace("#include <cstdint>", "#include <cstdint>\n#include <cstring>", 1)
    if "#include <cuda_runtime.h>" not in text:
        text = text.replace("#include <cstring>", "#include <cstring>\n#include <cuda_runtime.h>", 1)
    if text != wrapper.read_text():
        wrapper.write_text(text)

    cmake = pathweaver_dir / "csrc" / "cagra_graph" / "CMakeLists.txt"
    replace_once(
        cmake,
        "include_directories(/nfs/home/daisy1212/cagra-extended/local_build)",
        "# Use the RAFT and RMM headers provided by the configured CMake targets.\n"
        "# The upstream local_build copy is incomplete and shadows RAFT's mdspan headers.",
        "avoid the incomplete vendored RAFT headers that shadow the configured dependency",
    )

    setup = pathweaver_dir / "setup.py"
    replace_once(
        setup,
        "-arch=sm_80",
        f"-arch=sm_{args.cuda_arch}",
        "compile PathWeaver for the server's compute capability",
    )


if __name__ == "__main__":
    main()
