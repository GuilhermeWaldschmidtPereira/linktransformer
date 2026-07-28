from __future__ import annotations

import gc
import os
from typing import Iterator, List, Tuple

import numpy as np


DEFAULT_GLOBAL_BASE_CHUNK_SIZE = 50_000
DEFAULT_GLOBAL_QUERY_BATCH_SIZE = 10_000


def append_data_dir_candidates(
    repo_root: str,
    env_data_dir: str | None,
) -> List[str]:
    candidates: List[str] = []
    if env_data_dir:
        abs_env_data_dir = os.path.abspath(env_data_dir)
        candidates.append(abs_env_data_dir)

        parent_dir = os.path.dirname(abs_env_data_dir)
        if parent_dir and parent_dir != abs_env_data_dir:
            candidates.append(parent_dir)

    candidates.extend([
        os.path.join(repo_root, "data"),
        os.path.join(repo_root, "linktransformer", "data"),
    ])
    return list(dict.fromkeys(candidates))


def get_global_base_chunk_size(total_rows: int) -> int:
    value = os.environ.get("LINKTRANSFORMER_GLOBAL_BASE_CHUNK_SIZE")
    if value not in (None, ""):
        return max(1, min(int(value), total_rows))
    return max(1, min(DEFAULT_GLOBAL_BASE_CHUNK_SIZE, total_rows))


def get_global_query_batch_size(total_rows: int) -> int:
    value = os.environ.get("LINKTRANSFORMER_GLOBAL_QUERY_BATCH_SIZE")
    if value not in (None, ""):
        return max(1, min(int(value), total_rows))
    return max(1, min(DEFAULT_GLOBAL_QUERY_BATCH_SIZE, total_rows))


def should_use_chunked_global_search(total_rows: int) -> bool:
    value = os.environ.get("LINKTRANSFORMER_GLOBAL_CHUNKED")
    if value is not None:
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return total_rows > DEFAULT_GLOBAL_BASE_CHUNK_SIZE


def iter_row_ranges(total_rows: int, batch_size: int) -> Iterator[Tuple[int, int]]:
    effective_batch_size = max(1, batch_size)
    for start in range(0, total_rows, effective_batch_size):
        end = min(start + effective_batch_size, total_rows)
        yield start, end


def init_topk_buffers(
    num_rows: int,
    k: int,
    prefer_higher_scores: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    worst_score = -np.inf if prefer_higher_scores else np.inf
    scores = np.full((num_rows, k), worst_score, dtype=np.float32)
    indices = np.full((num_rows, k), -1, dtype=np.int64)
    return scores, indices


def merge_topk_buffers(
    current_scores: np.ndarray,
    current_indices: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_indices: np.ndarray,
    k: int,
    prefer_higher_scores: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    combined_scores = np.concatenate([current_scores, candidate_scores], axis=1)
    combined_indices = np.concatenate([current_indices, candidate_indices], axis=1)

    if prefer_higher_scores:
        order = np.argsort(-combined_scores, axis=1)[:, :k]
    else:
        order = np.argsort(combined_scores, axis=1)[:, :k]

    row_ids = np.arange(combined_scores.shape[0])[:, None]
    merged_scores = combined_scores[row_ids, order]
    merged_indices = combined_indices[row_ids, order]
    return merged_scores, merged_indices


def release_native_memory() -> None:
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
