# svs_vamana_indexer.py
import os
import struct
import tempfile
from typing import Any, Optional, Tuple

import numpy as np
import svs


class _NumpyL2Index:
    """Fallback simples quando a API esperada do SVS nao esta disponivel."""

    def __init__(self, base_embeddings: np.ndarray):
        self._base = np.asarray(base_embeddings, dtype=np.float32, order="C")

    def search(self, query_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(query_embeddings, dtype=np.float32, order="C")

        # Distancia L2 vetorizada: ||q-b||^2 = ||q||^2 + ||b||^2 - 2*q.b
        q2 = np.sum(queries * queries, axis=1, keepdims=True)
        b2 = np.sum(self._base * self._base, axis=1, keepdims=True).T
        distances = q2 + b2 - 2.0 * queries @ self._base.T

        k = max(1, min(int(k), self._base.shape[0]))
        idx_part = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        part_distances = np.take_along_axis(distances, idx_part, axis=1)

        order = np.argsort(part_distances, axis=1)
        indices = np.take_along_axis(idx_part, order, axis=1).astype(np.int64)
        dists = np.take_along_axis(part_distances, order, axis=1).astype(np.float32)
        return indices, dists


class VamanaIndexer:
    """
    Constrói um indice tipo Vamana (SVS) a partir de embeddings em memoria.
    Quando a API do SVS nao possui os simbolos esperados, cai para um fallback
    de busca L2 vetorizada em NumPy para manter o pipeline funcional.
    """

    def __init__(self, workdir: Optional[str] = None):
        self._owns_tmpdir = workdir is None
        self.workdir = workdir or tempfile.mkdtemp(prefix="svs_vamana_")
        os.makedirs(self.workdir, exist_ok=True)

        self._data_fvecs = os.path.join(self.workdir, "data.fvecs")
        self._queries_fvecs = os.path.join(self.workdir, "queries.fvecs")
        self.index = None
        self._backend = None

    @staticmethod
    def _save_fvecs(path: str, X: np.ndarray) -> None:
        """Formato .fvecs: [int32 d] + [d * float32] para cada vetor."""
        X = np.asarray(X, dtype=np.float32, order="C")
        if X.ndim != 2:
            raise ValueError(f"Esperado shape (N, D); recebido {X.shape}.")

        _, d = X.shape
        with open(path, "wb") as f:
            for row in X:
                f.write(struct.pack("i", d))
                f.write(row.tobytes(order="C"))

    def _can_use_svs_vamana(self) -> bool:
        required_symbols = [
            "VectorDataLoader",
            "DataType",
            "LeanVecLoader",
            "VamanaBuildParameters",
            "Vamana",
            "read_vecs",
        ]
        return all(hasattr(svs, symbol) for symbol in required_symbols)

    def build(
        self,
        base_embeddings: np.ndarray,
        *,
        reduced_dims: int = 128,
        graph_max_degree: int = 64,
        window_size: int = 128,
        distance: Any = "L2",
        num_threads: int = 4,
        primary_kind: Any = "lvq4",
        secondary_kind: Any = "lvq8",
        **kwargs: Any,
    ) -> "VamanaIndexer":
        """Constroi o indice e retorna o proprio wrapper para busca uniforme."""
        base = np.asarray(base_embeddings, dtype=np.float32, order="C")
        if base.ndim != 2:
            raise ValueError(f"base_embeddings deve ter shape (N, D); recebido {base.shape}.")

        # Compatibilidade com chamadas antigas: M/ef_construction
        graph_max_degree = int(kwargs.get("M", graph_max_degree))
        window_size = int(kwargs.get("ef_construction", window_size))

        if self._can_use_svs_vamana():
            self._save_fvecs(self._data_fvecs, base)
            uncompressed_loader = svs.VectorDataLoader(self._data_fvecs, svs.DataType.float32)
            lean_loader = svs.LeanVecLoader(
                uncompressed_loader,
                reduced_dims,
                primary_kind=primary_kind,
                secondary_kind=secondary_kind,
            )
            build_params = svs.VamanaBuildParameters(
                graph_max_degree=graph_max_degree,
                window_size=window_size,
            )
            self.index = svs.Vamana.build(
                build_params,
                lean_loader,
                distance,
                num_threads=num_threads,
            )
            self._backend = "svs"
            return self

        self.index = _NumpyL2Index(base)
        self._backend = "numpy"
        return self

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 1,
        search_window_size: int = 50,
        num_threads: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Busca k vizinhos e retorna (indices, distancias)."""
        if self.index is None:
            raise RuntimeError("Indice ainda nao foi construido. Chame build() antes de search().")

        queries = np.asarray(query_embeddings, dtype=np.float32, order="C")
        if queries.ndim != 2:
            raise ValueError(f"query_embeddings deve ter shape (Q, D); recebido {queries.shape}.")

        k = int(k)
        if self._backend == "svs":
            self._save_fvecs(self._queries_fvecs, queries)
            queries_svs = svs.read_vecs(self._queries_fvecs)

            self.index.search_window_size = search_window_size
            self.index.num_threads = num_threads

            first, second = self.index.search(queries_svs, k)
            first_np = np.asarray(first)
            second_np = np.asarray(second)

            if first_np.dtype.kind in ("i", "u"):
                return first_np.astype(np.int64), second_np.astype(np.float32)
            if second_np.dtype.kind in ("i", "u"):
                return second_np.astype(np.int64), first_np.astype(np.float32)
            return first_np.astype(np.int64), second_np.astype(np.float32)

        return self.index.search(queries, k)