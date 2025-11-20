# svs_vamana_indexer.py
import os
import struct
import tempfile
from typing import Optional, Tuple
import numpy as np

import numpy as np
import svs


class VamanaIndexer:
    """
    Constrói um índice Vamana (SVS) a partir de embeddings em memória (np.ndarray).
    - Usa LeanVec (LVQ4 primário / LVQ8 secundário) por padrão.
    - Distância padrão: L2.
    - Persiste .fvecs temporários só para alimentar os loaders do SVS.
    """

    def __init__(self, workdir: Optional[str] = None):
        # workdir opcional; se None, usa diretório temporário
        self._owns_tmpdir = workdir is None
        self.workdir = workdir or tempfile.mkdtemp(prefix="svs_vamana_")
        os.makedirs(self.workdir, exist_ok=True)

        # caminhos dos .fvecs
        self._data_fvecs = os.path.join(self.workdir, "data.fvecs")
        self._queries_fvecs = os.path.join(self.workdir, "queries.fvecs")  # preenchido no search
        self.index = None  # svs.VamanaIndex (após build)

    # ------------------------ helpers internos ------------------------

    @staticmethod
    def _save_fvecs(path: str, X: np.ndarray) -> None:
        """
        Formato .fvecs: para cada vetor: [int32 d] + [d * float32]
        """
        X = np.asarray(X, dtype=np.float32, order="C")
        if X.ndim != 2:
            raise ValueError(f"Esperado shape (N, D); recebido {X.shape}.")
        n, d = X.shape
        with open(path, "wb") as f:
            for i in range(n):
                f.write(struct.pack("i", d))
                f.write(X[i].tobytes(order="C"))

    # ------------------------ API pública ------------------------

    def build(
        self,
        base_embeddings: np.ndarray,
        *,
        reduced_dims: int = 128,
        graph_max_degree: int = 64,
        window_size: int = 128,
        distance: svs.DistanceType = svs.DistanceType.L2,
        num_threads: int = 4,
        primary_kind: svs.LeanVecKind = svs.LeanVecKind.lvq4,
        secondary_kind: svs.LeanVecKind = svs.LeanVecKind.lvq8,
    ) -> None:
        """
        Constrói o índice Vamana a partir dos embeddings da base (N x D).

        Parâmetros chave:
        - reduced_dims: dimensão alvo do LeanVec (ex.: 128 para 768->128).
        - graph_max_degree (R), window_size (alpha) do Vamana.
        - distance: svs.DistanceType.L2, etc.
        """
        base = np.asarray(base_embeddings, dtype=np.float32, order="C")
        if base.ndim != 2:
            raise ValueError(f"base_embeddings deve ter shape (N, D); recebido {base.shape}.")

        # 1) Salvar .fvecs para o loader do SVS
        self._save_fvecs(self._data_fvecs, base)

        # 2) Loaders (não-comprimido -> LeanVec)
        uncompressed_loader = svs.VectorDataLoader(self._data_fvecs, svs.DataType.float32)

        lean_loader = svs.LeanVecLoader(
            uncompressed_loader,
            reduced_dims,
            primary_kind=primary_kind,
            secondary_kind=secondary_kind,
        )

        # 3) Parâmetros e construção do Vamana
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
        
        return self.index

    def search(
        self,
        query_embeddings: np.ndarray,
        *,
        k: int = 10,
        search_window_size: int = 50,
        num_threads: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Busca k vizinhos para as queries (Q x D). Retorna (I, D):
        - I: índices (Q x k)
        - D: distâncias (Q x k)
        """
        
        if self.index is None:
            raise RuntimeError("Índice ainda não foi construído. Chame build() antes de search().")

        queries = np.asarray(query_embeddings, dtype=np.float32, order="C")
        if queries.ndim != 2:
            raise ValueError(f"query_embeddings deve ter shape (Q, D); recebido {queries.shape}.")

        # 1) Salvar .fvecs das queries e ler via SVS
        self._save_fvecs(self._queries_fvecs, queries)
        queries_svs = svs.read_vecs(self._queries_fvecs)

        # 2) Configurar parâmetros de busca
        self.index.search_window_size = search_window_size
        self.index.num_threads = num_threads

        # 3) Buscar
        I, D = self.index.search(queries_svs, 1)
        return I