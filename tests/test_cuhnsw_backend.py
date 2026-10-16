import numpy as np

from linktransformer.infer_main import _build_cuhnsw_index, _search_cuhnsw


class FakeCuHNSW:
    def __init__(self, options):
        self.options = options
        self.data = None
        self.built = False

    def set_data(self, data):
        self.data = data

    def build(self):
        self.built = True


class FakeSearchIndex:
    def __init__(self, result):
        self.result = result
        self.call = None

    def search_knn(self, queries, topk, ef_search):
        self.call = (queries, topk, ef_search)
        return self.result


def test_build_cuhnsw_index_uses_contiguous_float32(monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "cuhnsw", types.SimpleNamespace(CuHNSW=FakeCuHNSW))
    data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)[:, ::-1]

    index = _build_cuhnsw_index(data)

    assert index.built
    assert index.data.dtype == np.float32
    assert index.data.flags.c_contiguous
    assert index.options["dist_type"] == "cosine"
    assert index.options["nrz"] is True


def test_search_cuhnsw_converts_neighbors_to_cosine_scores(monkeypatch):
    monkeypatch.setenv("CUHNSW_EF_SEARCH", "7")
    base = np.array([[1.0, 0.0], [0.0, 1.0], [0.70710677, 0.70710677]], dtype=np.float32)
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = FakeSearchIndex((
        np.array([[0, 2], [1, 999]], dtype=np.int32),
        np.zeros((2, 2), dtype=np.float32),
        np.array([2, 1], dtype=np.int32),
    ))

    neighbors, scores, result_k = _search_cuhnsw(index, base, queries, 2)

    assert result_k == 1
    assert np.array_equal(neighbors, np.array([[0], [1]]))
    assert np.allclose(scores, [[1.0], [1.0]])
    assert index.call[1:] == (2, 7)