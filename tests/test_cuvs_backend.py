import sys
import types

import numpy as np

from linktransformer.infer_main import _build_cuvs_index, _search_cuvs


def install_fake_cuvs(monkeypatch, distances=None, neighbors=None):
    calls = {}

    class FakeCagra:
        class IndexParams:
            def __init__(self, **kwargs):
                calls["index_params"] = kwargs

        class SearchParams:
            def __init__(self, **kwargs):
                calls["search_params"] = kwargs

        @staticmethod
        def build(params, dataset):
            calls["dataset"] = dataset
            return object()

        @staticmethod
        def search(params, index, queries, k):
            calls["queries"] = queries
            calls["k"] = k
            return distances, neighbors

    cupy = types.ModuleType("cupy")
    cupy.float32 = np.float32
    cupy.asarray = lambda array, dtype=None: np.asarray(array, dtype=dtype)
    cupy.asnumpy = np.asarray

    cuvs = types.ModuleType("cuvs")
    cuvs.__path__ = []
    cuvs_neighbors = types.ModuleType("cuvs.neighbors")
    cuvs_neighbors.cagra = FakeCagra
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "cuvs", cuvs)
    monkeypatch.setitem(sys.modules, "cuvs.neighbors", cuvs_neighbors)
    return calls


def test_build_cuvs_index_uses_contiguous_float32_and_bounded_graph(monkeypatch):
    calls = install_fake_cuvs(monkeypatch)
    monkeypatch.setenv("CUVS_GRAPH_DEGREE", "64")
    data = np.arange(6, dtype=np.float64).reshape(3, 2)[:, ::-1]

    _, index, device_data = _build_cuvs_index(data)

    assert index is not None
    assert device_data.dtype == np.float32
    assert device_data.flags.c_contiguous
    assert calls["index_params"] == {
        "metric": "cosine",
        "graph_degree": 2,
        "intermediate_graph_degree": 2,
        "build_algo": "ivf_pq",
    }


def test_search_cuvs_converts_cosine_distances_to_scores(monkeypatch):
    distances = np.array([[0.0, 0.25], [1.25, 0.5]], dtype=np.float32)
    neighbors = np.array([[0, 2], [1, 0]], dtype=np.int32)
    calls = install_fake_cuvs(monkeypatch, distances, neighbors)
    base = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)[:, ::-1]

    index_bundle = _build_cuvs_index(base)
    result_neighbors, scores, result_k = _search_cuvs(index_bundle, queries, 2)

    assert np.array_equal(result_neighbors, neighbors)
    assert np.allclose(scores, [[1.0, 0.75], [-0.25, 0.5]])
    assert result_k == 2
    assert calls["queries"].dtype == np.float32
    assert calls["queries"].flags.c_contiguous
    assert calls["search_params"] == {"itopk_size": 64}
    assert calls["k"] == 2


def test_cuvs_handles_single_row_groups_without_gpu():
    base = np.array([[0.0, 1.0]], dtype=np.float32)
    queries = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    index_bundle = _build_cuvs_index(base)
    neighbors, scores, result_k = _search_cuvs(index_bundle, queries, 1)

    assert np.array_equal(neighbors, [[0], [0]])
    assert np.allclose(scores, [[1.0], [0.0]])
    assert result_k == 1