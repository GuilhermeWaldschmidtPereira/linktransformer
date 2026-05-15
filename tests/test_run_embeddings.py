import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_linktransformer" / "run_embeddings.py"
SPEC = importlib.util.spec_from_file_location("run_embeddings", MODULE_PATH)
RUN_EMBEDDINGS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUN_EMBEDDINGS)


def test_load_dataframe_supports_utf8_csv(tmp_path):
    path = tmp_path / "input.csv"
    expected = pd.DataFrame({"name": ["Joao", "Maria"], "value": [1, 2]})
    expected.to_csv(path, index=False, encoding="utf-8")

    loaded = RUN_EMBEDDINGS.load_dataframe(str(path))

    assert loaded.equals(expected)


def test_load_dataframe_supports_cp1252_csv_with_fallback(tmp_path):
    path = tmp_path / "input.csv"
    expected = pd.DataFrame({"name": ["José", "Maçã"], "value": [1, 2]})
    expected.to_csv(path, index=False, encoding="cp1252")

    loaded = RUN_EMBEDDINGS.load_dataframe(str(path))

    assert loaded.equals(expected)


def test_load_dataframe_supports_parquet(tmp_path):
    path = tmp_path / "input.parquet"
    expected = pd.DataFrame({"name": ["a", "b"], "value": [1, 2]})
    expected.to_parquet(path, index=False)

    loaded = RUN_EMBEDDINGS.load_dataframe(str(path))

    assert loaded.equals(expected)


def test_resolve_single_side_columns_uses_explicit_side():
    resolved = RUN_EMBEDDINGS.resolve_single_side_columns(["coluna_a"], "fallback", "right")
    assert resolved == "coluna_a"


def test_resolve_single_side_columns_uses_fallback():
    resolved = RUN_EMBEDDINGS.resolve_single_side_columns(None, ["coluna_a", "coluna_b"], "right")
    assert resolved == ["coluna_a", "coluna_b"]


def test_build_single_manifest_entry_for_query():
    manifest = RUN_EMBEDDINGS.build_single_manifest_entry("meu-modelo", "/tmp/query.npy", "query")
    assert manifest["mode"] == "query"
    assert manifest["query_embeddings_path"] == "/tmp/query.npy"
    assert "base_embeddings_path" not in manifest


def test_build_output_paths_generates_base_and_query_names(tmp_path):
    base_path, query_path = RUN_EMBEDDINGS.build_output_paths(str(tmp_path), "sentence-transformers/all-MiniLM-L6-v2")
    assert base_path.endswith("embeddings_base_sentence-transformers_all-MiniLM-L6-v2.npy")
    assert query_path.endswith("embeddings_query_sentence-transformers_all-MiniLM-L6-v2.npy")


def test_add_timing_to_manifest_adds_seconds_and_minutes():
    manifest = RUN_EMBEDDINGS.add_timing_to_manifest({"model": "m1"}, 120.0)
    assert manifest["elapsed_seconds"] == 120.0
    assert manifest["elapsed_minutes"] == 2.0


def test_add_memory_to_manifest_adds_memory_fields():
    manifest = RUN_EMBEDDINGS.add_memory_to_manifest({"model": "m1"}, 100.0, 130.0, 145.0)
    assert manifest["memory_start_mb"] == 100.0
    assert manifest["memory_end_mb"] == 130.0
    assert manifest["memory_peak_mb"] == 145.0
    assert manifest["memory_delta_mb"] == 30.0
