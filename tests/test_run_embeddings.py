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
    path = tmp_path / "query.csv"
    expected = pd.DataFrame({"name": ["Joao", "Maria"], "value": [1, 2]})
    expected.to_csv(path, index=False, encoding="utf-8")

    loaded = RUN_EMBEDDINGS.load_dataframe(str(path))

    assert loaded.equals(expected)


def test_load_dataframe_supports_cp1252_csv_with_fallback(tmp_path):
    path = tmp_path / "query.csv"
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


def test_load_dataframe_uses_semicolon_for_base_csv(tmp_path):
    path = tmp_path / "base.csv"
    expected = pd.DataFrame({"name": ["Joao", "Maria"], "value": [1, 2]})
    expected.to_csv(path, index=False, encoding="utf-8", sep=";")

    loaded = RUN_EMBEDDINGS.load_dataframe(str(path))

    assert loaded.equals(expected)


def test_load_dataframe_supports_comma_for_base_csv(tmp_path):
    path = tmp_path / "base.csv"
    expected = pd.DataFrame({"name": ["Joao", "Maria"], "value": [1, 2]})
    expected.to_csv(path, index=False, encoding="utf-8", sep=",")

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


def test_resolve_municipality_columns_defaults_to_id_municipio():
    class Args:
        municipality_column = None
        base_municipality_column = None
        query_municipality_column = None

    df_base = pd.DataFrame({"id_municipio": [1], "name": ["a"]})
    df_query = pd.DataFrame({"id_municipio": [1], "name": ["b"]})

    base_column, query_column = RUN_EMBEDDINGS.resolve_municipality_columns(
        Args,
        df_base=df_base,
        df_query=df_query,
    )

    assert base_column == "id_municipio"
    assert query_column == "id_municipio"


def test_resolve_municipality_columns_accepts_different_side_names():
    class Args:
        municipality_column = None
        base_municipality_column = "cod_mun_base"
        query_municipality_column = "cod_mun_query"

    df_base = pd.DataFrame({"cod_mun_base": [1], "name": ["a"]})
    df_query = pd.DataFrame({"cod_mun_query": [1], "name": ["b"]})

    base_column, query_column = RUN_EMBEDDINGS.resolve_municipality_columns(
        Args,
        df_base=df_base,
        df_query=df_query,
    )

    assert base_column == "cod_mun_base"
    assert query_column == "cod_mun_query"


def test_get_municipality_output_path_uses_side_directory_and_requested_name(tmp_path):
    path = RUN_EMBEDDINGS.get_municipality_output_path(
        output_dir=str(tmp_path),
        side="base",
        model_name="model/a",
        municipality_id=123,
        multiple_models=False,
    )

    assert path == os.path.join(str(tmp_path), "base", "embedding_base_123.npy")


def test_get_municipality_output_path_separates_multiple_models(tmp_path):
    path = RUN_EMBEDDINGS.get_municipality_output_path(
        output_dir=str(tmp_path),
        side="query",
        model_name="model/a",
        municipality_id="35/001",
        multiple_models=True,
    )

    assert path == os.path.join(str(tmp_path), "query", "model_a", "embedding_query_35_001.npy")


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
