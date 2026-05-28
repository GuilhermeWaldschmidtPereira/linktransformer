import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PARTITIONED_MODULE_PATH = Path(__file__).resolve().parents[1] / "run_linktransformer" / "run_embeddings_partitioned.py"
PARTITIONED_SPEC = importlib.util.spec_from_file_location("run_embeddings_partitioned", PARTITIONED_MODULE_PATH)
RUN_EMBEDDINGS_PARTITIONED = importlib.util.module_from_spec(PARTITIONED_SPEC)
assert PARTITIONED_SPEC.loader is not None
PARTITIONED_SPEC.loader.exec_module(RUN_EMBEDDINGS_PARTITIONED)

MERGE_MODULE_PATH = Path(__file__).resolve().parents[1] / "run_linktransformer" / "merge_partitions.py"
MERGE_SPEC = importlib.util.spec_from_file_location("merge_partitions", MERGE_MODULE_PATH)
MERGE_PARTITIONS = importlib.util.module_from_spec(MERGE_SPEC)
assert MERGE_SPEC.loader is not None
MERGE_SPEC.loader.exec_module(MERGE_PARTITIONS)


def test_count_rows_and_iter_dataframe_partitions_for_csv(tmp_path):
    path = tmp_path / "query.csv"
    df = pd.DataFrame({"name": ["a", "b", "c", "d", "e"], "value": [1, 2, 3, 4, 5]})
    df.to_csv(path, index=False)

    assert RUN_EMBEDDINGS_PARTITIONED.count_rows(str(path)) == 5

    partitions = list(
        RUN_EMBEDDINGS_PARTITIONED.iter_dataframe_partitions(
            str(path),
            partition_size=2,
        )
    )

    assert [len(partition) for partition in partitions] == [2, 2, 1]
    assert partitions[0].iloc[0]["name"] == "a"
    assert partitions[-1].iloc[0]["name"] == "e"


def test_count_rows_and_load_columns_for_parquet(tmp_path):
    path = tmp_path / "input.parquet"
    df = pd.DataFrame({"left": ["x", "y"], "right": [10, 20]})
    df.to_parquet(path, index=False)

    assert RUN_EMBEDDINGS_PARTITIONED.count_rows(str(path)) == 2

    loaded = RUN_EMBEDDINGS_PARTITIONED.load_dataframe_columns(str(path))

    assert list(loaded.columns) == ["left", "right"]


def test_write_to_merged_memmap_writes_partitions_without_concatenate(tmp_path):
    output_path = tmp_path / "merged.npy"
    offset = 0
    merged_memmap = None

    first = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    second = np.array([[5.0, 6.0]], dtype=np.float32)

    merged_memmap, offset = RUN_EMBEDDINGS_PARTITIONED.write_to_merged_memmap(
        merged_memmap,
        merged_output_path=str(output_path),
        total_rows=3,
        offset=offset,
        embeddings_partition=first,
    )
    merged_memmap, offset = RUN_EMBEDDINGS_PARTITIONED.write_to_merged_memmap(
        merged_memmap,
        merged_output_path=str(output_path),
        total_rows=3,
        offset=offset,
        embeddings_partition=second,
    )
    merged_memmap.flush()
    del merged_memmap

    merged = np.load(output_path)

    assert offset == 3
    assert merged.shape == (3, 2)
    assert np.allclose(merged, np.vstack([first, second]))


def test_merge_partitions_merges_saved_arrays_to_disk(tmp_path):
    partition_dir = tmp_path / "model_a"
    partition_dir.mkdir()

    np.save(partition_dir / "partition_000000_base.npy", np.array([[1.0, 2.0]], dtype=np.float32))
    np.save(partition_dir / "partition_000001_base.npy", np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32))

    output_path = tmp_path / "embeddings_base_model_a.npy"
    num_partitions, total_rows = MERGE_PARTITIONS.merge_partitions(
        partition_dir=partition_dir,
        side="base",
        output_path=str(output_path),
        verbose=False,
    )

    merged = np.load(output_path)

    assert num_partitions == 2
    assert total_rows == 3
    assert merged.shape == (3, 2)
    assert np.allclose(
        merged,
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )


def test_merge_partitions_can_delete_inputs_after_write(tmp_path):
    partition_dir = tmp_path / "model_a"
    partition_dir.mkdir()

    first_path = partition_dir / "partition_000000_query.npy"
    second_path = partition_dir / "partition_000001_query.npy"
    np.save(first_path, np.array([[1.0, 2.0]], dtype=np.float32))
    np.save(second_path, np.array([[3.0, 4.0]], dtype=np.float32))

    output_path = tmp_path / "embeddings_query_model_a.npy"
    num_partitions, total_rows = MERGE_PARTITIONS.merge_partitions(
        partition_dir=partition_dir,
        side="query",
        output_path=str(output_path),
        verbose=False,
        delete_partitions=True,
    )

    merged = np.load(output_path)

    assert num_partitions == 2
    assert total_rows == 2
    assert merged.shape == (2, 2)
    assert not first_path.exists()
    assert not second_path.exists()
