#!/usr/bin/env python3
import os
import sys


THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Importa nmslib antes dos outros módulos C++/pybind do projeto.
import nmslib  # noqa: F401, E402

from main_linktransformer import (  # noqa: E402
    MODELOS_A_UTILIZAR,
    assert_embeddings_exist,
    load_input_data,
    prepare_dataframes,
)
from linktransformer.infer_main import merge_knn_nmslib  # noqa: E402


def main() -> None:
    df_base, df_query = load_input_data()
    suffixes = ("_x", "_y")
    ks = [1]

    for modelo in MODELOS_A_UTILIZAR:
        assert_embeddings_exist(modelo)
        df1, df2 = prepare_dataframes(df_base, df_query)

        for k in ks:
            print(f">>> Rodando NMSLIB | modelo={modelo} | k={k}")
            merge_knn_nmslib(k, df1, df2, suffixes, modelo)


if __name__ == "__main__":
    main()
