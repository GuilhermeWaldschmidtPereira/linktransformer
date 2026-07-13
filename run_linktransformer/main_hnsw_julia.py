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

from main_linktransformer import (  # noqa: E402
    assert_embeddings_exist,
    load_input_data,
    prepare_dataframes,
)
from linktransformer.global_chunking import release_native_memory  # noqa: E402
from linktransformer.infer_main import (  # noqa: E402
    merge_knn_hnsw_julia,
    merge_knn_hnsw_julia_global,
)
from model_selection import (  # noqa: E402
    MODELOS_A_UTILIZAR,
    parse_embedding_model_args,
    resolve_embedding_models,
)


def main() -> None:
    args = parse_embedding_model_args(
        "Executa a indexacao HNSW Julia com embeddings pre-computados.",
    )
    modelos_a_executar = resolve_embedding_models(args.models)

    print(f">>> Modelos disponiveis: {MODELOS_A_UTILIZAR}", flush=True)
    print(f">>> Modelos selecionados: {modelos_a_executar}", flush=True)
    print(f">>> Escopo selecionado: {args.scope}", flush=True)

    df_base, df_query = load_input_data()
    suffixes = ("_x", "_y")
    ks = [1]

    for modelo in modelos_a_executar:
        assert_embeddings_exist(modelo, args.scope)
        df1, df2 = prepare_dataframes(df_base, df_query)

        for k in ks:
            print(f">>> Rodando HNSW Julia | modelo={modelo} | k={k} | scope={args.scope}")
            if args.scope == "geral":
                merge_knn_hnsw_julia_global(k, df1, df2, suffixes, modelo)
            else:
                merge_knn_hnsw_julia(k, df1, df2, suffixes, modelo)
            release_native_memory()


if __name__ == "__main__":
    main()
