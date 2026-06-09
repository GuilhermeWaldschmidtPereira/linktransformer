#!/usr/bin/env python3
import argparse
import os
from typing import List, Optional, Sequence


AVAILABLE_EMBEDDING_MODELS: List[str] = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/multilingual-e5-large",
    "neuralmind/bert-large-portuguese-cased",
]

# Mantem compatibilidade com os scripts que ja importavam este nome.
MODELOS_A_UTILIZAR = AVAILABLE_EMBEDDING_MODELS


def safe_model_name(modelo: str) -> str:
    safe_model = str(modelo).replace(os.sep, "_")
    if os.path.altsep:
        safe_model = safe_model.replace(os.path.altsep, "_")
    return safe_model


def resolve_embedding_models(selected_models: Optional[Sequence[str]]) -> List[str]:
    if not selected_models:
        return list(AVAILABLE_EMBEDDING_MODELS)

    aliases = {}
    for model in AVAILABLE_EMBEDDING_MODELS:
        aliases[model] = model
        aliases[safe_model_name(model)] = model

    resolved_models = []
    invalid_models = []
    for selected_model in selected_models:
        if selected_model in {"all", "todos", "*"}:
            return list(AVAILABLE_EMBEDDING_MODELS)

        resolved_model = aliases.get(selected_model)
        if resolved_model is None:
            invalid_models.append(selected_model)
        elif resolved_model not in resolved_models:
            resolved_models.append(resolved_model)

    if invalid_models:
        available = "\n".join(
            f"- {model} (ou {safe_model_name(model)})"
            for model in AVAILABLE_EMBEDDING_MODELS
        )
        invalid = ", ".join(invalid_models)
        raise ValueError(
            f"Modelo(s) de embedding invalido(s): {invalid}\n"
            "Use um dos modelos disponiveis:\n"
            f"{available}"
        )

    return resolved_models


def parse_embedding_model_args(
    description: str,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=None,
        help=(
            "Modelo de embedding para indexar. Pode repetir para rodar varios. "
            "Se omitido, roda todos. Tambem aceita: all, todos ou *."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=("municipio", "geral"),
        default="municipio",
        help=(
            "Escopo da indexacao/consulta. 'municipio' recria o comportamento atual, "
            "com um indice por municipio. 'geral' cria um indice unico para toda a base."
        ),
    )
    return parser.parse_args(argv)
