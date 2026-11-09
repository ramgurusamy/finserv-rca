"""Embedding backends behind one interface.

wordllama ships its weights inside the pip package, so the offline profile needs
no model download. fastembed (BGE-small) is the better laptop default.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class WordLlamaEmbedder:
    name = "wordllama-l2-supercat-256"
    dim = 256

    def __init__(self) -> None:
        import wordllama
        from wordllama import WordLlama

        # The package bundles its files; pointing cache_dir there avoids a download.
        self._model = WordLlama.load(cache_dir=Path(wordllama.__file__).parent)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self._model.embed(texts), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-9, None)


class FastEmbedEmbedder:
    def __init__(self, model: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)
        self.name = model
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.embed(texts)), dtype=np.float32)


@lru_cache(maxsize=4)
def get_embedder(backend: str, model: str) -> Embedder:
    if backend == "wordllama":
        return WordLlamaEmbedder()
    if backend == "fastembed":
        return FastEmbedEmbedder(model)
    raise ValueError(f"Unknown embedding backend: {backend}")
