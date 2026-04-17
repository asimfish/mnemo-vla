"""Pluggable embedders for the persistent memory store.

All embedders conform to the ``Embedder`` protocol in ``memory_store.py``
and expose a ``dim`` attribute plus an ``encode(text) -> np.ndarray`` method.

Two concrete implementations ship here:

* ``HashingEmbedder`` (re-exported from memory_store for convenience)
  is zero-dependency and deterministic; ideal for unit tests.
* ``SentenceTransformerEmbedder`` wraps the ``sentence-transformers``
  library; install the ``embeddings`` extra to use it.

Users can drop in any custom embedder that follows the same protocol.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .memory_store import HashingEmbedder


class SentenceTransformerEmbedder:
    """Semantic embedder backed by sentence-transformers.

    Example:
        from mnemo.persistent_memory.embedders import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        vec = embedder.encode("The knife is in the left drawer.")

    The sentence-transformers package is an optional dependency. Install
    it via ``pip install mnemo-vla[embeddings]``.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433
        except ImportError as err:  # pragma: no cover - tested by extras
            raise ImportError(
                "sentence-transformers is not installed. Install the "
                "optional extra via `pip install mnemo-vla[embeddings]`."
            ) from err

        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec.astype(np.float32)


class StackedEmbedder:
    """Wrap multiple embedders and concatenate their outputs.

    Useful when combining a lightweight lexical embedder (hashing) with
    a semantic one so that the store captures both surface-level and
    meaning-level similarity.
    """

    def __init__(self, embedders: list):
        if not embedders:
            raise ValueError("StackedEmbedder requires at least one embedder.")
        self._embedders = list(embedders)
        self.dim = sum(int(e.dim) for e in self._embedders)

    def encode(self, text: str) -> np.ndarray:
        vecs = [e.encode(text) for e in self._embedders]
        return np.concatenate(vecs).astype(np.float32)


__all__ = [
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "StackedEmbedder",
]
