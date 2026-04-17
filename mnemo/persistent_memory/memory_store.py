"""FAISS-backed persistent memory store for cross-session retrieval.

The MemoryStore indexes EpisodeSummary objects by their text embedding
and supports top-K similarity retrieval at the start of a new episode.

Design notes
------------
- We keep the FAISS index and the JSON-serialized summaries side by side
  on disk so that the store is easy to inspect and migrate.
- The embedder is pluggable: this skeleton ships with a light-weight
  HashingEmbedder for tests, and a SentenceTransformerEmbedder for
  real usage. Users can drop in Qwen3-VL or any other encoder.
- The index file grows monotonically; we rely on a later garbage-collect
  step (TODO) to prune stale entries when it gets too large.

Dependencies are kept optional: faiss, numpy and sentence-transformers
are imported lazily so that basic unit tests can run without them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np

from .episode_summary import EpisodeSummary


class Embedder(Protocol):
    """Protocol for any object that can embed text to a fixed-dim vector."""

    dim: int

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a (dim,) float32 numpy array."""
        ...


class HashingEmbedder:
    """Deterministic, dependency-free embedder for tests.

    Hashes the text into a fixed-dimension bucket vector. Not semantically
    meaningful -- only use for wiring tests.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in text.split():
            bucket = hash(token) % self.dim
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


@dataclass
class RetrievalHit:
    """One retrieved episode summary plus its relevance score."""

    summary: EpisodeSummary
    score: float


class MemoryStore:
    """Persistent memory store backed by FAISS (optional) or a flat index.

    The store keeps two files on disk:
      - summaries.jsonl: one JSON record per line, for EpisodeSummary data.
      - vectors.npy: matrix of (N, dim) float32 vectors.

    For simplicity the skeleton uses numpy for retrieval. Swap in
    faiss.IndexFlatIP for production to scale beyond 100K entries.
    """

    def __init__(
        self,
        root: str | Path,
        embedder: Optional[Embedder] = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbedder(dim=64)
        self._summaries_path = self.root / "summaries.jsonl"
        self._vectors_path = self.root / "vectors.npy"
        self._summaries: List[EpisodeSummary] = []
        self._vectors: Optional[np.ndarray] = None
        self._load()

    # ------------------------------------------------------------------ IO

    def _load(self) -> None:
        """Load summaries and vectors from disk if present."""
        if self._summaries_path.exists():
            with self._summaries_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    self._summaries.append(EpisodeSummary.from_dict(data))

        if self._vectors_path.exists():
            self._vectors = np.load(self._vectors_path)
        elif self._summaries:
            # Re-embed if the vectors file is missing but summaries exist.
            self._rebuild_vectors()

    def _rebuild_vectors(self) -> None:
        vectors = np.stack(
            [self.embedder.encode(s.to_text()) for s in self._summaries], axis=0
        )
        self._vectors = vectors.astype(np.float32)
        np.save(self._vectors_path, self._vectors)

    def _persist(self, summary: EpisodeSummary, vector: np.ndarray) -> None:
        # Append summary JSONL.
        with self._summaries_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(summary.to_dict(), ensure_ascii=False) + "\n")
        # Update vectors matrix on disk.
        if self._vectors is None:
            self._vectors = vector.reshape(1, -1).astype(np.float32)
        else:
            self._vectors = np.concatenate(
                [self._vectors, vector.reshape(1, -1).astype(np.float32)],
                axis=0,
            )
        np.save(self._vectors_path, self._vectors)

    # ------------------------------------------------------------------ API

    def __len__(self) -> int:
        return len(self._summaries)

    def add(self, summary: EpisodeSummary) -> None:
        """Index a new episode summary and persist it."""
        vector = self.embedder.encode(summary.to_text())
        self._summaries.append(summary)
        self._persist(summary, vector)

    def query(self, text: str, top_k: int = 3) -> List[RetrievalHit]:
        """Return the top-K most similar episodes to the given text.

        The score is cosine similarity (vectors are L2-normalized by the
        default HashingEmbedder, and we re-normalize here just in case).
        """
        if not self._summaries or self._vectors is None:
            return []

        query_vec = self.embedder.encode(text).reshape(1, -1)
        query_vec = _l2_normalize(query_vec)
        stored = _l2_normalize(self._vectors)
        scores = (stored @ query_vec.T).reshape(-1)  # (N,)

        # Top-K selection without requiring faiss.
        top_k = min(top_k, len(scores))
        top_idx = np.argsort(-scores)[:top_k]
        return [
            RetrievalHit(summary=self._summaries[i], score=float(scores[i]))
            for i in top_idx
        ]

    def all_summaries(self) -> List[EpisodeSummary]:
        return list(self._summaries)

    def clear(self) -> None:
        """Remove every stored summary (used mainly by tests)."""
        self._summaries = []
        self._vectors = None
        if self._summaries_path.exists():
            os.remove(self._summaries_path)
        if self._vectors_path.exists():
            os.remove(self._vectors_path)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with a small epsilon for stability."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return matrix / norms
