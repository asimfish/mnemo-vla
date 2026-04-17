"""Run the synthetic cross-session retrieval benchmark.

Usage::

    PYTHONPATH=. python examples/retrieval_benchmark.py

The script seeds a persistent memory store with 20 synthetic episodes
across 5 task families (kitchen / workshop / laundry / garden / office),
then evaluates Recall@1 / Recall@3 / Recall@5 on 20 paraphrased queries.

By default the hashing embedder is used so that the benchmark runs on
CPU without any additional installs. If sentence-transformers is
available, the script will also run with the MiniLM embedder for
comparison.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from mnemo.persistent_memory import (
    HashingEmbedder,
    MemoryStore,
    build_dataset,
    default_query_set,
    run_benchmark,
)


def run_with_embedder(embedder, label: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(root=Path(tmp) / label, embedder=embedder)
        for summary in build_dataset():
            store.add(summary)
        result = run_benchmark(
            store,
            default_query_set(),
            top_k=5,
        )
        print(f"\n--- {label} ---")
        print(result.as_table())


def main() -> None:
    run_with_embedder(HashingEmbedder(dim=128), label="HashingEmbedder(dim=128)")

    # Optional semantic embedder (only if available).
    try:
        from mnemo.persistent_memory import SentenceTransformerEmbedder
        st_embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    except (ImportError, OSError) as err:
        print(
            "\n[skip] SentenceTransformerEmbedder unavailable: "
            f"{type(err).__name__}"
        )
        return

    run_with_embedder(st_embedder, label="SentenceTransformer(MiniLM)")


if __name__ == "__main__":
    main()
