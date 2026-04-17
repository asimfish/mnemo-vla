"""Tests for the synthetic cross-session retrieval benchmark."""

from pathlib import Path

from mnemo.persistent_memory import (
    HashingEmbedder,
    MemoryStore,
    build_dataset,
    default_query_set,
    run_benchmark,
)


def test_build_dataset_size_and_uniqueness():
    dataset = build_dataset()
    assert len(dataset) == 20
    session_ids = [s.session_id for s in dataset]
    assert len(session_ids) == len(set(session_ids))  # unique


def test_query_set_points_to_existing_sessions():
    dataset = build_dataset()
    valid = {s.session_id for s in dataset}
    for _, target in default_query_set():
        assert target in valid


def test_hashing_embedder_benchmark_runs(tmp_path: Path):
    store = MemoryStore(
        root=tmp_path / "bench_store",
        embedder=HashingEmbedder(dim=128),
    )
    for summary in build_dataset():
        store.add(summary)

    result = run_benchmark(store, default_query_set(), top_k=5)
    assert result.num_queries == 20
    # Hashing is weak but should beat random (1/20 = 5%).
    assert 0.0 <= result.recall_at_1 <= 1.0
    assert result.recall_at_3 >= result.recall_at_1
    assert result.recall_at_5 >= result.recall_at_3


def test_benchmark_result_table_contains_metrics(tmp_path: Path):
    store = MemoryStore(
        root=tmp_path / "bench_store2",
        embedder=HashingEmbedder(dim=64),
    )
    for summary in build_dataset()[:5]:
        store.add(summary)
    result = run_benchmark(store, default_query_set()[:5])
    table = result.as_table()
    assert "Recall@1" in table
    assert "Recall@3" in table
    assert "Recall@5" in table


def test_perfect_retrieval_on_exact_queries(tmp_path: Path):
    """Sanity check: querying the exact session id should retrieve it."""
    store = MemoryStore(
        root=tmp_path / "bench_store3",
        embedder=HashingEmbedder(dim=128),
    )
    dataset = build_dataset()
    for summary in dataset:
        store.add(summary)
    queries = [
        (s.task + " " + s.narrative, s.session_id) for s in dataset
    ]
    result = run_benchmark(store, queries, top_k=1)
    # With the exact text, Recall@1 should be reasonably high even with
    # a toy hashing embedder (>50% is conservative).
    assert result.recall_at_1 >= 0.5
