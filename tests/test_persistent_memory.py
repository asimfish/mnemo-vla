"""Tests for mnemo.persistent_memory."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from mnemo.language_memory import Memory, SubtaskStatus
from mnemo.persistent_memory import (
    EpisodeFact,
    EpisodeSummary,
    HashingEmbedder,
    MemoryStore,
    chain_summaries,
    summarize_rule_based,
)


@pytest.fixture()
def tmp_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(
        root=tmp_path / "store",
        embedder=HashingEmbedder(dim=32),
    )
    yield store
    shutil.rmtree(tmp_path / "store", ignore_errors=True)


def _sample_summary(session_id: str, task: str) -> EpisodeSummary:
    return EpisodeSummary(
        session_id=session_id,
        task=task,
        outcome="success",
        narrative=f"Completed {task}.",
        facts=[
            EpisodeFact(kind="task_outcome", content=f"{task} succeeded"),
            EpisodeFact(kind="object_location", content="Knife in drawer"),
        ],
    )


def test_add_and_retrieve(tmp_store: MemoryStore):
    tmp_store.add(_sample_summary("ep1", "Make tomato soup"))
    tmp_store.add(_sample_summary("ep2", "Clean kitchen"))
    tmp_store.add(_sample_summary("ep3", "Prepare dinner"))

    hits = tmp_store.query("Make tomato soup", top_k=2)
    assert len(hits) == 2
    # The exact ordering depends on the hashing, but the relevant episode
    # should at least appear in the top-K.
    sessions = [h.summary.session_id for h in hits]
    assert "ep1" in sessions


def test_persistence_roundtrip(tmp_path: Path):
    store1 = MemoryStore(
        root=tmp_path / "roundtrip",
        embedder=HashingEmbedder(dim=32),
    )
    store1.add(_sample_summary("ep1", "Task A"))
    store1.add(_sample_summary("ep2", "Task B"))

    # Re-open the store and check that the data survives.
    store2 = MemoryStore(
        root=tmp_path / "roundtrip",
        embedder=HashingEmbedder(dim=32),
    )
    assert len(store2) == 2
    hits = store2.query("Task A", top_k=1)
    assert hits[0].summary.session_id == "ep1"


def test_summarize_rule_based_success():
    memory = Memory(task="Cook pasta")
    memory.add_subtask("Boil water", SubtaskStatus.DONE)
    memory.add_subtask("Add pasta", SubtaskStatus.DONE)
    memory.add_subtask("Drain pasta", SubtaskStatus.DONE)

    summary = summarize_rule_based(memory)
    assert summary.outcome == "success"
    assert "Boil water" in summary.narrative


def test_summarize_rule_based_partial_failure():
    memory = Memory(task="Cook pasta")
    memory.add_subtask("Boil water", SubtaskStatus.DONE)
    memory.add_subtask("Add pasta", SubtaskStatus.FAILED)
    memory.add_failure("Water boiled over")

    summary = summarize_rule_based(memory)
    assert summary.outcome == "failure"
    fact_kinds = {f.kind for f in summary.facts}
    assert "failure_cause" in fact_kinds


def test_chain_summaries_rendering():
    block = chain_summaries(
        [_sample_summary("a", "x"), _sample_summary("b", "y")]
    )
    assert "[EPISODE] a" in block
    assert "[EPISODE] b" in block


def test_embedder_determinism():
    embedder = HashingEmbedder(dim=16)
    v1 = embedder.encode("hello world")
    v2 = embedder.encode("hello world")
    assert np.allclose(v1, v2)


def test_clear(tmp_store: MemoryStore):
    tmp_store.add(_sample_summary("ep1", "Task A"))
    assert len(tmp_store) == 1
    tmp_store.clear()
    assert len(tmp_store) == 0
    hits = tmp_store.query("anything", top_k=5)
    assert hits == []
