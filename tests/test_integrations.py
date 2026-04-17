"""Tests for mnemo.integrations."""

from pathlib import Path

from mnemo.integrations.mem0_planner_patch import (
    build_cross_session_block,
    inject_into_prompt,
)
from mnemo.persistent_memory import (
    EpisodeFact,
    EpisodeSummary,
    HashingEmbedder,
    MemoryStore,
)


def _seed_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(
        root=tmp_path / "integration_store",
        embedder=HashingEmbedder(dim=32),
    )
    store.add(
        EpisodeSummary(
            session_id="yesterday",
            task="Organize tools",
            outcome="success",
            narrative="Put the hammer in the left drawer.",
            facts=[
                EpisodeFact(
                    kind="object_location",
                    content="Hammer is in the left drawer.",
                )
            ],
        )
    )
    return store


def test_build_cross_session_block_non_empty(tmp_path: Path):
    store = _seed_store(tmp_path)
    block = build_cross_session_block(
        store,
        task_goal="Find the hammer",
        top_k=1,
    )
    assert "cross_session_memory" in block
    assert "hammer" in block.lower() or "Hammer" in block


def test_build_cross_session_block_empty_for_empty_store(tmp_path: Path):
    store = MemoryStore(root=tmp_path / "empty_store")
    assert build_cross_session_block(store, task_goal="x") == ""


def test_inject_into_prompt_preserves_original_when_store_empty(tmp_path: Path):
    empty_store = MemoryStore(root=tmp_path / "empty_store")
    original_prompt = [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<global_task>: x\n"},
                {"type": "text", "text": "<finished_subtasks>: null.\n"},
            ],
        },
    ]
    patched = inject_into_prompt(
        [dict(m) for m in original_prompt],
        store=empty_store,
        task_goal="x",
    )
    # Original user content length is preserved.
    assert len(patched[1]["content"]) == 2


def test_inject_into_prompt_inserts_block_before_finished_subtasks(tmp_path: Path):
    store = _seed_store(tmp_path)
    prompt = [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<global_task>: Find the hammer.\n"},
                {"type": "text", "text": "<finished_subtasks>: null.\n"},
            ],
        },
    ]
    patched = inject_into_prompt(
        prompt,
        store=store,
        task_goal="Find the hammer",
        top_k=1,
    )
    user_content = patched[1]["content"]
    # Now three items: global_task, injected cross-session block, finished_subtasks.
    assert len(user_content) == 3
    injected = user_content[1]
    assert injected["type"] == "text"
    assert "cross_session_memory" in injected["text"]
    assert user_content[2]["text"].startswith("<finished_subtasks>")
