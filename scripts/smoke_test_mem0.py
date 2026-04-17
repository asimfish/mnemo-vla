"""Smoke test for the Mem-0 integration patch.

This script verifies that ``mnemo.integrations.mem0_planner_patch``
works against the actual Mem-0 planner prompt format, without
requiring the heavy Mem-0 runtime or a GPU.

Usage::

    PYTHONPATH=. python scripts/smoke_test_mem0.py

What the script does:
  1. Constructs a prompt that mirrors the format built by
     ``MemoryMattersPlanner.prepare_qwen_input`` (global task, initial
     observation placeholder, finished subtasks).
  2. Seeds a persistent memory store with three past episodes.
  3. Calls ``inject_into_prompt`` and asserts the output is
     syntactically valid and contains both the original user content
     and the injected cross-session block.
  4. Runs the injection on an empty store to confirm graceful fallback.

Exit status 0 means the patch is safe to apply to a real Mem-0
checkout.
"""

from __future__ import annotations

import sys
import tempfile
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


def _build_mem0_like_prompt(global_task: str) -> list:
    """Reconstruct the message schema used by Mem-0's Qwen planner."""
    system_prompt = (
        "You are a robotic assistant specialized in subtask planning."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"<global_task>: {global_task}\n"},
                {"type": "text", "text": "<initial_observation>: "},
                {"type": "image_url", "image_url": {"url": "fake://image.png"}},
                {"type": "text", "text": ".\n"},
                {"type": "text", "text": "<finished_subtasks>: null.\n"},
            ],
        },
    ]


def _seed_store(root: Path) -> MemoryStore:
    store = MemoryStore(root=root, embedder=HashingEmbedder(dim=64))
    store.add(
        EpisodeSummary(
            session_id="20270310",
            task="Organize kitchen utensils",
            outcome="success",
            narrative="Placed the hammer in the top-right drawer.",
            facts=[
                EpisodeFact(
                    kind="object_location",
                    content="Hammer is in the top-right drawer.",
                )
            ],
        )
    )
    store.add(
        EpisodeSummary(
            session_id="20270312",
            task="Prepare a tomato soup",
            outcome="partial",
            narrative="Heated oil and added tomatoes; ran out of salt.",
            facts=[
                EpisodeFact(
                    kind="failure_cause",
                    content="No salt left on the top shelf.",
                )
            ],
        )
    )
    store.add(
        EpisodeSummary(
            session_id="20270315",
            task="Find the hammer",
            outcome="success",
            narrative="Retrieved hammer from the top-right drawer.",
            facts=[
                EpisodeFact(
                    kind="user_preference",
                    content="User prefers the hammer returned to the top drawer.",
                )
            ],
        )
    )
    return store


def main() -> int:
    errors = []

    with tempfile.TemporaryDirectory() as tmp:
        store_root = Path(tmp) / "mem0_smoke_store"
        store = _seed_store(store_root)

        prompt = _build_mem0_like_prompt("Find the hammer.")
        block = build_cross_session_block(
            store,
            task_goal="Find the hammer.",
            top_k=3,
        )
        if "cross_session_memory" not in block:
            errors.append("cross_session_memory marker missing from block")

        patched = inject_into_prompt(
            [dict(m) for m in prompt],
            store=store,
            task_goal="Find the hammer.",
            top_k=3,
        )
        user_items = patched[1]["content"]
        if not any(
            isinstance(item, dict)
            and item.get("type") == "text"
            and "cross_session_memory" in item.get("text", "")
            for item in user_items
        ):
            errors.append("Injection did not add a cross_session_memory text item")

        # Ensure original <finished_subtasks> still present.
        if not any(
            isinstance(item, dict)
            and "<finished_subtasks>" in item.get("text", "")
            for item in user_items
        ):
            errors.append("Original <finished_subtasks> marker lost during injection")

    # Empty-store fallback.
    with tempfile.TemporaryDirectory() as tmp:
        empty_store = MemoryStore(root=Path(tmp) / "empty")
        prompt = _build_mem0_like_prompt("Cook rice.")
        patched = inject_into_prompt(prompt, store=empty_store, task_goal="Cook rice.")
        if len(patched[1]["content"]) != len(prompt[1]["content"]):
            errors.append("Empty-store injection altered prompt length")

    if errors:
        print("Smoke test FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Mem-0 integration smoke test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
