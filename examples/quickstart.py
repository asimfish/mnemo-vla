"""Minimal end-to-end demo of MNEMO-VLA primitives.

Run with::

    python examples/quickstart.py

It shows:
  1. Building a semi-structured Memory for a running episode.
  2. Summarizing the episode at its end.
  3. Persisting the summary into a FAISS-style store.
  4. Retrieving relevant memories at the start of the next episode.
  5. Formatting a cross-session block that can be injected into a
     Mem-0 planner prompt.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from mnemo.language_memory import Memory, SubtaskStatus
from mnemo.persistent_memory import (
    HashingEmbedder,
    MemoryStore,
    summarize_rule_based,
)
from mnemo.integrations.mem0_planner_patch import build_cross_session_block


def run_episode_one() -> Memory:
    memory = Memory(task="Organize kitchen tools.")
    memory.add_subtask("Open drawer", SubtaskStatus.DONE)
    memory.add_subtask("Place hammer in left slot", SubtaskStatus.DONE)
    memory.add_subtask("Close drawer", SubtaskStatus.DONE)
    return memory


def run_episode_two(store: MemoryStore) -> None:
    task_goal = "Find the hammer."

    cross_session_block = build_cross_session_block(
        store,
        task_goal=task_goal,
        top_k=2,
    )

    print("\n--- Cross-session memory injected into planner prompt ---")
    print(cross_session_block or "(none)")

    planner_prompt = [
        "You are a robotic assistant specialized in subtask planning.",
        "<global_task>: " + task_goal,
        cross_session_block,
        "<finished_subtasks>: null.",
    ]
    planner_prompt = [part for part in planner_prompt if part]
    print("\n--- Full planner prompt ---")
    print("\n".join(planner_prompt))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store_root = Path(tmp) / "persistent_memory"
        store = MemoryStore(
            root=store_root,
            embedder=HashingEmbedder(dim=32),
        )

        # --- Episode 1 ---
        memory = run_episode_one()
        summary = summarize_rule_based(memory)
        store.add(summary)

        print("--- Episode 1 stored ---")
        print(summary.to_text())

        # --- Episode 2: new session, uses cross-session memory ---
        run_episode_two(store)


if __name__ == "__main__":
    main()
