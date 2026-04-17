"""Adapter: inject MNEMO persistent memory into the Mem-0 Planner.

This module is a thin helper that does NOT import Mem-0 directly (to keep
this package dependency-free), but provides the glue code you would add
to Mem-0's ``MemoryMattersPlanner.prepare_qwen_input`` to incorporate
MNEMO's cross-session memory.

Reference
---------
- Mem-0 source: ``rmbench/policy/Mem-0/source/models/planning_module/memorymatters_planner.py``
- Patch target: ``prepare_qwen_input`` and ``update_image_or_video_input``

The adapter takes a MemoryStore and a task goal, retrieves the top-K
relevant past episodes, and formats them as a text block that can be
inserted before the ``<finished_subtasks>`` section of the Mem-0 prompt.
"""

from __future__ import annotations

from typing import List

from ..persistent_memory import MemoryStore, RetrievalHit


def build_cross_session_block(
    store: MemoryStore,
    task_goal: str,
    top_k: int = 3,
    min_score: float = 0.0,
) -> str:
    """Query the persistent store and render a prompt-ready text block.

    Args:
        store: the MNEMO persistent memory store.
        task_goal: current task description used as the retrieval query.
        top_k: number of past episodes to retrieve.
        min_score: drop retrieved episodes with score below this threshold.

    Returns:
        A text block ready to be inserted into the Mem-0 planner prompt.
        If nothing relevant is retrieved, returns an empty string.
    """
    hits: List[RetrievalHit] = store.query(text=task_goal, top_k=top_k)
    hits = [h for h in hits if h.score >= min_score]
    if not hits:
        return ""

    lines = ["<cross_session_memory>:"]
    for idx, hit in enumerate(hits, start=1):
        lines.append(
            f"  {idx}. [score={hit.score:.2f}] "
            f"Task '{hit.summary.task}' "
            f"(session {hit.summary.session_id}, {hit.summary.outcome}). "
            f"Summary: {hit.summary.narrative}"
        )
        for fact in hit.summary.facts:
            lines.append(f"     - ({fact.kind}) {fact.content}")
    return "\n".join(lines)


def inject_into_prompt(
    full_prompt_msg: list,
    store: MemoryStore,
    task_goal: str,
    top_k: int = 3,
) -> list:
    """Patched version of Mem-0's prepare_qwen_input that inserts MNEMO memory.

    The Mem-0 ``prepare_qwen_input`` function returns a list shaped like::

        [
          {"role": "system", "content": [...]},
          {"role": "user", "content": [<list of text/image dicts>]},
        ]

    This helper appends a text block with retrieved cross-session memory
    right before the ``<finished_subtasks>`` marker inside the user content.
    Mem-0 will then include this context when prompting Qwen3-VL.

    The original Mem-0 prompt remains untouched if the store is empty.

    Example (diff against memorymatters_planner.py::prepare_qwen_input):

        --- before returning full_prompt_msg ---
        from mnemo.integrations.mem0_planner_patch import inject_into_prompt
        full_prompt_msg = inject_into_prompt(
            full_prompt_msg,
            store=self.persistent_store,
            task_goal=self.global_task,
            top_k=3,
        )
        return full_prompt_msg
    """
    block = build_cross_session_block(store, task_goal=task_goal, top_k=top_k)
    if not block:
        return full_prompt_msg

    # Locate the user message (second entry in Mem-0's schema).
    for message in full_prompt_msg:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        # Insert just before the last text item that mentions finished_subtasks.
        insertion_idx = _find_finished_subtasks_idx(content)
        content.insert(insertion_idx, {"type": "text", "text": block + "\n"})
        break
    return full_prompt_msg


def _find_finished_subtasks_idx(content: list) -> int:
    """Return the index of the first <finished_subtasks> text entry.

    Falls back to the end of the list if no matching entry is found.
    """
    for idx, item in enumerate(content):
        if (
            isinstance(item, dict)
            and item.get("type") == "text"
            and "<finished_subtasks>" in item.get("text", "")
        ):
            return idx
    return len(content)
