"""Text parser for the semi-structured language memory format.

This module mirrors schema.to_text() so that LLM-generated memory updates
can be parsed back into Memory objects. Useful for verifying consistency
between planner output and the stored memory state.
"""

from __future__ import annotations

import re
from typing import List

from .schema import (
    CrossSessionFact,
    FailureEvent,
    Memory,
    Subtask,
    SubtaskStatus,
)


_STATUS_MAP = {
    "[X]": SubtaskStatus.DONE,
    "[>]": SubtaskStatus.CURRENT,
    "[ ]": SubtaskStatus.PENDING,
    "[!]": SubtaskStatus.FAILED,
}


def parse_memory_text(text: str) -> Memory:
    """Parse a memory text block (produced by Memory.to_text) back to an object.

    The parser is forgiving: missing sections are treated as empty, and
    malformed lines are skipped with no error.

    Args:
        text: the raw text block.

    Returns:
        A Memory object. If [TASK] is missing, task is set to an empty string.
    """
    task = ""
    session_id = ""
    subtasks: List[Subtask] = []
    cross_session: List[CrossSessionFact] = []
    failures: List[FailureEvent] = []
    section = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("[TASK]"):
            task = stripped[len("[TASK]"):].strip()
            section = None
            continue
        if stripped.startswith("[SESSION]"):
            session_id = stripped[len("[SESSION]"):].strip()
            section = None
            continue
        if stripped == "[SUBTASKS]":
            section = "subtasks"
            continue
        if stripped == "[CROSS-SESSION]":
            section = "cross_session"
            continue
        if stripped == "[FAILURES]":
            section = "failures"
            continue

        if section == "subtasks":
            subtask = _parse_subtask_line(stripped)
            if subtask is not None:
                subtasks.append(subtask)
        elif section == "cross_session":
            fact = _parse_cross_session_line(stripped)
            if fact is not None:
                cross_session.append(fact)
        elif section == "failures":
            failure = _parse_failure_line(stripped)
            if failure is not None:
                failures.append(failure)

    memory = Memory(task=task, session_id=session_id)
    memory.subtasks = subtasks
    memory.cross_session = cross_session
    memory.failures = failures
    return memory


def _parse_subtask_line(line: str) -> "Subtask | None":
    # Match patterns like: "1. [X] description (details)"
    match = re.match(
        r"^\d+\.\s+(\[.\])\s+(.+?)(?:\s+\((.+)\))?$",
        line,
    )
    if match is None:
        return None
    status_token, description, details = match.groups()
    status = _STATUS_MAP.get(status_token)
    if status is None:
        return None
    return Subtask(
        description=description,
        status=status,
        details=details,
    )


def _parse_cross_session_line(line: str) -> "CrossSessionFact | None":
    # Match patterns like: "- content (source: X, conf=0.95)"
    match = re.match(
        r"^-\s+(.+?)\s+\(source:\s*(.+?),\s*conf=([\d.]+)\)$",
        line,
    )
    if match is None:
        # Fallback: no metadata
        if line.startswith("-"):
            return CrossSessionFact(
                content=line[1:].strip(),
                source_session="unknown",
            )
        return None
    content, source, conf_str = match.groups()
    return CrossSessionFact(
        content=content,
        source_session=source,
        confidence=float(conf_str),
    )


def _parse_failure_line(line: str) -> "FailureEvent | None":
    # Match patterns like: "- description -> intervention"
    if not line.startswith("-"):
        return None
    body = line[1:].strip()
    if " -> " in body:
        description, intervention = body.split(" -> ", maxsplit=1)
        return FailureEvent(
            description=description.strip(),
            intervention=intervention.strip(),
        )
    return FailureEvent(description=body)
