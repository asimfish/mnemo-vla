"""Semi-structured language memory schema for MNEMO-VLA.

This module defines the data classes and serialization format for MNEMO's
language-based memory system. The design is inspired by MEM (Torne et al., 2026)
but extends it with a semi-structured format for easier parsing and
interpretability.

A Memory object captures:
  - TASK: the overall goal
  - SESSION: temporal metadata
  - SUBTASKS: ordered list with status (done / current / pending)
  - CROSS_SESSION: facts retrieved from persistent memory
  - FAILURES: error events and interventions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class SubtaskStatus(Enum):
    DONE = "done"
    CURRENT = "current"
    PENDING = "pending"
    FAILED = "failed"


@dataclass
class Subtask:
    """A single subtask in the execution plan."""

    description: str
    status: SubtaskStatus
    timestamp: Optional[datetime] = None
    details: Optional[str] = None


@dataclass
class CrossSessionFact:
    """A fact retrieved from persistent memory (cross-session)."""

    content: str
    source_session: str  # session id or timestamp
    confidence: float = 1.0


@dataclass
class FailureEvent:
    """An error or intervention event."""

    description: str
    timestamp: Optional[datetime] = None
    intervention: Optional[str] = None  # human correction or automatic recovery


@dataclass
class Memory:
    """Top-level semi-structured memory object for a single task/episode.

    This is the main object that flows through the planner and executor.
    It is serializable to/from text for LLM consumption, and to/from JSON
    for storage in persistent memory.

    Example:
        mem = Memory(task="Clean up the kitchen.")
        mem.add_subtask("Wash tomatoes", SubtaskStatus.DONE)
        mem.add_subtask("Cut tomatoes", SubtaskStatus.CURRENT)
        print(mem.to_text())
    """

    task: str
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%dT%H%M%S"))
    subtasks: List[Subtask] = field(default_factory=list)
    cross_session: List[CrossSessionFact] = field(default_factory=list)
    failures: List[FailureEvent] = field(default_factory=list)

    def add_subtask(
        self,
        description: str,
        status: SubtaskStatus = SubtaskStatus.PENDING,
        details: Optional[str] = None,
    ) -> Subtask:
        """Append a subtask and return it."""
        subtask = Subtask(
            description=description,
            status=status,
            timestamp=datetime.now(),
            details=details,
        )
        self.subtasks.append(subtask)
        return subtask

    def mark_current_done(self) -> None:
        """Mark the currently-in-progress subtask as done."""
        for subtask in self.subtasks:
            if subtask.status == SubtaskStatus.CURRENT:
                subtask.status = SubtaskStatus.DONE

    def start_next_pending(self) -> Optional[Subtask]:
        """Promote the first pending subtask to current."""
        for subtask in self.subtasks:
            if subtask.status == SubtaskStatus.PENDING:
                subtask.status = SubtaskStatus.CURRENT
                return subtask
        return None

    def add_cross_session(
        self,
        content: str,
        source_session: str,
        confidence: float = 1.0,
    ) -> None:
        """Append a cross-session fact from persistent memory."""
        self.cross_session.append(
            CrossSessionFact(
                content=content,
                source_session=source_session,
                confidence=confidence,
            )
        )

    def add_failure(
        self,
        description: str,
        intervention: Optional[str] = None,
    ) -> None:
        """Record a failure event."""
        self.failures.append(
            FailureEvent(
                description=description,
                timestamp=datetime.now(),
                intervention=intervention,
            )
        )

    def to_text(self) -> str:
        """Render memory as human-readable text (what the LLM sees).

        The format is stable and parser-friendly: every section starts
        with a [SECTION] tag on its own line.
        """
        lines: List[str] = []
        lines.append(f"[TASK] {self.task}")
        lines.append(f"[SESSION] {self.session_id}")

        if self.subtasks:
            lines.append("[SUBTASKS]")
            for idx, subtask in enumerate(self.subtasks, start=1):
                mark = {
                    SubtaskStatus.DONE: "[X]",
                    SubtaskStatus.CURRENT: "[>]",
                    SubtaskStatus.PENDING: "[ ]",
                    SubtaskStatus.FAILED: "[!]",
                }[subtask.status]
                line = f"  {idx}. {mark} {subtask.description}"
                if subtask.details:
                    line += f" ({subtask.details})"
                lines.append(line)

        if self.cross_session:
            lines.append("[CROSS-SESSION]")
            for fact in self.cross_session:
                lines.append(
                    f"  - {fact.content} "
                    f"(source: {fact.source_session}, conf={fact.confidence:.2f})"
                )

        if self.failures:
            lines.append("[FAILURES]")
            for failure in self.failures:
                line = f"  - {failure.description}"
                if failure.intervention:
                    line += f" -> {failure.intervention}"
                lines.append(line)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "task": self.task,
            "session_id": self.session_id,
            "subtasks": [
                {
                    "description": s.description,
                    "status": s.status.value,
                    "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                    "details": s.details,
                }
                for s in self.subtasks
            ],
            "cross_session": [
                {
                    "content": f.content,
                    "source_session": f.source_session,
                    "confidence": f.confidence,
                }
                for f in self.cross_session
            ],
            "failures": [
                {
                    "description": e.description,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "intervention": e.intervention,
                }
                for e in self.failures
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Deserialize from a dict produced by to_dict."""
        memory = cls(task=data["task"], session_id=data.get("session_id", ""))
        for s in data.get("subtasks", []):
            ts = s.get("timestamp")
            memory.subtasks.append(
                Subtask(
                    description=s["description"],
                    status=SubtaskStatus(s["status"]),
                    timestamp=datetime.fromisoformat(ts) if ts else None,
                    details=s.get("details"),
                )
            )
        for f in data.get("cross_session", []):
            memory.cross_session.append(
                CrossSessionFact(
                    content=f["content"],
                    source_session=f["source_session"],
                    confidence=f.get("confidence", 1.0),
                )
            )
        for e in data.get("failures", []):
            ts = e.get("timestamp")
            memory.failures.append(
                FailureEvent(
                    description=e["description"],
                    timestamp=datetime.fromisoformat(ts) if ts else None,
                    intervention=e.get("intervention"),
                )
            )
        return memory
