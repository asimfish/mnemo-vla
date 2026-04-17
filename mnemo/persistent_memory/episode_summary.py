"""Episode summary generation and structured facts extraction.

An EpisodeSummary captures what happened in a single episode and the key
facts that should persist across sessions (e.g., "the knife is in the
top-right drawer"). These summaries are the unit that goes into the
persistent memory store.

The summary is produced at episode end by:
  1. Taking the final Memory object from the planner.
  2. Extracting object-location / task-outcome / failure-cause facts.
  3. (Optional) calling an LLM to write a concise natural-language summary.

For the skeleton implementation here, we provide a rule-based extractor.
LLM integration is a planned extension (see integrations/).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional

from ..language_memory import Memory


@dataclass
class EpisodeFact:
    """A single structured fact extracted from an episode."""

    kind: str  # one of: "object_location", "task_outcome", "failure_cause", "user_preference"
    content: str  # free-text description of the fact
    entities: List[str] = field(default_factory=list)  # named entities involved


@dataclass
class EpisodeSummary:
    """Summary of a completed episode, ready for persistent storage.

    The summary contains:
      - session_id: unique id of the episode.
      - task: the task goal string.
      - outcome: "success" | "failure" | "partial".
      - narrative: natural-language summary (may be LLM-generated).
      - facts: structured list of extracted facts.
      - timestamp: when the episode ended.
    """

    session_id: str
    task: str
    outcome: str  # "success" | "failure" | "partial"
    narrative: str
    facts: List[EpisodeFact] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_text(self) -> str:
        """Render as a text block for embedding / retrieval."""
        lines = [
            f"[EPISODE] {self.session_id}",
            f"[TASK] {self.task}",
            f"[OUTCOME] {self.outcome}",
            f"[NARRATIVE] {self.narrative}",
        ]
        if self.facts:
            lines.append("[FACTS]")
            for fact in self.facts:
                lines.append(f"  - ({fact.kind}) {fact.content}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "task": self.task,
            "outcome": self.outcome,
            "narrative": self.narrative,
            "facts": [
                {
                    "kind": f.kind,
                    "content": f.content,
                    "entities": f.entities,
                }
                for f in self.facts
            ],
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeSummary":
        ts = data.get("timestamp")
        return cls(
            session_id=data["session_id"],
            task=data["task"],
            outcome=data["outcome"],
            narrative=data["narrative"],
            facts=[
                EpisodeFact(
                    kind=f["kind"],
                    content=f["content"],
                    entities=f.get("entities", []),
                )
                for f in data.get("facts", [])
            ],
            timestamp=datetime.fromisoformat(ts) if ts else datetime.now(),
        )


def summarize_rule_based(memory: Memory, outcome: Optional[str] = None) -> EpisodeSummary:
    """Create an EpisodeSummary from a Memory using simple rules.

    This is the skeleton implementation. For production, replace with an
    LLM-based summarizer that produces better narratives and extracts
    richer facts.

    Rules:
      - outcome = "success" if all subtasks are DONE, else "partial"
      - narrative = concatenation of done subtasks
      - facts: extract "task_outcome" and one "failure_cause" per failure
    """
    from ..language_memory import SubtaskStatus

    done = [s for s in memory.subtasks if s.status == SubtaskStatus.DONE]
    failed = [s for s in memory.subtasks if s.status == SubtaskStatus.FAILED]

    if outcome is None:
        if failed:
            outcome = "failure"
        elif len(done) == len(memory.subtasks) and memory.subtasks:
            outcome = "success"
        else:
            outcome = "partial"

    narrative_parts = [s.description for s in done]
    narrative = "; ".join(narrative_parts) if narrative_parts else "(no subtasks completed)"

    facts: List[EpisodeFact] = []
    facts.append(
        EpisodeFact(
            kind="task_outcome",
            content=f"Task '{memory.task}' ended with outcome '{outcome}'.",
        )
    )
    for failure in memory.failures:
        facts.append(
            EpisodeFact(
                kind="failure_cause",
                content=failure.description,
            )
        )

    return EpisodeSummary(
        session_id=memory.session_id,
        task=memory.task,
        outcome=outcome,
        narrative=narrative,
        facts=facts,
    )


def chain_summaries(summaries: Iterable[EpisodeSummary]) -> str:
    """Join several episode summaries into a readable block for the planner.

    Useful for presenting retrieved persistent memories to the LLM.
    """
    return "\n\n".join(s.to_text() for s in summaries)
