"""Task definitions for MNEMO-Bench.

A ``MemoryTask`` describes a single benchmark task independent of the
underlying simulator. The harness runs a policy against the task and
collects a ``TaskResult`` object. Adapters convert tasks from existing
benchmarks (RMBench, MemoryBench, MIKASA-Robo, RoboCerebra) into this
uniform schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


class MemoryType(Enum):
    """The seven memory types scored by MNEMO-Bench."""

    SPATIAL = "spatial"
    SEQUENTIAL = "sequential"
    EPISODIC = "episodic"
    TEMPORAL = "temporal"
    VISUAL = "visual"
    CAPACITY = "capacity"
    CROSS_SESSION = "cross_session"


@dataclass
class MemoryTask:
    """A single benchmark task.

    Fields:
        task_id: unique identifier, e.g. ``rmbench/press_button``.
        source_benchmark: one of ``rmbench | memorybench | mikasa | robocerebra | mnemo_real``.
        description: natural-language task description.
        memory_types: the memory types this task exercises (a task can
                      exercise multiple).
        horizon: average episode length in frames.
        difficulty: ``easy | hard`` (matches RoboTwin 2.0 easy/hard split).
        config: opaque dict of additional simulator-specific config.
    """

    task_id: str
    source_benchmark: str
    description: str
    memory_types: List[MemoryType]
    horizon: int = 300
    difficulty: str = "easy"
    config: Dict = field(default_factory=dict)


@dataclass
class TaskResult:
    """The outcome of running a policy on a single task.

    Fields:
        task_id: must match ``MemoryTask.task_id``.
        success_rate: over all rollouts, fraction that completed the task.
        num_rollouts: how many rollouts contributed to the statistic.
        avg_episode_length: mean episode length across successful rollouts.
        memory_recall: optional auxiliary metric (fraction of cross-session
                       facts correctly retrieved).
        notes: free-form diagnostic text.
    """

    task_id: str
    success_rate: float
    num_rollouts: int
    avg_episode_length: Optional[float] = None
    memory_recall: Optional[float] = None
    notes: str = ""


# Type alias for any callable that takes a task and returns a result.
PolicyRunner = Callable[[MemoryTask], TaskResult]
