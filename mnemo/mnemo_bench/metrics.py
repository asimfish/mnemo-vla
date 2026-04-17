"""Metric aggregation for MNEMO-Bench.

Given a list of ``TaskResult`` objects and the corresponding
``MemoryTask`` metadata, we compute per-memory-type success rates and
the overall memory score (geometric mean across types).

Design note: we average success rates over the tasks that exercise
each memory type. A task tagged with multiple memory types contributes
to each of them. Tasks with zero rollouts are ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .tasks import MemoryTask, MemoryType, TaskResult


@dataclass
class BenchmarkReport:
    """Aggregated results of a MNEMO-Bench run."""

    per_task: Dict[str, TaskResult]
    per_memory_type: Dict[MemoryType, float]
    per_benchmark: Dict[str, float]
    overall_score: float
    radar_scores: List[float] = field(default_factory=list)

    def summary_table(self) -> str:
        """Render a plain-text summary."""
        lines = [
            f"Total tasks         : {len(self.per_task)}",
            f"Overall memory score: {self.overall_score:.2%}",
            "",
            "Per memory type:",
        ]
        for mt, score in self.per_memory_type.items():
            lines.append(f"  {mt.value:<15s} {score:.2%}")
        lines.append("")
        lines.append("Per source benchmark:")
        for bench, score in self.per_benchmark.items():
            lines.append(f"  {bench:<15s} {score:.2%}")
        return "\n".join(lines)


def aggregate(
    tasks: Iterable[MemoryTask],
    results: Iterable[TaskResult],
) -> BenchmarkReport:
    """Compute a BenchmarkReport from per-task tasks and results."""
    task_by_id = {t.task_id: t for t in tasks}
    result_by_id = {r.task_id: r for r in results}

    missing = set(task_by_id) - set(result_by_id)
    if missing:
        raise ValueError(
            f"Missing results for task ids: {sorted(missing)[:5]}..."
        )

    # ----- per memory type
    per_memory_type_scores: Dict[MemoryType, List[float]] = {
        mt: [] for mt in MemoryType
    }
    for task_id, task in task_by_id.items():
        result = result_by_id[task_id]
        if result.num_rollouts <= 0:
            continue
        for mt in task.memory_types:
            per_memory_type_scores[mt].append(result.success_rate)

    per_memory_type: Dict[MemoryType, float] = {
        mt: (sum(scores) / len(scores)) if scores else 0.0
        for mt, scores in per_memory_type_scores.items()
    }

    # ----- per source benchmark
    per_benchmark_scores: Dict[str, List[float]] = {}
    for task_id, task in task_by_id.items():
        result = result_by_id[task_id]
        if result.num_rollouts <= 0:
            continue
        per_benchmark_scores.setdefault(task.source_benchmark, []).append(
            result.success_rate
        )
    per_benchmark: Dict[str, float] = {
        bench: (sum(scores) / len(scores)) if scores else 0.0
        for bench, scores in per_benchmark_scores.items()
    }

    # ----- overall (geometric mean across memory types with at least one task)
    active_scores = [
        score for score in per_memory_type.values() if score > 0.0
    ]
    overall = _geometric_mean(active_scores) if active_scores else 0.0

    radar_scores = [per_memory_type[mt] for mt in MemoryType]

    return BenchmarkReport(
        per_task=dict(result_by_id),
        per_memory_type=per_memory_type,
        per_benchmark=per_benchmark,
        overall_score=overall,
        radar_scores=radar_scores,
    )


def _geometric_mean(values: List[float]) -> float:
    """Geometric mean of a list of non-negative values."""
    if not values:
        return 0.0
    logs = [math.log(max(v, 1e-9)) for v in values]
    return math.exp(sum(logs) / len(logs))
