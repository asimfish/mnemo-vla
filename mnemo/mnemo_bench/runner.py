"""The MNEMO-Bench runner: iterate tasks, collect results, aggregate.

The runner does not know how to simulate robots. The caller supplies a
``PolicyRunner`` that takes a ``MemoryTask`` and returns a
``TaskResult``. The runner loops over the provided tasks, calls the
runner, and aggregates results via ``metrics.aggregate``.

This means MNEMO-Bench can be driven by:
  * A full simulator wrapper (e.g., a RoboTwin 2.0 evaluator).
  * A unit-test mock that returns fixed success rates.
  * A precomputed JSON of results (see ``load_results``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from .metrics import BenchmarkReport, aggregate
from .tasks import MemoryTask, PolicyRunner, TaskResult


def run(
    tasks: Iterable[MemoryTask],
    runner: PolicyRunner,
    save_to: Optional[str | Path] = None,
) -> BenchmarkReport:
    """Run a policy across all tasks and return an aggregated report.

    Args:
        tasks: iterable of tasks (typically from ``adapters.all_tasks()``).
        runner: callable that produces a TaskResult for a given task.
        save_to: optional path; if provided, per-task results are saved
                 to a JSONL file for later inspection.
    """
    tasks_list: List[MemoryTask] = list(tasks)
    results: List[TaskResult] = []
    save_path: Optional[Path] = Path(save_to) if save_to else None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text("")  # truncate

    for task in tasks_list:
        result = runner(task)
        results.append(result)
        if save_path is not None:
            with save_path.open("a", encoding="utf-8") as fp:
                fp.write(
                    json.dumps(
                        {
                            "task_id": result.task_id,
                            "success_rate": result.success_rate,
                            "num_rollouts": result.num_rollouts,
                            "avg_episode_length": result.avg_episode_length,
                            "memory_recall": result.memory_recall,
                            "notes": result.notes,
                        }
                    )
                    + "\n"
                )

    return aggregate(tasks_list, results)


def load_results(path: str | Path) -> List[TaskResult]:
    """Load per-task results from a JSONL file produced by ``run``."""
    results: List[TaskResult] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            results.append(
                TaskResult(
                    task_id=data["task_id"],
                    success_rate=data["success_rate"],
                    num_rollouts=data["num_rollouts"],
                    avg_episode_length=data.get("avg_episode_length"),
                    memory_recall=data.get("memory_recall"),
                    notes=data.get("notes", ""),
                )
            )
    return results


def mock_runner(
    success_map: Optional[dict] = None,
    default_success: float = 0.2,
    num_rollouts: int = 30,
) -> PolicyRunner:
    """Return a fake runner that returns canned success rates.

    Useful for unit tests and for sanity-checking the aggregation logic.
    """

    success_map = dict(success_map or {})

    def _runner(task: MemoryTask) -> TaskResult:
        rate = success_map.get(task.task_id, default_success)
        return TaskResult(
            task_id=task.task_id,
            success_rate=rate,
            num_rollouts=num_rollouts,
            avg_episode_length=float(task.horizon),
            notes="mock_runner",
        )

    return _runner
