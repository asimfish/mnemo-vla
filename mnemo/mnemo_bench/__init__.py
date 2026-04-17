"""MNEMO-Bench: a unified memory benchmark harness.

Public API:
    MemoryTask, MemoryType, TaskResult, PolicyRunner
    rmbench_tasks, memorybench_tasks, mikasa_tasks, robocerebra_tasks,
    all_tasks
    BenchmarkReport, aggregate
    run, load_results, mock_runner
    plot_radar, ascii_radar, render

Design: the harness is simulator-agnostic. Supply your own
``PolicyRunner`` to bind it to a specific simulator or real robot.
"""

from .adapters import (
    all_tasks,
    memorybench_tasks,
    mikasa_tasks,
    mnemo_real_tasks,
    rmbench_tasks,
    robocerebra_tasks,
)
from .metrics import BenchmarkReport, aggregate
from .radar import ascii_radar, plot_radar, render
from .runner import load_results, mock_runner, run
from .tasks import MemoryTask, MemoryType, PolicyRunner, TaskResult

__all__ = [
    "MemoryTask",
    "MemoryType",
    "TaskResult",
    "PolicyRunner",
    "BenchmarkReport",
    "aggregate",
    "rmbench_tasks",
    "memorybench_tasks",
    "mikasa_tasks",
    "robocerebra_tasks",
    "mnemo_real_tasks",
    "all_tasks",
    "run",
    "load_results",
    "mock_runner",
    "plot_radar",
    "ascii_radar",
    "render",
]
