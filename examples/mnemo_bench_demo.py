"""MNEMO-Bench end-to-end demo with a mock policy.

Usage::

    PYTHONPATH=. python examples/mnemo_bench_demo.py

The demo runs the full 32-task MNEMO-Bench with a mock policy that
returns canned success rates stratified by memory type. It prints:

  * A per-task progress table (via the runner).
  * Aggregated memory-type and benchmark-level scores.
  * An ASCII radar chart.
  * Optionally a PNG radar (if matplotlib is installed).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from mnemo.mnemo_bench import (
    MemoryType,
    all_tasks,
    ascii_radar,
    mock_runner,
    plot_radar,
    run,
)


def _memory_type_bias() -> dict:
    """Simulate a policy that is strong in some types and weak in others.

    The particular bias is contrived for demonstration only. Returns a
    mapping from task_id to success rate.
    """
    strong_types = {MemoryType.SPATIAL, MemoryType.VISUAL}
    medium_types = {MemoryType.SEQUENTIAL, MemoryType.EPISODIC}
    weak_types = {
        MemoryType.TEMPORAL,
        MemoryType.CAPACITY,
        MemoryType.CROSS_SESSION,
    }

    success = {}
    for task in all_tasks():
        if any(mt in strong_types for mt in task.memory_types):
            success[task.task_id] = 0.80
        elif any(mt in medium_types for mt in task.memory_types):
            success[task.task_id] = 0.55
        elif any(mt in weak_types for mt in task.memory_types):
            success[task.task_id] = 0.25
        else:
            success[task.task_id] = 0.40
    return success


def main() -> None:
    runner = mock_runner(
        success_map=_memory_type_bias(),
        num_rollouts=30,
    )

    with tempfile.TemporaryDirectory() as tmp:
        results_path = Path(tmp) / "results.jsonl"
        report = run(all_tasks(), runner, save_to=results_path)

    print(report.summary_table())
    print("\n" + ascii_radar(report))

    # Try to render the PNG version.
    png_path = Path("mnemo_bench_radar_demo.png")
    try:
        plot_radar(report, png_path, title="MNEMO-Bench (mock policy)")
        print(f"\nRadar PNG saved to {png_path.resolve()}")
    except ImportError:
        print("\n[skip] matplotlib not installed; PNG radar not rendered.")


if __name__ == "__main__":
    main()
