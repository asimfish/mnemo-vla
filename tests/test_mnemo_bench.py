"""Tests for the mnemo.mnemo_bench harness."""

from pathlib import Path

from mnemo.mnemo_bench import (
    BenchmarkReport,
    MemoryType,
    all_tasks,
    ascii_radar,
    memorybench_tasks,
    mikasa_tasks,
    mnemo_real_tasks,
    mock_runner,
    rmbench_tasks,
    robocerebra_tasks,
    run,
    load_results,
)


def test_task_counts():
    assert len(rmbench_tasks()) == 9
    assert len(memorybench_tasks()) == 3
    assert len(mikasa_tasks()) == 10
    assert len(robocerebra_tasks()) == 10
    assert len(mnemo_real_tasks()) == 5
    assert len(all_tasks()) == 37


def test_task_ids_unique():
    ids = [t.task_id for t in all_tasks()]
    assert len(ids) == len(set(ids))


def test_memory_type_coverage():
    tasks = all_tasks()
    covered = {mt for task in tasks for mt in task.memory_types}
    # The v1.0 selection must cover all 7 memory types.
    assert covered == set(MemoryType)


def test_run_with_mock_runner():
    tasks = all_tasks()
    runner = mock_runner(default_success=0.3)
    report = run(tasks, runner)
    assert isinstance(report, BenchmarkReport)
    assert len(report.per_task) == 37
    assert 0.0 <= report.overall_score <= 1.0


def test_aggregate_matches_canned_success_map():
    # Build a deterministic map where only cross-session tasks fail.
    runner = mock_runner(
        success_map={"rmbench/press_button": 0.0},
        default_success=1.0,
    )
    report = run(all_tasks(), runner)
    # With every other task at 100%, overall must be close to 1.0 but
    # strictly less than 1.0 because press_button drags capacity / episodic down.
    assert 0.6 < report.overall_score < 1.0


def test_save_and_load_results(tmp_path: Path):
    tasks = mikasa_tasks()
    runner = mock_runner(default_success=0.5)
    save_to = tmp_path / "mnemo_bench_results.jsonl"
    run(tasks, runner, save_to=save_to)

    loaded = load_results(save_to)
    assert len(loaded) == len(tasks)
    assert {r.task_id for r in loaded} == {t.task_id for t in tasks}


def test_ascii_radar_contains_all_memory_types():
    runner = mock_runner(default_success=0.4)
    report = run(all_tasks(), runner)
    chart = ascii_radar(report)
    for mt in MemoryType:
        assert mt.value in chart
