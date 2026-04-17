"""Adapters for each source benchmark that populate the MNEMO-Bench task set.

These adapters do not run the simulator themselves; they only translate
the task metadata from each benchmark into our uniform ``MemoryTask``
schema. Actual rollout execution is the responsibility of the
``PolicyRunner`` supplied by the user (see ``runner.py``).

The task lists below are the **MNEMO-Bench v1.0 selection**. Exactly
32 tasks span the four source benchmarks, chosen for coverage across
the seven memory types defined in ``tasks.py``. The list is checked
into source control so that the benchmark is fully reproducible.
"""

from __future__ import annotations

from typing import List

from .tasks import MemoryTask, MemoryType


# ----- RMBench (9 tasks, keep all) -----------------------------------------

_RMBENCH = [
    MemoryTask(
        task_id="rmbench/observe_and_pick_up",
        source_benchmark="rmbench",
        description=(
            "Observe a reference object on a shelf, then pick up the "
            "matching object from the table after the reference is hidden."
        ),
        memory_types=[MemoryType.VISUAL, MemoryType.EPISODIC],
        horizon=400,
    ),
    MemoryTask(
        task_id="rmbench/rearrange_blocks",
        source_benchmark="rmbench",
        description="Move a block onto an empty pad, press the button, and move another block to the middle.",
        memory_types=[MemoryType.SEQUENTIAL, MemoryType.SPATIAL],
        horizon=500,
    ),
    MemoryTask(
        task_id="rmbench/put_back_block",
        source_benchmark="rmbench",
        description="Move a block to the center, press the button, and return it to its original pad.",
        memory_types=[MemoryType.SPATIAL, MemoryType.EPISODIC],
    ),
    MemoryTask(
        task_id="rmbench/swap_blocks",
        source_benchmark="rmbench",
        description="Swap the positions of two blocks using an empty pad, then press the button.",
        memory_types=[MemoryType.SEQUENTIAL, MemoryType.SPATIAL],
    ),
    MemoryTask(
        task_id="rmbench/swap_T",
        source_benchmark="rmbench",
        description="Swap two T-shaped blocks' positions and orientations.",
        memory_types=[MemoryType.SPATIAL],
    ),
    MemoryTask(
        task_id="rmbench/battery_try",
        source_benchmark="rmbench",
        description=(
            "Repeatedly attempt different insertion orders for two "
            "batteries until the holder accepts them."
        ),
        memory_types=[MemoryType.EPISODIC, MemoryType.SEQUENTIAL],
    ),
    MemoryTask(
        task_id="rmbench/blocks_ranking_try",
        source_benchmark="rmbench",
        description="Repeatedly rearrange and press until the correct ordering is found.",
        memory_types=[MemoryType.EPISODIC, MemoryType.CAPACITY],
    ),
    MemoryTask(
        task_id="rmbench/cover_blocks",
        source_benchmark="rmbench",
        description=(
            "Cover blocks from left to right, then uncover them in "
            "red-green-blue order."
        ),
        memory_types=[MemoryType.SEQUENTIAL, MemoryType.SPATIAL],
    ),
    MemoryTask(
        task_id="rmbench/press_button",
        source_benchmark="rmbench",
        description=(
            "Press the left button N times and the middle button M times, "
            "then press confirm. N and M come from two number tiles."
        ),
        memory_types=[MemoryType.CAPACITY, MemoryType.EPISODIC],
    ),
]


# ----- MemoryBench (3 tasks, port to RoboTwin 2.0) -------------------------

_MEMORYBENCH = [
    MemoryTask(
        task_id="memorybench/reopen_drawer",
        source_benchmark="memorybench",
        description=(
            "Close an initially-open drawer, press a button, and reopen "
            "the same drawer after all drawers become visually identical."
        ),
        memory_types=[MemoryType.SPATIAL, MemoryType.EPISODIC],
        horizon=300,
    ),
    MemoryTask(
        task_id="memorybench/put_block_back",
        source_benchmark="memorybench",
        description="Move a block to the center pad, press button, return to original pad.",
        memory_types=[MemoryType.SPATIAL],
        horizon=300,
    ),
    MemoryTask(
        task_id="memorybench/rearrange_block",
        source_benchmark="memorybench",
        description="Backward-reasoning block rearrangement after a button press.",
        memory_types=[MemoryType.SEQUENTIAL, MemoryType.SPATIAL],
        horizon=300,
    ),
]


# ----- MIKASA-Robo subset (10 tasks covering the 12 memory categories) -----

_MIKASA = [
    MemoryTask(
        task_id="mikasa/shell_game_pick",
        source_benchmark="mikasa",
        description="Memorize which cup hides the ball, then pick that cup.",
        memory_types=[MemoryType.VISUAL, MemoryType.SPATIAL],
        horizon=90,
    ),
    MemoryTask(
        task_id="mikasa/intercept_medium",
        source_benchmark="mikasa",
        description="Estimate the rolling ball's velocity and intercept it.",
        memory_types=[MemoryType.TEMPORAL, MemoryType.SPATIAL],
        horizon=90,
    ),
    MemoryTask(
        task_id="mikasa/rotate_strict_pos",
        source_benchmark="mikasa",
        description="Rotate a peg by a specified angle without shifting the center.",
        memory_types=[MemoryType.SPATIAL],
        horizon=90,
    ),
    MemoryTask(
        task_id="mikasa/take_it_back",
        source_benchmark="mikasa",
        description="Move a cube to a target region, then return it to its initial position.",
        memory_types=[MemoryType.SPATIAL, MemoryType.EPISODIC],
        horizon=180,
    ),
    MemoryTask(
        task_id="mikasa/remember_color_5",
        source_benchmark="mikasa",
        description="Memorize a color among 5 and select it later.",
        memory_types=[MemoryType.VISUAL],
        horizon=60,
    ),
    MemoryTask(
        task_id="mikasa/remember_shape_5",
        source_benchmark="mikasa",
        description="Memorize a shape among 5 and select it later.",
        memory_types=[MemoryType.VISUAL],
        horizon=60,
    ),
    MemoryTask(
        task_id="mikasa/bunch_of_colors_5",
        source_benchmark="mikasa",
        description="Remember a set of 5 colored cubes shown at once, touch them in any order.",
        memory_types=[MemoryType.CAPACITY, MemoryType.VISUAL],
        horizon=120,
    ),
    MemoryTask(
        task_id="mikasa/seq_of_colors_5",
        source_benchmark="mikasa",
        description="Remember 5 colors shown sequentially; select them in any order.",
        memory_types=[MemoryType.CAPACITY, MemoryType.SEQUENTIAL],
        horizon=120,
    ),
    MemoryTask(
        task_id="mikasa/chain_of_colors_5",
        source_benchmark="mikasa",
        description="Remember 5 colors shown sequentially; reproduce the same order.",
        memory_types=[MemoryType.SEQUENTIAL, MemoryType.CAPACITY],
        horizon=120,
    ),
    MemoryTask(
        task_id="mikasa/intercept_grab_fast",
        source_benchmark="mikasa",
        description="Estimate ball velocity and catch it with the gripper.",
        memory_types=[MemoryType.TEMPORAL, MemoryType.SPATIAL],
        horizon=90,
    ),
]


# ----- RoboCerebra subset (10 tasks for long-horizon evaluation) -----------

_ROBOCEREBRA = [
    MemoryTask(
        task_id="robocerebra/long_kitchen_cleanup",
        source_benchmark="robocerebra",
        description=(
            "Clean up a kitchen by returning groceries to their storage "
            "locations, with visual distractors introduced mid-episode."
        ),
        memory_types=[MemoryType.SEQUENTIAL, MemoryType.SPATIAL],
        horizon=3000,
    ),
    MemoryTask(
        task_id="robocerebra/grilled_cheese",
        source_benchmark="robocerebra",
        description="Prepare a grilled cheese sandwich via a multi-step recipe.",
        memory_types=[MemoryType.SEQUENTIAL, MemoryType.TEMPORAL],
        horizon=2500,
    ),
    MemoryTask(
        task_id="robocerebra/serve_coffee",
        source_benchmark="robocerebra",
        description="Grind beans, brew coffee, and pour into the correct mug.",
        memory_types=[MemoryType.SEQUENTIAL, MemoryType.SPATIAL],
        horizon=2000,
    ),
    MemoryTask(
        task_id="robocerebra/reorganize_shelves",
        source_benchmark="robocerebra",
        description="Reorganize shelves by item category; handle dynamic object drops.",
        memory_types=[MemoryType.SPATIAL, MemoryType.EPISODIC],
        horizon=3200,
    ),
    MemoryTask(
        task_id="robocerebra/table_setting_4",
        source_benchmark="robocerebra",
        description="Set a table for 4 people with correct utensil placement.",
        memory_types=[MemoryType.SEQUENTIAL, MemoryType.CAPACITY],
        horizon=2200,
    ),
    MemoryTask(
        task_id="robocerebra/do_laundry",
        source_benchmark="robocerebra",
        description="Sort laundry, load washer, transfer to dryer, fold.",
        memory_types=[MemoryType.SEQUENTIAL, MemoryType.TEMPORAL, MemoryType.EPISODIC],
        horizon=3500,
    ),
    MemoryTask(
        task_id="robocerebra/office_filing",
        source_benchmark="robocerebra",
        description="File documents by category with interleaved shredding.",
        memory_types=[MemoryType.SEQUENTIAL, MemoryType.CAPACITY],
        horizon=2400,
    ),
    MemoryTask(
        task_id="robocerebra/medicine_sorting",
        source_benchmark="robocerebra",
        description="Sort medicines by name and dosage into labeled containers.",
        memory_types=[MemoryType.VISUAL, MemoryType.CAPACITY],
        horizon=1800,
    ),
    MemoryTask(
        task_id="robocerebra/garden_watering",
        source_benchmark="robocerebra",
        description="Water plants according to schedule without re-watering.",
        memory_types=[MemoryType.TEMPORAL, MemoryType.EPISODIC],
        horizon=2800,
    ),
    MemoryTask(
        task_id="robocerebra/birthday_party_setup",
        source_benchmark="robocerebra",
        description=(
            "Multi-phase party setup: decorations, cake, tableware, gifts."
        ),
        memory_types=[MemoryType.SEQUENTIAL, MemoryType.SPATIAL, MemoryType.CAPACITY],
        horizon=4000,
    ),
]


# ----- MNEMO-Real (5 new tasks, cross-session focus) -----------------------
# These tasks fill a structural gap in the four source benchmarks: none of
# them exercises genuine cross-session memory. MNEMO-Real tasks are meant
# to be executed on real hardware (Aloha-AgileX or similar); the
# specifications below are our v1.0 proposal.

_MNEMO_REAL = [
    MemoryTask(
        task_id="mnemo_real/find_lost_object",
        source_benchmark="mnemo_real",
        description=(
            "In a fresh session, retrieve an object whose location was "
            "established in an earlier session (persistent memory recall)."
        ),
        memory_types=[MemoryType.CROSS_SESSION, MemoryType.SPATIAL],
        horizon=1500,
    ),
    MemoryTask(
        task_id="mnemo_real/remember_user_preference",
        source_benchmark="mnemo_real",
        description=(
            "Execute a recipe whose step-level preferences (e.g., sugar "
            "amount) were taught in a previous session."
        ),
        memory_types=[MemoryType.CROSS_SESSION, MemoryType.SEQUENTIAL],
        horizon=2500,
    ),
    MemoryTask(
        task_id="mnemo_real/interrupted_task_resumption",
        source_benchmark="mnemo_real",
        description=(
            "Resume a multi-step task interrupted in a prior episode "
            "using the persisted subtask progress."
        ),
        memory_types=[MemoryType.CROSS_SESSION, MemoryType.EPISODIC],
        horizon=2000,
    ),
    MemoryTask(
        task_id="mnemo_real/user_correction_learning",
        source_benchmark="mnemo_real",
        description=(
            "Apply a correction (e.g., 'move slower near the fragile "
            "glass') taught in a prior session."
        ),
        memory_types=[MemoryType.CROSS_SESSION, MemoryType.VISUAL],
        horizon=1800,
    ),
    MemoryTask(
        task_id="mnemo_real/adaptive_grasp_from_history",
        source_benchmark="mnemo_real",
        description=(
            "Avoid a grasp strategy that previously failed on the same "
            "object by consulting the failure log in persistent memory."
        ),
        memory_types=[MemoryType.CROSS_SESSION, MemoryType.EPISODIC],
        horizon=1600,
    ),
]


# ----- Public API ----------------------------------------------------------


def rmbench_tasks() -> List[MemoryTask]:
    """Return the 9 RMBench tasks as MemoryTask objects."""
    return list(_RMBENCH)


def memorybench_tasks() -> List[MemoryTask]:
    """Return the 3 MemoryBench tasks (ported to RoboTwin 2.0)."""
    return list(_MEMORYBENCH)


def mikasa_tasks() -> List[MemoryTask]:
    """Return 10 representative MIKASA-Robo tasks."""
    return list(_MIKASA)


def robocerebra_tasks() -> List[MemoryTask]:
    """Return 10 representative RoboCerebra tasks."""
    return list(_ROBOCEREBRA)


def mnemo_real_tasks() -> List[MemoryTask]:
    """Return the 5 new MNEMO-Real cross-session tasks."""
    return list(_MNEMO_REAL)


def all_tasks() -> List[MemoryTask]:
    """Return the full MNEMO-Bench v1.0 task list (37 tasks)."""
    return (
        rmbench_tasks()
        + memorybench_tasks()
        + mikasa_tasks()
        + robocerebra_tasks()
        + mnemo_real_tasks()
    )
