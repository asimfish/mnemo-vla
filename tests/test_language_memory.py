"""Tests for mnemo.language_memory."""

from mnemo.language_memory import (
    Memory,
    SubtaskStatus,
    parse_memory_text,
)


def test_add_and_advance_subtasks():
    memory = Memory(task="Make tomato soup.")
    memory.add_subtask("Wash tomatoes")
    memory.add_subtask("Cut tomatoes")
    memory.add_subtask("Heat oil")

    started = memory.start_next_pending()
    assert started is not None
    assert started.status == SubtaskStatus.CURRENT
    assert started.description == "Wash tomatoes"

    memory.mark_current_done()
    assert memory.subtasks[0].status == SubtaskStatus.DONE

    nxt = memory.start_next_pending()
    assert nxt.description == "Cut tomatoes"


def test_cross_session_and_failures():
    memory = Memory(task="Find the knife.")
    memory.add_cross_session(
        content="Knife is in the top-right drawer.",
        source_session="2027-03-10",
        confidence=0.9,
    )
    memory.add_failure(
        description="Could not open drawer",
        intervention="Asked user",
    )
    assert len(memory.cross_session) == 1
    assert len(memory.failures) == 1


def test_text_roundtrip():
    original = Memory(task="Clean kitchen.", session_id="test-session-1")
    original.add_subtask("Wash dish", status=SubtaskStatus.DONE)
    original.add_subtask("Dry dish", status=SubtaskStatus.CURRENT, details="with towel")
    original.add_subtask("Store dish")
    original.add_cross_session(
        content="Dishrack is on the left counter.",
        source_session="yesterday",
        confidence=0.85,
    )
    original.add_failure(
        description="Slipped while grasping",
        intervention="Used two-hand grasp",
    )

    text = original.to_text()
    parsed = parse_memory_text(text)

    assert parsed.task == original.task
    assert parsed.session_id == original.session_id
    assert len(parsed.subtasks) == 3
    assert parsed.subtasks[0].status == SubtaskStatus.DONE
    assert parsed.subtasks[1].status == SubtaskStatus.CURRENT
    assert parsed.subtasks[1].details == "with towel"
    assert len(parsed.cross_session) == 1
    assert parsed.cross_session[0].confidence == 0.85
    assert len(parsed.failures) == 1
    assert parsed.failures[0].intervention == "Used two-hand grasp"


def test_dict_roundtrip():
    original = Memory(task="x")
    original.add_subtask("a", SubtaskStatus.DONE)
    restored = Memory.from_dict(original.to_dict())
    assert restored.task == "x"
    assert restored.subtasks[0].status == SubtaskStatus.DONE
