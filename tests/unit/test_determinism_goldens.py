"""Committed golden plans — byte-identity across runs, platforms and Python versions (G8).

A plan appears in an audit record, and a record nobody can reproduce is not evidence (spec §11
contract 4). The property suite asserts that *one* process derives the same plan twice; these
files assert that this process derives what a different machine, on a different Python, derived
when they were written. The CI matrix runs them on 3.12, 3.13 and 3.14.

Each golden is the plan's canonical JSON — sorted keys, minimal separators — so the file is the
bytes, not a rendering of them.

**Regenerating.** Delete the file and run the suite: a missing golden is written and the test
fails once, so the new bytes arrive as a reviewable diff rather than as a silent overwrite. A
changed plan for an unchanged policy version is a defect; a deliberate behaviour change bumps
``DropOldestPolicy.version`` and the golden's name changes with it (spec §19).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from baseaicore import canonical_json

from conftest import transcript, turn
from cutctx import CompactionBudget, CompactionExecutor, DropOldestPolicy, Role, Transcript

GOLDENS = Path(__file__).parent.parent / "goldens"
POLICY = DropOldestPolicy()

FIXTURES: dict[str, Transcript] = {
    "simple_chat": transcript(
        turn("s1", Role.SYSTEM, tokens=12, content="You are a careful assistant."),
        turn("u1", tokens=30, content="What changed in the build?"),
        turn("a1", Role.ASSISTANT, tokens=25, content="The cache key changed."),
        turn("u2", tokens=18, content="Why?"),
        turn("a2", Role.ASSISTANT, tokens=40, content="Because the lockfile moved."),
    ),
    "multi_call_exchange": transcript(
        turn("s1", Role.SYSTEM, tokens=12),
        turn("a1", Role.ASSISTANT, tokens=20, tool_call_id="c1"),
        turn("t1", Role.TOOL, tokens=400, tool_call_id="c1"),
        turn("t2", Role.TOOL, tokens=350, tool_call_id="c1"),
        turn("a2", Role.ASSISTANT, tokens=30),
    ),
    "displaced_result": transcript(
        turn("a1", Role.ASSISTANT, tokens=200, tool_call_id="c1"),
        turn("u1", tokens=60),
        turn("u2", tokens=60),
        turn("t1", Role.TOOL, tokens=15, tool_call_id="c1"),
    ),
    "entirely_pinned": transcript(
        turn("p1", tokens=10, pinned=True),
        turn("p2", tokens=10, pinned=True),
        turn("p3", tokens=10, pinned=True),
    ),
    "orphaned_result": transcript(
        turn("t0", Role.TOOL, tokens=500, tool_call_id="gone"),
        turn("u1", tokens=20),
        turn("a1", Role.ASSISTANT, tokens=20),
    ),
    "empty": Transcript(),
}

CASES = [
    ("simple_chat", 200, 2),
    ("simple_chat", 60, 1),
    ("multi_call_exchange", 100, 1),
    ("multi_call_exchange", 900, 2),
    ("displaced_result", 240, 1),
    ("entirely_pinned", 30, 0),
    ("orphaned_result", 45, 1),
    ("empty", 0, 4),
]


@pytest.mark.parametrize(("fixture", "max_tokens", "protected"), CASES)
def test_the_plan_is_byte_for_byte_what_it_was(
    fixture: str, max_tokens: int, protected: int
) -> None:
    budget = CompactionBudget(max_tokens=max_tokens, protected_recent_turns=protected)
    plan = POLICY.decide(FIXTURES[fixture], budget)
    name = f"{POLICY.name}-{POLICY.version}-{fixture}-{max_tokens}-{protected}.json"
    golden = GOLDENS / name
    produced = plan.canonical_json() + "\n"

    if not golden.exists():
        golden.write_text(produced, encoding="utf-8")
        pytest.fail(f"golden {name} did not exist; it has been written — review and commit it")

    assert produced == golden.read_text(encoding="utf-8")


@pytest.mark.parametrize(("fixture", "max_tokens", "protected"), CASES)
def test_the_report_is_byte_for_byte_what_it_was(
    fixture: str, max_tokens: int, protected: int
) -> None:
    """The ``context.compacted`` body is golden-locked too: two applications emit it unreshaped."""
    budget = CompactionBudget(max_tokens=max_tokens, protected_recent_turns=protected)
    source = FIXTURES[fixture]
    plan = POLICY.decide(source, budget)
    report = CompactionExecutor().apply(source, plan).report
    name = f"report-{POLICY.name}-{POLICY.version}-{fixture}-{max_tokens}-{protected}.json"
    golden = GOLDENS / name
    produced = canonical_json(report.to_dict()) + "\n"

    if not golden.exists():
        golden.write_text(produced, encoding="utf-8")
        pytest.fail(f"golden {name} did not exist; it has been written — review and commit it")

    assert produced == golden.read_text(encoding="utf-8")


def test_the_fixtures_cover_the_shapes_the_invariants_are_about() -> None:
    """A golden set that only held well-behaved transcripts would lock in nothing worth locking."""
    assert {"multi_call_exchange", "displaced_result", "entirely_pinned", "empty"} <= set(FIXTURES)
