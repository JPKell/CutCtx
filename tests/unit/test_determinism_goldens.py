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
from cutctx import (
    GROUP_ID_PREFIX,
    CompactionBudget,
    CompactionExecutor,
    CompactionPolicy,
    DropOldestPolicy,
    ObservationMaskingPolicy,
    Role,
    SummarizingPolicy,
    Transcript,
    default_chain,
)

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


# ------------------------------------------------------------------------------------------------
# Phase 2: the same fixtures and budgets, against every shipped policy and the default chain.
#
# Dev-plan AC2 is that *"the default chain compacts the Phase-1 golden transcripts to every budget
# in the golden set with byte-identical plans"*, so the cases below are the Phase-1 cases exactly —
# not a fresh set chosen to suit the new policies. A policy that cannot handle the transcripts the
# invariants were written against is a policy that has not been tested on them.
#
# The stub's text, the derived group id and the composed policy name are all inside these bytes, so
# a change to any of them is a change to every plan a consumer stored. Regenerating is the same
# deliberate act it was in Phase 1: delete the file, run the suite once, review the diff.
# ------------------------------------------------------------------------------------------------

PROMPT_ID = "compaction.summarize.v1"

PHASE_TWO_POLICIES: dict[str, CompactionPolicy] = {
    "masking": ObservationMaskingPolicy(),
    "summarizing": SummarizingPolicy(PROMPT_ID),
    "chain": default_chain(prompt_id=PROMPT_ID),
}


@pytest.mark.parametrize("label", sorted(PHASE_TWO_POLICIES))
@pytest.mark.parametrize(("fixture", "max_tokens", "protected"), CASES)
def test_every_phase_two_policy_is_byte_for_byte_what_it_was(
    label: str, fixture: str, max_tokens: int, protected: int
) -> None:
    policy = PHASE_TWO_POLICIES[label]
    budget = CompactionBudget(max_tokens=max_tokens, protected_recent_turns=protected)
    plan = policy.decide(FIXTURES[fixture], budget)
    name = f"{label}-{policy.version}-{fixture}-{max_tokens}-{protected}.json"
    golden = GOLDENS / name
    produced = plan.canonical_json() + "\n"

    if not golden.exists():
        golden.write_text(produced, encoding="utf-8")
        pytest.fail(f"golden {name} did not exist; it has been written — review and commit it")

    assert produced == golden.read_text(encoding="utf-8")


@pytest.mark.parametrize(("fixture", "max_tokens", "protected"), CASES)
def test_the_default_chains_report_is_byte_for_byte_what_it_was(
    fixture: str, max_tokens: int, protected: int
) -> None:
    """AC2's other half: the ``context.compacted`` body two applications emit unreshaped.

    The summaries are supplied here as a fixed string per group, which is what a caller does — and
    the report is unaffected by their length, because nothing in it is computed (`C1_HANDOFF.md`
    §3.1). A report that moved when the summary text moved would be a report that could contradict
    the plan it points at.
    """
    policy = PHASE_TWO_POLICIES["chain"]
    budget = CompactionBudget(max_tokens=max_tokens, protected_recent_turns=protected)
    source = FIXTURES[fixture]
    plan = policy.decide(source, budget)
    summaries = {request.group_id: "a summary" for request in plan.summarization_requests}
    report = CompactionExecutor().apply(source, plan, summaries).report
    name = f"report-chain-{policy.version}-{fixture}-{max_tokens}-{protected}.json"
    golden = GOLDENS / name
    produced = canonical_json(report.to_dict()) + "\n"

    if not golden.exists():
        golden.write_text(produced, encoding="utf-8")
        pytest.fail(f"golden {name} did not exist; it has been written — review and commit it")

    assert produced == golden.read_text(encoding="utf-8")


# The Phase-1 fixtures are small, and at every budget in `CASES` the shipped policies reach them by
# keeping or dropping — which locks in the *arithmetic* but not the stub's text, the derived group
# id or the composed name. Those are the bytes Phase 2 introduced, so they get fixtures big enough
# to produce them. Deliberately a separate set: `CASES` above is AC2's wording and stays as it is.

BIG_FIXTURES: dict[str, Transcript] = {
    "tool_heavy": transcript(
        turn("s1", Role.SYSTEM, tokens=10, content="You are careful."),
        turn("u1", tokens=20, content="Find the regression."),
        turn("a1", Role.ASSISTANT, tokens=25, tool_call_id="c1", content="Reading the log."),
        turn("t1", Role.TOOL, tokens=300, tool_call_id="c1", content="L" * 1200),
        turn("a2", Role.ASSISTANT, tokens=25, content="The cache key changed."),
        turn("a3", Role.ASSISTANT, tokens=25, tool_call_id="c2", content="Checking the diff."),
        turn("t2", Role.TOOL, tokens=300, tool_call_id="c2", content="D" * 1200),
        turn("a4", Role.ASSISTANT, tokens=25, content="Confirmed in the lockfile."),
        turn("a5", Role.ASSISTANT, tokens=25, tool_call_id="c3", content="One more look."),
        turn("t3", Role.TOOL, tokens=300, tool_call_id="c3", content="M" * 1200),
        turn("u2", tokens=15, content="Thanks."),
    ),
    "long_chat": transcript(
        turn("s1", Role.SYSTEM, tokens=10, content="You are careful."),
        *(
            turn(
                f"m{index}",
                Role.USER if index % 2 == 0 else Role.ASSISTANT,
                tokens=50,
                content=f"message {index}",
            )
            for index in range(10)
        ),
        turn("u9", tokens=15, content="Anything else?"),
    ),
}

BIG_CASES = [
    ("tool_heavy", 1_400, 0),
    ("tool_heavy", 700, 0),
    ("tool_heavy", 400, 2),
    ("tool_heavy", 200, 0),
    ("long_chat", 400, 0),
    ("long_chat", 200, 2),
    ("long_chat", 100, 0),
]


@pytest.mark.parametrize("label", sorted(PHASE_TWO_POLICIES))
@pytest.mark.parametrize(("fixture", "max_tokens", "protected"), BIG_CASES)
def test_the_phase_two_actions_are_byte_for_byte_what_they_were(
    label: str, fixture: str, max_tokens: int, protected: int
) -> None:
    """The stub's text, the derived group id and the composed policy name are in these bytes."""
    policy = PHASE_TWO_POLICIES[label]
    budget = CompactionBudget(max_tokens=max_tokens, protected_recent_turns=protected)
    plan = policy.decide(BIG_FIXTURES[fixture], budget)
    name = f"{label}-{policy.version}-{fixture}-{max_tokens}-{protected}.json"
    golden = GOLDENS / name
    produced = plan.canonical_json() + "\n"

    if not golden.exists():
        golden.write_text(produced, encoding="utf-8")
        pytest.fail(f"golden {name} did not exist; it has been written — review and commit it")

    assert produced == golden.read_text(encoding="utf-8")


def test_the_phase_two_goldens_actually_contain_a_stub_and_a_summary_group() -> None:
    """A golden set that locked in only ``keep`` and ``drop`` would lock in nothing new.

    Phase 1's fixtures are reached by keeping or dropping at every budget in `CASES`, so without
    this assertion the whole Phase-2 golden set could quietly stop exercising Phase-2 behaviour and
    still pass.
    """
    produced = "".join(
        PHASE_TWO_POLICIES[label]
        .decide(
            BIG_FIXTURES[fixture],
            CompactionBudget(max_tokens=max_tokens, protected_recent_turns=protected),
        )
        .canonical_json()
        for label in PHASE_TWO_POLICIES
        for fixture, max_tokens, protected in BIG_CASES
    )
    assert '"mask"' in produced
    assert '"summarize"' in produced
    assert "sha256:" in produced
    assert GROUP_ID_PREFIX in produced


@pytest.mark.parametrize("label", sorted(PHASE_TWO_POLICIES))
@pytest.mark.parametrize(("fixture", "max_tokens", "protected"), CASES)
def test_a_second_derivation_in_this_process_agrees_with_the_first(
    label: str, fixture: str, max_tokens: int, protected: int
) -> None:
    """The goldens cover other machines; this covers a policy that carries state between calls.

    A policy that accumulated anything — a counter for group ids, a cache keyed on the last
    transcript — would pass the golden on a cold run and differ on the second call in one process.
    """
    policy = PHASE_TWO_POLICIES[label]
    budget = CompactionBudget(max_tokens=max_tokens, protected_recent_turns=protected)
    first = policy.decide(FIXTURES[fixture], budget)
    second = policy.decide(FIXTURES[fixture], budget)
    assert first.canonical_json() == second.canonical_json()
