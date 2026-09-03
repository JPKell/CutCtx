"""The deterministic last resort, on the cases the property suite would have to be lucky to draw."""

from __future__ import annotations

import pytest

from conftest import transcript, turn
from cutctx import Action, BudgetUnsatisfiable, CompactionBudget, DropOldestPolicy, Role

POLICY = DropOldestPolicy()


def actions_of(source: object, budget: CompactionBudget) -> dict[str, Action]:
    """Return the plan's action per turn id."""
    plan = POLICY.decide(source, budget)  # type: ignore[arg-type]
    return {a.turn_id: a.action for a in plan.actions}


def test_a_transcript_that_already_fits_is_untouched() -> None:
    source = transcript(turn("a", tokens=10), turn("b", tokens=10))

    assert set(actions_of(source, CompactionBudget(100, 0)).values()) == {Action.KEEP}


def test_the_oldest_go_first_and_only_as_many_as_needed() -> None:
    source = transcript(
        turn("a", tokens=30), turn("b", tokens=30), turn("c", tokens=30), turn("d", tokens=30)
    )

    assert actions_of(source, CompactionBudget(90, protected_recent_turns=1)) == {
        "a": Action.DROP,
        "b": Action.KEEP,
        "c": Action.KEEP,
        "d": Action.KEEP,
    }


def test_a_tool_exchange_goes_as_a_unit() -> None:
    source = transcript(
        turn("a1", Role.ASSISTANT, tokens=10, tool_call_id="c1"),
        turn("t1", Role.TOOL, tokens=200, tool_call_id="c1"),
        turn("t2", Role.TOOL, tokens=200, tool_call_id="c1"),
        turn("u1", tokens=10),
    )

    assert actions_of(source, CompactionBudget(50, protected_recent_turns=1)) == {
        "a1": Action.DROP,
        "t1": Action.DROP,
        "t2": Action.DROP,
        "u1": Action.KEEP,
    }


def test_a_protected_result_keeps_its_distant_call_alive() -> None:
    """Why "everything dropped is older than everything kept" is not true of this policy."""
    source = transcript(
        turn("a1", Role.ASSISTANT, tokens=100, tool_call_id="c1"),
        turn("u1", tokens=100),
        turn("t1", Role.TOOL, tokens=10, tool_call_id="c1"),
    )

    assert actions_of(source, CompactionBudget(120, protected_recent_turns=1)) == {
        "a1": Action.KEEP,
        "u1": Action.DROP,
        "t1": Action.KEEP,
    }


def test_the_system_turn_and_pinned_turns_survive_a_budget_that_needs_everything_else() -> None:
    source = transcript(
        turn("s1", Role.SYSTEM, tokens=10),
        turn("u1", tokens=500),
        turn("p1", tokens=10, pinned=True),
        turn("u2", tokens=500),
    )

    assert actions_of(source, CompactionBudget(20, protected_recent_turns=0)) == {
        "s1": Action.KEEP,
        "u1": Action.DROP,
        "p1": Action.KEEP,
        "u2": Action.DROP,
    }


def test_an_exhausted_policy_says_so_rather_than_truncating() -> None:
    """``budget_unmet`` with a *satisfiable* floor — which for this policy needs exchange lock-in.

    Worth stating, because it is not obvious: if every remaining turn were untouchable the floor
    itself would exceed the budget and the refusal would have fired instead. The only way this
    policy exhausts its options while the budget is still theoretically meetable is a turn that is
    not untouchable but cannot leave — a huge old call whose result sits in the protected tail.
    Row E1's masking policy is what will be able to reduce this case; drop-oldest cannot, and says
    so rather than truncating (ADR-0023).
    """
    source = transcript(
        turn("a1", Role.ASSISTANT, tokens=500, tool_call_id="c1"),
        turn("u1", tokens=5),
        turn("t1", Role.TOOL, tokens=5, tool_call_id="c1"),
    )
    budget = CompactionBudget(max_tokens=20, protected_recent_turns=1)

    plan = POLICY.decide(source, budget)

    assert plan.budget_unmet is True
    assert plan.tokens_after_estimate == 505
    assert {a.turn_id: a.action for a in plan.actions} == {
        "a1": Action.KEEP,
        "u1": Action.DROP,
        "t1": Action.KEEP,
    }


def test_an_impossible_budget_refuses_before_any_work() -> None:
    source = transcript(turn("p1", tokens=500, pinned=True), turn("u1", tokens=10))

    with pytest.raises(BudgetUnsatisfiable) as refusal:
        POLICY.decide(source, CompactionBudget(100, protected_recent_turns=0))

    assert refusal.value.details["untouchable_tokens"] == 500
    assert refusal.value.details["max_tokens"] == 100


def test_an_empty_transcript_plans_to_nothing() -> None:
    from cutctx import Transcript  # noqa: PLC0415 — only this test needs the empty case

    plan = POLICY.decide(Transcript(), CompactionBudget(0, protected_recent_turns=4))

    assert plan.actions == ()
    assert (plan.tokens_before, plan.tokens_after_estimate, plan.budget_unmet) == (0, 0, False)


def test_the_policy_records_no_estimator_ratio_because_it_estimates_nothing() -> None:
    """ADR-0016: the ratio rides on the plan only when the character default produced a figure."""
    source = transcript(turn("a", tokens=10))

    assert POLICY.decide(source, CompactionBudget(100, 0)).estimator_ratio is None


def test_the_name_and_version_are_part_of_the_audit_record() -> None:
    source = transcript(turn("a", tokens=10))
    plan = POLICY.decide(source, CompactionBudget(100, 0))

    assert (plan.policy_name, plan.policy_version) == ("drop_oldest", "1.0.0")
