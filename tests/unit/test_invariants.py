"""The invariant module's refusals, one rule at a time, with the details a caller reads.

The property suite asserts that valid plans satisfy the invariants. These assert the other half —
that an *invalid* plan cannot be built — which is the half a property suite over one honest policy
cannot reach, because an honest policy never writes one.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from baseaicore import ValidationError

from conftest import rebuilt, transcript, turn
from cutctx import (
    Action,
    BudgetUnsatisfiable,
    CompactionBudget,
    CompactionPlan,
    DropOldestPolicy,
    Role,
    SummarizationRequest,
    Transcript,
    TurnAction,
    TurnReplacement,
    policies,
)
from cutctx._invariants import (
    build_plan,
    exchanges,
    removable_turn_ids,
    require_satisfiable_budget,
    untouchable_tokens,
    untouchable_turn_ids,
)

WIDE = CompactionBudget(max_tokens=10_000, protected_recent_turns=0)


def keeps(source: Transcript) -> tuple[TurnAction, ...]:
    """Return the identity plan's actions for a transcript."""
    return tuple(TurnAction(turn_id=t.turn_id, action=Action.KEEP) for t in source.turns)


def plan_of(source: Transcript, budget: CompactionBudget = WIDE) -> CompactionPlan:
    """Return the identity plan — a valid plan to mutilate."""
    return build_plan(
        transcript=source,
        budget=budget,
        actions=keeps(source),
        policy_name="test",
        policy_version="1.0.0",
    )


# --------------------------------------------------------------------------------------------
# Exchanges — the partition contract 3 is stated over
# --------------------------------------------------------------------------------------------


def test_a_multi_call_turn_and_all_its_results_are_one_exchange() -> None:
    source = transcript(
        turn("a1", Role.ASSISTANT, tool_call_id="c1"),
        turn("t1", Role.TOOL, tool_call_id="c1"),
        turn("t2", Role.TOOL, tool_call_id="c1"),
        turn("t3", Role.TOOL, tool_call_id="c1"),
    )

    assert exchanges(source) == (("a1", "t1", "t2", "t3"),)


def test_an_exchange_holds_together_across_intervening_turns() -> None:
    """A result far from its call is still its call's — position plays no part in the partition."""
    source = transcript(
        turn("a1", Role.ASSISTANT, tool_call_id="c1"),
        turn("u1"),
        turn("u2"),
        turn("t1", Role.TOOL, tool_call_id="c1"),
    )

    assert exchanges(source) == (("a1", "t1"), ("u1",), ("u2",))


def test_a_result_whose_call_is_gone_is_an_exchange_of_one() -> None:
    source = transcript(turn("t1", Role.TOOL, tool_call_id="vanished"))

    assert exchanges(source) == (("t1",),)


def test_a_tool_call_id_equal_to_another_turns_id_does_not_merge_exchanges() -> None:
    """The keys are namespaced, so an id reused across the two fields cannot fuse two exchanges."""
    source = transcript(turn("c1"), turn("a1", Role.ASSISTANT, tool_call_id="c1"))

    assert exchanges(source) == (("c1",), ("a1",))


def test_protection_closes_over_an_exchange() -> None:
    """A protected result makes its distant call unremovable — nothing else does."""
    source = transcript(
        turn("a1", Role.ASSISTANT, tool_call_id="c1"),
        turn("u1"),
        turn("t1", Role.TOOL, tool_call_id="c1"),
    )
    budget = CompactionBudget(max_tokens=100, protected_recent_turns=1)

    assert untouchable_turn_ids(source, budget) == {"t1"}
    assert removable_turn_ids(source, budget) == {"u1"}


def test_every_system_turn_is_untouchable_wherever_it_sits() -> None:
    source = transcript(turn("u1"), turn("s1", Role.SYSTEM), turn("u2"))

    assert untouchable_turn_ids(source, CompactionBudget(100, protected_recent_turns=0)) == {"s1"}


# --------------------------------------------------------------------------------------------
# BudgetUnsatisfiable — both numbers, and the boundary
# --------------------------------------------------------------------------------------------


def test_an_unsatisfiable_budget_names_both_numbers() -> None:
    source = transcript(turn("s1", Role.SYSTEM, tokens=40), turn("u1", tokens=10, pinned=True))
    budget = CompactionBudget(max_tokens=30, protected_recent_turns=0)

    with pytest.raises(BudgetUnsatisfiable) as refusal:
        require_satisfiable_budget(source, budget)

    assert refusal.value.code == "COMPACTION_BUDGET_UNSATISFIABLE"
    assert refusal.value.details["untouchable_tokens"] == 50
    assert refusal.value.details["max_tokens"] == 30
    assert refusal.value.details["untouchable_turn_count"] == 2
    assert "50" in str(refusal.value)
    assert "30" in str(refusal.value)


def test_a_budget_that_exactly_fits_the_untouchable_set_is_satisfiable() -> None:
    source = transcript(turn("s1", Role.SYSTEM, tokens=40))

    require_satisfiable_budget(source, CompactionBudget(40, protected_recent_turns=0))

    with pytest.raises(BudgetUnsatisfiable):
        require_satisfiable_budget(source, CompactionBudget(39, protected_recent_turns=0))


def test_the_untouchable_floor_counts_each_turn_once() -> None:
    """A turn that is pinned *and* in the protected tail is one turn, not two."""
    source = transcript(turn("u1", tokens=10, pinned=True))

    assert untouchable_tokens(source, CompactionBudget(100, protected_recent_turns=1)) == 10


# --------------------------------------------------------------------------------------------
# Rule 1 — coverage
# --------------------------------------------------------------------------------------------


def test_a_plan_that_forgets_a_turn_is_refused() -> None:
    source = transcript(turn("a"), turn("b"))
    plan = plan_of(source)

    with pytest.raises(ValidationError) as refusal:
        rebuilt(plan, source, WIDE, actions=plan.actions[:-1])

    assert refusal.value.details["unplanned_turn_ids"] == ["b"]


def test_a_plan_that_names_a_turn_twice_is_refused() -> None:
    source = transcript(turn("a"), turn("b"))
    plan = plan_of(source)

    with pytest.raises(ValidationError) as refusal:
        rebuilt(plan, source, WIDE, actions=(*plan.actions, plan.actions[0]))

    assert refusal.value.details["repeated_turn_ids"] == ["a"]


def test_a_plan_for_turns_this_transcript_does_not_have_is_refused() -> None:
    source = transcript(turn("a"))
    plan = plan_of(source)
    other = transcript(turn("z"))

    with pytest.raises(ValidationError) as refusal:
        rebuilt(plan, other, WIDE)

    assert refusal.value.details["unplanned_turn_ids"] == ["z"]
    assert refusal.value.details["unknown_turn_ids"] == ["a"]


def test_a_plan_whose_actions_are_out_of_order_is_refused() -> None:
    """The same ids in another order: every set-based check passes and the plan is still wrong."""
    source = transcript(turn("a"), turn("b"))
    plan = plan_of(source)

    with pytest.raises(ValidationError) as refusal:
        rebuilt(plan, source, WIDE, actions=tuple(reversed(plan.actions)))

    assert refusal.value.details["unplanned_turn_ids"] == []
    assert "order differs" in str(refusal.value)


# --------------------------------------------------------------------------------------------
# Rule 2 — the untouchable set
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "budget"),
    [
        ("s1", CompactionBudget(1_000, protected_recent_turns=0)),
        ("p1", CompactionBudget(1_000, protected_recent_turns=0)),
        ("u2", CompactionBudget(1_000, protected_recent_turns=1)),
    ],
    ids=["system", "pinned", "protected-tail"],
)
def test_a_plan_that_acts_on_an_untouchable_turn_is_refused(
    target: str, budget: CompactionBudget
) -> None:
    source = transcript(turn("s1", Role.SYSTEM), turn("p1", pinned=True), turn("u1"), turn("u2"))
    plan = plan_of(source, budget)
    actions = tuple(
        TurnAction(turn_id=a.turn_id, action=Action.DROP if a.turn_id == target else a.action)
        for a in plan.actions
    )

    with pytest.raises(ValidationError) as refusal:
        rebuilt(plan, source, budget, actions=actions, tokens_after_estimate=30, budget_unmet=False)

    assert refusal.value.details["touched_turn_ids"] == [target]


def test_masking_an_untouchable_turn_is_refused_too() -> None:
    """ "Never masked, summarized or dropped" — masking is not a loophole in contract 2."""
    source = transcript(turn("p1", tokens=10, pinned=True))
    plan = plan_of(source)
    actions = (
        TurnAction(
            turn_id="p1",
            action=Action.MASK,
            replacement=TurnReplacement(content="[masked]", token_estimate=1),
        ),
    )

    with pytest.raises(ValidationError) as refusal:
        rebuilt(plan, source, WIDE, actions=actions, tokens_after_estimate=1)

    assert refusal.value.details["touched_turn_ids"] == ["p1"]


# --------------------------------------------------------------------------------------------
# Rule 3 — tool exchanges
# --------------------------------------------------------------------------------------------


def test_dropping_a_call_without_its_result_is_refused() -> None:
    source = transcript(
        turn("a1", Role.ASSISTANT, tokens=10, tool_call_id="c1"),
        turn("t1", Role.TOOL, tokens=10, tool_call_id="c1"),
    )
    plan = plan_of(source)
    actions = (
        TurnAction(turn_id="a1", action=Action.DROP),
        TurnAction(turn_id="t1", action=Action.KEEP),
    )

    with pytest.raises(ValidationError) as refusal:
        rebuilt(plan, source, WIDE, actions=actions, tokens_after_estimate=10)

    assert refusal.value.details["removed_turn_ids"] == ["a1"]
    assert refusal.value.details["retained_turn_ids"] == ["t1"]


def test_dropping_one_result_of_a_multi_call_turn_is_refused() -> None:
    """The direction a one-to-one pair check misses: the call survives with partial answers."""
    source = transcript(
        turn("a1", Role.ASSISTANT, tokens=10, tool_call_id="c1"),
        turn("t1", Role.TOOL, tokens=10, tool_call_id="c1"),
        turn("t2", Role.TOOL, tokens=10, tool_call_id="c1"),
    )
    plan = plan_of(source)
    actions = (
        TurnAction(turn_id="a1", action=Action.KEEP),
        TurnAction(turn_id="t1", action=Action.KEEP),
        TurnAction(turn_id="t2", action=Action.DROP),
    )

    with pytest.raises(ValidationError) as refusal:
        rebuilt(plan, source, WIDE, actions=actions, tokens_after_estimate=20)

    assert refusal.value.details["removed_turn_ids"] == ["t2"]


def test_dropping_half_an_exchange_and_summarizing_the_other_half_is_refused() -> None:
    """It orphans nothing and describes nothing: not one of contract 3's three cases."""
    source = transcript(
        turn("a1", Role.ASSISTANT, tokens=10, tool_call_id="c1"),
        turn("t1", Role.TOOL, tokens=10, tool_call_id="c1"),
    )
    plan = plan_of(source)
    actions = (
        TurnAction(turn_id="a1", action=Action.DROP),
        TurnAction(turn_id="t1", action=Action.SUMMARIZE, summary_group="g1"),
    )
    requests = (
        SummarizationRequest(
            group_id="g1", turn_ids=("t1",), target_tokens=3, prompt_id="general.summarize"
        ),
    )

    with pytest.raises(ValidationError) as refusal:
        rebuilt(
            plan,
            source,
            WIDE,
            actions=actions,
            summarization_requests=requests,
            tokens_after_estimate=3,
        )

    assert refusal.value.details["actions"] == ["drop", "summarize"]


def test_summarizing_one_exchange_into_two_groups_is_refused() -> None:
    source = transcript(
        turn("a1", Role.ASSISTANT, tokens=10, tool_call_id="c1"),
        turn("t1", Role.TOOL, tokens=10, tool_call_id="c1"),
    )
    plan = plan_of(source)
    actions = (
        TurnAction(turn_id="a1", action=Action.SUMMARIZE, summary_group="g1"),
        TurnAction(turn_id="t1", action=Action.SUMMARIZE, summary_group="g2"),
    )
    requests = (
        SummarizationRequest("g1", ("a1",), 3, "general.summarize"),
        SummarizationRequest("g2", ("t1",), 4, "general.summarize"),
    )

    with pytest.raises(ValidationError) as refusal:
        rebuilt(
            plan,
            source,
            WIDE,
            actions=actions,
            summarization_requests=requests,
            tokens_after_estimate=7,
        )

    assert refusal.value.details["summary_groups"] == ["g1", "g2"]


def test_masking_part_of_an_exchange_is_allowed() -> None:
    """The reading contract 3 must have: masking a result beside a kept call orphans nothing.

    This is the case row E1's ``ObservationMaskingPolicy`` is: "reasoning stays, bulk goes". A
    stricter reading of "masked together" would forbid the policy the spec itself ships.
    """
    source = transcript(
        turn("a1", Role.ASSISTANT, tokens=10, tool_call_id="c1"),
        turn("t1", Role.TOOL, tokens=900, tool_call_id="c1"),
    )
    actions = (
        TurnAction(turn_id="a1", action=Action.KEEP),
        TurnAction(
            turn_id="t1",
            action=Action.MASK,
            replacement=TurnReplacement(content="[tool result masked]", token_estimate=5),
        ),
    )

    plan = build_plan(
        transcript=source,
        budget=WIDE,
        actions=actions,
        policy_name="test",
        policy_version="1.0.0",
    )

    assert plan.tokens_after_estimate == 15


# --------------------------------------------------------------------------------------------
# Rule 4 — summary groups and their requests
# --------------------------------------------------------------------------------------------


def test_a_summary_group_with_no_request_is_refused() -> None:
    source = transcript(turn("u1", tokens=10), turn("u2", tokens=10))
    plan = plan_of(source)
    actions = (
        TurnAction(turn_id="u1", action=Action.SUMMARIZE, summary_group="g1"),
        TurnAction(turn_id="u2", action=Action.KEEP),
    )

    with pytest.raises(ValidationError) as refusal:
        rebuilt(plan, source, WIDE, actions=actions, tokens_after_estimate=10)

    assert refusal.value.details["groups_without_request"] == ["g1"]


def test_a_request_naming_turns_the_actions_do_not_is_refused() -> None:
    source = transcript(turn("u1", tokens=10), turn("u2", tokens=10))
    plan = plan_of(source)
    actions = (
        TurnAction(turn_id="u1", action=Action.SUMMARIZE, summary_group="g1"),
        TurnAction(turn_id="u2", action=Action.KEEP),
    )
    requests = (SummarizationRequest("g1", ("u1", "u2"), 4, "general.summarize"),)

    with pytest.raises(ValidationError) as refusal:
        rebuilt(
            plan,
            source,
            WIDE,
            actions=actions,
            summarization_requests=requests,
            tokens_after_estimate=14,
        )

    assert refusal.value.details["group_id"] == "g1"


def test_repeated_group_ids_are_refused() -> None:
    source = transcript(turn("u1", tokens=10), turn("u2", tokens=10))
    plan = plan_of(source)
    actions = (
        TurnAction(turn_id="u1", action=Action.SUMMARIZE, summary_group="g1"),
        TurnAction(turn_id="u2", action=Action.SUMMARIZE, summary_group="g1"),
    )
    requests = (
        SummarizationRequest("g1", ("u1", "u2"), 4, "general.summarize"),
        SummarizationRequest("g1", ("u1", "u2"), 4, "general.summarize"),
    )

    with pytest.raises(ValidationError) as refusal:
        rebuilt(
            plan,
            source,
            WIDE,
            actions=actions,
            summarization_requests=requests,
            tokens_after_estimate=8,
        )

    assert refusal.value.details["group_ids"] == ["g1", "g1"]


def test_a_group_id_whose_summary_turn_would_collide_is_refused() -> None:
    source = transcript(turn("summary:g1", tokens=10), turn("u2", tokens=10))
    plan = plan_of(source)
    actions = (
        TurnAction(turn_id="summary:g1", action=Action.KEEP),
        TurnAction(turn_id="u2", action=Action.SUMMARIZE, summary_group="g1"),
    )
    requests = (SummarizationRequest("g1", ("u2",), 4, "general.summarize"),)

    with pytest.raises(ValidationError) as refusal:
        rebuilt(
            plan,
            source,
            WIDE,
            actions=actions,
            summarization_requests=requests,
            tokens_after_estimate=14,
        )

    assert refusal.value.details["group_ids"] == ["g1"]


def test_requests_out_of_position_order_are_refused() -> None:
    """Their order is part of the plan's bytes, so it may not come from a dict's history.

    This is the dict-ordering trap the development plan names for Phase 2's group assembly; the
    rule that catches it is written now, before the policy that could fall into it exists.
    """
    source = transcript(turn("u1", tokens=10), turn("u2", tokens=10))
    plan = plan_of(source)
    actions = (
        TurnAction(turn_id="u1", action=Action.SUMMARIZE, summary_group="g1"),
        TurnAction(turn_id="u2", action=Action.SUMMARIZE, summary_group="g2"),
    )
    forward = (
        SummarizationRequest("g1", ("u1",), 4, "general.summarize"),
        SummarizationRequest("g2", ("u2",), 4, "general.summarize"),
    )
    rebuilt(
        plan, source, WIDE, actions=actions, summarization_requests=forward, tokens_after_estimate=8
    )

    with pytest.raises(ValidationError) as refusal:
        rebuilt(
            plan,
            source,
            WIDE,
            actions=actions,
            summarization_requests=tuple(reversed(forward)),
            tokens_after_estimate=8,
        )

    assert "byte-identical" in str(refusal.value)


# --------------------------------------------------------------------------------------------
# Rules 5 and 6 — arithmetic and honesty
# --------------------------------------------------------------------------------------------


def test_a_plan_that_misstates_tokens_before_is_refused() -> None:
    source = transcript(turn("u1", tokens=10))
    plan = plan_of(source)

    with pytest.raises(ValidationError) as refusal:
        rebuilt(plan, source, WIDE, tokens_before=999)

    assert refusal.value.details == {"declared": 999, "transcript": 10}


def test_a_plan_that_misstates_tokens_after_is_refused() -> None:
    source = transcript(turn("u1", tokens=10))
    plan = plan_of(source)

    with pytest.raises(ValidationError) as refusal:
        rebuilt(plan, source, WIDE, tokens_after_estimate=1)

    assert refusal.value.details == {"declared": 1, "derived": 10}


def test_a_plan_that_claims_to_fit_while_being_over_is_refused() -> None:
    """The one the whole trichotomy rests on: there is no fourth case."""
    source = transcript(turn("u1", tokens=10, pinned=True))
    tight = CompactionBudget(max_tokens=5, protected_recent_turns=0)

    with pytest.raises(BudgetUnsatisfiable):
        plan_of(source, tight)

    # A floor that fits (3) with a plan that does not (13): the only shape `budget_unmet` covers.
    loose = transcript(turn("u1", tokens=10), turn("u2", tokens=3))
    reachable = CompactionBudget(max_tokens=5, protected_recent_turns=1)
    plan = build_plan(
        transcript=loose,
        budget=reachable,
        actions=keeps(loose),
        policy_name="test",
        policy_version="1.0.0",
    )
    assert plan.budget_unmet is True

    with pytest.raises(ValidationError) as refusal:
        rebuilt(plan, loose, reachable, budget_unmet=False)

    assert refusal.value.details["tokens_after_estimate"] == 13
    assert refusal.value.details["max_tokens"] == 5


# --------------------------------------------------------------------------------------------
# The guard: the invariant module is the only path to a plan
# --------------------------------------------------------------------------------------------


class SkipsTheInvariants:
    """A policy written in a hurry — the one this guard exists to stop.

    It never calls :func:`cutctx._invariants.build_plan`. It builds the plan itself, drops a
    pinned turn, and writes figures that suit it. Every one of its own tests would pass.
    """

    name = "delinquent"
    version = "0.0.0"

    def decide(self, source: Transcript, budget: CompactionBudget) -> CompactionPlan:
        """Return the plan a policy that skipped every rule would return."""
        return CompactionPlan(
            actions=tuple(TurnAction(turn_id=t.turn_id, action=Action.DROP) for t in source.turns),
            tokens_before=0,
            tokens_after_estimate=0,
            policy_name=self.name,
            policy_version=self.version,
            transcript=source,
            budget=budget,
        )


def test_a_policy_that_skips_the_invariants_still_cannot_produce_a_plan() -> None:
    """Acceptance criterion 2, proved by a delinquent policy rather than asserted.

    The guard is not a convention a policy can forget: the transcript and budget are constructor
    arguments, so the only plan this policy can reach is a validated one — and its is not valid.
    """
    source = transcript(turn("p1", tokens=10, pinned=True), turn("u1", tokens=10))

    with pytest.raises(ValidationError):
        SkipsTheInvariants().decide(source, WIDE)


def test_a_plan_cannot_be_constructed_without_the_transcript_it_is_a_plan_for() -> None:
    """There is no shorter constructor to find: omitting the InitVars is a ``TypeError``."""
    with pytest.raises(TypeError):
        CompactionPlan(  # type: ignore[call-arg]  # the omission is the point of the test
            actions=(),
            tokens_before=0,
            tokens_after_estimate=0,
            policy_name="x",
            policy_version="1",
        )


def test_an_honest_direct_construction_is_accepted() -> None:
    """The guard is about validity, not ceremony: a correct plan built by hand is a valid plan."""
    source = transcript(turn("u1", tokens=10))

    plan = CompactionPlan(
        actions=(TurnAction(turn_id="u1", action=Action.KEEP),),
        tokens_before=10,
        tokens_after_estimate=10,
        policy_name="by-hand",
        policy_version="1.0.0",
        transcript=source,
        budget=WIDE,
    )

    assert plan.policy_name == "by-hand"


@pytest.mark.contract
def test_no_shipped_policy_builds_its_own_plan() -> None:
    """Every module under ``policies/`` reaches a plan through ``build_plan`` and no other way.

    The second guard, and the cheaper one. Direct construction is *safe* — it validates — but it
    skips the arithmetic and the ``budget_unmet`` determination, which means a policy doing it is
    writing figures the validator is about to recompute. Row E1 adds three policies to this
    directory; this test is what tells their author, at CI time, that there is one way in.
    """
    directory = Path(inspect.getfile(policies)).parent
    modules = sorted(p for p in directory.glob("*.py") if p.name != "__init__.py")
    assert modules, "no policy modules found — this test would pass vacuously"

    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        # Constructed, not merely named: the type appears in every policy's return annotation.
        constructed = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "build_plan" in called, f"{module.name} does not build its plan through _invariants"
        assert "CompactionPlan" not in constructed, (
            f"{module.name} constructs a CompactionPlan directly, which skips build_plan's "
            "arithmetic and its budget_unmet determination"
        )


def test_the_shipped_policy_satisfies_the_policy_protocol() -> None:
    from cutctx import CompactionPolicy  # noqa: PLC0415 — the import is what is under test

    policy: CompactionPolicy = DropOldestPolicy()

    assert (policy.name, policy.version) == ("drop_oldest", "1.0.0")
