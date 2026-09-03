"""The invariants of spec §11, stated over arbitrary transcripts rather than chosen ones.

The choice of properties here is the deliverable, not their count and not their pass rate. Each
one below is a claim that would be **false** if the invariant module were subtly wrong, and each
is checked against transcripts the implementation was not written with in mind — multi-call
exchanges, results displaced far from their calls, entirely pinned transcripts, empty ones,
zero-token turns and million-token turns, and budgets aimed at the transcript's own boundaries
(see :mod:`strategies`).

The properties considered and **rejected** are recorded in ``C1_HANDOFF.md``; several of them look
obviously true and are not, and a suite that asserted them would have locked in a wrong intuition.
"""

from __future__ import annotations

import pytest
from baseaicore import ValidationError
from hypothesis import given
from hypothesis import strategies as st

from conftest import rebuilt
from cutctx import (
    Action,
    BudgetUnsatisfiable,
    CompactionBudget,
    CompactionExecutor,
    DropOldestPolicy,
    Transcript,
    TranscriptTurn,
)
from cutctx._invariants import exchanges, untouchable_turn_ids
from oracles import exchange_sets, untouchable_ids, untouchable_total
from strategies import budgets, transcripts, transcripts_and_budgets

POLICY = DropOldestPolicy()
EXECUTOR = CompactionExecutor()

_REMOVALS = {Action.DROP, Action.SUMMARIZE}


def _plan_or_refusal(transcript: Transcript, budget: CompactionBudget) -> object:
    """Return the plan, or the refusal, so a property can assert over both outcomes."""
    try:
        return POLICY.decide(transcript, budget)
    except BudgetUnsatisfiable as refusal:
        return refusal


# --------------------------------------------------------------------------------------------
# Contract 2 — the untouchable set
# --------------------------------------------------------------------------------------------


@given(transcripts_and_budgets())
def test_no_plan_acts_on_the_untouchable_set(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """Contract 2, over every transcript shape.

    Would catch: a policy that treats "pinned" as advisory; an off-by-one in the protected tail;
    a policy that protects the tail by *position* after having already reordered turns; a SYSTEM
    turn that is only protected when it is first.
    """
    transcript, budget = case
    outcome = _plan_or_refusal(transcript, budget)
    if isinstance(outcome, BudgetUnsatisfiable):
        return
    untouchable = untouchable_ids(transcript, budget)
    acted_on = [
        action.turn_id
        for action in outcome.actions  # type: ignore[attr-defined]  # narrowed by the guard above
        if action.turn_id in untouchable and action.action is not Action.KEEP
    ]
    assert acted_on == []


@given(transcripts_and_budgets())
def test_the_budget_refusal_is_exactly_the_untouchable_set_exceeding_the_budget(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """The refusal fires on one condition and no other, and it names both numbers.

    Would catch: a refusal computed over the *closure* rather than the strict untouchable set,
    which would refuse budgets Phase 2's masking policies can meet; a refusal that fires on
    equality, which would reject the budget that exactly fits its own floor.
    """
    transcript, budget = case
    floor = untouchable_total(transcript, budget.protected_recent_turns)
    outcome = _plan_or_refusal(transcript, budget)
    if isinstance(outcome, BudgetUnsatisfiable):
        assert floor > budget.max_tokens
        assert outcome.details["untouchable_tokens"] == floor
        assert outcome.details["max_tokens"] == budget.max_tokens
    else:
        assert floor <= budget.max_tokens


# --------------------------------------------------------------------------------------------
# Coverage — a plan is a total function over its transcript
# --------------------------------------------------------------------------------------------


@given(transcripts_and_budgets())
def test_every_turn_id_appears_exactly_once_in_transcript_order(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """A turn the plan forgot is the silent bug this property exists to catch.

    Would catch: a policy that builds actions from a *set* of ids and loses one to a duplicate; a
    policy that emits actions for dropped turns only; any construction whose action order came
    from a dict rather than from the transcript — which is the same trap Phase 2's group assembly
    is warned about.
    """
    transcript, budget = case
    outcome = _plan_or_refusal(transcript, budget)
    if isinstance(outcome, BudgetUnsatisfiable):
        return
    planned = tuple(a.turn_id for a in outcome.actions)  # type: ignore[attr-defined]
    assert planned == transcript.turn_ids()


# --------------------------------------------------------------------------------------------
# Contract 3 — a tool exchange travels together
# --------------------------------------------------------------------------------------------


@given(transcripts_and_budgets())
def test_a_tool_exchange_is_never_split_in_either_direction(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """No orphaned call, no orphaned result — including across a multi-call turn.

    Both directions matter and they fail differently: dropping the call and keeping the results
    leaves results referring to nothing, and dropping one of three results leaves a call whose
    answers are incomplete. Pairing is not one-to-one, so a check written as "for each call, find
    its result" would pass on the first and miss the second.

    Would catch: pair detection that assumes contiguity (the generator displaces results); pair
    detection keyed on adjacency rather than on the correlation id; a policy that drops turn by
    turn with a budget check between, stopping halfway through an exchange.
    """
    transcript, budget = case
    outcome = _plan_or_refusal(transcript, budget)
    if isinstance(outcome, BudgetUnsatisfiable):
        return
    by_turn = {a.turn_id: a for a in outcome.actions}  # type: ignore[attr-defined]
    for exchange in exchange_sets(transcript):
        removed = {t for t in exchange if by_turn[t].action in _REMOVALS}
        assert removed in (set(), exchange), (
            f"exchange {sorted(exchange)} was split: {sorted(removed)} removed"
        )
        signatures = {(by_turn[t].action, by_turn[t].summary_group) for t in exchange}
        assert len(signatures) == 1


@given(transcripts_and_budgets())
def test_protection_propagates_through_an_exchange(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """An untouchable member makes its whole exchange undroppable.

    The consequence that is not obvious: a tool result inside the protected tail whose call is
    forty turns back keeps that call alive, so "everything old is droppable" is false. This is
    also why "every dropped turn is older than every kept turn" is *not* a property of the output.

    Would catch: a policy that computes the untouchable set correctly and then drops from the
    remainder without closing over exchanges — which passes the contract-2 property and fails
    contract 3 only for the displaced shapes.
    """
    transcript, budget = case
    outcome = _plan_or_refusal(transcript, budget)
    if isinstance(outcome, BudgetUnsatisfiable):
        return
    untouchable = untouchable_ids(transcript, budget)
    by_turn = {a.turn_id: a for a in outcome.actions}  # type: ignore[attr-defined]
    for exchange in exchange_sets(transcript):
        if untouchable & exchange:
            assert all(by_turn[t].action is Action.KEEP for t in exchange)


# --------------------------------------------------------------------------------------------
# The executor — the view is what the plan said
# --------------------------------------------------------------------------------------------


@given(transcripts_and_budgets())
def test_apply_leaves_the_input_transcript_untouched(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """Contract 1: a plan is a view, never a deletion.

    Asserted on identity as well as equality, because a value that compared equal after being
    rebuilt would still have broken the caller's own references to its turns.
    """
    transcript, budget = case
    outcome = _plan_or_refusal(transcript, budget)
    if isinstance(outcome, BudgetUnsatisfiable):
        return
    before_ids = transcript.turn_ids()
    before_turns = [id(t) for t in transcript.turns]
    before_value = Transcript(turns=transcript.turns)

    EXECUTOR.apply(transcript, outcome)  # type: ignore[arg-type]

    assert transcript.turn_ids() == before_ids
    assert [id(t) for t in transcript.turns] == before_turns
    assert transcript == before_value


@given(transcripts_and_budgets())
def test_the_view_contains_exactly_what_the_plan_retained(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """No turn the plan dropped survives, and no turn it kept goes missing.

    Would catch: an executor that skipped an action class; an executor whose "drop" branch was a
    fall-through, so an unrecognised action silently kept a turn the plan removed.
    """
    transcript, budget = case
    outcome = _plan_or_refusal(transcript, budget)
    if isinstance(outcome, BudgetUnsatisfiable):
        return
    view = EXECUTOR.apply(transcript, outcome)  # type: ignore[arg-type]
    retained = {
        a.turn_id
        for a in outcome.actions  # type: ignore[attr-defined]
        if a.action in (Action.KEEP, Action.MASK)
    }
    assert set(view.transcript.turn_ids()) == retained
    # And in the transcript's own order, since "oldest" stays meaningful in the view.
    assert view.transcript.turn_ids() == tuple(t for t in transcript.turn_ids() if t in retained)


@given(transcripts_and_budgets())
def test_tokens_after_estimate_agrees_with_an_independent_sum_of_the_view(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """The plan's arithmetic and the view the executor built are two statements of one fact.

    The sum is independent of the plan's own computation: it comes from the turns the executor
    actually produced. "Token estimates drifting between plan and report" is the failure mode the
    development plan names for this phase, and this is the property that would see it.
    """
    transcript, budget = case
    outcome = _plan_or_refusal(transcript, budget)
    if isinstance(outcome, BudgetUnsatisfiable):
        return
    view = EXECUTOR.apply(transcript, outcome)  # type: ignore[arg-type]
    assert view.transcript.token_estimate() == outcome.tokens_after_estimate  # type: ignore[attr-defined]
    assert view.report.tokens_after_estimate == outcome.tokens_after_estimate  # type: ignore[attr-defined]
    assert view.report.tokens_before == transcript.token_estimate()


# --------------------------------------------------------------------------------------------
# Contract 4 — determinism
# --------------------------------------------------------------------------------------------


@given(transcripts_and_budgets())
def test_the_plan_is_byte_identical_on_re_derivation(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """Same transcript, same budget, same configuration ⇒ the same bytes.

    A plan appears in an audit record, and a record nobody can reproduce is not evidence.
    """
    transcript, budget = case
    first = _plan_or_refusal(transcript, budget)
    second = _plan_or_refusal(transcript, budget)
    if isinstance(first, BudgetUnsatisfiable):
        assert isinstance(second, BudgetUnsatisfiable)
        assert first.details == second.details
        return
    assert first.canonical_json() == second.canonical_json()  # type: ignore[attr-defined]
    assert first.plan_hash() == second.plan_hash()  # type: ignore[attr-defined]


@given(transcripts_and_budgets())
def test_an_equivalent_construction_path_produces_the_same_bytes(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """A transcript rebuilt by another route plans identically.

    The turns are reconstructed field by field and their ``metadata`` dicts are rebuilt in
    reversed insertion order. Nothing about how a caller assembled its transcript may reach the
    plan's bytes — dict insertion order least of all, since that is exactly the trap Phase 2's
    contiguous-group assembly is warned about.
    """
    transcript, budget = case
    rebuilt_transcript = Transcript(
        turns=tuple(
            TranscriptTurn(
                turn_id=t.turn_id,
                role=t.role,
                content=t.content,
                token_estimate=t.token_estimate,
                tool_call_id=t.tool_call_id,
                pinned=t.pinned,
                metadata=dict(reversed(list(t.metadata.items()))),
            )
            for t in transcript.turns
        )
    )
    original = _plan_or_refusal(transcript, budget)
    rebuilt_plan = _plan_or_refusal(rebuilt_transcript, budget)
    if isinstance(original, BudgetUnsatisfiable):
        assert isinstance(rebuilt_plan, BudgetUnsatisfiable)
        return
    assert original.canonical_json() == rebuilt_plan.canonical_json()  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------------
# The budget outcome — trichotomous and honest
# --------------------------------------------------------------------------------------------


@given(transcripts_and_budgets())
def test_the_budget_outcome_is_trichotomous_and_honest(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """Fits, or ``budget_unmet``, or ``BudgetUnsatisfiable``. There is no fourth case.

    In particular there is no plan that claims to fit while being over: ``budget_unmet`` is
    derived at construction from the plan's own arithmetic, so the flag and the figures cannot
    disagree. This is ADR-0023's refusal of silent truncation, in transcript form.
    """
    transcript, budget = case
    outcome = _plan_or_refusal(transcript, budget)
    if isinstance(outcome, BudgetUnsatisfiable):
        return
    over = outcome.tokens_after_estimate > budget.max_tokens  # type: ignore[attr-defined]
    assert outcome.budget_unmet is over  # type: ignore[attr-defined]


@given(transcripts_and_budgets())
def test_a_transcript_that_already_fits_is_left_alone(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """The last resort is a *last* resort: nothing is dropped from a transcript that already fits.

    Would catch: a policy whose loop drops one exchange before testing the budget, which is the
    natural way to write it and is wrong.
    """
    transcript, budget = case
    if transcript.token_estimate() > budget.max_tokens:
        return
    plan = POLICY.decide(transcript, budget)
    assert all(a.action is Action.KEEP for a in plan.actions)
    assert plan.tokens_after_estimate == plan.tokens_before


# --------------------------------------------------------------------------------------------
# The guard — the invariant module is the only path to a plan, and it bites
# --------------------------------------------------------------------------------------------


@given(transcripts_and_budgets(), st.integers(min_value=0, max_value=7))
def test_validation_refuses_every_mutilation_of_a_valid_plan(
    case: tuple[Transcript, CompactionBudget], mutation: int
) -> None:
    """Take a valid plan, break it one way, and it cannot be constructed.

    The mutation-testing property, and the strongest evidence that the guard bites: it does not
    check that a *correct* plan passes — every other property does that — it checks that the
    validator is not a decoration. Each mutation below is a plan some future policy could
    plausibly write:

    0. a forgotten turn, 1. a turn planned twice, 2. actions in the wrong order — the coverage rule;
    3. an untouchable turn dropped — contract 2;
    4. half an exchange dropped — contract 3;
    5. a ``tokens_before`` that is not the transcript's sum, 6. a ``tokens_after_estimate`` the
       actions do not produce — the arithmetic rule;
    7. a ``budget_unmet`` flag that disagrees with the figures — the honesty rule.

    A mutation whose precondition this example does not meet is skipped rather than weakened; the
    generator produces enough shapes that every branch is reached many times over a run.
    """
    transcript, budget = case
    try:
        plan = POLICY.decide(transcript, budget)
    except BudgetUnsatisfiable:
        return

    overrides = _mutate(plan, transcript, budget, mutation)
    if overrides is None:
        return
    with pytest.raises(ValidationError):
        rebuilt(plan, transcript, budget, **overrides)


def _mutate(
    plan: object, transcript: Transcript, budget: CompactionBudget, mutation: int
) -> dict[str, object] | None:
    """Return the field overrides for one mutation, or ``None`` if this plan cannot host it."""
    actions = plan.actions  # type: ignore[attr-defined]
    untouchable = untouchable_turn_ids(transcript, budget)
    if mutation == 0:
        return {"actions": actions[:-1]} if actions else None
    if mutation == 1:
        return {"actions": (*actions, actions[0])} if actions else None
    if mutation == 2:
        return {"actions": tuple(reversed(actions))} if len(set(actions)) > 1 else None
    if mutation == 3:
        target = next((a for a in actions if a.turn_id in untouchable), None)
        return None if target is None else {"actions": _with_action(actions, target, Action.DROP)}
    if mutation == 4:
        for exchange in exchanges(transcript):
            if len(exchange) > 1:
                target = next(a for a in actions if a.turn_id == exchange[0])
                flipped = Action.KEEP if target.action is Action.DROP else Action.DROP
                return {"actions": _with_action(actions, target, flipped)}
        return None
    if mutation == 5:
        return {"tokens_before": plan.tokens_before + 1}  # type: ignore[attr-defined]
    if mutation == 6:
        return {"tokens_after_estimate": plan.tokens_after_estimate + 1}  # type: ignore[attr-defined]
    return {"budget_unmet": not plan.budget_unmet}  # type: ignore[attr-defined]


def _with_action(actions: tuple[object, ...], target: object, action: Action) -> tuple[object, ...]:
    """Return ``actions`` with ``target``'s action replaced."""
    from cutctx import TurnAction  # noqa: PLC0415 — kept local to the mutation helper

    return tuple(
        TurnAction(turn_id=a.turn_id, action=action) if a is target else a  # type: ignore[attr-defined]
        for a in actions
    )


@given(transcripts())
def test_a_budget_is_satisfiable_exactly_when_its_floor_fits(transcript: Transcript) -> None:
    """The refusal boundary is an equality, and it is checked from both sides.

    Would catch: a ``>=`` where the spec says "smaller than", which would refuse the budget that
    exactly fits its own untouchable set — a budget for which the identity plan is correct.
    """
    protected = 2
    floor = untouchable_total(transcript, protected)
    exact = CompactionBudget(max_tokens=floor, protected_recent_turns=protected)
    assert POLICY.decide(transcript, exact) is not None
    if floor > 0:
        one_short = CompactionBudget(max_tokens=floor - 1, protected_recent_turns=protected)
        with pytest.raises(BudgetUnsatisfiable):
            POLICY.decide(transcript, one_short)


@given(transcripts(), st.data())
def test_the_report_only_repeats_figures_the_plan_earned(
    transcript: Transcript, data: st.DataObject
) -> None:
    """Every figure on the report is the plan's, copied — never a second computation.

    Would catch the drift the development plan names: a report that re-derived "after" from the
    view it was handed would agree today and disagree the first time a policy emits a MASK whose
    stub estimate is not what the stub's text would estimate to.
    """
    budget = data.draw(budgets(transcript))
    outcome = _plan_or_refusal(transcript, budget)
    if isinstance(outcome, BudgetUnsatisfiable):
        return
    report = EXECUTOR.apply(transcript, outcome).report  # type: ignore[arg-type]
    assert report.tokens_before == outcome.tokens_before  # type: ignore[attr-defined]
    assert report.tokens_after_estimate == outcome.tokens_after_estimate  # type: ignore[attr-defined]
    assert report.budget_unmet == outcome.budget_unmet  # type: ignore[attr-defined]
    assert report.estimator_ratio == outcome.estimator_ratio  # type: ignore[attr-defined]
    assert report.plan_hash == outcome.plan_hash()  # type: ignore[attr-defined]
    assert report.turns_before == len(transcript.turns)
