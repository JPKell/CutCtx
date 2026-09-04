"""Properties over the shipped policy set — the invariants, for all four, over generated shapes.

Phase 1's property suite is written around ``DropOldestPolicy``. This one runs the contract
properties against **every** shipped policy, because the invariants are properties of a plan and
not of the policy that wrote it — and adds the properties that are specific to what Phase 2's
policies do.

Every oracle here comes from ``tests/oracles.py``, restated from the spec in a different shape from
the implementation. `C1_HANDOFF.md` §9.2 records that as the trap that cost that session the most
time: an oracle imported from ``_invariants`` makes a property suite look thorough and prove
nothing, because the oracle moves with the bug. :func:`~oracles.maskable_ids` and
:func:`~oracles.exchange_rule_holds` are written that way deliberately — the first counts backwards
through the transcript where the policy slices a list, the second keys on the correlation id with
no notion of position at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cutctx import (
    Action,
    BudgetUnsatisfiable,
    CompactionExecutor,
    CompactionPlan,
    DropOldestPolicy,
    ObservationMaskingPolicy,
    PolicyChain,
    Role,
    SummarizingPolicy,
    default_chain,
)
from oracles import exchange_rule_holds, exchange_sets, maskable_ids, untouchable_ids
from strategies import transcripts_and_budgets

if TYPE_CHECKING:
    from cutctx import CompactionBudget, CompactionPolicy, Transcript

PROMPT = "compaction.summarize.v1"

KEEP_RECENT = 2

SHIPPED: dict[str, CompactionPolicy] = {
    "drop_oldest": DropOldestPolicy(),
    "masking": ObservationMaskingPolicy(KEEP_RECENT),
    "summarizing": SummarizingPolicy(PROMPT),
    "chain": default_chain(prompt_id=PROMPT, keep_recent_results=KEEP_RECENT),
}


def _plan(
    policy: CompactionPolicy, transcript: Transcript, budget: CompactionBudget
) -> CompactionPlan | None:
    """Return the plan, or ``None`` when the budget was refused before one existed."""
    try:
        return policy.decide(transcript, budget)
    except BudgetUnsatisfiable:
        return None


def _verdicts(plan: CompactionPlan) -> dict[str, str]:
    """Render a plan's actions in the vocabulary :func:`~oracles.exchange_rule_holds` reads."""
    return {
        action.turn_id: (
            f"summarize:{action.summary_group}"
            if action.action is Action.SUMMARIZE
            else action.action.value
        )
        for action in plan.actions
    }


# ------------------------------------------------------------------------------------------------
# The contracts, for every shipped policy
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("label", sorted(SHIPPED))
@given(transcripts_and_budgets())
def test_no_shipped_policy_acts_on_the_untouchable_set(
    label: str, case: tuple[Transcript, CompactionBudget]
) -> None:
    """Contract 2, for all four.

    Would catch: a masking policy that treats the protected tail as advisory because masking
    "removes nothing"; a summarizing span that runs into a pinned turn; a chain whose projection
    shortened the transcript and moved the tail with it.
    """
    transcript, budget = case
    plan = _plan(SHIPPED[label], transcript, budget)
    if plan is None:
        return
    untouchable = untouchable_ids(transcript, budget)
    assert [
        action.turn_id
        for action in plan.actions
        if action.turn_id in untouchable and action.action is not Action.KEEP
    ] == []


@pytest.mark.parametrize("label", sorted(SHIPPED))
@given(transcripts_and_budgets())
def test_no_shipped_policy_separates_a_tool_exchange(
    label: str, case: tuple[Transcript, CompactionBudget]
) -> None:
    """Contract 3 as `C1_HANDOFF.md` §4 reads it, for all four.

    Would catch: a summarizing span that clipped an exchange whose members are not adjacent; a
    chain that let a later policy drop a call whose result an earlier one masked.
    """
    transcript, budget = case
    plan = _plan(SHIPPED[label], transcript, budget)
    if plan is None:
        return
    assert exchange_rule_holds(_verdicts(plan), transcript)


@pytest.mark.parametrize("label", sorted(SHIPPED))
@given(transcripts_and_budgets())
def test_every_shipped_policy_is_byte_identical_on_re_derivation(
    label: str, case: tuple[Transcript, CompactionBudget]
) -> None:
    """Contract 4, for all four — including a policy that carried state between calls."""
    transcript, budget = case
    policy = SHIPPED[label]
    first = _plan(policy, transcript, budget)
    second = _plan(policy, transcript, budget)
    if first is None or second is None:
        assert first is second
        return
    assert first.canonical_json() == second.canonical_json()


@pytest.mark.parametrize("label", sorted(SHIPPED))
@given(transcripts_and_budgets())
def test_no_shipped_policy_makes_a_transcript_larger(
    label: str, case: tuple[Transcript, CompactionBudget]
) -> None:
    """A compaction that grew the transcript would be a compaction in name only.

    Would catch: masking a two-token result with a thirty-token stub; a summarization target above
    the span it replaces.
    """
    transcript, budget = case
    plan = _plan(SHIPPED[label], transcript, budget)
    if plan is None:
        return
    assert plan.tokens_after_estimate <= plan.tokens_before


@pytest.mark.parametrize("label", sorted(SHIPPED))
@given(transcripts_and_budgets())
def test_budget_unmet_is_derived_for_every_shipped_policy(
    label: str, case: tuple[Transcript, CompactionBudget]
) -> None:
    """`C1_HANDOFF.md` §3.2: derived, never declared. A plan never claims to fit while over."""
    transcript, budget = case
    plan = _plan(SHIPPED[label], transcript, budget)
    if plan is None:
        return
    assert plan.budget_unmet is (plan.tokens_after_estimate > budget.max_tokens)


@pytest.mark.parametrize("label", sorted(SHIPPED))
@given(transcripts_and_budgets())
def test_a_transcript_that_already_fits_is_left_alone_by_every_shipped_policy(
    label: str, case: tuple[Transcript, CompactionBudget]
) -> None:
    """Compaction is not something to do because a policy was asked to."""
    transcript, budget = case
    if transcript.token_estimate() > budget.max_tokens:
        return
    plan = _plan(SHIPPED[label], transcript, budget)
    if plan is None:
        return
    assert all(action.action is Action.KEEP for action in plan.actions)
    assert plan.summarization_requests == ()


# ------------------------------------------------------------------------------------------------
# ObservationMaskingPolicy
# ------------------------------------------------------------------------------------------------


@given(transcripts_and_budgets())
def test_masking_only_ever_masks_what_the_spec_says_is_maskable(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """Against a maskable set counted backwards through the transcript, not sliced from a list.

    Would catch: masking an assistant turn; masking within the recent floor; a ``keep=0``
    off-by-one where ``results[-0:]`` silently protects everything.
    """
    transcript, budget = case
    plan = _plan(SHIPPED["masking"], transcript, budget)
    if plan is None:
        return
    masked = {a.turn_id for a in plan.actions if a.action is Action.MASK}
    assert masked <= maskable_ids(transcript, budget, KEEP_RECENT)


@given(transcripts_and_budgets())
def test_a_masked_turn_stays_in_the_view_at_its_own_position(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """Masking removes nothing — which is what makes it legal under contract 3 at all."""
    transcript, budget = case
    plan = _plan(SHIPPED["masking"], transcript, budget)
    if plan is None:
        return
    view = CompactionExecutor().apply(transcript, plan).transcript
    assert [t.turn_id for t in view.turns] == [t.turn_id for t in transcript.turns]


@given(transcripts_and_budgets())
def test_masking_leaves_every_other_turn_byte_identical(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """The value of the policy is that the reasoning survives; a property, not a hope."""
    transcript, budget = case
    plan = _plan(SHIPPED["masking"], transcript, budget)
    if plan is None:
        return
    masked = {a.turn_id for a in plan.actions if a.action is Action.MASK}
    view = {t.turn_id: t for t in CompactionExecutor().apply(transcript, plan).transcript.turns}
    for original in transcript.turns:
        if original.turn_id not in masked:
            assert view[original.turn_id] == original


@given(transcripts_and_budgets())
def test_a_stub_holds_no_run_of_the_original(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """Spec §14: a hash, never a secret-bearing excerpt. Asserted over generated content.

    Would catch: a "first N characters" preview added later for readability, which is a leak with
    an extra step.
    """
    transcript, budget = case
    plan = _plan(SHIPPED["masking"], transcript, budget)
    if plan is None:
        return
    bodies = {t.turn_id: t.content for t in transcript.turns}
    for action in plan.actions:
        if action.replacement is None:
            continue
        original = bodies[action.turn_id]
        window = 8
        runs = {original[at : at + window] for at in range(max(0, len(original) - window + 1))}
        assert not any(run and run in action.replacement.content for run in runs)


# ------------------------------------------------------------------------------------------------
# SummarizingPolicy
# ------------------------------------------------------------------------------------------------


@given(transcripts_and_budgets())
def test_a_summary_group_is_a_contiguous_run_of_transcript_positions(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """ "the oldest **contiguous** unpinned span".

    Would catch the development plan's named trap directly: a span assembled by iterating the
    ``frozenset`` ``removable_turn_ids`` returns would be ordered by hash, and would pass a test
    that only checked membership.
    """
    transcript, budget = case
    plan = _plan(SHIPPED["summarizing"], transcript, budget)
    if plan is None or not plan.summarization_requests:
        return
    order = {turn.turn_id: index for index, turn in enumerate(transcript.turns)}
    for request in plan.summarization_requests:
        positions = [order[turn_id] for turn_id in request.turn_ids]
        assert positions == sorted(positions)
        assert positions == list(range(positions[0], positions[0] + len(positions)))


@given(transcripts_and_budgets())
def test_a_summary_group_contains_whole_exchanges_or_none_of_them(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """Against the correlation-id oracle, which has no notion of adjacency to be fooled by."""
    transcript, budget = case
    plan = _plan(SHIPPED["summarizing"], transcript, budget)
    if plan is None or not plan.summarization_requests:
        return
    for request in plan.summarization_requests:
        folded = set(request.turn_ids)
        for exchange in exchange_sets(transcript):
            overlap = exchange & folded
            assert not overlap or overlap == exchange


@given(transcripts_and_budgets())
def test_a_summary_never_costs_more_than_the_span_it_replaces(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    transcript, budget = case
    plan = _plan(SHIPPED["summarizing"], transcript, budget)
    if plan is None:
        return
    estimates = {turn.turn_id: turn.token_estimate for turn in transcript.turns}
    for request in plan.summarization_requests:
        assert request.target_tokens < sum(estimates[turn_id] for turn_id in request.turn_ids)


@given(transcripts_and_budgets())
def test_a_summarizing_plan_carries_no_prompt_text(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """ADR-0012: a ``prompt_id`` names a record. Nothing in this package holds prompt text."""
    transcript, budget = case
    plan = _plan(SHIPPED["summarizing"], transcript, budget)
    if plan is None:
        return
    for request in plan.summarization_requests:
        assert request.prompt_id == PROMPT


# ------------------------------------------------------------------------------------------------
# PolicyChain
# ------------------------------------------------------------------------------------------------


@given(transcripts_and_budgets())
def test_the_chains_estimate_is_the_estimate_of_the_view_it_produces(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """The projection is a claim about ``apply``; this is the claim checked against the thing.

    Would catch: a projection that put the summary stand-in at the wrong position, gave it the
    wrong estimate, or forgot that a masked turn keeps its place.
    """
    transcript, budget = case
    plan = _plan(SHIPPED["chain"], transcript, budget)
    if plan is None:
        return
    summaries = {
        request.group_id: "x" * request.target_tokens for request in plan.summarization_requests
    }
    view = CompactionExecutor().apply(transcript, plan, summaries).transcript
    assert view.token_estimate() == plan.tokens_after_estimate


@given(transcripts_and_budgets())
def test_a_chain_of_one_agrees_with_that_policy_alone(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """Composition of one is the identity, or the projection has changed something it should not.

    Only the policy *name* may differ: the chain names its members, which is the point of it.
    """
    transcript, budget = case
    alone = _plan(DropOldestPolicy(), transcript, budget)
    chained = _plan(PolicyChain((DropOldestPolicy(),)), transcript, budget)
    if alone is None or chained is None:
        assert alone is chained
        return
    assert [(a.turn_id, a.action) for a in alone.actions] == [
        (a.turn_id, a.action) for a in chained.actions
    ]
    assert alone.tokens_after_estimate == chained.tokens_after_estimate
    assert alone.budget_unmet == chained.budget_unmet


@given(transcripts_and_budgets(), st.integers(min_value=0, max_value=2))
def test_the_chain_never_ends_worse_than_its_first_policy_alone(
    case: tuple[Transcript, CompactionBudget], index: int
) -> None:
    """Running more policies cannot cost more tokens than running one of them.

    Would catch a projection that lost a reduction — a masked turn whose stub was forgotten, a
    summary group whose target was double-counted — which would show as the chain undoing work its
    own first step had done.
    """
    transcript, budget = case
    members = default_chain(prompt_id=PROMPT, keep_recent_results=KEEP_RECENT).policies
    first = _plan(members[index], transcript, budget)
    chained = _plan(SHIPPED["chain"], transcript, budget)
    if first is None or chained is None:
        return
    if index == 0:
        assert chained.tokens_after_estimate <= first.tokens_after_estimate


@given(transcripts_and_budgets())
def test_the_chain_retains_every_turn_the_view_needs_and_no_more(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """The view holds exactly the retained turns, plus one turn per summary group, in order."""
    transcript, budget = case
    plan = _plan(SHIPPED["chain"], transcript, budget)
    if plan is None:
        return
    summaries = {request.group_id: "s" for request in plan.summarization_requests}
    view = CompactionExecutor().apply(transcript, plan, summaries).transcript
    retained = [a.turn_id for a in plan.actions if a.action in (Action.KEEP, Action.MASK)]
    summary_ids = {r.summary_turn_id for r in plan.summarization_requests}
    assert [t.turn_id for t in view.turns if t.turn_id not in summary_ids] == retained
    assert {t.turn_id for t in view.turns} & summary_ids == summary_ids


@given(transcripts_and_budgets())
def test_the_chain_never_plans_a_group_no_request_covers(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """Escalation removes requests as well as actions; a stale one would be unfulfillable."""
    transcript, budget = case
    plan = _plan(SHIPPED["chain"], transcript, budget)
    if plan is None:
        return
    named = {a.summary_group for a in plan.actions if a.summary_group is not None}
    assert named == {request.group_id for request in plan.summarization_requests}


@given(transcripts_and_budgets())
def test_no_shipped_policy_masks_a_turn_that_is_not_a_tool_result(
    case: tuple[Transcript, CompactionBudget],
) -> None:
    """Including through a chain, where a projection could have changed a turn's role."""
    transcript, budget = case
    roles = {turn.turn_id: turn.role for turn in transcript.turns}
    for policy in SHIPPED.values():
        plan = _plan(policy, transcript, budget)
        if plan is None:
            continue
        for action in plan.actions:
            if action.action is Action.MASK:
                assert roles[action.turn_id] is Role.TOOL
