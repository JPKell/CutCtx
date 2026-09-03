"""Properties of ``apply`` over **arbitrary valid plans**, not just the one policy that ships.

``DropOldestPolicy`` emits ``KEEP`` and ``DROP`` and nothing else, so a suite built on it alone
would leave the executor's masking and summarization paths — and the ``MASK`` branch of the token
arithmetic — asserted only by the handful of examples someone thought to write. Those are exactly
the paths row E1's policies will drive, over transcripts nobody has seen. So the plans here are
drawn (:func:`strategies.plans_over`): every action class, groups spanning several
exchanges, stubs of every size, over the same awkward transcripts.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cutctx import (
    SUMMARY_TURN_ID_PREFIX,
    Action,
    CompactionExecutor,
    CompactionPlan,
    PlanTranscriptMismatch,
    Role,
    SummaryMissing,
    Transcript,
)
from strategies import transcripts_budgets_and_plans

EXECUTOR = CompactionExecutor()

Case = tuple[Transcript, object, CompactionPlan]


def _summaries(plan: CompactionPlan) -> dict[str, str]:
    """Fulfil every request the plan carries, the way a caller would."""
    return {
        request.group_id: f"a summary of {len(request.turn_ids)} turns"
        for request in plan.summarization_requests
    }


@given(transcripts_budgets_and_plans())
def test_every_action_class_lands_in_the_view_as_the_plan_said(case: Case) -> None:
    """KEEP passes through, MASK substitutes in place, SUMMARIZE folds, DROP vanishes.

    Would catch: a MASK that lost the turn's role, ``pinned`` or ``tool_call_id``, which would
    move a masked result out of its own tool exchange and let the *next* round separate a pair; a
    MASK that kept the original body; a summary turn emitted once per member rather than once per
    group; a summary placed at the group's last turn rather than its first, which reorders the
    view against the transcript it came from.
    """
    transcript, _, plan = case
    view = EXECUTOR.apply(transcript, plan, _summaries(plan))
    by_id = {turn.turn_id: turn for turn in view.transcript.turns}
    originals = {turn.turn_id: turn for turn in transcript.turns}

    for action in plan.actions:
        if action.action is Action.KEEP:
            assert by_id[action.turn_id] == originals[action.turn_id]
        elif action.action is Action.MASK:
            masked, original = by_id[action.turn_id], originals[action.turn_id]
            assert action.replacement is not None
            assert masked.content == action.replacement.content
            assert masked.token_estimate == action.replacement.token_estimate
            assert (masked.role, masked.pinned, masked.tool_call_id) == (
                original.role,
                original.pinned,
                original.tool_call_id,
            )
        else:
            assert action.turn_id not in by_id

    order = transcript.turn_ids()
    view_ids = view.transcript.turn_ids()
    survivors = [t for t in view_ids if not t.startswith(SUMMARY_TURN_ID_PREFIX)]
    # The surviving originals keep the transcript's relative order: "oldest" still means oldest.
    assert survivors == [t for t in order if t in set(survivors)]

    for request in plan.summarization_requests:
        summary = by_id[request.summary_turn_id]
        assert summary.role is Role.ASSISTANT
        assert summary.pinned is False
        assert summary.token_estimate == request.target_tokens
        # The summary sits where its group's *earliest* member was: every surviving original is on
        # the side of it that it was on in the transcript.
        folds_from = min(order.index(t) for t in request.turn_ids)
        summary_at = view_ids.index(request.summary_turn_id)
        for position, turn_id in enumerate(view_ids):
            if turn_id.startswith(SUMMARY_TURN_ID_PREFIX):
                continue
            assert (position < summary_at) == (order.index(turn_id) < folds_from)


@given(transcripts_budgets_and_plans())
def test_the_views_estimate_is_the_plans_estimate_for_every_action_class(case: Case) -> None:
    """The arithmetic holds when masks and summaries are in play, not only drops.

    A mask contributes its stub's estimate and a summary its ``target_tokens``; both are figures
    the plan chose and neither is recoverable from the text. An executor that re-estimated either
    would produce a report that disagreed with its own plan the first time a stub's text did not
    estimate to the number the policy planned for it.
    """
    transcript, _, plan = case
    view = EXECUTOR.apply(transcript, plan, _summaries(plan))
    assert view.transcript.token_estimate() == plan.tokens_after_estimate


@given(transcripts_budgets_and_plans())
def test_the_report_partitions_the_transcript_exactly_once(case: Case) -> None:
    """Every turn is in exactly one of the report's four id lists, in transcript order.

    The report is the ``context.compacted`` event body, and an operator reads it to know what the
    model was shown. A turn missing from all four lists, or in two of them, makes that reading
    wrong in a way no error would ever surface.
    """
    transcript, _, plan = case
    report = EXECUTOR.apply(transcript, plan, _summaries(plan)).report
    lists = (
        report.kept_turn_ids,
        report.masked_turn_ids,
        report.summarized_turn_ids,
        report.dropped_turn_ids,
    )
    flat = [turn_id for group in lists for turn_id in group]
    assert sorted(flat) == sorted(transcript.turn_ids())
    for group in lists:
        assert list(group) == [t for t in transcript.turn_ids() if t in set(group)]
    assert report.turns_before == len(transcript.turns)
    assert report.turns_after == len(
        EXECUTOR.apply(transcript, plan, _summaries(plan)).transcript.turns
    )
    assert set(report.summary_turn_ids) == {
        request.summary_turn_id for request in plan.summarization_requests
    }
    assert all(t.startswith(SUMMARY_TURN_ID_PREFIX) for t in report.summary_turn_ids)


@given(transcripts_budgets_and_plans())
def test_a_missing_summary_refuses_and_names_the_group(case: Case) -> None:
    """The loud failure of the step callers forget — for **every** one of the plan's groups.

    Every group is omitted in turn rather than one being drawn. Drawing looked equivalent and was
    not: multi-group plans were 2 % of draws, so "check only the first request" — a mutant
    deliberately introduced into the executor — survived the drawn version of this test and dies
    against this one. A property whose interesting case is rare in the generator is a property
    that is not being run.
    """
    transcript, _, plan = case
    summaries = _summaries(plan)
    for omitted in sorted(summaries):
        partial = {group: text for group, text in summaries.items() if group != omitted}

        with pytest.raises(SummaryMissing) as refusal:
            EXECUTOR.apply(transcript, plan, partial)

        assert refusal.value.details["group_id"] == omitted
        assert refusal.value.code == "COMPACTION_SUMMARY_MISSING"


@given(transcripts_budgets_and_plans())
def test_a_surplus_summary_is_ignored_rather_than_refused(case: Case) -> None:
    """A caller may hold one summaries mapping across several plans; that is not a mistake.

    A deliberate non-refusal, and the only one in the package — recorded as a property so that a
    later change to refuse it is a decision somebody makes rather than one that leaks in.
    """
    transcript, _, plan = case
    summaries = _summaries(plan) | {"a-group-this-plan-does-not-have": "text"}
    with_surplus = EXECUTOR.apply(transcript, plan, summaries)
    assert with_surplus.transcript == EXECUTOR.apply(transcript, plan, _summaries(plan)).transcript


@given(transcripts_budgets_and_plans())
def test_applying_a_plan_twice_produces_identical_results(case: Case) -> None:
    """``apply`` is a pure function of its three arguments (contract 4, downstream half)."""
    transcript, _, plan = case
    summaries = _summaries(plan)
    first = EXECUTOR.apply(transcript, plan, summaries)
    second = EXECUTOR.apply(transcript, plan, summaries)
    assert first.transcript == second.transcript
    assert first.report == second.report


@given(transcripts_budgets_and_plans(), st.booleans())
def test_a_plan_never_applies_to_a_transcript_it_was_not_built_for(
    case: Case, by_reordering: bool
) -> None:
    """Different ids, or the same ids in a different order — both refuse.

    Reordering matters as much as substitution and is the easier one to get wrong: the ids all
    match, so a set-based check passes, and the plan then acts on turns by position that it chose
    by identity. "Oldest" and the protected tail are both positional.
    """
    transcript, _, plan = case
    if len(transcript.turns) < 2:
        return
    other = (
        Transcript(turns=(transcript.turns[-1], *transcript.turns[:-1]))
        if by_reordering
        else Transcript(turns=transcript.turns[:-1])
    )
    if other.turn_ids() == transcript.turn_ids():
        return

    with pytest.raises(PlanTranscriptMismatch) as refusal:
        EXECUTOR.apply(other, plan, _summaries(plan))

    assert refusal.value.details["reordered"] is by_reordering
