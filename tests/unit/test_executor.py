"""Applying a plan: the refusals, the substitutions, and the immutability of the input."""

from __future__ import annotations

import pytest
from baseaicore import canonical_json

from conftest import transcript, turn
from cutctx import (
    Action,
    CompactionBudget,
    CompactionExecutor,
    PlanTranscriptMismatch,
    Role,
    SummarizationRequest,
    SummaryMissing,
    Transcript,
    TurnAction,
    TurnReplacement,
)
from cutctx._invariants import build_plan

WIDE = CompactionBudget(max_tokens=100_000, protected_recent_turns=0)
EXECUTOR = CompactionExecutor()


def planned(source: Transcript, *actions: TurnAction, **kwargs: object) -> object:
    """Build a plan over ``source`` from explicit actions."""
    return build_plan(
        transcript=source,
        budget=WIDE,
        actions=actions,
        policy_name="test",
        policy_version="1.0.0",
        **kwargs,  # type: ignore[arg-type]  # the tests pass only declared keywords
    )


class TestMismatch:
    def test_a_plan_for_other_turns_refuses_and_names_both_differences(self) -> None:
        source = transcript(turn("a"), turn("b"))
        plan = planned(source, TurnAction("a", Action.KEEP), TurnAction("b", Action.KEEP))

        with pytest.raises(PlanTranscriptMismatch) as refusal:
            EXECUTOR.apply(transcript(turn("a"), turn("z")), plan)  # type: ignore[arg-type]

        assert refusal.value.code == "COMPACTION_PLAN_MISMATCH"
        assert refusal.value.details["unknown_turn_ids"] == ["b"]
        assert refusal.value.details["unplanned_turn_ids"] == ["z"]
        assert refusal.value.details["reordered"] is False

    def test_the_same_ids_in_another_order_refuse_and_say_so(self) -> None:
        source = transcript(turn("a"), turn("b"))
        plan = planned(source, TurnAction("a", Action.KEEP), TurnAction("b", Action.KEEP))

        with pytest.raises(PlanTranscriptMismatch) as refusal:
            EXECUTOR.apply(transcript(turn("b"), turn("a")), plan)  # type: ignore[arg-type]

        assert refusal.value.details["reordered"] is True
        assert "order differs" in str(refusal.value)


class TestSummaries:
    def test_a_missing_summary_refuses_with_everything_needed_to_fulfil_it(self) -> None:
        source = transcript(turn("a", tokens=50), turn("b", tokens=50))
        plan = planned(
            source,
            TurnAction("a", Action.SUMMARIZE, summary_group="g1"),
            TurnAction("b", Action.SUMMARIZE, summary_group="g1"),
            summarization_requests=(
                SummarizationRequest("g1", ("a", "b"), 12, "general.summarize"),
            ),
        )

        with pytest.raises(SummaryMissing) as refusal:
            EXECUTOR.apply(source, plan)  # type: ignore[arg-type]

        assert refusal.value.details == {
            "group_id": "g1",
            "turn_ids": ["a", "b"],
            "target_tokens": 12,
            "prompt_id": "general.summarize",
        }

    def test_a_fulfilled_group_becomes_one_labelled_assistant_turn(self) -> None:
        source = transcript(turn("a", tokens=50), turn("b", tokens=50), turn("c", tokens=5))
        plan = planned(
            source,
            TurnAction("a", Action.SUMMARIZE, summary_group="g1"),
            TurnAction("b", Action.SUMMARIZE, summary_group="g1"),
            TurnAction("c", Action.KEEP),
            summarization_requests=(
                SummarizationRequest("g1", ("a", "b"), 12, "general.summarize"),
            ),
        )

        view = EXECUTOR.apply(source, plan, {"g1": "Earlier, the user asked about the build."})  # type: ignore[arg-type]

        summary = view.transcript.turns[0]
        assert summary.turn_id == "summary:g1"
        assert summary.role is Role.ASSISTANT
        assert summary.pinned is False
        assert summary.content == "Earlier, the user asked about the build."
        assert summary.token_estimate == 12
        assert summary.metadata == {
            "cutctx.kind": "summary",
            "cutctx.summary_group": "g1",
            "cutctx.replaced_turn_count": "2",
        }
        assert view.transcript.turn_ids() == ("summary:g1", "c")

    def test_the_summary_turn_carries_the_planned_estimate_not_a_measurement(self) -> None:
        """The plan's figure is the audited one; a report disagreeing with it would be unusable."""
        source = transcript(turn("a", tokens=500))
        plan = planned(
            source,
            TurnAction("a", Action.SUMMARIZE, summary_group="g1"),
            summarization_requests=(SummarizationRequest("g1", ("a",), 9, "general.summarize"),),
        )

        view = EXECUTOR.apply(source, plan, {"g1": "x" * 4_000})  # type: ignore[arg-type]

        assert view.transcript.token_estimate() == 9
        assert view.report.tokens_after_estimate == 9

    def test_a_surplus_summary_is_ignored(self) -> None:
        source = transcript(turn("a", tokens=5))
        plan = planned(source, TurnAction("a", Action.KEEP))

        view = EXECUTOR.apply(source, plan, {"not-in-this-plan": "text"})  # type: ignore[arg-type]

        assert view.transcript.turn_ids() == ("a",)


class TestMasking:
    def test_a_mask_replaces_the_body_in_place_and_keeps_the_turn_in_its_exchange(self) -> None:
        source = transcript(
            turn("a1", Role.ASSISTANT, tokens=10, tool_call_id="c1"),
            turn("t1", Role.TOOL, tokens=900, tool_call_id="c1", pinned=False),
        )
        plan = planned(
            source,
            TurnAction("a1", Action.KEEP),
            TurnAction(
                "t1",
                Action.MASK,
                replacement=TurnReplacement(
                    content="[tool result masked: sha256:ab…]", token_estimate=8
                ),
            ),
        )

        view = EXECUTOR.apply(source, plan)  # type: ignore[arg-type]

        masked = view.transcript.turns[1]
        assert masked.turn_id == "t1"
        assert masked.role is Role.TOOL
        assert masked.tool_call_id == "c1"
        assert masked.content == "[tool result masked: sha256:ab…]"
        assert masked.token_estimate == 8
        assert masked.metadata["cutctx.kind"] == "masked"
        assert view.report.masked_turn_ids == ("t1",)

    def test_a_mask_keeps_the_callers_own_metadata_beside_its_own_label(self) -> None:
        source = Transcript(turns=(turn("a", tokens=10),))
        source = Transcript(
            turns=(
                type(source.turns[0])(
                    turn_id="a",
                    role=Role.USER,
                    content="x",
                    token_estimate=10,
                    metadata={"kind": "note"},
                ),
            )
        )
        plan = planned(
            source,
            TurnAction(
                "a", Action.MASK, replacement=TurnReplacement(content="[masked]", token_estimate=1)
            ),
        )

        view = EXECUTOR.apply(source, plan)  # type: ignore[arg-type]

        assert view.transcript.turns[0].metadata == {"kind": "note", "cutctx.kind": "masked"}


class TestImmutabilityAndReport:
    def test_the_input_transcript_is_untouched_down_to_object_identity(self) -> None:
        source = transcript(turn("a", tokens=10), turn("b", tokens=10))
        before = canonical_json([t.turn_id for t in source.turns])
        identities = [id(t) for t in source.turns]
        plan = planned(source, TurnAction("a", Action.DROP), TurnAction("b", Action.KEEP))

        EXECUTOR.apply(source, plan)  # type: ignore[arg-type]

        assert canonical_json([t.turn_id for t in source.turns]) == before
        assert [id(t) for t in source.turns] == identities

    def test_a_kept_turn_is_the_very_same_object(self) -> None:
        """Nothing is rebuilt for the sake of it: a KEEP is a pass-through."""
        source = transcript(turn("a", tokens=10))
        plan = planned(source, TurnAction("a", Action.KEEP))

        view = EXECUTOR.apply(source, plan)  # type: ignore[arg-type]

        assert view.transcript.turns[0] is source.turns[0]

    def test_the_report_copies_the_plans_figures_and_links_back_by_hash(self) -> None:
        source = transcript(turn("a", tokens=10), turn("b", tokens=90))
        plan = planned(
            source,
            TurnAction("a", Action.DROP),
            TurnAction("b", Action.KEEP),
            estimator_ratio=4.0,
        )

        report = EXECUTOR.apply(source, plan).report  # type: ignore[arg-type]

        assert report.plan_hash == plan.plan_hash()  # type: ignore[attr-defined]
        assert (report.tokens_before, report.tokens_after_estimate) == (100, 90)
        assert report.estimator_ratio == 4.0
        assert (report.turns_before, report.turns_after) == (2, 1)
        assert report.dropped_turn_ids == ("a",)
        assert report.kept_turn_ids == ("b",)
        assert report.policy_name == "test"

    def test_an_empty_transcript_applies_to_an_empty_view(self) -> None:
        empty = Transcript()
        plan = planned(empty)

        view = EXECUTOR.apply(empty, plan)  # type: ignore[arg-type]

        assert view.transcript.turns == ()
        assert view.report.turns_after == 0
