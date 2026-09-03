"""The vocabulary's own refusals and the shapes settled for row E1 to inherit."""

from __future__ import annotations

import pytest
from baseaicore import ValidationError, canonical_json

from conftest import transcript, turn
from cutctx import (
    EMPTY_METADATA,
    Action,
    CompactionBudget,
    CompactionExecutor,
    CompactionPlan,
    CompactionReport,
    DropOldestPolicy,
    Role,
    SummarizationRequest,
    Transcript,
    TranscriptTurn,
    TurnAction,
    TurnReplacement,
)


class TestTranscriptTurn:
    def test_an_empty_turn_id_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="non-empty turn_id"):
            TranscriptTurn(turn_id="", role=Role.USER, content="x", token_estimate=1)

    def test_a_negative_token_estimate_is_refused(self) -> None:
        with pytest.raises(ValidationError) as refusal:
            TranscriptTurn(turn_id="a", role=Role.USER, content="x", token_estimate=-1)

        assert refusal.value.details["token_estimate"] == -1

    def test_metadata_is_copied_so_a_later_mutation_cannot_reach_the_turn(self) -> None:
        caller_owned = {"kind": "note"}
        built = turn("a")
        built = TranscriptTurn("a", Role.USER, "x", 1, metadata=caller_owned)

        caller_owned["kind"] = "changed"

        assert built.metadata == {"kind": "note"}

    def test_metadata_cannot_be_written_through(self) -> None:
        built = TranscriptTurn("a", Role.USER, "x", 1, metadata={"kind": "note"})

        with pytest.raises(TypeError):
            built.metadata["kind"] = "changed"  # type: ignore[index]  # the refusal is the point

    def test_the_default_metadata_is_an_empty_immutable_mapping(self) -> None:
        assert turn("a").metadata == {}
        assert EMPTY_METADATA == {}
        with pytest.raises(TypeError):
            EMPTY_METADATA["k"] = "v"  # a Mapping has no __setitem__; the refusal is the point

    def test_turns_compare_by_value_regardless_of_metadata_ordering(self) -> None:
        one = TranscriptTurn("a", Role.USER, "x", 1, metadata={"k1": "a", "k2": "b"})
        two = TranscriptTurn("a", Role.USER, "x", 1, metadata={"k2": "b", "k1": "a"})

        assert one == two

    def test_a_frozen_turn_refuses_assignment(self) -> None:
        with pytest.raises(AttributeError):
            turn("a").content = "changed"  # type: ignore[misc]  # the refusal is the point


class TestTranscript:
    def test_duplicate_turn_ids_are_refused(self) -> None:
        with pytest.raises(ValidationError) as refusal:
            Transcript(turns=(turn("a"), turn("b"), turn("a")))

        assert refusal.value.details["duplicate_turn_ids"] == ["a"]

    def test_an_empty_transcript_estimates_at_zero(self) -> None:
        assert Transcript().token_estimate() == 0
        assert Transcript().turn_ids() == ()

    def test_the_estimate_is_the_sum_of_the_callers_own_figures(self) -> None:
        assert transcript(turn("a", tokens=3), turn("b", tokens=4)).token_estimate() == 7


class TestCompactionBudget:
    def test_a_zero_budget_is_legal(self) -> None:
        assert CompactionBudget(max_tokens=0).max_tokens == 0

    def test_the_default_protected_tail_is_four(self) -> None:
        assert CompactionBudget(max_tokens=10).protected_recent_turns == 4

    @pytest.mark.parametrize(
        ("kwargs", "field"),
        [
            ({"max_tokens": -1}, "max_tokens"),
            ({"max_tokens": 1, "protected_recent_turns": -1}, "protected_recent_turns"),
        ],
    )
    def test_negative_values_are_refused(self, kwargs: dict[str, int], field: str) -> None:
        with pytest.raises(ValidationError) as refusal:
            CompactionBudget(**kwargs)

        assert field in refusal.value.details


class TestTurnAction:
    def test_a_summarize_without_a_group_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="summary_group is set exactly"):
            TurnAction(turn_id="a", action=Action.SUMMARIZE)

    def test_a_group_without_a_summarize_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="summary_group is set exactly"):
            TurnAction(turn_id="a", action=Action.KEEP, summary_group="g1")

    def test_a_mask_without_a_replacement_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="replacement is set exactly"):
            TurnAction(turn_id="a", action=Action.MASK)

    def test_a_replacement_without_a_mask_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="replacement is set exactly"):
            TurnAction(
                turn_id="a",
                action=Action.DROP,
                replacement=TurnReplacement(content="x", token_estimate=1),
            )

    def test_the_json_shape_names_every_field(self) -> None:
        action = TurnAction(
            turn_id="a",
            action=Action.MASK,
            replacement=TurnReplacement(content="stub", token_estimate=2),
        )

        assert action.to_dict() == {
            "turn_id": "a",
            "action": "mask",
            "summary_group": None,
            "replacement": {"content": "stub", "token_estimate": 2},
        }


class TestTurnReplacement:
    def test_a_negative_estimate_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            TurnReplacement(content="x", token_estimate=-1)


class TestSummarizationRequest:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"group_id": ""}, "non-empty group_id"),
            ({"prompt_id": ""}, "versioned prompt record"),
            ({"turn_ids": ()}, "covers no turns"),
            ({"turn_ids": ("a", "a")}, "repeats a turn id"),
            ({"target_tokens": -1}, "negative target_tokens"),
        ],
    )
    def test_a_request_no_caller_could_fulfil_is_refused(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        base: dict[str, object] = {
            "group_id": "g1",
            "turn_ids": ("a",),
            "target_tokens": 4,
            "prompt_id": "general.summarize",
        }

        with pytest.raises(ValidationError, match=match):
            SummarizationRequest(**{**base, **kwargs})  # type: ignore[arg-type]  # deliberate

    def test_the_summary_turn_id_is_derived_from_the_group_id(self) -> None:
        request = SummarizationRequest("g1", ("a",), 4, "general.summarize")

        assert request.summary_turn_id == "summary:g1"

    def test_the_json_shape_carries_the_prompt_name_and_never_prompt_text(self) -> None:
        request = SummarizationRequest("g1", ("a", "b"), 4, "general.summarize")

        assert request.to_dict() == {
            "group_id": "g1",
            "turn_ids": ["a", "b"],
            "target_tokens": 4,
            "prompt_id": "general.summarize",
        }


class TestCompactionReport:
    def test_the_event_body_is_scalars_and_id_lists_and_carries_no_content(self) -> None:
        """Contract 7: two applications emit this without reshaping, and neither quotes a turn."""
        report = CompactionReport(
            plan_hash="abc",
            policy_name="drop_oldest",
            policy_version="1.0.0",
            tokens_before=100,
            tokens_after_estimate=40,
            estimator_ratio=None,
            budget_unmet=False,
            turns_before=5,
            turns_after=2,
            kept_turn_ids=("a",),
            masked_turn_ids=(),
            summarized_turn_ids=("b", "c"),
            dropped_turn_ids=("d",),
            summary_turn_ids=("summary:g1",),
        )

        body = report.to_dict()

        assert body["tokens_after_estimate"] == 40
        assert body["summarized_turn_ids"] == ["b", "c"]
        assert "timestamp" not in body and "created_at" not in body
        assert canonical_json(body)  # serializable with no further conversion

    def test_it_carries_no_timestamp_field_at_all(self) -> None:
        """The caller stamps: a plan carrying a time could not be re-derived byte-identically."""
        fields = set(CompactionReport.__dataclass_fields__)

        assert not any("time" in name or "_at" in name for name in fields)


class TestCompactionPlanIntrospection:
    def test_the_initvar_names_resolve_so_a_pretty_printer_cannot_crash(self) -> None:
        """``__dataclass_fields__`` includes InitVars, and printers iterate it with ``getattr``.

        Without this, printing a plan raises ``AttributeError`` — which is how a failing property
        test about a plan would report the crash instead of the plan. Caught exactly that way.
        """
        source = transcript(turn("a", tokens=10))
        plan = DropOldestPolicy().decide(source, CompactionBudget(100, protected_recent_turns=0))

        for name, field in CompactionPlan.__dataclass_fields__.items():
            if field.init:
                assert getattr(plan, name) is not Ellipsis  # resolves rather than raising

        assert "not carried on the plan" in repr(plan.transcript)  # type: ignore[attr-defined]

    def test_a_plan_carries_no_transcript_in_its_value(self) -> None:
        """The InitVars leave no trace: not in ``==``, not in ``repr``, not in ``to_dict``."""
        source = transcript(turn("a", tokens=10))
        budget = CompactionBudget(100, protected_recent_turns=0)
        plan = DropOldestPolicy().decide(source, budget)

        assert plan == DropOldestPolicy().decide(source, budget)
        assert "TranscriptTurn" not in repr(plan)
        assert set(plan.to_dict()) == {
            "policy_name",
            "policy_version",
            "tokens_before",
            "tokens_after_estimate",
            "estimator_ratio",
            "budget_unmet",
            "actions",
            "summarization_requests",
        }


def test_the_packages_own_documented_example_still_holds() -> None:
    """The README and the package docstring quote figures; this is what stops them rotting."""
    source = Transcript(
        (
            TranscriptTurn("s", Role.SYSTEM, "You are a careful assistant.", 12),
            TranscriptTurn("u1", Role.USER, "What changed in the build?", 30),
            TranscriptTurn("a1", Role.ASSISTANT, "Let me look.", 20, tool_call_id="c1"),
            TranscriptTurn("t1", Role.TOOL, "<4 kB of build log>", 900, tool_call_id="c1"),
            TranscriptTurn("a2", Role.ASSISTANT, "The cache key changed.", 25),
        )
    )
    plan = DropOldestPolicy().decide(
        source, CompactionBudget(max_tokens=100, protected_recent_turns=1)
    )
    view = CompactionExecutor().apply(source, plan)

    assert (plan.tokens_before, plan.tokens_after_estimate, plan.budget_unmet) == (987, 37, False)
    assert view.transcript.turn_ids() == ("s", "a2")
    assert view.report.dropped_turn_ids == ("u1", "a1", "t1")
    assert source.turn_ids() == ("s", "u1", "a1", "t1", "a2")
