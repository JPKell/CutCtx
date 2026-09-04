"""``SummarizingPolicy`` — which span, why that one, and the two honesty points.

The span rules are where the work is: contiguous, removable, and **whole in its exchanges**, which
is not implied by the first two because an exchange's members need not be adjacent. Everything
else in this module is about what the policy refuses to plan.
"""

from __future__ import annotations

import pytest
from baseaicore import ValidationError

from conftest import transcript, turn
from cutctx import (
    GROUP_ID_PREFIX,
    SUMMARY_TURN_ID_PREFIX,
    Action,
    CompactionBudget,
    CompactionExecutor,
    Role,
    SummarizingPolicy,
    Transcript,
)
from cutctx.policies.summarizing import _spans, group_id_for

PROMPT = "compaction.summarize.v1"


def _long_chat(count: int = 10) -> Transcript:
    """A system turn and ``count`` alternating user/assistant turns, nothing pinned."""
    turns = [turn("s1", Role.SYSTEM, tokens=10)]
    for index in range(count):
        role = Role.USER if index % 2 == 0 else Role.ASSISTANT
        turns.append(turn(f"m{index}", role, tokens=50))
    return transcript(*turns)


def _open(max_tokens: int) -> CompactionBudget:
    """A budget with no protected tail, so a test names a number and means it."""
    return CompactionBudget(max_tokens=max_tokens, protected_recent_turns=0)


class TestConfigurationIsValidatedAtConstruction:
    """Caller configuration, so it fails at construction rather than on a transcript."""

    @pytest.mark.parametrize("value", ["", "  ", None, 17])
    def test_an_absent_prompt_id_is_refused(self, value: object) -> None:
        """ADR-0012: a prompt is a versioned record's name. This package carries no prompt text."""
        with pytest.raises(ValidationError, match="prompt_id"):
            SummarizingPolicy(value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [0, -0.5, float("inf"), float("nan"), "0.2", True, None])
    def test_a_ratio_that_could_not_produce_a_summary_is_refused(self, value: object) -> None:
        with pytest.raises(ValidationError, match="target_ratio"):
            SummarizingPolicy(PROMPT, target_ratio=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [0, -1, 2.0, "4", True, None])
    def test_a_min_span_that_is_not_a_span_is_refused(self, value: object) -> None:
        with pytest.raises(ValidationError, match="min_span_turns"):
            SummarizingPolicy(PROMPT, min_span_turns=value)  # type: ignore[arg-type]

    def test_the_configuration_is_readable_back(self) -> None:
        """A consumer reading a plan may need to know what produced it; the policy says so."""
        policy = SummarizingPolicy(PROMPT, target_ratio=0.35, min_span_turns=6)
        assert policy.prompt_id == PROMPT
        assert policy.target_ratio == 0.35
        assert policy.min_span_turns == 6
        assert (policy.name, policy.version) == ("summarizing", "1.0.0")


class TestTheGroupId:
    """Contract 4 lives or dies here: the id is in the plan's bytes and in the summary turn's id."""

    def test_it_is_derived_from_the_span_and_not_from_a_counter(self) -> None:
        assert group_id_for(("a", "b")) == group_id_for(("a", "b"))
        assert group_id_for(("a", "b")) != group_id_for(("b", "a"))

    def test_ids_that_differ_only_in_where_a_separator_falls_do_not_collide(self) -> None:
        """A printable separator would let ``["a", "b-c"]`` and ``["a-b", "c"]`` digest alike."""
        assert group_id_for(("a", "b-c")) != group_id_for(("a-b", "c"))

    def test_it_is_recognisable_as_a_derived_id(self) -> None:
        assert group_id_for(("a",)).startswith(GROUP_ID_PREFIX)

    def test_the_summary_turn_id_derives_from_it(self) -> None:
        plan = SummarizingPolicy(PROMPT).decide(_long_chat(), _open(200))
        request = plan.summarization_requests[0]
        assert request.summary_turn_id == f"{SUMMARY_TURN_ID_PREFIX}{request.group_id}"


class TestWhichSpan:
    """Oldest, contiguous, removable, and whole in its exchanges."""

    def test_the_oldest_qualifying_span_is_taken(self) -> None:
        plan = SummarizingPolicy(PROMPT, min_span_turns=4).decide(_long_chat(), _open(200))
        folded = [a.turn_id for a in plan.actions if a.action is Action.SUMMARIZE]
        assert folded[0] == "m0"

    def test_the_system_turn_is_never_in_a_span(self) -> None:
        plan = SummarizingPolicy(PROMPT).decide(_long_chat(), _open(200))
        folded = {a.turn_id for a in plan.actions if a.action is Action.SUMMARIZE}
        assert "s1" not in folded

    def test_a_pinned_turn_breaks_a_span_rather_than_joining_it(self) -> None:
        source = transcript(
            turn("u1", tokens=50),
            turn("u2", tokens=50),
            turn("p1", tokens=50, pinned=True),
            turn("u3", tokens=50),
            turn("u4", tokens=50),
            turn("u5", tokens=50),
            turn("u6", tokens=50),
        )
        spans = _spans(source, _open(10))
        assert ("u1", "u2") in spans
        assert all("p1" not in span for span in spans)

    def test_a_span_never_clips_an_exchange_whose_members_are_not_adjacent(self) -> None:
        """The case a "contiguous and removable" rule alone would get wrong.

        ``a1`` and ``t1`` are one exchange with a pinned turn between them, so the contiguous
        removable run ``(a1, u1)`` contains half an exchange. Folding it would separate the pair,
        which contract 3 forbids — so ``a1`` is filtered out and the run becomes ``(u1,)``.
        """
        source = transcript(
            turn("a1", Role.ASSISTANT, tokens=50, tool_call_id="c1"),
            turn("u1", tokens=50),
            turn("p1", tokens=50, pinned=True),
            turn("u2", tokens=50),
            turn("t1", Role.TOOL, tokens=50, tool_call_id="c1"),
        )
        spans = _spans(source, _open(10))
        for span in spans:
            assert not ({"a1", "t1"} & set(span)) or {"a1", "t1"} <= set(span)

    def test_a_whole_exchange_inside_one_run_is_kept_together(self) -> None:
        source = transcript(
            turn("a1", Role.ASSISTANT, tokens=50, tool_call_id="c1"),
            turn("u1", tokens=50),
            turn("t1", Role.TOOL, tokens=50, tool_call_id="c1"),
            turn("u2", tokens=50),
        )
        assert _spans(source, _open(10)) == [("a1", "u1", "t1", "u2")]

    def test_filtering_a_half_exchange_can_fragment_a_run_and_the_fragments_stay_ordered(
        self,
    ) -> None:
        """Removing a turn from the middle of a run leaves two runs, not one with a gap."""
        source = transcript(
            turn("u0", tokens=50),
            turn("a1", Role.ASSISTANT, tokens=50, tool_call_id="c1"),
            turn("u1", tokens=50),
            turn("p1", tokens=50, pinned=True),
            turn("t1", Role.TOOL, tokens=50, tool_call_id="c1"),
        )
        spans = _spans(source, _open(10))
        assert spans == [("u0",), ("u1",)]

    def test_the_protected_tail_is_outside_every_span(self) -> None:
        source = _long_chat()
        budget = CompactionBudget(max_tokens=200, protected_recent_turns=3)
        folded = {
            a.turn_id
            for a in SummarizingPolicy(PROMPT).decide(source, budget).actions
            if a.action is Action.SUMMARIZE
        }
        tail = {t.turn_id for t in source.turns[-3:]}
        assert not folded & tail


class TestWhenItPlansNothing:
    """A policy that finds nothing to do says so by planning nothing, never by raising."""

    def test_a_span_shorter_than_the_minimum_is_not_folded(self) -> None:
        source = transcript(turn("s1", Role.SYSTEM, tokens=10), turn("u1", tokens=500))
        plan = SummarizingPolicy(PROMPT, min_span_turns=4).decide(source, _open(20))
        assert plan.summarization_requests == ()
        assert all(a.action is Action.KEEP for a in plan.actions)
        assert plan.budget_unmet is True

    def test_a_ratio_that_would_grow_the_span_folds_nothing(self) -> None:
        """A "compaction" that made the transcript bigger would be one in name only."""
        plan = SummarizingPolicy(PROMPT, target_ratio=1.5).decide(_long_chat(), _open(200))
        assert plan.summarization_requests == ()
        assert plan.tokens_after_estimate == plan.tokens_before

    def test_a_transcript_with_nothing_removable_folds_nothing(self) -> None:
        source = transcript(
            turn("s1", Role.SYSTEM, tokens=10),
            turn("p1", tokens=10, pinned=True),
            turn("p2", tokens=10, pinned=True),
        )
        plan = SummarizingPolicy(PROMPT, min_span_turns=1).decide(source, _open(30))
        assert plan.summarization_requests == ()

    def test_an_empty_transcript_folds_nothing(self) -> None:
        plan = SummarizingPolicy(PROMPT).decide(Transcript(), _open(0))
        assert plan.summarization_requests == ()
        assert plan.actions == ()


class TestTheRequest:
    """What the caller is asked to do, and what the plan counts for it."""

    def test_the_target_honours_the_ratio(self) -> None:
        plan = SummarizingPolicy(PROMPT, target_ratio=0.2, min_span_turns=4).decide(
            _long_chat(10), _open(200)
        )
        request = plan.summarization_requests[0]
        span_tokens = 50 * len(request.turn_ids)
        assert request.target_tokens == round(span_tokens * 0.2)

    def test_the_target_is_never_zero(self) -> None:
        """A summary planned to cost nothing is a model call that says nothing."""
        source = transcript(*(turn(f"m{i}", tokens=1) for i in range(8)))
        plan = SummarizingPolicy(PROMPT, target_ratio=0.01).decide(source, _open(4))
        assert plan.summarization_requests[0].target_tokens == 1

    def test_the_request_names_the_prompt_record_and_carries_no_prompt_text(self) -> None:
        plan = SummarizingPolicy(PROMPT).decide(_long_chat(), _open(200))
        assert plan.summarization_requests[0].prompt_id == PROMPT

    def test_the_request_and_the_actions_are_two_statements_of_one_fact(self) -> None:
        plan = SummarizingPolicy(PROMPT).decide(_long_chat(), _open(200))
        request = plan.summarization_requests[0]
        folded = tuple(a.turn_id for a in plan.actions if a.action is Action.SUMMARIZE)
        assert request.turn_ids == folded
        assert {a.summary_group for a in plan.actions if a.action is Action.SUMMARIZE} == {
            request.group_id
        }

    def test_exactly_one_span_is_folded_per_plan(self) -> None:
        """The spec's "the oldest contiguous unpinned span", taken literally: one model call."""
        source = transcript(
            turn("u0", tokens=50),
            turn("u1", tokens=50),
            turn("u2", tokens=50),
            turn("u3", tokens=50),
            turn("p1", tokens=1, pinned=True),
            turn("u4", tokens=50),
            turn("u5", tokens=50),
            turn("u6", tokens=50),
            turn("u7", tokens=50),
        )
        plan = SummarizingPolicy(PROMPT, min_span_turns=4).decide(source, _open(10))
        assert len(plan.summarization_requests) == 1
        assert plan.budget_unmet is True

    def test_the_plan_counts_the_target_and_the_executor_does_not_re_estimate(self) -> None:
        """Contract 5 working as intended: the figure is an estimate of a text that does not exist.

        The executor is handed a summary far longer than the target and the view still reports the
        target, because plan and report must be incapable of disagreeing (`C1_HANDOFF.md` §3.1).
        """
        source = _long_chat()
        plan = SummarizingPolicy(PROMPT).decide(source, _open(200))
        request = plan.summarization_requests[0]
        compacted = CompactionExecutor().apply(source, plan, {request.group_id: "x" * 10_000})
        summary_turn = next(
            t for t in compacted.transcript.turns if t.turn_id == request.summary_turn_id
        )
        assert summary_turn.token_estimate == request.target_tokens
        assert compacted.report.tokens_after_estimate == plan.tokens_after_estimate

    def test_no_ratio_is_recorded_because_nothing_was_estimated_from_text(self) -> None:
        plan = SummarizingPolicy(PROMPT).decide(_long_chat(), _open(200))
        assert plan.estimator_ratio is None


class TestDeterminism:
    """Contract 4, at the level of this policy — and the named trap."""

    @pytest.mark.parametrize("max_tokens", [10, 200, 400, 5000])
    def test_the_same_inputs_give_the_same_bytes(self, max_tokens: int) -> None:
        source = _long_chat()
        budget = _open(max_tokens)
        policy = SummarizingPolicy(PROMPT)
        assert policy.decide(source, budget).canonical_json() == (
            policy.decide(source, budget).canonical_json()
        )

    def test_span_order_comes_from_transcript_position_and_not_from_a_set(self) -> None:
        """The failure mode the development plan names, asserted structurally.

        ``removable_turn_ids`` returns a ``frozenset``; a span assembled by iterating it would be
        ordered by hash. Every span here must be an ascending run of transcript indices.
        """
        source = _long_chat(12)
        order = {t.turn_id: index for index, t in enumerate(source.turns)}
        for span in _spans(source, _open(10)):
            positions = [order[turn_id] for turn_id in span]
            assert positions == sorted(positions)
            assert positions == list(range(positions[0], positions[0] + len(positions)))
