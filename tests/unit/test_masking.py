"""``ObservationMaskingPolicy`` — what it stubs, what it refuses to touch, and what a stub says.

The stub's format is this phase's to define and it is golden-locked (spec §19), so most of what
follows is about the *edges*: which turns are eligible, which are never eligible whatever the
budget, and the two cases where masking would make things worse.
"""

from __future__ import annotations

import pytest
from baseaicore import ValidationError, sha256_of

from conftest import transcript, turn
from cutctx import (
    DEFAULT_PLACEHOLDER,
    Action,
    CompactionBudget,
    CompactionExecutor,
    ObservationMaskingPolicy,
    Role,
    Transcript,
)
from cutctx.policies.masking import stub_for


def _tool_heavy() -> Transcript:
    """A transcript whose bulk is four tool results, with reasoning between them."""
    return transcript(
        turn("s1", Role.SYSTEM, tokens=10, content="system"),
        turn("a1", Role.ASSISTANT, tokens=20, tool_call_id="c1", content="I will look."),
        turn("t1", Role.TOOL, tokens=400, tool_call_id="c1", content="A" * 1600),
        turn("a2", Role.ASSISTANT, tokens=25, content="It says the cache key changed."),
        turn("a3", Role.ASSISTANT, tokens=20, tool_call_id="c2", content="I will look again."),
        turn("t2", Role.TOOL, tokens=400, tool_call_id="c2", content="B" * 1600),
        turn("a4", Role.ASSISTANT, tokens=25, content="Confirmed."),
        turn("a5", Role.ASSISTANT, tokens=20, tool_call_id="c3", content="Once more."),
        turn("t3", Role.TOOL, tokens=400, tool_call_id="c3", content="C" * 1600),
        turn("a6", Role.ASSISTANT, tokens=20, tool_call_id="c4", content="And again."),
        turn("t4", Role.TOOL, tokens=400, tool_call_id="c4", content="D" * 1600),
        turn("u1", tokens=15, content="Thanks."),
    )


def _tight(max_tokens: int) -> CompactionBudget:
    """A budget with no protected tail, so a test can name a number and mean it.

    With the default tail of four, most of these transcripts' untouchable floor exceeds any tight
    budget and `BudgetUnsatisfiable` fires before the policy runs — correctly, but it would be
    testing the floor rather than the policy.
    """
    return CompactionBudget(max_tokens=max_tokens, protected_recent_turns=0)


def _actions(plan: object) -> dict[str, Action]:
    return {action.turn_id: action.action for action in plan.actions}  # type: ignore[attr-defined]


class TestConfigurationIsValidatedAtConstruction:
    """Caller configuration fails where the caller is, not on the transcript that tripped it."""

    @pytest.mark.parametrize("value", [-1, -10])
    def test_a_negative_keep_count_is_refused(self, value: int) -> None:
        with pytest.raises(ValidationError, match="keep_recent_results"):
            ObservationMaskingPolicy(value)

    @pytest.mark.parametrize("value", [True, 2.0, "2", None])
    def test_a_keep_count_that_is_not_an_int_is_refused(self, value: object) -> None:
        with pytest.raises(ValidationError, match="keep_recent_results"):
            ObservationMaskingPolicy(value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", ["", "   ", None, 17])
    def test_a_placeholder_that_says_nothing_is_refused(self, value: object) -> None:
        """A model cannot tell an empty stub from a turn that said nothing."""
        with pytest.raises(ValidationError, match="placeholder"):
            ObservationMaskingPolicy(placeholder=value)  # type: ignore[arg-type]

    def test_an_estimator_that_does_not_estimate_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="estimator"):
            ObservationMaskingPolicy(estimator=object())  # type: ignore[arg-type]

    def test_zero_is_a_policy_and_is_accepted(self) -> None:
        """ "Mask every eligible result" is a choice; a negative number is not."""
        assert ObservationMaskingPolicy(0).keep_recent_results == 0

    def test_the_configuration_is_readable_back(self) -> None:
        """A consumer reading a plan may need to know what produced it; the policy says so."""
        policy = ObservationMaskingPolicy(3, placeholder="[gone]")
        assert policy.keep_recent_results == 3
        assert policy.placeholder == "[gone]"
        assert (policy.name, policy.version) == ("observation_masking", "1.0.0")


class TestTheStub:
    """Spec §14: a hash of the original, never a secret-bearing excerpt."""

    def test_the_stub_carries_the_originals_digest_and_estimate(self) -> None:
        original = turn("t1", Role.TOOL, tokens=400, content="secret token sk-abc123")
        stub = stub_for(original, placeholder=DEFAULT_PLACEHOLDER)
        assert sha256_of("secret token sk-abc123") in stub
        assert "400 tokens" in stub

    def test_the_stub_holds_no_part_of_the_original(self) -> None:
        """Not a prefix, not a suffix, not an ellipsis of one. An excerpt is a leak with steps."""
        original = turn("t1", Role.TOOL, tokens=400, content="AKIAIOSFODNN7EXAMPLE and more")
        stub = stub_for(original, placeholder=DEFAULT_PLACEHOLDER)
        assert "AKIA" not in stub
        assert "EXAMPLE" not in stub

    def test_the_label_is_configurable_and_the_evidence_is_not(self) -> None:
        """A placeholder that could omit the digest would make §14 a matter of discipline."""
        original = turn("t1", Role.TOOL, tokens=400, content="body")
        stub = stub_for(original, placeholder="[gone]")
        assert stub.startswith("[gone]")
        assert sha256_of("body") in stub

    def test_two_turns_with_the_same_body_produce_the_same_stub(self) -> None:
        """The digest is of content alone, so a repeated observation is visibly the same one."""
        first = turn("t1", Role.TOOL, tokens=400, content="same")
        second = turn("t2", Role.TOOL, tokens=400, content="same")
        assert stub_for(first, placeholder="[x]") == stub_for(second, placeholder="[x]")


class TestWhatIsMasked:
    """Only TOOL bodies, only outside the untouchable set, only beyond the recent floor."""

    def test_only_tool_turns_are_masked(self) -> None:
        plan = ObservationMaskingPolicy(0).decide(_tool_heavy(), _tight(100))
        masked = {tid for tid, action in _actions(plan).items() if action is Action.MASK}
        assert masked <= {"t1", "t2", "t3", "t4"}

    def test_reasoning_turns_stay_byte_identical(self) -> None:
        """The value of this policy is that the reasoning survives; assert it, do not assume it."""
        source = _tool_heavy()
        plan = ObservationMaskingPolicy(0).decide(source, _tight(100))
        view = CompactionExecutor().apply(source, plan).transcript
        by_id = {t.turn_id: t for t in view.turns}
        for original in source.turns:
            if original.role is not Role.TOOL:
                assert by_id[original.turn_id] == original, original.turn_id

    def test_the_most_recent_results_are_never_masked_however_tight_the_budget(self) -> None:
        """A hard floor. A model that can see no recent observation is working blind."""
        plan = ObservationMaskingPolicy(2).decide(_tool_heavy(), _tight(11))
        actions = _actions(plan)
        assert actions["t3"] is Action.KEEP
        assert actions["t4"] is Action.KEEP
        assert plan.budget_unmet is True

    @pytest.mark.parametrize("keep", [0, 1, 2, 3, 4, 9])
    def test_n_is_respected_exactly(self, keep: int) -> None:
        plan = ObservationMaskingPolicy(keep).decide(_tool_heavy(), _tight(11))
        actions = _actions(plan)
        results = ["t1", "t2", "t3", "t4"]
        protected = results[len(results) - keep :] if keep else []
        for turn_id in protected:
            assert actions[turn_id] is Action.KEEP, turn_id

    def test_a_pinned_result_is_not_masked(self) -> None:
        source = transcript(
            turn("t1", Role.TOOL, tokens=400, pinned=True, content="A" * 1600),
            turn("t2", Role.TOOL, tokens=400, content="B" * 1600),
            turn("u1", tokens=10),
        )
        plan = ObservationMaskingPolicy(0).decide(source, _tight(401))
        assert _actions(plan)["t1"] is Action.KEEP

    def test_a_result_in_the_protected_tail_is_not_masked(self) -> None:
        source = _tool_heavy()
        budget = CompactionBudget(max_tokens=500, protected_recent_turns=2)
        plan = ObservationMaskingPolicy(0).decide(source, budget)
        actions = _actions(plan)
        assert actions["t4"] is Action.KEEP, "a result in the protected tail was masked"
        assert actions["t3"] is Action.MASK, "a result outside the tail was left alone"

    def test_masking_is_not_limited_by_removability(self) -> None:
        """A result whose *call* is untouchable is still maskable: masking removes nothing.

        This is the difference between `untouchable_turn_ids` and `removable_turn_ids`, and it is
        the reason `ObservationMaskingPolicy` is legal under contract 3 at all.
        """
        source = transcript(
            turn("t1", Role.TOOL, tokens=400, tool_call_id="c1", content="A" * 1600),
            turn("u1", tokens=10),
            turn("u2", tokens=10),
            turn("a1", Role.ASSISTANT, tokens=20, tool_call_id="c1", pinned=True),
        )
        plan = ObservationMaskingPolicy(0).decide(source, _tight(100))
        assert _actions(plan)["t1"] is Action.MASK


class TestWhereMaskingWouldNotHelp:
    """Two cases where the policy declines, both of which would otherwise make things worse."""

    def test_a_result_smaller_than_its_own_stub_is_left_alone(self) -> None:
        """A transcript of tiny results would *grow* under a policy whose purpose is shrinking."""
        source = transcript(
            turn("t1", Role.TOOL, tokens=2, content="ok"),
            turn("t2", Role.TOOL, tokens=2, content="ok"),
            turn("u1", tokens=10),
        )
        plan = ObservationMaskingPolicy(0).decide(source, _tight(1))
        assert all(action is Action.KEEP for action in _actions(plan).values())
        assert plan.tokens_after_estimate == plan.tokens_before

    def test_masking_stops_as_soon_as_the_budget_fits(self) -> None:
        """Masking nine results when two would do costs the model seven observations."""
        source = _tool_heavy()
        generous = ObservationMaskingPolicy(0).decide(source, _tight(1400))
        tight = ObservationMaskingPolicy(0).decide(source, _tight(200))
        generous_masked = sum(1 for a in generous.actions if a.action is Action.MASK)
        tight_masked = sum(1 for a in tight.actions if a.action is Action.MASK)
        assert 0 < generous_masked < tight_masked

    def test_a_transcript_already_within_budget_is_untouched(self) -> None:
        source = _tool_heavy()
        plan = ObservationMaskingPolicy().decide(
            source, CompactionBudget(max_tokens=source.token_estimate())
        )
        assert all(action is Action.KEEP for action in _actions(plan).values())

    def test_masking_the_oldest_first(self) -> None:
        """Order is transcript position: the observation reasoned about longest ago goes first."""
        plan = ObservationMaskingPolicy(0).decide(_tool_heavy(), _tight(1400))
        actions = _actions(plan)
        assert actions["t1"] is Action.MASK
        assert actions["t4"] is Action.KEEP


class TestTheEstimatorRatio:
    """ADR-0016: a reader must be able to tell an estimate from a count."""

    def test_the_default_estimators_ratio_rides_on_a_plan_that_masked(self) -> None:
        plan = ObservationMaskingPolicy(0).decide(_tool_heavy(), _tight(100))
        assert plan.estimator_ratio == 4.0

    def test_a_plan_that_masked_nothing_records_no_ratio(self) -> None:
        """Recording one would describe a computation that did not happen."""
        source = _tool_heavy()
        plan = ObservationMaskingPolicy().decide(
            source, CompactionBudget(max_tokens=source.token_estimate())
        )
        assert plan.estimator_ratio is None

    def test_an_injected_estimator_records_no_ratio(self) -> None:
        """CutCtx will not describe a ratio somebody else's tokenizer does not have."""

        class Fixed:
            def estimate_tokens(self, text: str) -> int:
                return len(text) // 10

        plan = ObservationMaskingPolicy(0, estimator=Fixed()).decide(_tool_heavy(), _tight(100))
        assert plan.estimator_ratio is None
        assert any(a.action is Action.MASK for a in plan.actions)

    def test_an_injected_estimator_decides_the_stubs_cost(self) -> None:
        class Fixed:
            def estimate_tokens(self, text: str) -> int:
                del text
                return 7

        plan = ObservationMaskingPolicy(0, estimator=Fixed()).decide(_tool_heavy(), _tight(100))
        masked = [a for a in plan.actions if a.replacement is not None]
        assert masked
        assert all(a.replacement.token_estimate == 7 for a in masked)  # type: ignore[union-attr]


class TestDeterminism:
    """Contract 4, at the level of this policy."""

    @pytest.mark.parametrize("max_tokens", [11, 100, 800, 1400, 5000])
    def test_the_same_inputs_give_the_same_bytes(self, max_tokens: int) -> None:
        source = _tool_heavy()
        budget = _tight(max_tokens)
        policy = ObservationMaskingPolicy(1)
        assert policy.decide(source, budget).canonical_json() == (
            policy.decide(source, budget).canonical_json()
        )
