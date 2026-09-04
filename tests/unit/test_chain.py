"""``PolicyChain`` — the projection, the escalation, and the one plan built at the end.

The chain is the only policy that reads another policy's output, so it is the only one that can be
wrong about what a plan *means*. Three things are asserted here rather than assumed: that the
projection is what the executor would actually produce, that a later policy removing a summary turn
removes everything folded into it, and that ordering is transcript position at every step.
"""

from __future__ import annotations

import pytest
from baseaicore import ValidationError

from conftest import transcript, turn
from cutctx import (
    Action,
    CompactionBudget,
    CompactionExecutor,
    CompactionPlan,
    DropOldestPolicy,
    ObservationMaskingPolicy,
    PolicyChain,
    Role,
    SummarizingPolicy,
    Transcript,
    default_chain,
)
from cutctx.policies.chain import _project, _reconcile_exchanges, _Resolution

PROMPT = "compaction.summarize.v1"


def _agent_transcript() -> Transcript:
    """A system turn, six tool exchanges with reasoning between them, and a closing question."""
    turns = [turn("s1", Role.SYSTEM, tokens=10, content="system")]
    for index in range(1, 7):
        turns.append(turn(f"u{index}", tokens=20))
        turns.append(turn(f"a{index}", Role.ASSISTANT, tokens=25, tool_call_id=f"c{index}"))
        turns.append(
            turn(
                f"t{index}",
                Role.TOOL,
                tokens=300,
                tool_call_id=f"c{index}",
                content=chr(ord("A") + index) * 1200,
            )
        )
    turns.append(turn("u9", tokens=15))
    return transcript(*turns)


def _open(max_tokens: int) -> CompactionBudget:
    return CompactionBudget(max_tokens=max_tokens, protected_recent_turns=0)


def _counts(plan: CompactionPlan) -> dict[str, int]:
    counted: dict[str, int] = {}
    for action in plan.actions:
        counted[action.action.value] = counted.get(action.action.value, 0) + 1
    return counted


class TestConstruction:
    """A chain is caller configuration, so a broken one fails where the caller is."""

    def test_an_empty_chain_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="at least one"):
            PolicyChain(())

    def test_a_member_that_is_not_a_policy_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="CompactionPolicy"):
            PolicyChain((DropOldestPolicy(), object()))  # type: ignore[arg-type]

    def test_the_name_says_which_policies_at_which_versions(self) -> None:
        """A report saying only "chain" leaves an auditor unable to tell what produced the view."""
        chain = PolicyChain((ObservationMaskingPolicy(), DropOldestPolicy()))
        assert chain.name == "chain(observation_masking@1.0.0+drop_oldest@1.0.0)"
        assert chain.version == "1.0.0"

    def test_the_default_chain_is_mask_then_summarize_then_drop(self) -> None:
        """The order is an argument about cost: free, then a model call, then irreversible loss."""
        chain = default_chain(prompt_id=PROMPT)
        assert [p.name for p in chain.policies] == [
            "observation_masking",
            "summarizing",
            "drop_oldest",
        ]


class TestStoppingEarly:
    """A chain runs a policy only while the budget is unmet."""

    def test_a_transcript_already_within_budget_runs_no_policy(self) -> None:
        source = _agent_transcript()
        plan = default_chain(prompt_id=PROMPT).decide(source, _open(source.token_estimate()))
        assert _counts(plan) == {"keep": len(source.turns)}
        assert plan.summarization_requests == ()
        assert plan.budget_unmet is False

    def test_masking_alone_is_enough_so_nothing_is_summarized(self) -> None:
        """Dev-plan AC: "masking reduces enough → no summarization planned"."""
        source = _agent_transcript()
        # Masking every result but the two most recent brings 2 095 down to 1 023, so a budget
        # above that is met without a model call ever being planned.
        plan = default_chain(prompt_id=PROMPT).decide(source, _open(1_100))
        assert plan.summarization_requests == ()
        assert _counts(plan).get("drop", 0) == 0
        assert _counts(plan)["mask"] > 0
        assert plan.budget_unmet is False

    def test_a_later_policy_runs_when_an_earlier_one_could_not_finish(self) -> None:
        source = _agent_transcript()
        plan = default_chain(prompt_id=PROMPT).decide(source, _open(300))
        assert _counts(plan).get("summarize", 0) + _counts(plan).get("drop", 0) > 0
        assert plan.budget_unmet is False


class TestTheProjectionMatchesTheView:
    """The projection is a claim about what `apply` would produce. It is checked against it."""

    @pytest.mark.parametrize("max_tokens", [2200, 1500, 900, 500, 300, 120])
    def test_the_plans_estimate_is_the_applied_views_estimate(self, max_tokens: int) -> None:
        """If the projection were wrong, the composed plan's arithmetic would be wrong with it."""
        source = _agent_transcript()
        plan = default_chain(prompt_id=PROMPT).decide(source, _open(max_tokens))
        summaries = {
            request.group_id: "x" * request.target_tokens for request in plan.summarization_requests
        }
        view = CompactionExecutor().apply(source, plan, summaries).transcript
        assert view.token_estimate() == plan.tokens_after_estimate

    def test_a_masked_turn_keeps_its_id_and_its_correlation_id_in_the_projection(self) -> None:
        """A later policy must see the same exchanges, or it will split one."""
        source = _agent_transcript()
        resolved = {t.turn_id: _Resolution(Action.KEEP) for t in source.turns}
        masked = ObservationMaskingPolicy(0).decide(source, _open(900))
        for action in masked.actions:
            if action.action is Action.MASK:
                resolved[action.turn_id] = _Resolution(Action.MASK, replacement=action.replacement)
        projection = _project(source, resolved, {})
        by_id = {t.turn_id: t for t in projection.turns}
        for original in source.turns:
            assert by_id[original.turn_id].tool_call_id == original.tool_call_id
            assert by_id[original.turn_id].role is original.role


class TestEscalation:
    """A later policy removing a summary turn removes everything folded into it."""

    def test_dropping_a_summary_turn_drops_its_whole_group(self) -> None:
        source = transcript(
            turn("s1", Role.SYSTEM, tokens=10),
            *(turn(f"m{i}", tokens=60) for i in range(8)),
            turn("u9", tokens=15),
        )
        chain = PolicyChain((SummarizingPolicy(PROMPT, min_span_turns=4), DropOldestPolicy()))
        plan = chain.decide(source, _open(40))
        assert plan.summarization_requests == (), "a request survived its summary turn's removal"
        dropped = {a.turn_id for a in plan.actions if a.action is Action.DROP}
        assert {f"m{i}" for i in range(8)} <= dropped
        assert not any(a.action is Action.SUMMARIZE for a in plan.actions)

    def test_a_group_that_survives_keeps_its_request(self) -> None:
        source = transcript(
            turn("s1", Role.SYSTEM, tokens=10),
            *(turn(f"m{i}", tokens=60) for i in range(8)),
            turn("u9", tokens=15),
        )
        chain = PolicyChain((SummarizingPolicy(PROMPT, min_span_turns=4), DropOldestPolicy()))
        plan = chain.decide(source, _open(200))
        assert len(plan.summarization_requests) == 1
        # Everything but the system turn is one contiguous removable run, so the span is all of it.
        assert plan.summarization_requests[0].turn_ids == (
            *(f"m{i}" for i in range(8)),
            "u9",
        )


class TestExchangeReconciliation:
    """Contract 3 as `C1_HANDOFF.md` §4 reads it, enforced where the chain composes.

    ``_reconcile_exchanges`` is exercised directly. End to end it is defence in depth — each policy
    runs against a projection that preserves ids and correlation ids, so its own ``build_plan``
    already refuses a split on its own view — but that argument is a property of the projection
    rather than of the contract, so the function is tested against the states it exists to fix.
    """

    @staticmethod
    def _exchange() -> Transcript:
        return transcript(
            turn("a1", Role.ASSISTANT, tokens=20, tool_call_id="c1"),
            turn("t1", Role.TOOL, tokens=300, tool_call_id="c1"),
            turn("u1", tokens=10),
        )

    def test_a_dropped_call_takes_its_masked_result_with_it(self) -> None:
        """The exact shape the kickoff names: a later policy drops a call an earlier one masked."""
        source = self._exchange()
        resolved = {
            "a1": _Resolution(Action.DROP),
            "t1": _Resolution(Action.MASK),
            "u1": _Resolution(Action.KEEP),
        }
        _reconcile_exchanges(resolved, source, {})
        assert resolved["t1"].action is Action.DROP
        assert resolved["t1"].replacement is None
        assert resolved["u1"].action is Action.KEEP

    def test_a_dropped_member_beats_a_summarized_one(self) -> None:
        """A member already gone cannot be folded, and summarizing the rest misdescribes it."""
        source = self._exchange()
        resolved = {
            "a1": _Resolution(Action.SUMMARIZE, group="g1"),
            "t1": _Resolution(Action.DROP),
            "u1": _Resolution(Action.KEEP),
        }
        _reconcile_exchanges(resolved, source, {})
        assert resolved["a1"].action is Action.DROP
        assert resolved["a1"].group is None

    def test_a_half_summarized_exchange_is_completed_into_the_earliest_group(self) -> None:
        source = self._exchange()
        resolved = {
            "a1": _Resolution(Action.SUMMARIZE, group="g1"),
            "t1": _Resolution(Action.KEEP),
            "u1": _Resolution(Action.KEEP),
        }
        _reconcile_exchanges(resolved, source, {})
        assert resolved["t1"].action is Action.SUMMARIZE
        assert resolved["t1"].group == "g1"

    def test_two_groups_over_one_exchange_resolve_to_the_earliest_by_position(self) -> None:
        """By transcript position — never by whatever order a mapping happened to yield."""
        source = self._exchange()
        resolved = {
            "a1": _Resolution(Action.SUMMARIZE, group="early"),
            "t1": _Resolution(Action.SUMMARIZE, group="late"),
            "u1": _Resolution(Action.KEEP),
        }
        _reconcile_exchanges(resolved, source, {})
        assert resolved["a1"].group == "early"
        assert resolved["t1"].group == "early"

    def test_a_retained_exchange_mixing_keep_and_mask_is_left_alone(self) -> None:
        """Masking removes nothing, so the mix is legal — this is why masking is legal at all."""
        source = self._exchange()
        resolved = {
            "a1": _Resolution(Action.KEEP),
            "t1": _Resolution(Action.MASK),
            "u1": _Resolution(Action.KEEP),
        }
        _reconcile_exchanges(resolved, source, {})
        assert resolved["t1"].action is Action.MASK

    def test_a_wholly_dropped_exchange_is_left_alone(self) -> None:
        source = self._exchange()
        resolved = {
            "a1": _Resolution(Action.DROP),
            "t1": _Resolution(Action.DROP),
            "u1": _Resolution(Action.KEEP),
        }
        _reconcile_exchanges(resolved, source, {})
        assert resolved["a1"].action is Action.DROP
        assert resolved["t1"].action is Action.DROP

    def test_a_group_emptied_by_escalation_loses_its_request(self) -> None:
        from cutctx import SummarizationRequest

        source = self._exchange()
        resolved = {
            "a1": _Resolution(Action.SUMMARIZE, group="g1"),
            "t1": _Resolution(Action.DROP),
            "u1": _Resolution(Action.KEEP),
        }
        groups = {
            "g1": SummarizationRequest(
                group_id="g1", turn_ids=("a1",), target_tokens=5, prompt_id=PROMPT
            )
        }
        _reconcile_exchanges(resolved, source, groups)
        assert groups == {}

    def test_a_group_widened_by_escalation_has_its_request_widened_with_it(self) -> None:
        from cutctx import SummarizationRequest

        source = self._exchange()
        resolved = {
            "a1": _Resolution(Action.SUMMARIZE, group="g1"),
            "t1": _Resolution(Action.KEEP),
            "u1": _Resolution(Action.KEEP),
        }
        groups = {
            "g1": SummarizationRequest(
                group_id="g1", turn_ids=("a1",), target_tokens=5, prompt_id=PROMPT
            )
        }
        _reconcile_exchanges(resolved, source, groups)
        assert groups["g1"].turn_ids == ("a1", "t1")


class TestTheComposedPlan:
    """One plan, built once, against the real transcript, through the invariants."""

    @pytest.mark.parametrize("max_tokens", [2200, 1500, 900, 500, 300, 120])
    def test_the_composed_plan_is_valid_by_construction(self, max_tokens: int) -> None:
        """It went through ``build_plan``, so it covers the transcript and keeps every exchange."""
        source = _agent_transcript()
        plan = default_chain(prompt_id=PROMPT).decide(source, _open(max_tokens))
        assert [a.turn_id for a in plan.actions] == [t.turn_id for t in source.turns]

    def test_budget_unmet_is_derived_and_not_declared(self) -> None:
        """`C1_HANDOFF.md` §3.2: `build_plan` recomputes it and refuses a plan that disagrees."""
        source = _agent_transcript()
        plan = default_chain(prompt_id=PROMPT).decide(source, _open(2200))
        assert plan.budget_unmet is (plan.tokens_after_estimate > 2200)

    def test_an_exhausted_chain_says_so_rather_than_truncating(self) -> None:
        """Spec §13's last row. Never a silent truncation.

        The shape that exhausts every shipped policy at once is tool-exchange lock-in
        (`C1_HANDOFF.md` §3.2): a huge **assistant** call whose small result sits in the protected
        tail. The result is untouchable, so the exchange is not removable and neither summarizing
        nor dropping may take the call; and the call is not a ``TOOL`` turn, so masking may not
        either. Nothing can reduce it, the floor is still under the budget so
        `BudgetUnsatisfiable` does not fire, and the honest answer is a plan that says it did not
        get there.
        """
        source = transcript(
            turn("a1", Role.ASSISTANT, tokens=500, tool_call_id="c1"),
            turn("u1", tokens=20),
            turn("t1", Role.TOOL, tokens=10, tool_call_id="c1"),
        )
        budget = CompactionBudget(max_tokens=100, protected_recent_turns=1)
        plan = default_chain(prompt_id=PROMPT).decide(source, budget)
        assert plan.budget_unmet is True
        assert plan.tokens_after_estimate > 100
        assert {a.turn_id for a in plan.actions if a.action is Action.DROP} == {"u1"}

    def test_the_ratio_of_the_first_policy_that_estimated_rides_on_the_plan(self) -> None:
        source = _agent_transcript()
        plan = default_chain(prompt_id=PROMPT).decide(source, _open(900))
        assert plan.estimator_ratio == 4.0

    def test_a_chain_that_estimated_nothing_records_no_ratio(self) -> None:
        source = _agent_transcript()
        chain = PolicyChain((SummarizingPolicy(PROMPT), DropOldestPolicy()))
        assert chain.decide(source, _open(400)).estimator_ratio is None


class TestDeterminism:
    """Contract 4 for the composition, and the named trap at chain level."""

    @pytest.mark.parametrize("max_tokens", [2200, 1500, 900, 500, 300, 120])
    def test_the_same_inputs_give_the_same_bytes(self, max_tokens: int) -> None:
        source = _agent_transcript()
        budget = _open(max_tokens)
        assert default_chain(prompt_id=PROMPT).decide(source, budget).canonical_json() == (
            default_chain(prompt_id=PROMPT).decide(source, budget).canonical_json()
        )

    def test_requests_are_ordered_by_the_position_of_their_earliest_turn(self) -> None:
        source = _agent_transcript()
        plan = default_chain(prompt_id=PROMPT).decide(source, _open(300))
        order = {t.turn_id: index for index, t in enumerate(source.turns)}
        positions = [order[r.turn_ids[0]] for r in plan.summarization_requests]
        assert positions == sorted(positions)

    def test_a_group_covers_its_turns_in_transcript_order(self) -> None:
        """Assembled from a mapping, so the ordering has to be imposed rather than inherited."""
        source = _agent_transcript()
        plan = default_chain(prompt_id=PROMPT).decide(source, _open(300))
        order = {t.turn_id: index for index, t in enumerate(source.turns)}
        for request in plan.summarization_requests:
            positions = [order[turn_id] for turn_id in request.turn_ids]
            assert positions == sorted(positions)
