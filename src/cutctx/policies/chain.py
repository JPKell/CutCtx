"""Compose policies over a projection, and build exactly one plan at the end.

A chain runs its policies in order and stops when the budget fits. What it cannot do is what a naive
composition would: apply the intermediate plan and run the next policy against the resulting view.
Applying requires summaries, and under :doc:`ADR-0052 <adr>` the summaries do not exist — this
package never produces one. So the chain composes over a **projection**: the view a plan *would*
produce, built from the plan alone.

**The projection, and why every part of it is derivable.**

* ``KEEP`` — the turn, unchanged.
* ``MASK`` — the same turn with the plan's :class:`~cutctx.types.TurnReplacement` as its content and
  estimate. Known exactly; the stub is in the plan.
* ``DROP`` — absent.
* ``SUMMARIZE`` — absent, and at the position of the group's **earliest** turn a stand-in appears,
  with the derived id ``summary:<group_id>``, role ``ASSISTANT``, and ``target_tokens`` as its
  estimate. Its *content* is the one thing that does not exist yet, so the stand-in carries none —
  and no shipped policy reads content, which is why that is survivable. A policy that did would be
  reading text a chain cannot give it, and this is the place that would have to change.

Every one of those is a function of the previous plan, so the projection is deterministic and the
composed plan is byte-identical on re-derivation (spec §11 contract 4).

**Ordering is transcript position, everywhere.** Actions are emitted by walking the transcript, the
projection is built by walking it, groups are keyed by the position of their earliest turn, and
requests are ordered by that position. Nothing is ordered by iterating a ``dict`` or a ``set``. The
development plan names *"nondeterminism via dict ordering in group assembly"* as this phase's likely
failure mode, and it is the kind that reproduces only under an unlucky ``pytest-randomly`` seed
unless the ordering is structural rather than incidental.

**One plan, built once, through the invariants.** Whatever the composition works out, the result
goes through :func:`cutctx._invariants.build_plan` against the **real** transcript, which recomputes
``tokens_before``, ``tokens_after_estimate`` and ``budget_unmet`` and refuses a plan that disagrees
with its own actions. ``budget_unmet`` is derived and never declared (`C1_HANDOFF.md` §3.2), and
``test_no_shipped_policy_builds_its_own_plan`` scans this directory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from baseaicore import ValidationError

from cutctx import _invariants
from cutctx.policies.drop_oldest import DropOldestPolicy
from cutctx.policies.masking import ObservationMaskingPolicy
from cutctx.policies.summarizing import SummarizingPolicy
from cutctx.types import (
    SUMMARY_TURN_ID_PREFIX,
    Action,
    Role,
    SummarizationRequest,
    Transcript,
    TranscriptTurn,
    TurnAction,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cutctx.types import CompactionBudget, CompactionPlan, CompactionPolicy, TurnReplacement

__all__ = ["PolicyChain", "default_chain"]

_VERSION: Final = "1.0.0"

_REMOVING: Final = frozenset({Action.DROP, Action.SUMMARIZE})
"""The two actions that take a turn out of the view. ``KEEP`` and ``MASK`` leave it in place."""


@dataclass(slots=True)
class _Resolution:
    """What the chain has decided about one real turn so far, before the plan is built."""

    action: Action
    replacement: TurnReplacement | None = None
    group: str | None = None


class PolicyChain:
    """Runs policies in order over a projection, stopping as soon as the budget fits.

    The composed plan's ``policy_name`` names every constituent and its version, because a report
    saying only "chain" would leave an auditor unable to tell which policies produced the view they
    are looking at. ``version`` is the chain's own composition rules, which is what this class
    would bump if the projection changed.

    Args:
        policies: The policies, in the order they run. At least one.

    Raises:
        ValidationError: If ``policies`` is empty or holds something that is not a policy. An empty
            chain would be a policy that plans nothing while claiming to have tried.
    """

    version: str = _VERSION

    def __init__(self, policies: Sequence[CompactionPolicy]) -> None:
        """Validate the composition and bind it."""
        members = tuple(policies)
        if not members:
            raise ValidationError(
                "A PolicyChain needs at least one policy. An empty chain returns a plan that "
                "changed nothing while reporting that a chain ran, which is worse than either.",
                details={},
            )
        for policy in members:
            if not callable(getattr(policy, "decide", None)) or not isinstance(
                getattr(policy, "name", None), str
            ):
                raise ValidationError(
                    "Every member of a PolicyChain must be a CompactionPolicy: a `name`, a "
                    "`version` and a `decide(transcript, budget)`.",
                    details={"member": type(policy).__name__},
                )
        self._policies = members
        self.name = "chain(" + "+".join(f"{p.name}@{p.version}" for p in members) + ")"

    @property
    def policies(self) -> tuple[CompactionPolicy, ...]:
        """The constituent policies, in the order they run."""
        return self._policies

    def decide(self, transcript: Transcript, budget: CompactionBudget) -> CompactionPlan:
        """Return the one plan the whole chain composes.

        Args:
            transcript: The transcript to compact.
            budget: The window it must fit, and the tail no policy may touch.

        Returns:
            The plan. ``budget_unmet`` is ``True`` when every policy has run and the view is still
            over budget — the chain refuses rather than truncating (spec §13's last row), and the
            caller decides.

        Raises:
            BudgetUnsatisfiable: If the untouchable turns alone exceed ``budget.max_tokens``.
                Raised once, before any policy runs.
        """
        _invariants.require_satisfiable_budget(transcript, budget)

        resolved: dict[str, _Resolution] = {
            turn.turn_id: _Resolution(Action.KEEP) for turn in transcript.turns
        }
        groups: dict[str, SummarizationRequest] = {}
        projection = transcript
        estimator_ratio: float | None = None

        for policy in self._policies:
            if projection.token_estimate() <= budget.max_tokens:
                break
            plan = policy.decide(projection, budget)
            if estimator_ratio is None:
                estimator_ratio = plan.estimator_ratio
            self._absorb(plan, transcript, resolved, groups)
            projection = _project(transcript, resolved, groups)

        _reconcile_exchanges(resolved, transcript, groups)
        actions = tuple(
            TurnAction(
                turn_id=turn.turn_id,
                action=resolved[turn.turn_id].action,
                summary_group=resolved[turn.turn_id].group,
                replacement=resolved[turn.turn_id].replacement,
            )
            for turn in transcript.turns
        )
        return _invariants.build_plan(
            transcript=transcript,
            budget=budget,
            actions=actions,
            summarization_requests=_ordered_requests(groups, transcript),
            policy_name=self.name,
            policy_version=self.version,
            estimator_ratio=estimator_ratio,
        )

    @staticmethod
    def _absorb(
        plan: CompactionPlan,
        transcript: Transcript,
        resolved: dict[str, _Resolution],
        groups: dict[str, SummarizationRequest],
    ) -> None:
        """Fold one plan-over-the-projection back onto the real transcript's turns.

        A projected turn is either a real turn under its own id — the ids are preserved through
        ``KEEP`` and ``MASK`` — or a summary stand-in named ``summary:<group_id>``. The first kind
        maps to itself. The second maps to **every real turn already folded into that group**,
        which is the escalation the composition turns on: a later policy that drops a summary turn
        has decided to drop everything the earlier policy folded into it, and a later policy that
        summarizes one has decided to fold that whole group into a larger one.
        """
        for action in plan.actions:
            members = _members_of(action.turn_id, groups)
            for turn_id in members:
                current = resolved[turn_id]
                if action.action is Action.KEEP:
                    continue
                if action.action is Action.MASK:
                    current.action = Action.MASK
                    current.replacement = action.replacement
                    current.group = None
                elif action.action is Action.DROP:
                    current.action = Action.DROP
                    current.replacement = None
                    current.group = None
                else:
                    current.action = Action.SUMMARIZE
                    current.replacement = None
                    current.group = action.summary_group
            if action.action in _REMOVING and action.turn_id.startswith(SUMMARY_TURN_ID_PREFIX):
                # The group this stand-in represented has been superseded; its request goes.
                groups.pop(action.turn_id[len(SUMMARY_TURN_ID_PREFIX) :], None)

        # Ordered by the plan's own request order, which `validate_plan` already pinned to
        # transcript position — never by iterating a mapping.
        order = {turn.turn_id: index for index, turn in enumerate(transcript.turns)}
        for request in plan.summarization_requests:
            covered = sorted(
                (turn_id for turn_id, state in resolved.items() if state.group == request.group_id),
                key=lambda turn_id: order[turn_id],
            )
            groups[request.group_id] = SummarizationRequest(
                group_id=request.group_id,
                turn_ids=tuple(covered),
                target_tokens=request.target_tokens,
                prompt_id=request.prompt_id,
            )


def _members_of(projected_id: str, groups: Mapping[str, SummarizationRequest]) -> tuple[str, ...]:
    """Return the real turn ids one projected turn stands for."""
    if projected_id.startswith(SUMMARY_TURN_ID_PREFIX):
        group_id = projected_id[len(SUMMARY_TURN_ID_PREFIX) :]
        request = groups.get(group_id)
        return () if request is None else request.turn_ids
    return (projected_id,)


def _project(
    transcript: Transcript,
    resolved: Mapping[str, _Resolution],
    groups: Mapping[str, SummarizationRequest],
) -> Transcript:
    """Build the view the decisions so far would produce, walking the transcript in order.

    The summary stand-in appears at the position of its group's earliest turn — the same position
    :meth:`~cutctx.executor.CompactionExecutor.apply` puts the real summary turn — so a policy
    reading positions sees the view it will actually get.
    """
    turns: list[TranscriptTurn] = []
    emitted: set[str] = set()
    for turn in transcript.turns:
        state = resolved[turn.turn_id]
        if state.action is Action.KEEP:
            turns.append(turn)
        elif state.action is Action.MASK and state.replacement is not None:
            turns.append(
                replace(
                    turn,
                    content=state.replacement.content,
                    token_estimate=state.replacement.token_estimate,
                )
            )
        elif state.action is Action.SUMMARIZE and state.group is not None:
            if state.group in emitted:
                continue
            emitted.add(state.group)
            request = groups[state.group]
            turns.append(
                TranscriptTurn(
                    turn_id=request.summary_turn_id,
                    role=Role.ASSISTANT,
                    content="",
                    token_estimate=request.target_tokens,
                )
            )
    return Transcript(turns=tuple(turns))


def _reconcile_exchanges(
    resolved: dict[str, _Resolution],
    transcript: Transcript,
    groups: dict[str, SummarizationRequest],
) -> None:
    """Escalate any exchange that composition left half-removed, and refresh the groups.

    Spec §11 contract 3 as `C1_HANDOFF.md` §4 reads it: within one exchange, either every member is
    retained (``KEEP``/``MASK``, mixed freely) or every member is removed by the **same** action.
    Composition could reach a plan that breaks it — a later policy dropping a call whose result an
    earlier one masked — and the answer is to escalate the whole exchange to the removing action
    here, rather than to hand ``build_plan`` a plan it will reject and leave the caller with an
    exception where a compaction was asked for.

    ``DROP`` wins over ``SUMMARIZE``: a member already gone cannot be folded into a summary, and a
    summary of the rest would describe an exchange the view no longer contains. Where several
    members are summarized into different groups, the group of the **earliest** member wins, by
    transcript position rather than by whatever order a mapping happened to yield.

    With the projection as it is built today this is defence in depth: each policy runs against a
    projection that preserves every retained turn's id and ``tool_call_id``, so its own
    ``build_plan`` already refuses a split on its own view. It is run unconditionally anyway,
    because that argument is a property of the projection rather than of the contract — anyone
    changing the projection changes whether it holds, and a guarantee that depends on an
    invariant two modules away is one worth enforcing where it is stated.
    """
    order = {turn.turn_id: index for index, turn in enumerate(transcript.turns)}
    for exchange in _invariants.exchanges(transcript):
        actions = {resolved[turn_id].action for turn_id in exchange}
        if not actions & _REMOVING:
            continue  # every member retained: KEEP and MASK mix freely.
        if Action.DROP in actions:
            if actions != {Action.DROP}:
                for turn_id in exchange:
                    resolved[turn_id] = _Resolution(Action.DROP)
            continue
        summarized = [turn_id for turn_id in exchange if resolved[turn_id].group is not None]
        distinct = {resolved[turn_id].group for turn_id in summarized}
        if actions == {Action.SUMMARIZE} and len(distinct) == 1:
            continue  # already one group, whole.
        winner = resolved[min(summarized, key=lambda turn_id: order[turn_id])].group
        for turn_id in exchange:
            resolved[turn_id] = _Resolution(Action.SUMMARIZE, group=winner)

    for group_id in list(groups):
        covered = sorted(
            (turn_id for turn_id, state in resolved.items() if state.group == group_id),
            key=lambda turn_id: order[turn_id],
        )
        if not covered:
            del groups[group_id]
            continue
        groups[group_id] = SummarizationRequest(
            group_id=group_id,
            turn_ids=tuple(covered),
            target_tokens=groups[group_id].target_tokens,
            prompt_id=groups[group_id].prompt_id,
        )


def _ordered_requests(
    groups: Mapping[str, SummarizationRequest], transcript: Transcript
) -> tuple[SummarizationRequest, ...]:
    """Return the requests ordered by the position of each group's earliest turn.

    ``_invariants.validate_plan`` requires exactly this order and says why: the order is part of
    the plan's bytes, and a plan whose order came from a mapping's insertion history is not
    byte-identical on re-derivation.
    """
    order = {turn.turn_id: index for index, turn in enumerate(transcript.turns)}
    return tuple(sorted(groups.values(), key=lambda request: order[request.turn_ids[0]]))


def default_chain(
    *,
    prompt_id: str,
    keep_recent_results: int = 2,
    target_ratio: float = 0.2,
    min_span_turns: int = 4,
) -> PolicyChain:
    """Build the chain the development plan calls the default: mask, then summarize, then drop.

    The order is an argument about cost. Masking is free and reversible in the sense that
    matters — the original is in the caller's store and the stub names its digest — so it goes
    first. Summarizing costs a model call but keeps the *substance* of what it folds, so it is
    second. Dropping keeps nothing at all, so it is the last resort, which is what
    `DropOldestPolicy` calls itself.

    Args:
        prompt_id: The versioned prompt record the summarizing step names (ADR-0012).
        keep_recent_results: Passed to :class:`~cutctx.policies.masking.ObservationMaskingPolicy`.
        target_ratio: Passed to :class:`~cutctx.policies.summarizing.SummarizingPolicy`.
        min_span_turns: Passed to :class:`~cutctx.policies.summarizing.SummarizingPolicy`.

    Returns:
        The chain. It is an ordinary :class:`PolicyChain`, so a caller who wants a different order
        or different members builds one directly — this is a convenience with a documented opinion,
        not a special case.
    """
    return PolicyChain(
        (
            ObservationMaskingPolicy(keep_recent_results=keep_recent_results),
            SummarizingPolicy(
                prompt_id=prompt_id, target_ratio=target_ratio, min_span_turns=min_span_turns
            ),
            DropOldestPolicy(),
        )
    )
