"""The rules of spec §11, in one place, enforced by validation and not by convention.

**Why this module is private and still the most public thing in the package.** Every policy
CutCtx will ever ship routes through it: row E1 adds observation masking, summarization and
``PolicyChain`` against it, PromptCadence wires it in at row I1, and IdeaPress's prose reduction
order becomes a chain over it at row J3. It is where a rule is written once, so that the fourth
policy cannot get it subtly different from the first.

The invariants
--------------

1. **A plan is a total function over its transcript.** ``actions`` names every turn exactly once,
   in transcript order. A turn a plan forgot is the silent bug: it would be neither kept nor
   dropped, and what happened to it would depend on which loop read the plan.
2. **The untouchable set is untouched** (contract 2). Every ``SYSTEM`` turn, every ``pinned``
   turn, and the last ``protected_recent_turns`` turns take :attr:`~cutctx.types.Action.KEEP` —
   not ``MASK``, not ``SUMMARIZE``, not ``DROP``.
3. **A tool exchange travels together** (contract 3). See :func:`exchanges`.
4. **A summary group and its request agree** exactly, in both directions.
5. **The arithmetic follows from the actions.** ``tokens_before`` and ``tokens_after_estimate``
   are recomputed here and compared, so a plan cannot carry a number its own actions do not
   produce, and a report that copies them cannot drift from a plan that earned them.
6. **The budget outcome is trichotomous and honest.** Either the untouchable set alone exceeds
   the budget and :class:`~cutctx.errors.BudgetUnsatisfiable` is raised before any plan exists, or
   a plan exists and ``budget_unmet`` is exactly ``tokens_after_estimate > max_tokens``. There is
   no fourth case, and no plan that claims to fit while being over.

How contract 3 is read
----------------------

Contract 3 says a tool call and its result are "masked together, summarized in the same group, or
dropped as a pair — never separated, because an orphaned call or result is a malformed transcript
to every provider". The operative prohibition is *separation*, and what separates is **removal**:
:attr:`~cutctx.types.Action.KEEP` and :attr:`~cutctx.types.Action.MASK` leave the turn in the view
at its own position, so masking a tool result beside a kept call orphans nothing.

Read the other way it would forbid the masking policy the spec itself ships — "masks TOOL-result
bodies beyond the N most recent … reasoning stays, bulk goes" (spec §7) masks a result while
keeping the assistant turn that called it. So the rule enforced here is:

    Within one exchange, either every member is retained (``KEEP``/``MASK``, mixed freely), or
    every member is removed by the **same** action — all ``DROP``, or all ``SUMMARIZE`` into the
    **same** group.

That is stricter than "never orphaned" by one step (it also forbids dropping half an exchange and
summarizing the other half, which orphans nothing but describes nothing either), and it is exactly
the three cases the contract enumerates.

Protection propagates through an exchange
-----------------------------------------

A consequence worth stating, because it is not obvious and it is load-bearing: if **any** member
of an exchange is untouchable, no member may be removed. A tool result sitting in the protected
tail whose call is forty turns back makes that call undroppable — the pair would be separated
otherwise. :func:`removable_turn_ids` is the set that survives this closure, and it is what a
policy may act on.

The closure does **not** move the :class:`~cutctx.errors.BudgetUnsatisfiable` threshold. That
threshold is the strict untouchable set of contract 2, whose turns can never be reduced by any
means; a turn locked only by the closure is still maskable, so refusing on its account would
refuse budgets that row E1's policies can meet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from baseaicore import ValidationError

from cutctx.errors import BudgetUnsatisfiable
from cutctx.types import Action, CompactionPlan, Role

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from cutctx.types import CompactionBudget, SummarizationRequest, Transcript, TurnAction

__all__ = [
    "build_plan",
    "exchanges",
    "removable_turn_ids",
    "require_satisfiable_budget",
    "untouchable_tokens",
    "untouchable_turn_ids",
    "validate_plan",
]

_REMOVAL_ACTIONS = frozenset({Action.DROP, Action.SUMMARIZE})
"""The actions that take a turn out of the view, and so are the ones that can orphan a pair."""

_ID_SAMPLE_LIMIT = 10
"""How many ids an error's ``details`` carries. A 2 000-turn mismatch is not made clearer by
2 000 ids, and ``details`` travels into API error envelopes."""


def exchanges(transcript: Transcript) -> tuple[tuple[str, ...], ...]:
    """Return the tool exchanges of a transcript: the units that must travel together.

    An **exchange** is every turn sharing one ``tool_call_id`` — the assistant turn that issued
    the calls and every ``TOOL`` turn carrying a result, however far apart they sit. A turn with
    no ``tool_call_id`` is an exchange of one, and so is a result whose call is already gone.

    ``tool_call_id`` is a correlation id rather than a per-call one, which is what makes this a
    partition rather than a graph: one assistant turn may issue several calls, so the call/result
    relation is not one-to-one, and the transitive closure of a per-call relation would be this
    same partition anyway (see :class:`~cutctx.types.TranscriptTurn`, "Multi-call turns"). A
    partition needs no traversal order, and so has nothing for determinism to pin down.

    Args:
        transcript: The transcript to partition.

    Returns:
        The exchanges, each a tuple of turn ids in transcript order, ordered by the position of
        each exchange's earliest turn. Every turn id appears in exactly one exchange.
    """
    members: dict[str, list[str]] = {}
    order: list[str] = []
    for turn in transcript.turns:
        # Namespaced so that a tool_call_id equal to some other turn's id cannot merge two
        # exchanges that share nothing.
        key = (
            f"c\x00{turn.tool_call_id}" if turn.tool_call_id is not None else f"t\x00{turn.turn_id}"
        )
        if key not in members:
            members[key] = []
            order.append(key)
        members[key].append(turn.turn_id)
    return tuple(tuple(members[key]) for key in order)


def untouchable_turn_ids(transcript: Transcript, budget: CompactionBudget) -> frozenset[str]:
    """Return the turns contract 2 forbids masking, summarizing or dropping.

    Three sources, unioned: every :attr:`~cutctx.types.Role.SYSTEM` turn, every ``pinned`` turn,
    and the last ``budget.protected_recent_turns`` turns of the transcript.

    Every ``SYSTEM`` turn, not "the" system turn: the spec's singular describes the usual case,
    and a transcript that carries two — an application that appends an instruction mid-run — has
    two turns whose loss would change what the model was told. Protecting both is the reading that
    cannot silently discard instructions.

    Args:
        transcript: The transcript.
        budget: The budget, for its ``protected_recent_turns``.

    Returns:
        The untouchable turn ids. Possibly the whole transcript; possibly empty.
    """
    protected_from = len(transcript.turns) - budget.protected_recent_turns
    return frozenset(
        turn.turn_id
        for index, turn in enumerate(transcript.turns)
        if turn.role is Role.SYSTEM or turn.pinned or index >= protected_from
    )


def untouchable_tokens(transcript: Transcript, budget: CompactionBudget) -> int:
    """Return the estimated tokens of the untouchable set — the floor no policy can go below."""
    untouchable = untouchable_turn_ids(transcript, budget)
    return sum(turn.token_estimate for turn in transcript.turns if turn.turn_id in untouchable)


def removable_turn_ids(transcript: Transcript, budget: CompactionBudget) -> frozenset[str]:
    """Return the turns a policy may remove — the untouchable set closed over tool exchanges.

    A turn is removable when its whole exchange is free of untouchable turns. Removing any other
    turn would either touch the untouchable set (contract 2) or separate a pair (contract 3).

    This is the set every policy should work from. Masking is *not* limited by it — a maskable
    turn is any turn outside :func:`untouchable_turn_ids`, since masking removes nothing.

    Args:
        transcript: The transcript.
        budget: The budget, for its ``protected_recent_turns``.

    Returns:
        The removable turn ids.
    """
    untouchable = untouchable_turn_ids(transcript, budget)
    removable: set[str] = set()
    for exchange in exchanges(transcript):
        if not untouchable.intersection(exchange):
            removable.update(exchange)
    return frozenset(removable)


def require_satisfiable_budget(transcript: Transcript, budget: CompactionBudget) -> None:
    """Refuse a budget smaller than the turns no policy is allowed to reduce.

    Called before a policy does any work, and again when the plan is constructed, so a policy that
    forgets it still cannot produce a plan for a contradictory budget.

    Args:
        transcript: The transcript.
        budget: The budget.

    Raises:
        BudgetUnsatisfiable: If the untouchable turns' combined estimate exceeds
            ``budget.max_tokens``. ``details`` names **both** numbers and the count of turns
            behind the first, because "budget too small" without the figures leaves an operator
            unable to choose between raising the budget and unpinning something. Equality is
            satisfiable: a budget that exactly fits the untouchable set admits the plan that keeps
            it and drops everything else.
    """
    floor = untouchable_tokens(transcript, budget)
    if floor > budget.max_tokens:
        untouchable = untouchable_turn_ids(transcript, budget)
        raise BudgetUnsatisfiable(
            f"The untouchable turns alone are estimated at {floor} tokens, which exceeds the "
            f"budget of {budget.max_tokens}. {len(untouchable)} turns are untouchable: every "
            f"SYSTEM turn, every pinned turn, and the last {budget.protected_recent_turns}. "
            "Raise max_tokens, unpin turns, or shorten the protected tail — no compaction can "
            "fit this budget without violating an invariant.",
            details={
                "untouchable_tokens": floor,
                "max_tokens": budget.max_tokens,
                "untouchable_turn_count": len(untouchable),
                "protected_recent_turns": budget.protected_recent_turns,
            },
        )


def estimate_after(
    transcript: Transcript,
    actions: Sequence[TurnAction],
    summarization_requests: Sequence[SummarizationRequest],
) -> int:
    """Return the estimated tokens of the view these actions would produce.

    The single computation of the "after" figure. The plan carries its result, the report copies
    the plan's, and the executor builds a view whose per-turn estimates add up to the same number
    — one derivation, so the drift the development plan names as this phase's likely failure mode
    has nowhere to happen.

    Per action: ``KEEP`` contributes the turn's own estimate, ``MASK`` its replacement's,
    ``DROP`` and ``SUMMARIZE`` nothing. Each summarization request contributes its
    ``target_tokens`` once, for the one summary turn it becomes.

    Args:
        transcript: The transcript planned over.
        actions: One action per turn.
        summarization_requests: The plan's requests.

    Returns:
        The estimated token cost of the applied view.
    """
    by_id = {turn.turn_id: turn for turn in transcript.turns}
    total = 0
    for action in actions:
        if action.action is Action.KEEP:
            total += by_id[action.turn_id].token_estimate
        elif action.replacement is not None:
            total += action.replacement.token_estimate
    return total + sum(request.target_tokens for request in summarization_requests)


def build_plan(
    *,
    transcript: Transcript,
    budget: CompactionBudget,
    actions: Sequence[TurnAction],
    summarization_requests: Sequence[SummarizationRequest] = (),
    policy_name: str,
    policy_version: str,
    estimator_ratio: float | None = None,
) -> CompactionPlan:
    """Build a validated plan from a policy's decisions — the way every policy makes a plan.

    Does the two things no policy should do for itself: the token arithmetic
    (:func:`estimate_after`) and the ``budget_unmet`` determination. A policy that constructs a
    :class:`~cutctx.types.CompactionPlan` directly is validated identically — validation is on the
    constructor, not here — but it would be writing figures this function derives and the
    validator immediately recomputes, so the only thing direct construction can achieve is being
    rejected.

    Args:
        transcript: The transcript planned over.
        budget: The budget planned against.
        actions: One action per turn, in transcript order.
        summarization_requests: The requests the caller must fulfil, ordered by the position of
            each group's earliest turn.
        policy_name: The deciding policy's name.
        policy_version: The deciding policy's version.
        estimator_ratio: The character-ratio default's ``chars_per_token`` when it produced an
            estimate on this plan; ``None`` otherwise.

    Returns:
        The validated plan.

    Raises:
        BudgetUnsatisfiable: If the untouchable turns alone exceed the budget.
        ValidationError: If any invariant of spec §11 is broken.
    """
    after = estimate_after(transcript, actions, summarization_requests)
    return CompactionPlan(
        actions=tuple(actions),
        summarization_requests=tuple(summarization_requests),
        tokens_before=transcript.token_estimate(),
        tokens_after_estimate=after,
        policy_name=policy_name,
        policy_version=policy_version,
        estimator_ratio=estimator_ratio,
        budget_unmet=after > budget.max_tokens,
        transcript=transcript,
        budget=budget,
    )


def validate_plan(plan: CompactionPlan, transcript: Transcript, budget: CompactionBudget) -> None:
    """Refuse a plan that breaks any rule in this module's list.

    Called from :meth:`cutctx.types.CompactionPlan.__post_init__`, which is why there is no
    unvalidated plan anywhere: the transcript and budget are constructor arguments, so a caller
    cannot reach a plan object without handing over what it takes to check it.

    Args:
        plan: The plan under construction.
        transcript: The transcript it is a plan for.
        budget: The budget it is a plan against.

    Raises:
        BudgetUnsatisfiable: If the untouchable turns alone exceed the budget. Checked first, so
            an impossible budget is reported as impossible rather than as whatever the policy did
            about it.
        ValidationError: If the plan does not cover the transcript exactly once in order, acts on
            an untouchable turn, splits a tool exchange, disagrees with its own summarization
            requests, or carries arithmetic its actions do not produce.
    """
    require_satisfiable_budget(transcript, budget)
    _validate_coverage(plan.actions, transcript)
    by_turn = {action.turn_id: action for action in plan.actions}
    _validate_untouchable(by_turn, transcript, budget)
    _validate_exchanges(by_turn, transcript)
    _validate_summary_groups(plan, transcript)
    _validate_arithmetic(plan, transcript, budget)


def _validate_coverage(actions: Sequence[TurnAction], transcript: Transcript) -> None:
    """Rule 1: the actions name every turn exactly once, in transcript order."""
    planned = tuple(action.turn_id for action in actions)
    expected = transcript.turn_ids()
    if planned == expected:
        return
    unplanned = sorted(set(expected) - set(planned))
    unknown = sorted(set(planned) - set(expected))
    repeated = sorted({turn_id for turn_id in planned if planned.count(turn_id) > 1})
    raise ValidationError(
        "A plan must name every turn of its transcript exactly once, in transcript order; this "
        f"one names {len(planned)} actions for {len(expected)} turns "
        f"({len(unplanned)} unplanned, {len(unknown)} unknown, {len(repeated)} repeated"
        + (", order differs)" if not (unplanned or unknown or repeated) else ")")
        + ". A turn the plan forgot would be neither kept nor dropped.",
        details={
            # `unplanned`: in the transcript, forgotten by the plan. `unknown`: named by the plan,
            # absent from the transcript. Two different defects with two different fixes.
            "unplanned_turn_ids": unplanned[:_ID_SAMPLE_LIMIT],
            "unknown_turn_ids": unknown[:_ID_SAMPLE_LIMIT],
            "repeated_turn_ids": repeated[:_ID_SAMPLE_LIMIT],
        },
    )


def _validate_untouchable(
    by_turn: Mapping[str, TurnAction], transcript: Transcript, budget: CompactionBudget
) -> None:
    """Rule 2: every untouchable turn is kept, whole and in place."""
    touched = sorted(
        turn_id
        for turn_id in untouchable_turn_ids(transcript, budget)
        if by_turn[turn_id].action is not Action.KEEP
    )
    if touched:
        raise ValidationError(
            f"{len(touched)} untouchable turns are acted on: a SYSTEM turn, a pinned turn and "
            f"any of the last {budget.protected_recent_turns} turns must be KEEP (spec §11 "
            "contract 2). A budget that cannot be met without touching them raises "
            "BudgetUnsatisfiable instead.",
            details={
                "touched_turn_ids": touched[:_ID_SAMPLE_LIMIT],
                "actions": [by_turn[t].action.value for t in touched[:_ID_SAMPLE_LIMIT]],
                "protected_recent_turns": budget.protected_recent_turns,
            },
        )


def _validate_exchanges(by_turn: Mapping[str, TurnAction], transcript: Transcript) -> None:
    """Rule 3: within an exchange, all retained, or all removed by one identical action."""
    for exchange in exchanges(transcript):
        removed = [t for t in exchange if by_turn[t].action in _REMOVAL_ACTIONS]
        if not removed:
            continue
        if len(removed) != len(exchange):
            retained = [t for t in exchange if t not in set(removed)]
            raise ValidationError(
                f"A tool exchange of {len(exchange)} turns is split: {len(removed)} removed, "
                f"{len(retained)} retained. An orphaned call or result is a malformed transcript "
                "to every provider (spec §11 contract 3), so an exchange is removed whole or not "
                "at all.",
                details={
                    "removed_turn_ids": sorted(removed)[:_ID_SAMPLE_LIMIT],
                    "retained_turn_ids": sorted(retained)[:_ID_SAMPLE_LIMIT],
                },
            )
        signatures = {(by_turn[t].action, by_turn[t].summary_group) for t in exchange}
        if len(signatures) != 1:
            raise ValidationError(
                f"A tool exchange of {len(exchange)} turns is removed by more than one action or "
                "into more than one summary group. Contract 3's three cases are 'masked together, "
                "summarized in the same group, or dropped as a pair' — a half-dropped, "
                "half-summarized exchange is none of them.",
                details={
                    "turn_ids": sorted(exchange)[:_ID_SAMPLE_LIMIT],
                    "actions": sorted({by_turn[t].action.value for t in exchange}),
                    "summary_groups": sorted({by_turn[t].summary_group or "" for t in exchange})[
                        :_ID_SAMPLE_LIMIT
                    ],
                },
            )


def _validate_summary_groups(plan: CompactionPlan, transcript: Transcript) -> None:
    """Rule 4: every group named by an action has a request, covering exactly those turns."""
    requests = plan.summarization_requests
    group_ids = [request.group_id for request in requests]
    if len(set(group_ids)) != len(group_ids):
        raise ValidationError(
            "A plan's summarization requests must have distinct group_ids; a repeat would make "
            "the summaries mapping ambiguous at apply time.",
            details={"group_ids": sorted(group_ids)[:_ID_SAMPLE_LIMIT]},
        )

    planned_groups: dict[str, list[str]] = {}
    for action in plan.actions:
        if action.summary_group is not None:
            planned_groups.setdefault(action.summary_group, []).append(action.turn_id)

    if set(planned_groups) != set(group_ids):
        raise ValidationError(
            "Every summary group named by an action needs a SummarizationRequest, and every "
            "request needs turns; the two sets disagree. A group without a request could not be "
            "fulfilled; a request without turns would spend a model call to fold nothing.",
            details={
                "groups_without_request": sorted(set(planned_groups) - set(group_ids))[
                    :_ID_SAMPLE_LIMIT
                ],
                "requests_without_turns": sorted(set(group_ids) - set(planned_groups))[
                    :_ID_SAMPLE_LIMIT
                ],
            },
        )

    for request in requests:
        if tuple(planned_groups[request.group_id]) != request.turn_ids:
            raise ValidationError(
                f"Summarization request {request.group_id!r} covers turns the plan's actions do "
                "not, or covers them in another order. The request and the actions are two "
                "statements of one fact and must agree exactly, in transcript order.",
                details={
                    "group_id": request.group_id,
                    "request_turn_ids": list(request.turn_ids)[:_ID_SAMPLE_LIMIT],
                    "action_turn_ids": planned_groups[request.group_id][:_ID_SAMPLE_LIMIT],
                },
            )

    _validate_summary_turn_ids(requests, transcript.turn_ids())

    order = {turn_id: index for index, turn_id in enumerate(transcript.turn_ids())}
    positions = [order[request.turn_ids[0]] for request in requests]
    if positions != sorted(positions):
        raise ValidationError(
            "Summarization requests must be ordered by the position of each group's earliest "
            "turn. Their order is part of the plan's bytes, and a plan whose order came from a "
            "dict's insertion history is not byte-identical on re-derivation (contract 4).",
            details={"group_ids": list(group_ids)[:_ID_SAMPLE_LIMIT]},
        )


def _validate_summary_turn_ids(
    requests: Iterable[SummarizationRequest], existing: Iterable[str]
) -> None:
    """Rule 4, continued: a summary turn's derived id must be free."""
    taken = set(existing)
    collisions = sorted(
        request.group_id for request in requests if request.summary_turn_id in taken
    )
    if collisions:
        raise ValidationError(
            f"{len(collisions)} summarization groups would produce a summary turn whose id "
            "already belongs to a turn in the transcript. The summary turn's id is derived from "
            "the group id rather than generated, because a counter or a random source would make "
            "two applications of one plan differ (contract 4) — so the collision is refused here "
            "instead.",
            details={"group_ids": collisions[:_ID_SAMPLE_LIMIT]},
        )


def _validate_arithmetic(
    plan: CompactionPlan, transcript: Transcript, budget: CompactionBudget
) -> None:
    """Rules 5 and 6: the figures follow from the actions, and the verdict follows from them."""
    before = transcript.token_estimate()
    if plan.tokens_before != before:
        raise ValidationError(
            f"tokens_before is {plan.tokens_before} but the transcript's turns are estimated at "
            f"{before}. The 'before' figure is the transcript's own sum, not a policy's opinion.",
            details={"declared": plan.tokens_before, "transcript": before},
        )
    after = estimate_after(transcript, plan.actions, plan.summarization_requests)
    if plan.tokens_after_estimate != after:
        raise ValidationError(
            f"tokens_after_estimate is {plan.tokens_after_estimate} but the plan's own actions "
            f"produce a view estimated at {after}. A report copies these figures from the plan, "
            "so a plan allowed to carry a figure its actions do not produce is a report that "
            "silently misstates what the model was shown.",
            details={"declared": plan.tokens_after_estimate, "derived": after},
        )
    unmet = after > budget.max_tokens
    if plan.budget_unmet != unmet:
        raise ValidationError(
            f"budget_unmet is {plan.budget_unmet} but the plan's estimate of {after} tokens "
            f"against a budget of {budget.max_tokens} makes it {unmet}. The flag is derived, not "
            "declared: a plan that could claim to fit while being over is exactly the silent "
            "truncation ADR-0023 refuses.",
            details={
                "declared": plan.budget_unmet,
                "tokens_after_estimate": after,
                "max_tokens": budget.max_tokens,
            },
        )
