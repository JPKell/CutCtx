"""The untouchable set and the tool exchanges, **restated from the spec** rather than imported.

A property that asks the implementation what the untouchable set is, and then checks that the plan
did not touch it, tests self-consistency and not correctness: change
``index >= protected_from`` to ``index > protected_from`` in
:func:`cutctx._invariants.untouchable_turn_ids` and every such property still passes, because the
oracle moved with the bug. Two deliberately-introduced mutants survived the first draft of this
suite for exactly that reason.

So the oracles below are written from spec §11's wording, in a different shape from the module they
check — a slice for the tail rather than an index comparison, a dictionary keyed on the correlation
id rather than an ordered partition. They are allowed to be slower and clumsier than the real thing.
They are not allowed to share its code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cutctx import CompactionBudget, Role

if TYPE_CHECKING:
    from cutctx import Transcript

__all__ = [
    "untouchable_ids",
    "untouchable_total",
    "exchange_sets",
    "maskable_ids",
    "exchange_rule_holds",
]


def untouchable_ids(transcript: Transcript, budget: CompactionBudget) -> set[str]:
    """Return the turns spec §11 contract 2 forbids masking, summarizing or dropping.

    "The system turn and pinned turns are untouchable, and the ``protected_recent_turns`` tail is
    never masked, summarized or dropped."
    """
    turns = transcript.turns
    tail_starts_at = max(0, len(turns) - budget.protected_recent_turns)
    tail = {turn.turn_id for turn in turns[tail_starts_at:]}
    system = {turn.turn_id for turn in turns if turn.role is Role.SYSTEM}
    pinned = {turn.turn_id for turn in turns if turn.pinned}
    return tail | system | pinned


def untouchable_total(transcript: Transcript, protected_recent_turns: int) -> int:
    """Return the estimated tokens of the untouchable set — the floor no policy can go below."""
    budget = CompactionBudget(0, protected_recent_turns=protected_recent_turns)
    untouchable = untouchable_ids(transcript, budget)
    return sum(turn.token_estimate for turn in transcript.turns if turn.turn_id in untouchable)


def exchange_sets(transcript: Transcript) -> list[set[str]]:
    """Return the tool exchanges as sets: turns sharing a correlation id, and singletons.

    "A tool call and its result travel together … never separated." Built as a dictionary keyed on
    ``tool_call_id`` with no notion of position at all, so an implementation that quietly assumed
    a call and its results are adjacent cannot hide behind this oracle.
    """
    by_call: dict[str, set[str]] = {}
    singletons: list[set[str]] = []
    for turn in transcript.turns:
        if turn.tool_call_id is None:
            singletons.append({turn.turn_id})
        else:
            by_call.setdefault(turn.tool_call_id, set()).add(turn.turn_id)
    return singletons + list(by_call.values())


def maskable_ids(
    transcript: Transcript, budget: CompactionBudget, keep_recent_results: int
) -> set[str]:
    """Return the turns spec §7 says ``ObservationMaskingPolicy`` may mask.

    "masks TOOL-result bodies beyond the N most recent" — so: a ``TOOL`` turn, outside the
    untouchable set (contract 2), and not among the last ``N`` ``TOOL`` turns of the transcript.

    Restated from the spec, in a different shape from the policy: the recent set is built by
    counting *backwards* through the transcript rather than by slicing a list of results, so a
    policy that sliced wrongly — ``results[-0:]`` is the whole list, not none of it — cannot hide
    behind this oracle.
    """
    untouchable = untouchable_ids(transcript, budget)
    recent: set[str] = set()
    for turn in reversed(transcript.turns):
        if len(recent) >= keep_recent_results:
            break
        if turn.role is Role.TOOL:
            recent.add(turn.turn_id)
    return {
        turn.turn_id
        for turn in transcript.turns
        if turn.role is Role.TOOL and turn.turn_id not in untouchable and turn.turn_id not in recent
    }


def exchange_rule_holds(actions: dict[str, str], transcript: Transcript) -> bool:
    """Return whether every exchange is wholly retained or wholly removed by one action.

    Spec §11 contract 3 as `C1_HANDOFF.md` §4 reads it, restated:

        Within one exchange, either every member is retained (``keep``/``mask``, mixed freely), or
        every member is removed by the **same** action — all ``drop``, or all ``summarize`` into
        the same group.

    Args:
        actions: turn id → ``"keep"``, ``"mask"``, ``"drop"``, or ``"summarize:<group>"``.
        transcript: The transcript the actions are over.

    Returns:
        Whether the rule holds for every exchange.
    """
    retained = {"keep", "mask"}
    for exchange in exchange_sets(transcript):
        verdicts = {actions[turn_id] for turn_id in exchange}
        if verdicts <= retained:
            continue
        if len(verdicts) != 1:
            return False
    return True
