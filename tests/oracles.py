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

__all__ = ["untouchable_ids", "untouchable_total", "exchange_sets"]


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
