"""Generators for the invariant properties — designed before the properties, on purpose.

A ``Transcript`` strategy that only emits well-paired, well-ordered, modestly sized transcripts
passes everything and proves nothing. These build the shapes the implementation was *not* written
against:

* **multi-call assistant turns** — one call turn with two or three results sharing its correlation
  id, because the call/result relation is not one-to-one;
* **results far from their call** — a result displaced many turns later, so that a *recent*
  protected result can make an *old* call undroppable;
* **orphaned results** — a ``TOOL`` turn whose call is not in the transcript, as an earlier
  compaction round would leave;
* **runs of pinned turns, and entirely pinned transcripts**;
* **more than one ``SYSTEM`` turn, and none, and one that is not first**;
* **transcripts shorter than ``protected_recent_turns``, and empty ones**;
* **degenerate budgets** — zero, one, exactly the untouchable total, one below and one above it,
  exactly the transcript total, and one above that;
* **token estimates of zero and of millions**, so that "fits" and "cannot possibly fit" are both
  reachable and neither is the common case.

Everything is generated **by construction** rather than by filtering. Heavy filtering gives
`filter_too_much` health-check failures and useless shrinking; a constructive generator shrinks
towards the small transcript that still breaks the property, which is the example worth reading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hypothesis import strategies as st

from cutctx import (
    Action,
    CompactionBudget,
    CompactionPlan,
    Role,
    SummarizationRequest,
    Transcript,
    TranscriptTurn,
    TurnAction,
    TurnReplacement,
)
from cutctx._invariants import build_plan
from oracles import untouchable_ids, untouchable_total

if TYPE_CHECKING:
    from collections.abc import Sequence

MAX_BLOCKS = 9
"""Blocks, not turns: one block can be a four-turn tool exchange, so transcripts reach the high
teens. Large enough for a displaced result to be genuinely distant, small enough that a shrunk
counterexample fits on a screen."""

TOKEN_ESTIMATES = st.one_of(
    st.sampled_from([0, 1, 2, 7, 64, 1_000, 1_000_000]),
    st.integers(min_value=0, max_value=5_000),
)
"""Zero and a million are both in range: a zero-token turn makes "dropping reduces the total" false
and a million-token turn makes a budget unmeetable without any of the pins being involved."""

METADATA = st.dictionaries(
    st.sampled_from(["distance", "kind", "source"]),
    st.sampled_from(["1", "3", "note", "unit"]),
    max_size=3,
)
"""Caller-owned and opaque. Drawn so that "metadata never reaches a plan's bytes" is a claim the
determinism properties can actually falsify rather than one they trivially satisfy."""

PIN_MODES = st.sampled_from(["none", "some", "all"])
"""``all`` is not an edge case worth one example — it is the shape where every policy must produce
the identity plan or raise, so it gets a third of the draws."""


@st.composite
def transcripts(draw: st.DrawFn, *, max_blocks: int = MAX_BLOCKS) -> Transcript:
    """Draw a structurally varied transcript, oldest first.

    Args:
        draw: Hypothesis' draw function.
        max_blocks: How many blocks to draw at most; a block is one plain turn or one whole tool
            exchange.

    Returns:
        The transcript. May be empty; may be entirely pinned; may contain orphaned tool results.
    """
    block_count = draw(st.integers(min_value=0, max_value=max_blocks))
    pin_mode = draw(PIN_MODES)
    turns: list[TranscriptTurn] = []
    counter = 0

    for block in range(block_count):
        kind = draw(st.sampled_from(["plain", "plain", "exchange", "orphan"]))
        if kind == "plain":
            role = draw(st.sampled_from([Role.USER, Role.ASSISTANT, Role.ASSISTANT, Role.SYSTEM]))
            turns.append(_turn(draw, f"t{counter:02d}", role, pin_mode))
            counter += 1
        elif kind == "exchange":
            call_id = f"c{block:02d}"
            turns.append(_turn(draw, f"t{counter:02d}", Role.ASSISTANT, pin_mode, call_id))
            counter += 1
            for _ in range(draw(st.integers(min_value=1, max_value=3))):
                turns.append(_turn(draw, f"t{counter:02d}", Role.TOOL, pin_mode, call_id))
                counter += 1
        else:
            # A result whose call is not in the transcript: an earlier compaction round dropped
            # it, or the caller's window starts mid-exchange. It is an exchange of one.
            turns.append(_turn(draw, f"t{counter:02d}", Role.TOOL, pin_mode, f"gone{block:02d}"))
            counter += 1

    return Transcript(turns=tuple(_displace(draw, turns)))


def _turn(
    draw: st.DrawFn,
    turn_id: str,
    role: Role,
    pin_mode: str,
    tool_call_id: str | None = None,
) -> TranscriptTurn:
    """Draw one turn's variable parts."""
    pinned = pin_mode == "all" or (pin_mode == "some" and draw(st.booleans()))
    return TranscriptTurn(
        turn_id=turn_id,
        role=role,
        content=turn_id,
        token_estimate=draw(TOKEN_ESTIMATES),
        tool_call_id=tool_call_id,
        pinned=pinned,
        # Opaque to every policy (spec §4) and absent from a plan's bytes — which is precisely
        # what the "equivalent construction path" property is there to keep true.
        metadata=draw(METADATA),
    )


def _displace(draw: st.DrawFn, turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
    """Move some tool results later, so a result and its call can be far apart.

    Without this every exchange is contiguous, and the interesting case — a result inside the
    protected tail whose call is at the very start, making the call undroppable — is unreachable.
    """
    if len(turns) < 3:
        return turns
    moved = list(turns)
    for _ in range(draw(st.integers(min_value=0, max_value=2))):
        source = draw(st.integers(min_value=0, max_value=len(moved) - 1))
        if moved[source].role is not Role.TOOL:
            continue
        target = draw(st.integers(min_value=source, max_value=len(moved) - 1))
        turn = moved.pop(source)
        moved.insert(target, turn)
    return moved


@st.composite
def budgets(draw: st.DrawFn, transcript: Transcript) -> CompactionBudget:
    """Draw a budget for a specific transcript, aimed at that transcript's own boundaries.

    The interesting budgets are not uniform integers: they are the untouchable total, one either
    side of it, the transcript total, and zero. Those are drawn from the transcript rather than
    hoped for, which is why this takes the transcript as an argument.
    """
    # Weighted rather than uniform. A uniform draw over `0..len+3` protects the whole transcript
    # about as often as not, and combined with the deliberately-unsatisfiable boundaries below it
    # sent 63 % of examples down the `BudgetUnsatisfiable` path — where every property about a
    # *plan* returns early. The extremes are kept; they are just no longer the common case.
    protected = draw(
        st.one_of(
            st.just(0),
            st.integers(min_value=0, max_value=2),
            st.integers(min_value=0, max_value=len(transcript.turns) + 3),
        )
    )
    floor = untouchable_total(transcript, protected)
    total = transcript.token_estimate()
    boundaries: Sequence[int] = [
        # Unsatisfiable by construction: the refusal path, and its exact boundary.
        0,
        1,
        max(floor - 1, 0),
        # Satisfiable, at and around the two figures that decide the outcome.
        floor,
        floor,
        floor + 1,
        floor + 1,
        max(total - 1, 0),
        max(total - 1, 0),
        total,
        total,
        total + 1,
    ]
    max_tokens = draw(
        st.one_of(
            st.sampled_from(boundaries),
            st.integers(min_value=0, max_value=total + 10),
        )
    )
    return CompactionBudget(max_tokens=max_tokens, protected_recent_turns=protected)


@st.composite
def transcripts_and_budgets(draw: st.DrawFn) -> tuple[Transcript, CompactionBudget]:
    """Draw a transcript together with a budget aimed at its own boundaries."""
    transcript = draw(transcripts())
    return transcript, draw(budgets(transcript))


def ordered_exchanges(transcript: Transcript) -> list[tuple[str, ...]]:
    """Return the tool exchanges as transcript-ordered tuples, ordered by earliest member."""
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for turn in transcript.turns:
        key = turn.tool_call_id if turn.tool_call_id is not None else f"\x00{turn.turn_id}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(turn.turn_id)
    return [tuple(groups[key]) for key in order]


@st.composite
def satisfiable_budgets(draw: st.DrawFn, transcript: Transcript) -> CompactionBudget:
    """Draw a budget a plan can exist under — at or above the untouchable floor.

    The protected tail is drawn **short** here, unlike in :func:`budgets`. A uniform draw over
    ``0..len+3`` protects half the transcript on average and, combined with the all-pinned pin
    mode, left 92 % of drawn plans with no summarization group at all — a generator that reached
    the interesting action classes twice a run and called it coverage. The invariant properties
    still want the full range, which is why that strategy keeps it.
    """
    protected = draw(
        st.one_of(
            st.just(0),
            st.integers(min_value=0, max_value=2),
            st.integers(min_value=0, max_value=len(transcript.turns) + 3),
        )
    )
    floor = untouchable_total(transcript, protected)
    total = transcript.token_estimate()
    max_tokens = draw(
        st.one_of(
            st.sampled_from([floor, floor + 1, max(total, floor), max(total, floor) + 1]),
            st.integers(min_value=floor, max_value=floor + total + 10),
        )
    )
    return CompactionBudget(max_tokens=max_tokens, protected_recent_turns=protected)


@st.composite
def plans_over(draw: st.DrawFn, transcript: Transcript, budget: CompactionBudget) -> CompactionPlan:
    """Draw a **valid** plan over a transcript, using every action class.

    ``DropOldestPolicy`` only ever emits ``KEEP`` and ``DROP``, so a property suite built on it
    alone would leave the executor's masking and summarization paths, and the ``MASK`` branch of
    the token arithmetic, untested against arbitrary shapes — and those are exactly the paths row
    E1's policies will drive. This strategy stands in for the policies that do not exist yet: it
    decides per exchange, respecting the invariants by construction, and lets ``build_plan`` do
    the arithmetic.

    Summary groups may span several exchanges, because a real ``SummarizingPolicy`` folds a
    contiguous *span* rather than one exchange, and a group whose turns are not adjacent is the
    shape most likely to break request ordering.
    """
    untouchable = untouchable_ids(transcript, budget)
    decided: dict[str, TurnAction] = {}
    group_members: dict[str, list[str]] = {}
    group_count = 0

    for exchange in ordered_exchanges(transcript):
        if untouchable.intersection(exchange):
            for turn_id in exchange:
                decided[turn_id] = _keep_or_mask(draw, turn_id, forced_keep=turn_id in untouchable)
            continue
        kind = draw(st.sampled_from(["retain", "drop", "summarize", "summarize"]))
        if kind == "retain":
            for turn_id in exchange:
                decided[turn_id] = _keep_or_mask(draw, turn_id, forced_keep=False)
        elif kind == "drop":
            for turn_id in exchange:
                decided[turn_id] = TurnAction(turn_id=turn_id, action=Action.DROP)
        else:
            index = draw(st.integers(min_value=0, max_value=group_count))
            if index == group_count:
                group_count += 1
            group_id = f"g{index}"
            group_members.setdefault(group_id, [])
            for turn_id in exchange:
                decided[turn_id] = TurnAction(
                    turn_id=turn_id, action=Action.SUMMARIZE, summary_group=group_id
                )

    actions = tuple(decided[turn.turn_id] for turn in transcript.turns)
    # Collected from the actions in transcript order, which is the order the validator recomputes
    # them in; a strategy that built requests while iterating exchanges would order a displaced
    # group wrongly and would be testing its own bug.
    for action in actions:
        if action.summary_group is not None:
            group_members[action.summary_group].append(action.turn_id)
    requests = tuple(
        SummarizationRequest(
            group_id=group_id,
            turn_ids=tuple(turn_ids),
            target_tokens=draw(st.integers(min_value=0, max_value=500)),
            prompt_id="general.summarize",
        )
        for group_id, turn_ids in sorted(
            group_members.items(), key=lambda item: transcript.turn_ids().index(item[1][0])
        )
    )
    return build_plan(
        transcript=transcript,
        budget=budget,
        actions=actions,
        summarization_requests=requests,
        policy_name="drawn_policy",
        policy_version="0.0.1",
        estimator_ratio=draw(st.sampled_from([None, 4.0, 3.5])),
    )


def _keep_or_mask(draw: st.DrawFn, turn_id: str, *, forced_keep: bool) -> TurnAction:
    """Draw ``KEEP`` or a ``MASK`` with a stub, for a turn that stays in the view."""
    if forced_keep or draw(st.booleans()):
        return TurnAction(turn_id=turn_id, action=Action.KEEP)
    return TurnAction(
        turn_id=turn_id,
        action=Action.MASK,
        replacement=TurnReplacement(
            content=f"[masked {turn_id}]",
            token_estimate=draw(st.integers(min_value=0, max_value=80)),
        ),
    )


@st.composite
def transcripts_budgets_and_plans(
    draw: st.DrawFn,
) -> tuple[Transcript, CompactionBudget, CompactionPlan]:
    """Draw a transcript, a satisfiable budget for it, and a valid plan over both."""
    transcript = draw(transcripts())
    budget = draw(satisfiable_budgets(transcript))
    return transcript, budget, draw(plans_over(transcript, budget))
