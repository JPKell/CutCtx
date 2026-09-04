"""Plan a summarization of the oldest contiguous span. Never perform one.

:doc:`ADR-0052 <adr>` decision 3 is the whole shape of this module: **the package plans a
summarization and never executes one.** There is no model here, no HTTP client, no prompt text and
no way to acquire any — what this policy produces is a
:class:`~cutctx.types.SummarizationRequest` naming turns, a token target and a *prompt id*, and the
caller fulfils it through its own governed inference path before handing the text back to
:meth:`~cutctx.executor.CompactionExecutor.apply`.

Two honesty points, both of which look like defects until the reasoning is written down.

**A whole exchange goes into a group or none of it does.** Spec §11 contract 3 as
`C1_HANDOFF.md` §4 reads it: within one exchange, either every member is retained or every member
is removed by the *same* action. Summarizing is a removal, so a span that clips half an exchange is
not a span this policy may take — and since an exchange's members need not be adjacent, "contiguous"
is not enough on its own. See :func:`_spans`.

**The planned summary's token figure is an estimate of a text that does not exist yet.** It is
``target_tokens``, derived from the span and the ratio, and the executor does not re-estimate the
summary it is handed — so a caller whose model returns something longer has a view that costs more
than the plan said. That is spec §11 contract 5 working as intended (an estimate is never presented
as a count), and the fix is emphatically *not* to recompute anything in the report: `C1_HANDOFF.md`
§3.1 settled that nothing in the report is computed, only copied from the plan, because one
computation is what makes plan and report incapable of disagreeing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from baseaicore import ValidationError, sha256_of

from cutctx import _invariants
from cutctx.types import Action, SummarizationRequest, TurnAction

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cutctx.types import CompactionBudget, CompactionPlan, Transcript

__all__ = ["GROUP_ID_PREFIX", "SummarizingPolicy"]

_NAME: Final = "summarizing"
_VERSION: Final = "1.0.0"

GROUP_ID_PREFIX: Final = "span-"
"""What a derived group id starts with, so a reader can tell one from a caller's own name."""

_GROUP_DIGEST_CHARS: Final = 16
"""How much of the span digest a group id carries.

Sixty-four hex characters in an id that appears in every action of the group, in the request, and
in the summary turn's own id is noise in an audit record; sixteen is 64 bits, which no transcript
will collide within a single plan. The plan lists the span's ``turn_ids`` beside the group anyway,
so the id identifies rather than proves.
"""


def group_id_for(turn_ids: Sequence[str]) -> str:
    """Return the deterministic group id for a span.

    **Derived from the span, never generated.** A counter would renumber when a chain composed the
    policies differently; a uuid would differ on every run. Either would break spec §11 contract 4
    outright, because the group id is in the plan's bytes *and* in the summary turn's derived id
    (``summary:<group_id>``).

    Args:
        turn_ids: The span's turn ids, in transcript order.

    Returns:
        ``"span-"`` followed by the first sixteen hex characters of the sha256 of the ids joined by
        NUL. NUL rather than a printable separator so that ``["a", "b-c"]`` and ``["a-b", "c"]``
        cannot digest alike, whatever a caller's ids look like.
    """
    return f"{GROUP_ID_PREFIX}{sha256_of(chr(0).join(turn_ids))[:_GROUP_DIGEST_CHARS]}"


def _spans(transcript: Transcript, budget: CompactionBudget) -> list[tuple[str, ...]]:
    """Return the summarizable spans, oldest first.

    A span is a run of turns that are **adjacent in the transcript**, **removable**, and **whole in
    their exchanges**. The three conditions are applied in that order and the third is why this
    function exists rather than a comprehension:

    :func:`~cutctx._invariants.removable_turn_ids` is already closed over exchanges, so no
    removable turn has an untouchable partner. But an exchange's members need not be *adjacent* —
    an assistant call at turn 2 and its result at turn 9 are one exchange — so a contiguous run of
    removable turns can still contain half of one. Summarizing that run would separate the pair,
    which contract 3 forbids and ``build_plan`` would reject. So each run is filtered down to the
    turns whose whole exchange it contains, and the filtering is allowed to fragment it: the
    fragments are re-split and offered in order.

    Ordering comes from **transcript position at every step** — never from iterating the ``set``
    that ``removable_turn_ids`` returns or the dict that groups exchanges. That is the failure mode
    the development plan names for this phase, and it is the kind that only shows up under an
    unlucky ``pytest-randomly`` seed unless the ordering is structural.

    Args:
        transcript: The transcript.
        budget: The budget, for the protected tail.

    Returns:
        The spans, each a tuple of turn ids in transcript order, ordered by position. Possibly
        empty. Spans are not filtered by length here; that is the caller's ``min_span_turns``.
    """
    removable = _invariants.removable_turn_ids(transcript, budget)
    order = {turn.turn_id: index for index, turn in enumerate(transcript.turns)}
    exchange_of = {
        turn_id: exchange for exchange in _invariants.exchanges(transcript) for turn_id in exchange
    }

    runs: list[list[str]] = []
    current: list[str] = []
    for turn in transcript.turns:
        if turn.turn_id in removable:
            current.append(turn.turn_id)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    spans: list[tuple[str, ...]] = []
    for run in runs:
        inside = set(run)
        whole = [
            turn_id for turn_id in run if all(member in inside for member in exchange_of[turn_id])
        ]
        # Re-split: dropping a half-exchange can leave a gap in what was one run.
        fragment: list[str] = []
        for turn_id in whole:
            if fragment and order[turn_id] != order[fragment[-1]] + 1:
                spans.append(tuple(fragment))
                fragment = []
            fragment.append(turn_id)
        if fragment:
            spans.append(tuple(fragment))
    return spans


class SummarizingPolicy:
    """Folds the oldest contiguous unpinned span into one planned summary turn.

    One span per plan, and one :class:`~cutctx.types.SummarizationRequest` for it — the spec's
    "replaces the oldest contiguous unpinned span with one summary turn", taken literally. A policy
    that folded several spans would spend several model calls on one compaction, and the caller who
    wants that composes this policy into a :class:`~cutctx.policies.chain.PolicyChain` twice.

    What it will not do:

    * **Split an exchange.** See :func:`_spans`.
    * **Touch the untouchable set** (spec §11 contract 2), which ``removable_turn_ids`` already
      excludes.
    * **Take a span shorter than ``min_span_turns``.** Folding two turns into a summary costs a
      model call to save very little and makes the transcript harder to read for it.
    * **Plan a summary that costs more than the span it replaces.** With the default ratio that
      cannot happen, but a caller may configure ``target_ratio=1.5``; a "compaction" that grew the
      transcript would be a compaction in name.
    * **Produce prompt text.** ``prompt_id`` names a versioned prompt record (ADR-0012). There is
      no prompt string anywhere in this package and a test asserts it.

    Determinism: the span is the first qualifying one in transcript order, and its ``group_id`` is
    a digest of its turn ids, so the same transcript and budget give a byte-identical plan
    (contract 4).

    Args:
        prompt_id: The versioned prompt record that will produce the summary. A name, never text.
        target_ratio: What fraction of the span's estimate the summary is planned to cost.
        min_span_turns: The shortest span worth folding.

    Raises:
        ValidationError: If ``prompt_id`` is empty, ``target_ratio`` is not a positive finite
            number, or ``min_span_turns`` is below one. Caller configuration, so it fails at
            construction.
    """

    name: str = _NAME
    version: str = _VERSION

    def __init__(self, prompt_id: str, target_ratio: float = 0.2, min_span_turns: int = 4) -> None:
        """Validate the configuration and bind it."""
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValidationError(
                "prompt_id must name a versioned prompt record (ADR-0012); CutCtx carries no "
                "prompt text and cannot invent one.",
                details={"prompt_id": repr(prompt_id)},
            )
        if (
            isinstance(target_ratio, bool)
            or not isinstance(target_ratio, int | float)
            or not (0 < float(target_ratio) < float("inf"))
        ):
            raise ValidationError(
                f"target_ratio must be a finite number greater than zero; got {target_ratio!r}. "
                "Zero would plan a summary of nothing, which costs a model call and says nothing.",
                details={"target_ratio": repr(target_ratio)},
            )
        if isinstance(min_span_turns, bool) or not isinstance(min_span_turns, int):
            raise ValidationError(
                f"min_span_turns must be an int; got {type(min_span_turns).__name__}.",
                details={"min_span_turns": repr(min_span_turns)},
            )
        if min_span_turns < 1:
            raise ValidationError(
                f"min_span_turns must be at least 1; got {min_span_turns}. A span of no turns is "
                "not a span.",
                details={"min_span_turns": min_span_turns},
            )
        self._prompt_id = prompt_id
        self._target_ratio = float(target_ratio)
        self._min_span_turns = min_span_turns

    @property
    def prompt_id(self) -> str:
        """The versioned prompt record the request names."""
        return self._prompt_id

    @property
    def target_ratio(self) -> float:
        """What fraction of a span's estimate its summary is planned to cost."""
        return self._target_ratio

    @property
    def min_span_turns(self) -> int:
        """The shortest span this policy will fold."""
        return self._min_span_turns

    def decide(self, transcript: Transcript, budget: CompactionBudget) -> CompactionPlan:
        """Return the plan that folds the oldest qualifying span, or the plan that folds nothing.

        Args:
            transcript: The transcript to compact.
            budget: The window it must fit, and the tail no policy may touch.

        Returns:
            The plan. When the transcript already fits, or no span qualifies — nothing removable,
            every run too short, or the arithmetic saying a summary would not help — every turn is
            ``KEEP`` and there are no requests; ``budget_unmet`` then says whether the untouched
            transcript fits. A policy that finds nothing to do says so by planning nothing, not by
            raising.

        Raises:
            BudgetUnsatisfiable: If the untouchable turns alone exceed ``budget.max_tokens``.
        """
        _invariants.require_satisfiable_budget(transcript, budget)
        estimates = {turn.turn_id: turn.token_estimate for turn in transcript.turns}

        chosen: tuple[str, ...] | None = None
        target_tokens = 0
        # A transcript that already fits is left alone. Every other shipped policy checks this in
        # its loop; this one has no loop over reductions, so it is checked here — and it matters
        # more here than anywhere, because folding a span that did not need folding spends a model
        # call and replaces turns the model could still have read.
        spans = (
            [] if transcript.token_estimate() <= budget.max_tokens else _spans(transcript, budget)
        )
        for span in spans:
            if len(span) < self._min_span_turns:
                continue
            span_tokens = sum(estimates[turn_id] for turn_id in span)
            target = max(1, round(span_tokens * self._target_ratio))
            if target >= span_tokens:
                continue
            chosen, target_tokens = span, target
            break

        if chosen is None:
            actions = tuple(
                TurnAction(turn_id=turn.turn_id, action=Action.KEEP) for turn in transcript.turns
            )
            return _invariants.build_plan(
                transcript=transcript,
                budget=budget,
                actions=actions,
                policy_name=self.name,
                policy_version=self.version,
            )

        group_id = group_id_for(chosen)
        folded = frozenset(chosen)
        actions = tuple(
            TurnAction(
                turn_id=turn.turn_id,
                action=Action.SUMMARIZE if turn.turn_id in folded else Action.KEEP,
                summary_group=group_id if turn.turn_id in folded else None,
            )
            for turn in transcript.turns
        )
        request = SummarizationRequest(
            group_id=group_id,
            turn_ids=chosen,
            target_tokens=target_tokens,
            prompt_id=self._prompt_id,
        )
        return _invariants.build_plan(
            transcript=transcript,
            budget=budget,
            actions=actions,
            summarization_requests=(request,),
            policy_name=self.name,
            policy_version=self.version,
        )
