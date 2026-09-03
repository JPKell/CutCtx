"""Typed refusals — the error hierarchy CutCtx raises, per spec §7 and §13.

Every error subclasses :class:`baseaicore.SuiteError`, so a caller that already handles suite
errors handles these, and every ``code`` is part of the public contract: codes appear in API error
envelopes, in stored event rows and in CLI exit-code mapping, so changing what one means is a
breaking change.

Two refusals are deliberately **not** here.

* A malformed *input* — a transcript with duplicate turn ids, a negative token estimate, a
  character ratio of zero — raises :class:`baseaicore.ValidationError` from the constructor that
  received it. It is a defect in the caller's mapping, not an outcome of a compaction, and spec
  §13's table describes outcomes.
* ``COMPACTION_FAILED`` belongs to PromptCadence (spec §13's last row, and
  :doc:`ADR-0052 <adr>` decision 6). CutCtx hands back a plan flagged ``budget_unmet``; deciding
  that an unmet budget ends a trajectory is the application's policy, and a package that raised it
  would be making a decision the ADR assigns to its caller.
"""

from __future__ import annotations

from typing import ClassVar

from baseaicore import SuiteError

__all__ = [
    "BudgetUnsatisfiable",
    "CompactionError",
    "PlanTranscriptMismatch",
    "SummaryMissing",
]


class CompactionError(SuiteError):
    """Base for every error this package raises.

    Nothing raises it directly; it exists so a caller can catch every CutCtx refusal with one
    ``except`` without also catching unrelated suite errors.
    """

    code: ClassVar[str] = "COMPACTION_ERROR"


class BudgetUnsatisfiable(CompactionError):
    """The untouchable turns alone need more tokens than the budget allows.

    The system turn, every pinned turn and the ``protected_recent_turns`` tail are never masked,
    summarized or dropped (spec §11 contract 2), so their combined estimate is a floor no policy
    can go below. A budget under that floor is not a compaction problem to solve — it is a
    contradiction, and it is refused rather than "solved" by violating an invariant
    (:doc:`ADR-0052 <adr>` decision 6).

    ``details`` names **both numbers** — ``untouchable_tokens`` and ``max_tokens`` — plus
    ``untouchable_turn_count``, because "budget too small" without the figures leaves the operator
    unable to choose between raising the budget and unpinning something.
    """

    code: ClassVar[str] = "COMPACTION_BUDGET_UNSATISFIABLE"


class SummaryMissing(CompactionError):
    """A plan's :class:`~cutctx.types.SummarizationRequest` has no summary supplied to ``apply``.

    The loud failure of the step callers forget. CutCtx plans a summarization and never performs
    it (:doc:`ADR-0052 <adr>` decision 3): the application fulfils the request through its own
    governed inference path and passes the text back. Applying the plan without it would silently
    skip a reduction the plan's arithmetic already counted, so it refuses instead.

    ``details`` names the ``group_id``, the ``turn_ids`` it covers and the ``target_tokens`` that
    was planned for it — everything needed to go and fulfil it.
    """

    code: ClassVar[str] = "COMPACTION_SUMMARY_MISSING"


class PlanTranscriptMismatch(CompactionError):
    """A plan was applied to a transcript it was not built for.

    A plan is a total function over one transcript: every turn id appears in ``actions`` exactly
    once, in transcript order. Applying it to a transcript with different ids — or with the same
    ids in a different order — would produce a view nobody planned, so it refuses.

    ``details`` names ``unplanned_turn_ids`` (in the transcript, absent from the plan),
    ``unknown_turn_ids`` (named by the plan, absent from the transcript) and ``reordered``. Two
    different defects with two different fixes, so they are never merged into one list — and both
    are capped, since a 2 000-turn mismatch is not made clearer by 2 000 ids.
    """

    code: ClassVar[str] = "COMPACTION_PLAN_MISMATCH"
