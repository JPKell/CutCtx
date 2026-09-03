"""CutCtx — the suite's one answer to "this transcript has outgrown the window it must fit".

Given a transcript and a token budget, CutCtx decides — deterministically, explicably, and without
touching a model, a database or a disk — which turns to keep, mask, summarize or drop, and applies
that decision to produce a compacted view plus an auditable account of what was done to it.

The shape is **two-phase**, and it is the thing to understand first:

    >>> from cutctx import CompactionBudget, CompactionExecutor, DropOldestPolicy
    >>> from cutctx import Role, Transcript, TranscriptTurn
    >>> transcript = Transcript((
    ...     TranscriptTurn("s", Role.SYSTEM, "rules", 10),
    ...     TranscriptTurn("a", Role.USER, "old question", 40),
    ...     TranscriptTurn("b", Role.ASSISTANT, "old answer", 40),
    ... ))
    >>> plan = DropOldestPolicy().decide(transcript, CompactionBudget(60, protected_recent_turns=1))
    >>> plan.tokens_before, plan.tokens_after_estimate, plan.budget_unmet
    (90, 50, False)
    >>> view = CompactionExecutor().apply(transcript, plan)
    >>> view.transcript.turn_ids()
    ('s', 'b')
    >>> transcript.turn_ids()          # the input is a value, and values do not change
    ('s', 'a', 'b')

**plan → fulfil → apply.** The middle step exists because CutCtx never calls a model
(:doc:`ADR-0052 <adr>`). A plan that wants a span summarized carries a
:class:`SummarizationRequest`; the *application* fulfils it through its own governed inference
path — PromptCadence via LoadCoach, IdeaPress via its inference port — and hands the text back to
:meth:`~cutctx.executor.CompactionExecutor.apply` in ``summaries``. Skip the middle step and
:class:`SummaryMissing` says so by name, loudly, rather than quietly dropping a reduction the
plan's arithmetic already counted.

Anything not listed in ``__all__`` is private and may change without a version bump, whatever its
module happens to be named — :mod:`cutctx._invariants` included, though every policy routes
through it.
"""

from __future__ import annotations

from cutctx.__about__ import __version__
from cutctx.errors import (
    BudgetUnsatisfiable,
    CompactionError,
    PlanTranscriptMismatch,
    SummaryMissing,
)
from cutctx.estimator import CharRatioEstimator, TokenEstimator
from cutctx.executor import CompactionExecutor
from cutctx.policies import DropOldestPolicy
from cutctx.types import (
    EMPTY_METADATA,
    SUMMARY_TURN_ID_PREFIX,
    Action,
    CompactedTranscript,
    CompactionBudget,
    CompactionPlan,
    CompactionPolicy,
    CompactionReport,
    Role,
    SummarizationRequest,
    Transcript,
    TranscriptTurn,
    TurnAction,
    TurnReplacement,
)

__all__ = [
    "EMPTY_METADATA",
    "SUMMARY_TURN_ID_PREFIX",
    "Action",
    "BudgetUnsatisfiable",
    "CharRatioEstimator",
    "CompactedTranscript",
    "CompactionBudget",
    "CompactionError",
    "CompactionExecutor",
    "CompactionPlan",
    "CompactionPolicy",
    "CompactionReport",
    "DropOldestPolicy",
    "PlanTranscriptMismatch",
    "Role",
    "SummarizationRequest",
    "SummaryMissing",
    "TokenEstimator",
    "Transcript",
    "TranscriptTurn",
    "TurnAction",
    "TurnReplacement",
    "__version__",
]
