"""Applying a plan — the second half of the two-phase protocol (spec §7).

The package's shape is **plan, fulfil, apply**, and the middle step is the caller's:
:class:`~cutctx.types.SummarizationRequest` s cross the boundary and come back as text, because a
package below the applications must not hold a second path to a model (:doc:`ADR-0052 <adr>`).
Forgetting the middle step is the mistake this module is loudest about —
:class:`~cutctx.errors.SummaryMissing` names the group rather than quietly skipping a reduction
the plan's arithmetic already counted.

Nothing here computes a token figure. The view's per-turn estimates come from the plan, and the
report copies the plan's totals, so there is exactly one derivation of "after" in the package
(:func:`cutctx._invariants.estimate_after`) and nothing for the two halves to drift apart over.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from cutctx.errors import PlanTranscriptMismatch, SummaryMissing
from cutctx.types import (
    Action,
    CompactedTranscript,
    CompactionReport,
    Role,
    Transcript,
    TranscriptTurn,
    TurnReplacement,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cutctx.types import CompactionPlan, SummarizationRequest

__all__ = ["CompactionExecutor"]

_NO_SUMMARIES: Final[Mapping[str, str]] = MappingProxyType({})

_ID_SAMPLE_LIMIT: Final = 10
"""How many ids an error's ``details`` carries; ``details`` travels into API error envelopes."""

SUMMARY_KIND: Final = "summary"
MASKED_KIND: Final = "masked"
KIND_KEY: Final = "cutctx.kind"
"""The metadata key the executor stamps on turns it produced or altered.

Namespaced so it cannot collide with a caller's own keys by accident, and stamped because a view
in which a summary is indistinguishable from a turn somebody actually took is a view nobody can
explain afterwards. Policies still treat ``metadata`` as opaque (spec §4) — this is the executor
labelling its own output, not a channel between policies.
"""

GROUP_KEY: Final = "cutctx.summary_group"
REPLACED_COUNT_KEY: Final = "cutctx.replaced_turn_count"


class CompactionExecutor:
    """Applies a plan, with the caller's summaries, to produce the view and its report.

    Stateless and configuration-free: every input is an argument, so two executors are
    interchangeable and an application can hold one for its lifetime.
    """

    def apply(
        self,
        transcript: Transcript,
        plan: CompactionPlan,
        summaries: Mapping[str, str] = _NO_SUMMARIES,
    ) -> CompactedTranscript:
        """Apply ``plan`` to ``transcript`` and return the compacted view with its report.

        The input transcript is **not** modified — it cannot be; every type here is a frozen
        dataclass over tuples, and the returned view is a new value (spec §11 contract 1). What
        the caller retains is the caller's data-ownership decision.

        Each action produces its turn's contribution to the view:

        * ``KEEP`` — the turn, unchanged and at its position.
        * ``MASK`` — the same turn with the plan's replacement body and estimate, at its position,
          stamped ``cutctx.kind=masked``. Role, ``pinned`` and ``tool_call_id`` are preserved, so
          a masked result is still the same tool exchange's member.
        * ``SUMMARIZE`` — nothing; the group's summary turn takes the position of the group's
          **earliest** turn, and the rest vanish.
        * ``DROP`` — nothing.

        A summary turn is ``role=ASSISTANT``, unpinned, with ``turn_id`` ``summary:<group_id>``
        and the request's ``target_tokens`` as its estimate. Not ``SYSTEM``: a system turn is
        untouchable, so summaries injected as system turns would accumulate across compaction
        rounds until they were the only thing a policy could not reduce. Not pinned, for the same
        reason. Its estimate is the plan's ``target_tokens`` and not a measurement of the supplied
        text, because the plan's figure is the audited one — a caller who needs the delivered size
        measures the returned transcript.

        Args:
            transcript: The transcript the plan was built for.
            plan: The plan to apply.
            summaries: The fulfilled summaries, keyed by ``group_id``. Entries for groups this
                plan does not contain are **ignored**, deliberately: a caller may reasonably hold
                one mapping across several plans, and failing a compaction over a surplus entry
                would refuse something that is not wrong. A *missing* one is a different matter
                and refuses.

        Returns:
            The compacted view and the ``context.compacted`` report body.

        Raises:
            PlanTranscriptMismatch: If the plan's actions do not name this transcript's turn ids,
                exactly once each, in this order. A plan is a total function over one transcript;
                applied to another it would produce a view nobody planned.
            SummaryMissing: If a summarization request has no entry in ``summaries``, naming the
                group, its turns and its target.
        """
        self._require_matching_transcript(transcript, plan)
        by_turn = {action.turn_id: action for action in plan.actions}
        requests = {request.group_id: request for request in plan.summarization_requests}
        for request in plan.summarization_requests:
            self._require_summary(request, summaries)

        turns: list[TranscriptTurn] = []
        kept: list[str] = []
        masked: list[str] = []
        summarized: list[str] = []
        dropped: list[str] = []
        summary_turn_ids: list[str] = []
        emitted_groups: set[str] = set()

        # Branching on the optional fields rather than on `action.action` is deliberate: it is the
        # same discrimination (TurnAction refuses a MASK without a replacement and a SUMMARIZE
        # without a group), and it narrows the types without an `assert` that -O would remove.
        for turn in transcript.turns:
            action = by_turn[turn.turn_id]
            if action.action is Action.KEEP:
                turns.append(turn)
                kept.append(turn.turn_id)
            elif action.replacement is not None:
                turns.append(self._masked(turn, action.replacement))
                masked.append(turn.turn_id)
            elif action.summary_group is not None:
                summarized.append(turn.turn_id)
                if action.summary_group not in emitted_groups:
                    emitted_groups.add(action.summary_group)
                    request = requests[action.summary_group]
                    turns.append(self._summary_turn(request, summaries[action.summary_group]))
                    summary_turn_ids.append(request.summary_turn_id)
            else:
                dropped.append(turn.turn_id)

        view = Transcript(turns=tuple(turns))
        report = CompactionReport(
            plan_hash=plan.plan_hash(),
            policy_name=plan.policy_name,
            policy_version=plan.policy_version,
            tokens_before=plan.tokens_before,
            tokens_after_estimate=plan.tokens_after_estimate,
            estimator_ratio=plan.estimator_ratio,
            budget_unmet=plan.budget_unmet,
            turns_before=len(transcript.turns),
            turns_after=len(view.turns),
            kept_turn_ids=tuple(kept),
            masked_turn_ids=tuple(masked),
            summarized_turn_ids=tuple(summarized),
            dropped_turn_ids=tuple(dropped),
            summary_turn_ids=tuple(summary_turn_ids),
        )
        return CompactedTranscript(transcript=view, report=report)

    @staticmethod
    def _require_matching_transcript(transcript: Transcript, plan: CompactionPlan) -> None:
        """Refuse a plan built for other turns, or for these turns in another order."""
        planned = tuple(action.turn_id for action in plan.actions)
        actual = transcript.turn_ids()
        if planned == actual:
            return
        unknown = sorted(set(planned) - set(actual))
        unplanned = sorted(set(actual) - set(planned))
        reordered = not unknown and not unplanned
        raise PlanTranscriptMismatch(
            f"This plan was built for {len(planned)} turns and the transcript has {len(actual)}; "
            + (
                "the ids match but their order differs, and a plan's actions are positional — "
                "'oldest' and the protected tail both are."
                if reordered
                else f"{len(unknown)} planned turns are absent from it and {len(unplanned)} of "
                "its turns are unplanned."
            ),
            details={
                "unknown_turn_ids": unknown[:_ID_SAMPLE_LIMIT],
                "unplanned_turn_ids": unplanned[:_ID_SAMPLE_LIMIT],
                "reordered": reordered,
            },
        )

    @staticmethod
    def _require_summary(request: SummarizationRequest, summaries: Mapping[str, str]) -> None:
        """Refuse to apply a plan whose summarization work was not done."""
        if request.group_id in summaries:
            return
        raise SummaryMissing(
            f"Summarization group {request.group_id!r} has no supplied summary. CutCtx plans a "
            "summarization and never performs it (ADR-0052): fulfil the request through your own "
            "governed inference path — on a local tier when the transcript is confidential — and "
            "pass the text back in `summaries`.",
            details={
                "group_id": request.group_id,
                "turn_ids": list(request.turn_ids)[:_ID_SAMPLE_LIMIT],
                "target_tokens": request.target_tokens,
                "prompt_id": request.prompt_id,
            },
        )

    @staticmethod
    def _masked(turn: TranscriptTurn, replacement: TurnReplacement) -> TranscriptTurn:
        """Return ``turn`` with the plan's stub in place of its body."""
        return TranscriptTurn(
            turn_id=turn.turn_id,
            role=turn.role,
            content=replacement.content,
            token_estimate=replacement.token_estimate,
            tool_call_id=turn.tool_call_id,
            pinned=turn.pinned,
            metadata={**turn.metadata, KIND_KEY: MASKED_KIND},
        )

    @staticmethod
    def _summary_turn(request: SummarizationRequest, summary: str) -> TranscriptTurn:
        """Return the one turn a fulfilled summarization request becomes."""
        return TranscriptTurn(
            turn_id=request.summary_turn_id,
            role=Role.ASSISTANT,
            content=summary,
            token_estimate=request.target_tokens,
            metadata={
                KIND_KEY: SUMMARY_KIND,
                GROUP_KEY: request.group_id,
                REPLACED_COUNT_KEY: str(len(request.turn_ids)),
            },
        )
