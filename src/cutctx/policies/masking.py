"""Mask old tool-result bodies: keep the reasoning, drop the bulk, leave the turn where it is.

The policy that makes an agent transcript survivable. A tool result is usually the largest thing in
a transcript and the least useful once it has been reasoned about — the assistant turn that
*interprets* a 4 000-token file listing is worth keeping; the listing itself, twelve turns later, is
not. So the turn stays at its position with a labelled stub in place of its body.

**Masking removes nothing**, which is the whole reason this policy is legal under spec §11
contract 3. A masked turn is still in the view, still carries its ``tool_call_id``, and still sits
beside the call that produced it — so no exchange is separated and nothing is orphaned. That is the
reading `C1_HANDOFF.md` §4 settled and this module depends on it: within one exchange, either every
member is retained (``KEEP``/``MASK``, mixed freely) or every member is removed by the same action.

**The stub carries a hash, never an excerpt** (spec §14). The original may hold a secret, and a
"first 200 characters" preview is a secret-bearing excerpt with extra steps. What the stub carries
is what an operator needs to find the original in the record that stored it — its digest — and what
the model needs in order to know that something *was* there and how big it was.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from baseaicore import ValidationError, sha256_of

from cutctx import _invariants
from cutctx.estimator import CharRatioEstimator
from cutctx.types import Action, Role, TurnAction, TurnReplacement

if TYPE_CHECKING:
    from cutctx.estimator import TokenEstimator
    from cutctx.types import CompactionBudget, CompactionPlan, Transcript, TranscriptTurn

__all__ = ["DEFAULT_PLACEHOLDER", "ObservationMaskingPolicy"]

_NAME: Final = "observation_masking"
_VERSION: Final = "1.0.0"

DEFAULT_PLACEHOLDER: Final = "[tool result masked by cutctx]"
"""The label a masked body is replaced by, before its evidence is appended.

Configurable, because the phrase a model reads is a prompt-design question and the caller owns it.
What is **not** configurable is the evidence that follows it: the original's token estimate and its
sha256 are appended by :func:`stub_for` whatever the label says. A placeholder template that could
omit the digest would make spec §14's guarantee a matter of caller discipline, and a guarantee that
depends on discipline is a default rather than a guarantee.
"""


def stub_for(turn: TranscriptTurn, *, placeholder: str) -> str:
    """Return the text that replaces a masked turn's body.

    The format is golden-locked: it appears in every plan that masks anything, and a plan is an
    audit record (spec §11 contract 4). Changing it is a policy version bump.

    Args:
        turn: The turn being masked. Only its content and estimate are read.
        placeholder: The label, from the policy's configuration.

    Returns:
        ``"<placeholder> (original: <n> tokens, sha256:<digest>)"``. The digest is of the original
        content, complete and untruncated: it is what links the stub to whatever stored the
        original, and a shortened one would be a collision question nobody asked to have.
    """
    return (
        f"{placeholder} (original: {turn.token_estimate} tokens, sha256:{sha256_of(turn.content)})"
    )


class ObservationMaskingPolicy:
    """Replaces old ``TOOL`` bodies with labelled stubs until the budget fits.

    What it will not do:

    * **Touch anything but a ``TOOL`` turn.** Assistant reasoning, user messages and system turns
      are left byte-identical. The value of this policy is precisely that the reasoning survives.
    * **Mask within the ``keep_recent_results`` most recent results.** That is a hard floor and it
      does not move for the budget: a model that cannot see any recent observation is a model
      working blind, and a compaction that produced one has met its number by breaking the task.
    * **Touch the untouchable set.** Every ``SYSTEM`` turn, every pinned turn and the
      ``protected_recent_turns`` tail (spec §11 contract 2). Note that masking is bounded by
      :func:`~cutctx._invariants.untouchable_turn_ids` and **not** by ``removable_turn_ids``:
      masking removes nothing, so an exchange with an untouchable member may still have its other
      members masked.
    * **Make a turn bigger.** A stub costs about thirty tokens, so masking a two-token result would
      *increase* the estimate. Such a turn is left alone. Without this, a transcript of many tiny
      results would grow under a policy whose purpose is shrinking.
    * **Truncate silently.** Having masked everything it may, it returns the plan it reached; if
      that is still over budget the plan says ``budget_unmet`` and the caller — or the next policy
      in a chain — decides.

    **It stops as soon as the budget fits**, oldest result first, which is `DropOldestPolicy`'s
    rule applied to a different action. Masking every eligible result unconditionally would be
    simpler to describe and would throw away observations for nothing: if masking two results meets
    the budget, masking nine costs the model seven observations it could have had. The floor above
    is what keeps that from being a slippery slope.

    Determinism: results are considered in transcript order and masking stops at the first point
    the running estimate fits, so the same transcript and budget give a byte-identical plan
    (contract 4).

    Args:
        keep_recent_results: How many of the most recent ``TOOL`` turns are never masked.
        placeholder: The stub's label. See :data:`DEFAULT_PLACEHOLDER`.
        estimator: What estimates the stub's cost. ``None`` means the documented default,
            :class:`~cutctx.estimator.CharRatioEstimator`, and then the plan carries its ratio in
            ``estimator_ratio`` so a reader can tell an estimate from a count (ADR-0016). An
            injected estimator leaves the field ``None``, because CutCtx cannot describe a ratio
            somebody else's tokenizer does not have.

    Raises:
        ValidationError: If ``keep_recent_results`` is negative, the placeholder is empty, or the
            estimator does not estimate. All three are caller configuration, so they fail where the
            caller is rather than on the transcript that happened to trip them.
    """

    name: str = _NAME
    version: str = _VERSION

    def __init__(
        self,
        keep_recent_results: int = 2,
        placeholder: str = DEFAULT_PLACEHOLDER,
        *,
        estimator: TokenEstimator | None = None,
    ) -> None:
        """Validate the configuration and bind it."""
        if isinstance(keep_recent_results, bool) or not isinstance(keep_recent_results, int):
            raise ValidationError(
                f"keep_recent_results must be an int; got {type(keep_recent_results).__name__}.",
                details={"keep_recent_results": repr(keep_recent_results)},
            )
        if keep_recent_results < 0:
            raise ValidationError(
                f"keep_recent_results must not be negative; got {keep_recent_results}. Zero means "
                "every eligible result may be masked, which is a policy; a negative number is not.",
                details={"keep_recent_results": keep_recent_results},
            )
        if not isinstance(placeholder, str) or not placeholder.strip():
            raise ValidationError(
                "placeholder must be a non-empty string: it is what the model reads in place of "
                "the body, and an empty one leaves it unable to tell a masked turn from a turn "
                "that said nothing.",
                details={"placeholder": repr(placeholder)},
            )
        if estimator is not None and not callable(getattr(estimator, "estimate_tokens", None)):
            raise ValidationError(
                "estimator must implement estimate_tokens(text) -> int.",
                details={"estimator": type(estimator).__name__},
            )
        self._keep_recent_results = keep_recent_results
        self._placeholder = placeholder
        self._injected = estimator
        self._estimator: TokenEstimator = CharRatioEstimator() if estimator is None else estimator

    @property
    def keep_recent_results(self) -> int:
        """How many of the most recent ``TOOL`` turns this policy will never mask."""
        return self._keep_recent_results

    @property
    def placeholder(self) -> str:
        """The stub's label."""
        return self._placeholder

    def decide(self, transcript: Transcript, budget: CompactionBudget) -> CompactionPlan:
        """Return the plan that masks the oldest eligible tool results until the budget fits.

        Args:
            transcript: The transcript to compact.
            budget: The window it must fit, and the tail no policy may touch.

        Returns:
            The plan. ``budget_unmet`` is ``True`` when everything maskable was masked and the
            view is still over budget.

        Raises:
            BudgetUnsatisfiable: If the untouchable turns alone exceed ``budget.max_tokens``.
                Raised before any work, so an impossible budget is reported as impossible rather
                than as whatever a doomed policy did about it.
        """
        _invariants.require_satisfiable_budget(transcript, budget)
        untouchable = _invariants.untouchable_turn_ids(transcript, budget)

        results = [turn for turn in transcript.turns if turn.role is Role.TOOL]
        # A slice, not an index comparison: `keep_recent_results=0` must keep nothing, and
        # `results[-0:]` is the whole list.
        kept_recent = (
            frozenset(turn.turn_id for turn in results[-self._keep_recent_results :])
            if self._keep_recent_results
            else frozenset()
        )

        running = transcript.token_estimate()
        replacements: dict[str, TurnReplacement] = {}
        for turn in results:
            if running <= budget.max_tokens:
                break
            if turn.turn_id in untouchable or turn.turn_id in kept_recent:
                continue
            content = stub_for(turn, placeholder=self._placeholder)
            estimate = self._estimator.estimate_tokens(content)
            if estimate >= turn.token_estimate:
                # Masking would cost more than it saves. Skipping keeps the policy monotonic:
                # every plan it produces is no larger than the transcript it planned over.
                continue
            replacements[turn.turn_id] = TurnReplacement(content=content, token_estimate=estimate)
            running -= turn.token_estimate - estimate

        actions = tuple(
            TurnAction(
                turn_id=turn.turn_id,
                action=Action.MASK if turn.turn_id in replacements else Action.KEEP,
                replacement=replacements.get(turn.turn_id),
            )
            for turn in transcript.turns
        )
        return _invariants.build_plan(
            transcript=transcript,
            budget=budget,
            actions=actions,
            policy_name=self.name,
            policy_version=self.version,
            estimator_ratio=self.estimator_ratio_if_used(bool(replacements)),
        )

    def estimator_ratio_if_used(self, estimated_anything: bool) -> float | None:
        """Return the ratio to record on a plan, or ``None``.

        Args:
            estimated_anything: Whether this plan actually contains a stub. A plan that masked
                nothing estimated nothing, and recording a ratio on it would describe a
                computation that did not happen.

        Returns:
            The default estimator's ``chars_per_token`` when it produced a figure on this plan;
            ``None`` when an estimator was injected or when nothing was estimated.
        """
        if not estimated_anything or self._injected is not None:
            return None
        default = self._estimator
        return default.chars_per_token if isinstance(default, CharRatioEstimator) else None
