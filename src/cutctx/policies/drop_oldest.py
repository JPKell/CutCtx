"""The deterministic last resort: drop the oldest turns nothing protects, exchanges whole.

The worked example every later policy copies. It is short on purpose — the judgment is in
:mod:`cutctx._invariants`, and a policy's job is to choose, not to re-derive the rules or the
arithmetic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from cutctx import _invariants
from cutctx.types import Action, TurnAction

if TYPE_CHECKING:
    from cutctx.types import CompactionBudget, CompactionPlan, Transcript

__all__ = ["DropOldestPolicy"]

_NAME: Final = "drop_oldest"
_VERSION: Final = "1.0.0"


class DropOldestPolicy:
    """Drops whole tool exchanges, oldest first, until the budget fits or nothing is left to drop.

    The policy of last resort, and the only one that needs no estimator: it removes turns whose
    estimates the caller already supplied, so no figure on its plans comes from the character-ratio
    default and :attr:`~cutctx.types.CompactionPlan.estimator_ratio` is always ``None``.

    What it will not do:

    * **Touch the untouchable set.** Every ``SYSTEM`` turn, every pinned turn and the
      ``protected_recent_turns`` tail are kept (spec §11 contract 2).
    * **Split a tool exchange.** It drops exchanges, not turns, so a call never loses its result
      and a result never loses its call (contract 3). An exchange with an untouchable member is
      undroppable in its entirety — which is why a *recent* tool result can make an *old*
      assistant turn survive, and why "dropped turns are older than kept turns" is not a property
      of this policy's output.
    * **Truncate silently.** If it runs out of droppable exchanges while still over budget, it
      returns the plan it reached with ``budget_unmet=True`` and lets the caller decide
      (ADR-0023's rule applied to transcripts). If the untouchable set alone exceeds the budget it
      raises :class:`~cutctx.errors.BudgetUnsatisfiable` with both numbers, before doing any work.

    Determinism: exchanges are considered in the order of their earliest turn, and dropping stops
    at the first point the running estimate fits. Same transcript, same budget ⇒ byte-identical
    plan (contract 4).
    """

    name: str = _NAME
    version: str = _VERSION

    def decide(self, transcript: Transcript, budget: CompactionBudget) -> CompactionPlan:
        """Return the plan that drops the oldest droppable exchanges until the budget fits.

        Args:
            transcript: The transcript to compact.
            budget: The window it must fit, and the tail no policy may touch.

        Returns:
            The plan. ``budget_unmet`` is ``True`` when every droppable exchange was dropped and
            the view is still over budget.

        Raises:
            BudgetUnsatisfiable: If the untouchable turns alone exceed ``budget.max_tokens``.
                Raised before any work, so the caller learns the budget is impossible rather than
                learning what a doomed policy did about it.
        """
        _invariants.require_satisfiable_budget(transcript, budget)

        removable = _invariants.removable_turn_ids(transcript, budget)
        estimates = {turn.turn_id: turn.token_estimate for turn in transcript.turns}
        running = transcript.token_estimate()
        dropped: set[str] = set()

        for exchange in _invariants.exchanges(transcript):
            if running <= budget.max_tokens:
                break
            # `removable_turn_ids` is closed over whole exchanges, so one member answers for all.
            if exchange[0] not in removable:
                continue
            dropped.update(exchange)
            running -= sum(estimates[turn_id] for turn_id in exchange)

        actions = tuple(
            TurnAction(
                turn_id=turn.turn_id,
                action=Action.DROP if turn.turn_id in dropped else Action.KEEP,
            )
            for turn in transcript.turns
        )
        return _invariants.build_plan(
            transcript=transcript,
            budget=budget,
            actions=actions,
            policy_name=self.name,
            policy_version=self.version,
        )
