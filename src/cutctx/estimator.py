"""Token estimation — injected, documented, and never presented as a count.

CutCtx ships no tokenizer (spec §3). A tokenizer would be a heavyweight dependency in a package
whose whole value is purity, and it would tie every plan to one tokenizer's version
(:doc:`ADR-0052 <adr>`, "Count tokens exactly with a tokenizer"). So the caller injects an
estimator, and when the character-ratio default produced a figure the ratio rides on the plan —
:attr:`~cutctx.types.CompactionPlan.estimator_ratio` — so that a reader can tell an estimate from
a count (ADR-0016).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from baseaicore import ValidationError

__all__ = ["CharRatioEstimator", "TokenEstimator"]


@runtime_checkable
class TokenEstimator(Protocol):
    """Estimates how many tokens a piece of text costs.

    Implementations must be **pure and deterministic**: the same text yields the same number, in
    every process and on every platform, because a plan built from a wobbling estimate is not
    byte-identical on re-derivation (spec §11 contract 4).
    """

    def estimate_tokens(self, text: str) -> int:
        """Return the estimated token cost of ``text``. Never negative."""
        ...


@dataclass(frozen=True, slots=True)
class CharRatioEstimator:
    """The documented default: characters divided by a ratio, rounded up.

    Crude on purpose. It needs no vocabulary, no model and no download, it is exactly
    reproducible, and it is honest about being an estimate — which is the property that matters,
    since :class:`~cutctx.types.CompactionPlan` records the ratio beside every figure it produced.
    A caller who needs accuracy injects their provider's tokenizer behind
    :class:`TokenEstimator`; CutCtx will record ``estimator_ratio=None`` and say nothing it cannot
    support.

    Rounding is **up**, so an estimate is never optimistic: a compaction that undershot its own
    estimate would be discovered by the provider, as a rejected request.

    Args:
        chars_per_token: How many characters one token is assumed to hold. The default of ``4.0``
            is the spec's.

    Raises:
        ValidationError: If ``chars_per_token`` is not finite and positive. Zero would divide, and
            a negative ratio would produce negative estimates that make a transcript's total fall
            below the sum of its untouchable turns.
    """

    chars_per_token: float = 4.0

    def __post_init__(self) -> None:
        """Refuse a ratio that could not produce an estimate."""
        if not math.isfinite(self.chars_per_token) or self.chars_per_token <= 0:
            raise ValidationError(
                f"chars_per_token must be a finite positive number; got {self.chars_per_token!r}.",
                details={"chars_per_token": repr(self.chars_per_token)},
            )

    def estimate_tokens(self, text: str) -> int:
        """Return ``ceil(len(text) / chars_per_token)``; zero for empty text.

        Counts Python characters (code points), not bytes or graphemes: it is a stable, defined
        quantity on every platform, which is what determinism needs from it.
        """
        if not text:
            return 0
        return math.ceil(len(text) / self.chars_per_token)
