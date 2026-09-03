"""The character-ratio default, and the honesty rule it exists to serve."""

from __future__ import annotations

import pytest
from baseaicore import ValidationError

from cutctx import CharRatioEstimator, TokenEstimator


def test_the_documented_default_is_four_characters_per_token() -> None:
    assert CharRatioEstimator().chars_per_token == 4.0


def test_empty_text_estimates_at_zero() -> None:
    assert CharRatioEstimator().estimate_tokens("") == 0


@pytest.mark.parametrize(("text", "expected"), [("a", 1), ("abcd", 1), ("abcde", 2), ("a" * 8, 2)])
def test_rounding_is_up_so_an_estimate_is_never_optimistic(text: str, expected: int) -> None:
    """An estimate that undershot would be discovered by the provider, as a rejected request."""
    assert CharRatioEstimator().estimate_tokens(text) == expected


def test_the_ratio_is_configurable() -> None:
    assert CharRatioEstimator(chars_per_token=2.0).estimate_tokens("abcd") == 2


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("nan"), float("inf")])
def test_a_ratio_that_could_not_produce_an_estimate_is_refused(ratio: float) -> None:
    with pytest.raises(ValidationError, match="finite positive"):
        CharRatioEstimator(chars_per_token=ratio)


def test_the_estimate_is_the_same_on_every_call() -> None:
    estimator = CharRatioEstimator()
    text = "a transcript turn, of no particular length"

    assert estimator.estimate_tokens(text) == estimator.estimate_tokens(text)


def test_the_default_satisfies_the_protocol() -> None:
    assert isinstance(CharRatioEstimator(), TokenEstimator)


def test_the_estimator_counts_code_points_not_bytes() -> None:
    """A defined, platform-stable quantity — which is what determinism needs from it."""
    assert CharRatioEstimator(chars_per_token=1.0).estimate_tokens("é€") == 2
