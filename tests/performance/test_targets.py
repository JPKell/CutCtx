"""Spec §15's performance targets, behind the ``performance`` marker (excluded by default).

The figures are budgets, not benchmarks: they exist so that a change which makes planning
quadratic is noticed by CI rather than by PromptCadence's turn loop. Each is measured on the
median of several runs, because a single timing on a shared runner is noise.
"""

from __future__ import annotations

import statistics
import time

import pytest

from conftest import turn
from cutctx import CompactionBudget, CompactionExecutor, DropOldestPolicy, Role, Transcript

POLICY = DropOldestPolicy()
EXECUTOR = CompactionExecutor()
RUNS = 5


def build(turn_count: int) -> Transcript:
    """Build a transcript of ``turn_count`` turns, a third of them in tool exchanges."""
    turns = [turn("s1", Role.SYSTEM, tokens=20)]
    for index in range(turn_count - 1):
        if index % 3 == 0:
            turns.append(turn(f"a{index}", Role.ASSISTANT, tokens=40, tool_call_id=f"c{index}"))
        elif index % 3 == 1:
            turns.append(turn(f"t{index}", Role.TOOL, tokens=300, tool_call_id=f"c{index - 1}"))
        else:
            turns.append(turn(f"u{index}", tokens=30))
    return Transcript(turns=tuple(turns))


def median_ms(work: object, repeats: int = RUNS) -> float:
    """Return the median wall time of ``work`` in milliseconds."""
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        work()  # type: ignore[operator]  # a zero-argument callable by construction
        timings.append((time.perf_counter() - started) * 1000)
    return statistics.median(timings)


@pytest.mark.performance
def test_planning_two_hundred_turns_is_under_fifty_milliseconds() -> None:
    transcript = build(200)
    budget = CompactionBudget(max_tokens=5_000, protected_recent_turns=4)

    elapsed_ms = median_ms(lambda: POLICY.decide(transcript, budget))

    assert elapsed_ms <= 50, f"planning 200 turns took {elapsed_ms:.1f} ms"


@pytest.mark.performance
def test_planning_two_thousand_turns_is_under_five_hundred_milliseconds() -> None:
    transcript = build(2_000)
    budget = CompactionBudget(max_tokens=50_000, protected_recent_turns=4)

    elapsed_ms = median_ms(lambda: POLICY.decide(transcript, budget))

    assert elapsed_ms <= 500, f"planning 2000 turns took {elapsed_ms:.1f} ms"


@pytest.mark.performance
def test_applying_two_hundred_turns_is_under_ten_milliseconds() -> None:
    transcript = build(200)
    plan = POLICY.decide(transcript, CompactionBudget(5_000, protected_recent_turns=4))

    elapsed_ms = median_ms(lambda: EXECUTOR.apply(transcript, plan))

    assert elapsed_ms <= 10, f"applying 200 turns took {elapsed_ms:.1f} ms"


@pytest.mark.performance
def test_planning_scales_linearly_rather_than_quadratically() -> None:
    """The shape, not the constant: a ten-fold transcript must not cost a hundred-fold.

    The absolute budgets above would pass a quadratic implementation for a long time — 2 000 turns
    is small — and would then fail on the first real 20 000-turn trajectory. This is the test that
    would have noticed.
    """
    small = build(200)
    large = build(2_000)
    small_ms = median_ms(lambda: POLICY.decide(small, CompactionBudget(5_000, 4)))
    large_ms = median_ms(lambda: POLICY.decide(large, CompactionBudget(50_000, 4)))

    assert large_ms <= small_ms * 40, (
        f"10x the turns cost {large_ms / max(small_ms, 1e-9):.1f}x the time"
    )
