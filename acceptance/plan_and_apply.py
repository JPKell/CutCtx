#!/usr/bin/env python3
"""Spec §20 criterion 2: plan and apply a compaction with a hand-supplied summary.

Run it in a throwaway virtualenv holding nothing but this package and its one dependency::

    python -m venv /tmp/cutctx-acceptance
    /tmp/cutctx-acceptance/bin/pip install .
    /tmp/cutctx-acceptance/bin/python acceptance/plan_and_apply.py

It exits ``0`` when every claim below holds and non-zero with a message when one does not, so it is
a check rather than a demonstration — M10's exit condition is *"clean-venv acceptance scripts
pass"*, and a script that only printed things would pass while being wrong. Nothing here imports
pytest, this repository's test helpers, or any suite application: the point is that an application
with `cutctx` installed and nothing else can do this.

It is also the quickstart, and it shows the thing a new caller most needs to see: **CutCtx plans a
summarization and never performs one** (ADR-0052). The summary below is a hand-written string, which
is exactly the shape a real caller's is — text it obtained from its own governed inference path and
handed back.
"""

from __future__ import annotations

import sys

from cutctx import (
    Action,
    CompactionBudget,
    CompactionExecutor,
    Role,
    Transcript,
    TranscriptTurn,
    default_chain,
)

PROMPT_ID = "compaction.summarize.v1"
"""A versioned prompt record's *name*. No prompt text exists anywhere in this package (ADR-0012)."""


def _transcript() -> Transcript:
    """An agent transcript: a system turn, six tool exchanges, and a closing question."""
    turns = [
        TranscriptTurn(
            turn_id="s1",
            role=Role.SYSTEM,
            content="You are a careful assistant.",
            token_estimate=10,
        )
    ]
    for index in range(1, 7):
        turns.append(
            TranscriptTurn(
                turn_id=f"u{index}",
                role=Role.USER,
                content=f"Question {index}?",
                token_estimate=20,
            )
        )
        turns.append(
            TranscriptTurn(
                turn_id=f"a{index}",
                role=Role.ASSISTANT,
                content=f"Looking at file {index}.",
                token_estimate=25,
                tool_call_id=f"c{index}",
            )
        )
        turns.append(
            TranscriptTurn(
                turn_id=f"t{index}",
                role=Role.TOOL,
                content=f"line {index}\n" * 300,
                token_estimate=300,
                tool_call_id=f"c{index}",
            )
        )
    turns.append(
        TranscriptTurn(turn_id="u9", role=Role.USER, content="Anything else?", token_estimate=15)
    )
    return Transcript(turns=tuple(turns))


def _check(claim: str, condition: bool) -> None:  # noqa: FBT001 — a check takes the answer
    """Print the claim and stop the script if it does not hold."""
    print(f"{'ok  ' if condition else 'FAIL'}  {claim}")
    if not condition:
        sys.exit(f"acceptance failed: {claim}")


def main() -> None:
    """Compact the same transcript twice: once by masking alone, once with a planned summary."""
    transcript = _transcript()
    turns_before = len(transcript.turns)
    chain = default_chain(prompt_id=PROMPT_ID)

    # ---------------------------------------------------------------------------------------
    # A budget masking alone can meet. No model call is planned, and every turn stays in place.
    # ---------------------------------------------------------------------------------------
    roomy = CompactionBudget(max_tokens=1_200, protected_recent_turns=2)
    masked_plan = chain.decide(transcript, roomy)
    _check(
        "a plan covers every turn exactly once, in order",
        [a.turn_id for a in masked_plan.actions] == [t.turn_id for t in transcript.turns],
    )
    _check("masking alone met the budget", masked_plan.budget_unmet is False)
    _check("so no model call was planned", masked_plan.summarization_requests == ())

    masked_view = CompactionExecutor().apply(transcript, masked_plan).transcript
    _check(
        "a masked turn stays in the view at its own position",
        [t.turn_id for t in masked_view.turns] == [t.turn_id for t in transcript.turns],
    )
    masked_ids = [a.turn_id for a in masked_plan.actions if a.action is Action.MASK]
    _check("and something was actually masked", len(masked_ids) > 0)
    for turn_id in masked_ids:
        stub = next(t for t in masked_view.turns if t.turn_id == turn_id).content
        _check(
            f"the stub for {turn_id} carries a digest and no excerpt",
            "sha256:" in stub and "line 1\n" not in stub,
        )

    # ---------------------------------------------------------------------------------------
    # A tighter budget. CutCtx *plans* a summarization; the caller performs it. This package
    # holds no model, no HTTP client and no prompt text, and could not perform one (ADR-0052).
    # ---------------------------------------------------------------------------------------
    tight = CompactionBudget(max_tokens=1_000, protected_recent_turns=2)
    plan = chain.decide(transcript, tight)
    _check("the tighter budget needed a summary", len(plan.summarization_requests) == 1)
    _check("the plan fits it", plan.budget_unmet is False)
    _check(
        "and it says so honestly",
        plan.tokens_after_estimate <= tight.max_tokens < plan.tokens_before,
    )

    summaries = {}
    for request in plan.summarization_requests:
        _check(
            f"request {request.group_id} names a prompt record, not prompt text",
            request.prompt_id == PROMPT_ID,
        )
        # Hand-supplied, which is the shape a real caller's is: text it obtained from its own
        # governed inference path and handed back.
        summaries[request.group_id] = (
            "Earlier: the user asked six questions and six files were read. The build "
            "regression came from a changed cache key."
        )

    compacted = CompactionExecutor().apply(transcript, plan, summaries)
    view, report = compacted.transcript, compacted.report

    _check("the input transcript is not mutated", len(transcript.turns) == turns_before)
    _check("the view is smaller", len(view.turns) < turns_before)
    _check("the system turn survived", any(t.turn_id == "s1" for t in view.turns))
    _check("the protected tail survived", {"t6", "u9"} <= {t.turn_id for t in view.turns})
    _check(
        "the supplied summary is in the view, labelled as one",
        any(t.metadata.get("cutctx.kind") == "summary" for t in view.turns),
    )
    _check(
        "the report repeats the plan's figures rather than recomputing them",
        report.tokens_after_estimate == plan.tokens_after_estimate
        and report.tokens_before == plan.tokens_before,
    )
    _check("the report links to the plan by hash", report.plan_hash == plan.plan_hash())
    _check(
        "and it names which policies produced the view",
        "observation_masking" in report.policy_name and "summarizing" in report.policy_name,
    )
    _check(
        "the same inputs give a byte-identical plan",
        chain.decide(transcript, tight).canonical_json() == plan.canonical_json(),
    )

    print(
        f"\nmasking alone: {masked_plan.tokens_before} -> "
        f"{masked_plan.tokens_after_estimate} tokens, {len(masked_ids)} results stubbed."
    )
    print(
        f"with a summary: {plan.tokens_before} -> {plan.tokens_after_estimate} tokens, "
        f"{turns_before} turns -> {len(view.turns)}."
    )
    print("All acceptance checks passed.")


if __name__ == "__main__":
    main()
