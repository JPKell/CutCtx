"""A ten-line CutCtx quickstart: plan a compaction, then apply it.

Needs nothing but `pip install cutctx`. No server, no model, no configuration file.
"""

from cutctx import (
    CompactionBudget,
    CompactionExecutor,
    DropOldestPolicy,
    Role,
    Transcript,
    TranscriptTurn,
)

transcript = Transcript(
    (
        TranscriptTurn("s", Role.SYSTEM, "You are a careful assistant.", 12),
        TranscriptTurn("u1", Role.USER, "What changed in the build?", 30),
        TranscriptTurn("a1", Role.ASSISTANT, "Let me look.", 20, tool_call_id="c1"),
        TranscriptTurn("t1", Role.TOOL, "<4 kB of build log>", 900, tool_call_id="c1"),
        TranscriptTurn("a2", Role.ASSISTANT, "The cache key changed.", 25),
    )
)

plan = DropOldestPolicy().decide(
    transcript, CompactionBudget(max_tokens=100, protected_recent_turns=1)
)
view = CompactionExecutor().apply(transcript, plan)

print("kept:", view.transcript.turn_ids())
print("dropped:", view.report.dropped_turn_ids)
print("tokens before/after:", plan.tokens_before, plan.tokens_after_estimate)
