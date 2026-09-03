# CutCtx — Specification

**Type:** Python package · **Import/distribution name:** `cutctx` · **Layer:** 3 (capability package)
**Status:** Specified, not implemented. Part of the PromptCadence arc
([roadmap](../../roadmap/promptcadence-roadmap.md)); decision record D-8 (compaction is a view; packages
plan summarization, applications execute it).

---

## 1. Purpose

Be the suite's one answer to "this transcript has outgrown the window it must fit". Given a
transcript and a token budget, decide — deterministically, explicably and without touching a model
or a database — which turns to keep, mask, summarize or drop, and produce a compacted view plus an
auditable account of what was done to it. Two consumers have this problem today: PromptCadence's growing
multi-turn agent transcripts, and IdeaPress's `project_review` stage, which reasons over all
committed units of a project and already documents a hand-rolled reduction order
([Workflows](../../apps/ideapress/workflows.md): "research notes → distant unit summaries →
adjacent unit…") — exactly a compaction policy, currently expressed as prose.

## 2. Scope

* A generic transcript representation: `Transcript`, `TranscriptTurn` — roles, content, tool
  payloads, token estimates, pins, metadata. Generic on purpose: PromptCadence maps its turns into it,
  IdeaPress maps units and notes into it.
* The `CompactionPolicy` protocol and shipped policies: `ObservationMaskingPolicy`,
  `SummarizingPolicy`, `DropOldestPolicy`, plus `PolicyChain` composition.
* `CompactionPlan`: the value object naming what happens to each turn and the token estimates
  before and after.
* `CompactionExecutor`: applies a plan (with any summaries the caller produced) to yield a
  `CompactedTranscript`.
* Token estimation via an injected `TokenEstimator`, with a documented character-ratio default.

## 3. Explicit non-goals

* **No model calls.** `SummarizingPolicy` *plans* a summarization — which turns, what target
  budget, which prompt record — as a `SummarizationRequest` inside the plan; the application
  executes it (PromptCadence via LoadCoach, IdeaPress via its inference port) and hands the text back.
  A package below the applications must not contain a second, ungoverned path to a model; ModelRack
  is the suite's only model client and CutCtx does not import it.
* **No persistence, no database, no I/O of any kind.** Pure functions over values.
* **No deletion semantics.** A plan describes a *view*; whether originals are retained is the
  caller's data-ownership decision (PromptCadence retains everything).
* No tokenizer. Token counts are estimates from the injected estimator, labelled as such.
* No prompt text. The summarization prompt is named by `prompt_id`/version; the record lives in
  the application's prompt pack ([ADR-0012](../../adr/0012-prompt-storage-format.md)).

## 4. Responsibilities

| Responsibility | Detail |
|---|---|
| Transcript model | Roles, content, tool call/result pairing, per-turn token estimate, `pinned`, opaque `metadata` |
| Policies | Decide `keep / mask / summarize / drop` per turn against a budget, deterministically |
| Composition | `PolicyChain` runs policies in order until the budget fits or the chain is exhausted |
| Plan | Which turns, what action, estimate before/after, the `SummarizationRequest`s required |
| Execution | Apply a plan + supplied summaries → `CompactedTranscript` + a `CompactionReport` |
| Invariants | The rules in §11, enforced by validation, not convention |

## 5. Dependencies

`baseaicore`. Nothing else.

## 6. Consumers

PromptCadence (per-turn transcript compaction), IdeaPress `project_review` and stage-context assembly
(adoption phase, [roadmap §6](../../roadmap/promptcadence-roadmap.md)).

## 7. Public API

```python
@dataclass(frozen=True, slots=True)
class TranscriptTurn:
    turn_id: str
    role: Role                        # SYSTEM | USER | ASSISTANT | TOOL
    content: str
    token_estimate: int               # from the caller's estimator; units in the name
    tool_call_id: str | None = None   # links an assistant tool call to its TOOL result
    pinned: bool = False              # never masked, summarized or dropped
    metadata: Mapping[str, str] = …   # opaque to policies

@dataclass(frozen=True, slots=True)
class Transcript:
    turns: tuple[TranscriptTurn, ...]
    def token_estimate(self) -> int: ...

class TokenEstimator(Protocol):
    def estimate_tokens(self, text: str) -> int: ...

CharRatioEstimator(chars_per_token: float = 4.0)   # the documented default; ratio recorded on plans

@dataclass(frozen=True, slots=True)
class CompactionBudget:
    max_tokens: int
    protected_recent_turns: int = 4   # a tail window policies may not touch

class CompactionPolicy(Protocol):
    name: str
    version: str
    def decide(self, transcript: Transcript, budget: CompactionBudget) -> CompactionPlan: ...

ObservationMaskingPolicy(keep_recent_results: int = 2, placeholder: str = …)
    # masks TOOL-result bodies beyond the N most recent, keeping a labelled stub with the
    # original's hash and token estimate — reasoning stays, bulk goes
SummarizingPolicy(prompt_id: str, target_ratio: float = 0.2, min_span_turns: int = 4)
    # replaces the oldest contiguous unpinned span with one summary turn; emits a
    # SummarizationRequest the caller must fulfil
DropOldestPolicy()
    # drops oldest unpinned turns (tool pairs together) until the budget fits — the
    # deterministic last resort
PolicyChain(policies: Sequence[CompactionPolicy])

@dataclass(frozen=True, slots=True)
class TurnAction:
    turn_id: str
    action: Action                    # KEEP | MASK | SUMMARIZE | DROP
    summary_group: str | None         # which SummarizationRequest consumes it

@dataclass(frozen=True, slots=True)
class SummarizationRequest:
    group_id: str
    turn_ids: tuple[str, ...]
    target_tokens: int
    prompt_id: str                    # a versioned prompt record name, never prompt text

@dataclass(frozen=True, slots=True)
class CompactionPlan:
    actions: tuple[TurnAction, ...]
    summarization_requests: tuple[SummarizationRequest, ...]
    tokens_before: int
    tokens_after_estimate: int
    policy_name: str
    policy_version: str
    estimator_ratio: float | None     # set when the char-ratio default estimated

class CompactionExecutor:
    def apply(self, transcript: Transcript, plan: CompactionPlan,
              summaries: Mapping[str, str] = …) -> CompactedTranscript: ...
    # raises SummaryMissing when a plan's request has no supplied summary

@dataclass(frozen=True, slots=True)
class CompactedTranscript:
    transcript: Transcript            # the view to send
    report: CompactionReport          # what was done: counts, ids, before/after — the audit event body

# Errors (subclass baseaicore.SuiteError)
CompactionError                       COMPACTION_ERROR
├── BudgetUnsatisfiable               COMPACTION_BUDGET_UNSATISFIABLE   # pinned+protected > budget
├── SummaryMissing                    COMPACTION_SUMMARY_MISSING
└── PlanTranscriptMismatch            COMPACTION_PLAN_MISMATCH          # plan built for other turns
```

## 8. Inputs

A `Transcript`, a `CompactionBudget`, policy configuration, and (at execution) the summaries the
caller produced for the plan's requests.

## 9. Outputs

A `CompactionPlan`, a `CompactedTranscript` with its `CompactionReport`, typed errors.

## 10. Data ownership

None. CutCtx never persists anything; the caller stores plans and reports (PromptCadence's
`compactions` table, IdeaPress's stage records).

## 11. Public contracts

1. **A plan is a view, never a deletion.** Applying a plan does not mutate the input transcript;
   the caller decides retention.
2. **The system turn and pinned turns are untouchable**, and the `protected_recent_turns` tail is
   never masked, summarized or dropped. A budget smaller than the untouchable set raises
   `BudgetUnsatisfiable` with the numbers — it is never "solved" by violating the invariant.
3. **A tool call and its result travel together**: masked together, summarized in the same group,
   or dropped as a pair — never separated, because an orphaned call or result is a malformed
   transcript to every provider.
4. **Determinism.** Same transcript + budget + policy configuration ⇒ byte-identical plan, on
   every platform and Python version — golden-tested, because plans appear in audit records.
5. **Honest estimates.** Every plan carries `tokens_before`, `tokens_after_estimate` and the
   estimator ratio when the character default produced them; an estimate is never presented as a
   count.
6. **No model access.** The package imports no HTTP client and no provider; summarization crosses
   the boundary as a request object. A test asserts the import graph.
7. `CompactionReport` is exactly the body of the suite's `context.compacted` event, so consumers
   emit it without reshaping.

## 12. Configuration

Constructor arguments only. CutCtx reads no environment and no files.

## 13. Error behaviour

| Condition | Error |
|---|---|
| Pinned + protected turns alone exceed the budget | `BudgetUnsatisfiable`, naming both numbers |
| Plan applied to a transcript with different turn ids | `PlanTranscriptMismatch` |
| A `SummarizationRequest` with no supplied summary | `SummaryMissing`, naming the group |
| Chain exhausted and still over budget | The plan is returned with `tokens_after_estimate` over budget and a `budget_unmet` flag — the caller decides (PromptCadence halts with `COMPACTION_FAILED`); never a silent truncation |

## 14. Security considerations

Transcript content is untrusted model output; policies treat it as opaque text — no parsing, no
interpolation, no execution. Masked stubs carry a hash of the original, never a secret-bearing
excerpt. The package logs nothing.

## 15. Performance

| Measure | Target |
|---|---|
| Plan for a 200-turn transcript | ≤ 50 ms |
| Plan for a 2 000-turn transcript | ≤ 500 ms |
| Apply, 200 turns | ≤ 10 ms |
| Memory | O(transcript); no copy of content except the compacted view |

## 16. Cross-platform

Pure Python; fully portable. Determinism asserted across the CI matrix.

## 17. Observability

No logging. The `CompactionReport` carries everything a consumer logs or emits.

## 18. Test strategy

| Area | Tests |
|---|---|
| Invariants | System/pinned/protected untouched under every policy; tool pairs never separated; property-based (hypothesis) over random transcripts |
| Policies | Each policy against golden transcripts; masking stubs carry hash + estimate; summarize groups contiguous spans only; drop is last resort in the default chain |
| Determinism | Byte-identical plans across repeated runs and across the CI matrix |
| Budget maths | Fits, near-misses, unsatisfiable, chain-exhausted with `budget_unmet` |
| Executor | Summary substitution, missing summary, transcript mismatch, immutability of input |
| Errors | Every §13 row produced |

Coverage floor: **95 %**.

## 19. Compatibility and versioning

Semver, pre-1.0 `0.x`. Policy `version` strings are part of the audit record: changing a policy's
behaviour bumps its version, and golden plans for old versions are kept for the life of a major.

## 20. Acceptance criteria

1. PromptCadence compacts a 100-turn transcript to a 16 384-token budget through the default chain, with
   the summary produced by its own LoadCoach call, and the report reconstructs exactly what
   happened.
2. A standalone script using only `cutctx` + `baseaicore` plans and applies a compaction
   with a hand-supplied summary — no suite application installed.
3. Determinism goldens pass across the CI matrix.
4. `mypy --strict`, `ruff`, `lint-imports` clean; coverage ≥ 95 %.

## 21. Future extensions

* An embedding-relevance policy (keep turns similar to the current step), once an embeddings API
  exists in ModelRack — arrives as another `CompactionPolicy`, no core change.
* IdeaPress's stage-context assembly expressed as a shipped policy, after its adoption phase
  proves the mapping.
* Structured note extraction (`NoteExtractionPolicy` from the skeleton) — deferred until a
  consumer defines what a "note" is; a policy without a consumer-defined output shape is
  speculation.
