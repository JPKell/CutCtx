# CutCtx — Development Plan

**Sequence position:** PromptCadence arc, stream P ([roadmap §4](../../roadmap/promptcadence-roadmap.md)).
Depends on `baseaicore>=0.4.1` (for nothing beyond the error base and clock helpers — the
`DataClassification` addition is not used here, but the arc pins one baseline).
**Target:** `cutctx 0.1.0` at the end of Phase 2; `0.2.0` (hardened) before PromptCadence 1.0.

The ordering principle is the suite's usual one: the deterministic core and its invariants first,
so PromptCadence's loop can be built against a compaction library whose behaviour is already
golden-locked; the policy variety second.

---

## Phase 1 — Transcript model, plan, executor, DropOldestPolicy

**Goal:** the vocabulary and the invariants exist and are enforced; one honest policy works end to
end.

**Prerequisites:** the D-8 ADR accepted (compaction is a view; packages plan summarization,
applications execute it — [roadmap §2](../../roadmap/promptcadence-roadmap.md)).

**Work**
* Repository skeleton (standard toolchain: hatchling, ruff, mypy strict, import-linter, pytest).
* `types.py`: `TranscriptTurn`, `Transcript`, `CompactionBudget`, `TurnAction`,
  `SummarizationRequest`, `CompactionPlan`, `CompactedTranscript`, `CompactionReport`.
* `estimator.py`: `TokenEstimator` protocol, `CharRatioEstimator` with the ratio recorded on
  plans.
* `executor.py`: `CompactionExecutor.apply` with immutability, mismatch detection and
  `SummaryMissing`.
* `policies/drop_oldest.py`: the deterministic last resort, tool pairs dropped together.
* `errors.py` per spec §7.
* Invariant validation shared by all policies (system/pinned/protected untouchable; pair
  integrity) in one place, `_invariants.py`, so no future policy can skip it.

**Files/subsystems**
```text
src/cutctx/{__init__,__about__,types,estimator,executor,errors,_invariants}.py
src/cutctx/policies/{__init__,drop_oldest}.py
tests/unit/{test_types,test_estimator,test_executor,test_drop_oldest,test_invariants}.py
tests/property/test_invariants_property.py     # hypothesis
```

**Tests**
* Invariants property-based over random transcripts: no plan from any policy touches the
  untouchable set or separates a pair.
* `BudgetUnsatisfiable` with the numbers when pinned + protected exceed the budget.
* Executor immutability; `PlanTranscriptMismatch`; report totals equal plan estimates.
* Determinism: repeated runs byte-identical; goldens committed.

**Acceptance criteria**
1. `mypy --strict` clean; a plan for a 200-turn transcript in ≤ 50 ms.
2. The invariant module is the only path to a valid plan (policies construct plans through it —
   asserted structurally).
3. Coverage ≥ 95 %.

**Known risks:** designing `TranscriptTurn` PromptCadence-shaped. Mitigated by writing the IdeaPress
mapping sketch (units → turns) into the docstrings now, before PromptCadence exists to bias it.
**Likely failure modes:** token estimates drifting between plan and report; pair detection
missing multi-call turns.
**Gold standards:** deterministic pure core; unsupported-safe honesty about estimates.
**Deferred:** masking, summarizing, chains.

---

## Phase 2 — ObservationMaskingPolicy, SummarizingPolicy, PolicyChain — publish 0.1.0

**Goal:** the shipped policy set and composition; the package PromptCadence's beta consumes.

**Prerequisites:** Phase 1.

**Work**
* `policies/masking.py`: mask tool-result bodies beyond the N most recent; stub carries hash +
  original estimate.
* `policies/summarizing.py`: oldest contiguous unpinned span → one summary turn +
  `SummarizationRequest`; `target_ratio`, `min_span_turns`.
* `policies/chain.py`: run in order, stop when the budget fits; `budget_unmet` flag when
  exhausted.
* README, quickstart with a hand-supplied-summary example; publish `cutctx 0.1.0`.

**Tests**
* Masking keeps reasoning turns byte-identical; only TOOL bodies stubbed; N respected.
* Summarizing groups are contiguous, unpinned, and meet `min_span_turns`; the request's
  `target_tokens` honours the ratio.
* Chain ordering golden: masking reduces enough → no summarization planned; chain exhausted →
  `budget_unmet`, never truncation.
* Cross-matrix determinism goldens for all policies and the default chain.

**Acceptance criteria**
1. Spec §20 criteria 2–4 met; `cutctx 0.1.0` published and installable standalone.
2. The default chain compacts the Phase-1 golden transcripts to every budget in the golden set
   with byte-identical plans.

**Known risks:** a summarize/mask boundary that leaves the model a confusing transcript. Mitigated
by stub and summary-turn formats being explicit, labelled and golden-tested, and by PromptCadence's live
phase feeding real transcripts back before 0.2.0.
**Likely failure modes:** nondeterminism via dict ordering in group assembly; masking a result
whose call was already dropped.
**Gold standards:** shipped policies deterministic and explicable; refusal (`budget_unmet`) over
silent truncation.
**Deferred:** embedding-relevance policy; note extraction; IdeaPress-shaped shipped policy
(0.2.0 hardening happens inside PromptCadence P8, driven by real transcripts).
