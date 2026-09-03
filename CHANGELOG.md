# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0: `0.x`).

## [Unreleased]

Phase 1 of the [development plan](docs/packages/cutctx/development-plan.md) — the vocabulary, the
invariants and one honest policy. Nothing is published yet; `cutctx 0.1.0` ships at the end of
Phase 2.

### Added

* **Transcript model** — `TranscriptTurn`, `Transcript`, `Role`, `Action`, `CompactionBudget`,
  with `tool_call_id` as a *correlation* id so that a multi-call assistant turn and all of its
  results form one tool exchange.
* **Plan vocabulary** — `TurnAction`, `TurnReplacement`, `SummarizationRequest`, `CompactionPlan`,
  `CompactedTranscript`, `CompactionReport`, `CompactionPolicy`. `CompactionReport` is the body of
  the suite's `context.compacted` event (spec §11 contract 7) and carries no timestamp.
* **`cutctx._invariants`** — spec §11 contracts 1–3 in one place, plus the token arithmetic and
  the `budget_unmet` determination. `CompactionPlan` takes the transcript and budget it is a plan
  *for* as `InitVar`s and validates against them at construction, so there is no path to an
  unvalidated plan.
* **`BudgetUnsatisfiable`** naming both numbers, `SummaryMissing` naming the group,
  `PlanTranscriptMismatch` naming the difference — all under `CompactionError`, all subclassing
  `baseaicore.SuiteError`.
* **`CharRatioEstimator`** (`chars_per_token=4.0`) behind the `TokenEstimator` protocol, with the
  ratio recorded on plans that used it and never presented as a count.
* **`CompactionExecutor.apply`** — masking, summary substitution, `summaries` fulfilment, and a
  report whose figures are copied from the plan rather than recomputed.
* **`DropOldestPolicy`** — drops whole tool exchanges, oldest first, until the budget fits;
  `budget_unmet` when it cannot, `BudgetUnsatisfiable` when nothing could.
* Property-based invariant suite (`hypothesis`), determinism goldens, an import-graph purity test,
  and the spec §15 performance targets behind the `performance` marker.

[Unreleased]: https://github.com/JPKell/CutCtx/commits/main
