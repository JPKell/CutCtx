# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0: `0.x`).

## [Unreleased]

## [0.1.0] — 2026-09-03

The first release. Phases 1 and 2 of the
[development plan](docs/packages/cutctx/development-plan.md) — the vocabulary, the invariants, and
the four shipped policies.

### Added — Phase 2, the policy set

* **`ObservationMaskingPolicy`** — replaces old `TOOL` bodies with a labelled stub carrying the
  original's **sha256 and token estimate, never an excerpt** (spec §14): the original may hold a
  secret, and a "first 200 characters" preview is a leak with an extra step. The turn stays in the
  view at its own position, which is why masking orphans nothing and is legal under contract 3.
  Reasoning turns are left byte-identical — that is the whole value of the policy, and a property
  asserts it rather than hoping for it.
* **`SummarizingPolicy`** — folds the oldest contiguous unpinned span into one planned summary turn
  and one `SummarizationRequest`. It **plans a summarization and never performs one** (ADR-0052):
  no model, no HTTP client, no prompt text, and a `prompt_id` naming a versioned record.
* **`PolicyChain`** and **`default_chain`** — masking → summarizing → drop-oldest, stopping as soon
  as the budget fits.
* **`DEFAULT_PLACEHOLDER`** and **`GROUP_ID_PREFIX`**, so a consumer reading a plan can recognise a
  shipped policy's output.
* **Cross-matrix determinism goldens** for all four policies, over the Phase-1 fixtures at every
  Phase-1 budget (dev-plan AC2) and over a second fixture set large enough to actually produce a
  stub and a summary group — because the Phase-1 transcripts are reached by keeping and dropping
  alone, and a golden set that locked in only those would lock in nothing new.
* **A property suite over every shipped policy**, against oracles restated from the spec in a
  different shape from the implementation (`maskable_ids` counts backwards where the policy slices;
  `exchange_rule_holds` keys on the correlation id with no notion of position).
* **`acceptance/plan_and_apply.py`** — spec §20 criterion 2 as a runnable check rather than a
  demonstration, exiting non-zero when a claim fails.

### Decided — Phase 2

* **Contract 3 said the wrong thing, and the document was amended.** Read literally, "masked
  together" forbade the `ObservationMaskingPolicy` §7 ships. Phase 1 enforced the reading that the
  prohibition is on *separation* — and what separates is *removal* — and left the decision to this
  row. Having built the policy, the reading is confirmed: within one exchange, either every member
  is retained (`KEEP`/`MASK`, mixed freely) or every member is removed by the same action. The spec
  now states the enforced rule.
* **A chain composes over a projection, not over an applied view.** Applying requires summaries and
  this package produces none, so the chain builds the view each plan *would* produce — a masked
  turn's replacement is in the plan, a summary turn's id is derived and its estimate is
  `target_tokens` — and lifts the next policy's decisions back onto the real transcript. **One
  plan is built, once, through `_invariants.build_plan`.** A later policy that removes a summary
  turn removes everything folded into it, and a half-removed exchange is escalated to the removing
  action before the plan is built rather than being rejected after.
* **Ordering is transcript position at every step**, never a `dict` or `set` iteration order. The
  development plan named "nondeterminism via dict ordering in group assembly" as this phase's
  likely failure mode, and it is the kind that reproduces only under an unlucky `pytest-randomly`
  seed unless the ordering is structural.
* **A summary group's id is a digest of its span's turn ids** — never a counter and never a uuid.
  A counter would renumber when a chain composed the policies differently, and the id is in the
  plan's bytes *and* in the summary turn's derived id.
* **Masking stops as soon as the budget fits**, oldest result first, the way `DropOldestPolicy`
  does; `keep_recent_results` is a floor that does not move for the budget. Masking every eligible
  result would cost the model observations for nothing.
* **A result smaller than its own stub is left alone.** A stub costs about thirty tokens, so
  masking a two-token result would *increase* the estimate, and a transcript of many tiny results
  would grow under a policy whose purpose is shrinking.
* **A planned summary's token figure is `target_tokens`** — an estimate of a text that does not
  exist yet — and the executor does not re-estimate the text it is handed. That is contract 5
  working as intended; recomputing anything in the report is exactly what makes plan and report
  able to disagree.

### Fixed — Phase 2

* **`SummarizingPolicy` folded a span even when the transcript already fitted the budget**, which
  spent a model call and replaced turns the model could still have read. Found by the property
  `test_a_transcript_that_already_fits_is_left_alone_by_every_shipped_policy`, which runs over
  every shipped policy: the other three check the budget inside a loop over reductions, and this
  one has no such loop, so it needed the check stated explicitly.

### Added — Phase 1

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
