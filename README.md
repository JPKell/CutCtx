# CutCtx

Deterministic transcript compaction for the Local AI Suite. Given a transcript and a token budget,
CutCtx decides which turns to **keep, mask, summarize or drop**, and applies that decision to
produce a compacted view plus an auditable account of what was done to it.

**Status: Phase 2 complete, `0.1.0` — on PyPI.** The transcript model, the invariants, the
plan/executor split and all four shipped policies — `ObservationMaskingPolicy`,
`SummarizingPolicy`, `DropOldestPolicy` and `PolicyChain` — are built and gated.

* Import name and distribution name: `cutctx`
* Runtime dependencies: `baseaicore`, and nothing else
* Python: 3.12+
* Specification: [`docs/packages/cutctx/spec.md`](docs/packages/cutctx/spec.md) ·
  plan: [`docs/packages/cutctx/development-plan.md`](docs/packages/cutctx/development-plan.md)

## The shape: plan → fulfil → apply

This is the one thing to understand before using the package, and it is friction on purpose.

```python
from cutctx import CompactionBudget, CompactionExecutor, DropOldestPolicy
from cutctx import Role, Transcript, TranscriptTurn

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
plan.tokens_before, plan.tokens_after_estimate, plan.budget_unmet
# (987, 37, False)

view = CompactionExecutor().apply(transcript, plan)
view.transcript.turn_ids()  # ('s', 'a2') — the tool exchange left as a unit
view.report.dropped_turn_ids  # ('u1', 'a1', 't1')
transcript.turn_ids()  # ('s', 'u1', 'a1', 't1', 'a2') — the input is untouched
```

**CutCtx never calls a model.** A policy that wants a span summarized does not summarize it; it
puts a `SummarizationRequest(group_id, turn_ids, target_tokens, prompt_id)` in the plan. *You*
fulfil it — through your own governed inference path, on a local tier if the transcript is
confidential — and pass the text back:

```python
view = executor.apply(transcript, plan, summaries={"g1": "Earlier: the user asked about …"})
```

Forget that middle step and you get `SummaryMissing` naming the group, not a quietly shorter
prompt. That is the whole reason the two-phase protocol exists
([ADR-0052](docs/packages/cutctx/spec.md)): a package below the applications must not hold a
second, ungoverned path to a model, and separating the decision from the expensive part is what
makes the decision auditable *before* anything is spent.

## What it guarantees

| | |
|---|---|
| **A plan is a view, never a deletion** | `apply()` returns a new transcript; the input is unchanged and what you retain is your call |
| **The untouchable set is untouchable** | Every `SYSTEM` turn, every `pinned` turn and the `protected_recent_turns` tail are kept under every policy |
| **Tool exchanges travel together** | A call and its results are dropped as a unit or summarized into one group — never separated |
| **The budget outcome is trichotomous** | It fits, or the plan says `budget_unmet`, or `BudgetUnsatisfiable` names *both* numbers. There is no plan that claims to fit while being over, and never a silent truncation |
| **Plans are byte-identical on re-derivation** | Same transcript, budget and configuration ⇒ the same bytes, on every platform — because a plan appears in an audit record |
| **An estimate is never a count** | Token figures are estimates; the character-ratio default's ratio rides on the plan that used it |
| **Purity is proven, not claimed** | No model, no HTTP client, no database, no filesystem, no clock, no logging — asserted by an import-graph test and by `import-linter` |
| **A masked stub carries a hash, never an excerpt** | The original may hold a secret, and a "first 200 characters" preview is a leak with an extra step |
| **A summarization is planned, never performed** | The package emits a `SummarizationRequest` naming turns, a token target and a *prompt id*; the caller fulfils it through its own inference path |
| **A chain composes over a projection** | Applying needs summaries, and the package produces none — so composition works from what a plan *would* produce, and one plan is built once against the real transcript |

## The four shipped policies

| Policy | What it does | What it will not do |
|---|---|---|
| `ObservationMaskingPolicy` | Replaces old `TOOL` bodies with a labelled stub carrying the original's digest and estimate | Touch a non-`TOOL` turn, mask within the `keep_recent_results` floor, or make a turn bigger than it was |
| `SummarizingPolicy` | Plans one summarization of the oldest contiguous unpinned span | Split an exchange, take a span shorter than `min_span_turns`, or produce prompt text |
| `DropOldestPolicy` | Drops whole exchanges, oldest first — the deterministic last resort | Split an exchange, or truncate silently |
| `PolicyChain` / `default_chain` | Runs them in order, stopping when the budget fits | Apply an intermediate plan, or build more than one plan |

`default_chain(prompt_id=…)` is masking → summarizing → drop-oldest, and the order is an argument
about cost: masking is free, summarizing costs a model call but keeps the substance, dropping keeps
nothing at all.

## Acceptance

`acceptance/plan_and_apply.py` is spec §20 criterion 2 as a runnable check: it plans and applies a
compaction with a hand-supplied summary, using only `cutctx` and `baseaicore`. It exits non-zero
when a claim fails, so it is a check rather than a demonstration.

```bash
python -m venv /tmp/cutctx-acceptance
/tmp/cutctx-acceptance/bin/pip install .
/tmp/cutctx-acceptance/bin/python acceptance/plan_and_apply.py
```

The rules live in one module, `cutctx._invariants`, and **there is no way to build a plan that
skips it**: `CompactionPlan`'s constructor takes the transcript and budget the plan is *for* and
validates against them before the object exists.

## Consumers

* **PromptCadence** — per-turn transcript compaction, above `compaction.threshold` of the tier's
  context budget; emits the `CompactionReport` as the `context.compacted` event body.
* **IdeaPress** — `project_review` and stage-context assembly, whose documented reduction order
  ("research notes → distant unit summaries → adjacent unit summaries", with the unit
  specification and requirements never dropped) is a compaction policy currently written in
  English. The mapping from its context rows onto `TranscriptTurn` is in `cutctx.types`'s module
  docstring.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

The full gate, identical to CI:

```bash
ruff format --check . && ruff check . && mypy src tests && lint-imports \
  && pytest -m "not live and not performance"
```

Property tests use `hypothesis` (a development dependency; the runtime budget stays at zero
non-suite packages). A failing example prints a `@reproduce_failure` decorator you can paste onto
the test to replay it — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
