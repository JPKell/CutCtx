# Contributing to CutCtx

This repository is one component of the Local AI Suite. Before changing anything, read
`docs/packages/cutctx/spec.md` and the current
phase in `development-plan.md` — both are in this repository's `docs/` folder, copied from the suite's
central documentation set so this repository can be worked on independently.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Required reading, in order

1. This component's spec — purpose, scope, non-goals, and especially §11, the public contracts
   `cutctx._invariants` exists to enforce.
2. `development-plan.md` in the same folder — the phase you are implementing, its acceptance
   criteria and its tests.
3. `src/cutctx/_invariants.py`'s module docstring — the rules, how contract 3 is read, and why
   protection propagates through a tool exchange.

## Rules that apply to every change here

* Follow the architecture's dependency direction.
  This repository's `.importlinter` enforces it in CI; do not weaken that file to make an import work.
* **Purity is the product.** No I/O of any kind: no network, no filesystem, no database, no
  environment, no clock, no randomness, no logging. Five `import-linter` contracts and an
  import-allowlist test enforce it; do not weaken either to make an import work.
* **Every policy routes through `cutctx._invariants`.** A `CompactionPlan` cannot be constructed
  without the transcript and budget it is a plan *for*, and construction validates against them —
  so there is no unvalidated plan. Build yours with `_invariants.build_plan`, which also does the
  token arithmetic and the `budget_unmet` determination; a test asserts that every module under
  `policies/` does.
* **An estimate is never presented as a count** (ADR-0016): token figures are estimates, and the
  character-ratio default's ratio rides on the plan that used it.
* **Prompts are named, never carried** — `prompt_id` refers to a versioned record in the
  *application's* prompt pack. No prompt text belongs in this package.
* **Plans are byte-identical on re-derivation.** Nothing may depend on dict insertion order, on a
  clock, or on randomness. Changing a policy's behaviour bumps its `version` and its goldens.
* Every phase's acceptance criteria in `development-plan.md` must be demonstrable, not merely
  test-covered — the plan states what to run and what a person should see.

## Before opening a pull request

```bash
ruff format --check .
ruff check .
mypy src tests
lint-imports
pytest -m "not live and not performance"
```

All of the above run in CI (`.github/workflows/ci.yml`); a red CI run blocks merge. Coverage floor
is **95 %** (a shared package, not an application).

### Replaying a failing property test

The invariant suite is property-based. `pytest-randomly` reseeds every test, so a failure is not
reproducible from the command line alone — but the hypothesis profile in `tests/conftest.py` sets
`print_blob=True`, so each failure prints a `@reproduce_failure(...)` decorator. Paste it onto the
failing test to replay that exact example; the `--randomly-seed=` line pytest prints reproduces the
*ordering*. A failure that only appears under one seed is a real bug, never a reason to pin one.

New properties belong beside the existing ones, and new generators in `tests/strategies.py`. Build
transcripts **by construction**, never by `filter`: heavy filtering gives flaky health-check
failures and useless shrinking. And state a property against `tests/oracles.py` — the untouchable
set and the tool exchanges restated from the spec — rather than against the implementation's own
helpers, or the property moves with the bug.

## Commit style

Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `perf:`, `build:`,
`ci:`), with `!` or a `BREAKING CHANGE:` footer for breaking changes. Update `CHANGELOG.md` under
`## [Unreleased]` for any user-visible change.
