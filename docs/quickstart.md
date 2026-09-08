# CutCtx — Quickstart

CutCtx decides which turns of a transcript to keep, mask, summarize or drop against a token
budget, and applies that decision to produce a compacted view plus an auditable report. It never
calls a model itself — a policy that wants a span summarized emits a `SummarizationRequest` and the
caller fulfils it.

## Install

```bash
pip install cutctx
```

The only runtime dependency is `baseaicore`.

## Run it

[`quickstart.py`](quickstart.py) in this directory is a standalone script that builds a five-turn
transcript, plans a compaction with `DropOldestPolicy`, and applies it. It needs nothing but
`cutctx`: no server, no model, no configuration file.

```bash
pip install cutctx
python docs/quickstart.py
```

Its real output:

```text
kept: ('s', 'a2')
dropped: ('u1', 'a1', 't1')
tokens before/after: 987 37
```

The tool exchange (`a1`, its call, and `t1`, the result) is dropped as one unit, never split — the
`SYSTEM` turn and the most recent turn are always kept. See [the README](../README.md) for the full
plan → fulfil → apply protocol, including how a policy asks for a summary instead of a drop.
