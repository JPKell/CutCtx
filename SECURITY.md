# Security Policy

CutCtx is part of the Local AI Suite, whose default posture is local-first. This component is a
**pure library with no I/O of any kind**: it opens no socket, reads no file, reads no environment
variable, touches no database, reads no clock and writes no log. It has no network surface of its
own, and an import-graph test plus five `import-linter` contracts assert that it stays that way
([ADR-0052](docs/packages/cutctx/spec.md), gold standards §2).

That absence is itself the security property. CutCtx sits below the applications, where a path to
a model would be a second inference path that no budget debits and no egress policy governs — so
summarization crosses the boundary as a request object and the application executes it through its
own governed path. There is no HTTP client here to repurpose.

See the suite's security standards for the full trust-boundary model, and ADR-0026 (local HTTP
hardening) for the design the applications that consume this package follow.

## Handling of transcript content

Transcript content is **untrusted model output** (spec §14) and is treated as opaque text:

* never parsed, never interpolated, never executed;
* never quoted in an error message, in a `CompactionReport`, or in `details` — turns are named by
  id, and `details` travels into API error envelopes;
* a mask stub carries a hash of the original, never a secret-bearing excerpt;
* no prompt text appears anywhere in the package. A prompt is named by `prompt_id`
  ([ADR-0012](docs/packages/cutctx/spec.md)), and a test refuses long string literals in `src/`.

A `CompactionReport` is safe to log and to emit as an event body: it is scalars, token estimates
and turn ids.

## Reporting a vulnerability

Please do not open a public issue for a suspected security vulnerability.

Instead, report it privately to the maintainer with:

* A description of the issue and its potential impact.
* Steps to reproduce, including the transcript shape, the budget and the policy configuration
  involved — this package's whole configuration surface is its constructor arguments (spec §12).
  Please redact transcript content; a turn's id, role, `pinned` flag, `tool_call_id` and
  `token_estimate` are enough to reproduce anything CutCtx does, because it never reads content.
* The installed package version (`pip show cutctx`). This package ships no CLI.

You should expect an acknowledgement within a reasonable time and, once a fix is available, credit
in the release notes unless you ask otherwise.

## Scope

In scope: this repository's own code and its documented configuration surface. Vulnerabilities in
the operating system or in a third-party dependency should be reported to that project directly —
`pip-audit` runs in this repository's CI to catch known vulnerable dependency versions, over the
locked sets rather than over the job's own environment.

Out of scope by construction: anything requiring CutCtx to perform I/O, since it performs none.
A report that depends on it doing so is a report that one of the five boundary contracts has been
weakened, and that is the finding.
