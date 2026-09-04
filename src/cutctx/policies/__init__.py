"""The shipped compaction policies.

Three of them so far; the chain that composes them lands next. Every one builds its plan through
:func:`cutctx._invariants.build_plan`, which is where every rule of spec §11 lives, and
``tests/unit/test_packaging.py::test_no_shipped_policy_builds_its_own_plan`` scans this directory to
say so at CI time.

* :class:`~cutctx.policies.masking.ObservationMaskingPolicy` — stubs old tool-result bodies,
  keeping the reasoning. Removes nothing, so it orphans nothing.
* :class:`~cutctx.policies.summarizing.SummarizingPolicy` — **plans** one summarization of the
  oldest contiguous span and never performs it (ADR-0052).
* :class:`~cutctx.policies.drop_oldest.DropOldestPolicy` — the deterministic last resort.
"""

from __future__ import annotations

from cutctx.policies.drop_oldest import DropOldestPolicy
from cutctx.policies.masking import DEFAULT_PLACEHOLDER, ObservationMaskingPolicy
from cutctx.policies.summarizing import GROUP_ID_PREFIX, SummarizingPolicy

__all__ = [
    "DEFAULT_PLACEHOLDER",
    "GROUP_ID_PREFIX",
    "DropOldestPolicy",
    "ObservationMaskingPolicy",
    "SummarizingPolicy",
]
