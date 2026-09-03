"""The shipped compaction policies.

Phase 1 ships one: :class:`~cutctx.policies.drop_oldest.DropOldestPolicy`, the deterministic last
resort. Row E1 adds ``ObservationMaskingPolicy``, ``SummarizingPolicy`` and ``PolicyChain``; each
is built the same way, through :func:`cutctx._invariants.build_plan`, which is where every rule of
spec §11 lives.
"""

from __future__ import annotations

from cutctx.policies.drop_oldest import DropOldestPolicy

__all__ = ["DropOldestPolicy"]
