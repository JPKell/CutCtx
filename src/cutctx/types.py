"""The vocabulary a compaction is expressed in — transcripts, budgets, plans, reports (spec §7).

Pure data. Nothing here performs I/O, reads a clock or touches a model: CutCtx decides *what*
should happen to a transcript and produces a value describing it, and the application does the
expensive half (:doc:`ADR-0052 <adr>`).

Three rules run through every type in this module.

* **A plan is a view, never a deletion** (spec §11 contract 1). Applying one returns a new
  :class:`CompactedTranscript`; the input is untouched, and what the caller retains is the
  caller's data-ownership decision.
* **No time, anywhere.** There is no timestamp on any type here and no clock is injected, because
  a plan must be byte-identical on re-derivation (contract 4) and a plan appears in an audit
  record. If a consumer's event needs a time, the consumer stamps it.
* **An estimate is never presented as a count** (ADR-0016). Every token figure in this module is
  an estimate, named ``*_estimate`` or ``*_tokens`` and carried alongside the
  :attr:`CompactionPlan.estimator_ratio` that produced it when the character-ratio default did.

Mapping a caller's world onto these types
-----------------------------------------

The transcript model is deliberately generic, because it has two consumers with different shapes
and only one of them is a chat loop. Designing it around PromptCadence's turns is the named risk
for this phase (development plan, "Known risks"), so the second consumer's mapping is written down
here, now, before PromptCadence exists to bias it.

**PromptCadence** (:doc:`lifecycle §7 <lifecycle>`) maps one thread turn to one
:class:`TranscriptTurn`: the trajectory's system prompt is the :attr:`Role.SYSTEM` turn, model and
user turns map by role, a tool result is a :attr:`Role.TOOL` turn, and
:attr:`~CompactionBudget.protected_recent_turns` guards the live tail.

**IdeaPress** (:doc:`workflows §7 <workflows>`) has no turns at all; it assembles a *stage
context* out of documents, and its documented reduction order — "research notes → distant unit
summaries → adjacent unit summaries", with the unit specification and its requirements never
dropped — is a compaction policy written in English. It maps as:

============================================  ==========================================
IdeaPress context row                          :class:`TranscriptTurn`
============================================  ==========================================
system prompt record                           ``role=SYSTEM``, ``pinned=True``
unit specification + its requirements          ``role=USER``, ``pinned=True`` — never dropped,
                                               so unsatisfiable-with-numbers falls out of
                                               :class:`~cutctx.errors.BudgetUnsatisfiable`
                                               rather than needing its own code path
project glossary + style constraints           ``role=USER``, ``pinned=True``
adjacent committed unit summaries              ``role=USER``, ``metadata={"distance": "1"}``
distant committed unit summaries               ``role=USER``, ``metadata={"distance": "3"}``
research notes                                 ``role=USER``, ``metadata={"kind": "note"}``
previous attempt's findings (repair/revision)   ``role=USER``, ``pinned=True``
============================================  ==========================================

The order of the rows *is* the transcript order, oldest-first, so that dropping "oldest first"
means dropping notes before summaries; the reduction order is a
:class:`~cutctx.policies.DropOldestPolicy`-shaped chain over that ordering, and the ``metadata``
column is how a later, IdeaPress-shaped policy would express "distant before adjacent". **No
policy reads ``metadata`` today, and none should learn to without a shipped policy that documents
the keys it consumes** (spec §4 calls it opaque).
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol

from baseaicore import ValidationError, canonical_json, sha256_of

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "Action",
    "CompactedTranscript",
    "CompactionBudget",
    "CompactionPlan",
    "CompactionPolicy",
    "CompactionReport",
    "Role",
    "SummarizationRequest",
    "Transcript",
    "TranscriptTurn",
    "TurnAction",
    "TurnReplacement",
]

EMPTY_METADATA: Final[Mapping[str, str]] = MappingProxyType({})
"""The immutable default for :attr:`TranscriptTurn.metadata`.

A read-only proxy rather than ``{}``: a frozen dataclass whose default was a mutable dict would be
frozen in name only, and every turn constructed without metadata would share one dict that any
caller could edit. It compares equal to an empty mapping, so ``turn.metadata == {}`` holds.
"""

SUMMARY_TURN_ID_PREFIX: Final = "summary:"
"""Prefix of the turn id the executor gives a summary turn: ``summary:<group_id>``.

Derived, not generated. An id from a counter or a random source would make two applications of the
same plan differ, which contract 4 forbids; a plan whose ``group_id`` would collide with an
existing turn id is refused at construction instead.
"""


class Role(StrEnum):
    """Who produced a turn.

    CutCtx's own vocabulary. ``baseaicore`` has no message or role type and is not gaining one for
    this: a role is a transcript concept, and the transcript lives here. A
    :class:`~enum.StrEnum` rather than a bare :class:`~enum.Enum`, matching every other
    enumeration in the suite — the member's value is its serialized form, so a role crossing a
    payload boundary spells itself the same way it does in code.
    """

    SYSTEM = "system"
    """The instructions the whole exchange runs under. Untouchable under every policy (spec §11
    contract 2), whether or not it is also ``pinned``, and whatever its position."""

    USER = "user"
    """A turn from the human, or from the application speaking on the human's behalf."""

    ASSISTANT = "assistant"
    """A turn from the model, including one that issues tool calls."""

    TOOL = "tool"
    """The result of a tool call. Travels with the turn that called it (contract 3)."""


class Action(StrEnum):
    """What a plan does to one turn.

    The four members split into two classes, and the split is what
    :mod:`cutctx._invariants` enforces contract 3 over:

    * **Retention** — :attr:`KEEP` and :attr:`MASK`. The turn stays in the view at its own
      position. Masking replaces a body with a stub; it orphans nothing, which is why a tool
      result may be masked while the turn that called it is kept.
    * **Removal** — :attr:`DROP` and :attr:`SUMMARIZE`. The turn leaves the view. This is the
      class that can orphan a tool call, so a removal applied to any member of a tool exchange
      must be applied to all of them, identically.
    """

    KEEP = "keep"
    """Pass the turn through unchanged."""

    MASK = "mask"
    """Substitute :attr:`TurnAction.replacement` for the turn's content, in place."""

    SUMMARIZE = "summarize"
    """Fold the turn into the :class:`SummarizationRequest` named by
    :attr:`TurnAction.summary_group`, whose fulfilled text becomes one new turn."""

    DROP = "drop"
    """Remove the turn from the view. The deterministic last resort."""


@dataclass(frozen=True, slots=True)
class TranscriptTurn:
    """One turn of a transcript, as the caller has already estimated it.

    Args:
        turn_id: The caller's identifier, unique within the transcript. Opaque to CutCtx.
        role: Who produced the turn.
        content: The turn's text. **Untrusted model output** (spec §14): CutCtx never parses,
            interpolates or executes it, and never puts an excerpt of it in an error, a report or
            a log.
        token_estimate: What this turn costs, in tokens, as estimated by the *caller*. CutCtx does
            not re-estimate a turn it was handed — the caller knows which tokenizer its provider
            uses and CutCtx does not (spec §3: no tokenizer). Must not be negative.
        tool_call_id: The correlation id of the tool exchange this turn belongs to, or ``None``
            for a turn that is not part of one. See "Multi-call turns" below.
        pinned: Untouchable under every policy — never masked, summarized or dropped.
        metadata: Caller-owned key/value context, **opaque to every policy** (spec §4). Copied
            into a read-only mapping at construction, so a caller that mutates its own dict
            afterwards cannot change a frozen turn.

    Raises:
        ValidationError: If ``turn_id`` is empty, or ``token_estimate`` is negative. Both are
            defects in the caller's mapping rather than compaction outcomes, so they refuse at the
            boundary rather than becoming a plan nobody can reproduce.

    Multi-call turns:
        One assistant turn may issue several tool calls, so the call/result relation is **not**
        one-to-one and ``tool_call_id`` is a *correlation* id, not a per-call one: every turn
        carrying the same value belongs to one **tool exchange**, and an exchange is the atomic
        unit that travels together (contract 3). An assistant turn issuing three calls and the
        three results it produced share one id.

        This is a deliberate choice over a per-call list, and it costs nothing: the unit that must
        travel together is the transitive closure of the call/result relation anyway — dropping
        two of three results would orphan the third against a call that survived — so per-call
        granularity would collapse to the same partition and buy only a graph traversal whose
        order is one more thing determinism has to pin down. Callers whose provider gives distinct
        per-call ids assign one of them, or a synthetic exchange id, to all four turns; the
        provider-side ids stay in the caller's own store, which is where the mapping back happens.

        A ``TOOL`` turn whose correlation id matches nothing else is not an error — its call may
        have been compacted away in an earlier round. It forms an exchange of one, and travels
        alone.

    Note:
        Turns are compared by value and are **not hashable**: ``metadata`` is a mapping. Index
        turns by :attr:`turn_id`, which is what every plan does.
    """

    turn_id: str
    role: Role
    content: str
    token_estimate: int
    tool_call_id: str | None = None
    pinned: bool = False
    metadata: Mapping[str, str] = EMPTY_METADATA

    def __post_init__(self) -> None:
        """Refuse a malformed turn, and freeze ``metadata`` into a read-only mapping."""
        if not self.turn_id:
            raise ValidationError(
                "A TranscriptTurn needs a non-empty turn_id: a plan names every turn by id, and "
                "an empty id cannot be told apart from another empty id.",
                details={"role": self.role.value},
            )
        if self.token_estimate < 0:
            raise ValidationError(
                f"token_estimate must not be negative; turn {self.turn_id!r} declares "
                f"{self.token_estimate}. A negative estimate would let a transcript's total fall "
                "below the sum of the turns a policy cannot touch.",
                details={"turn_id": self.turn_id, "token_estimate": self.token_estimate},
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class Transcript:
    """An ordered body of turns, oldest first.

    Order is meaning: "oldest" and the ``protected_recent_turns`` tail are both positional, and a
    plan's actions are emitted in this order so that two derivations of the same plan produce the
    same bytes.

    Args:
        turns: The turns, oldest first. May be empty.

    Raises:
        ValidationError: If two turns share a ``turn_id``. A duplicate makes "every turn id
            appears in the plan exactly once" unstatable and would let a plan act on one turn
            while a caller applied it to the other.
    """

    turns: tuple[TranscriptTurn, ...] = ()

    def __post_init__(self) -> None:
        """Refuse duplicate turn ids."""
        seen: set[str] = set()
        duplicates: list[str] = []
        for turn in self.turns:
            if turn.turn_id in seen:
                duplicates.append(turn.turn_id)
            seen.add(turn.turn_id)
        if duplicates:
            raise ValidationError(
                f"Transcript turn ids must be unique; {len(duplicates)} are repeated "
                f"(first: {duplicates[0]!r}).",
                details={"duplicate_turn_ids": sorted(set(duplicates))[:10]},
            )

    def token_estimate(self) -> int:
        """Return the sum of the turns' declared estimates.

        An estimate, never a count (ADR-0016): it is the caller's per-turn figures added up, and
        CutCtx neither produced nor checked them.
        """
        return sum(turn.token_estimate for turn in self.turns)

    def turn_ids(self) -> tuple[str, ...]:
        """Return the turn ids in transcript order."""
        return tuple(turn.turn_id for turn in self.turns)


@dataclass(frozen=True, slots=True)
class CompactionBudget:
    """The window a transcript must fit, and the tail no policy may touch.

    Args:
        max_tokens: The estimate the compacted view must not exceed. Zero is legal and means
            "nothing but an empty transcript fits" — a degenerate budget, not an error.
        protected_recent_turns: How many turns at the **end** of the transcript are untouchable —
            never masked, summarized or dropped (spec §11 contract 2). A value at or above the
            transcript's length protects the whole transcript, which is a satisfiable state only
            if the whole transcript already fits.

    Raises:
        ValidationError: If either value is negative.
    """

    max_tokens: int
    protected_recent_turns: int = 4

    def __post_init__(self) -> None:
        """Refuse a budget that could not describe any window."""
        if self.max_tokens < 0:
            raise ValidationError(
                f"max_tokens must not be negative; got {self.max_tokens}. Zero is the smallest "
                "meaningful budget and already means 'only an empty view fits'.",
                details={"max_tokens": self.max_tokens},
            )
        if self.protected_recent_turns < 0:
            raise ValidationError(
                f"protected_recent_turns must not be negative; got {self.protected_recent_turns}.",
                details={"protected_recent_turns": self.protected_recent_turns},
            )


@dataclass(frozen=True, slots=True)
class TurnReplacement:
    """The body a :attr:`Action.MASK` puts in a turn's place, and what it is estimated to cost.

    One value rather than two optional fields on :class:`TurnAction`, so that "a mask has a body
    **and** an estimate" is a shape rather than a rule someone has to remember.

    The stub's *format* belongs to the policy that writes it —
    :func:`cutctx.policies.masking.stub_for`: a labelled placeholder carrying the original's hash
    and token estimate, **never an excerpt** (spec §14). The executor builds no stub; it
    substitutes the one the plan carries.

    Args:
        content: The substitute text. Goes into the view verbatim.
        token_estimate: What the substitute is estimated to cost. Must not be negative.

    Raises:
        ValidationError: If ``token_estimate`` is negative.
    """

    content: str
    token_estimate: int

    def __post_init__(self) -> None:
        """Refuse a negative estimate."""
        if self.token_estimate < 0:
            raise ValidationError(
                f"A TurnReplacement's token_estimate must not be negative; got "
                f"{self.token_estimate}.",
                details={"token_estimate": self.token_estimate},
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped form used in plans and goldens."""
        return {"content": self.content, "token_estimate": self.token_estimate}


@dataclass(frozen=True, slots=True)
class TurnAction:
    """What a plan does to one turn.

    Args:
        turn_id: The turn this action is about.
        action: What happens to it.
        summary_group: The :attr:`SummarizationRequest.group_id` that consumes this turn. Set
            **exactly** when ``action`` is :attr:`Action.SUMMARIZE`.
        replacement: The stub substituted for the turn's body. Set **exactly** when ``action`` is
            :attr:`Action.MASK`.

    Raises:
        ValidationError: If ``summary_group`` or ``replacement`` is present without its action, or
            absent with it. A ``MASK`` with no body and a ``SUMMARIZE`` with no group are both
            plans the executor could not apply, and they are refused where they are written rather
            than where they are read.
    """

    turn_id: str
    action: Action
    summary_group: str | None = None
    replacement: TurnReplacement | None = None

    def __post_init__(self) -> None:
        """Refuse an action whose optional fields disagree with its ``action``."""
        wants_group = self.action is Action.SUMMARIZE
        if wants_group != (self.summary_group is not None):
            raise ValidationError(
                f"summary_group is set exactly when the action is SUMMARIZE; turn "
                f"{self.turn_id!r} has action {self.action.value!r} and "
                f"summary_group={self.summary_group!r}.",
                details={"turn_id": self.turn_id, "action": self.action.value},
            )
        wants_replacement = self.action is Action.MASK
        if wants_replacement != (self.replacement is not None):
            raise ValidationError(
                f"replacement is set exactly when the action is MASK; turn {self.turn_id!r} has "
                f"action {self.action.value!r} and "
                f"{'a replacement' if self.replacement else 'none'}.",
                details={"turn_id": self.turn_id, "action": self.action.value},
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped form used in plans and goldens."""
        return {
            "turn_id": self.turn_id,
            "action": self.action.value,
            "summary_group": self.summary_group,
            "replacement": None if self.replacement is None else self.replacement.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SummarizationRequest:
    """Work the caller must do before a plan can be applied — never work CutCtx does.

    The whole of :doc:`ADR-0052 <adr>` decision 3 in one value: CutCtx decides *which* turns to
    summarize, to *what* budget, under *which* prompt record, and the application executes it
    through its own governed inference path — PromptCadence via LoadCoach on the trajectory's
    cheapest admissible **local** tier, IdeaPress via its inference port — and hands the text back
    to :meth:`~cutctx.executor.CompactionExecutor.apply`.

    Args:
        group_id: Names this request within the plan, and derives the summary turn's id
            (``summary:<group_id>``). Unique within a plan.
        turn_ids: The turns folded into it, in transcript order. Never empty.
        target_tokens: The estimate the summary is planned to cost. It is what the plan's
            arithmetic counts and what the summary turn carries in the applied view — the executor
            does **not** re-estimate the text it is handed, because a report that disagreed with
            its own plan would make the plan unauditable. The figure is an estimate and is
            labelled as one (ADR-0016); a caller wanting the delivered size measures the returned
            transcript.
        prompt_id: The versioned prompt record that produces the summary — a **name**, never
            prompt text (ADR-0012). No prompt text appears anywhere in this package.

    Raises:
        ValidationError: If ``group_id`` or ``prompt_id`` is empty, ``turn_ids`` is empty or
            repeats an id, or ``target_tokens`` is negative.
    """

    group_id: str
    turn_ids: tuple[str, ...]
    target_tokens: int
    prompt_id: str

    def __post_init__(self) -> None:
        """Refuse a request no caller could fulfil."""
        if not self.group_id:
            raise ValidationError(
                "A SummarizationRequest needs a non-empty group_id: it names the request in the "
                "plan, in SummaryMissing, and in the summary turn's id.",
                details={},
            )
        if not self.prompt_id:
            raise ValidationError(
                f"Summarization request {self.group_id!r} needs a prompt_id naming a versioned "
                "prompt record (ADR-0012). CutCtx carries no prompt text.",
                details={"group_id": self.group_id},
            )
        if not self.turn_ids:
            raise ValidationError(
                f"Summarization request {self.group_id!r} covers no turns. A request that folds "
                "nothing would cost a model call and reduce nothing.",
                details={"group_id": self.group_id},
            )
        if len(set(self.turn_ids)) != len(self.turn_ids):
            raise ValidationError(
                f"Summarization request {self.group_id!r} repeats a turn id.",
                details={"group_id": self.group_id},
            )
        if self.target_tokens < 0:
            raise ValidationError(
                f"Summarization request {self.group_id!r} has a negative target_tokens "
                f"({self.target_tokens}).",
                details={"group_id": self.group_id, "target_tokens": self.target_tokens},
            )

    @property
    def summary_turn_id(self) -> str:
        """Return the id the executor gives the summary turn: ``summary:<group_id>``."""
        return f"{SUMMARY_TURN_ID_PREFIX}{self.group_id}"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped form used in plans and goldens."""
        return {
            "group_id": self.group_id,
            "turn_ids": list(self.turn_ids),
            "target_tokens": self.target_tokens,
            "prompt_id": self.prompt_id,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionPlan:
    """What a policy decided, for one transcript against one budget — the auditable artifact.

    **This class cannot be built without validation.** ``transcript`` and ``budget`` are
    :class:`~dataclasses.InitVar` s: constructing a plan requires handing over the transcript and
    budget it is a plan *for*, and ``__post_init__`` runs
    :func:`cutctx._invariants.validate_plan` over the three of them before the object exists.
    There is no unchecked path — not a private constructor a policy can find, not a factory it can
    forget to call, not a base class it can subclass past. The invariant module is the only path
    to a valid plan because it is on the only path to a plan at all (development plan Phase 1,
    acceptance criterion 2).

    The InitVars leave no trace: they are not fields, so they are absent from ``==``, from
    ``repr``, and from :meth:`to_dict`. A plan carries no transcript.

    Args:
        actions: One action per turn, in transcript order — a plan is a **total function** over
            the transcript, so a turn the plan forgot is refused rather than silently kept.
        summarization_requests: The work the caller must fulfil, ordered by the position of each
            group's earliest turn.
        tokens_before: :meth:`Transcript.token_estimate` of the transcript planned over.
        tokens_after_estimate: What the applied view is estimated to cost. Recomputed and checked
            at construction, so a plan cannot carry a number its own actions do not produce.
        policy_name: The deciding policy's name — part of the audit record.
        policy_version: The deciding policy's version. Changing a policy's behaviour bumps it
            (spec §19); golden plans for old versions are kept for the life of a major.
        estimator_ratio: The ``chars_per_token`` of the character-ratio default when it produced
            an estimate on this plan, and ``None`` when no estimate on this plan came from it —
            every plan from :class:`~cutctx.policies.DropOldestPolicy`, which only adds up figures
            the caller supplied. It rides on the plan so that an estimate is never mistaken for a
            count (ADR-0016).
        budget_unmet: ``True`` when the plan is still over budget (spec §13's last row). It is
            not a policy's opinion — construction computes it as
            ``tokens_after_estimate > budget.max_tokens`` and refuses a plan that disagrees, which
            is what makes the budget outcome trichotomous and honest: a plan fits, or it says it
            does not, or :class:`~cutctx.errors.BudgetUnsatisfiable` was raised before any plan
            existed. There is no plan that claims to fit while being over.
        transcript: The transcript this is a plan for. Validation input only; not stored.
        budget: The budget this is a plan for. Validation input only; not stored.

    Raises:
        BudgetUnsatisfiable: If the untouchable turns alone exceed ``budget.max_tokens``.
        ValidationError: If any invariant of spec §11 is broken — a turn missing from ``actions``
            or named twice, an untouchable turn acted on, a tool exchange split, a summary group
            that does not match its request, or arithmetic that does not follow from the actions.
            A broken invariant is a defect in the policy that wrote the plan, not an outcome a
            caller chose between, which is why it raises the suite's validation error rather than
            one of spec §13's outcome codes.
    """

    actions: tuple[TurnAction, ...]
    summarization_requests: tuple[SummarizationRequest, ...] = ()
    tokens_before: int
    tokens_after_estimate: int
    policy_name: str
    policy_version: str
    estimator_ratio: float | None = None
    budget_unmet: bool = False
    transcript: InitVar[Transcript]
    budget: InitVar[CompactionBudget]

    def __post_init__(self, transcript: Transcript, budget: CompactionBudget) -> None:
        """Validate the plan against the transcript and budget it was built for."""
        # Imported here rather than at module scope because `_invariants` imports this module for
        # its vocabulary; deferring the direction that closes the cycle keeps both modules
        # importable in either order. The module is already loaded by the second plan.
        from cutctx import _invariants

        _invariants.validate_plan(self, transcript, budget)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped form: the golden format, and what a caller persists.

        Keys are fixed here and inherited by every later policy. Pass the result through
        :func:`baseaicore.canonical_json` for bytes that are identical across runs, platforms and
        Python versions (gold standard G8).
        """
        return {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "tokens_before": self.tokens_before,
            "tokens_after_estimate": self.tokens_after_estimate,
            "estimator_ratio": self.estimator_ratio,
            "budget_unmet": self.budget_unmet,
            "actions": [action.to_dict() for action in self.actions],
            "summarization_requests": [
                request.to_dict() for request in self.summarization_requests
            ],
        }

    def plan_hash(self) -> str:
        """Return the sha256 of this plan's canonical JSON — the id a report is linked by."""
        return sha256_of(self.to_dict())

    def canonical_json(self) -> str:
        """Return this plan's canonical JSON. Byte-identical for equal plans (contract 4)."""
        return canonical_json(self.to_dict())


class CompactionPolicy(Protocol):
    """Decides what happens to a transcript that must fit a budget.

    A policy is a **pure function of its inputs and its own configuration**: same transcript, same
    budget, same configuration ⇒ byte-identical plan, on every platform and Python version (spec
    §11 contract 4). It performs no I/O and calls no model — a policy that needed a model call
    *during* planning is what would reopen :doc:`ADR-0052 <adr>`, and none does.

    Implementations build their plan through :func:`cutctx._invariants.build_plan`, which is where
    the token arithmetic, the ``budget_unmet`` determination and every §11 invariant live. A
    policy that constructs a :class:`CompactionPlan` directly gets the same validation — that is
    the point of the InitVars — but not the arithmetic, so it would be writing figures the
    validator is about to recompute and reject.
    """

    name: str
    """Stable identifier, recorded on every plan and report."""

    version: str
    """The behaviour's version. Bumped whenever the decisions change (spec §19)."""

    def decide(self, transcript: Transcript, budget: CompactionBudget) -> CompactionPlan:
        """Return the plan for this transcript and budget.

        Raises:
            BudgetUnsatisfiable: If the untouchable turns alone exceed the budget.
        """
        ...


class _ValidationInput:
    """The value the plan's ``InitVar`` names resolve to on the class.

    ``__dataclass_fields__`` includes ``InitVar`` entries, and introspecting pretty-printers
    (IPython's, and the copy of it hypothesis vendors for its counterexamples) iterate it and call
    ``getattr`` for every entry whose ``init`` is true. Without something to find, printing a plan
    raises ``AttributeError`` — so a failing property test about a plan would report the crash
    instead of the plan. These class attributes answer that read honestly: a plan carries no
    transcript and no budget, and this says so where a ``None`` would have implied it had one and
    lost it.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the phrase that appears wherever a plan is pretty-printed."""
        return "<validation input; not carried on the plan>"


_NOT_CARRIED: Final = _ValidationInput()

for _init_var_name in ("transcript", "budget"):
    # Set here rather than in the class body: the names are already annotated there as InitVars,
    # and an assignment beside the annotation would give the InitVar a default — which is exactly
    # the optional-argument bypass the InitVars exist to close.
    setattr(CompactionPlan, _init_var_name, _NOT_CARRIED)


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionReport:
    """What was done to a transcript — **exactly** the body of the suite's ``context.compacted``
    event, so a consumer emits it without reshaping (spec §11 contract 7).

    Shaped as though it were a SetSpec payload, because two applications will emit it and neither
    should have to translate: every field is a scalar or a list of ids, nothing nests a transcript
    or a turn, and no field carries content — turns are named by id, never quoted (spec §14).

    It carries **no timestamp**. PromptCadence's event frame already stamps and sequences the
    event it wraps; a second time here would be a second answer to the same question, and a plan
    that carried one could not be re-derived byte-identically.

    Nothing in this class is computed. Every token figure is copied from the plan
    (:attr:`plan_hash` ties them to it), because the development plan names "token estimates
    drifting between plan and report" as this phase's likely failure mode, and the way to make a
    drift impossible is for there to be only one computation. The executor produces the view; the
    plan produced the numbers.

    Attributes:
        plan_hash: :meth:`CompactionPlan.plan_hash` — the audit link back to the plan that a
            consumer persisted, and the reason the report can stay this small.
        policy_name: From the plan.
        policy_version: From the plan.
        tokens_before: From the plan.
        tokens_after_estimate: From the plan. An estimate, never a count.
        estimator_ratio: From the plan; ``None`` when no figure on it came from the
            character-ratio default.
        budget_unmet: From the plan. ``True`` means the view is still over budget and the caller
            must decide — PromptCadence halts with ``COMPACTION_FAILED``. It is never a silent
            truncation (ADR-0023's rule, applied to transcripts).
        turns_before: How many turns went in.
        turns_after: How many are in the view, summary turns included.
        kept_turn_ids: Passed through unchanged, in transcript order.
        masked_turn_ids: Body replaced by a stub, in transcript order.
        summarized_turn_ids: Folded into a summary, in transcript order.
        dropped_turn_ids: Removed from the view, in transcript order.
        summary_turn_ids: The ids of the summary turns that replaced them, one per fulfilled
            request, in plan order — so "which turns were summarized into what" is answerable
            from the report alone.
    """

    plan_hash: str
    policy_name: str
    policy_version: str
    tokens_before: int
    tokens_after_estimate: int
    estimator_ratio: float | None
    budget_unmet: bool
    turns_before: int
    turns_after: int
    kept_turn_ids: tuple[str, ...]
    masked_turn_ids: tuple[str, ...]
    summarized_turn_ids: tuple[str, ...]
    dropped_turn_ids: tuple[str, ...]
    summary_turn_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the ``context.compacted`` event body, ready to serialize."""
        return {
            "plan_hash": self.plan_hash,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "tokens_before": self.tokens_before,
            "tokens_after_estimate": self.tokens_after_estimate,
            "estimator_ratio": self.estimator_ratio,
            "budget_unmet": self.budget_unmet,
            "turns_before": self.turns_before,
            "turns_after": self.turns_after,
            "kept_turn_ids": list(self.kept_turn_ids),
            "masked_turn_ids": list(self.masked_turn_ids),
            "summarized_turn_ids": list(self.summarized_turn_ids),
            "dropped_turn_ids": list(self.dropped_turn_ids),
            "summary_turn_ids": list(self.summary_turn_ids),
        }


@dataclass(frozen=True, slots=True)
class CompactedTranscript:
    """The view to send, and the account of how it was reached.

    Args:
        transcript: The compacted view. A **new** value — the input transcript is unchanged, and
            whether the originals are retained is the caller's data-ownership decision (spec §11
            contract 1; PromptCadence retains everything).
        report: What was done, as the ``context.compacted`` event body.
    """

    transcript: Transcript
    report: CompactionReport
