"""Packaging and purity — the claims the rest of the suite relies on, proved rather than asserted.

Gold standards §2 gives CutCtx six bullets. Two of them are about what is *absent*: no HTTP
client, no provider, no database, no filesystem. Absence is the hardest thing to test by using a
package, so it is tested by reading it: what an import actually pulls in, and what the source
actually names.
"""

from __future__ import annotations

import ast
import importlib.metadata
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

import cutctx

EXPECTED_PUBLIC_API = {
    # Vocabulary
    "Action",
    "Role",
    "Transcript",
    "TranscriptTurn",
    "CompactionBudget",
    "EMPTY_METADATA",
    "SUMMARY_TURN_ID_PREFIX",
    # Plans
    "CompactionPlan",
    "CompactionPolicy",
    "SummarizationRequest",
    "TurnAction",
    "TurnReplacement",
    # Execution
    "CompactedTranscript",
    "CompactionExecutor",
    "CompactionReport",
    # Estimation
    "CharRatioEstimator",
    "TokenEstimator",
    # Policies
    "DropOldestPolicy",
    # Errors
    "BudgetUnsatisfiable",
    "CompactionError",
    "PlanTranscriptMismatch",
    "SummaryMissing",
    # Metadata
    "__version__",
}

SOURCE_ROOT = Path(inspect.getfile(cutctx)).parent

ALLOWED_IMPORTS = {
    # The package's own modules.
    "cutctx",
    # The only runtime dependency (gold standards §1.1: budget 0 non-suite).
    "baseaicore",
    # The standard library this package is allowed. Every name here is pure computation: no I/O,
    # no clock, no randomness, no process. Adding to this list is the moment to ask whether the
    # purity claim still holds.
    "__future__",
    "collections",
    "dataclasses",
    "enum",
    "math",
    "types",
    "typing",
}


@pytest.mark.contract
def test_the_public_api_is_exactly_what_the_spec_documents() -> None:
    assert set(cutctx.__all__) == EXPECTED_PUBLIC_API


@pytest.mark.contract
def test_every_exported_name_resolves() -> None:
    assert [name for name in cutctx.__all__ if not hasattr(cutctx, name)] == []


@pytest.mark.contract
def test_all_lists_each_name_once() -> None:
    assert len(cutctx.__all__) == len(set(cutctx.__all__))


@pytest.mark.contract
def test_the_invariant_module_is_not_part_of_the_public_surface() -> None:
    """Every policy routes through it and it is still private: it is a rule, not an API."""
    assert "_invariants" not in cutctx.__all__
    assert not hasattr(cutctx, "validate_plan")
    assert not hasattr(cutctx, "build_plan")


@pytest.mark.contract
def test_every_public_class_has_a_docstring() -> None:
    """Gold standard G15: every public symbol states its contract."""
    undocumented = [
        name
        for name in cutctx.__all__
        if isinstance(getattr(cutctx, name), type) and not getattr(cutctx, name).__doc__
    ]

    assert undocumented == []


@pytest.mark.contract
def test_importing_the_package_pulls_in_nothing_but_the_stdlib_and_baseaicore() -> None:
    """Measure what an import adds to ``sys.modules``, in a clean process.

    In a subprocess because pytest, coverage and the plugins have already imported half the world
    by the time a test runs — measured in *this* process the assertion would be vacuous.
    """
    program = (
        "import json, sys;"
        "before = set(sys.modules);"
        "import cutctx;"
        "added = {name.split('.')[0] for name in set(sys.modules) - before};"
        "print(json.dumps(sorted(added)))"
    )

    result = subprocess.run(  # noqa: S603 — our own interpreter, no shell, literal argument list
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    added = json.loads(result.stdout)

    third_party = [
        name
        for name in added
        if name not in {"cutctx", "baseaicore"}
        and not name.startswith("_")
        and name not in sys.stdlib_module_names
    ]
    assert third_party == [], f"cutctx imported unexpected modules: {third_party}"


@pytest.mark.contract
def test_no_module_imports_a_client_a_store_a_clock_or_a_source_of_randomness() -> None:
    """Read the source rather than the import graph: an import inside a function still counts.

    ``import-linter`` covers the module graph; this covers the rest of the file. The four things
    it is looking for are the four ways this package could stop being a pure function —
    ``httpx``/``socket`` (a second path to a model, ADR-0052 decision 2), ``sqlite3``/``pathlib``
    (persistence CutCtx does not own, spec §10), ``datetime``/``time`` (a plan that could not be
    re-derived, contract 4), ``random``/``secrets`` (the same, from the other direction).
    """
    offenders: list[str] = []
    for module in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            offenders += [
                f"{module.name}: {name}"
                for name in names
                if name.split(".")[0] not in ALLOWED_IMPORTS
            ]

    assert offenders == [], f"imports outside the allowlist: {offenders}"


@pytest.mark.contract
def test_the_package_logs_nothing() -> None:
    """Spec §17: the ``CompactionReport`` is what a consumer logs; the package emits nothing."""
    sources = [p.read_text(encoding="utf-8") for p in SOURCE_ROOT.rglob("*.py")]

    assert not any("logging" in text or "print(" in text for text in sources)


@pytest.mark.contract
def test_no_prompt_text_appears_anywhere_in_the_package() -> None:
    """ADR-0012: a prompt is named by ``prompt_id``. A literal here would be a prompt in hiding."""
    for module in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign | ast.AnnAssign) and isinstance(
                node.value, ast.Constant
            ):
                value = node.value.value
                assert not (isinstance(value, str) and len(value) > 120), (
                    f"{module.name} assigns a long string literal — a prompt would look like this"
                )


@pytest.mark.contract
def test_the_distribution_declares_only_baseaicore_at_runtime() -> None:
    """Gold standard G16 §1.1: CutCtx's non-suite runtime dependency budget is zero."""
    requirements = importlib.metadata.requires("cutctx") or []

    runtime = [r for r in requirements if "extra ==" not in r]
    assert [r.split(">")[0].split("<")[0].split("=")[0].strip() for r in runtime] == ["baseaicore"]


@pytest.mark.contract
def test_the_py_typed_marker_ships_so_consumers_get_the_types() -> None:
    assert (SOURCE_ROOT / "py.typed").is_file()


@pytest.mark.contract
def test_the_version_is_what_the_metadata_says() -> None:
    assert cutctx.__version__ == importlib.metadata.version("cutctx")


def test_the_standalone_script_from_the_spec_works() -> None:
    """Spec §20 acceptance criterion 2, as a test rather than as a promise.

    ``cutctx`` + ``baseaicore``, no suite application, a hand-supplied summary, plan and apply.
    """
    from cutctx import (  # noqa: PLC0415 — importing it here is what the script does
        Action,
        CompactionBudget,
        CompactionExecutor,
        CompactionPlan,
        Role,
        SummarizationRequest,
        Transcript,
        TranscriptTurn,
        TurnAction,
    )

    transcript = Transcript(
        (
            TranscriptTurn("s", Role.SYSTEM, "Be brief.", 5),
            TranscriptTurn("u1", Role.USER, "A long question.", 200),
            TranscriptTurn("a1", Role.ASSISTANT, "A long answer.", 200),
            TranscriptTurn("u2", Role.USER, "And now?", 10),
        )
    )
    budget = CompactionBudget(max_tokens=100, protected_recent_turns=1)
    plan = CompactionPlan(
        actions=(
            TurnAction("s", Action.KEEP),
            TurnAction("u1", Action.SUMMARIZE, summary_group="g1"),
            TurnAction("a1", Action.SUMMARIZE, summary_group="g1"),
            TurnAction("u2", Action.KEEP),
        ),
        summarization_requests=(SummarizationRequest("g1", ("u1", "a1"), 20, "general.summarize"),),
        tokens_before=415,
        tokens_after_estimate=35,
        policy_name="hand_written",
        policy_version="1.0.0",
        transcript=transcript,
        budget=budget,
    )

    compacted = CompactionExecutor().apply(
        transcript, plan, {"g1": "The user asked a long question and got a long answer."}
    )

    assert compacted.transcript.turn_ids() == ("s", "summary:g1", "u2")
    assert compacted.report.tokens_after_estimate == 35
    assert compacted.report.budget_unmet is False
    assert transcript.turn_ids() == ("s", "u1", "a1", "u2")


@pytest.mark.contract
def test_the_package_docstrings_examples_actually_run() -> None:
    """The README and the module docstring both quote figures; a stale one is a wrong promise.

    ``--doctest-modules`` is not in the suite's shared ``addopts``, so the doctest is run from
    here rather than left to a flag nobody passes.
    """
    import doctest  # noqa: PLC0415 — only this test needs it

    results = doctest.testmod(cutctx, verbose=False)

    assert results.attempted > 0, "the package docstring has no runnable example"
    assert results.failed == 0
