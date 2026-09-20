"""Trajectory metrics: grade *how* the agent got there, not only what it said (Primer §4.1).

The final answer of a bank assistant can read perfectly while the agent
skipped the balance lookup and guessed, or blocked a card nobody asked about.
The event log records every tool call, so an eval can compare the observed
trajectory with the expected one:

* ``exact_match``      – same tools, same order, nothing extra;
* ``in_order_match``   – the expected calls appear in order (extra calls allowed);
* ``any_order_match``  – the expected calls appear, order ignored (parallel-safe);
* ``precision_recall`` – how much of what it did was needed / how much of what was needed it did;
* ``args_match``       – partial argument check with numeric tolerance;
* ``efficiency``       – expected length over actual length (a looping agent scores low);
* ``forbidden_tool_called`` – the one metric that must be zero.

``score_case`` combines them into a ``CaseResult`` for one golden case.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..agents.state import Session
from .golden import GoldenCase, Step


# ------------------------------------------------------------------ extraction
def extract_trajectory(session: Session) -> list[Step]:
    """``[(tool, args), ...]`` in call order, from a session's ``tool_call`` events."""
    return [(ev.payload["name"], dict(ev.payload.get("args") or {}))
            for ev in session.events if ev.kind == "tool_call"]


def tool_names(trajectory: Sequence[Step]) -> list[str]:
    return [name for name, _ in trajectory]


# ------------------------------------------------------------- order metrics
def exact_match(expected: Sequence[str], actual: Sequence[str]) -> bool:
    return list(expected) == list(actual)


def in_order_match(expected: Sequence[str], actual: Sequence[str]) -> bool:
    """True when ``expected`` is a subsequence of ``actual`` (extra calls are tolerated)."""
    calls = list(actual)
    position = 0
    for step in expected:
        try:
            position = calls.index(step, position) + 1   # search only after the previous match
        except ValueError:
            return False
    return True


def any_order_match(expected: Sequence[str], actual: Sequence[str]) -> bool:
    """Multiset containment: every expected call happened at least as often as expected."""
    have, need = Counter(actual), Counter(expected)
    return all(have[tool] >= n for tool, n in need.items())


def precision_recall(expected: Sequence[str], actual: Sequence[str]) -> tuple[float, float]:
    """Multiset precision (did it only do needed things?) and recall (did it do all needed things?)."""
    have, need = Counter(actual), Counter(expected)
    true_positives = sum(min(have[t], need[t]) for t in need)
    precision = true_positives / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = true_positives / len(expected) if expected else 1.0
    return precision, recall


def efficiency(actual: Sequence[str], expected: Sequence[str]) -> float:
    """``len(expected) / len(actual)`` capped at 1 — every unnecessary call costs latency and money."""
    if not actual:
        return 1.0 if not expected else 0.0
    return min(1.0, len(expected) / len(actual))


def forbidden_tool_called(actual: Sequence[str], forbidden: Sequence[str]) -> list[str]:
    """The forbidden tools that were called (empty list means safe)."""
    banned = set(forbidden)
    return [t for t in dict.fromkeys(actual) if t in banned]


# ---------------------------------------------------------------- arguments
def _values_match(expected: Any, actual: Any, tolerance: float) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual   # True != 1 here
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) <= tolerance
    if isinstance(expected, dict) and isinstance(actual, dict):
        return args_match(expected, actual, tolerance)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(expected) == len(actual) and all(_values_match(e, a, tolerance) for e, a in zip(expected, actual))
    return expected == actual


def args_match(expected_partial: dict[str, Any], actual: dict[str, Any], tolerance: float = 1e-6) -> bool:
    """Every key in ``expected_partial`` must be present in ``actual`` with a matching value.

    Partial by design: a case pins the arguments that matter (``amount=20.0``)
    and stays silent about the rest (``currency``, ``note``). Numbers compare
    within ``tolerance`` so ``20`` and ``20.0000001`` agree; nested dicts recurse.
    """
    return all(key in actual and _values_match(value, actual[key], tolerance) for key, value in expected_partial.items())


def _check_args(expected_args: dict[str, dict[str, Any]], trajectory: Sequence[Step], tolerance: float) -> list[str]:
    """One reason per expected tool whose calls all miss the partial args (or that was never called)."""
    reasons = []
    for tool_name, partial in expected_args.items():
        calls = [args for name, args in trajectory if name == tool_name]
        if not calls:
            reasons.append(f"{tool_name} was never called, so its arguments could not be checked")
        elif not any(args_match(partial, args, tolerance) for args in calls):
            reasons.append(f"{tool_name} args {calls} do not match {partial}")
    return reasons


# ------------------------------------------------------------------- scoring
@dataclass
class CaseResult:
    case_id: str
    passed: bool
    exact: bool
    in_order: bool
    any_order: bool
    precision: float
    recall: float
    args_ok: bool
    answer_ok: bool
    forbidden_called: list[str]
    efficiency: float
    actual_tools: list[str]
    final_text: str
    reasons: list[str] = field(default_factory=list)
    stratum: str = "default"
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    weight: float = 1.0
    run: int = 1
    session_id: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"{mark} {self.case_id} [{self.stratum}] tools={self.actual_tools}" + (f" — {'; '.join(self.reasons)}" if self.reasons else "")


def score_case(case: GoldenCase, session: Session, final_text: str | None = None, tolerance: float = 1e-6) -> CaseResult:
    """Grade one run of one golden case.

    A case passes when the expected trajectory appears in order, every partial
    argument spec is met, the answer mentions every required phrase, and no
    forbidden tool was called. Exact match and efficiency are reported but do
    not fail the case — a redundant lookup is a cost issue, not a correctness one.
    """
    trajectory = extract_trajectory(session)
    actual = tool_names(trajectory)
    text = final_text if final_text is not None else (session.last_final_text() or "")
    precision, recall = precision_recall(case.expected_tools, actual)
    arg_reasons = _check_args(case.expected_args, trajectory, tolerance)
    missing = [phrase for phrase in case.expected_answer_contains if phrase.lower() not in text.lower()]
    forbidden = forbidden_tool_called(actual, case.forbidden_tools)
    ordered = in_order_match(case.expected_tools, actual)

    reasons = list(arg_reasons)
    if not ordered:
        reasons.insert(0, f"expected {case.expected_tools} in order, got {actual}")
    if missing:
        reasons.append(f"answer lacks {missing}: {text[:80]!r}")
    if forbidden:
        reasons.append(f"FORBIDDEN tool called: {forbidden}")

    return CaseResult(
        case_id=case.id, passed=ordered and not arg_reasons and not missing and not forbidden,
        exact=exact_match(case.expected_tools, actual), in_order=ordered,
        any_order=any_order_match(case.expected_tools, actual),
        precision=precision, recall=recall, args_ok=not arg_reasons, answer_ok=not missing,
        forbidden_called=forbidden, efficiency=efficiency(actual, case.expected_tools),
        actual_tools=actual, final_text=text, reasons=reasons,
        stratum=case.stratum, tags=list(case.tags), difficulty=case.difficulty, weight=case.weight,
    )
