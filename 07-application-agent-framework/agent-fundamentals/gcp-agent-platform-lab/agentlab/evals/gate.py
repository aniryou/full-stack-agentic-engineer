"""Run a golden set and decide whether a change ships (Primer §4.1).

``run_eval`` drives the agent through every case ``n_runs`` times with a fresh
session per run, so flakiness is measured rather than averaged away. ``Gate``
turns the results into a release decision:

* a threshold on a *rate* metric is judged on the observed value and reported
  with a Wilson confidence interval — a 9/10 pass rate is "somewhere between
  60% and 98%", and the gate says so instead of pretending it is 90%;
* an ``absolute`` threshold demands 100% across every run: the pattern for
  irreversible actions (blocking a card, moving money) where one miss in a
  thousand is an incident, not a statistic;
* ``regression_vs`` compares two eval runs and flags a drop only when it is
  larger than the run-to-run noise, so a 2-point wobble on a 15-case set does
  not block a release while a broken intent does.
"""
from __future__ import annotations

import inspect
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..agents import InMemorySessionStore, Runner, Session
from .golden import GoldenSet
from .trajectory import CaseResult, score_case

# Metrics that are a fraction of case-runs satisfying a predicate (binomial → Wilson CI applies).
RATE_METRICS: dict[str, Callable[[CaseResult], bool]] = {
    "pass_rate": lambda r: r.passed,
    "exact_rate": lambda r: r.exact,
    "in_order_rate": lambda r: r.in_order,
    "any_order_rate": lambda r: r.any_order,
    "args_rate": lambda r: r.args_ok,
    "answer_rate": lambda r: r.answer_ok,
    "no_forbidden_rate": lambda r: not r.forbidden_called,
}
# Metrics that are means of a per-case score (no binomial CI).
MEAN_METRICS: dict[str, Callable[[CaseResult], float]] = {
    "efficiency": lambda r: r.efficiency,
    "precision": lambda r: r.precision,
    "recall": lambda r: r.recall,
}


# ------------------------------------------------------------------ statistics
def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (well-behaved at 0/n and n/n, unlike ±1.96·SE)."""
    if n == 0:
        return (0.0, 1.0)
    p = passes / n
    z2 = z * z
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def half_width_rule_of_thumb(n: int) -> float:
    """≈ 1/√n: the confidence half-width a pass rate near 50% carries — 100 cases ≈ ±10 points."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 1 / math.sqrt(n)


# -------------------------------------------------------------------- eval run
@dataclass
class EvalRun:
    results: list[CaseResult]
    n_runs: int
    golden_name: str = "golden"
    seed: int | None = None
    sessions: dict[str, Session] = field(default_factory=dict)  # session_id -> Session, for triage

    # -- selection ----------------------------------------------------------
    def select(self, scope: str = "all", run: int | None = None) -> list[CaseResult]:
        """``scope`` is ``"all"``, ``"stratum:<name>"`` or ``"tag:<name>"``."""
        rows = self.results if run is None else [r for r in self.results if r.run == run]
        if scope == "all":
            return rows
        kind, _, value = scope.partition(":")
        if kind == "stratum":
            return [r for r in rows if r.stratum == value]
        if kind == "tag":
            return [r for r in rows if value in r.tags]
        raise ValueError(f"unknown scope {scope!r}; use 'all', 'stratum:<name>' or 'tag:<name>'")

    def rate(self, metric: str, scope: str = "all", run: int | None = None) -> tuple[float, int, int]:
        """``(value, passes, n)`` for a rate metric."""
        rows = self.select(scope, run)
        passes = sum(1 for r in rows if RATE_METRICS[metric](r))
        return (passes / len(rows) if rows else 0.0), passes, len(rows)

    def metric(self, metric: str, scope: str = "all", run: int | None = None) -> float:
        if metric in RATE_METRICS:
            return self.rate(metric, scope, run)[0]
        rows = self.select(scope, run)
        return sum(MEAN_METRICS[metric](r) for r in rows) / len(rows) if rows else 0.0

    def pass_rate(self, scope: str = "all") -> float:
        return self.rate("pass_rate", scope)[0]

    def weighted_pass_rate(self, scope: str = "all") -> float:
        rows = self.select(scope)
        total = sum(r.weight for r in rows)
        return sum(r.weight for r in rows if r.passed) / total if total else 0.0

    # -- views ---------------------------------------------------------------------
    def strata(self) -> list[str]:
        return list(dict.fromkeys(r.stratum for r in self.results))

    def by_stratum(self) -> dict[str, tuple[float, int, int]]:
        return {s: self.rate("pass_rate", f"stratum:{s}") for s in self.strata()}

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]

    def flaky_cases(self) -> list[str]:
        """Cases that passed in some runs and failed in others — the ones a single run would misreport."""
        outcomes: dict[str, set[bool]] = {}
        for r in self.results:
            outcomes.setdefault(r.case_id, set()).add(r.passed)
        return [cid for cid, seen in outcomes.items() if len(seen) > 1]

    def summary(self) -> dict[str, Any]:
        value, passes, n = self.rate("pass_rate")
        return {
            "golden": self.golden_name, "runs": self.n_runs, "case_runs": n, "passes": passes,
            "pass_rate": value, "wilson_95": wilson_interval(passes, n),
            "weighted_pass_rate": self.weighted_pass_rate(),
            "by_stratum": {s: {"pass_rate": v, "passes": p, "n": k, "wilson_95": wilson_interval(p, k)}
                           for s, (v, p, k) in self.by_stratum().items()},
            "flaky_cases": self.flaky_cases(),
            "efficiency": self.metric("efficiency"),
        }

    def render(self) -> str:
        lines = [f"{self.golden_name}: {self.n_runs} run(s)"]
        for stratum, (value, passes, n) in self.by_stratum().items():
            lo, hi = wilson_interval(passes, n)
            lines.append(f"  {stratum:12s} {passes:3d}/{n:<3d} = {value:6.1%}   95% CI [{lo:.2f}, {hi:.2f}]")
        value, passes, n = self.rate("pass_rate")
        lo, hi = wilson_interval(passes, n)
        lines.append(f"  {'all':12s} {passes:3d}/{n:<3d} = {value:6.1%}   95% CI [{lo:.2f}, {hi:.2f}]")
        return "\n".join(lines)


def _accepts_seed(factory: Callable[..., Any]) -> bool:
    try:
        return "seed" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False


async def run_eval(agent_factory: Callable[..., Any], golden_set: GoldenSet, n_runs: int = 1, seed: int | None = None,
                   runner_kwargs: dict[str, Any] | None = None, *, user: Any = None, auto_approve: bool = True) -> EvalRun:
    """Run every case ``n_runs`` times through a fresh agent, runner session and score it.

    ``agent_factory()`` returns a ``BaseAgent``; if it accepts a ``seed`` keyword
    it receives a per-run seed derived from ``seed`` so stochastic policies are
    reproducible run by run. Confirmation pauses are approved automatically
    (the harness plays the human) unless ``auto_approve`` is False.
    """
    rng = random.Random(seed)
    run_seeds = [rng.randrange(2**31) for _ in range(n_runs)]
    kwargs = dict(runner_kwargs or {})
    kwargs.setdefault("store", InMemorySessionStore())
    results: list[CaseResult] = []
    sessions: dict[str, Session] = {}
    for run_no, run_seed in enumerate(run_seeds, 1):
        for case in golden_set:
            agent = agent_factory(seed=run_seed) if _accepts_seed(agent_factory) else agent_factory()
            runner = Runner(agent, **kwargs)
            session_id = f"{case.id}#run{run_no}"
            result = await runner.run(session_id, case.input, user=user)
            while result.paused and auto_approve:
                result = await runner.approve(session_id, True, user=user)
            scored = score_case(case, result.session)
            scored.run, scored.session_id = run_no, session_id
            if result.error:
                scored.reasons.append(f"runtime error: {result.error}")
            results.append(scored)
            sessions[session_id] = result.session
    return EvalRun(results=results, n_runs=n_runs, golden_name=golden_set.name, seed=seed, sessions=sessions)


# ------------------------------------------------------------------------ gate
@dataclass(frozen=True)
class Threshold:
    """``metric`` must reach ``min_value`` within ``scope``; ``absolute`` means 100% across all runs."""

    metric: str
    min_value: float
    scope: str = "all"                  # "all" | "stratum:<name>" | "tag:<name>"
    absolute: bool = False

    def __post_init__(self) -> None:
        if self.metric not in RATE_METRICS and self.metric not in MEAN_METRICS:
            raise ValueError(f"unknown metric {self.metric!r}; rates: {sorted(RATE_METRICS)}, means: {sorted(MEAN_METRICS)}")
        if self.absolute and self.metric not in RATE_METRICS:
            raise ValueError("absolute thresholds only make sense for rate metrics")

    def label(self) -> str:
        bar = "100% of every run" if self.absolute else f">= {self.min_value:.0%}"
        return f"{self.metric} @ {self.scope} {bar}"


@dataclass
class ThresholdResult:
    threshold: Threshold
    observed: float
    n: int
    passes: int | None
    ci: tuple[float, float] | None
    ok: bool
    underpowered: bool                  # passes on the point estimate but the CI straddles the bar
    reason: str

    def __str__(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        ci = f" CI [{self.ci[0]:.2f}, {self.ci[1]:.2f}]" if self.ci else ""
        warn = "  (underpowered: add cases)" if self.underpowered else ""
        return f"{mark} {self.threshold.label():48s} observed {self.observed:.3f}{ci} — {self.reason}{warn}"


@dataclass
class ScopeDelta:
    metric: str
    scope: str
    baseline: float
    candidate: float
    noise_band: float

    @property
    def drop(self) -> float:
        return self.baseline - self.candidate

    @property
    def regressed(self) -> bool:
        return self.drop > self.noise_band

    def __str__(self) -> str:
        verdict = "REGRESSION" if self.regressed else ("improved" if self.drop < 0 else "within noise")
        return f"{self.metric} @ {self.scope}: {self.baseline:.3f} → {self.candidate:.3f} (drop {self.drop:+.3f}, noise ±{self.noise_band:.3f}) {verdict}"


@dataclass
class RegressionReport:
    deltas: list[ScopeDelta]

    @property
    def regressed(self) -> bool:
        return any(d.regressed for d in self.deltas)

    def regressions(self) -> list[ScopeDelta]:
        return [d for d in self.deltas if d.regressed]

    def render(self) -> str:
        return "\n".join(str(d) for d in self.deltas)


def run_to_run_spread(eval_run: EvalRun, metric: str, scope: str) -> float:
    """max − min of a metric across the runs of one eval — the noise a single number hides."""
    values = [eval_run.metric(metric, scope, run) for run in range(1, eval_run.n_runs + 1)]
    return max(values) - min(values) if values else 0.0


@dataclass
class GateReport:
    eval_run: EvalRun
    results: list[ThresholdResult]

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.results)

    def failures(self) -> list[ThresholdResult]:
        return [r for r in self.results if not r.ok]

    def render(self) -> str:
        return "\n".join(str(r) for r in self.results) + f"\n=> gate {'PASSED' if self.passed else 'FAILED'}"

    def regression_vs(self, baseline: EvalRun, noise_band: float | None = None) -> RegressionReport:
        """Compare this report's run against a baseline on every gated (metric, scope).

        ``noise_band`` defaults to the larger run-to-run spread seen in either
        eval; with a single run on both sides it is zero and any drop counts.
        """
        deltas = []
        for scope_key in dict.fromkeys((r.threshold.metric, r.threshold.scope) for r in self.results):
            metric, scope = scope_key
            band = noise_band if noise_band is not None else max(
                run_to_run_spread(baseline, metric, scope), run_to_run_spread(self.eval_run, metric, scope))
            deltas.append(ScopeDelta(metric, scope, baseline.metric(metric, scope), self.eval_run.metric(metric, scope), band))
        return RegressionReport(deltas)


class Gate:
    """A list of thresholds an eval run must clear before a change ships."""

    def __init__(self, thresholds: Iterable[Threshold]):
        self.thresholds = list(thresholds)
        if not self.thresholds:
            raise ValueError("a gate needs at least one threshold")

    def evaluate(self, eval_run: EvalRun) -> GateReport:
        return GateReport(eval_run, [self._check(t, eval_run) for t in self.thresholds])

    @staticmethod
    def _check(t: Threshold, eval_run: EvalRun) -> ThresholdResult:
        if t.metric in RATE_METRICS:
            observed, passes, n = eval_run.rate(t.metric, t.scope)
            ci: tuple[float, float] | None = wilson_interval(passes, n)
        else:
            observed, passes, n, ci = eval_run.metric(t.metric, t.scope), None, len(eval_run.select(t.scope)), None
        if n == 0:
            return ThresholdResult(t, observed, n, passes, ci, ok=False, underpowered=False, reason="no cases in scope")
        if t.absolute:
            ok = passes == n
            return ThresholdResult(t, observed, n, passes, ci, ok=ok, underpowered=False,
                                   reason=f"{passes}/{n} case-runs across {eval_run.n_runs} run(s); absolute thresholds allow no misses")
        ok = observed >= t.min_value
        underpowered = ok and ci is not None and ci[0] < t.min_value
        detail = f"{passes}/{n} case-runs" if passes is not None else f"mean over {n} case-runs"
        return ThresholdResult(t, observed, n, passes, ci, ok=ok, underpowered=underpowered,
                               reason=f"{detail} vs bar {t.min_value:.2f}")
