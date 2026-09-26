"""bench.py — what each rung of the ladder costs in start-up time, and how many sandboxes that means.

One idea: isolation is paid for in *latency before the first instruction runs*, and a warm pool
buys that latency back with idle capacity (PRIMER §6 "Latency, throughput and cost per action").
This module measures the start-up and run time of every level this machine can run — fork/exec,
a Python interpreter, the process sandbox, Docker (runc, runsc) and a Kubernetes Job or warm pod
when reachable — and sizes a pool with Little's law and Erlang C.

Numbers from ``measure()`` are **measured on this machine**. Levels this machine cannot run come
from ``data/samples/bench_latency.json``: **sample output in the documented format
(illustrative)**, with each row's source. Nothing here invents a measurement.
"""
from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .process import Budgets, ProcessSandbox

SAMPLES = Path(__file__).parent / "data" / "samples" / "bench_latency.json"


def percentile(xs: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = math.floor(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class Measurement:
    level: str
    n: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    label: str = "measured on this machine"
    source: str = ""

    def row(self) -> str:
        return f"{self.level:<28} p50 {self.p50_ms:9.1f} ms   p95 {self.p95_ms:9.1f} ms   n={self.n:<3} [{self.label}]"


def _summ(level: str, xs_s: list[float], label: str = "measured on this machine", source: str = "") -> Measurement:
    ms = [x * 1000 for x in xs_s]
    return Measurement(level, len(ms), round(percentile(ms, 0.5), 2), round(percentile(ms, 0.95), 2),
                       round(statistics.fmean(ms), 2), label, source)


def time_cmd(argv: list[str], n: int = 20) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out.append(time.perf_counter() - t0)
    return out


def time_executor(executor, n: int = 10, code: str = "print(1)") -> list[float]:
    out = []
    for _ in range(n):
        r = executor.run(code, Budgets(cpu_s=2, wall_s=10))
        if r.exit_reason != "ok":
            raise RuntimeError(f"{getattr(executor, 'name', executor)}: {r.exit_reason} {r.stderr[-200:]}")
        out.append(r.total_s)
    return out


def measure(levels: list[str] | None = None, n: int = 20) -> list[Measurement]:
    """Round-trip time to run ``print(1)`` (or ``true``) at each level this machine can run."""
    from . import env
    caps = env.capabilities()
    levels = levels or ["fork_exec", "python", "python_isolated", "process", "process+netns"]
    out = []
    for lv in levels:
        if lv == "fork_exec":
            out.append(_summ(lv, time_cmd(["/bin/true"], n)))
        elif lv == "python":
            out.append(_summ(lv, time_cmd([sys.executable, "-c", "pass"], n)))
        elif lv == "python_isolated":
            out.append(_summ(lv, time_cmd([sys.executable, "-I", "-S", "-c", "pass"], n)))
        elif lv == "process":
            out.append(_summ(lv, time_executor(ProcessSandbox(netns=False), max(5, n // 2))))
        elif lv == "process+netns" and caps.netns:
            out.append(_summ(lv, time_executor(ProcessSandbox(netns=True), max(5, n // 2))))
        elif lv.startswith("docker:") and caps.docker and (lv != "docker:runsc" or caps.runsc):
            from .docker import DockerSandbox
            out.append(_summ(lv, time_executor(DockerSandbox.hardened(lv.split(":")[1] if lv != "docker:runc" else None),
                                               max(3, n // 4))))
    return out


def samples() -> tuple[str, list[Measurement]]:
    d = json.loads(SAMPLES.read_text())
    return d["_label"], [Measurement(**r) for r in d["rows"]]


def ladder(measured: list[Measurement], *, include_samples: bool = True) -> list[Measurement]:
    """Measured rows first; sample rows only for levels not measured here."""
    have = {m.level for m in measured}
    rows = list(measured)
    if include_samples:
        rows += [s for s in samples()[1] if s.level not in have]
    return rows


# ---- sizing a pool -------------------------------------------------------------------------------------------
def littles_law_pool(arrival_per_s: float, exec_s: float, cold_start_s: float) -> dict:
    """Replace-after-use pool that never makes a request wait for a cold start: sandboxes busy
    running code (lambda x t_exec) plus sandboxes being prepared to replace them (lambda x t_cold)."""
    busy = arrival_per_s * exec_s
    warming = arrival_per_s * cold_start_s
    return {"busy": busy, "warming": warming, "total": busy + warming,
            "total_ceil": math.ceil(busy + warming - 1e-9)}


def erlang_c(offered_load: float, servers: int) -> float:
    """P(an arrival waits) in M/M/c with offered load a = lambda/mu (Erlang C)."""
    a, c = offered_load, servers
    if c <= a:
        return 1.0
    top = a ** c / math.factorial(c) * c / (c - a)
    return top / (sum(a ** k / math.factorial(k) for k in range(c)) + top)


def mean_wait_s(arrival_per_s: float, exec_s: float, servers: int) -> float:
    """E[Wq] = C(a, c) / (c*mu - lambda), mu = 1/t_exec."""
    mu = 1.0 / exec_s
    if servers * mu <= arrival_per_s:
        return math.inf
    return erlang_c(arrival_per_s * exec_s, servers) / (servers * mu - arrival_per_s)


def servers_for(arrival_per_s: float, exec_s: float, *, max_p_wait: float | None = None,
                max_mean_wait_s: float | None = None) -> int:
    c = math.floor(arrival_per_s * exec_s) + 1
    while True:
        ok = (max_p_wait is None or erlang_c(arrival_per_s * exec_s, c) <= max_p_wait) and \
             (max_mean_wait_s is None or mean_wait_s(arrival_per_s, exec_s, c) <= max_mean_wait_s)
        if ok:
            return c
        c += 1


def cost_per_execution(held_s: float, node_usd_per_hour: float, sandboxes_per_node: int,
                       fixed_usd: float = 0.0) -> float:
    """Sandbox-seconds held x the per-second price of one sandbox's share of a node, plus any
    per-execution fixed cost. Prices are inputs (see COMPUTE.md; mark any figure you use (verify))."""
    return held_s * node_usd_per_hour / 3600.0 / sandboxes_per_node + fixed_usd


def to_json(rows: list[Measurement]) -> str:
    return json.dumps([asdict(r) for r in rows], indent=1)
