"""How many sandboxes, and what each execution costs: Little's law, Erlang C, and a warm-pool simulator.

The one idea: a sandbox per execution has a cold start, so the number you must keep ready follows the same
queueing arithmetic as any service. Little's law sizes the *busy* set (arrival rate × execution time); a
replace-after-use warm pool also holds a *warming* set (arrival rate × cold-start time). Erlang C turns a
target "fraction that waits" into a server count. The discrete-event ``simulate()`` checks the closed-form
numbers against an actual run and shows what cold starts do to tail latency. The arrival rate ties back to
the scaling primer: at peak, 27.1 tool calls/s (§3.2); if a fifth are ``run_code``, λ ≈ 5.4/s. Cost per
action closes the loop with §1: actions per turn × cost per action. Every number here is computed by the
functions the primer names; latencies are inputs, and any $ figure is (verify) against COMPUTE.md.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass


def busy_sandboxes(arrival_rate: float, exec_s: float) -> float:
    """Little's law: mean number of sandboxes actively running code = λ · t_exec."""
    return arrival_rate * exec_s


def warming_sandboxes(arrival_rate: float, cold_start_s: float) -> float:
    """A replace-after-use pool also holds sandboxes warming up = λ · t_cold."""
    return arrival_rate * cold_start_s


def pool_size(arrival_rate: float, exec_s: float, cold_start_s: float) -> int:
    """Sandboxes to hold so a request rarely waits on a cold start: ceil(busy + warming)."""
    return math.ceil(busy_sandboxes(arrival_rate, exec_s) + warming_sandboxes(arrival_rate, cold_start_s))


def erlang_c(offered_load: float, servers: int) -> float:
    """P(wait) for an M/M/c queue. offered_load a = λ·t_exec (Erlangs); servers c > a or it never clears."""
    a, c = offered_load, servers
    if c <= a:
        return 1.0
    # sum_{k<c} a^k/k!  and the last term a^c/c! · c/(c-a), computed iteratively to avoid overflow.
    term = 1.0
    s = 1.0
    for k in range(1, c):
        term *= a / k
        s += term
    last = term * (a / c) * (c / (c - a))
    return last / (s + last)


def expected_wait_s(arrival_rate: float, exec_s: float, servers: int) -> float:
    """E[Wq] for M/M/c: C(a,c) / (c·μ − λ), μ = 1/t_exec. Zero when the queue never forms."""
    a = arrival_rate * exec_s
    mu = 1.0 / exec_s
    denom = servers * mu - arrival_rate
    if denom <= 0:
        return math.inf
    return erlang_c(a, servers) / denom


def servers_for_wait_target(arrival_rate: float, exec_s: float, max_p_wait: float) -> int:
    """Smallest server count whose P(wait) is at or below the target."""
    a = arrival_rate * exec_s
    c = math.floor(a) + 1
    while erlang_c(a, c) > max_p_wait:
        c += 1
    return c


def cost_per_execution(exec_s: float, cold_start_s: float, node_cost_per_s: float,
                       fixed_per_exec: float = 0.0, warm_share: float = 1.0) -> float:
    """(sandbox-seconds held × $/s) + fixed. warm_share=1 for pay-per-use; the fraction of cold start
    charged (0 if a warm pool absorbs it, 1 if each execution pays its own cold start)."""
    held_s = exec_s + warm_share * cold_start_s
    return held_s * node_cost_per_s + fixed_per_exec


def actions_cost(actions_per_turn: float, cost_per_action: float) -> float:
    """Scaling primer §1: the per-turn code-execution bill is actions/turn × cost/action."""
    return actions_per_turn * cost_per_action


@dataclass
class SimResult:
    completed: int
    mean_wait_s: float
    p95_wait_s: float
    frac_waited: float
    cold_starts: int
    max_busy: int

    def as_row(self) -> dict:
        return {"completed": self.completed, "mean_wait_s": round(self.mean_wait_s, 4),
                "p95_wait_s": round(self.p95_wait_s, 4), "frac_waited": round(self.frac_waited, 3),
                "cold_starts": self.cold_starts, "max_busy": self.max_busy}


def simulate(arrival_rate: float, exec_s: float, cold_start_s: float, servers: int, *,
             n: int = 20000, warm_pool: bool = True, seed: int = 0) -> SimResult:
    """A discrete-event M/M/c-style run of a fixed-size sandbox pool. SIMULATED — labels its output so.

    ``servers`` sandbox slots. With ``warm_pool`` a slot is already warm when a request grabs it (no cold
    start on the request's path); without it, an idle slot must cold-start first, adding ``cold_start_s`` to
    the request's wait. Exponential inter-arrivals and service; a request that finds no free slot queues
    FCFS. Returns waits (queueing + any cold start) so you can compare with ``expected_wait_s``.
    """
    import random
    rng = random.Random(seed)
    mu = 1.0 / exec_s
    free_at = [0.0] * servers            # when each slot next becomes free
    warm = [warm_pool] * servers         # is the slot already warm?
    t = 0.0
    waits: list[float] = []
    cold_starts = 0
    max_busy = 0
    for _ in range(n):
        t += rng.expovariate(arrival_rate)
        slot = min(range(servers), key=lambda i: free_at[i])
        start = max(t, free_at[slot])
        cold = 0.0
        if not warm[slot]:
            cold = cold_start_s
            cold_starts += 1
            warm[slot] = True            # warmed for its next use
        wait = (start - t) + cold
        service = rng.expovariate(mu)
        free_at[slot] = start + cold + service
        if not warm_pool:
            warm[slot] = False           # torn down after use, cold again next time
        waits.append(wait)
        busy = sum(1 for f in free_at if f > t)
        max_busy = max(max_busy, busy)
    waits.sort()
    p95 = waits[int(0.95 * len(waits)) - 1]
    return SimResult(n, sum(waits) / len(waits), p95,
                     sum(1 for w in waits if w > 1e-6) / len(waits), cold_starts, max_busy)
