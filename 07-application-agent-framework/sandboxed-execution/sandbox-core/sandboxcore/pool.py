"""How many sandboxes, and what each execution costs: Little's law, Erlang C, and a pool simulator.

The one idea: a sandbox per execution has a cold start, so the number you must keep ready follows the same
queueing arithmetic as any service. In a **replace-after-use** warm pool (the safe kind: a sandbox runs one
execution and is destroyed, and a fresh one warms up in the background to replace it) each execution holds a
*slot* for its run **and** for the replacement's cold start. So Little's law gives the **mean** occupancy,
λ·(t_exec + t_cold) — busy plus warming — which is the offered load *a* in Erlangs. It is the floor, not the
pool size: with exactly *a* slots the queue never clears. Erlang C on that *a* turns a target "fraction of
requests that wait for a warm sandbox" into a slot count. A **reuse** pool (one warm sandbox serves many
executions) holds a slot for t_exec only — cheaper and faster, but state carries between executions. The
discrete-event ``simulate()`` runs all three modes (plus **cold-on-demand**, no pool: every request pays the
cold start on its own path) so the closed forms can be checked. The arrival rate ties back to the scaling
primer: at peak, 27.1 tool calls/s (§3.2); if a fifth are ``run_code``, λ ≈ 5.4/s. Cost per action closes the
loop with §1: actions per turn × cost per action. Latencies are inputs; any $ figure is (verify) against
COMPUTE.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MODES = ("reuse", "replace_after_use", "cold_on_demand")


def busy_sandboxes(arrival_rate: float, exec_s: float) -> float:
    """Little's law: mean number of sandboxes actively running code = λ · t_exec."""
    return arrival_rate * exec_s


def warming_sandboxes(arrival_rate: float, cold_start_s: float) -> float:
    """A replace-after-use pool also holds, on average, λ · t_cold sandboxes warming up."""
    return arrival_rate * cold_start_s


def mean_occupancy(arrival_rate: float, exec_s: float, cold_start_s: float, *,
                   replace_after_use: bool = True) -> float:
    """Little's law on the slot cycle: λ · (t_exec + t_cold) slots busy or warming, ON AVERAGE.

    This is the offered load *a* (Erlangs). It is the floor, not a pool size: ``erlang_c(a, ceil(a))`` is
    close to 1, and at c = a exactly the queue never clears. Size with ``slots_for_wait_target``.
    """
    hold = exec_s + (cold_start_s if replace_after_use else 0.0)
    return arrival_rate * hold


def erlang_c(offered_load: float, servers: int) -> float:
    """P(wait) for an M/M/c queue. offered_load a = λ·(slot hold time) in Erlangs; c must exceed a."""
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


def expected_wait_s(arrival_rate: float, hold_s: float, servers: int) -> float:
    """E[Wq] for M/M/c: C(a,c) / (c·μ − λ), μ = 1/hold. Infinite when the queue never clears."""
    a = arrival_rate * hold_s
    mu = 1.0 / hold_s
    denom = servers * mu - arrival_rate
    if denom <= 0:
        return math.inf
    return erlang_c(a, servers) / denom


def servers_for_wait_target(arrival_rate: float, hold_s: float, max_p_wait: float) -> int:
    """Smallest server count whose P(wait) is at or below the target, for slots held ``hold_s`` each."""
    a = arrival_rate * hold_s
    c = math.floor(a) + 1
    while erlang_c(a, c) > max_p_wait:
        c += 1
    return c


def slots_for_wait_target(arrival_rate: float, exec_s: float, cold_start_s: float, max_p_wait: float, *,
                          replace_after_use: bool = True) -> int:
    """Pool slots so at most ``max_p_wait`` of requests wait for a warm sandbox.

    Erlang C on a = λ·(t_exec + t_cold) for replace-after-use (λ·t_exec for a reuse pool). Exact for
    exponential hold times; with a fixed cold start the hold is less variable than exponential, so the real
    P(wait) comes out a little lower — ``simulate()`` shows by how much.
    """
    hold = exec_s + (cold_start_s if replace_after_use else 0.0)
    return servers_for_wait_target(arrival_rate, hold, max_p_wait)


def cost_per_execution(exec_s: float, cold_start_s: float, node_cost_per_s: float, *,
                       reuse: bool = False, fixed_per_exec: float = 0.0) -> float:
    """Sandbox-seconds one execution consumes × the node's $/s (+ any fixed per-execution fee).

    One-shot sandboxes — a pod per call, a replace-after-use warm pool, pay-per-use services — each spend a
    cold start that someone pays for, so held = t_exec + t_cold: a warm pool moves the cold start off the
    *latency* path, not off the *bill*. ``reuse=True`` (one warm sandbox serving many executions) amortises
    the cold start to ~0 per execution, at the price of state carried between executions. Idle headroom
    comes on top; ``pool_cost_per_execution`` is the fleet view that includes it.
    """
    held_s = exec_s + (0.0 if reuse else cold_start_s)
    return held_s * node_cost_per_s + fixed_per_exec


def pool_cost_per_execution(arrival_rate: float, slots: int, node_cost_per_s: float,
                            fixed_per_exec: float = 0.0) -> float:
    """The fleet view: every slot is paid for all the time, so cost/execution = slots × $/s ÷ λ.

    It equals ``cost_per_execution`` plus the idle headroom (c − a)/λ sandbox-seconds per execution that
    the wait target buys.
    """
    return slots * node_cost_per_s / arrival_rate + fixed_per_exec


def actions_cost(actions_per_turn: float, cost_per_action: float) -> float:
    """Scaling primer §1: the per-turn code-execution bill is actions/turn × cost/action."""
    return actions_per_turn * cost_per_action


@dataclass
class SimResult:
    completed: int
    mean_wait_s: float
    p95_wait_s: float
    frac_waited: float
    cold_on_path: int        # cold starts a request waited through
    rewarms: int             # cold starts done in the background (replace-after-use)
    max_busy: int            # slots busy or warming at an arrival, at most

    def as_row(self) -> dict:
        return {"completed": self.completed, "mean_wait_s": round(self.mean_wait_s, 4),
                "p95_wait_s": round(self.p95_wait_s, 4), "frac_waited": round(self.frac_waited, 3),
                "cold_on_path": self.cold_on_path, "rewarms": self.rewarms, "max_busy": self.max_busy}


def simulate(arrival_rate: float, exec_s: float, cold_start_s: float, servers: int, *,
             n: int = 20000, mode: str = "replace_after_use", seed: int = 0) -> SimResult:
    """A discrete-event run of a fixed pool of ``servers`` slots. SIMULATED — label its output so.

    Poisson arrivals, exponential execution times, a fixed cold start, FCFS. Modes:

    * ``reuse`` — slots stay warm across executions (a session pool): hold = t_exec.
    * ``replace_after_use`` — each slot starts warm; after one execution its sandbox is destroyed and a
      replacement warms for t_cold **in the background**: hold = t_exec + t_cold, and a request waits only
      when no warm sandbox is ready.
    * ``cold_on_demand`` — no pool: every request cold-starts its own sandbox **on its path**, so its wait
      includes t_cold, and the slot is held for t_cold + t_exec.
    """
    import random
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rng = random.Random(seed)
    mu = 1.0 / exec_s
    free_at = [0.0] * servers            # when each slot next has a warm sandbox ready
    t = 0.0
    waits: list[float] = []
    cold_on_path = rewarms = max_busy = 0
    for _ in range(n):
        t += rng.expovariate(arrival_rate)
        slot = min(range(servers), key=lambda i: free_at[i])
        start = max(t, free_at[slot])
        service = rng.expovariate(mu)
        if mode == "cold_on_demand":
            wait = (start - t) + cold_start_s
            free_at[slot] = start + cold_start_s + service
            cold_on_path += 1
        elif mode == "replace_after_use":
            wait = start - t
            free_at[slot] = start + service + cold_start_s     # the replacement warms off the request path
            rewarms += 1
        else:
            wait = start - t
            free_at[slot] = start + service
        waits.append(wait)
        max_busy = max(max_busy, sum(1 for f in free_at if f > t))
    waits.sort()
    p95 = waits[int(0.95 * len(waits)) - 1]
    return SimResult(n, sum(waits) / len(waits), p95, sum(1 for w in waits if w > 1e-6) / len(waits),
                     cold_on_path, rewarms, max_busy)
