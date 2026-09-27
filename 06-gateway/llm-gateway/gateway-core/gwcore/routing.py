"""Model routing and fallback chains (PRIMER §2).

The one idea: an alias ("chat") resolves to an ordered chain of (provider, model, region) targets,
filtered by what the request needs; a request falls through to the next target only for a failure
that another target could fix (429, 5xx, a timeout, a context too long) -- never for a bad request,
bad credentials or a content-policy refusal -- and only before the first byte has reached the client.
A breaker per target turns a dead provider into an instant fall-through instead of a timeout per request.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass

from .providers import CATALOGUE

# 408: the provider timed out the request; 529: Anthropic's overloaded_error. None: no answer before our timeout.
FALLS_THROUGH = {408, 429, 500, 502, 503, 504, 529, None}


def falls_through(status: int | None, code: str | None = None) -> bool:
    """Would the next target plausibly succeed where this one failed?"""
    if status == 400 and code == "context_length_exceeded":
        return True                                       # a longer-context target can take it
    return status in FALLS_THROUGH                        # 400/401/403/404/content policy: same answer everywhere


@dataclass(frozen=True)
class Target:
    provider: str
    model: str
    region: str = "global"
    weight: float = 1.0                                   # canary share when the policy is "canary"


class Breaker:
    """The 07.2 lab's rule (``agentlab/reliability/breaker.py``): open after ``threshold`` *consecutive*
    failures, fail fast for ``cooldown`` s, then let one probe decide. (``scalelab``'s breaker trips on a
    failure *ratio* in a window instead; both are fine per target, pick one and say which.)

    A probe whose outcome says nothing about the provider's health -- a 400 for this request, our own 401 --
    must not leave the breaker half-open forever: ``release()`` hands the probe back so the next request probes."""

    def __init__(self, threshold: int = 3, cooldown: float = 30.0):
        self.threshold, self.cooldown = threshold, cooldown
        self.failures, self.opened_at, self.probing, self.trips = 0, None, False, 0

    def state(self, now: float) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if now - self.opened_at >= self.cooldown else "open"

    def allow(self, now: float) -> bool:
        s = self.state(now)
        if s == "open" or (s == "half_open" and self.probing):
            return False
        self.probing = s == "half_open"
        return True

    def release(self) -> None:
        """The probe ended without a health verdict: let the next request probe instead."""
        self.probing = False

    def record(self, ok: bool, now: float) -> None:
        if ok:
            self.failures, self.opened_at, self.probing = 0, None, False
            return
        self.failures += 1
        if self.probing or self.failures >= self.threshold:
            self.opened_at, self.probing, self.trips = now, False, self.trips + 1


def ewma(prev: float | None, x: float, alpha: float = 0.2) -> float:
    return x if prev is None else alpha * x + (1 - alpha) * prev


def needs(request: dict) -> set:
    """Capabilities a request requires of its target."""
    n = set()
    if request.get("tools"):
        n.add("tools")
    if request.get("reasoning_effort") not in (None, "none", "minimal"):
        n.add("reasoning")                                # effort routing: only thinking models take high effort
    return n


def blended_cost(model: str, prompt_tokens: int, output_tokens: int) -> float:
    p = CATALOGUE[model].price
    return 0.0 if p is None else (prompt_tokens * p[0] + output_tokens * p[1]) / 1e6


class Router:
    """Chains per alias (and per tenant tier: ``chat@gold``), capability and region filters, a policy, breakers."""

    def __init__(self, chains: dict, *, threshold: int = 3, cooldown: float = 30.0):
        self.chains = chains
        self.breakers = defaultdict(lambda: Breaker(threshold, cooldown))
        self.ttft: dict = {}

    def candidates(self, alias: str, request: dict, *, now: float, tier: str | None = None, policy: str = "ordered",
                   prompt_tokens: int = 0, regions: set | None = None, request_id: str = "") -> list:
        chain = self.chains.get(f"{alias}@{tier}") or self.chains[alias]
        want, out_tokens = needs(request), request.get("max_completion_tokens") or 1024
        ok = [t for t in chain if want <= CATALOGUE[t.model].capabilities
              and prompt_tokens + out_tokens <= CATALOGUE[t.model].context
              and (regions is None or t.region in regions)]     # residency: "global" may serve from anywhere
        if policy == "cheapest":
            ok.sort(key=lambda t: blended_cost(t.model, prompt_tokens, out_tokens))
        elif policy == "ewma_ttft":
            ok.sort(key=lambda t: self.ttft.get(t, 0.0))  # unmeasured targets first, so they get measured
        elif policy == "canary" and len(ok) > 1:          # a stable share of requests to the second target
            share = ok[1].weight / (ok[0].weight + ok[1].weight)
            if int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) / 2 ** 32 < share:
                ok[0], ok[1] = ok[1], ok[0]
        return ok

    def allow(self, target: Target, now: float) -> bool:
        return self.breakers[target].allow(now)

    def release(self, target: Target) -> None:
        self.breakers[target].release()

    def observe(self, target: Target, ok: bool, now: float, ttft: float | None = None) -> None:
        self.breakers[target].record(ok, now)
        if ok and ttft is not None:
            self.ttft[target] = ewma(self.ttft.get(target), ttft)


def chain_availability(avails: list, common_mode: float = 0.0) -> float:
    """P(some target answers) = (1 - common-mode failure) x (1 - product of independent failures)."""
    return (1 - common_mode) * (1 - math.prod(1 - a for a in avails))


def chain_cost(steps: list) -> dict:
    """Expected latency and cost of walking a chain. Each step: p_ok, t_ok, t_fail, cost_ok (cost_fail optional).
    A failure's time is paid by every later outcome, so a slow failure (a timeout) dominates."""
    reach, t, c = 1.0, 0.0, 0.0
    for s in steps:
        p = s["p_ok"]
        t += reach * (p * s["t_ok"] + (1 - p) * s["t_fail"])
        c += reach * (p * s["cost_ok"] + (1 - p) * s.get("cost_fail", 0.0))
        reach *= 1 - p
    return {"latency": t, "cost": c, "p_fail": reach}
