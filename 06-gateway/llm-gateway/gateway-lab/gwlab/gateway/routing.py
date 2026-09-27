"""Model routing and fallback chains: which target serves a request, and what happens when it fails.

The one idea (PRIMER §2): a client names an *alias*; the gateway turns it into an ordered chain of (provider,
model) targets — filtered by capability (tools, context window) and residency, ordered by a policy (as
listed, cheapest for this request's shape, fastest by an EWMA of observed TTFT, or a canary share) — and tries
them in order. Three rules make a chain safe:

1. Only some failures fall through: 429, 5xx, 529, timeouts, connection errors and context-length errors
   (a bigger-context target can take it). A 400, an auth failure or a content-policy refusal fails the request:
   the next provider would refuse it too, or the failure is ours to fix.
2. Fall back only **before the first byte**. Once a chunk has reached the client, splicing another model's
   output onto it is wrong; the gateway surfaces the error in the stream instead.
3. A **breaker per target** stops the chain from paying a timeout on a dead target every request. This one
   follows 07.2's `CircuitBreaker` (gcp-agent-platform-lab notebook 10, the canonical home): open after
   `failure_threshold` *consecutive* failures, half-open after `recovery_timeout_s`, one probe decides.
   (`scalelab.resilience.CircuitBreaker` counts a failure *ratio* in a window instead; either works, pick one.)

A self-hosted target is a *pool*: the gateway picks the pool, the 05 endpoint picker picks the pod
(05 PRIMER §1.3, §7).
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field

FALLTHROUGH, FAIL = "fallthrough", "fail"
FALLTHROUGH_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})


def classify(status: int | None = None, code: str | None = None, *, timeout: bool = False,
             connect_error: bool = False) -> str:
    """Does this failure fall through to the next target, or fail the request?"""
    if timeout or connect_error:
        return FALLTHROUGH
    if code == "context_length_exceeded":
        return FALLTHROUGH
    if code == "content_filter":
        return FAIL
    return FALLTHROUGH if status in FALLTHROUGH_STATUS else FAIL


class Breaker:
    """closed -> open after `threshold` consecutive failures; open -> half_open after `recovery_s`;
    half_open admits one probe at a time: success closes, failure re-opens."""

    def __init__(self, threshold: int = 3, recovery_s: float = 5.0, clock=time.monotonic):
        self.threshold, self.recovery_s, self.clock = threshold, recovery_s, clock
        self._state, self.failures, self.opened_at, self.probing = "closed", 0, 0.0, False
        self.opens = self.rejections = 0

    @property
    def state(self) -> str:
        if self._state == "open" and self.clock() - self.opened_at >= self.recovery_s:
            self._state, self.probing = "half_open", False
        return self._state

    def allow(self) -> bool:
        s = self.state
        if s == "closed":
            return True
        if s == "half_open" and not self.probing:
            self.probing = True
            return True
        self.rejections += 1
        return False

    def record(self, ok: bool) -> None:
        s = self.state
        if ok:
            self._state, self.failures, self.probing = "closed", 0, False
            return
        self.failures += 1
        if s == "half_open" or self.failures >= self.threshold:
            self._state, self.opened_at, self.failures, self.probing = "open", self.clock(), 0, False
            self.opens += 1

    def release(self) -> None:
        """A probe that ended without a verdict (e.g. a non-fallthrough error) frees the half-open slot."""
        self.probing = False


@dataclass
class Needs:
    prompt_tokens: int
    output_tokens: int
    tools: bool = False
    request_id: str = ""


@dataclass
class Plan:
    alias: str
    policy: str
    targets: list                          # Model objects, in the order they will be tried
    skipped: list = field(default_factory=list)   # (model id, reason)

    def ids(self) -> list:
        return [m.id for m in self.targets]


class Router:
    def __init__(self, cfg, clock=time.monotonic, alpha: float = 0.3, ttft_prior_s: float = 0.5):
        self.cfg, self.clock, self.alpha, self.prior = cfg, clock, alpha, ttft_prior_s
        self.breakers = {m: Breaker(cfg.breaker.failure_threshold, cfg.breaker.recovery_timeout_s, clock)
                         for m in cfg.models}
        self.ewma: dict = {}

    def plan(self, alias_name: str, tenant, needs: Needs) -> Plan:
        alias = self.cfg.aliases[alias_name]
        keep, skipped = [], []
        for mid in alias.targets:
            m = self.cfg.models[mid]
            prov = self.cfg.providers[m.provider]
            if needs.tools and not m.tools:
                skipped.append((mid, "no_tools"))
            elif needs.prompt_tokens + needs.output_tokens > m.context_window:
                skipped.append((mid, "context_window"))
            elif tenant is not None and tenant.regions and prov.region not in tenant.regions:
                skipped.append((mid, "residency"))
            else:
                keep.append(m)
        keep = self.order(alias, keep, needs)
        return Plan(alias_name, alias.policy, keep, skipped)

    def order(self, alias, models: list, needs: Needs) -> list:
        if alias.policy == "cheapest":
            return sorted(models, key=lambda m: m.price.input * needs.prompt_tokens + m.price.output * needs.output_tokens)
        if alias.policy == "ewma_ttft":
            return sorted(models, key=lambda m: self.ewma.get(m.id, self.prior))
        if alias.policy == "canary" and len(models) >= 2:
            if zlib.crc32(needs.request_id.encode()) % 1000 < alias.canary_weight * 1000:
                return [models[1], models[0]] + models[2:]
        return list(models)

    def record(self, model_id: str, ok: bool, ttft_s: float | None = None) -> None:
        self.breakers[model_id].record(ok)
        if ok and ttft_s is not None:
            prev = self.ewma.get(model_id)
            self.ewma[model_id] = ttft_s if prev is None else self.alpha * ttft_s + (1 - self.alpha) * prev

    def state(self) -> dict:
        return {m: {"breaker": b.state, "opens": b.opens, "rejections": b.rejections,
                    "ewma_ttft_ms": round(self.ewma[m] * 1e3, 1) if m in self.ewma else None}
                for m, b in self.breakers.items()}


def chain_availability(availability: list, common_mode: float = 0.0) -> float:
    """P(some target answers): independent failures, times (1 − P(a common-mode event takes all down))."""
    p_all_fail = 1.0
    for a in availability:
        p_all_fail *= 1.0 - a
    return (1.0 - common_mode) * (1.0 - p_all_fail)


def expected_chain(steps: list) -> dict:
    """Expected outcome of trying targets in order.

    steps: [(p_fail, fail_latency_s, ttft_s, cost_usd), ...] — a failed attempt costs its `fail_latency_s` and no
    tokens (a 429 or 5xx before the first byte is not billed). Returns P(success), E[TTFT | success] and
    E[cost | success], counting the time spent on the failures before the target that answered.
    """
    p_reach, p_ok, e_ttft, e_cost, waited = 1.0, 0.0, 0.0, 0.0, 0.0
    for p_fail, fail_s, ttft_s, cost in steps:
        p_here = p_reach * (1 - p_fail)
        p_ok += p_here
        e_ttft += p_here * (waited + ttft_s)
        e_cost += p_here * cost
        waited += fail_s
        p_reach *= p_fail
    return {"p_success": p_ok, "ttft_s": e_ttft / p_ok if p_ok else float("nan"),
            "cost_usd": e_cost / p_ok if p_ok else float("nan")}
