"""Streaming-aware rate limits: requests per minute AND tokens per minute, reserved -> streamed -> reconciled.

The one idea (PRIMER §4, on top of the scaling primer §5.1 and §5.3): a request's cost is unknown when it is
admitted — the output length is decided while it streams and is heavy-tailed (00.5 PRIMER §7). A bucket that
charges each request a fixed guess up front (`per_request` mode, the `TokenBucket` rule of
`scalelab.resilience`) over-admits by (prompt + actual output) / (prompt + guess): cheap to build, wrong
exactly when outputs grow. `reserve` mode instead

    admit:     reserve prompt_estimate + min(requested cap or default estimate, hard cap) from the tenant's TPM
    stream:    if the output passes the reservation, debit the excess as it streams (the bucket may go into debt)
    reconcile: at the end, refund reservation − actual (or debit the difference) from the provider's `usage`

and enforces RPM beside TPM, per tenant, under an optional gateway-wide TPM (a provider share): hierarchical
limits. One process here; with two gateway instances the buckets move to Redis behind a Lua script (one
round trip per check, scaling primer §5.1).

The "minute" is `Limits.minute_s` seconds so a notebook can compress time (it says so when it does).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


class Bucket:
    """`capacity` tokens, refilled at `per_s` per second. The level may go negative (debt) through `take`."""

    def __init__(self, capacity: float, per_s: float, clock=time.monotonic):
        self.capacity, self.per_s, self.clock = float(capacity), float(per_s), clock
        self.level = float(capacity)
        self.t = clock()

    def _refill(self) -> None:
        now = self.clock()
        self.level = min(self.capacity, self.level + (now - self.t) * self.per_s)
        self.t = now

    def available(self) -> float:
        self._refill()
        return self.level

    def can_take(self, n: float) -> bool:
        """A request larger than the burst is admitted only from a full bucket (then it goes into debt)."""
        return self.available() >= min(n, self.capacity) - 1e-9

    def take(self, n: float) -> None:
        self._refill()
        self.level -= n

    def refund(self, n: float) -> None:
        self._refill()
        self.level = min(self.capacity, self.level + n)

    def retry_after(self, n: float) -> float:
        need = min(n, self.capacity) - self.available()
        return max(0.0, need / self.per_s) if self.per_s > 0 else math.inf


@dataclass
class Reservation:
    tenant: str
    mode: str
    prompt_estimate: int
    reserved_output: int
    charged_output: int                   # output tokens charged so far (reservation, then streamed excess)
    buckets: list = field(default_factory=list)

    @property
    def charged(self) -> int:
        return self.prompt_estimate + self.charged_output


@dataclass
class Decision:
    ok: bool
    limit: str = ""                       # which limit said no: rpm | tpm | global_tpm
    retry_after: float = 0.0
    reservation: Reservation | None = None
    headers: dict = field(default_factory=dict)


class Limiter:
    def __init__(self, limits, tenants: dict, clock=time.monotonic):
        self.cfg, self.tenants, self.clock = limits, tenants, clock
        self._b: dict = {}
        self.global_tpm = Bucket(limits.global_tpm, limits.global_tpm / limits.minute_s, clock) if limits.global_tpm else None
        self.rejections: dict = {}

    def _buckets(self, tenant: str, rpm: int | None = None, tpm: int | None = None):
        if tenant not in self._b:
            t = self.tenants.get(tenant)
            rpm = rpm or (t.rpm if t else 600)
            tpm = tpm or (t.tpm if t else 200_000)
            m = self.cfg.minute_s
            self._b[tenant] = (Bucket(rpm, rpm / m, self.clock), Bucket(tpm, tpm / m, self.clock))
        return self._b[tenant]

    def output_reservation(self, requested_output: int | None) -> int:
        c = self.cfg
        if c.mode == "per_request":
            return c.per_request_output_guess
        return min(requested_output or c.default_output_estimate, c.hard_output_cap)

    def admit(self, tenant: str, prompt_estimate: int, requested_output: int | None = None, *,
              rpm: int | None = None, tpm: int | None = None) -> Decision:
        """Check RPM, tenant TPM and the global TPM together; take from all three only if all admit."""
        rpm_b, tpm_b = self._buckets(tenant, rpm, tpm)
        out = self.output_reservation(requested_output)
        tokens = prompt_estimate + out
        checks = [("rpm", rpm_b, 1), ("tpm", tpm_b, tokens)]
        if self.global_tpm:
            checks.append(("global_tpm", self.global_tpm, tokens))
        headers = self.headers(tenant)
        for name, b, n in checks:
            if not b.can_take(n):
                ra = b.retry_after(n)
                self.rejections[(tenant, name)] = self.rejections.get((tenant, name), 0) + 1
                return Decision(False, name, ra, None, {**headers, "Retry-After": str(max(1, math.ceil(ra)))})
        for _, b, n in checks:
            b.take(n)
        res = Reservation(tenant, self.cfg.mode, prompt_estimate, out, out,
                          [b for name, b, _ in checks if name != "rpm"])
        return Decision(True, reservation=res, headers=self.headers(tenant))

    def on_output(self, res: Reservation, streamed_tokens: int) -> bool:
        """Called as output streams. Reserve mode debits tokens beyond what was charged; returns False once the
        hard cap is reached (the relay then stops the stream)."""
        if res.mode == "reserve" and streamed_tokens > res.charged_output:
            extra = streamed_tokens - res.charged_output
            for b in res.buckets:
                b.take(extra)
            res.charged_output = streamed_tokens
        return streamed_tokens < self.cfg.hard_output_cap

    def reconcile(self, res: Reservation, prompt_tokens: int, completion_tokens: int) -> int:
        """Settle a reservation against the actual counts; returns tokens refunded (negative = debited).
        `per_request` mode never reconciles: that is its flaw."""
        if res.mode != "reserve":
            return 0
        delta = res.charged - (prompt_tokens + completion_tokens)
        for b in res.buckets:
            if delta >= 0:
                b.refund(delta)
            else:
                b.take(-delta)
        res.charged_output -= delta
        return delta

    def headers(self, tenant: str) -> dict:
        """`x-ratelimit-*` headers as OpenAI's docs describe them (prose docs, not the OpenAPI spec: verify)."""
        rpm_b, tpm_b = self._buckets(tenant)
        m = self.cfg.minute_s
        return {"x-ratelimit-limit-requests": str(int(rpm_b.capacity)),
                "x-ratelimit-remaining-requests": str(max(0, int(rpm_b.available()))),
                "x-ratelimit-limit-tokens": str(int(tpm_b.capacity)),
                "x-ratelimit-remaining-tokens": str(max(0, int(tpm_b.available()))),
                "x-ratelimit-reset-tokens": f"{max(0.0, (tpm_b.capacity - tpm_b.available()) / tpm_b.per_s):.2f}s",
                "x-gwlab-minute-s": f"{m:g}"}

    def state(self) -> dict:
        return {t: {"rpm_available": round(r.available(), 2), "tpm_available": round(k.available(), 1),
                    "rpm": r.capacity, "tpm": k.capacity} for t, (r, k) in self._b.items()}


def over_admission(prompt_tokens: float, mean_output: float, charged_output: float) -> float:
    """How many times its TPM a bucket admits when it charges `prompt + charged_output` per request but requests
    actually cost `prompt + mean_output` (per_request mode at saturation)."""
    return (prompt_tokens + mean_output) / (prompt_tokens + charged_output)
