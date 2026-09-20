"""Simulated systems of record with the two things that matter: latency and a QPS ceiling.

The CRM takes 200 requests/s, the billing mainframe 40. Reads are idempotent; ``create_ticket``
is a write — the loop must never run it twice for the same turn.
"""

from __future__ import annotations

import math
import random

from .clock import CLOCK
from .resilience import TokenBucket


class ToolError(Exception):
    pass


class System:
    def __init__(self, name: str, p50_s: float, max_qps: float, rng: random.Random):
        self.name, self.p50_s, self.rng = name, p50_s, rng
        self.bucket = TokenBucket(max_qps, max_qps / 4)
        self.calls = self.rejected = 0

    async def call(self, result: dict) -> dict:
        self.calls += 1
        if self.bucket.tokens < 1:  # the real system answers 429 instead of queueing
            self.rejected += 1
            raise ToolError(f"{self.name}: rate limited")
        await self.bucket.acquire(1)
        await CLOCK.sleep(self.p50_s * math.exp(self.rng.gauss(0, 0.5)))
        return result


class Tools:
    """name -> (idempotent?, handler). ``executed`` records every write for the idempotency tests."""

    def __init__(self, seed: int = 11):
        rng = random.Random(seed)
        self.crm = System("crm", 0.12, 200, rng)
        self.billing = System("billing", 0.60, 40, rng)
        self.network = System("network", 0.05, 500, rng)
        self.tickets: list[str] = []
        self.specs = {
            "get_customer": (True, lambda a: self.crm.call({"customer_id": a.get("customer_id", "c1"), "plan": "plus-50"})),
            "get_invoice": (True, lambda a: self.billing.call({"total": 41.5, "lines": ["monthly 28.0", "pro-rated 9.5", "roaming 4.0"]})),
            "list_plans": (True, lambda a: self.crm.call({"plans": ["lite-20", "plus-50", "max-unl"]})),
            "check_network": (True, lambda a: self.network.call({"incident": True, "eta": "2h"})),
            "search_kb": (True, lambda a: self.network.call({"results": ["kb-101 troubleshooting"]})),
            "create_ticket": (False, self._create_ticket),
        }

    async def _create_ticket(self, args: dict) -> dict:
        ticket = await self.crm.call({"ticket_id": f"T-{1000 + len(self.tickets) + 1}"})
        self.tickets.append(ticket["ticket_id"])
        return ticket

    def names(self, degrade_level: int = 0) -> list[str]:
        """Tools offered to the model; at level 2 writes and the slow billing call are withheld."""
        out = list(self.specs)
        if degrade_level >= 2:
            out = [n for n in out if self.specs[n][0] and n != "get_invoice"]
        return out

    def is_idempotent(self, name: str) -> bool:
        return self.specs[name][0]

    async def run(self, name: str, args: dict) -> dict:
        try:
            return await self.specs[name][1](args)
        except ToolError as e:
            return {"error": str(e)}   # structured error: the model can answer around it
