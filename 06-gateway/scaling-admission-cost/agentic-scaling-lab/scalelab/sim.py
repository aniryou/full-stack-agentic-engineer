"""A load generator: many users, one shared pool, one admission controller.

``simulate`` runs ``users`` virtual customers for ``duration_s`` virtual seconds. Each user
submits a turn, waits for it, thinks, and repeats. Every turn passes through the admission
controller (which may shed it) and the model gateway pieces (bucket, breaker, retries).
The result is a DataFrame of per-turn samples plus a one-second timeline of in-flight turns,
degrade level and pool utilisation — enough to see the overload feedback loop and to
derive Cloud Run settings from measurements.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .admission import AdmissionConfig, AdmissionController
from .clock import CLOCK
from .loop import Budget, Store, run_turn
from .model import FakeModel, SharedPool
from .resilience import CircuitBreaker, TokenBucket
from .tools import Tools

MESSAGES = ["my bill is higher than usual this month", "no signal since this morning, is there an outage",
            "which plan should I be on, I keep running out of data", "I want to complain and speak to a human",
            "how does roaming work"]


@dataclass
class Setup:
    """Everything one simulation shares. Build with ``make_setup``."""
    standard: FakeModel
    lite: FakeModel
    tools: Tools
    store: Store
    admission: AdmissionController
    bucket: TokenBucket | None
    breakers: dict[str, CircuitBreaker]      # one per model: a sibling model has its own pool and its own breaker
    budget: Budget = field(default_factory=Budget)


def make_setup(*, pool_tpm: int | None = 2_000_000, max_inflight: int = 85, client_tpm: int | None = None,
               dwell_s: float = 15.0, naive: bool = False, seed: int = 1) -> Setup:
    """``pool_tpm`` is the provider's shared pool (None = unlimited); ``client_tpm`` a client-side smoothing bucket.

    ``naive=True`` switches every protection off (no degrade levels, no cap, no breaker): retries only.
    """
    pool = SharedPool(pool_tpm) if pool_tpm else None
    standard = FakeModel(pool=pool, rng=random.Random(seed))
    lite = FakeModel(model="gemini-3.5-flash-lite", pool=SharedPool(pool_tpm * 2) if pool_tpm else None,
                     ttft_median_s=0.35, tokens_per_s=350, answer_tokens=150, rng=random.Random(seed + 1))
    bucket = TokenBucket(client_tpm / 60, client_tpm / 60 * 6) if client_tpm else None
    if naive:
        cfg = AdmissionConfig(max_inflight=10**9, rate_limited_degrade=10.0, queue_age_degrade_s=1e9, queue_age_shed_s=1e9)
        breakers = {k: CircuitBreaker(threshold=10**9, min_calls=10**9) for k in ("standard", "lite")}
    else:
        cfg = AdmissionConfig(max_inflight=max_inflight, dwell_s=dwell_s)
        breakers = {"standard": CircuitBreaker(), "lite": CircuitBreaker()}
    return Setup(standard, lite, Tools(seed), Store(), AdmissionController(cfg), bucket, breakers)


async def simulate(setup: Setup, *, users: int = 40, duration_s: float = 90.0, think_s: float = 5.0,
                   priority: int = 5, seed: int = 1) -> "SimResult":
    rng = random.Random(seed)
    samples: list[dict] = []
    timeline: list[dict] = []
    t_start = CLOCK.now()
    t_stop = t_start + duration_s

    async def one_user(uid: int):
        n = 0
        while CLOCK.now() < t_stop:
            n += 1
            text = rng.choice(MESSAGES)
            submitted = CLOCK.now()
            decision = setup.admission.admit(priority, circuit_open=setup.breakers["standard"].state != "closed")
            if not decision.admitted:
                samples.append({"user": uid, "outcome": "shed", "level": decision.level, "t": submitted - t_start,
                                "latency_s": None, "cost_usd": 0.0, "tokens": 0, "rate_limited": 0, "steps": 0, "error": None})
                await CLOCK.sleep(decision.retry_after * rng.uniform(0.5, 1.5))  # jittered client retry
                continue
            tier = "lite" if decision.level >= 1 else "standard"
            try:
                r = await run_turn(f"u{uid}-{n}", text, model=getattr(setup, tier), tools=setup.tools, store=setup.store,
                                   budget=setup.budget, degrade_level=decision.level, bucket=setup.bucket, breaker=setup.breakers[tier])
            finally:
                setup.admission.release()
            for c in (True,) * r.rate_limited + (False,) * max(0, r.model_attempts - r.rate_limited):
                setup.admission.note_model_call(c)
            samples.append({"user": uid, "outcome": r.status, "level": decision.level, "t": submitted - t_start,
                            "latency_s": r.latency_s, "cost_usd": r.cost_usd, "tokens": r.tokens,
                            "rate_limited": r.rate_limited, "steps": r.steps, "error": r.error})
            await CLOCK.sleep(rng.expovariate(1 / think_s))

    async def sampler():
        while True:
            pool = setup.standard.pool
            timeline.append({"t": CLOCK.now() - t_start, "inflight": setup.admission.inflight, "level": setup.admission.level,
                             "pool_utilisation": pool.utilisation if pool else 0.0,
                             "rate_limited_ratio": setup.admission.rate_limited_ratio(),
                             "breaker": setup.breakers["standard"].state})
            await CLOCK.sleep(1.0)

    task = asyncio.create_task(sampler())
    await asyncio.gather(*(one_user(u) for u in range(users)))
    task.cancel()
    return SimResult(pd.DataFrame(samples), pd.DataFrame(timeline), CLOCK.now() - t_start, setup)


@dataclass
class SimResult:
    samples: pd.DataFrame
    timeline: pd.DataFrame
    duration_s: float
    setup: Setup

    def summary(self) -> dict:
        df, done = self.samples, self.samples[self.samples.outcome == "completed"]
        lat = done.latency_s.to_numpy(dtype=float) if len(done) else np.array([np.nan])
        return {
            "turns": len(df), "completed": len(done), "failed": int((df.outcome == "failed").sum()),
            "shed": int((df.outcome == "shed").sum()), "shed_rate": float((df.outcome == "shed").mean()) if len(df) else 0.0,
            "throughput_turns_per_s": len(done) / self.duration_s,
            "p50_s": float(np.nanpercentile(lat, 50)), "p95_s": float(np.nanpercentile(lat, 95)),
            "rate_limited_calls": int(df.rate_limited.sum()), "max_level": int(df.level.max()) if len(df) else 0,
            "degraded_share": float((df.level >= 1).mean()) if len(df) else 0.0,
            "breaker_trips": self.setup.breakers["standard"].trips,
            "cost_per_turn_usd": float(done.cost_usd.mean()) if len(done) else float("nan"),
            "pool_rejected": self.setup.standard.pool.rejected if self.setup.standard.pool else 0,
        }


def compare(results: dict[str, SimResult]) -> pd.DataFrame:
    return pd.DataFrame({name: r.summary() for name, r in results.items()}).T
