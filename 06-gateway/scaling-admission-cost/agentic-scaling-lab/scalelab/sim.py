"""A load generator: many users, one model backend, one admission controller.

``simulate`` runs ``users`` virtual customers for ``duration_s`` virtual seconds. Each user
submits a turn, waits for it, thinks, and repeats. Every turn passes through the admission
controller (which may shed it) and the model gateway pieces (bucket, breaker, retries).

Three backends, chosen with ``make_setup(mode=...)`` (or ``SCALELAB_BACKEND`` when ``mode`` is not given):

    "hosted"  a provider's API — a shared TPM (and RPS) limit that answers 429 (the default)
    "local"   your own vLLM replicas — they queue; latency is the batch you are running
    "hybrid"  the fleet first, the API when the fleet is saturated (spill-over)

and two providers, chosen with ``provider=`` (or ``SCALELAB_PROVIDER``): ``"gemini"``, the Google Cloud anchor
scenario and the default on the API, and ``"mistral"``, whose open-weight models are what a fleet serves
(the default, and the only choice, for "local" and "hybrid").

The result is a DataFrame of per-turn samples plus a one-second timeline of in-flight turns,
degrade level, backend saturation and queue depth — enough to see the overload feedback loop in
both of its forms (a 429 storm; a silent latency collapse) and to derive Cloud Run, Kubernetes and
vLLM settings from measurements.
"""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .admission import AdmissionConfig, AdmissionController
from .clock import CLOCK
from .loop import Budget, Store, run_turn
from .model import FakeModel, HostedBackend, HybridBackend, ServerPool, SharedPool, fake_model
from .resilience import CircuitBreaker, TokenBucket
from .serving import replica
from .tools import Tools

MESSAGES = ["my bill is higher than usual this month", "no signal since this morning, is there an outage",
            "which plan should I be on, I keep running out of data", "I want to complain and speak to a human",
            "how does roaming work"]
CONTEXT_TOKENS, PREFIX_TOKENS, TURN_SECONDS = 5_200, 3_000, 6.0
MODES = ("hosted", "local", "hybrid")
BACKEND_ENV, PROVIDER_ENV = "SCALELAB_BACKEND", "SCALELAB_PROVIDER"


@dataclass
class Setup:
    """Everything one simulation shares. Build with ``make_setup``."""
    mode: str
    provider: str
    standard: FakeModel
    lite: FakeModel
    tools: Tools
    store: Store
    admission: AdmissionController
    bucket: TokenBucket | None
    breakers: dict[str, CircuitBreaker]      # one per model: a sibling model has its own pool and its own breaker
    server: ServerPool | None = None
    budget: Budget = field(default_factory=Budget)


def resolve_backend(mode: str | None = None, provider: str | None = None) -> tuple[str, str]:
    """``(mode, provider)``: the arguments if given, else ``SCALELAB_BACKEND`` / ``SCALELAB_PROVIDER``, else the
    defaults — "hosted" on "gemini"; a fleet ("local", "hybrid") serves Mistral's open-weight models."""
    mode = (mode or os.environ.get(BACKEND_ENV) or "hosted").strip().lower()
    if mode not in MODES:
        raise ValueError(f"unknown backend {mode!r} (from the argument or {BACKEND_ENV}); choose one of {MODES}")
    provider = (provider or os.environ.get(PROVIDER_ENV) or ("gemini" if mode == "hosted" else "mistral")).strip().lower()
    if provider not in ("gemini", "mistral"):
        raise ValueError(f"unknown provider {provider!r} (from the argument or {PROVIDER_ENV}); choose gemini or mistral")
    if mode != "hosted" and provider != "mistral":
        raise ValueError(f"backend {mode!r} runs a vLLM fleet of open-weight Mistral models: use provider='mistral' "
                         f"(or unset {PROVIDER_ENV})")
    return mode, provider


def make_setup(mode: str | None = None, *, provider: str | None = None, pool_tpm: int | None = 2_000_000,
               pool_rps: int | None = None, max_inflight: int | None = None, client_tpm: int | None = None,
               dwell_s: float = 15.0, naive: bool = False, replicas: int = 2, open_model: str = "ministral-14b",
               gpu: str = "h100", max_queued: int | None = None, target_tpot_s: float = 0.02, spill_at: float = 0.9,
               seed: int = 1) -> Setup:
    """``mode``/``provider``: see ``resolve_backend``. ``pool_tpm``/``pool_rps`` describe the API's limit (None =
    unlimited); ``client_tpm`` a client-side smoothing bucket; ``replicas``/``max_queued`` the fleet.

    ``max_inflight`` defaults to 85 on the Gemini pool (the anchor scenario's cap), else to the value the capacity
    model derives (from the TPM limit, or from the fleet at the target TPOT). ``naive=True`` switches every
    protection off (no degrade levels, no cap, no breaker): retries only.
    """
    mode, provider = resolve_backend(mode, provider)
    server = None
    if mode == "hosted" and provider == "gemini":
        pool = SharedPool(pool_tpm, pool_rps) if pool_tpm else None
        standard = fake_model("gemini", pool=pool, rng=random.Random(seed))
        lite = fake_model("gemini", "lite", pool=SharedPool(pool_tpm * 2, pool_rps) if pool_tpm else None, rng=random.Random(seed + 1))
        default_cap = 85
    elif mode == "hosted":
        hosted = HostedBackend(SharedPool(pool_tpm, pool_rps) if pool_tpm else None)
        standard = fake_model("mistral", backend=hosted, rng=random.Random(seed))
        lite = fake_model("mistral", "lite", backend=HostedBackend(SharedPool(pool_tpm * 2, pool_rps) if pool_tpm else None, 0.35, 250),
                          rng=random.Random(seed + 1))
        tokens_per_turn = 2.2 * CONTEXT_TOKENS
        default_cap = int((pool_tpm / 60) / (tokens_per_turn / TURN_SECONDS)) if pool_tpm else 10**9
    else:
        hosted = HostedBackend(SharedPool(pool_tpm, pool_rps) if pool_tpm else None)
        r = replica(open_model, gpu)
        server = ServerPool(r, replicas, CONTEXT_TOKENS, PREFIX_TOKENS, max_queued=max_queued, target_tpot_s=target_tpot_s)
        backend = HybridBackend(server, hosted, spill_at) if mode == "hybrid" else server
        standard = fake_model("mistral", backend=backend, rng=random.Random(seed))
        lite = fake_model("mistral", backend=backend, answer_tokens=150, rng=random.Random(seed + 1))   # on a fleet the lever is shorter answers
        batch = server.target_batch
        call_s = r.call_seconds(batch, CONTEXT_TOKENS - PREFIX_TOKENS, CONTEXT_TOKENS, 200)
        default_cap = int(replicas * batch * (2.2 * call_s + 1.0) / (2.2 * call_s))
        if mode == "hybrid" and pool_tpm:   # the API adds its own share of capacity
            default_cap += int((pool_tpm / 60) / (2.2 * CONTEXT_TOKENS / TURN_SECONDS))
    bucket = TokenBucket(client_tpm / 60, client_tpm / 60 * 6) if client_tpm else None
    if naive:
        cfg = AdmissionConfig(max_inflight=10**9, rate_limited_degrade=10.0, queue_age_degrade_s=1e9, queue_age_shed_s=1e9,
                              saturation_degrade=1e9)
        breakers = {k: CircuitBreaker(threshold=10**9, min_calls=10**9) for k in ("standard", "lite")}
    else:
        # The Gemini anchor scenario's controller reads three signals (in-flight, 429 ratio, queue age), as the
        # primer calibrates it; the Mistral scenario adds the backend's saturation (pool use or fleet batch).
        cfg = AdmissionConfig(max_inflight=max_inflight or default_cap, dwell_s=dwell_s,
                              saturation_degrade=0.8 if provider == "mistral" else float("inf"))
        breakers = {"standard": CircuitBreaker(), "lite": CircuitBreaker()}
    return Setup(mode, provider, standard, lite, Tools(seed), Store(), AdmissionController(cfg), bucket, breakers, server)


async def simulate(setup: Setup, *, users: int = 40, duration_s: float = 90.0, think_s: float = 5.0,
                   priority: int = 5, seed: int = 1) -> "SimResult":
    rng = random.Random(seed)
    samples: list[dict] = []
    timeline: list[dict] = []
    t_start = CLOCK.now()
    t_stop = t_start + duration_s
    backend = setup.standard.backend

    async def one_user(uid: int):
        n = 0
        while CLOCK.now() < t_stop:
            n += 1
            text = rng.choice(MESSAGES)
            submitted = CLOCK.now()
            decision = setup.admission.admit(priority, circuit_open=setup.breakers["standard"].state != "closed",
                                             saturation=backend.saturation)
            if not decision.admitted:
                samples.append({"user": uid, "outcome": "shed", "level": decision.level, "t": submitted - t_start,
                                "latency_s": None, "cost_usd": 0.0, "tokens": 0, "rate_limited": 0, "steps": 0,
                                "hosted_calls": 0, "model_calls": 0, "error": None})
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
                            "rate_limited": r.rate_limited, "steps": r.steps, "hosted_calls": r.hosted_calls,
                            "model_calls": r.model_attempts - r.rate_limited, "error": r.error})
            await CLOCK.sleep(rng.expovariate(1 / think_s))

    async def sampler():
        while True:
            pool = setup.standard.backend.pool if isinstance(setup.standard.backend, HostedBackend) else None
            srv = setup.server
            timeline.append({"t": CLOCK.now() - t_start, "inflight": setup.admission.inflight, "level": setup.admission.level,
                             "pool_utilisation": pool.utilisation if pool else 0.0,
                             "rate_limited_ratio": setup.admission.rate_limited_ratio(),
                             "breaker": setup.breakers["standard"].state,
                             "running": srv.running if srv else 0, "waiting": srv.waiting if srv else 0,
                             "kv_usage": srv.kv_usage if srv else 0.0, "batch": srv.batch_per_replica() if srv else 0})
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
        srv, backend = self.setup.server, self.setup.standard.backend
        pool = backend.pool if isinstance(backend, HostedBackend) else backend.hosted.pool if isinstance(backend, HybridBackend) else None
        calls = int(df.model_calls.sum())
        return {
            "turns": len(df), "completed": len(done), "failed": int((df.outcome == "failed").sum()),
            "shed": int((df.outcome == "shed").sum()), "shed_rate": float((df.outcome == "shed").mean()) if len(df) else 0.0,
            "throughput_turns_per_s": len(done) / self.duration_s,
            "p50_s": float(np.nanpercentile(lat, 50)), "p95_s": float(np.nanpercentile(lat, 95)),
            "rate_limited_calls": int(df.rate_limited.sum()),    # 429s from the API, 503s from a capped queue …
            "pushback_calls": int(df.rate_limited.sum()),        # … the same count, under the name the fleet notebooks use
            "max_level": int(df.level.max()) if len(df) else 0,
            "degraded_share": float((df.level >= 1).mean()) if len(df) else 0.0,
            "breaker_trips": self.setup.breakers["standard"].trips,
            "cost_per_turn_usd": float(done.cost_usd.mean()) if len(done) else float("nan"),
            "api_share_of_calls": float(df.hosted_calls.sum() / calls) if calls else 0.0,
            "max_queue": int(self.timeline.waiting.max()) if len(self.timeline) else 0,
            "max_batch": int(self.timeline.batch.max()) if len(self.timeline) else 0,
            "pool_rejected": pool.rejected if pool else 0,
            "server_rejected": srv.rejected if srv else 0,
        }


def compare(results: dict[str, SimResult]) -> pd.DataFrame:
    return pd.DataFrame({name: r.summary() for name, r in results.items()}).T
