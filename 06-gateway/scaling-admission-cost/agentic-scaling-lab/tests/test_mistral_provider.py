"""The Mistral provider and the self-hosted backends: one test per concept.

These are the cases of the suite of the former Mistral copy of this lab, run against the merged package:
its capacity model is now ``scalelab.mistral``, its fake model ``fake_model("mistral")``, and its hosted
simulations pass ``provider="mistral"``. The cases that were identical to tests/test_scalelab.py (token
bucket, backoff, breaker, retries, the turn's budget, crash-and-resume, degrade level 2, hysteresis) run
there once.
"""

import asyncio
import random

import pytest

from scalelab.admission import AdmissionConfig, AdmissionController
from scalelab.clock import CLOCK, run_in_virtual_time
from scalelab.loop import Store, run_turn
from scalelab.mistral import Scenario, cost_per_call, fleet, plan
from scalelab.model import HostedBackend, HybridBackend, ServerOverloaded, ServerPool, SharedPool, fake_model
from scalelab.resilience import RateLimited
from scalelab.serving import GPUS, OPEN_MODELS, Replica, kv_bytes_per_token, kv_bytes_per_token_mla, replica
from scalelab.sim import compare, make_setup, simulate
from scalelab.tools import Tools


@pytest.fixture(autouse=True)
def fast_clock():
    CLOCK.reset(0.02)  # 50x faster than real time


# --- capacity -------------------------------------------------------------------------------

def test_capacity_plan_reproduces_the_primer_numbers():
    p = plan(Scenario())
    assert abs(p["rates"]["peak"]["calls_per_s"] - 45.8) < 0.1
    assert abs(p["tokens"]["peak"]["total_tpm"] / 1e6 - 14.3) < 0.05
    assert abs(p["concurrency"]["peak"]["inflight_turns"] - 125) < 0.5
    assert 174 < p["concurrency"]["max_inflight_for_limit"] < 176
    assert p["hosted"]["limit_request"] == {"rps": 60, "tpm": 19_000_000, "tokens_per_month": pytest.approx(2.09e11, rel=0.01)}
    mix = p["hosted"]["cost_per_conversation"][p["hosted"]["planning_mix"]]
    assert 0.0130 < mix < 0.0132
    assert 0.0066 < p["hosted"]["cost_per_conversation"]["mistral-small-2603 cached"] < 0.0068
    f = p["self_hosted"]
    assert f["batch"] == 24 and f["replicas"] == {"average": 4, "peak": 11, "incident": 34}
    assert 1.3 < f["vs_hosted_planning_mix"] < 1.5                      # on-demand H100s cost more than the API
    assert 4.9 < f["breakeven_gpu_usd_per_hour"] < 5.0                  # … the fleet wins below ≈ $4.95 per GPU-hour
    assert f["vs_hosted_by_price"]["neocloud"] < 0.8 < 1 < f["vs_hosted_by_price"]["gcp on-demand"]
    assert 12_000 < f["floor_conversations_per_day_by_price"]["neocloud"] < 15_000
    assert p["breaks_first"][0]["level"] == "incident" and "hosted" in p["breaks_first"][0]["resource"]


def test_pricing_arithmetic():
    # 2,300 uncached × $0.15 + 2,700 cached × $0.015 + 200 out × $0.60 per 1M tokens
    assert abs(cost_per_call("mistral-small-2603", 5000, 200, 2700) - (345 + 40.5 + 120) / 1e6) < 1e-12
    assert abs(cost_per_call("mistral-small-2603", 5000, 200, 2700, tier="priority") / cost_per_call("mistral-small-2603", 5000, 200, 2700) - 1.75) < 1e-9


# --- serving --------------------------------------------------------------------------------

def test_kv_cache_arithmetic():
    assert kv_bytes_per_token(40, 8, 128) == 163_840                       # Ministral 3 14B: 160 KiB per token
    assert kv_bytes_per_token_mla(36, 256, 64) == 23_040                    # Mistral Small 4 (MLA): 22.5 KiB
    assert OPEN_MODELS["mistral-large-3"].kv_bytes_per_token == 70_272
    r = replica("ministral-14b", "h100")
    assert 50 < r.kv_budget_gb < 56 and r.max_seqs(5_200) == 63 and r.max_seqs(5_200, 2_700) == 128   # 0.85 GB per 5.2k-token call
    small4 = OPEN_MODELS["mistral-small-4"]
    assert not Replica(small4, GPUS["h100"], 1).fits and replica("mistral-small-4", "h100").tp == 2   # 121 GB of FP8 weights


def test_batch_trades_latency_for_throughput():
    r = replica("ministral-14b", "h100")
    tpots = [r.tpot(b, 5_200) for b in (1, 8, 32, 64)]
    assert tpots == sorted(tpots) and 0.009 < tpots[0] < 0.012 and 0.02 < tpots[2] < 0.026
    tps = [r.tokens_per_s(b, 5_200) for b in (1, 8, 32, 64)]
    assert tps == sorted(tps) and 1_300 < tps[2] < 1_450
    assert r.batch_for_tpot(0.02, 5_200) == 24 and r.batch_for_tpot(0.005, 5_200) == 0
    s4 = replica("mistral-small-4", "h100")
    assert s4.weights_read_gb(1) < 10 < s4.weights_read_gb(128) < 121   # MoE: a big batch touches every expert


def test_fleet_sizing_scales_with_gpu_price_not_demand():
    s = Scenario()
    a, b = fleet(s, price_key="aws on-demand"), fleet(s, price_key="neocloud")
    assert a["replicas"] == b["replicas"] and a["monthly_usd"]["peak"] > 1.5 * b["monthly_usd"]["peak"]
    assert fleet(s, model_key="mistral-small-4", gpu_key="h100")["gpus"]["peak"] % 2 == 0   # TP=2 replicas


# --- the backends -----------------------------------------------------------------------------

def test_hosted_pool_answers_429_on_tokens_and_on_rps():
    pool = SharedPool(tpm=100_000, rps=5)
    for _ in range(5):
        pool.admit(5_000)
    with pytest.raises(RateLimited):
        pool.admit(5_000)                      # the 6th request in the same second
    assert pool.rejected == 1


async def test_server_pool_queues_instead_of_429_and_slows_with_batch():
    srv = ServerPool(replica("ministral-14b", "h100"), replicas=1, context_tokens=5_200, shared_prefix_tokens=3_000)
    rng = random.Random(0)
    alone, _ = await srv.serve(5_000, 3_000, 200, rng)
    t0 = CLOCK.now()
    crowd = await asyncio.gather(*(srv.serve(5_000, 3_000, 200, rng) for _ in range(48)))
    assert srv.rejected == 0 and max(l for l, _ in crowd) > 1.8 * alone      # nobody was refused; everybody was slower
    assert srv.saturation == 0 and srv.served == 49
    capped = ServerPool(replica("ministral-14b", "h100"), replicas=1, context_tokens=5_200, shared_prefix_tokens=3_000, max_queued=0)
    capped.running = capped.capacity                             # full: the next request must queue …
    with pytest.raises(ServerOverloaded):
        await capped.serve(5_000, 3_000, 200, rng)               # … and the queue cap turns that into a 503
    assert capped.rejected == 1


async def test_hybrid_spills_to_the_api_when_the_fleet_is_saturated():
    srv = ServerPool(replica("ministral-14b", "h100"), replicas=1, context_tokens=5_200, shared_prefix_tokens=3_000, target_tpot_s=0.02)
    hybrid = HybridBackend(srv, HostedBackend(SharedPool(10_000_000)), spill_at=0.9)
    rng = random.Random(0)
    where = await asyncio.gather(*(hybrid.serve(5_000, 3_000, 200, rng) for _ in range(60)))
    served = [w for _, w in where]
    assert served.count("self-hosted") >= 20 and served.count("hosted") >= 20


# --- the loop ---------------------------------------------------------------------------------

async def test_turn_runs_tools_then_answers():
    r = await run_turn("t1", "my bill is higher than usual", model=fake_model("mistral"), tools=Tools(), store=Store())
    assert r.status == "completed" and r.tool_names == ["get_customer", "get_invoice"] and r.steps == 5
    assert r.cost_usd > 0 and 3 < r.latency_s < 9 and r.hosted_calls == 3


async def test_a_fleet_call_costs_nothing_per_token():
    srv = ServerPool(replica("ministral-14b", "h100"), replicas=1, context_tokens=5_200, shared_prefix_tokens=3_000)
    r = await run_turn("t9", "my bill is higher than usual", model=fake_model("mistral", backend=srv), tools=Tools(), store=Store())
    assert r.status == "completed" and r.hosted_calls == 0 and r.cost_usd == 0.0   # the GPUs are paid for by the hour


# --- admission ---------------------------------------------------------------------------------

def test_admission_reads_the_fleets_saturation():
    ac = AdmissionController(AdmissionConfig(max_inflight=4, dwell_s=0))
    for _ in range(10):
        ac.note_model_call(rate_limited=True)
    assert ac.compute_level() == 2                # 100 % of calls pushed back
    ac._recent.clear()
    assert ac.compute_level() == 0
    assert ac.compute_level(saturation=0.85) == 1 and ac.compute_level(saturation=1.2) == 2   # the fleet's signal


# --- simulation ---------------------------------------------------------------------------------

def test_overload_regimes_hosted_and_local():
    # Deterministic: a virtual-time event loop (sleeps cost no wall-clock, so a busy CPU cannot shift
    # the numbers) and seeded randomness, including the retry jitter's module-level `random`.
    async def regimes():
        results = {}
        for name, mode, kw in [("hosted naive", "hosted", dict(provider="mistral", pool_tpm=3_000_000, naive=True)),
                               ("hosted capped", "hosted", dict(provider="mistral", pool_tpm=3_000_000, max_inflight=30)),
                               ("local naive", "local", dict(replicas=2, naive=True)),
                               ("local capped", "local", dict(replicas=2, max_inflight=30)),
                               ("hybrid", "hybrid", dict(replicas=2, pool_tpm=3_000_000))]:
            CLOCK.reset(0.02)
            results[name] = await simulate(make_setup(mode, **kw), users=100, duration_s=40, think_s=5)
        return results

    random.seed(0)
    results = run_in_virtual_time(regimes())
    s = compare(results)
    assert s.loc["hosted naive", "pushback_calls"] > 20 and s.loc["hosted naive", "p95_s"] > s.loc["hosted capped", "p95_s"]
    assert s.loc["hosted capped", "pushback_calls"] == 0 and s.loc["hosted capped", "shed"] > 0 and s.loc["hosted capped", "failed"] == 0
    assert s.loc["local naive", "pushback_calls"] == 0 and s.loc["local naive", "failed"] == 0        # no errors …
    assert s.loc["local naive", "p95_s"] > 1.4 * s.loc["local capped", "p95_s"]                        # … just latency
    assert s.loc["local naive", "max_batch"] > s.loc["local capped", "max_batch"]
    assert s.loc["hybrid", "api_share_of_calls"] > 0.2 and s.loc["hybrid", "shed_rate"] < s.loc["local capped", "shed_rate"]
