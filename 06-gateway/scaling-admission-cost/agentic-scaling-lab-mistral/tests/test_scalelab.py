"""One test per concept. Run with `pytest -q` (about 10 seconds)."""

import asyncio
import random

import pytest

from scalelab.admission import AdmissionConfig, AdmissionController
from scalelab.capacity import Scenario, cost_per_call, fleet, plan
from scalelab.clock import CLOCK, run_in_virtual_time
from scalelab.loop import Budget, Store, run_turn
from scalelab.model import FakeModel, HostedBackend, HybridBackend, ServerOverloaded, ServerPool, SharedPool
from scalelab.resilience import CircuitBreaker, CircuitOpen, RateLimited, TokenBucket, backoff, call_with_retries
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


# --- resilience -----------------------------------------------------------------------------

async def test_token_bucket_paces_to_its_rate():
    b = TokenBucket(rate=10, capacity=10)
    t0 = CLOCK.now()
    for _ in range(30):
        await b.acquire(1)
    assert 1.7 < CLOCK.now() - t0 < 2.6  # 10 from the burst, then 20 at 10/s


def test_backoff_is_capped_and_jittered():
    rng = random.Random(1)
    assert backoff(4, jitter=False) == 4.0 and backoff(9, jitter=False) == 8.0
    assert all(0 <= backoff(a, rng=rng) <= 8.0 for a in range(1, 10) for _ in range(20))


async def test_breaker_trips_on_failure_ratio_and_recovers():
    cb = CircuitBreaker(threshold=3, min_calls=4, ratio=0.5, cooldown=2.0)
    for ok in (True, False, False, False):
        cb.record(ok)
    assert cb.state == "open"
    with pytest.raises(CircuitOpen):
        cb.before_call()
    await CLOCK.sleep(2.1)
    assert cb.state == "half_open"
    cb.record(True)
    assert cb.state == "closed"


async def test_call_with_retries_honours_deadline():
    calls = []

    async def always_429():
        calls.append(1)
        raise RateLimited(retry_after=5.0)

    with pytest.raises(RateLimited):
        await call_with_retries(always_429, deadline=CLOCK.now() + 3.0)
    assert len(calls) == 1  # a 5 s hint cannot fit in a 3 s deadline: no second attempt


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


# --- the loop -------------------------------------------------------------------------------

async def test_turn_runs_tools_then_answers():
    r = await run_turn("t1", "my bill is higher than usual", model=FakeModel(), tools=Tools(), store=Store())
    assert r.status == "completed" and r.tool_names == ["get_customer", "get_invoice"] and r.steps == 5
    assert r.cost_usd > 0 and 3 < r.latency_s < 9 and r.hosted_calls == 3


async def test_budget_ends_the_turn_gracefully():
    r = await run_turn("t2", "my bill is wrong", model=FakeModel(), tools=Tools(), store=Store(), budget=Budget(max_steps=2))
    assert r.status == "failed" and r.error == "budget:steps" and "sorry" in r.text


async def test_crash_and_resume_creates_one_ticket():
    store, tools, model = Store(), Tools(), FakeModel()

    class Crash(Exception):
        pass

    original = store.append_step

    def crash_after_ticket(turn_id, step):
        original(turn_id, step)
        if step["kind"] == "tool" and "create_ticket" in step["payload"]["tools"]:
            raise Crash()

    store.append_step = crash_after_ticket
    with pytest.raises(Crash):
        await run_turn("c1", "I want to complain to a human", model=model, tools=tools, store=store)
    store.append_step = original
    r = await run_turn("c1", "I want to complain to a human", model=model, tools=tools, store=store)  # the redelivery
    assert r.status == "completed" and r.resumed_steps == 4 and r.model_attempts == 1
    assert tools.tickets == ["T-1001"]  # exactly one side effect


async def test_degrade_level_2_withholds_writes():
    tools = Tools()
    r = await run_turn("d1", "I want to complain to a human", model=FakeModel(), tools=tools, store=Store(), degrade_level=2)
    assert "create_ticket" not in r.tool_names and tools.tickets == []


# --- admission --------------------------------------------------------------------------------

def test_admission_levels_and_shedding():
    ac = AdmissionController(AdmissionConfig(max_inflight=4, dwell_s=0))
    assert ac.admit().admitted and ac.level == 0
    for _ in range(3):
        ac.admit()
    d = ac.admit()
    assert not d.admitted and d.level == 3 and d.retry_after
    assert ac.admit(priority=1).admitted          # priority bypasses the cap
    for _ in range(10):
        ac.note_model_call(rate_limited=True)
    for _ in range(5):
        ac.release()
    assert ac.compute_level() == 2                # 67 % of calls pushed back
    ac._recent.clear()
    assert ac.compute_level(saturation=0.85) == 1 and ac.compute_level(saturation=1.2) == 2   # the fleet's signal


def test_level_hysteresis():
    ac = AdmissionController(AdmissionConfig(max_inflight=100, dwell_s=10))
    for _ in range(4):
        ac.note_model_call(True)
    assert ac.compute_level() == 2
    ac._recent.clear()
    assert ac.compute_level() == 2  # held for the dwell time


# --- simulation ---------------------------------------------------------------------------------

def test_overload_regimes_hosted_and_local():
    # Deterministic: a virtual-time event loop (sleeps cost no wall-clock, so a busy CPU cannot shift
    # the numbers) and seeded randomness, including the retry jitter's module-level `random`.
    async def regimes():
        results = {}
        for name, mode, kw in [("hosted naive", "hosted", dict(pool_tpm=3_000_000, naive=True)),
                               ("hosted capped", "hosted", dict(pool_tpm=3_000_000, max_inflight=30)),
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
