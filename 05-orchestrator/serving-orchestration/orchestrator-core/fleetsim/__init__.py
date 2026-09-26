"""fleetsim — a fleet of LLM engines, simulated: routing, autoscaling, prefill/decode disaggregation, KV tiers.

    from fleetsim import Fleet, L4_8B, agentic, PowerOfTwo

    res = Fleet(L4_8B, replicas=4, router=PowerOfTwo(seed=1)).run(agentic(0.5, 120, seed=1))
    print(res.summary())              # every number is SIMULATED — a model of an engine, not a measurement

Pure standard library and deterministic under the seeds you pass. Read the modules in this order: workload.py,
replica.py, routers.py, sim.py, metrics.py, autoscale.py, disagg.py, kvtier.py. The lab next door
(../inference-gateway-lab) puts the same decisions in front of real OpenAI-compatible servers.
"""
from .autoscale import HPA, Autoscaler, ColdStart, Policy, Rules, external_metric_replicas, pods_metric_replicas
from .disagg import decode_step_s, kv_bytes, max_decode_batch, pd_plan, run_pd, search_pd, transfer_s
from .kvtier import Tier, TieredKV, breakeven_gb_s, onload_s, recompute_s, simulate_sessions, working_set_gb
from .metrics import imbalance, percentile, sparkline, summarize, table
from .replica import H100_8B, L4_8B, LLAMA_8B_KV, BlockPool, EngineProfile, Replica, engine_profile, step_time
from .routers import (ApproxPrefixIndex, ConsistentHashBoundedLoad, HashRing, KVCacheUtilizationScorer,
                      LeastOutstanding, LoraAffinityFilter, PowerOfTwo, PrefixAffinityFilter, PrefixCacheScorer,
                      PrefixHash, PreciseIndex, QueueScorer, RoundRobin, Router, TokenLoadScorer, WeightedScorer,
                      epp, sticky_until_saturated)
from .sim import Fleet, Result
from .workload import HashChain, Request, agentic, arrivals, burst, chat, expand, mix, mix64, rag

__all__ = [n for n in dir() if not n.startswith("_")]
