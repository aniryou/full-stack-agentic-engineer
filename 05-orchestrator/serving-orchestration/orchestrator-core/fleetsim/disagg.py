"""Prefill/decode disaggregation: run the two phases on separate pools and pay to move the KV between them.

    kv bytes     = prompt tokens x KV bytes per token
    transfer_s   = link latency + bytes x 8 / (link Gb/s x 1e9)          (x (1 - overlap) if streamed per layer)
    decode step  = overhead + max(batch / compute speed, weight read + batch x context x KV read)
    P:D ratio    = (prefill tokens/s demanded / prefill tokens/s per replica)
                 : (output tokens/s demanded / decode tokens/s per replica at the ITL SLO)

It pays when prefill chunks stall decodes (long prompts, tight ITL SLO) and the link is fast. It hurts when prompts
are short (transfer + an extra hop for no interference removed), when the link is slow (the transfer rivals the
prefill), and when a fixed P:D split does not match the traffic's input/output ratio (one pool idles, the other
queues). `search_pd` runs the simulator over every split of a fixed GPU budget.
"""
from __future__ import annotations

import math

from .replica import EngineProfile, step_time


def kv_bytes(tokens: int, kv_bytes_per_token: int) -> int:
    return tokens * kv_bytes_per_token


def transfer_s(tokens, kv_bytes_per_token, link_gbps, latency_s=0.0, overlap=0.0) -> float:
    """Time to move a prompt's KV over a link of `link_gbps` gigabits/s; `overlap` = share hidden behind prefill."""
    return latency_s + kv_bytes(tokens, kv_bytes_per_token) * 8 / (link_gbps * 1e9) * (1 - overlap)


def decode_step_s(p: EngineProfile, batch: int, ctx: int) -> float:
    """A pure decode step for `batch` requests with `ctx` tokens of context each."""
    return step_time(p, batch, batch * (ctx + 1))


def max_decode_batch(p: EngineProfile, ctx: int, itl_slo_s: float) -> int:
    """Largest decode batch that meets the ITL SLO and whose KV fits in the pool (0 if even 1 misses)."""
    fits = min(p.max_seqs, p.kv_blocks // math.ceil((ctx + 1) / p.block))
    b = 0
    while b < fits and decode_step_s(p, b + 1, ctx) <= itl_slo_s:
        b += 1
    return b


def pd_plan(p: EngineProfile, rate, isl, osl, itl_slo_s, prefill_util=0.7, hit_rate=0.0) -> dict:
    """Analytic P:D sizing for `rate` req/s of `isl` input / `osl` output tokens."""
    prefill_tok_s = rate * isl * (1 - hit_rate)
    n_prefill = prefill_tok_s / (p.compute_tok_s * prefill_util)
    ctx = isl + osl // 2
    b = max_decode_batch(p, ctx, itl_slo_s)
    per_decode = b / decode_step_s(p, b, ctx) if b else 0.0      # output tokens/s one decode replica sustains
    n_decode = rate * osl / per_decode if per_decode else math.inf
    return {"prefill_replicas": n_prefill, "decode_replicas": n_decode, "ratio_p_to_d": n_prefill / n_decode,
            "decode_batch": b, "decode_tok_s_per_replica": per_decode, "prefill_tok_s_demand": prefill_tok_s}


def run_pd(p: EngineProfile, requests, n_prefill, n_decode, *, router=None, link_gbps=100.0,
           link_latency_s=0.001, pd_threshold=0):
    """Simulate xPyD (n_prefill + n_decode replicas; n_prefill = 0 means aggregated serving)."""
    from .sim import Fleet
    fleet = Fleet(p, n_prefill + n_decode, router, prefill=n_prefill, link_gbps=link_gbps,
                  link_latency_s=link_latency_s, pd_threshold=pd_threshold)
    return fleet.run(requests)


def search_pd(p: EngineProfile, make_requests, total, *, ttft_slo, tpot_slo, make_router=None, **kw) -> list[dict]:
    """Every split of `total` replicas (0 prefill = aggregated) on the same traffic; sorted by goodput."""
    rows = []
    for n_p in range(0, total):
        res = run_pd(p, make_requests(), n_p, total - n_p, router=make_router() if make_router else None, **kw)
        s = res.summary(ttft_slo=ttft_slo, tpot_slo=tpot_slo)
        rows.append({"split": f"{n_p}P{total - n_p}D" if n_p else f"{total} aggregated",
                     "goodput_rps": s["goodput_rps"], "slo_attainment": s["slo_attainment"],
                     "ttft_p95": s["ttft_p95"], "itl_p99": s["itl_p99"], "tpot_p95": s["tpot_p95"]})
    return sorted(rows, key=lambda r: -r["goodput_rps"])
