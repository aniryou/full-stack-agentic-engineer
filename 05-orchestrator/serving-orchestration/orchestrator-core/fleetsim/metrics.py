"""Metrics: what "good" means for a fleet, computed the way the serving world reports it.

    TTFT  = first token - arrival            (queueing + prefill; what a user waits for)
    ITL   = gap between consecutive tokens   (every gap, so a prefill stall shows up in p99)
    TPOT  = (done - first) / (output - 1)    (per-request mean; vLLM request_time_per_output_token)
    goodput = requests/s meeting BOTH the TTFT and the TPOT SLO (DistServe's definition)
    hit rate = prompt tokens served from the prefix cache / prompt tokens (vllm prefix_cache_hits / queries)
    imbalance = max / mean tokens computed per replica (1.0 = perfectly even)

Percentiles interpolate linearly between closest ranks (numpy's default). Every table is labelled simulated.
"""
from __future__ import annotations

import math


def percentile(xs, p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def imbalance(values) -> float:
    values = [v for v in values]
    mean = sum(values) / len(values) if values else 0
    return max(values) / mean if mean else float("nan")


def summarize(res, ttft_slo: float = 2.0, tpot_slo: float = 0.15) -> dict:
    """The one-line scorecard of a run (simulated)."""
    done = [r for r in res.requests if r.t_done is not None]
    ttft = [r.t_first - r.arrival for r in done]
    tpot = [r.tpot for r in done if r.output > 1]
    good = sum(1 for r in done if r.t_first - r.arrival <= ttft_slo and (r.output <= 1 or r.tpot <= tpot_slo))
    served = [r for r in res.replicas if r.stats["tokens"]]
    prompt = sum(r.prompt for r in done)
    return {
        "requests": len(done),
        "ttft_p50": percentile(ttft, 50), "ttft_p95": percentile(ttft, 95), "ttft_p99": percentile(ttft, 99),
        "itl_p50": percentile(res.itl, 50), "itl_p99": percentile(res.itl, 99),
        "tpot_p50": percentile(tpot, 50), "tpot_p95": percentile(tpot, 95),
        "e2e_p50": percentile([r.t_done - r.arrival for r in done], 50),
        "goodput_rps": good / res.duration if res.duration else 0.0,
        "slo_attainment": good / len(done) if done else 0.0,
        "hit_rate": sum(r.cached for r in done) / prompt if prompt else 0.0,
        "imbalance": imbalance([r.stats["tokens"] for r in served]) if served else float("nan"),
        "preemptions": sum(r.stats["preempted"] for r in res.replicas),
        "gpu_hours": sum(r.died - r.born for r in res.replicas) / 3600,
        "out_tok_s": sum(r.output for r in done) / res.duration if res.duration else 0.0,
    }


def table(rows, cols=None, title="simulated") -> str:
    """Format a list of dicts as a fixed-width text table (floats to 3 significant figures)."""
    cols = cols or list(rows[0])
    fmt = lambda v: (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}") if isinstance(v, float) else str(v)  # noqa: E731
    cells = [[fmt(r.get(c, "")) for c in cols] for r in rows]
    w = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    line = lambda vals: "  ".join(v.rjust(w[i]) for i, v in enumerate(vals))     # noqa: E731
    return "\n".join([f"[{title}]", line(cols), line(["-" * x for x in w])] + [line(row) for row in cells])


def sparkline(values, lo=None, hi=None) -> str:
    """A one-line chart for a notebook without matplotlib: ▁▂▃▄▅▆▇█."""
    bars = "▁▂▃▄▅▆▇█"
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    span = (hi - lo) or 1
    return "".join(bars[min(7, max(0, int((v - lo) / span * 7.999)))] for v in values)
