"""Metering, the ledger and chargeback (PRIMER §5).

The one idea: the provider's ``usage`` is the bill -- prompt tokens (the cached ones at their own price)
and completion tokens (reasoning included) -- so the ledger prices *usage*, not text. The gateway
estimates only where no usage exists (at admission, and for a stream cut before its usage chunk),
marks those rows estimated, and reconciles its totals against the provider's counters. A shared
self-hosted pool has no per-token price: split its bill by what each tenant made the GPUs do.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .providers import CATALOGUE


def price_call(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    """Dollars for one call from canonical usage: ``prompt_tokens`` includes the cached ones (as in
    ``scalelab.capacity.cost_per_call``); ``completion_tokens`` includes reasoning."""
    inp, out, cached = CATALOGUE[model].price
    return ((prompt_tokens - cached_tokens) * inp + cached_tokens * cached + completion_tokens * out) / 1e6


def cost_per_million(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    """Blended $ per 1M tokens (input + output) for one call shape."""
    return price_call(model, prompt_tokens, completion_tokens, cached_tokens) / (prompt_tokens + completion_tokens) * 1e6


def self_hosted_per_million(price_per_gpu_hour: float, tokens_per_s: float, utilisation: float = 1.0, n_gpus: int = 1) -> float:
    """$ per 1M generated tokens on your own GPUs -- ``roofline.cost.cost_per_million_tokens`` (01 PRIMER §8.1)."""
    return price_per_gpu_hour * n_gpus / (tokens_per_s * 3600 * utilisation) * 1e6


@dataclass
class LedgerRow:
    request_id: str
    tenant: str
    key_id: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    estimated: bool = False            # True when no provider usage covered this row (a cut stream)
    status: str = "ok"
    ttft: float | None = None
    duration: float = 0.0
    trace_id: str = ""


def row_from_usage(request_id: str, tenant: str, key_id: str, model: str, usage: dict, *, estimated: bool = False,
                   **extra) -> LedgerRow:
    m = CATALOGUE[model]
    cost = 0.0 if m.price is None else price_call(model, usage["prompt_tokens"], usage["completion_tokens"],
                                                  usage.get("cached_tokens", 0))
    return LedgerRow(request_id, tenant, key_id, model, m.provider, usage["prompt_tokens"], usage["completion_tokens"],
                     usage.get("cached_tokens", 0), usage.get("reasoning_tokens", 0), cost, estimated, **extra)


class Ledger:
    """One row per request, billing-grade (every row, not a sample -- unlike traces)."""

    def __init__(self):
        self.rows: list = []

    def add(self, row: LedgerRow) -> None:
        self.rows.append(row)

    def totals(self, by: str = "tenant") -> dict:
        out = defaultdict(lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "estimated": 0})
        for r in self.rows:
            t = out[getattr(r, by)]
            t["requests"] += 1
            t["prompt_tokens"] += r.prompt_tokens
            t["completion_tokens"] += r.completion_tokens
            t["cost"] += r.cost
            t["estimated"] += r.estimated
        return dict(out)

    def reconcile(self, provider_counts: dict, tolerance: float = 0.01) -> dict:
        """Compare our per-model token totals with the provider's (its usage export or invoice)."""
        ours, report = self.totals(by="model"), {}
        for model, theirs in provider_counts.items():
            mine = ours.get(model, {"prompt_tokens": 0, "completion_tokens": 0})
            for field in ("prompt_tokens", "completion_tokens"):
                diff = theirs[field] - mine[field]
                report[(model, field)] = {"ledger": mine[field], "provider": theirs[field], "diff": diff,
                                          "ok": abs(diff) <= tolerance * max(1, theirs[field])}
        return report


def chargeback(usage_by_tenant: dict, pool_cost: float, *, by: str = "tokens", prefill_tps: float = 68_000.0,
               decode_tps: float = 6_846.5) -> dict:
    """Split a shared pool's bill. ``by="tokens"``: by prompt + completion tokens. ``by="gpu_seconds"``: by
    the GPU time each tenant used -- prompt tokens at the prefill rate, completion tokens at the decode rate
    (defaults: Llama-3.1-8B on an H100 at 2K context from 01 PRIMER §8.1, ~68,000 prefill and 6,846.5 decode tok/s)."""
    if by == "tokens":
        w = {t: u["prompt_tokens"] + u["completion_tokens"] for t, u in usage_by_tenant.items()}
    else:
        w = {t: u["prompt_tokens"] / prefill_tps + u["completion_tokens"] / decode_tps for t, u in usage_by_tenant.items()}
    total = sum(w.values())
    return {t: pool_cost * x / total for t, x in w.items()}
