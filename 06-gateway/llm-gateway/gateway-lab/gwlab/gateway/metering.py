"""Metering: one ledger row per request, priced from the provider's `usage`, and chargeback per tenant.

The one idea: the provider's `usage` is the bill. The gateway prices it with a dated price table, writes one
row per request (tenant, key, alias, the target that actually served it, tokens by kind, dollars, latency,
cache outcome, whether the counts came from the provider or from an estimate) and answers "who spent what"
from those rows — not from sampled traces (PRIMER §5 Metering, tracing and chargeback).

    cost = (prompt − cached) × input + cached × cached_input + cache_write × write + completion × output   [$ / 1M]

`prompt` includes cached tokens and `completion` includes reasoning tokens — a thinking token is billed as an
output token (00.5 PRIMER §5). The hosted rows below are the scaling lab's `scalelab.capacity.PRICES` (checked
2026-09-05) plus rows read from LiteLLM's `model_prices_and_context_window.json` at 849f303; all (verify,
2026-09-26). Self-hosted rows come from the GPU-hour price and a throughput (`cost_per_million_tokens`, the
same formula as `roofline.cost` and 01 PRIMER §8.1).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

PRICES_DATE = "2026-09-26"


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""
    input: float
    output: float
    cached: float
    cache_write: float | None = None


PRICES = {                    # (verify) — dated 2026-09-26
    "gemini-3.5-flash":      Price(1.50, 9.00, 0.15),
    "gemini-3.5-flash-lite": Price(0.30, 2.50, 0.03),
    "gpt-5.4-mini":          Price(0.75, 4.50, 0.075),
    "claude-haiku-4-5":      Price(1.00, 5.00, 0.10, 1.25),
}


def cost_usd(usage, price: Price) -> float:
    """Dollars for one request from canonical usage (prompt includes cached; completion includes reasoning)."""
    uncached = max(0, usage.prompt_tokens - usage.cached_tokens - usage.cache_write_tokens)
    write = price.cache_write if price.cache_write is not None else price.input
    return (uncached * price.input + usage.cached_tokens * price.cached + usage.cache_write_tokens * write
            + usage.completion_tokens * price.output) / 1e6


def blended_per_million(price: Price, prompt: int, cached: int, completion: int) -> float:
    """$ per 1M tokens (prompt + completion) for a request of this shape."""
    from .adapters import Usage
    return cost_usd(Usage(prompt, completion, cached), price) / (prompt + completion) * 1e6


def cost_per_million_tokens(price_per_gpu_hour: float, tokens_per_s: float, utilisation: float = 1.0,
                            n_gpus: int = 1) -> float:
    """Self-hosted $ per 1M tokens: GPU dollars per hour over tokens per hour (01 PRIMER §8.1, `roofline.cost`)."""
    return price_per_gpu_hour * n_gpus / (tokens_per_s * 3600 * utilisation) * 1e6


def self_hosted_price(gpu_hour_usd: float, tokens_per_s: float, utilisation: float = 1.0) -> Price:
    """A self-hosted model as a price row: every output token at the GPU's $/1M; prompt tokens at 0.

    An allocation choice, not a law: prefill is batched and compute-bound, so its GPU time per token is small
    next to decode's, and the decode rate at this utilisation already pays for the GPU hour. `chargeback(...,
    by="gpu_seconds")` is the alternative that weights both.
    """
    out = cost_per_million_tokens(gpu_hour_usd, tokens_per_s, utilisation)
    return Price(0.0, out, 0.0)


@dataclass
class LedgerRow:
    request_id: str
    ts: float
    tenant: str
    key_id: str
    alias: str
    target: str                  # the catalogue id that served it ("" when nothing did)
    provider: str
    status: int
    stream: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    usage_source: str = "provider"   # provider | estimate | none
    cost_usd: float = 0.0
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    cache: str = "miss"              # miss | exact | semantic | bypass
    attempts: int = 0
    error: str = ""
    finish_reason: str | None = None
    trace_id: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def gpu_seconds(prompt_tokens: int, completion_tokens: int, prefill_s_per_token: float,
                decode_s_per_token: float) -> float:
    """GPU time a request occupies, in a two-rate model: prefill tokens are cheap, decode tokens dear."""
    return prompt_tokens * prefill_s_per_token + completion_tokens * decode_s_per_token


def chargeback(rows, total_usd: float, by: str = "tokens", prefill_s_per_token: float = 0.0,
               decode_s_per_token: float = 1.0) -> dict:
    """Split `total_usd` (e.g. a shared GPU's bill) across tenants in proportion to tokens or GPU-seconds.

    rows: dicts or LedgerRows with tenant, prompt_tokens, completion_tokens.
    """
    weight: dict[str, float] = {}
    for r in rows:
        r = r if isinstance(r, dict) else r.as_dict()
        if by == "tokens":
            w = r["prompt_tokens"] + r["completion_tokens"]
        elif by == "gpu_seconds":
            w = gpu_seconds(r["prompt_tokens"], r["completion_tokens"], prefill_s_per_token, decode_s_per_token)
        else:
            raise ValueError("by must be tokens or gpu_seconds")
        weight[r["tenant"]] = weight.get(r["tenant"], 0.0) + w
    total_w = sum(weight.values())
    return {t: (total_usd * w / total_w if total_w else 0.0) for t, w in sorted(weight.items())}


def totals(rows, key: str = "tenant") -> dict:
    """Sum tokens and dollars per `key` (tenant, target, alias, key_id)."""
    out: dict = {}
    for r in rows:
        r = r if isinstance(r, dict) else r.as_dict()
        t = out.setdefault(r[key], {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
                                    "reasoning_tokens": 0, "cost_usd": 0.0, "estimated_rows": 0})
        t["requests"] += 1
        for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens", "cost_usd"):
            t[k] += r[k] or 0
        t["estimated_rows"] += r["usage_source"] == "estimate"
    return out


def reconcile(rows, provider_counters: dict) -> dict:
    """Compare the ledger with a provider's own counters, per provider.

    provider_counters: {provider: {"prompt_tokens": n, "generation_tokens": n}} — e.g. the difference of two scrapes
    of vLLM's `vllm:prompt_tokens_total` / `vllm:generation_tokens_total`. Returns the ledger's sums, the
    provider's, the gaps, and how many ledger rows carried estimated counts (cut streams).
    """
    out = {}
    for prov, counters in provider_counters.items():
        mine = [r if isinstance(r, dict) else r.as_dict() for r in rows]
        mine = [r for r in mine if r["provider"] == prov and r["cache"] in ("miss", "bypass") and r["status"] < 500]
        lp = sum(r["prompt_tokens"] for r in mine)
        lc = sum(r["completion_tokens"] for r in mine)
        out[prov] = {"ledger_prompt": lp, "provider_prompt": counters.get("prompt_tokens", 0),
                     "ledger_completion": lc, "provider_completion": counters.get("generation_tokens", 0),
                     "prompt_gap": lp - counters.get("prompt_tokens", 0),
                     "completion_gap": lc - counters.get("generation_tokens", 0),
                     "estimated_rows": sum(r["usage_source"] == "estimate" for r in mine)}
    return out
