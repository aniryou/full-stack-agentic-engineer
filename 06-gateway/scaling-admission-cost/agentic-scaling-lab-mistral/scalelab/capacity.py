"""The arithmetic of agent scale, for Mistral models — hosted and self-hosted.

Everything in the primer's Part 3 comes from these functions. The method matters more than the
numbers: conversations → turns → model calls → tokens per minute; Little's law for concurrency;
then two ways to pay for the tokens — Mistral's API (a rate limit and a price per token) or your
own GPUs (a fleet sized for peak, paid for whether busy or not) — and the GPU price at which one
becomes cheaper than the other; then "what breaks first".

Prices are Mistral Studio list prices (USD per 1M tokens) and GPU prices checked 19 Sep 2026.
Verify before quoting: model ids, prices and tier limits move every few weeks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .serving import GPU_USD_PER_HOUR, HOURS_PER_MONTH, OPEN_MODELS, Replica, replica, usd_per_hour

# --- Mistral's hosted API ----------------------------------------------------------------------
# model id -> (input, output, cached input) per 1M tokens. Cached input is 10 % of input (64-token blocks,
# `prompt_cache_key`). Batch is half price; EU/US regional endpoints +10 %; Priority Tier +75 %.
PRICES = {
    "mistral-small-2603":  (0.15, 0.60, 0.015),    # Mistral Small 4, Apache 2.0, MoE 119B / 6.5B active
    "mistral-large-2512":  (0.50, 1.50, 0.05),     # Mistral Large 3, Apache 2.0, MoE 675B / 41B active
    "mistral-medium-3-5":  (1.50, 7.50, 0.15),     # Mistral Medium 3.5, 128B dense, Modified MIT
    "ministral-14b-2512":  (0.20, 0.20, 0.02),
    "ministral-8b-2512":   (0.15, 0.15, 0.015),
    "ministral-3b-2512":   (0.10, 0.10, 0.01),
}
TIER_MULTIPLIER = {"global": 1.0, "regional": 1.10, "priority": 1.75, "batch": 0.5}
FREE_TIER = {"rps": 1, "tpm": 500_000, "tokens_per_month": 1_000_000_000}   # paid tiers: console only


def cost_per_call(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0, tier: str = "global") -> float:
    """Dollars for one model call on Mistral's API."""
    inp, out, cached = PRICES[model]
    uncached = input_tokens - cached_tokens
    return (uncached * inp + cached_tokens * cached + output_tokens * out) / 1e6 * TIER_MULTIPLIER[tier]


# --- the scenario --------------------------------------------------------------------------------


@dataclass
class Scenario:
    conversations_per_day: int = 100_000
    turns_per_conversation: float = 6
    calls_per_turn: float = 2.2          # plan, answer, sometimes a third
    input_tokens: int = 5_000            # per model call
    cached_tokens: int = 2_700           # of which cached (3,000-token prefix at a 90 % hit rate)
    output_tokens: int = 200             # reasoning off: ~60-token tool calls, ~300-token answers
    peak_factor: float = 3.0             # peak hour vs daily average
    incident_factor: float = 10.0        # an outage sends everyone to the chat
    turn_seconds: float = 6.0            # p50 turn duration on the hosted API
    think_seconds: float = 60.0          # user pause between turns
    tool_seconds_per_turn: float = 1.0   # CRM / billing time per turn (used for the self-hosted turn)
    # hosted
    model: str = "mistral-small-2603"
    premium_model: str = "mistral-medium-3-5"
    premium_share: float = 0.10          # share of calls routed to the premium model
    tpm_limit: int = 20_000_000          # the per-model TPM limit negotiated with Mistral (console-only; an assumption)
    rps_limit: int = 60
    headroom: float = 1.3
    # self-hosted
    open_model: str = "ministral-14b"
    gpu: str = "h100"
    gpu_price: str = "aws on-demand"
    target_tpot_s: float = 0.02          # 20 ms per token ≈ 50 tokens/s per stream
    min_replicas: int = 2                # never one: a rollout or a node loss must not take the fleet down
    turns_per_orchestrator_pod: int = 80


def hosted_costs(s: Scenario) -> dict:
    """Cost per conversation under the configurations worth comparing."""
    calls = s.turns_per_conversation * s.calls_per_turn
    c = lambda m, cached=True, tier="global": cost_per_call(m, s.input_tokens, s.output_tokens, s.cached_tokens if cached else 0, tier)  # noqa: E731
    mix = (1 - s.premium_share) * c(s.model) + s.premium_share * c(s.premium_model)
    return {
        f"{s.model} no cache": calls * c(s.model, cached=False),
        f"{s.model} cached": calls * c(s.model),
        f"planning mix ({1 - s.premium_share:.0%} {s.model} + {s.premium_share:.0%} {s.premium_model}) cached": calls * mix,
        "planning mix, priority tier": calls * mix * TIER_MULTIPLIER["priority"],
        "planning mix, regional endpoint": calls * mix * TIER_MULTIPLIER["regional"],
        "mistral-large-2512 cached": calls * c("mistral-large-2512"),
        "mistral-medium-3-5 cached": calls * c("mistral-medium-3-5"),
        "ministral-14b-2512 cached": calls * c("ministral-14b-2512"),
    }


def fleet(s: Scenario, model_key: str | None = None, gpu_key: str | None = None, price_key: str | None = None,
          target_tpot_s: float | None = None) -> dict:
    """Size a self-hosted vLLM fleet for the scenario and price it."""
    model_key, gpu_key = model_key or s.open_model, gpu_key or s.gpu
    price_key, target = price_key or s.gpu_price, target_tpot_s or s.target_tpot_s
    r: Replica = replica(model_key, gpu_key)
    context = s.input_tokens + s.output_tokens
    batch = min(r.batch_for_tpot(target, context), r.max_seqs(context, s.cached_tokens))
    if batch == 0:
        raise ValueError(f"{r.model.name} on {r.gpu.name} cannot reach {target * 1e3:.0f} ms TPOT even at batch 1")
    tok_s = r.tokens_per_s(batch, context)
    call_s = r.call_seconds(batch, s.input_tokens - s.cached_tokens, context, s.output_tokens)
    turn_s = s.calls_per_turn * call_s + s.tool_seconds_per_turn
    price = usd_per_hour(gpu_key, price_key)
    out = {"model": r.model.name, "gpu": f"{r.tp} × {r.gpu.name}", "price_key": price_key, "usd_per_gpu_hour": price,
           "batch": batch, "tpot_ms": r.tpot(batch, context) * 1e3, "ttft_ms": r.ttft(s.input_tokens - s.cached_tokens) * 1e3,
           "tok_s_per_replica": tok_s, "call_seconds": call_s, "turn_seconds": turn_s,
           "replicas": {}, "gpus": {}, "monthly_usd": {}}
    for level, f in {"average": 1.0, "peak": s.peak_factor, "incident": s.incident_factor}.items():
        calls_s = s.conversations_per_day / 86_400 * f * s.turns_per_conversation * s.calls_per_turn
        by_tokens = calls_s * s.output_tokens / tok_s                    # decode throughput
        by_seqs = calls_s * call_s / batch                               # Little: calls in flight ÷ batch per replica
        n = max(s.min_replicas, math.ceil(max(by_tokens, by_seqs) * s.headroom))
        out["replicas"][level], out["gpus"][level] = n, n * r.tp
        out["monthly_usd"][level] = n * r.tp * price * HOURS_PER_MONTH
    peak = out["replicas"]["peak"]
    conversations_per_month = s.conversations_per_day * 30.4
    out["max_inflight_turns_peak_fleet"] = peak * batch * turn_s / (s.calls_per_turn * call_s)
    out["utilisation_average"] = out["replicas"]["average"] / s.headroom / peak
    out["cost_per_conversation_peak_fleet"] = out["monthly_usd"]["peak"] / conversations_per_month
    out["monthly_usd_peak_by_price"] = {k: peak * r.tp * p * HOURS_PER_MONTH for k, p in GPU_USD_PER_HOUR[gpu_key].items()}
    out["cost_per_conversation_by_price"] = {k: v / conversations_per_month for k, v in out["monthly_usd_peak_by_price"].items()}
    out["gpu_hours_per_conversation"] = peak * r.tp * HOURS_PER_MONTH / conversations_per_month
    return out


def plan(s: Scenario = Scenario()) -> dict:
    """The capacity plan: rates, tokens, concurrency, hosted, self_hosted, breaks_first."""
    out: dict = {"rates": {}, "tokens": {}, "concurrency": {}, "orchestrator_pods": {}}
    per_call = s.input_tokens + s.output_tokens
    for level, f in {"average": 1.0, "peak": s.peak_factor, "incident": s.incident_factor}.items():
        conv_s = s.conversations_per_day / 86_400 * f
        turns_s = conv_s * s.turns_per_conversation
        calls_s = turns_s * s.calls_per_turn
        out["rates"][level] = {"conversations_per_s": conv_s, "turns_per_s": turns_s, "calls_per_s": calls_s}
        out["tokens"][level] = {
            "input_tpm": calls_s * 60 * s.input_tokens,
            "uncached_input_tpm": calls_s * 60 * (s.input_tokens - s.cached_tokens),
            "output_tpm": calls_s * 60 * s.output_tokens,
            "total_tpm": calls_s * 60 * per_call,
            "vs_limit": calls_s * 60 * per_call / s.tpm_limit,
        }
        # Little's law: things in the system = arrival rate × time each spends in it.
        out["concurrency"][level] = {
            "inflight_turns": turns_s * s.turn_seconds,
            "concurrent_sessions": conv_s * s.turns_per_conversation * (s.turn_seconds + s.think_seconds),
        }
        out["orchestrator_pods"][level] = math.ceil(turns_s * s.turn_seconds * 1.4 / s.turns_per_orchestrator_pod)

    # The TPM limit caps concurrency: how many turns can be generating tokens at once?
    tokens_per_turn = s.calls_per_turn * per_call
    out["concurrency"]["max_inflight_for_limit"] = (s.tpm_limit / 60) / (tokens_per_turn / s.turn_seconds)
    out["concurrency"]["max_turns_per_s_for_limit"] = (s.tpm_limit / 60) / tokens_per_turn

    # Hosted: the rate-limit request, and the cost.
    peak = out["rates"]["peak"]["calls_per_s"]
    costs = hosted_costs(s)
    mix_key = next(k for k in costs if k.startswith("planning mix ("))
    daily_tokens = out["rates"]["average"]["calls_per_s"] * 86_400 * per_call
    out["hosted"] = {
        "limit_request": {"rps": math.ceil(peak * s.headroom), "tpm": math.ceil(peak * 60 * per_call * s.headroom / 1e6) * 1_000_000,
                          "tokens_per_month": daily_tokens * 30.4},
        "free_tier_share_of_average": {"rps": FREE_TIER["rps"] / out["rates"]["average"]["calls_per_s"],
                                       "tpm": FREE_TIER["tpm"] / out["tokens"]["average"]["total_tpm"]},
        "cost_per_conversation": costs,
        "monthly_usd": {k: v * s.conversations_per_day * 30.4 for k, v in costs.items()},
        "planning_mix": mix_key,
    }

    # Self-hosted: the fleet, and where it beats the API.
    fl = fleet(s)
    hosted_month = out["hosted"]["monthly_usd"][mix_key]
    # Both bills grow with volume (the fleet in steps), so the break-even is a GPU price, not a volume: the hourly
    # price at which the peak-sized fleet costs what the API would for the same conversations.
    fl["breakeven_gpu_usd_per_hour"] = costs[mix_key] / fl["gpu_hours_per_conversation"]
    fl["vs_hosted_by_price"] = {k: v / hosted_month for k, v in fl["monthly_usd_peak_by_price"].items()}
    fl["vs_hosted_planning_mix"] = fl["monthly_usd"]["peak"] / hosted_month
    # The one place volume matters: below this many conversations/day the minimum (HA) fleet is not amortised.
    tp = fl["gpus"]["peak"] // fl["replicas"]["peak"]
    fl["floor_conversations_per_day_by_price"] = {k: s.min_replicas * tp * p * HOURS_PER_MONTH / (costs[mix_key] * 30.4)
                                                  for k, p in GPU_USD_PER_HOUR[s.gpu].items()}
    out["self_hosted"] = fl

    # What breaks first: demand vs limit, sorted by ratio.
    limits = {"hosted TPM limit": s.tpm_limit, "hosted RPS limit": s.rps_limit,
              "self-hosted fleet (sized for peak)": fl["replicas"]["peak"] * fl["tok_s_per_replica"],
              "billing system QPS (40)": 40, "CRM QPS (200)": 200}
    checks = []
    for level in ("peak", "incident"):
        r_, t_ = out["rates"][level], out["tokens"][level]
        demand = {"hosted TPM limit": t_["total_tpm"], "hosted RPS limit": r_["calls_per_s"],
                  "self-hosted fleet (sized for peak)": r_["calls_per_s"] * s.output_tokens,
                  "billing system QPS (40)": r_["turns_per_s"] * 0.25, "CRM QPS (200)": r_["turns_per_s"] * 0.7}
        for name, d in demand.items():
            checks.append({"level": level, "resource": name, "demand": d, "limit": limits[name], "ratio": d / limits[name]})
    out["breaks_first"] = sorted(checks, key=lambda c: -c["ratio"])
    return out


def _fmt(x: float) -> str:
    return f"{x / 1e6:,.2f} M" if x >= 1e6 else f"{x / 1e3:,.1f} k" if x >= 1e3 else f"{x:,.2f}" if x < 100 else f"{x:,.0f}"


def plan_markdown(s: Scenario = Scenario()) -> str:
    p = plan(s)
    L = ("average", "peak", "incident")
    rows = [
        ("Conversations / s", [p["rates"][l]["conversations_per_s"] for l in L]),
        ("Turns / s", [p["rates"][l]["turns_per_s"] for l in L]),
        ("Model calls / s", [p["rates"][l]["calls_per_s"] for l in L]),
        ("Input tokens / min", [p["tokens"][l]["input_tpm"] for l in L]),
        ("… of which uncached", [p["tokens"][l]["uncached_input_tpm"] for l in L]),
        ("Output tokens / min", [p["tokens"][l]["output_tpm"] for l in L]),
        ("Total TPM vs the hosted limit (×)", [p["tokens"][l]["vs_limit"] for l in L]),
        ("In-flight turns, hosted (Little's law)", [p["concurrency"][l]["inflight_turns"] for l in L]),
        ("Concurrent sessions", [p["concurrency"][l]["concurrent_sessions"] for l in L]),
        ("Orchestrator pods", [p["orchestrator_pods"][l] for l in L]),
    ]
    h, f = p["hosted"], p["self_hosted"]
    lines = [f"# Capacity plan — {s.conversations_per_day:,} conversations/day", "",
             f"{s.turns_per_conversation:g} turns/conversation, {s.calls_per_turn:g} calls/turn, {s.input_tokens:,} in "
             f"({s.cached_tokens:,} cached) / {s.output_tokens} out tokens per call, peak ×{s.peak_factor:g}, "
             f"incident ×{s.incident_factor:g}, hosted turn {s.turn_seconds:g} s, think {s.think_seconds:g} s. "
             f"Hosted: {s.model} with {s.premium_share:.0%} of calls on {s.premium_model}, assumed TPM limit "
             f"{s.tpm_limit / 1e6:.0f} M. Self-hosted: {f['model']} on {f['gpu']} at {f['price_key']} "
             f"(${f['usd_per_gpu_hour']:.2f}/GPU-h), target TPOT {s.target_tpot_s * 1e3:.0f} ms.", "",
             "| Quantity | Average | Peak | Incident |", "|---|---:|---:|---:|"]
    lines += [f"| {name} | " + " | ".join(_fmt(v) for v in vals) + " |" for name, vals in rows]
    lines += ["", "## Hosted (Mistral API)", "",
              f"- rate-limit request for peak with ×{s.headroom:g} headroom: **{h['limit_request']['rps']} RPS, "
              f"{h['limit_request']['tpm'] / 1e6:.0f} M TPM, {h['limit_request']['tokens_per_month'] / 1e9:.0f} B tokens/month** "
              f"(the free tier covers {h['free_tier_share_of_average']['tpm']:.0%} of the average TPM)",
              f"- the assumed {s.tpm_limit / 1e6:.0f} M TPM limit sustains {p['concurrency']['max_turns_per_s_for_limit']:.1f} turns/s, "
              f"i.e. {p['concurrency']['max_inflight_for_limit']:.0f} turns in flight — the starting value for the in-flight cap", "",
              "| Configuration | $ / conversation | $ / month |", "|---|---:|---:|"]
    lines += [f"| {k} | {v:.4f} | {h['monthly_usd'][k]:,.0f} |" for k, v in h["cost_per_conversation"].items()]
    lines += ["", f"## Self-hosted ({f['model']} on {f['gpu']}, vLLM)", "",
              f"- batch {f['batch']} per replica keeps TPOT at {f['tpot_ms']:.0f} ms → {f['tok_s_per_replica']:,.0f} tokens/s per replica; "
              f"TTFT {f['ttft_ms']:.0f} ms; a call takes {f['call_seconds']:.1f} s and a turn {f['turn_seconds']:.1f} s "
              f"(vs {s.turn_seconds:g} s hosted)",
              "- replicas (×{:g} headroom): average {}, peak {}, incident {} → GPUs {} / {} / {}".format(
                  s.headroom, f["replicas"]["average"], f["replicas"]["peak"], f["replicas"]["incident"],
                  f["gpus"]["average"], f["gpus"]["peak"], f["gpus"]["incident"]),
              f"- the peak-sized fleet costs ${f['monthly_usd']['peak']:,.0f}/month at {f['price_key']} "
              f"(${f['cost_per_conversation_peak_fleet']:.4f}/conversation; average utilisation {f['utilisation_average']:.0%}); "
              f"it can hold {f['max_inflight_turns_peak_fleet']:.0f} turns in flight — the in-flight cap in self-hosted mode",
              f"- versus the hosted planning mix (${h['monthly_usd'][h['planning_mix']]:,.0f}/month): {f['vs_hosted_planning_mix']:.2f}× — "
              f"both bills grow with volume, so the break-even is a GPU price: **${f['breakeven_gpu_usd_per_hour']:.2f} per GPU-hour** "
              f"({f['gpu_hours_per_conversation'] * 60:.2f} GPU-minutes per conversation at {f['utilisation_average']:.0%} average utilisation)", "",
              "| GPU price | $ / GPU-h | Peak fleet $ / month | $ / conversation | vs hosted | Floor: conversations / day to amortise "
              f"{s.min_replicas} replicas |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {k} | {GPU_USD_PER_HOUR[s.gpu][k]:.2f} | {v:,.0f} | {f['cost_per_conversation_by_price'][k]:.4f} | "
              f"{f['vs_hosted_by_price'][k]:.2f}× | {f['floor_conversations_per_day_by_price'][k]:,.0f} |" for k, v in f["monthly_usd_peak_by_price"].items()]
    lines += ["", "## What breaks first", "", "| Level | Resource | Demand | Limit | Ratio |", "|---|---|---:|---:|---:|"]
    lines += [f"| {b['level']} | {b['resource']} | {_fmt(b['demand'])} | {_fmt(b['limit'])} | {b['ratio']:.2f}× |" for b in p["breaks_first"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(plan_markdown())
