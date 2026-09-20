"""The arithmetic of agent scale.

Everything in the primer's Part 3 comes from these few functions. The method matters more
than the numbers: conversations → turns → model calls → tokens per minute; then Little's
law for concurrency; then Provisioned Throughput units and dollars; then "what breaks first".

Prices and throughput figures are for Gemini on the global endpoint, checked 5 Sep 2026.
Verify them before quoting: model ids and prices move every few weeks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- price table (USD per 1M tokens) and Provisioned Throughput data ------------------------
# PT: one generative-scale unit (GSU) delivers `gsu_tokens_per_s` burndown tokens per second, where
# burndown = uncached input × 1 + cached input × 0.1 + output × `output_burn`.
PRICES = {
    #  model                      input   output  cached  gsu tok/s  output burn
    "gemini-3.5-flash":        (1.50,   9.00,   0.15,   675,       6),
    "gemini-3.5-flash-lite":   (0.30,   2.50,   0.03,   3360,      9),
    "gemini-3.1-flash-lite":   (0.25,   1.50,   0.025,  4030,      6),
    "gemini-3.8-flash":        (0.75,   3.75,   0.075,  675,       5),   # introductory price to 31 Dec 2026
    "gemini-3.1-pro-preview":  (2.00,  12.00,   0.20,   500,       6),   # 2× above 200k context
}
PT_USD_PER_GSU_HOUR = {"1w": 7.14, "1m": 3.6986, "3m": 3.2877, "1y": 2.7397}
HOURS_PER_MONTH = 730


def cost_per_call(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """Dollars for one model call. Cached input is billed at ~10 % of input."""
    inp, out, cached, *_ = PRICES[model]
    uncached = input_tokens - cached_tokens
    return (uncached * inp + cached_tokens * cached + output_tokens * out) / 1e6


def burndown_tokens(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """How much of a GSU's throughput one call consumes."""
    *_, output_burn = PRICES[model]
    return (input_tokens - cached_tokens) + 0.1 * cached_tokens + output_burn * output_tokens


def gsus_needed(model: str, calls_per_s: float, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    gsu_tokens_per_s = PRICES[model][3]
    return calls_per_s * burndown_tokens(model, input_tokens, output_tokens, cached_tokens) / gsu_tokens_per_s


def pt_usd_per_million_burndown(model: str, term: str) -> float:
    """What a fully-used GSU costs per million burndown tokens — compare with pay-as-you-go."""
    return PT_USD_PER_GSU_HOUR[term] / (PRICES[model][3] * 3600) * 1e6


# --- the scenario --------------------------------------------------------------------------


@dataclass
class Scenario:
    conversations_per_day: int = 100_000
    turns_per_conversation: float = 6
    calls_per_turn: float = 2.2          # plan, answer, sometimes a third
    input_tokens: int = 5_000            # per model call
    cached_tokens: int = 2_700           # of which cached (3,000-token prefix at a 90 % hit rate)
    output_tokens: int = 350             # including thinking
    peak_factor: float = 3.0             # peak hour vs daily average
    incident_factor: float = 10.0        # an outage sends everyone to the chat
    turn_seconds: float = 6.0            # p50 turn duration
    think_seconds: float = 60.0          # user pause between turns
    model: str = "gemini-3.5-flash"
    lite_model: str = "gemini-3.5-flash-lite"
    lite_share: float = 0.35             # share of calls routed to the lite tier
    tpm_baseline: int = 10_000_000       # the org's Flash tier-3 baseline
    turns_per_orchestrator_instance: int = 80


def plan(s: Scenario = Scenario()) -> dict:
    """The capacity plan as a dict of dicts: rates, tokens, concurrency, cost, pt, breaks_first."""
    out: dict = {"rates": {}, "tokens": {}, "concurrency": {}, "instances": {}}
    for level, f in {"average": 1.0, "peak": s.peak_factor, "incident": s.incident_factor}.items():
        conv_s = s.conversations_per_day / 86_400 * f
        turns_s = conv_s * s.turns_per_conversation
        calls_s = turns_s * s.calls_per_turn
        out["rates"][level] = {"conversations_per_s": conv_s, "turns_per_s": turns_s, "calls_per_s": calls_s}
        out["tokens"][level] = {
            "input_tpm": calls_s * 60 * s.input_tokens,
            "uncached_input_tpm": calls_s * 60 * (s.input_tokens - s.cached_tokens),
            "output_tpm": calls_s * 60 * s.output_tokens,
            "vs_baseline": calls_s * 60 * s.input_tokens / s.tpm_baseline,
        }
        # Little's law: things in the system = arrival rate × time each spends in it.
        out["concurrency"][level] = {
            "inflight_turns": turns_s * s.turn_seconds,
            "concurrent_sessions": conv_s * s.turns_per_conversation * (s.turn_seconds + s.think_seconds),
        }
        out["instances"][level] = math.ceil(turns_s * s.turn_seconds * 1.4 / s.turns_per_orchestrator_instance)

    # The token budget also caps concurrency: how many turns can be generating tokens at once?
    tokens_per_turn = s.calls_per_turn * (s.input_tokens + s.output_tokens)
    out["concurrency"]["max_inflight_for_baseline"] = (s.tpm_baseline / 60) / (tokens_per_turn / s.turn_seconds)
    out["concurrency"]["max_turns_per_s_for_baseline"] = (s.tpm_baseline / 60) / tokens_per_turn

    # Cost per conversation under four configurations.
    calls = s.turns_per_conversation * s.calls_per_turn
    std_nocache = cost_per_call(s.model, s.input_tokens, s.output_tokens, 0)
    std_cached = cost_per_call(s.model, s.input_tokens, s.output_tokens, s.cached_tokens)
    lite_cached = cost_per_call(s.lite_model, s.input_tokens, s.output_tokens, s.cached_tokens)
    routed = (1 - s.lite_share) * std_cached + s.lite_share * lite_cached
    out["cost"] = {
        "per_conversation_no_cache": calls * std_nocache,
        "per_conversation_cached": calls * std_cached,
        "per_conversation_routed_cached": calls * routed,
        "per_month_routed_cached": calls * routed * s.conversations_per_day * 30.4,
        "per_month_no_cache": calls * std_nocache * s.conversations_per_day * 30.4,
    }

    # Provisioned Throughput for the standard-model share.
    std_calls_s = {k: v["calls_per_s"] * (1 - s.lite_share) for k, v in out["rates"].items()}
    gsus = {k: gsus_needed(s.model, c, s.input_tokens, s.output_tokens, s.cached_tokens) for k, c in std_calls_s.items()}
    paygo_per_m = std_cached / burndown_tokens(s.model, s.input_tokens, s.output_tokens, s.cached_tokens) * 1e6
    out["pt"] = {
        "gsus": {k: math.ceil(v) for k, v in gsus.items()},
        "monthly_usd_for_average_gsus": {t: math.ceil(gsus["average"]) * p * HOURS_PER_MONTH for t, p in PT_USD_PER_GSU_HOUR.items()},
        "usd_per_million_burndown": {"paygo": paygo_per_m, **{t: pt_usd_per_million_burndown(s.model, t) for t in PT_USD_PER_GSU_HOUR}},
        "breakeven_utilisation": {t: pt_usd_per_million_burndown(s.model, t) / paygo_per_m for t in PT_USD_PER_GSU_HOUR},
    }

    # What breaks first: demand vs limit, sorted by ratio.
    limits = {"model TPM baseline": s.tpm_baseline, "billing system QPS (40)": 40, "CRM QPS (200)": 200}
    checks = []
    for level in ("peak", "incident"):
        turns_s = out["rates"][level]["turns_per_s"]
        demand = {"model TPM baseline": out["tokens"][level]["input_tpm"], "billing system QPS (40)": turns_s * 0.25,
                  "CRM QPS (200)": turns_s * 0.7}
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
        ("Input TPM vs baseline (×)", [p["tokens"][l]["vs_baseline"] for l in L]),
        ("In-flight turns (Little's law)", [p["concurrency"][l]["inflight_turns"] for l in L]),
        ("Concurrent sessions", [p["concurrency"][l]["concurrent_sessions"] for l in L]),
        ("Orchestrator instances", [p["instances"][l] for l in L]),
    ]
    lines = [f"# Capacity plan — {s.conversations_per_day:,} conversations/day", "",
             f"{s.turns_per_conversation:g} turns/conversation, {s.calls_per_turn:g} calls/turn, {s.input_tokens:,} in "
             f"({s.cached_tokens:,} cached) / {s.output_tokens} out tokens per call, peak ×{s.peak_factor:g}, "
             f"incident ×{s.incident_factor:g}, turn {s.turn_seconds:g} s, think {s.think_seconds:g} s, "
             f"{s.model} with {s.lite_share:.0%} of calls on {s.lite_model}, baseline {s.tpm_baseline / 1e6:.0f} M TPM.", "",
             "| Quantity | Average | Peak | Incident |", "|---|---:|---:|---:|"]
    lines += [f"| {name} | " + " | ".join(_fmt(v) for v in vals) + " |" for name, vals in rows]
    c, pt = p["cost"], p["pt"]
    lines += ["", f"The {s.tpm_baseline / 1e6:.0f} M TPM baseline sustains about {p['concurrency']['max_turns_per_s_for_baseline']:.1f} turns/s, "
              f"i.e. {p['concurrency']['max_inflight_for_baseline']:.0f} turns in flight — the starting value for the in-flight cap.", "",
              "## Cost per conversation", "",
              f"- all {s.model}, no caching: ${c['per_conversation_no_cache']:.4f}",
              f"- all {s.model}, cached prefix: ${c['per_conversation_cached']:.4f}",
              f"- routed + cached: **${c['per_conversation_routed_cached']:.4f}** → ${c['per_month_routed_cached']:,.0f}/month "
              f"(vs ${c['per_month_no_cache']:,.0f} unoptimised)", "",
              "## Provisioned Throughput (standard-model share)", "",
              f"- GSUs: average {pt['gsus']['average']}, peak {pt['gsus']['peak']}, incident {pt['gsus']['incident']}",
              f"- monthly cost of the average GSUs: 1-year ${pt['monthly_usd_for_average_gsus']['1y']:,.0f}, "
              f"1-month ${pt['monthly_usd_for_average_gsus']['1m']:,.0f}",
              "- $/M burndown tokens: " + ", ".join(f"{k} {v:.2f}" for k, v in pt["usd_per_million_burndown"].items()),
              "- break-even utilisation: " + ", ".join(f"{k} {v:.0%}" for k, v in pt["breakeven_utilisation"].items()), "",
              "## What breaks first", "", "| Level | Resource | Demand | Limit | Ratio |", "|---|---|---:|---:|---:|"]
    lines += [f"| {b['level']} | {b['resource']} | {_fmt(b['demand'])} | {_fmt(b['limit'])} | {b['ratio']:.2f}× |" for b in p["breaks_first"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(plan_markdown())
