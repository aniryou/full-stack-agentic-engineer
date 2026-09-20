# %% [markdown]
# # 12 · Resource estimation: cost, throughput, latency
#
# A design review does not want a spreadsheet; it wants to see that you know **what drives** cost and latency,
# can put an order of magnitude on it in a minute, and know which levers move it. Every number in this
# notebook is arithmetic you can redo on a whiteboard — the library only gives the arithmetic names.
#
# **Primer sections:** 5.1–5.4 (token anchors, cost scenarios, capacity, latency budgets), with §2.5 (context caching).
#
# In this notebook you will:
# 1. reproduce the primer's cost scenarios A–D and the capacity numbers (peak TPM, concurrency) from first principles;
# 2. build a latency budget with parallel tool calls and read it as an ASCII waterfall;
# 3. apply the optimisation playbook lever by lever: **$0.80 → $0.15 per conversation and 14 s → 4 s per turn**.

# %%
from dataclasses import dataclass, replace
from typing import Callable

from agentlab.estimation import (PRICE_DISCLAIMER, PRICES, Scenario, Segment, ci_half_width, compounded_reliability,
                               human_bytes, latency_budget, littles_law, throughput_for_backlog, token_cost,
                               vector_store_bytes, waterfall_text)

# %% [markdown]
# ## 1. Prices and the cost of one call
#
# Three prices per model — fresh input, *cached* input, output — per million tokens. Output is 6× input;
# cached input is 10× cheaper than fresh. Those two ratios explain most cost decisions.

# %%
print(PRICE_DISCLAIMER)
print(f"{'model':24s}{'input':>8s}{'cached':>8s}{'output':>8s}")
for name, p in PRICES.items():
    print(f"{name:24s}{p.input:8.2f}{p.cached_input:8.2f}{p.output:8.2f}" + (f"   (>{p.long_context_threshold // 1000}k in: {p.long_context.input:.2f}/{p.long_context.cached_input:.2f}/{p.long_context.output:.2f})" if p.long_context else ""))

pro = PRICES["gemini-3.1-pro"]
print(f"\none support turn on Pro, 6,000 in / 400 out:  ${token_cost(6_000, 400, pro):.4f}")
print(f"same turn with 4,000 of the 6,000 input cached: ${token_cost(6_000, 400, pro, cached_share=4_000 / 6_000):.4f}")

# %% [markdown]
# ### Exercise 1.1 — implement `token_cost` with a cached share
#
# `my_token_cost(in_tokens, out_tokens, price, cached_share=0.0, batch=False)` in USD:
# fresh input at `price.input`, the cached share of input at `price.cached_input`, output at `price.output`,
# all per million tokens; halve the result when `batch` is true. (Ignore the long-context tier here.)

# %% exercise
def my_token_cost(in_tokens: float, out_tokens: float, price, cached_share: float = 0.0, batch: bool = False) -> float:
    ### BEGIN SOLUTION
    cached = in_tokens * cached_share
    usd = ((in_tokens - cached) * price.input + cached * price.cached_input + out_tokens * price.output) / 1e6
    return usd * (0.5 if batch else 1.0)
    ### END SOLUTION

# %% check
assert round(my_token_cost(6_000, 400, pro), 4) == 0.0168
assert round(my_token_cost(6_000, 400, pro, cached_share=4_000 / 6_000), 4) == 0.0096
assert round(my_token_cost(1_000, 100, PRICES["gemini-3.5-flash-lite"], batch=True), 6) == 0.000275
for args in ((6_000, 400, pro, 0.0, False), (6_000, 400, pro, 2 / 3, False), (1_000, 100, PRICES["gemini-3-flash"], 0.5, True)):
    assert abs(my_token_cost(*args) - token_cost(*args)) < 1e-12, args
print("✅ token_cost reproduces the per-call numbers")

# %% [markdown]
# ## 2. Primer §5.3 — scenarios A to D
#
# 50,000 conversations a day, 8 model calls each, 6,000 tokens in and 400 out per call. Same traffic, four designs.

# %%
base = dict(units_per_day=50_000, calls_per_unit=8, in_tokens=6_000, out_tokens=400)
scenarios = [
    Scenario("A · all Pro", **base),
    Scenario("B · all 3-Flash", **base, model_mix={"gemini-3-flash": 1.0}),
    Scenario("C · 70% Flash / 30% Pro", **base, model_mix={"gemini-3-flash": 0.7, "gemini-3.1-pro": 0.3}),
    Scenario("D · Pro, 4k of 6k cached", **base, cached_share=4_000 / 6_000),
]
print(f"{'scenario':28s}{'$/day':>12s}{'$/conv':>10s}{'$/year':>14s}")
for sc in scenarios:
    print(f"{sc.name:28s}{sc.daily_cost():>12,.0f}{sc.cost_per_unit():>10.4f}{sc.annual_cost():>14,.0f}")

# %% [markdown]
# Two things to say out loud: routing 70% of calls to Flash halves the bill *without touching the prompt*, and caching
# the stable 4k prefix on Pro (**$3,840**) beats the mixed fleet on quality-per-dollar — which is why prompt layout is a
# cost lever, not a style choice (Notebook 00).
#
# The full report for scenario A carries the capacity numbers as well:

# %%
print(scenarios[0].report())

# %% [markdown]
# ### Exercise 2.1 — peak input TPM and concurrency from first principles
#
# * `peak_input_tpm(sc)`: calls/day ÷ 86,400 × `peak_factor` × `in_tokens` × 60 — the number you compare with the model's quota.
# * `concurrency(sc)`: Little's law, L = λW — peak calls per second × `seconds_per_call` — the number of requests in flight,
#   which sizes worker pools and connection limits.
#
# Use only the scenario's fields; do not call the library methods.

# %% exercise
def peak_input_tpm(sc: Scenario) -> float:
    ### BEGIN SOLUTION
    peak_calls_per_s = sc.units_per_day * sc.calls_per_unit / 86_400 * sc.peak_factor
    return peak_calls_per_s * sc.in_tokens * 60
    ### END SOLUTION

def concurrency(sc: Scenario) -> float:
    ### BEGIN SOLUTION
    peak_calls_per_s = sc.units_per_day * sc.calls_per_unit / 86_400 * sc.peak_factor
    return peak_calls_per_s * sc.seconds_per_call
    ### END SOLUTION

# %% check
a = scenarios[0]
assert round(peak_input_tpm(a)) == 5_000_000, peak_input_tpm(a)
assert round(concurrency(a), 1) == 55.6, concurrency(a)
b = Scenario("probe", 10_000, 5, 2_000, 300, peak_factor=2.0, seconds_per_call=6.0)
assert abs(peak_input_tpm(b) - b.peak_input_tpm()) < 1e-6 and abs(concurrency(b) - b.concurrency()) < 1e-9
print(f"✅ peak {round(a.peak_calls_per_sec())} calls/s → {peak_input_tpm(a) / 1e6:.1f}M input TPM, {concurrency(a):.0f} calls in flight at {a.seconds_per_call:g} s/call")

# %% [markdown]
# ## 3. A document backlog: online vs batch, and the throughput it needs
#
# 20 M documents, 3 calls per document at 1,000 input tokens, 300 output tokens per document (100 per call), on Flash-Lite.
# Backlogs are the case for **batch** pricing: nobody is waiting, so pay half.

# %%
backlog = Scenario("backlog · Flash-Lite online", 20_000_000, 3, 1_000, 100, model_mix={"gemini-3.5-flash-lite": 1.0})
batched = replace(backlog, name="backlog · Flash-Lite batch", batch=True)
print(f"online: ${backlog.daily_cost():,.0f}   batch: ${batched.daily_cost():,.0f}   input tokens: {backlog.calls_per_day * backlog.in_tokens / 1e9:.0f}B")
rate = throughput_for_backlog(backlog.calls_per_day * backlog.in_tokens, days=30)
print(f"to finish in 30 days: {rate.tokens_per_s:,.0f} input tokens/s ≈ {rate.tokens_per_min / 1e6:.2f}M TPM sustained — check that against the quota before promising a date")

# %% [markdown]
# ## 4. Three small formulas that change conversations
#
# * **Little's law** sizes anything with a queue: in flight = arrival rate × time in system.
# * **Confidence half-width** tells you whether an eval delta is real: with 100 cases, ±9.8 points at 95%.
# * **Compounded reliability**: ten 99%-reliable steps make a 90%-reliable agent — the argument for fewer hops and for retries.

# %%
print(f"Little: 14 calls/s × 4 s        → {littles_law(14, 4):.0f} in flight")
for n in (100, 400, 1_000, 4_000):
    print(f"eval set n={n:<5d} 95% half-width ±{ci_half_width(n) * 100:4.1f} points")
for p, hops in ((0.99, 10), (0.999, 10), (0.99, 25)):
    print(f"p={p} over {hops:2d} hops → {compounded_reliability(p, hops):.3f}")

# %% [markdown]
# ## 5. The latency budget and its waterfall
#
# The primer's turn: plan **1.1 s**, two tool calls of **0.4 s** and **0.5 s**, an answer of **2.7 s** whose first token
# arrives 0.7 s in. Sequential tools give 4.7 s; running the two lookups in parallel gives 4.3 s and a first token at 2.3 s.

# %%
sequential = [Segment("plan", 1.1), Segment("lookup_a", 0.4), Segment("lookup_b", 0.5), Segment("answer", 2.7)]
parallel = [Segment("plan", 1.1), Segment("lookup_a", 0.4, parallel_group="tools"), Segment("lookup_b", 0.5, parallel_group="tools"), Segment("answer", 2.7)]
for label, segs in (("SEQUENTIAL", sequential), ("PARALLEL", parallel)):
    print(label)
    print(waterfall_text(segs, first_token_segment="answer", first_token_offset_s=0.7), "\n")

# %% [markdown]
# ### Exercise 5.1 — total time with parallel groups
#
# Implement `turn_total_seconds(segments)`: sequential segments add up; *adjacent* segments that share a
# `parallel_group` overlap, so the group contributes only its longest member.

# %% exercise
def turn_total_seconds(segments: list[Segment]) -> float:
    ### BEGIN SOLUTION
    total, group, group_longest = 0.0, None, 0.0
    for seg in segments:
        if seg.parallel_group is not None and seg.parallel_group == group:
            group_longest = max(group_longest, seg.seconds)
        else:
            total += group_longest
            group, group_longest = seg.parallel_group, seg.seconds
    return total + group_longest
    ### END SOLUTION

# %% check
assert round(turn_total_seconds(sequential), 2) == 4.7
assert round(turn_total_seconds(parallel), 2) == 4.3
three = [Segment("plan", 1.0), Segment("a", 2.0, "g1"), Segment("b", 0.5, "g1"), Segment("c", 0.5, "g1"), Segment("verify", 0.3), Segment("d", 1.0, "g2"), Segment("e", 1.5, "g2"), Segment("answer", 2.0)]
assert round(turn_total_seconds(three), 2) == round(latency_budget(three).total_s, 2) == 6.8
print("✅ parallel groups overlap; the turn is 4.7 s sequential, 4.3 s parallel")

# %% [markdown]
# ## 6. Sizing a vector store
#
# 5 M chunks × 768 dimensions × 4 bytes = **15.36 GB** of raw vectors; an HNSW index and metadata add roughly 50%.
# Say the raw number first, then the overhead — it shows you know where the bytes come from.

# %% [markdown]
# ### Exercise 6.1 — implement `vector_store_bytes`
#
# `my_vector_store_bytes(chunks, dims, bytes_per_dim=4, index_overhead=1.5)` → bytes including the index overhead.

# %% exercise
def my_vector_store_bytes(chunks: int, dims: int, bytes_per_dim: int = 4, index_overhead: float = 1.5) -> float:
    ### BEGIN SOLUTION
    return chunks * dims * bytes_per_dim * index_overhead
    ### END SOLUTION

# %% check
assert my_vector_store_bytes(5_000_000, 768, index_overhead=1.0) == 15_360_000_000
assert human_bytes(my_vector_store_bytes(5_000_000, 768)) == "23.04 GB"
assert my_vector_store_bytes(1_000_000, 1536, bytes_per_dim=2, index_overhead=1.2) == vector_store_bytes(1_000_000, 1536, bytes_per_dim=2, index_overhead=1.2)
print(f"✅ raw {human_bytes(my_vector_store_bytes(5_000_000, 768, index_overhead=1.0))}, with index ≈ {human_bytes(my_vector_store_bytes(5_000_000, 768))}")

# %% [markdown]
# ## 7. The playbook: $0.80 → $0.15 per conversation, 14 s → 4 s per turn
#
# The starting point is a real-looking first version: 20 model calls per conversation with a 17k-token prompt
# (system prompt, policies, tool schemas, the whole transcript) and 500-token answers on Pro; a turn is a
# plan call, three sequential lookups and the answer.

# %%
v0 = Scenario("assistant v0", units_per_day=50_000, calls_per_unit=20, in_tokens=17_000, out_tokens=500, seconds_per_call=4.0)
turn_v0 = [Segment("plan", 4.0), Segment("crm", 1.5), Segment("orders", 1.5), Segment("policy", 1.0), Segment("answer", 6.0)]
print(f"v0: ${v0.cost_per_unit():.2f} per conversation, {latency_budget(turn_v0).total_s:.1f} s per turn")

# %% [markdown]
# Each lever is a small transformation of the scenario and of the turn. They are listed here **out of order** on purpose;
# the exercise is to apply them in the right one.

# %%
def retime(segments, **seconds):
    return [replace(s, seconds=seconds.get(s.name, s.seconds)) for s in segments]

def parallelise(segments, names, group):
    return [replace(s, parallel_group=group) if s.name in names else s for s in segments]

@dataclass
class Lever:
    name: str
    risk_rank: int                    # 0 = no behaviour change … 4 = changes the answers themselves
    risk: str
    on_cost: Callable[[Scenario], Scenario]
    on_turn: Callable[[list[Segment]], list[Segment]]

LEVERS = [
    Lever("route 50% of calls to Flash", 3, "changes the model → eval before/after",
          lambda sc: replace(sc, model_mix={"gemini-3.1-pro": 0.5, "gemini-3-flash": 0.5}), lambda t: retime(t, plan=1.2, answer=2.3)),
    Lever("cap answers at 400 tokens", 4, "changes the answers → check completeness",
          lambda sc: replace(sc, out_tokens=400), lambda t: retime(t, answer=1.3)),
    Lever("cache the 6k stable prefix", 0, "none: bytes identical, only the bill changes",
          lambda sc: replace(sc, cached_share=6_000 / sc.in_tokens), lambda t: retime(t, plan=3.0, answer=5.0)),
    Lever("trim context 17k → 9k", 2, "changes the input → check quality on the golden set",
          lambda sc: replace(sc, in_tokens=9_000, cached_share=6_000 / 9_000), lambda t: retime(t, plan=2.5, answer=4.5)),
    Lever("run the three lookups in parallel", 1, "none for independent tools; verify they are",
          lambda sc: sc, lambda t: parallelise(t, {"crm", "orders", "policy"}, "lookups")),
]

# %% [markdown]
# ### Exercise 7.1 — order the levers and compute the cumulative effect
#
# Implement `apply_playbook(scenario, turn, levers)` returning a list of rows `(name, cost_per_conv, turn_seconds)`,
# starting with a `"baseline"` row, then one row per lever **applied cumulatively in increasing `risk_rank`**
# (the playbook rule: change nothing about model behaviour before you have changed everything else).
# Use `Scenario.cost_per_unit()` and `latency_budget(turn).total_s`.

# %% exercise
def apply_playbook(scenario: Scenario, turn: list[Segment], levers: list[Lever]) -> list[tuple[str, float, float]]:
    ### BEGIN SOLUTION
    rows = [("baseline", scenario.cost_per_unit(), latency_budget(turn).total_s)]
    for lever in sorted(levers, key=lambda lv: lv.risk_rank):
        scenario, turn = lever.on_cost(scenario), lever.on_turn(turn)
        rows.append((lever.name, scenario.cost_per_unit(), latency_budget(turn).total_s))
    return rows
    ### END SOLUTION

# %% check
rows = apply_playbook(v0, turn_v0, LEVERS)
assert [r[0] for r in rows] == ["baseline", "cache the 6k stable prefix", "run the three lookups in parallel", "trim context 17k → 9k",
                                "route 50% of calls to Flash", "cap answers at 400 tokens"], [r[0] for r in rows]
assert round(rows[0][1], 2) == 0.80 and round(rows[0][2], 1) == 14.0
assert round(rows[-1][1], 2) == 0.15, rows[-1]
assert round(rows[-1][2], 1) == 4.0, rows[-1]
costs, times = [r[1] for r in rows], [r[2] for r in rows]
assert all(a >= b - 1e-9 for a, b in zip(costs, costs[1:])) and all(a >= b - 1e-9 for a, b in zip(times, times[1:])), "every lever must be monotone"
print(f"{'step':36s}{'$/conv':>8s}{'Δ$':>8s}{'s/turn':>8s}{'Δs':>6s}")
for (name, cost, secs), (_, prev_cost, prev_secs) in zip(rows, [rows[0]] + rows[:-1]):
    print(f"{name:36s}{cost:8.3f}{cost - prev_cost:+8.3f}{secs:8.1f}{secs - prev_secs:+6.1f}")
print(f"\n✅ cumulative: ${rows[0][1]:.2f} → ${rows[-1][1]:.2f} per conversation ({1 - rows[-1][1] / rows[0][1]:.0%} less), "
      f"{rows[0][2]:.0f} s → {rows[-1][2]:.0f} s per turn")

# %% [markdown]
# Read the table the way you would present it: the two **zero-risk** levers (caching, parallel lookups) already
# remove 27% of cost and 32% of latency without an eval run; the model-touching levers come after, each gated
# by the golden set (see the evals notebook). The final turn waterfall:

# %%
turn_final = turn_v0
for lever in sorted(LEVERS, key=lambda lv: lv.risk_rank):
    turn_final = lever.on_turn(turn_final)
print(waterfall_text(turn_final, first_token_segment="answer", first_token_offset_s=0.5))

# %% [markdown]
# ## The one-minute version
#
# Estimate out loud, in this order, and round aggressively:
#
# 1. **Volume** → calls/day → calls/s (÷ 86,400) → peak (× 3). *"400k calls a day is 4.6/s, call it 14/s at peak."*
# 2. **Tokens per call** → peak input TPM against the quota; concurrency by Little's law. *"14 × 6k × 60 ≈ 5M TPM; 14 × 4 s ≈ 56 in flight."*
# 3. **Cost** = calls × (in × p_in + cached × p_cached + out × p_out). Say the driver: *"input tokens on Pro are 70% of this bill."*
# 4. **Levers, ordered by risk**: cache the prefix, parallelise, trim context, route by difficulty, constrain output, batch the offline work.
# 5. **Latency** as a waterfall with a first-token marker; parallel tools and streaming change what the user *feels*.
# 6. **Uncertainty**: quote the eval set's confidence interval, and remind everyone that reliability compounds per hop.
#
# Then flag it: *"these are list prices I would verify before a proposal"* — the disclaimer is part of the answer.
