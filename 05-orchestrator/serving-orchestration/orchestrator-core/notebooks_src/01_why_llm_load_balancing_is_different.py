# %% [markdown]
# # 01 · Why LLM load balancing is different
#
# **Tier:** T0 — CPU only, a few seconds, no network. Every number below is **simulated** by `fleetsim` (an engine
# model built from spec-sheet arithmetic), not measured on a GPU.
#
# ## The one-minute version
# A web load balancer assumes requests are small, similar and stateless, so spreading them evenly *by count* spreads
# the *work* evenly. LLM requests break all three assumptions:
#
# 1. **Cost varies by one to two orders of magnitude and is unknown up front.** A 6,000-token RAG prompt is five
#    times the prefill of a chat turn, and nobody knows the output length until generation stops.
# 2. **Capacity is KV-cache memory and step time, not CPU.** A replica is "full" when its KV blocks are; overload
#    shows up as queueing and preemption long before anything looks like a CPU alarm.
# 3. **Replicas are caches.** The replica that already holds a prompt's prefix serves it for a fraction of the cost —
#    that is notebook 02.
#
# After this notebook you can explain, with numbers, why round-robin and least-connections leave latency on the
# table, why power-of-two-choices is the robust default when load is all a router looks at, and why a router's own
# dispatch counters beat stale scraped metrics. Primer: `05-orchestrator/serving-orchestration/PRIMER.md` §1–2.

# %%
from fleetsim import (L4_8B, Fleet, LeastOutstanding, PowerOfTwo, RoundRobin, chat, epp, expand, mix, rag,
                      step_time, table)
import random

p = L4_8B
print(p.name)
print(f"every step streams the weights once: {p.weight_read_s * 1e3:.1f} ms | prefill speed "
      f"{p.compute_tok_s:,.0f} tok/s | KV pool {p.kv_blocks:,} blocks x {p.block} = {p.kv_blocks * p.block:,} tokens")
rows = []
for label, tokens, kv in [("decode, batch 1, 2,000-token context", 1, 2001),
                          ("decode, batch 16, 2,000-token context", 16, 16 * 2001),
                          ("decode, batch 64, 500-token context", 64, 64 * 501),
                          ("prefill chunk of 2,048 tokens", 2048, 2048)]:
    t = step_time(p, tokens, kv)
    rows.append({"one engine step": label, "ms": t * 1e3, "tokens/s": tokens / t})
print(table(rows, title="simulated step times, 8B bf16 on one L4"))

# %% [markdown]
# Read the table as a roofline. A decode step costs about the same whether it carries 1 request or 16 — the weights
# are streamed once either way — so batching is nearly free throughput. A prefill chunk is compute-bound and ten
# times longer; every request decoding in the same batch waits for it (notebook 04 is about that stall).
#
# ## Exercise 1.1 — how different are two requests?
# A request occupies a replica in three currencies: **prefill compute** (its uncached prompt tokens divided by
# `p.compute_tok_s`), **KV memory** (`ceil((prompt + output) / p.block)` blocks at the end), and **residency** (it
# stays in the batch for roughly `output` decode steps; take `step_s = 0.07`, a typical busy step here).
# Write `footprint(p, prompt, output, cached=0, step_s=0.07)` returning a dict with keys `prefill_s`, `kv_blocks`,
# `residency_s` (prefill time + output x step_s) and `block_seconds` (kv_blocks x residency_s — the memory-time
# product a replica has to give this request).

# %% exercise
import math


def footprint(p, prompt, output, cached=0, step_s=0.07):
    ### BEGIN SOLUTION
    prefill_s = (prompt - cached) / p.compute_tok_s
    kv_blocks = math.ceil((prompt + output) / p.block)
    residency_s = prefill_s + output * step_s
    return {"prefill_s": prefill_s, "kv_blocks": kv_blocks, "residency_s": residency_s,
            "block_seconds": kv_blocks * residency_s}
    ### END SOLUTION

# %% check
f_chat, f_rag = footprint(p, 1200, 150), footprint(p, 6000, 250)
assert f_chat["kv_blocks"] == 85 and f_rag["kv_blocks"] == 391
assert abs(f_chat["prefill_s"] - 1200 / 3781.25) < 1e-9 and abs(f_rag["residency_s"] - (6000 / 3781.25 + 17.5)) < 1e-9
assert abs(footprint(p, 12000, 100, cached=11000)["prefill_s"] - 1000 / 3781.25) < 1e-9
print(f"RAG vs chat: prefill {f_rag['prefill_s'] / f_chat['prefill_s']:.1f}x, "
      f"block-seconds {f_rag['block_seconds'] / f_chat['block_seconds']:.1f}x")
work = sorted(footprint(p, r.prompt, r.output)["block_seconds"]
              for r in expand(mix(chat(3.0, 180, seed=1), rag(0.8, 180, seed=2, corpus=1000, zipf=0.6))))
p1, p99 = work[len(work) // 100], work[len(work) * 99 // 100]
print(f"across a chat + RAG mix: p1 {p1:,.0f}, p50 {work[len(work) // 2]:,.0f}, p99 {p99:,.0f} block-seconds "
      f"-> the p99 request costs {p99 / p1:.0f}x the p1 request (simulated workload)")
print("✅ footprint works")

# %% [markdown]
# ## Worked example — five routers, one heterogeneous fleet
# Four L4 replicas serve a mix of chat turns and RAG requests near saturation. Every router below looks at **load
# only** (notebook 02 adds the cache). `PowerOfTwo(d=1)` is plain random choice.

# %%
def mixed():
    return mix(chat(4.0, 180, seed=1), rag(1.0, 180, seed=2, corpus=1000, zipf=0.6))


routers = {"random": lambda: PowerOfTwo(d=1, seed=1), "round-robin": lambda: RoundRobin(),
           "least-outstanding": lambda: LeastOutstanding(), "power-of-two": lambda: PowerOfTwo(seed=1),
           "least-waiting (scraped, 50 ms)": lambda: LeastOutstanding(load="waiting")}
results = {}
for name, make in routers.items():
    res = Fleet(p, 4, make()).run(mixed())
    results[name] = res.summary(ttft_slo=2.0, tpot_slo=0.15)
cols = ["ttft_p50", "ttft_p95", "ttft_p99", "tpot_p95", "imbalance", "preemptions", "slo_attainment"]
print(table([{"router": k, **{c: v[c] for c in cols}} for k, v in results.items()], ["router"] + cols,
            title=f"simulated: {len(expand(mixed()))} requests, 4x L4, chat + RAG"))

# %% [markdown]
# Random choice has the worst tail, and round-robin — perfectly fair *in request count* — is not far behind: it
# hands a replica its next request on schedule even when that replica is chewing through two RAG prompts.
# Least-outstanding and power-of-two react to what is actually in flight and cut the p95 roughly in half. The best
# row looks at the engine's own **queue**: waiting requests are the direct cause of TTFT.
#
# ## Exercise 1.2 — power of two choices
# Write `p2c(loads, rng)`: sample **two distinct** indices uniformly at random (`rng.sample`), return the one with the
# lower load (ties: the first sampled). This is the whole algorithm; it needs no global scan and no coordination.

# %% exercise
def p2c(loads, rng):
    ### BEGIN SOLUTION
    a, b = rng.sample(range(len(loads)), 2)
    return b if loads[b] < loads[a] else a
    ### END SOLUTION

# %% check
rng = random.Random(0)
picks = [p2c([0, 5, 5, 5], rng) for _ in range(20000)]
share = picks.count(0) / len(picks)
assert abs(share - 0.5) < 0.02, share           # 1 - C(3,2)/C(4,2): replica 0 wins whenever it is sampled
assert all(p2c([9, 1, 1, 1], rng) != 0 for _ in range(2000))    # the unique maximum is never chosen
print(f"✅ p2c works — the idle replica gets {share:.1%} of new requests, the busiest never gets one")

# %% [markdown]
# ## Exercise 1.3 — Little's law, the sizing identity
# In steady state, **requests in the system = arrival rate x mean time in the system** (L = λW). It holds for any
# queueing discipline and any router, which makes it the fastest sanity check on a load test. Write
# `littles_law(rate, mean_e2e_s)` and check it against the simulator's own occupancy samples.

# %% exercise
def littles_law(rate, mean_e2e_s):
    ### BEGIN SOLUTION
    return rate * mean_e2e_s
    ### END SOLUTION

# %% check
reqs = chat(2.0, 600, seed=3)
res = Fleet(p, 2, PowerOfTwo(seed=1), sample_s=5.0).run(reqs)
lam = len(reqs) / 600
w = sum(r.t_done - r.arrival for r in res.requests) / len(reqs)
seen = [row["waiting"] + row["running"] for row in res.timeline if 60 <= row["t"] <= 600]
observed = sum(seen) / len(seen)
assert abs(littles_law(lam, w) - observed) / observed < 0.15
print(f"✅ λ = {lam:.2f}/s x W = {w:.1f} s -> L = {littles_law(lam, w):.1f} in flight; the simulator saw {observed:.1f}")

# %% [markdown]
# ## Worked example — stale metrics make an argmin router herd
# An endpoint picker that scrapes `/metrics` sees a snapshot. If it sends every request to the replica with the
# lowest scraped `num_requests_running`, then every request in a burst sees the *same* minimum and lands on the same
# replica until the next scrape. Below, traffic arrives in 5-second bursts of 16 req/s; the snapshot is refreshed at
# most every `metrics_age` seconds. (llm-d's EPP refreshes every 50 ms by default; a router fed by a 15-s Prometheus
# scrape would be far staler.)

# %%
bursts = [(t, 16.0 if (t // 5) % 4 == 1 else 2.0) for t in range(0, 240, 5)]
rows = []
for age in (0.05, 2.0, 10.0):
    for name, make in [("argmin(running, scraped)", lambda: LeastOutstanding(load="running")),
                       ("p2c(running, scraped)", lambda: PowerOfTwo(load="running", seed=1)),
                       ("argmin(own counters)", lambda: LeastOutstanding())]:
        s = Fleet(p, 4, make(), metrics_age=age).run(chat(bursts, 240, seed=21, system=800, user=250)).summary(ttft_slo=1.0)
        rows.append({"metrics age s": age, "router": name, "ttft_p95": s["ttft_p95"], "ttft_p99": s["ttft_p99"],
                     "slo_attainment": s["slo_attainment"]})
print(table(rows, title="simulated: bursty chat, 4x L4"))

# %% [markdown]
# The argmin router degrades as its data ages; power-of-two on the *same* stale data barely moves, because two random
# candidates rarely both look idle (Mitzenmacher, "How useful is old information?"). The router's own counters —
# requests it dispatched and has not seen finish — are never stale, which is why load-aware routers keep them.
#
# ## Exercise 1.4 — predict the herd
# A router holds the stale snapshot `loads = [3, 1, 4, 2]` for a whole burst of 8 requests (it counts nothing
# itself). Write `herd(loads, burst, picker, rng)` that applies `picker(loads, rng)` to each request of the burst
# **without updating `loads`**, and returns how many requests each replica received. Then predict: how many of the 8
# land on replica 1 with an argmin picker, and how many (on average) with `p2c`?

# %% exercise
def herd(loads, burst, picker, rng):
    ### BEGIN SOLUTION
    got = [0] * len(loads)
    for _ in range(burst):
        got[picker(loads, rng)] += 1
    return got
    ### END SOLUTION


def argmin(loads, rng):
    return min(range(len(loads)), key=lambda i: loads[i])


argmin_on_replica_1 = None   # your prediction: an integer
p2c_on_replica_1 = None      # your prediction: expected value, e.g. 3.0
### BEGIN SOLUTION
argmin_on_replica_1 = 8      # every request sees the same minimum
p2c_on_replica_1 = 4.0       # replica 1 is the lighter of any pair it is in: P(sampled) = 1 - C(3,2)/C(4,2) = 1/2
### END SOLUTION

# %% check
rng = random.Random(7)
assert herd([3, 1, 4, 2], 8, argmin, rng) == [0, 8, 0, 0] == [0, argmin_on_replica_1, 0, 0]
trials = [herd([3, 1, 4, 2], 8, p2c, rng)[1] for _ in range(5000)]
mean = sum(trials) / len(trials)
assert abs(mean - p2c_on_replica_1) < 0.1, mean
print(f"✅ argmin sends all 8 to replica 1; p2c sends {mean:.2f} on average and spreads the rest")

# %% [markdown]
# ## Exercise 1.5 — pick the load signal
# For each situation, choose the router from `{"round-robin", "least-outstanding", "power-of-two"}` that a design
# review should accept, using what this notebook showed. Put your answers in `choice`.
#
# * **a** — one router process, requests of very different sizes, the router can count its own in-flight requests.
# * **b** — twenty router replicas (each sees only its own traffic) reading a shared metrics snapshot that is a few
#   seconds old.
# * **c** — identical, tiny requests (e.g. an embedding model with fixed-length inputs) and a cheap, dumb proxy.

# %% exercise
choice = {"a": None, "b": None, "c": None}
### BEGIN SOLUTION
choice = {"a": "least-outstanding",   # exact counts of what it sent; works while one process sees all traffic
          "b": "power-of-two",        # robust to stale and partial information, no coordination between routers
          "c": "round-robin"}         # when requests really are uniform, counting them is counting work
### END SOLUTION

# %% check
assert choice == {"a": "least-outstanding", "b": "power-of-two", "c": "round-robin"}, choice
print("✅ right signal for each situation")

# %% [markdown]
# ## In a design review
# **Two-minute version.** "LLM requests are not interchangeable: prefill work, KV memory and residency time each vary
# by an order of magnitude or more across a realistic mix, and the output length is unknown up front. So counting
# requests does not balance work. Load-aware routing needs a signal close to the cause of latency — the engine's
# waiting queue and its KV usage — plus the router's own in-flight counters, which are never stale. When several
# router replicas act on scraped metrics, argmin herds; power-of-two-choices keeps nearly all the benefit with none
# of the coordination. And all of this is before the biggest lever: replicas are caches (notebook 02)."
#
# **Drills**
# 1. *Why does least-connections work for web servers but not here?* A connection is a proxy for work only when
#    requests are similar; here a streaming connection can be 0.3 s or 30 s of engine time and 80 or 800 KV blocks.
# 2. *Our router scrapes vLLM every 15 s via Prometheus. What breaks?* Argmin routing herds on the stale minimum; use
#    the router's own in-flight counts, scrape the engines directly at sub-second intervals, or use power-of-two.
# 3. *What does Little's law tell you about a 20-replica fleet at 50 req/s with a 12 s mean E2E?* 600 requests in
#    flight — about 30 per replica, which you check against each replica's KV capacity and `max_num_seqs`.
