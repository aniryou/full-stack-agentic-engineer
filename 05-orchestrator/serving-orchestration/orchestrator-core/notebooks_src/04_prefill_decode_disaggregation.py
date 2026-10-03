# %% [markdown]
# # 04 · Prefill/decode disaggregation
#
# **Tier:** T0 (CPU only, about 15 seconds, no network). Every number in this notebook is a **simulated** number from
# `fleetsim`.
#
# ## The one-minute version
# Prefill is compute-bound, and its load comes in bursts. Decode is memory-bound, and its load is steady. On a shared
# engine, one 2,048-token prefill chunk makes every decode request in the batch wait half a second (on an L4).
#
# **Disaggregation** runs prefill on a pool of its own. It sends the KV cache of each prompt to a decode replica. Thus
# the decode steps no longer stop for short times. The costs are:
#
# * The **transfer**: $\text{prompt tokens} \:\times$ $\text{KV bytes/token} \:\div$ $\text{link bandwidth}$. The
#   transfer adds to TTFT. On fast GPUs, the transfer needs RDMA-class links.
# * **Two pools to size**: the P:D ratio must match the input/output mix of the traffic. If it does not, one pool is
#   idle while requests wait in the queue of the other pool. Small fleets have bad fragmentation.
# * An extra hop on short prompts. On short prompts, this hop gives no benefit.
#
# The first solution to try is a smaller prefill chunk, because chunked prefill already interleaves prefill with
# decode. Disaggregation is worth its cost when you have long prompts, tight inter-token SLOs, fast links, and fleets
# that are sufficiently large to divide. Read Primer §5 (and `01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md` §8).

# %%
import dataclasses

from fleetsim import (H100_8B, L4_8B, LLAMA_8B_KV, PowerOfTwo, chat, decode_step_s, max_decode_batch, mix, pd_plan,
                      rag, run_pd, search_pd, step_time, table)
import math

p = L4_8B


def long_prompts(rate=0.5, duration=150, seed=11):
    """RAG-style: ~6,000-token prompts (4 retrieved chunks), 250-token answers."""
    return rag(rate, duration, seed=seed, system=400, docs=4, doc=1400, corpus=5000, zipf=0.0, output=250)


res = run_pd(p, long_prompts(), 0, 4, router=PowerOfTwo(seed=1))
s = res.summary(ttft_slo=3.0, tpot_slo=0.1)
print(f"a 2,048-token chunk takes {step_time(p, 2048, 2048) * 1e3:.0f} ms; a 16-request decode step "
      f"{step_time(p, 16, 16 * 6100) * 1e3:.0f} ms")
print(table([{"ITL p50": s["itl_p50"], "ITL p99": s["itl_p99"], "TPOT p95": s["tpot_p95"], "TTFT p95": s["ttft_p95"]}],
            title="simulated: 4 aggregated L4 replicas, ~6k-token prompts"))

# %% [markdown]
# The median gap between tokens is one ordinary decode step. The p99 gap is a prefill chunk that a different request
# brought into the batch. Users see this gap as a stream that stops for a moment in the middle of a sentence.
#
# ## Exercise 4.1 — what the KV transfer costs
# Write `kv_transfer_s(tokens, kv_bytes_per_token, link_gbps, latency_s=0.0)`. Convert the KV bytes of the prompt to
# bits. Divide the bits by the link speed, `link_gbps` gigabits per second. Then add a constant latency.

# %% exercise
def kv_transfer_s(tokens, kv_bytes_per_token, link_gbps, latency_s=0.0):
    ### BEGIN SOLUTION
    return latency_s + tokens * kv_bytes_per_token * 8 / (link_gbps * 1e9)
    ### END SOLUTION

# %% check
assert kv_transfer_s(4096, LLAMA_8B_KV, 400) == 4096 * 131072 * 8 / 400e9      # 512 MiB over 400 Gb/s: 10.7 ms
assert abs(kv_transfer_s(4096, LLAMA_8B_KV, 10) - 0.4295) < 1e-4                # the same over 10 GbE: 0.43 s
assert kv_transfer_s(1, 1, 8, latency_s=0.001) == 0.001 + 1e-9
print("✅ kv_transfer_s works — 128 KiB per token adds up: a 4k prompt is half a gigabyte")

# %% [markdown]
# ## Exercise 4.2 — fast GPUs need fast links
# A transfer is acceptable if it costs at most 10 % of the prefill that it replaces. For a 6,000-token prompt, return
# the links that meet this budget on an **L4** and on an **H100**. Select the links from `links` (in Gb/s). The
# prefill time is `tokens / profile.compute_tok_s`.

# %% exercise
links = {"10 GbE": 10, "25 GbE": 25, "100 GbE": 100, "400G RDMA": 400,
         "NVLink-class": 3600}           # ~450 GB/s one way on H100 (verify for your part)


def links_within_budget(profile, tokens=6000, budget=0.10):
    ### BEGIN SOLUTION
    prefill = tokens / profile.compute_tok_s
    return [name for name, gbps in links.items() if kv_transfer_s(tokens, LLAMA_8B_KV, gbps) <= budget * prefill]
    ### END SOLUTION

# %% check
on_l4, on_h100 = links_within_budget(L4_8B), links_within_budget(H100_8B)
assert on_l4 == ["100 GbE", "400G RDMA", "NVLink-class"] and on_h100 == ["400G RDMA", "NVLink-class"]
print(f"✅ L4 (prefill {6000 / L4_8B.compute_tok_s:.2f} s): {on_l4}\n   H100 (prefill {6000 / H100_8B.compute_tok_s:.2f} s): "
      f"{on_h100} — an 8x faster prefill needs an 8x faster link for the same overhead")

# %% [markdown]
# ## Worked example — every split of eight GPUs
# The fleet has eight L4 replicas and a 100 Gb/s link. The traffic is 1.2 req/s of ~6k-token prompts. `search_pd`
# runs every xPyD split on the same traffic (0 prefill = aggregated). The SLO is a TTFT of at most 3 s and a
# TPOT of at most 80 ms. The analytic plan is next to the results.
#
# The fleets in this example have a constant size. `fleetsim` does not autoscale P/D pools, but the planners of
# primer §4.5 do.

# %%
def heavier():
    return long_prompts(rate=1.2)


rows = search_pd(p, heavier, 8, ttft_slo=3.0, tpot_slo=0.08, make_router=lambda: PowerOfTwo(seed=1))
print(table(rows, title="simulated: xPyD splits of 8x L4"))
plan = pd_plan(p, rate=1.2, isl=6060, osl=250, itl_slo_s=0.08)
print({k: round(v, 2) for k, v in plan.items()})

# %% [markdown]
# The analytic plan asks for about 2.75 prefill replicas and 4.2 decode replicas. Rounded up, this plan is 3P5D, and
# 3P5D is also the best split in the simulator. All the other splits have a bad balance. With too few prefill
# replicas, TTFT increases very fast in the prefill queue. With too few decode replicas, TPOT increases very fast. The
# aggregated fleet meets the TTFT SLO but not the tight TPOT SLO, because of the chunk stall.
#
# ## Exercise 4.3 — the decode side of the plan
# The prefill side is a division: 1.2 req/s x 6,060 tokens over 3,781 tok/s at a 0.7 utilisation cap is 2.75
# replicas. The decode side depends on one number: the number of requests that a decode replica can batch. Three
# limits cap this number:
#
# * `profile.max_seqs`.
# * The KV pool, which must hold the context of every request. The pool has `profile.kv_blocks` blocks of
#   `profile.block` tokens. `ctx` tokens of context plus the next token need
#   $\lceil (\text{ctx} + 1) / \text{block} \rceil$ blocks.
# * The ITL SLO. One decode step for the whole batch, `decode_step_s(profile, batch, ctx)`, must meet this SLO.
#
# Write `largest_decode_batch(profile, ctx, itl_slo_s)`. If even one request misses the SLO, return 0. Then
# **predict** which cap is the limit for the plan of the worked example: `"kv"` or `"itl"`. For this plan, `ctx` =
# 6,060 + 250 / 2 = 6,185 on the L4, with an 80 ms SLO.

# %% exercise
def largest_decode_batch(profile, ctx, itl_slo_s):
    ### BEGIN SOLUTION
    fits = min(profile.max_seqs, profile.kv_blocks // math.ceil((ctx + 1) / profile.block))
    b = 0
    while b < fits and decode_step_s(profile, b + 1, ctx) <= itl_slo_s:
        b += 1
    return b
    ### END SOLUTION


binding = None      # "kv" or "itl"
### BEGIN SOLUTION
binding = "kv"      # 2,193 blocks // 387 per request = 5, while the ITL alone would allow 8
### END SOLUTION

# %% check
for prof, ctx, slo in ((L4_8B, 6185, 0.08), (L4_8B, 500, 0.07), (L4_8B, 6185, 0.05), (H100_8B, 6185, 0.012),
                       (H100_8B, 2000, 0.03)):
    assert largest_decode_batch(prof, ctx, slo) == max_decode_batch(prof, ctx, slo), (prof.name, ctx, slo)
b = largest_decode_batch(p, 6185, 0.08)
kv_fit = p.kv_blocks // math.ceil(6186 / p.block)
itl_fit = largest_decode_batch(dataclasses.replace(p, kv_blocks=10**9), 6185, 0.08)
assert binding == ("kv" if kv_fit < itl_fit else "itl"), (kv_fit, itl_fit)
n_d = 1.2 * 250 / (b / decode_step_s(p, b, 6185))           # output tok/s demanded / per-replica decode tok/s
assert abs(n_d - plan["decode_replicas"]) < 1e-9
print(f"✅ batch {b} (KV fits {kv_fit}, the ITL SLO alone {itl_fit}) -> {n_d:.2f} decode replicas against "
      f"{plan['prefill_replicas']:.2f} prefill: 3P5D")

# %% [markdown]
# ## Worked example — try smaller prefill chunks first
# Chunked prefill already interleaves prefill with decode. A smaller token budget per step makes each stall shorter.
# Because a decode step is memory-bound, a small prefill chunk goes into the same step at almost no cost (the insight
# of Sarathi-Serve).

# %%
rows = []
for budget in (2048, 1024, 512, 256):
    prof = dataclasses.replace(p, max_tokens=budget)
    s = run_pd(prof, heavier(), 0, 8, router=PowerOfTwo(seed=1)).summary(ttft_slo=3.0, tpot_slo=0.08)
    rows.append({"max_num_batched_tokens": budget, "ttft_p95": s["ttft_p95"], "itl_p99": s["itl_p99"],
                 "tpot_p95": s["tpot_p95"], "slo_attainment": s["slo_attainment"]})
print(table(rows, title="simulated: 8 aggregated L4 replicas, chunk budget sweep"))

# %% [markdown]
# On this model and GPU, the chunk-budget knob gives results as good as the best split. This solution uses one pool,
# no network, and no ratio to maintain. Two things that the step model leaves out make this result the optimistic end:
#
# * The step model has no per-chunk efficiency loss. Real engines lose some prefill efficiency at small chunks.
# * The prefill compute of the step model is linear in tokens. The step model has no attention FLOPs, which add about
#   +10 % for these 6k-token prompts, and more for longer prompts. Thus the TTFT of every row looks better than it is.
#
# Larger models make each decode step shorter in comparison with a chunk. With these models, disaggregation becomes
# the better choice (DistServe, Splitwise, the llm-d P/D guide: medium-large models, long inputs).
#
# ## Exercise 4.4 — disaggregate only the prompts worth it
# In practice, traffic mixes short chat turns with long RAG prompts. In llm-d, the endpoint picker decides for each
# request. First, it selects the decode pod. Then its `prefix-based-pd-decider` examines the cache of that decode pod.
# It sends the prompt to a prefill pod only if at least `nonCachedTokens` tokens of the prompt are not in that cache.
#
# In this notebook, `pd_threshold` is this limit. The sidecar of the decode pod does what the decision says.
#
# **Predict** which threshold gives a 2P6D fleet the best SLO attainment on this mix. The choices are:
#
# * `0` (disaggregate everything).
# * `2048` (only long prompts).
# * `100_000` (never: the prefill pool is idle).

# %% exercise
best_threshold = None
### BEGIN SOLUTION
best_threshold = 2048     # short prompts gain nothing from the hop; long ones shed their stall
### END SOLUTION

# %% check
def traffic_mix():
    return mix(chat(3.0, 150, seed=31, system=300, user=200, output=150),
               rag(0.8, 150, seed=32, docs=4, doc=1400, corpus=5000, zipf=0.0, output=150))


got = {}
for thr in (0, 2048, 100_000):
    got[thr] = run_pd(p, traffic_mix(), 2, 6, router=PowerOfTwo(seed=1), pd_threshold=thr).summary(
        ttft_slo=2.0, tpot_slo=0.1)["slo_attainment"]
print(table([{"pd_threshold": k, "slo_attainment": v} for k, v in got.items()], title="simulated: 2P6D, chat + RAG"))
assert best_threshold == max(got, key=got.get), got
print("✅ conditional disaggregation: pay the hop only where the stall it removes is larger")

# %% [markdown]
# ## In a design review
# **Two-minute version.** "Disaggregation separates two phases that have opposite bottlenecks. Thus we can set the
# batch and the parallelism of each pool for its own phase, and decode never waits for a prefill chunk. Before I
# propose it, I will examine three numbers:
#
# "First, the stall that it removes: the p99 inter-token latency against our SLO, after we adjust the chunk budget.
# Second, the KV transfer that it adds: $\text{prompt tokens} \times \text{KV bytes per token}$ over our link, in
# comparison with the prefill time. Third, the size of the fleet: is it sufficiently large to divide at the P:D
# ratio of the traffic without fragmentation?
#
# "If the answer is yes, we use xPyD, sized from that ratio, conditional on prompt length, over RDMA: llm-d or Dynamo
# with NIXL. If the answer is no, we use an aggregated fleet with an adjusted chunk budget."
#
# **Drills**
# 1. *When does disaggregation make things worse?* Short prompts make it worse: there is no stall to remove, and the
#    hop is extra. Slow links make it worse: the transfer takes about as long as the prefill. A split that does not
#    match the input/output ratio and small fleets also make it worse.
# 2. *How do you size the pools?* Calculate the prefill replicas from the prompt tokens/s at a utilisation cap.
#    Calculate the decode replicas from the output tokens/s at the ITL SLO, with the KV capacity as a limit.
#    Calculate them again each time the input/output mix changes.
# 3. *Why does a faster GPU make the network matter more?* The prefill time decreases as the FLOP/s increase, but
#    the KV bytes stay the same. Thus the same transfer becomes a larger fraction of TTFT, and H100-class prefill
#    needs RDMA or NVLink, not 100 GbE.
