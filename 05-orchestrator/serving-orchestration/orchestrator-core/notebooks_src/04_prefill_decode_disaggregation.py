# %% [markdown]
# # 04 · Prefill/decode disaggregation
#
# **Tier:** T0 — CPU only, about 20 seconds, no network. Every number below is **simulated** by `fleetsim`.
#
# ## The one-minute version
# Prefill is compute-bound and bursty; decode is memory-bound and steady. On a shared engine, one 2,048-token prefill
# chunk makes every decoding request in the batch wait half a second (on an L4). **Disaggregation** runs prefill on
# its own pool and ships each prompt's KV cache to a decode replica: decodes stop stuttering. The bill:
#
# * the **transfer** — `prompt tokens x KV bytes/token / link bandwidth` — added to TTFT, which needs RDMA-class
#   links on fast GPUs;
# * **two pools to size**: the P:D ratio must match the traffic's input/output mix, or one pool idles while the other
#   queues — small fleets fragment badly;
# * an extra hop, for nothing, on short prompts.
#
# The first fix to try is a smaller prefill chunk (chunked prefill already interleaves prefill with decode).
# Disaggregation earns its keep with long prompts, tight inter-token SLOs, fast links, and fleets large enough to
# split. Primer §5 (and `01-hardware-gpu-fabric/gpu-deployment/gpu-deployment-primer.md` §8).

# %%
import dataclasses

from fleetsim import (H100_8B, L4_8B, LLAMA_8B_KV, PowerOfTwo, chat, expand, mix, pd_plan, rag, run_pd, search_pd,
                      step_time, table)

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
# The median gap between tokens is one ordinary decode step; the p99 gap is a prefill chunk that some other request
# brought into the batch. Users see that as the stream freezing mid-sentence.
#
# ## Exercise 4.1 — what the KV transfer costs
# Write `kv_transfer_s(tokens, kv_bytes_per_token, link_gbps, latency_s=0.0)`: the prompt's KV bytes, converted to
# bits, over a link of `link_gbps` gigabits per second, plus a fixed latency.

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
# A transfer is tolerable if it costs at most 10 % of the prefill it replaces. For a 6,000-token prompt, return the
# links (from `links`, in Gb/s) that meet that budget on an **L4** and on an **H100** (prefill time =
# `tokens / profile.compute_tok_s`).

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
# Eight L4 replicas, 1.2 req/s of ~6k-token prompts, 100 Gb/s link. `search_pd` runs every xPyD split (0 prefill =
# aggregated) on the same traffic; the SLO is TTFT <= 3 s and TPOT <= 80 ms. Next to it, the analytic plan.

# %%
def heavier():
    return long_prompts(rate=1.2)


rows = search_pd(p, heavier, 8, ttft_slo=3.0, tpot_slo=0.08, make_router=lambda: PowerOfTwo(seed=1))
print(table(rows, title="simulated: xPyD splits of 8x L4"))
plan = pd_plan(p, rate=1.2, isl=6060, osl=250, itl_slo_s=0.08)
print({k: round(v, 2) for k, v in plan.items()})

# %% [markdown]
# The analytic plan asks for about 2.75 prefill and 4.2 decode replicas — 3P5D, which is also the simulator's winner.
# Everything else is lopsided: too few prefill replicas and TTFT explodes in the prefill queue; too few decode
# replicas and TPOT does. Aggregated serving meets the TTFT SLO but not the tight TPOT one, because of the chunk stall.
#
# ## Exercise 4.3 — the P:D ratio from first principles
# Prefill replicas needed = prefill tokens per second demanded / (prefill tokens per second one replica sustains x a
# utilisation cap). Decode replicas needed = output tokens per second demanded / output tokens per second one decode
# replica sustains at the ITL SLO. Write `pd_ratio(rate, isl, osl, prefill_tok_s, util, decode_tok_s)` returning
# `(prefill_replicas, decode_replicas)`.

# %% exercise
def pd_ratio(rate, isl, osl, prefill_tok_s, util, decode_tok_s):
    ### BEGIN SOLUTION
    return rate * isl / (prefill_tok_s * util), rate * osl / decode_tok_s
    ### END SOLUTION

# %% check
n_p, n_d = pd_ratio(1.2, 6060, 250, p.compute_tok_s, 0.7, plan["decode_tok_s_per_replica"])
assert abs(n_p - plan["prefill_replicas"]) < 1e-9 and abs(n_d - plan["decode_replicas"]) < 1e-9
print(f"✅ {n_p:.2f} prefill : {n_d:.2f} decode — round to the split that keeps both pools under their limits")

# %% [markdown]
# ## Worked example — try smaller prefill chunks first
# Chunked prefill already interleaves prefill with decode; a smaller token budget per step makes each stall shorter.
# Because a decode step is memory-bound, a small prefill chunk rides along almost for free (Sarathi-Serve's insight).

# %%
rows = []
for budget in (2048, 1024, 512, 256):
    prof = dataclasses.replace(p, max_tokens=budget)
    s = run_pd(prof, heavier(), 0, 8, router=PowerOfTwo(seed=1)).summary(ttft_slo=3.0, tpot_slo=0.08)
    rows.append({"max_num_batched_tokens": budget, "ttft_p95": s["ttft_p95"], "itl_p99": s["itl_p99"],
                 "tpot_p95": s["tpot_p95"], "slo_attainment": s["slo_attainment"]})
print(table(rows, title="simulated: 8 aggregated L4 replicas, chunk budget sweep"))

# %% [markdown]
# On this model and GPU the chunk-budget knob does as well as the best split — one pool, no network, no ratio to
# maintain. The simulator's step model has no per-chunk efficiency loss, so treat this as the optimistic end; real
# engines lose some prefill efficiency at small chunks, and bigger models make each decode step shorter relative to
# a chunk — which is where disaggregation pulls ahead (DistServe, Splitwise, the llm-d P/D guide: medium-large models,
# long inputs).
#
# ## Exercise 4.4 — disaggregate only the prompts worth it
# Real traffic mixes short chat turns with long RAG prompts. llm-d's P/D sidecar can serve a request locally on its
# decode replica when the uncached prompt is short (`pd_threshold` here). **Predict** which threshold gives a 2P6D
# fleet the best SLO attainment on this mix: `0` (disaggregate everything), `2048` (only long prompts) or `100_000`
# (never — the prefill pool idles).

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
# **Two-minute version.** "Disaggregation separates two phases with opposite bottlenecks so each pool can be batched
# and parallelised for its own phase, and decode never waits for a prefill chunk. Before proposing it I would check
# three numbers: the stall it removes — p99 inter-token latency against our SLO, after tuning the chunk budget; the
# KV transfer it adds — prompt tokens x KV bytes per token over our link, compared with the prefill time; and whether
# the fleet is big enough to split at the traffic's P:D ratio without fragmenting. If yes, xPyD sized from that ratio,
# conditional on prompt length, over RDMA — llm-d or Dynamo with NIXL. If not, aggregated with a tuned chunk budget."
#
# **Drills**
# 1. *When does disaggregation make things worse?* Short prompts (no stall to remove, an extra hop), slow links (the
#    transfer rivals the prefill), a split that does not match the input/output ratio, and small fleets.
# 2. *How do you size the pools?* Prefill replicas from prompt tokens/s at a utilisation cap; decode replicas from
#    output tokens/s at the ITL SLO (bounded by KV capacity); re-derive whenever the input/output mix shifts.
# 3. *Why does a faster GPU make the network matter more?* Prefill time shrinks with FLOP/s but the KV bytes do not;
#    the same transfer becomes a larger fraction of TTFT, so H100-class prefill wants RDMA or NVLink, not 100 GbE.
