# %% [markdown]
# # 04 · Loading, reliability and the cost of a token
#
# **Tier:** T0 — arithmetic with stated assumptions; no GPU. The measured counterpart for loading
# is `gpu-bench-lab` notebook `04_weights_loading_and_cold_start` (T0 on your disk, T1 with a GPU).
#
# ## The one-minute version
# Three fleet-level numbers come straight from the hardware. **Cold start**: weight bytes ÷ the
# slowest bandwidth on the path, plus provisioning, image pull and engine init — parallel and
# streamed loading attack the first term, warm pools and caches the rest. **Reliability**: failure
# rates add, so a 16K-GPU job is interrupted every few hours even though each GPU runs for years;
# training answers with checkpoints every `√(2 · checkpoint time · MTBF)`, inference with spare
# replicas, and bigger replicas are bigger failure domains. **Cost**: `$/M tokens = $/GPU-hr ÷
# (tokens/s × 3600 × utilisation) × 1e6` — every lever in this layer (batching, quantization, the
# right GPU, keeping it busy) shows up in that one line. Primer: `../PRIMER.md` §6–8.

# %%
import math

from roofline import cost, llm, reliability, specs, storage

m8, m70 = llm.PRESETS["llama-3.1-8b"], llm.PRESETS["llama-3.1-70b"]
b8, b70 = storage.checkpoint_bytes(m8.params()), storage.checkpoint_bytes(m70.params())
print(f"checkpoints in bf16: Llama-3.1-8B {b8 / 1e9:.1f} GB, Llama-3.1-70B {b70 / 1e9:.1f} GB")
print("\nassumed tier bandwidths (planning inputs, not measurements - replace with yours):")
for tier, gbs in storage.ASSUMED_GBS.items():
    print(f"  {tier:22s} {gbs:6.1f} GB/s -> 8B in {storage.read_time(b8, gbs):7.1f} s, 70B in {storage.read_time(b70, gbs):8.1f} s")

# %% [markdown]
# ## Parallel and streamed loading
# One HTTP stream from an object store is slow; many range reads in parallel add up until the NIC
# (or the disk, or the service) caps them. Streaming chunks straight to the GPUs overlaps fetch and
# copy, so the time tends to the slower hop instead of the sum.

# %%
fetch = storage.parallel_gbs(128, 0.1, storage.ASSUMED_GBS["nic-100g"])
print(f"128 streams x 0.1 GB/s, capped by a 100 Gb/s NIC: {fetch} GB/s")
seq = storage.load_time(b70, fetch, 50, gpus=8)
stm = storage.load_time(b70, fetch, 50, gpus=8, streamed=True)
print(f"70B onto 8 GPUs (PCIe Gen5 pinned): sequential {seq:.1f} s, streamed {stm:.1f} s")

# %% [markdown]
# ## A cold start, stage by stage
# All stage times below are **assumptions** you should replace with measurements (provisioning
# and image pull depend on your platform — see layers 03 and 05; engine init on the engine and its
# settings — layer 04). The shape is the lesson: fix the biggest bar first.

# %%
scenarios = {
    "new node, one stream": dict(fetch_gbs=0.1, provision_s=120, image_s=60, init_s=90),
    "new node, parallel + streamed": dict(fetch_gbs=fetch, provision_s=120, image_s=60, init_s=90, streamed=True),
    "warm node, page cache": dict(fetch_gbs=storage.ASSUMED_GBS["page-cache"], init_s=90, streamed=True),
}
for name, kw in scenarios.items():
    print(f"-- {name}\n{storage.cold_start(b70, h2d_gbs=50, gpus=8, **kw).table()}\n")

# %% [markdown]
# ## Failures are a rate, and rates add
# The Llama 3 report (Meta, 2024) counts 419 unexpected interruptions in 54 days of pre-training on
# 16,384 H100s. Backing out a per-GPU figure — it lumps GPU, host, network and software-detected
# faults together — gives the rate every fleet calculation starts from.

# %%
M = reliability.component_mtbf_from_observation(419, 54 * 24, 16384)
print(f"per-GPU MTBF equivalent: {M:,.0f} GPU-hours ({M / reliability.HOURS_PER_YEAR:.1f} years)")
for n in (8, 64, 1024, 16384):
    print(f"  {n:6d} GPUs: an interruption every {reliability.cluster_mtbf(M, n):8.1f} h, "
          f"{reliability.failures_per_day(n, M):6.2f}/day, P(no failure in 24 h) = {reliability.p_survive(24, n, M):.3f}")

# %% [markdown]
# ## Checkpoint interval: Young/Daly
# Checkpoint too often and you pay the write each time; too rarely and each failure throws away
# more work. With checkpoint time δ and system MTBF M the waste is ≈ δ/τ + τ/(2M), minimised at
# τ = √(2δM), where it equals √(2δ/M). Faster (asynchronous) checkpoints help as a square root.

# %%
Ms = reliability.cluster_mtbf(M, 16384) * 3600
for delta in (60, 10):
    tau = reliability.young_daly_interval(delta, Ms)
    print(f"16,384 GPUs, checkpoint {delta:2d} s: tau = {tau / 60:4.1f} min (Daly: {reliability.young_daly_interval(delta, Ms, True) / 60:.1f}), "
          f"waste {reliability.wasted_fraction(tau, delta, Ms):.1%} vs {reliability.wasted_fraction(3600, delta, Ms):.1%} checkpointing hourly")

# %% [markdown]
# ## Inference: replicas are failure domains
# A TP=8 replica is down whenever any of its 8 GPUs is, so its MTBF is M/8. Needing 8 replicas up,
# how many do you deploy? (MTTR of 48 h assumed: detect, drain, swap or repair the node.)

# %%
a8 = reliability.replica_availability(M, 8, 48)
print(f"TP=8 replica: MTBF {reliability.cluster_mtbf(M, 8):,.0f} h, availability {a8:.5f}")
for n in (8, 9, 10):
    print(f"  deploy {n:2d}: P(at least 8 up) = {reliability.p_at_least(8, n, a8):.5f}")

# %% [markdown]
# ## The cost of a token
# Throughput from notebook 02's roofline (an upper bound, so these are lower bounds on cost), and
# on-demand / Spot prices that you must check against today's list (`COMPUTE.md` at the repo root).

# %%
PRICES = {"L4 on-demand": 0.70, "H100 on-demand": 11.0, "H100 Spot": 3.7}   # $/GPU-hr, us-central1, Sep 2026 (verify)
l4, h100 = specs.get("l4"), specs.get("h100-sxm")
b_l4 = llm.max_batch_by_memory(m8, l4, 2048)
thr = {"L4 on-demand": llm.decode(m8, l4, b_l4, 2048), "H100 on-demand": llm.decode(m8, h100, 64, 2048),
       "H100 Spot": llm.decode(m8, h100, 64, 2048)}
for name, st in thr.items():
    print(f"{name:15s} batch {st.tokens:3d}: {st.tokens_per_s:6.0f} tok/s ({1 / st.time:5.1f} per user) -> "
          f"${cost.cost_per_million_tokens(PRICES[name], st.tokens_per_s):.3f}/M output tokens at 100% utilisation")
u = cost.utilisation([0.2] * 8 + [1.0] * 8 + [0.6] * 8)
print(f"\na fleet sized for its peak, loaded 20% / 100% / 60% across the day: utilisation {u:.0%} "
      f"-> every $/M above is {1 / u:.2f}x higher")

# %% [markdown]
# ## Exercise 4.1 — streamed loading
# Write `streamed_load_time(n, fetch_gbs, h2d_gbs, chunk)`: chunks flow disk/network → host → GPU,
# so the slower hop sets the rate and the faster hop adds one chunk of pipeline fill.

# %% exercise
def streamed_load_time(n, fetch_gbs, h2d_gbs, chunk=0.25 * 2**30):
    ### BEGIN SOLUTION
    slow = min(fetch_gbs, h2d_gbs) * 1e9
    return n / slow + chunk / (fetch_gbs * 1e9) + chunk / (h2d_gbs * 1e9) - chunk / slow
    ### END SOLUTION

# %% check
for n, f, h in [(b8, 2.0, 25), (b70, 12.5, 400), (b8, 30, 25)]:
    assert abs(streamed_load_time(n, f, h) - storage.load_time(n, f, h, streamed=True)) < 1e-9
print(f"✅ streamed ≈ bytes / slower hop: 8B at 2 GB/s fetch -> {streamed_load_time(b8, 2.0, 25):.2f} s "
      f"(sequential would be {storage.load_time(b8, 2.0, 25):.2f} s)")

# %% [markdown]
# ## Exercise 4.2 — a cold-start budget
# Autoscaling wants a new Llama-3.1-70B replica (bf16, 8 GPUs) serving within 120 s on a warm
# node pool: image pull 20 s and engine init 60 s (assumed). Write `required_fetch_gbs(n_bytes, budget_s, other_s)`
# (weights streamed, so fetch is the bottleneck) and `streams_needed(required_gbs, per_stream_gbs, cap_gbs)`
# (return `None` if even unlimited streams would hit the cap first).

# %% exercise
def required_fetch_gbs(n_bytes, budget_s, other_s):
    ### BEGIN SOLUTION
    return n_bytes / (budget_s - other_s) / 1e9
    ### END SOLUTION


def streams_needed(required_gbs, per_stream_gbs, cap_gbs):
    ### BEGIN SOLUTION
    if required_gbs > cap_gbs:
        return None
    return math.ceil(required_gbs / per_stream_gbs)
    ### END SOLUTION

# %% check
need = required_fetch_gbs(b70, 120, 20 + 60)
assert abs(need - 3.53) < 0.01
assert streams_needed(need, 0.1, 12.5) == 36
assert streams_needed(required_fetch_gbs(b70, 30, 20), 0.1, 12.5) is None    # 14 GB/s > a 100 Gb/s NIC
print(f"✅ 70B in 40 s needs {need:.2f} GB/s: 36 parallel streams; a 30 s budget needs a bigger pipe or a local cache")

# %% [markdown]
# ## Exercise 4.3 — how often should a 4,096-GPU job checkpoint?
# Using the per-GPU rate `M` above, write `young(delta_s, mtbf_s)` and `waste(tau_s, delta_s, mtbf_s)`
# (first-order, no restart cost), and compute them for a 4,096-GPU job writing a checkpoint in 30 s.

# %% exercise
def young(delta_s, mtbf_s):
    ### BEGIN SOLUTION
    return math.sqrt(2 * delta_s * mtbf_s)
    ### END SOLUTION


def waste(tau_s, delta_s, mtbf_s):
    ### BEGIN SOLUTION
    return delta_s / tau_s + tau_s / (2 * mtbf_s)
    ### END SOLUTION

# %% check
m4096 = reliability.cluster_mtbf(M, 4096) * 3600
tau = young(30, m4096)
assert abs(tau - reliability.young_daly_interval(30, m4096)) < 1e-9 and abs(tau / 60 - 27.2) < 0.05
assert abs(waste(tau, 30, m4096) - math.sqrt(2 * 30 / m4096)) < 1e-12
assert waste(tau, 30, m4096) < min(waste(tau / 2, 30, m4096), waste(2 * tau, 30, m4096))
print(f"✅ 4,096 GPUs (MTBF {m4096 / 3600:.1f} h), 30 s checkpoints: every {tau / 60:.1f} min, {waste(tau, 30, m4096):.1%} waste")

# %% [markdown]
# ## Exercise 4.4 — spares, and the size of a failure domain
# Write `p_at_least(k, n, a)` (binomial) and `replicas_needed(k, a, target)`. Then compare two ways to
# serve the same capacity at 99.9%: 8 replicas of TP=8, or (with FP8 weights so it fits) 16 replicas of TP=4.

# %% exercise
def p_at_least(k, n, a):
    ### BEGIN SOLUTION
    return sum(math.comb(n, i) * a ** i * (1 - a) ** (n - i) for i in range(k, n + 1))
    ### END SOLUTION


def replicas_needed(k, a, target):
    ### BEGIN SOLUTION
    n = k
    while p_at_least(k, n, a) < target:
        n += 1
    return n
    ### END SOLUTION

# %% check
a4 = reliability.replica_availability(M, 4, 48)
assert abs(p_at_least(8, 9, a8) - reliability.p_at_least(8, 9, a8)) < 1e-12
assert replicas_needed(8, a8, 0.999) == 10 and replicas_needed(16, a4, 0.999) == 18
print(f"✅ TP=8: 10 replicas = 80 GPUs for 64 GPUs of capacity; TP=4: 18 replicas = 72 GPUs — "
      "smaller failure domains need fewer spare GPUs")

# %% [markdown]
# ## Exercise 4.5 — dollars per million tokens
# Write `cost_per_mtok(price_per_gpu_hr, tokens_per_s, utilisation)` and fill `table` with the cost of each
# option in `thr` at 60% utilisation. Which GPU gives the cheaper token for Llama-3.1-8B — and which the faster one?

# %% exercise
def cost_per_mtok(price_per_gpu_hr, tokens_per_s, utilisation=1.0):
    ### BEGIN SOLUTION
    return price_per_gpu_hr / (tokens_per_s * 3600 * utilisation) * 1e6
    ### END SOLUTION


table = {}
### BEGIN SOLUTION
table = {name: cost_per_mtok(PRICES[name], st.tokens_per_s, 0.6) for name, st in thr.items()}
### END SOLUTION

# %% check
assert abs(cost_per_mtok(2.0, 1000) - 2 / 3.6) < 1e-12
assert abs(table["L4 on-demand"] - 1.10) < 0.01 and abs(table["H100 on-demand"] - 0.765) < 0.01
assert abs(table["H100 Spot"] - 0.257) < 0.01
print("✅ at 60% busy: L4 $1.10, H100 on-demand $0.77, H100 Spot $0.26 per M tokens — the H100's bandwidth and "
      "capacity (batch 64 vs 20) buy both the cheaper and the faster token here")

# %% [markdown]
# ## Exercise 4.6 — rent or own?
# An 8×H100 server: $300k (assumed), 4-year life, 10.2 kW at full load (verify), PUE 1.3, $0.10/kWh,
# $30k/year of fixed opex. Write `breakeven(rent_per_gpu_hr, fixed_per_gpu_hr, energy_per_gpu_hr)` — the
# utilisation above which owning is cheaper — and compare against on-demand and Spot rental.

# %%
fixed, energy = cost.owned_cost_per_hour(300_000, 4, 10.2, pue=1.3, usd_per_kwh=0.10, opex_per_year=30_000)
fixed_gpu, energy_gpu = fixed / 8, energy / 8
print(f"owned: ${fixed_gpu:.2f}/GPU-hr fixed + ${energy_gpu:.3f}/GPU-hr energy while busy")

# %% exercise
def breakeven(rent_per_gpu_hr, fixed_per_gpu_hr, energy_per_gpu_hr):
    ### BEGIN SOLUTION
    if rent_per_gpu_hr <= energy_per_gpu_hr:
        return math.inf
    return fixed_per_gpu_hr / (rent_per_gpu_hr - energy_per_gpu_hr)
    ### END SOLUTION

# %% check
assert abs(breakeven(11.0, fixed_gpu, energy_gpu) - 0.138) < 0.001
assert abs(breakeven(3.7, fixed_gpu, energy_gpu) - 0.424) < 0.001
assert breakeven(0.1, fixed_gpu, energy_gpu) == math.inf
print("✅ owning beats on-demand above ~14% utilisation but Spot only above ~42%: "
      "the break-even depends on which rental you would really use")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Cold start is bytes over the slowest bandwidth plus fixed stages:
# 141 GB from one object-store stream is 23 minutes, from 128 parallel streams at a 100 Gb/s NIC
# about 11 s, so I stream weights in parallel, keep a warm pool and cache the checkpoint locally,
# and then engine init dominates. Reliability: failure rates add, so a 16K-GPU run is interrupted
# every ~3 hours; checkpoint every √(2δM) — about 19 minutes at a 60 s checkpoint — and make
# checkpoints asynchronous. For serving, each TP=8 replica is an 8-GPU failure domain: 8 needed means
# 10 deployed for 99.9%. Cost is $/GPU-hr over tokens/s × utilisation, so I quote $/M tokens at a
# realistic utilisation, not at the roofline."
#
# **Drill.**
# 1. *Autoscaling takes 10 minutes; where do you look first?* — the stage table: usually the weight
#    fetch (one slow stream) — parallel/streamed loads, a local or regional cache, smaller precision.
# 2. *Checkpoint writes got 4× faster. How much less waste?* — waste ∝ √δ at the optimum: half.
# 3. *Should we buy GPUs because we are 60% utilised?* — only if 60% is above the break-even against
#    the rental you would actually use: ~14% vs on-demand, ~42% vs Spot, higher against cheaper rentals.
