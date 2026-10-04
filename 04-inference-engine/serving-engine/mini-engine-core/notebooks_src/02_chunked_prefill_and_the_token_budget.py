# %% [markdown]
# # 02 · Chunked prefill and the token budget
#
# **Tier:** T0. It needs a CPU only and no network, and it runs in about twenty seconds. Each latency in this
# notebook is a **SIMULATED** value: a roofline model (`minengine.perf`) gives the time of each step of the real scheduler.
# Measure the real curve with vLLM in `vllm-serving-lab` (notebook `03_knobs_and_tradeoffs`, T1).
#
# ## The one-minute version
# The cost of a step depends on the number of tokens in it. Up to a **knee** of a few hundred tokens, a step is
# **memory-bound**. It takes as long as the weight read, so more tokens are almost free. After the knee, the step is
# **compute-bound**, and each token costs FLOPs. If the engine processes a long prompt in one step, that step is
# long. Then each request that decodes in the same step sees one large gap between two tokens: prefill
# **interferes** with decode.
#
# **Chunked prefill** caps the tokens per step (`max_num_batched_tokens`) and divides long prompts across steps.
# Thus the decodes continue with no stop (Sarathi-Serve's "stall-free batching").
#
# The budget is the knob. A larger budget gives fewer steps per prompt and a lower TTFT. A smaller budget gives a
# smoother inter-token latency (ITL). Under saturation, a budget slightly past the knee also gives the most
# throughput. The reason is that each step then mixes compute-bound prefill with memory-bound decodes.
#
# The *other* budget is KV memory. When the requests that run become larger than the block pool, the scheduler
# **preempts** the newest request, and the engine recomputes it later. At the end of this notebook, you can put
# numbers on all of that.
#
# Primer: §3 *Chunked prefill and prefill/decode interference*, §4 *KV cache management revisited*,
# §11 *Measuring an engine* (`../../PRIMER.md`).

# %%
import math

import numpy as np

from minengine import Engine, SamplingParams, TinyLM, encode, perf

model = TinyLM()

# %% [markdown]
# ## Worked example 1 — a long prompt arrives while three requests decode
# The first run uses chunked prefill (a budget of 16 tokens per step). The second run does not. Without chunked
# prefill, the budget must hold a whole prompt, so the cell sets it to `max_model_len`.

# %%
LATE = "The keys and values of every token stay in a cache of small blocks, one table each."
print("the late prompt is", len(encode(LATE)), "tokens")


def arrive_late(**kw):                                # prefix caching off: it is Notebook 03's topic
    eng = Engine(model, num_blocks=64, block_size=4, max_model_len=128, enable_prefix_caching=False, **kw)
    for p in ["The engine ", "Each step", "When memory"]:
        eng.add_request(p, SamplingParams(max_tokens=14, temperature=0))
    for _ in range(3):
        eng.step()                                    # three requests are now decoding
    eng.add_request(LATE, SamplingParams(max_tokens=2, temperature=0))
    for _ in range(6):
        eng.step()
    return eng

print("chunked prefill, budget 16:")
print(arrive_late(max_num_batched_tokens=16).trace(6))
print("\nno chunking, budget 128:")
print(arrive_late(max_num_batched_tokens=128, enable_chunked_prefill=False).trace(6))

# %% [markdown]
# With chunked prefill, the 83-token prompt goes in as chunks of 13 (16 minus the three decodes). During that time,
# `r0..r2` still get one token in each step. Without chunked prefill, the prompt goes in whole: one step with 86
# tokens. On this toy model, that causes no problem. But on a real GPU that step is long, and the three users in
# decode feel it. The next example calculates the cost of steps on real hardware.
#
# ## Worked example 2 — what a step costs (SIMULATED)
# `perf.step_cost` gives a step this cost:
#
# $$
# \max\left(\frac{\text{bytes}}{\text{bandwidth}}, \frac{\text{FLOPs}}{\text{peak}}\right) + \text{overhead}
# $$
#
# It uses 80% of datasheet bandwidth, 60% of datasheet FLOP/s and 2 ms of overhead per step. These values are
# assumptions. Replace them with measurements.

# %%
for gname, mname in [("L4", "qwen2.5-1.5b"), ("H100-SXM", "llama-3.1-8b")]:
    g, m = perf.GPUS[gname], perf.LLMS[mname]
    print(f"{gname} + {mname}: ideal knee {perf.knee_tokens(g, m):.0f} tokens, "
          f"with the efficiencies {perf.knee_tokens(g, m, 0.6, 0.8):.0f}  (SIMULATED)")
    for n in [1, 64, 256, 512, 2048, 8192]:
        c = perf.step_cost(g, m, [(0, n)])
        print(f"   {n:5d} tokens in the step: {1e3 * c['t']:6.1f} ms  ({c['bound']:7} bound)  "
              f"{1e3 * c['t'] / n:7.3f} ms per token")

# %% [markdown]
# From 1 to 256 tokens, the step time almost does not change. This is why decode batching is nearly free. It is
# also why a single decode token has a high cost per token. After the knee, the time increases linearly with the
# tokens.
#
# ## Worked example 3 — the knob, swept (SIMULATED)
# The workload is 200 requests on an H100 with Llama-3.1-8B. The prompts are 500–6,000 tokens, and the outputs are
# 100–400. The first run is an open-loop Poisson stream at 6 requests/s, a load that the engine can serve as fast as
# it arrives. The second run sends all 200 at once. This saturates the engine, so the run measures **capacity**.
# `goodput` counts the requests per second that met a 1 s TTFT **and** a 50 ms TPOT SLO (`SimResult.goodput`,
# primer §11).

# %%
g, m = perf.GPUS["H100-SXM"], perf.LLMS["llama-3.1-8b"]
for rate in [6, math.inf]:
    work = perf.Workload(n_requests=200, rate=rate, prompt_len=(500, 6000), output_len=(100, 400), seed=1)
    print("Poisson at 6/s:" if rate == 6 else "\nall 200 at t = 0 (saturated):")
    results = [perf.simulate(g, m, work, enable_chunked_prefill=False, label="no chunking")]
    for budget in [8192, 2048, 512, 256]:
        results.append(perf.simulate(g, m, work, max_num_batched_tokens=budget, label=f"budget {budget}"))
    for r in results:
        print(r.summary(), f"| goodput {r.goodput(ttft_slo=1.0, tpot_slo=0.05):4.2f} req/s")

# %% [markdown]
# Compare the columns with each other:
#
# * **ITL p99** decreases by a large factor as the budget decreases. A step can no longer hold a whole 6k-token
#   prompt.
# * **TTFT** increases as the budget decreases. A prompt now needs several steps, and each step also holds decodes.
# * **Throughput at 6/s is the same in every row**. The reason is not that the knob is free. The reason is that the
#   engine keeps up with the load. An open-loop run below capacity finishes exactly the offered load (6/s × 250
#   tokens on average). Such a run cannot show an effect on throughput, but a run that saturates the engine can.
# * **Saturated, the budget sets capacity.** 512 does best. Each step mixes a prompt chunk (compute-bound) with the
#   KV reads of the decodes that run (memory-bound). Thus the tensor cores and the HBM are busy at the same time.
#   That is Sarathi-Serve's case for hybrid batches. Whole prompts or 8,192-token steps alternate compute-heavy
#   prefill steps with memory-bound decode steps.
#
#   Part of the gain comes from memory, not from overlap. Read the `preempt` and `peak KV` columns. The three larger
#   settings fill the KV pool and preempt a few requests, and the engine recomputes the work of those requests. 512
#   never fills the pool.
#
#   At 256, a step is just above the knee with these efficiencies (~221 tokens, next exercise). Its time is mostly
#   the weight read, the KV reads of the decodes and the 2 ms overhead. The prompt gets small chunks, which spread
#   the constant costs over few tokens. TTFT doubles at 6/s, and capacity decreases by about a quarter.
# * **Goodput** is the honest summary. Saturated, each row produces ~1,700–2,300 tokens/s, but almost no request
#   meets the SLO. The defaults of vLLM lie between the extremes (2,048 on an L4/A100-class GPU, 8,192 on H100-class for
#   the API server, as of Sep 2026, verify).
#
# ## Worked example 4 — when blocks run out: preemption by recompute
# The budget limits the tokens per step. The KV pool limits the tokens in flight. Four requests, each of which grows
# to ~9 blocks, share a 16-block pool.
#
# If a request that runs needs a block and no block is free, the scheduler preempts the **newest** request that runs.
# The scheduler frees the blocks of that request and sets the request back to zero computed tokens. The scheduler also
# puts the request at the front of the queue. The request keeps its generated tokens. Later, the engine does the
# prefill of the request again ("recompute"), and the request continues.

# %%
prompts = ["The engine runs a loop", "Each step it picks the", "When memory runs out,", "A request that finishes"]
eng = Engine(model, num_blocks=16, block_size=4, max_num_batched_tokens=32, max_num_seqs=4)
outs = eng.generate(prompts, SamplingParams(max_tokens=20, temperature=0))
for rec in eng.history:
    if rec.preempted or any(kind == "recompute" for _, kind, *_ in rec.batch):
        print(rec)
print("\npreemptions:", eng.scheduler.num_preemptions, "(vLLM exports this as vllm:num_preemptions)")
print("outputs identical to the dense reference:",
      all(o.token_ids == model.generate_dense(encode(p), 20) for p, o in zip(prompts, outs)))

# %% [markdown]
# Look at the `recompute` lines. `r1` starts again at position 20, not at 0. Its freed blocks were still in the
# prefix cache (Notebook 03), so the engine recomputed only the tail. Preemption is correct: the outputs match. But
# it has a high cost. The engine does the work of the preempted request again, and each request waits
# while memory is short.
#
# vLLM V1 preempts by recompute only. A swap of blocks to CPU memory was a V0 mode (verify). In production,
# preemptions are a signal to change the size of the deployment. The corrections are more KV memory
# (`gpu_memory_utilization`, FP8 KV, a smaller model), fewer concurrent sequences, or more replicas.
#
# ## Exercise 2.1 — the chunk schedule
# The scheduler admits a `prompt_len`-token prompt while `num_decodes` requests decode. Each decode gets 1 token per
# step, and the scheduler schedules the decodes first. With the budget `budget`, list the chunk sizes of the prompt,
# step by step.

# %% exercise
def chunk_sizes(prompt_len, budget, num_decodes):
    ### BEGIN SOLUTION
    sizes, left = [], prompt_len
    while left > 0:
        n = min(left, budget - num_decodes)
        sizes.append(n)
        left -= n
    return sizes
    ### END SOLUTION

# %% check
assert chunk_sizes(40, 16, 3) == [13, 13, 13, 1]
eng = arrive_late(max_num_batched_tokens=16)
for _ in range(3):
    eng.step()                                    # let the late prompt finish
measured = [n for rec in eng.history for rid, kind, s, n, t in rec.batch if rid == "r3" and kind != "decode"]
assert measured == chunk_sizes(len(encode(LATE)), 16, 3), measured
print("✅ chunks:", measured, "- the engine agrees")

# %% [markdown]
# ## Exercise 2.2 — the roofline step time
# Do not include KV reads, attention and overheads. Then a step must read each weight once and do
# $2 \times \mathtt{params}$ FLOPs per token. Write `roofline_step_ms(weight_bytes, params, tokens, peak_flops, hbm_bw)`.

# %% exercise
def roofline_step_ms(weight_bytes, params, tokens, peak_flops, hbm_bw):
    ### BEGIN SOLUTION
    return 1e3 * max(weight_bytes / hbm_bw, 2 * params * tokens / peak_flops)
    ### END SOLUTION

# %% check
g, m = perf.GPUS["L4"], perf.LLMS["qwen2.5-1.5b"]
for n in [1, 64, 512, 2048]:
    mine = roofline_step_ms(m.weight_bytes, m.params, n, g.peak_flops, g.hbm_bw)
    full = 1e3 * perf.step_time(g, m, [(0, 1)] * n, flop_eff=1, bw_eff=1, overhead_s=0)
    assert abs(mine - full) / full < 0.01, (n, mine, full)
print(f"✅ L4 + Qwen2.5-1.5B: 1 token {roofline_step_ms(m.weight_bytes, m.params, 1, g.peak_flops, g.hbm_bw):.1f} ms, "
      f"2048 tokens {roofline_step_ms(m.weight_bytes, m.params, 2048, g.peak_flops, g.hbm_bw):.1f} ms")

# %% [markdown]
# ## Exercise 2.3 — find the knee, predict the bound
# Solve
#
# $$
# \frac{\mathtt{weight\_bytes}}{\mathtt{hbm\_bw}} = \frac{2 \times \mathtt{params} \times n}{\mathtt{peak\_flops}}
# $$
#
# for $n$. Then predict if each step in the next cell is memory-bound or compute-bound **with the default
# efficiencies**. These are 60% FLOPs and 80% bandwidth, so the knee moves to 0.6 / 0.8 = 0.75 of the ideal one.

# %% exercise
def knee(peak_flops, hbm_bw, bytes_per_param):
    ### BEGIN SOLUTION
    return peak_flops * bytes_per_param / (2 * hbm_bw)
    ### END SOLUTION


predictions = {                                   # "memory" or "compute"
    ("L4", "qwen2.5-1.5b", 256): None,
    ("H100-SXM", "llama-3.1-8b", 256): None,
    ("L4", "qwen2.5-1.5b", 1024): None,
    ("H100-SXM", "llama-3.1-8b", 128): None,
}
### BEGIN SOLUTION
predictions = {("L4", "qwen2.5-1.5b", 256): "memory",        # knee 403 x 0.75 = 302 > 256
               ("H100-SXM", "llama-3.1-8b", 256): "compute", # knee 295 x 0.75 = 221 < 256
               ("L4", "qwen2.5-1.5b", 1024): "compute",
               ("H100-SXM", "llama-3.1-8b", 128): "memory"}
### END SOLUTION

# %% check
assert round(knee(989e12, 3.35e12, 2)) == 295 and round(knee(121e12, 300e9, 2)) == 403
for (gn, mn, n), guess in predictions.items():
    assert perf.step_cost(perf.GPUS[gn], perf.LLMS[mn], [(0, n)])["bound"] == guess, (gn, mn, n)
print("✅ knees: H100 bf16 295 tokens, L4 bf16 403 tokens (x 0.75 with the default efficiencies)")

# %% [markdown]
# ## Exercise 2.4 — pick `max_num_batched_tokens` for an ITL SLO
# The setup is H100 + Llama-3.1-8B. Up to 64 requests decode at ~2,000 tokens of context, and the p99 ITL SLO is
# 25 ms. The worst step is 64 decodes plus a prefill chunk that fills the rest of the budget. Select the **largest**
# budget in the list whose worst step meets the SLO. Use `perf.step_time`, with the chunk at context 0.

# %% exercise
g, m = perf.GPUS["H100-SXM"], perf.LLMS["llama-3.1-8b"]
candidates = [256, 512, 1024, 2048, 4096, 8192]
### BEGIN SOLUTION
worst = {b: perf.step_time(g, m, [(2000, 1)] * 64 + [(0, b - 64)]) for b in candidates}
budget = max(b for b in candidates if worst[b] <= 0.025)
### END SOLUTION

# %% check
assert budget == 512
print(f"✅ budget {budget}: worst step {1e3 * perf.step_time(g, m, [(2000, 1)] * 64 + [(0, budget - 64)]):.1f} ms "
      f"(1024 would be {1e3 * perf.step_time(g, m, [(2000, 1)] * 64 + [(0, 960)]):.1f} ms) - SIMULATED")

# %% [markdown]
# ## Exercise 2.5 — how long is the stall?
# An L4 serves Qwen2.5-1.5B to 16 users who decode at ~1,000 tokens of context. An 8,000-token prompt arrives.
# Calculate these values with `perf.step_time`:
#
# * `stall_ms`: the gap that the 16 users see if the prompt runs in one step (no chunked prefill).
# * `chunked_steps` and `worst_chunked_ms`: the same with a 512-token budget. 16 tokens go to the decodes, and the
#   prompt gets the rest. Chunk $i$ starts at context $i \times 496$.

# %% exercise
g, m = perf.GPUS["L4"], perf.LLMS["qwen2.5-1.5b"]
decodes = [(1000, 1)] * 16
### BEGIN SOLUTION
stall_ms = 1e3 * perf.step_time(g, m, decodes + [(0, 8000)])
per_chunk = 512 - 16
chunked_steps = math.ceil(8000 / per_chunk)
worst_chunked_ms = max(1e3 * perf.step_time(g, m, decodes + [(i * per_chunk, min(per_chunk, 8000 - i * per_chunk))])
                       for i in range(chunked_steps))
### END SOLUTION

# %% check
normal = 1e3 * perf.step_time(g, m, decodes)
assert 350 < stall_ms < 380 and chunked_steps == 17 and worst_chunked_ms < 40
print(f"✅ normal ITL {normal:.1f} ms; unchunked stall {stall_ms:.0f} ms; "
      f"chunked: {chunked_steps} steps, worst gap {worst_chunked_ms:.1f} ms - SIMULATED")

# %% [markdown]
# ## Exercise 2.6 — size the KV cache so nobody is preempted
# The setup is H100 + Llama-3.1-8B, with 200 requests at 6/s. The prompts are 500–3,000 tokens, and the outputs are
# 100–400 (next cell). Find these two values:
#
# * the smallest `num_blocks` (16 tokens each) in `candidates` for which the simulated run has **zero** preemptions,
# * `kv_gb`, the KV memory of those blocks in GB (`perf.LLMS["llama-3.1-8b"].kv_bytes_per_token` bytes per token).
#
# Then, at 2,000 blocks, turn off the **whole-prompt admission check** (`admit_whole_prompt=False`). The scheduler
# then admits a request when its first chunk fits, as a scheduler without vLLM's `scheduler_reserve_full_isl` does.
# Predict first: more preemptions or fewer? Then record the count in `without_check`.

# %%
g, m = perf.GPUS["H100-SXM"], perf.LLMS["llama-3.1-8b"]
load = perf.Workload(n_requests=200, rate=6, prompt_len=(500, 3000), output_len=(100, 400), seed=2)
candidates = [2000, 3000, 4000, 5000, 6000, 8000]

# %% exercise
### BEGIN SOLUTION
runs = {nb: perf.simulate(g, m, load, num_blocks=nb) for nb in candidates}
min_blocks = min(nb for nb, r in runs.items() if r.preemptions == 0)
kv_gb = min_blocks * 16 * m.kv_bytes_per_token / 1e9
without_check = perf.simulate(g, m, load, num_blocks=2000, admit_whole_prompt=False).preemptions
### END SOLUTION

# %% check
assert min_blocks == 4000 and abs(kv_gb - 8.39) < 0.01
for nb in [2000, 3000, min_blocks]:
    r = perf.simulate(g, m, load, num_blocks=nb, label=f"{nb} blocks")
    print(r.summary())
with_check = perf.simulate(g, m, load, num_blocks=2000).preemptions
assert without_check > 2 * with_check
print(f"✅ {min_blocks} blocks = {kv_gb:.1f} GB of KV; an H100 left ~{perf.kv_cache_blocks(g, m)} blocks after the "
      "weights - short KV memory shows up first as preemptions and exploding TTFT, not as errors")
print(f"   at 2,000 blocks: {with_check} preemptions with the whole-prompt check, {without_check} without - "
      "admitting prompts memory cannot finish just moves the preemption a few steps later (SIMULATED)")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "A forward pass is memory-bound until a few hundred tokens, because the weight read
# takes most of the time. After that, it is compute-bound. Thus decode tokens go into a batch almost for free, and
# the cost of a long prompt is proportional to its length. If an 8k-token prompt runs in one step, each user who
# decodes in that step waits for it. On an L4 with a 1.5B model, their 17 ms token gap becomes about 370 ms.
#
# "Chunked prefill caps each step at `max_num_batched_tokens`. The scheduler schedules the decodes first, and the
# prompt gets the rest. Thus the worst gap stays near 30 ms at a 512 budget. The cost is a slightly later first
# token for the long prompt.
#
# "At moderate load, the budget is a trade between TTFT and ITL. Saturated, the budget also sets capacity. A budget a
# few hundred tokens past the knee packs prefill FLOPs and decode KV reads into the same steps. A budget at the knee
# loses steps to constant costs. I select the largest budget whose worst step meets the ITL SLO, and I examine
# capacity under a load that saturates the engine.
#
# "The second budget is KV memory. If the requests that run become larger than the block pool, the scheduler
# preempts the newest one, and the engine recomputes it. The outputs stay correct, but TTFT increases by a large factor
# long before an error occurs. Thus we give the KV cache sufficient blocks for the peak working set, and we alert on
# `vllm:num_preemptions`. If the prefill and decode SLOs still conflict at our scale, that is the argument for
# prefill/decode disaggregation (05)."
#
# **Drill questions**
# 1. *Why does a larger batch not make decode slower?* Below the knee, a step is the weight read, and the tokens
#    share it.
# 2. *Chunked prefill is on, yet p99 ITL is bad. What do you examine?* The budget. If it is far past the knee, each
#    step is still long. Decrease it until the worst step meets the SLO.
# 3. *`vllm:num_preemptions` increases. What does it mean and what do you change?* The requests that run become
#    larger than the KV pool, and each preemption discards work. Add KV memory (utilisation, FP8 KV), cap
#    `max_num_seqs`, or scale out.
