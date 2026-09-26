# %% [markdown]
# # 02 · Chunked prefill and the token budget
#
# **Tier:** T0 — CPU only, no network, about twenty seconds. Every latency in this notebook is **SIMULATED**
# by a roofline model (`minengine.perf`) driving the real scheduler; measure the real curve with vLLM in
# `vllm-serving-lab` (notebook `03_knobs_and_tradeoffs`, T1).
#
# ## The one-minute version
# A step's cost depends on how many tokens are in it. Up to a **knee** of a few hundred tokens a step is
# **memory-bound** — it takes as long as reading the weights, so extra tokens are almost free. Past the knee it is
# **compute-bound** and every token costs FLOPs. A long prompt processed in one step makes that step long, and
# every request decoding alongside it sees one huge gap between two tokens — prefill **interferes** with decode.
# **Chunked prefill** caps the tokens per step (`max_num_batched_tokens`) and splits long prompts across steps,
# so decodes keep flowing (Sarathi-Serve's "stall-free batching"). The budget is the knob: bigger means better
# prefill efficiency and lower TTFT, smaller means smoother inter-token latency (ITL). The *other* budget is KV
# memory: when running requests outgrow the block pool, the newest is **preempted** and later recomputed. You will
# be able to put numbers on all of that.
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
# First with chunked prefill (budget 16 tokens per step), then without (the budget must then cover a whole
# prompt, so it is raised to `max_model_len`).

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
# With chunking, the 83-token prompt goes in as chunks of 13 (16 minus the three decodes) while `r0..r2` still
# get one token every step. Without chunking it goes in whole: one step with 86 tokens. On this toy that is harmless; on a real
# GPU that step is long, and the three decoding users feel it. Let's price steps on real hardware.
#
# ## Worked example 2 — what a step costs (SIMULATED)
# `perf.step_cost` charges a step `max(bytes / bandwidth, FLOPs / peak) + overhead` with 80% of datasheet
# bandwidth, 60% of datasheet FLOP/s and 2 ms per step of overhead — assumptions to replace with measurements.

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
# From 1 to 256 tokens the step time barely moves: this is why decode batching is nearly free, and why a
# single decode token is so expensive per token. Past the knee, time grows linearly with tokens.
#
# ## Worked example 3 — the knob, swept (SIMULATED)
# 200 requests, Poisson arrivals at 6/s, prompts 500–6,000 tokens, outputs 100–400, on an H100 with Llama-3.1-8B.

# %%
g, m = perf.GPUS["H100-SXM"], perf.LLMS["llama-3.1-8b"]
work = perf.Workload(n_requests=200, rate=6, prompt_len=(500, 6000), output_len=(100, 400), seed=1)
results = [perf.simulate(g, m, work, enable_chunked_prefill=False, label="no chunking")]
for budget in [8192, 2048, 512, 256]:
    results.append(perf.simulate(g, m, work, max_num_batched_tokens=budget, label=f"budget {budget}"))
for r in results:
    print(r.summary())

# %% [markdown]
# Read the columns against each other:
#
# * **ITL p99** collapses as the budget shrinks — a step can no longer contain a whole 6k-token prompt.
# * **TTFT** rises as the budget shrinks — a prompt now needs several steps, each also carrying decodes.
# * **Throughput** hardly moves — the same work, scheduled differently. The knob moves latency *between*
#   TTFT and ITL; it does not create capacity.
# * Below the knee (256) steps are memory-bound: ITL is smoothest, but prefill runs in small inefficient
#   pieces and TTFT doubles. vLLM's defaults (2,048 on an L4/A100-class GPU, 8,192 on H100-class for the API
#   server, as of Sep 2026, verify) sit between the extremes.
#
# ## Worked example 4 — when blocks run out: preemption by recompute
# The budget limits tokens per step; the KV pool limits tokens in flight. Four requests that each grow to ~9
# blocks share a 16-block pool. When a running request needs a block and none is free, the scheduler preempts the
# **newest** running request: frees its blocks, resets it to zero computed tokens and puts it at the front of the
# queue. Its generated tokens are kept; later it is re-prefilled ("recompute") and continues.

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
# Look at the `recompute` lines: `r1` restarts at position 20, not 0. Its freed blocks were still in the prefix cache
# (Notebook 03), so only the tail had to be recomputed. Preemption is correct — the outputs match — but costly:
# the victim's work is redone and every request waits while memory is short. vLLM V1 preempts by recompute only;
# swapping blocks to CPU memory was a V0 mode (verify). Preemptions in production are a sizing signal: more KV
# memory (`gpu_memory_utilization`, FP8 KV, a smaller model), fewer concurrent sequences, or more replicas.
#
# ## Exercise 2.1 — the chunk schedule
# A `prompt_len`-token prompt is admitted while `num_decodes` requests decode (1 token each per step, scheduled
# first). With budget `budget`, list the prompt's chunk sizes step by step.

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
# Ignoring KV reads, attention and overheads: a step must read every weight once and do `2 x params` FLOPs per
# token. Write `roofline_step_ms(weight_bytes, params, tokens, peak_flops, hbm_bw)`.

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
# Solve `weight_bytes / hbm_bw = 2 x params x n / peak_flops` for `n`, then predict whether each step below is
# memory- or compute-bound **with the default efficiencies** (60% FLOPs, 80% bandwidth: the knee moves to
# 0.6 / 0.8 = 0.75 of the ideal one).

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
# H100 + Llama-3.1-8B, up to 64 requests decoding at ~2,000 tokens of context, p99 ITL SLO 25 ms. The worst step
# is 64 decodes plus a prefill chunk that fills the rest of the budget. Pick the **largest** budget from the list
# whose worst step fits the SLO (use `perf.step_time`, chunk at context 0).

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
# An L4 serves Qwen2.5-1.5B to 16 users decoding at ~1,000 tokens of context. An 8,000-token prompt arrives.
# Compute (with `perf.step_time`): `stall_ms` — the gap the 16 users see if the prompt runs in one step (no
# chunking); `chunked_steps` and `worst_chunked_ms` — with a 512-token budget (16 go to decodes, the prompt gets
# the rest; chunk `i` starts at context `i x 496`).

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
assert 400 < stall_ms < 440 and chunked_steps == 17 and worst_chunked_ms < 40
print(f"✅ normal ITL {normal:.1f} ms; unchunked stall {stall_ms:.0f} ms; "
      f"chunked: {chunked_steps} steps, worst gap {worst_chunked_ms:.1f} ms - SIMULATED")

# %% [markdown]
# ## Exercise 2.6 — size the KV cache so nobody is preempted
# H100 + Llama-3.1-8B, 200 requests at 6/s, prompts 500–3,000 tokens, outputs 100–400 (below). Find the smallest
# `num_blocks` (16 tokens each) among `candidates` for which the simulated run has **zero** preemptions, and
# `kv_gb`, the KV memory that takes in GB (`perf.LLMS["llama-3.1-8b"].kv_bytes_per_token` bytes per token).

# %%
g, m = perf.GPUS["H100-SXM"], perf.LLMS["llama-3.1-8b"]
load = perf.Workload(n_requests=200, rate=6, prompt_len=(500, 3000), output_len=(100, 400), seed=2)
candidates = [2000, 3000, 4000, 5000, 6000, 8000]

# %% exercise
### BEGIN SOLUTION
runs = {nb: perf.simulate(g, m, load, num_blocks=nb) for nb in candidates}
min_blocks = min(nb for nb, r in runs.items() if r.preemptions == 0)
kv_gb = min_blocks * 16 * m.kv_bytes_per_token / 1e9
### END SOLUTION

# %% check
assert min_blocks == 5000 and abs(kv_gb - 10.49) < 0.01
for nb in [3000, min_blocks]:
    r = perf.simulate(g, m, load, num_blocks=nb, label=f"{nb} blocks")
    print(r.summary())
print(f"✅ {min_blocks} blocks = {kv_gb:.1f} GB of KV; an H100 left ~{perf.kv_cache_blocks(g, m)} blocks after the "
      "weights - short KV memory shows up first as preemptions and exploding TTFT, not as errors")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "A forward pass is memory-bound until a few hundred tokens — the weight read
# dominates — and compute-bound after. So decode tokens batch almost for free, and a long prompt is expensive in
# proportion to its length. If an 8k-token prompt runs in one step, every user decoding in that step waits for
# it: on an L4 with a 1.5B model their 17 ms token gap becomes about 420 ms. Chunked prefill caps each step at
# `max_num_batched_tokens`; decodes are scheduled first and the prompt gets the rest, so the worst gap stays near
# 30 ms at a 512 budget, at the price of a slightly later first token for the long prompt. The budget trades TTFT
# against ITL; throughput barely changes. I pick the largest budget whose worst step meets the ITL SLO. The second
# budget is KV memory: if running requests outgrow the block pool, the newest is preempted and recomputed — outputs
# stay correct, but TTFT explodes long before anything errors. So we size KV for the peak working set and alert on
# `vllm:num_preemptions`; if prefill and decode SLOs still conflict at our scale, that is the argument for
# disaggregating them (05)."
#
# **Drill questions**
# 1. *Why doesn't a larger batch make decode slower?* — Below the knee a step is the weight read; tokens share it.
# 2. *Chunked prefill is on, yet p99 ITL is bad. What do you check?* — The budget: if it is far past the knee,
#    each step is still long; lower it until the worst step meets the SLO.
# 3. *`vllm:num_preemptions` is climbing. What does it mean and what do you change?* — Running requests outgrow the
#    KV pool; each preemption throws work away. Add KV memory (utilisation, FP8 KV), cap `max_num_seqs`, or scale out.
