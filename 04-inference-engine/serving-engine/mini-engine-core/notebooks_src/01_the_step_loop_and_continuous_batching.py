# %% [markdown]
# # 01 · The step loop and continuous batching
#
# **Tier:** T0. It needs a CPU only and no network, and it runs in about ten seconds. The same ideas on a real GPU
# with vLLM are in `vllm-serving-lab` (notebook `02_serve_and_measure`, T1).
#
# ## The one-minute version
# An inference engine is a loop. Each iteration of the loop is a **step**. In each step, the engine does these
# things:
#
# 1. It asks the **scheduler** which requests run and how many tokens each request gets in this step.
# 2. It packs all of those tokens into **one flat batch**. The batch can hold a prompt chunk for one request and a
#    single decode token for another. For each token, the batch holds its position and the place to write its K/V
#    (the *slot mapping*). For each request, the batch holds its *block table*.
# 3. It runs **one forward pass** over that batch.
# 4. It **samples** a token for each request whose prompt is now fully processed.
# 5. It **updates** the state. Finished requests leave and free their KV blocks. Requests that wait can join in the
#    next step.
#
# The membership changes in each step, not in each batch. That is **continuous (iteration-level) batching**. It is
# the reason that a short request never waits for a long one. At the end of this notebook, you can explain these
# things:
#
# * what one step is,
# * why the batch is flat,
# * how much continuous batching gains over static batching,
# * how KV blocks cap concurrency,
# * how the engine divides one step across two GPUs (tensor parallelism).
#
# Primer: §1 *Anatomy of an engine*, §2 *Continuous batching*, §9 *Parallelism inside the engine*
# (`../../PRIMER.md`).

# %%
import numpy as np

from minengine import Engine, SamplingParams, TinyLM, decode, encode, perf

model = TinyLM()                       # a tiny deterministic Llama-style model, byte-level vocabulary
print("vocabulary: 256 bytes + EOS;", "'The' ->", encode("The"))
print("parameters:", model.num_params(), "| KV bytes per token (bf16):", model.cfg.kv_bytes_per_token())
print("greedy continuation of 'The engine ':", repr(decode(model.generate_dense(encode("The engine "), 24))))

# %% [markdown]
# The weights of the model are random, except its embedding. The embedding encodes English letter-pair statistics.
# Thus the model produces meaningless text that looks like English. Greedy decoding falls into a loop (`tofofof`),
# and Notebook 04 repairs that with penalties. **The engine does not care what the model says.** All of the rest of
# this notebook is about the machinery.
#
# ## Worked example 1 — three requests through the engine
# The numbers are small, so that each step fits on one line. A block holds 4 tokens, and the budget is 16 tokens per
# step.

# %%
eng = Engine(model, num_blocks=32, block_size=4, max_num_batched_tokens=16, max_num_seqs=4)
prompts = ["The engine ", "When memory runs out", "A request that finishes leaves"]
for p, n in zip(prompts, [10, 6, 3]):
    eng.add_request(p, SamplingParams(max_tokens=n, temperature=0))
while eng.scheduler.has_unfinished():
    eng.step()
print(eng.trace())

# %% [markdown]
# Read the trace line by line:
#
# * **step 1**: `r0`'s 11-token prompt runs whole (`prefill`) and samples its first token. `r1` gets only the
#   5 tokens that the budget still has (`chunk`). The engine samples nothing for `r1` yet. `r2` waits.
# * **step 3**: two decodes (1 token each) and a 14-token chunk of `r2`. *Decode and prefill share a step*.
#   A modern engine has no separate "prefill steps".
# * **finished**: in the step where a request gets to `max_tokens`, the request frees its blocks. Thus the `kv` value
#   on the next line is lower, because `kv` counts the blocks held while that step ran. A request that waits can take
#   the slot immediately, at the next step.
#
# ## Worked example 2 — one step, taken apart
# `Engine.step` has about 30 lines. The next cell does its stages by hand, so that you can see the flat batch.

# %%
eng = Engine(model, num_blocks=32, block_size=4, max_num_batched_tokens=16)
eng.add_request("The engine ", SamplingParams(max_tokens=4, temperature=0))
eng.add_request("Each step", SamplingParams(max_tokens=4, temperature=0))
eng.step()                                            # r0's prompt (11) + 5 of r1's 9 prompt tokens

out = eng.scheduler.schedule()                        # 1. schedule: who runs, how many tokens
print("scheduled:", [(r.request_id, n) for r, n in out.scheduled], "-> r0 decodes 1, r1 finishes its prompt")
batch = eng.build_batch(out.scheduled)                # 2. flatten
print("token_ids   :", batch.token_ids, " (r0's last sampled token, then r1's last 4 prompt tokens)")
print("positions   :", batch.positions)
print("slot_mapping:", batch.slot_mapping, " = block_id * 4 + offset")
print("block_tables:", batch.block_tables, " context_lens:", batch.context_lens)
logits = model.forward(batch, eng.cache)              # 3. one forward pass for everyone
print("logits shape:", logits.shape, "-> one row per request (only the last token of each is unembedded)")
sampled = {r.request_id: int(np.argmax(row)) for (r, n), row in zip(out.scheduled, logits)}   # 4. sample
eng.scheduler.update(out, sampled)                    # 5. update
print("sampled:", {k: decode([v]) for k, v in sampled.items()})

# %% [markdown]
# One batch mixes a decode token and a prompt chunk. The engine reads the weights **once** for the whole batch, and
# the tokens of each request share that one read. Attention is the only per-request part. Each request reads *its
# own* K/V through *its own* block table. The engine turns only the last token of each request into logits.
#
# ## Worked example 3 — the paged, batched engine computes exactly the reference
# `TinyLM.generate_dense` recomputes the whole sequence for each token. It uses textbook attention and no cache. The
# engine divides prompts into chunks, puts requests into batches and keeps the KV cache in pages. It must also produce
# the same tokens.

# %%
eng = Engine(model, num_blocks=32, block_size=4, max_num_batched_tokens=8, max_num_seqs=2)
outs = eng.generate(prompts, SamplingParams(max_tokens=12, temperature=0))
for p, o in zip(prompts, outs):
    ref = model.generate_dense(encode(p), 12)
    print(f"{p!r:34} engine == dense: {o.token_ids == ref}")

# %% [markdown]
# ## Worked example 4 — static vs continuous batching
# Static batching fills a batch and runs it until the **longest** request finishes. Then it starts the next batch.
# Continuous batching fills a slot again in the step that frees the slot. The requests are the same, and the slot
# count is the same:

# %%
lens = [2, 9, 3, 4]                                   # output lengths of four requests, 2 slots
static = max(lens[:2]) + max(lens[2:])                # batch {2, 9} then batch {3, 4}
print("static:     ", static, "steps, slot utilisation", f"{sum(lens) / (2 * static):.0%}")
print("continuous: ", 9, "steps, slot utilisation", f"{sum(lens) / (2 * 9):.0%}  (slot 1: 2+3+4 = 9, slot 2: 9)")

# %% [markdown]
# ## Exercise 1.1 — how many tokens does a request get this step?
# In vLLM V1's unified scheduler, a request is only two numbers. `num_tokens` is the prompt tokens plus the tokens
# generated until now. `num_computed_tokens` is the tokens whose K/V are already in the cache. Write
# `num_new_tokens(num_tokens, num_computed, budget_left)`. It returns the tokens that the scheduler gives to the
# request in this step, with chunked prefill on.

# %% exercise
def num_new_tokens(num_tokens, num_computed, budget_left):
    ### BEGIN SOLUTION
    return min(num_tokens - num_computed, budget_left)
    ### END SOLUTION

# %% check
assert num_new_tokens(12, 11, 16) == 1          # a decoding request: its one new token
assert num_new_tokens(40, 0, 16) == 16          # a long prompt: a chunk the size of the budget
assert num_new_tokens(40, 32, 16) == 8          # the prompt's last chunk
assert num_new_tokens(40, 32, 0) == 0           # no budget left this step
print("✅ one rule covers prefill, chunked prefill and decode")

# %% [markdown]
# ## Exercise 1.2 — the slot mapping
# The engine writes the K/V of each new token to a *slot*:
# `block_table[pos // block_size] * block_size + pos % block_size`. Write
# `slot_mapping(block_table, start, n, block_size)` for the `n` tokens at positions `start .. start+n-1`.

# %% exercise
def slot_mapping(block_table, start, n, block_size):
    ### BEGIN SOLUTION
    return [block_table[p // block_size] * block_size + p % block_size for p in range(start, start + n)]
    ### END SOLUTION

# %% check
assert slot_mapping([7, 2, 9], 3, 3, 4) == [31, 8, 9]    # pos 3 is in block 7; pos 4, 5 in block 2
eng = Engine(model, num_blocks=16, block_size=4, max_num_batched_tokens=16)
eng.add_request("When memory runs out", SamplingParams(max_tokens=2))
out = eng.scheduler.schedule()
(req, n), = out.scheduled
assert slot_mapping(eng.kv.tables[req.request_id], 0, n, 4) == list(eng.build_batch(out.scheduled).slot_mapping)
print("✅ slot mapping matches the engine's:", slot_mapping(eng.kv.tables[req.request_id], 0, n, 4))

# %% [markdown]
# ## Exercise 1.3 — predict a request's peak KV blocks
# A request has a $P$-token prompt and generates $O$ tokens. How many blocks does it hold at its peak? Be careful.
# The engine returns the **last** sampled token to the user, but it never sends that token back through the model.
# Thus that token never gets a K/V slot. Write `peak_blocks(P, O, block_size)`.

# %% exercise
def peak_blocks(P, O, block_size):
    ### BEGIN SOLUTION
    return -(-(P + O - 1) // block_size)
    ### END SOLUTION

# %% check
def measured_peak(P, O, B=4):
    e = Engine(model, num_blocks=64, block_size=B, max_num_batched_tokens=64)
    e.generate([[65] * P], SamplingParams(max_tokens=O, ignore_eos=True))
    return max(rec.kv_used for rec in e.history)          # blocks held while each step ran
for P, O in [(8, 1), (8, 2), (5, 4), (13, 20)]:
    assert peak_blocks(P, O, 4) == measured_peak(P, O), (P, O)
print("✅ peak blocks = ceil((P + O - 1) / B), e.g. P=8, O=1 ->", peak_blocks(8, 1, 4), "block(s)")

# %% [markdown]
# ## Exercise 1.4 — static vs continuous batching, in general
# Write the two step counters for decode-only requests with the given output lengths (FCFS order):
#
# * `static_steps(lens, slots)`: batches of `slots` requests, in order. Each batch takes as long as its longest
#   request.
# * `continuous_steps(lens, slots)`: each slot takes the next request in the queue at the moment that the slot
#   becomes free.

# %% exercise
def static_steps(lens, slots):
    ### BEGIN SOLUTION
    return sum(max(lens[i:i + slots]) for i in range(0, len(lens), slots))
    ### END SOLUTION


def continuous_steps(lens, slots):
    ### BEGIN SOLUTION
    free_at = [0] * slots                          # the step at which each slot becomes free
    for n in lens:
        i = free_at.index(min(free_at))            # the earliest-free slot takes the next request
        free_at[i] += n
    return max(free_at)
    ### END SOLUTION

# %% check
assert static_steps([2, 9, 3, 4], 2) == 13 and continuous_steps([2, 9, 3, 4], 2) == 9
rng = np.random.default_rng(0)
lens = list(rng.integers(10, 400, 64))            # a long-tailed mix of output lengths
s, c = static_steps(lens, 8), continuous_steps(lens, 8)
print(f"64 requests, 8 slots: static {s} steps, continuous {c} steps -> {s / c:.2f}x the throughput")
assert c < s
print("✅ continuous batching never waits for the longest request in a batch")

# %% [markdown]
# ## Exercise 1.5 — blocks cap concurrency (a real model)
# If Llama-3.1-8B in bf16 runs on one 24 GB L4 with `gpu_memory_utilization=0.9`, it leaves
# `perf.kv_cache_blocks(...)` blocks of 16 tokens for the KV cache. On average, a chat request has 1,000 prompt tokens
# and 200 output tokens. How many requests can be in memory at the same time? Set `concurrent` (use your
# `peak_blocks`).

# %% exercise
gpu, llm = perf.GPUS["L4"], perf.LLMS["llama-3.1-8b"]
blocks = perf.kv_cache_blocks(gpu, llm)
### BEGIN SOLUTION
concurrent = blocks // peak_blocks(1000, 200, 16)
### END SOLUTION

# %% check
assert blocks == 2164 and concurrent == 28
print(f"✅ {blocks} blocks / {peak_blocks(1000, 200, 16)} per request = {concurrent} concurrent requests")
print("   the same arithmetic as 00-foundations/gpu-capacity-planning; Notebook 06 moves it with quantization")

# %% [markdown]
# ## Exercise 1.6 — one step on two GPUs (tensor parallelism)
# When one GPU is too small or too slow, the engine divides each layer across GPUs (the Megatron pattern). The
# MLP $(\operatorname{silu}(h W_{\text{gate}}) \odot (h W_{\text{up}}))\,W_{\text{down}}$ divides cleanly:
#
# * Each rank holds **half the columns** of $W_{\text{gate}}$ and $W_{\text{up}}$. This is *column-parallel*. The
#   rank calculates half of the hidden features with no communication, because the gate operation is elementwise.
# * Each rank also holds the **half of the rows** of $W_{\text{down}}$ that matches its columns. This is *row-parallel*. The rank
#   produces a full-size **partial sum**.
#
# The addition of the partial sums of the ranks is the **all-reduce**. Write `tp_mlp_partials(L, h, world)`. It
# returns the `world` partial outputs, and rank $r$ uses only its shard of the three weights. $h$ is the normalised
# input. `L` is the weights of one layer, for example `L["w_gate"]` of shape `(d_model, d_ff)`.

# %% exercise
def tp_mlp_partials(L, h, world):
    ### BEGIN SOLUTION
    d_ff = L["w_gate"].shape[1]
    parts = []
    for r in range(world):
        cols = slice(r * d_ff // world, (r + 1) * d_ff // world)      # this rank's hidden features
        g = h @ L["w_gate"][:, cols]
        parts.append((g / (1 + np.exp(-g)) * (h @ L["w_up"][:, cols])) @ L["w_down"][cols, :])
    return parts
    ### END SOLUTION

# %% check
from minengine.model import rms_norm

L, h = model.layers[0], rms_norm(model.emb[encode("The engine runs")])   # 15 tokens
g = h @ L["w_gate"]
dense = (g / (1 + np.exp(-g)) * (h @ L["w_up"])) @ L["w_down"]            # the unsplit MLP
parts = tp_mlp_partials(L, h, 2)
assert len(parts) == 2 and all(p.shape == dense.shape for p in parts)
assert all(not np.allclose(p, dense) for p in parts)                     # one rank alone holds a partial sum
np.testing.assert_allclose(parts[0] + parts[1], dense, atol=1e-12)       # the all-reduce restores the output
np.testing.assert_allclose(sum(tp_mlp_partials(L, h, 4)), dense, atol=1e-12)
n_ar, nbytes = perf.tp_allreduces(model.cfg.n_layers, model.cfg.d_model, len(h))
print(f"✅ two ranks with half the MLP weights each + one all-reduce == the dense MLP")
print(f"   with attention split by heads the same way: {n_ar} all-reduces per forward pass "
      f"({model.cfg.n_layers} layers x 2), each {len(h)} tokens x {model.cfg.d_model} values")
print("   a 70B model at 64 decodes:", perf.tp_allreduces(80, 8192, 64), "= (all-reduces, bytes each), primer §9")

# %% [markdown]
# Attention divides in the same way. The Q/K/V projections are column-parallel **by head**: each rank owns whole
# heads, and thus also their slice of the KV cache. The output projection is row-parallel, and its partial sums need
# the second all-reduce of the layer. Each step has two all-reduces per layer on its critical path. That is why
# tensor parallelism stays inside one NVLink domain. To run it for real, you need two GPUs (tier T2,
# `vllm-serving-lab`).
#
# ## In a design review
# **The two-minute version.** "An engine is a loop around one forward pass. In each step, the scheduler gives out a
# token budget. The requests that run get tokens first. A request in decode gets one token, and a request that is
# still in prefill gets a chunk. Then new requests get tokens while the budget, the sequence slots and the KV blocks
# last.
#
# "The engine puts all scheduled tokens into one flat batch, so it reads the weights one time for all requests.
# Attention reads the history of each request through its block table.
#
# "After the pass, we sample for the requests whose prompt is complete. Finished requests free their blocks
# immediately, so a request that waits joins at the next step. That is continuous batching. The scheduler alone
# gives about 1.6× the throughput of static batching on a mix of 10–400-token outputs (more with longer tails or
# more slots).
#
# "KV blocks cap concurrency, not compute. For an 8B model on an L4, the cap is about 28 chat requests. When a model
# needs more than one GPU, tensor parallelism divides each layer and pays two all-reduces per layer per step."
#
# **Drill questions**
# 1. *Why does the engine not have prefill steps and decode steps?* Because the scheduler records only two counts: the
#    computed tokens and the total tokens. A step mixes decode tokens and prefill chunks under one token budget.
# 2. *Why is batching nearly free during decode?* Decode is memory-bound. The step time is the weight read. All
#    requests in the batch share that read (Notebook 02 puts numbers on it).
# 3. *A request's prompt is 8 tokens and it generates 1 token with block size 4. How many blocks?* Two. The
#    only sampled token never goes back through the model, so it needs no slot.
