# %% [markdown]
# # 01 · The step loop and continuous batching
#
# **Tier:** T0 — CPU only, no network, about ten seconds. The same ideas on a real GPU with vLLM are in
# `vllm-serving-lab` (notebook `02_serve_and_measure`, T1).
#
# ## The one-minute version
# An inference engine is a loop. Each iteration — a **step** — it:
#
# 1. asks the **scheduler** which requests run and how many tokens each gets this step;
# 2. packs all of those tokens into **one flat batch** (a prompt chunk here, a single decode token there),
#    with each token's position, where to write its K/V (the *slot mapping*) and each request's *block table*;
# 3. runs **one forward pass** over that batch;
# 4. **samples** a token for every request whose prompt is now fully processed;
# 5. **updates** state: finished requests leave and free their KV blocks, waiting ones can join next step.
#
# Membership changes every step, not every batch — that is **continuous (iteration-level) batching**, and it
# is why a short request never waits for a long one. By the end you can explain what one step is, why the batch
# is flat, how much continuous batching buys over static batching, and how KV blocks cap concurrency.
#
# Primer: §1 *Anatomy of an engine*, §2 *Continuous batching* (`../../PRIMER.md`).

# %%
import numpy as np

from minengine import Engine, SamplingParams, TinyLM, decode, encode, perf

model = TinyLM()                       # a tiny deterministic Llama-style model, byte-level vocabulary
print("vocabulary: 256 bytes + EOS;", "'The' ->", encode("The"))
print("parameters:", model.num_params(), "| KV bytes per token (bf16):", model.cfg.kv_bytes_per_token())
print("greedy continuation of 'The engine ':", repr(decode(model.generate_dense(encode("The engine "), 24))))

# %% [markdown]
# The model's weights are random except its embedding, which encodes English letter-pair statistics, so it
# babbles English-looking text — and greedy decoding falls into a loop (`tofofof`), which Notebook 04 fixes with
# penalties. **The engine does not care what the model says.** Everything below is about the machinery.
#
# ## Worked example 1 — three requests through the engine
# Small numbers so every step fits on a line: blocks of 4 tokens, a budget of 16 tokens per step.

# %%
eng = Engine(model, num_blocks=32, block_size=4, max_num_batched_tokens=16, max_num_seqs=4)
prompts = ["The engine ", "When memory runs out", "A request that finishes leaves"]
for p, n in zip(prompts, [10, 6, 3]):
    eng.add_request(p, SamplingParams(max_tokens=n, temperature=0))
while eng.scheduler.has_unfinished():
    eng.step()
print(eng.trace())

# %% [markdown]
# Read it line by line:
#
# * **step 1** — `r0`'s 11-token prompt runs whole (`prefill`) and samples its first token; `r1` gets only the
#   5 tokens left in the budget (`chunk`: nothing is sampled for it yet); `r2` waits.
# * **step 3** — two decodes (1 token each) and a 14-token chunk of `r2`: *decode and prefill share a step*.
#   There are no separate "prefill steps" in a modern engine.
# * **finished** — the step a request hits `max_tokens` it frees its blocks (the next line's `kv`, which counts
#   the blocks held while that step ran, is lower) and its slot is open to a waiting request at the very next step.
#
# ## Worked example 2 — one step, taken apart
# `Engine.step` is about 30 lines. Here are its stages by hand, so you can see the flat batch.

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
# One batch mixes a decode token and a prompt chunk. The weights are read **once** for the whole batch; every
# request's tokens ride along. Attention is the only per-request part — each request reads *its own* K/V
# through *its own* block table, and only each request's last token is turned into logits.
#
# ## Worked example 3 — the paged, batched engine computes exactly the reference
# `TinyLM.generate_dense` recomputes the whole sequence every token with textbook attention and no cache.
# The engine chunks, batches and pages — and must produce the same tokens.

# %%
eng = Engine(model, num_blocks=32, block_size=4, max_num_batched_tokens=8, max_num_seqs=2)
outs = eng.generate(prompts, SamplingParams(max_tokens=12, temperature=0))
for p, o in zip(prompts, outs):
    ref = model.generate_dense(encode(p), 12)
    print(f"{p!r:34} engine == dense: {o.token_ids == ref}")

# %% [markdown]
# ## Worked example 4 — static vs continuous batching
# Static batching fills a batch, runs it until the **longest** request finishes, then starts the next batch.
# Continuous batching refills a slot the step it frees. Same requests, same slot count:

# %%
lens = [2, 9, 3, 4]                                   # output lengths of four requests, 2 slots
static = max(lens[:2]) + max(lens[2:])                # batch {2, 9} then batch {3, 4}
print("static:     ", static, "steps, slot utilisation", f"{sum(lens) / (2 * static):.0%}")
print("continuous: ", 9, "steps, slot utilisation", f"{sum(lens) / (2 * 9):.0%}  (slot 1: 2+3+4 = 9, slot 2: 9)")

# %% [markdown]
# ## Exercise 1.1 — how many tokens does a request get this step?
# In vLLM V1's unified scheduler a request is just `num_tokens` (prompt + generated so far) and
# `num_computed_tokens` (tokens whose K/V are already cached). Write `num_new_tokens(num_tokens, num_computed,
# budget_left)`: the tokens it would be scheduled this step (chunked prefill on).

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
# Every new token's K/V is written to a *slot*: `block_table[pos // block_size] * block_size + pos % block_size`.
# Write `slot_mapping(block_table, start, n, block_size)` for the `n` tokens at positions `start .. start+n-1`.

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
# A request with a `P`-token prompt that generates `O` tokens: how many blocks does it hold at its peak?
# Careful — the **last** sampled token is returned to the user but never fed back through the model, so it
# never gets a K/V slot. Write `peak_blocks(P, O, block_size)`.

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
# Write both step counters for decode-only requests with the given output lengths (FCFS order):
#
# * `static_steps(lens, slots)` — batches of `slots` requests in order; each batch takes as long as its longest;
# * `continuous_steps(lens, slots)` — each slot takes the next waiting request the moment it frees.

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
# Llama-3.1-8B in bf16 on one 24 GB L4 with `gpu_memory_utilization=0.9` leaves `perf.kv_cache_blocks(...)`
# blocks of 16 tokens for the KV cache. Chat requests average 1,000 prompt + 200 output tokens. How many can be
# resident at once? Set `concurrent` (use your `peak_blocks`).

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
# ## In a design review
# **The two-minute version.** "An engine is a loop around one forward pass. Each step the scheduler hands out a
# token budget: running requests first — one token each if they are decoding, a chunk if they are still
# prefilling — then new requests while budget, sequence slots and KV blocks last. All scheduled tokens are
# flattened into one batch, so the weights are read once for everyone; attention reads each request's history
# through its block table. After the pass we sample for requests whose prompt is done, and finished requests
# free their blocks immediately, so a waiting request joins at the next step — that is continuous batching,
# worth several-fold throughput over static batching on long-tailed output lengths. Concurrency is capped by
# KV blocks, not by compute: for an 8B model on an L4 it is about 28 chat requests."
#
# **Drill questions**
# 1. *Why does the engine not have prefill steps and decode steps?* — Because the scheduler only tracks
#    computed vs total tokens; a step mixes decode tokens and prefill chunks under one token budget.
# 2. *Why is batching nearly free during decode?* — Decode is memory-bound: the step time is the weight read,
#    which is shared by every request in the batch (Notebook 02 puts numbers on it).
# 3. *A request's prompt is 8 tokens and it generates 1 token with block size 4. How many blocks?* — Two: the
#    only sampled token is never fed back, so it needs no slot.
