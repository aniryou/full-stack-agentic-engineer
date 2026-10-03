# %% [markdown]
# # 03 · Prefix caching
#
# **Tier:** T0. It needs a CPU only and no network, and it runs in a few seconds. `vllm-serving-lab` notebook
# `04_prefix_caching_for_agents` (T1) measures the same effect on a real vLLM server. It also reads the
# `vllm:prefix_cache_queries` / `vllm:prefix_cache_hits` counters of that server.
#
# ## The one-minute version
# Two requests whose prompts start with the same tokens compute the same K/V for that prefix. Thus compute it one
# time. The engine gives each **full** KV block a name: `hash(parent block's name, the block's tokens, extra keys)`.
# The parent is part of the name. Thus a name identifies the *entire* prefix, not only the 16 tokens inside the
# block.
#
# A new request goes along the chain of names of its prompt. It adopts each block that is already in the cache
# (refcount + 1), and it does not recompute that block. Thus its TTFT decreases, and it uses no new memory. When a
# request finishes, its blocks go, **still named**, to an LRU free queue. Thus they continue to give hits until the
# engine actually needs the memory.
#
# Agents have long stable system prompts, tool schemas and append-only histories. For agents, prefix caching often
# gives the single largest gain in the engine. The prompt layout decides if you get it.
#
# Primer: §5 *Prefix caching*, §4 *KV cache management revisited* (`../../PRIMER.md`). Background:
# `04-inference-engine/paged-attention/` (block tables, copy-on-write).

# %%
import math

from minengine import Engine, SamplingParams, TinyLM, block_hashes, encode, hash_block, perf
from minengine.kv import KVCacheManager

model = TinyLM()
greedy = SamplingParams(max_tokens=8, temperature=0)

# %% [markdown]
# ## Worked example 1 — block names form a chain

# %%
a = encode("SYSTEM: be brief. Q1")
b = encode("SYSTEM: be brief. Q2")
c = encode("xxxxxxx: be brief. Q2")                  # same tokens from block 2 on, different start
for name, toks in [("a", a), ("b", b), ("c", c)]:
    print(name, [h.hex()[:6] for h in block_hashes(toks, 4)])

# %% [markdown]
# `a` and `b` share their first four names (the first 16 tokens are identical). They become different at the block
# that holds `Q1` / `Q2`. `c` has the same tokens as `b` in blocks 2 and 3 (`" be "`, `"brie"`), but it has
# **different names**. Its first block is different, and the name of each later block depends on it. The K/V of a
# block depend on each token before it (attention), so the name must depend on those tokens too. The partial last
# block has no name, because the engine caches only full blocks.
#
# ## Worked example 2 — a shared system prompt

# %%
SYSTEM = ("You are a support agent for an online shop. Tools: search_orders(query), get_order(id), "
          "refund(order_id, reason), escalate(summary). Always check the order before a refund. Be brief.\n")
eng = Engine(model, num_blocks=128, block_size=16, max_num_batched_tokens=512)
for q in ["User: where is my order?", "User: I want a refund.", "User: my parcel is damaged."]:
    rid = eng.add_request(SYSTEM + q, greedy)
    eng.step()                                            # admit it (and prefill it) before the next arrives
    print(f"{rid}: {len(encode(SYSTEM + q))} prompt tokens, {eng.requests[rid].num_cached_tokens} from cache")
print("\nblock tables (id:'contents' x refcount, * = published):")
for rid in ["r0", "r1", "r2"]:
    print(rid, eng.blocks(rid)[:120], "...")
print("\nstats:", eng.kv.stats)

# %% [markdown]
# `r1` and `r2` adopt `r0`'s system-prompt blocks. The same physical block ids are in all three tables with a
# refcount of 3. `r1` and `r2` prefill only their own question. The engine counts hits in tokens. That is exactly
# what vLLM reports as `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries`.
#
# ## Worked example 3 — a burst: three requests in the same step
# An agent sends out three tool calls at once (a fan-out), all behind the same system prompt. Thus the scheduler
# admits all three in **one** step. The engine publishes a block when the scheduler **schedules** the tokens that
# fill it (vLLM does it in `allocate_slots`). Thus the second and third requests adopt `r0`'s blocks while `r0` still
# computes them.

# %%
eng = Engine(model, num_blocks=128, block_size=16, max_num_batched_tokens=1024)
for q in ["User: where is my order?", "User: I want a refund.", "User: my parcel is damaged."]:
    eng.add_request(SYSTEM + q, greedy)
eng.step()
print(eng.trace())
print("from cache:", [eng.requests[r].num_cached_tokens for r in ["r0", "r1", "r2"]], "| blocks in use:",
      eng.kv.num_blocks - eng.kv.num_free_blocks)

# %% [markdown]
# In one step, `r0` prefills its whole prompt, and `r1` and `r2` prefill only the tokens after token 176. That is
# safe, because the forward pass writes the K/V of each layer for the *whole* step before a request attends at that
# layer. The model code does `cache.write(...)` for all tokens, then the per-request attention.
#
# It also explains a rule of the scheduler. The scheduler publishes the blocks of a request that runs only when it
# can no longer preempt a request in that step. A request that the scheduler removes from the batch must not publish blocks that it will
# never compute. If an engine publishes blocks only after the step, a burst computes the shared prefix one time per
# request. That engine also holds one copy per request.
#
# ## Worked example 4 — freed blocks keep producing hits, until memory is needed

# %%
eng = Engine(model, num_blocks=24, block_size=16, max_num_batched_tokens=512)
print("one at a time, each finishing before the next:")
for q in ["User: hi.", "User: hello?", "User: are you there?"]:
    o = eng.generate([SYSTEM + q], greedy)[0]
    print(f"   {o.num_cached_tokens:3d} of {len(encode(SYSTEM + q))} tokens from cache; blocks in use afterwards: "
          f"{eng.kv.num_blocks - eng.kv.num_free_blocks}")
filler = eng.generate(["Z" * 330], greedy)[0]            # needs 21 of the 24 blocks: cached blocks get evicted
print("after a big unrelated request, evictions so far:", eng.kv.stats.evictions)
o = eng.generate([SYSTEM + "User: back again."], greedy)[0]
print(f"   {o.num_cached_tokens} tokens from cache now")

# %% [markdown]
# The blocks of a finished request stay in the free queue with their names. They count as **free**: the allocator
# can take them, and `vllm:kv_cache_usage_perc` does not count them. But they still give hits until a new allocation
# takes them from the front of the queue and **evicts** the name. Prefix caching costs no memory. It uses only
# memory that no other request needs yet.
#
# Note *which* 32 tokens stayed in the cache after the large request: the **head** of the system prompt. Requests
# free their blocks tail first, so the most widely shared blocks are the last to go.
#
# ## Worked example 5 — why the parent must be in the name
# Replace the hash with a hash that ignores the parent. Two prompts can have the same tokens in their second block
# after different first blocks. Now they share a name, and the engine serves K/V computed under the incorrect
# prefix. The greedy tokens of this small model almost do not depend on context. Thus compare what the engine
# *computed*: the logprob of each generated token, against the dense reference.

# %%
from minengine.sampler import log_softmax


def logprob_error(out, prompt):
    """Largest |engine logprob - reference logprob| over the generated tokens."""
    p = encode(prompt)
    ref = model.forward_dense(p + out.token_ids)
    return max(abs(lp - log_softmax(ref[len(p) - 1 + i])[t])
               for i, ((lp, _), t) in enumerate(zip(out.logprobs, out.token_ids)))


parentless = lambda parent, tokens, extra=None: hash_block(None, tokens, extra)
prompts = ["BBBBxxxx", "AAAAthe engine", "BBBBthe engine"]   # the 3rd shares block 0 with the 1st, block 1 with the 2nd
for label, fn in [("chained   ", hash_block), ("parentless", parentless)]:
    eng = Engine(model, num_blocks=32, block_size=4, hash_fn=fn)
    outs = [eng.generate([p], greedy)[0] for p in prompts]
    print(f"{label}: 3rd request reused {outs[2].num_cached_tokens:2d} tokens, "
          f"logprob error vs reference {logprob_error(outs[2], prompts[2]):.1e}")

# %% [markdown]
# The chained cache uses again only the block whose whole prefix is the same. Its result agrees with the reference
# to rounding error. The parentless cache uses more blocks again, and it computes different probabilities. There is
# no error and no crash, only outputs that are subtly incorrect (with a real model, text that is clearly incorrect).
#
# That is why engines use a chained, collision-resistant hash (vLLM defaults to SHA-256 as of Sep 2026, verify). It
# is also why tenant isolation adds a salt to the root of the chain (`cache_salt` in vLLM). Without the salt, a
# timing side channel can show if someone else sent a prefix.
#
# ## Worked example 6 — what it buys on a real GPU (SIMULATED)
# The workload is 60 requests on H100 + Llama-3.1-8B. Their 2,000–2,200-token prompts share their first 1,800
# tokens (a long system prompt with tool schemas). They go through `perf.simulate`: the scheduler and the KV manager
# of this package on a roofline clock. In the first run, they arrive at 6/s. In the second run, they arrive all at
# once.

# %%
g, m = perf.GPUS["H100-SXM"], perf.LLMS["llama-3.1-8b"]
for rate in [6, math.inf]:
    w = perf.Workload(n_requests=60, rate=rate, prompt_len=(2000, 2200), output_len=(40, 80), shared_prefix=1800)
    for caching in [True, False]:
        label = f"{'6/s' if rate == 6 else 'burst'}, caching {'on' if caching else 'off'}"
        print(perf.simulate(g, m, w, enable_prefix_caching=caching, label=label).summary())

# %% [markdown]
# At 6/s, each request after the first skips 1,792 of its ~2,100 prompt tokens. TTFT p50 decreases by about 6×. The
# engine holds the shared blocks one time, not 60 times, so the peak KV use decreases too.
#
# The burst shows an even larger gain. Without caching, 60 copies of the same prefill wait in the queue behind
# one another. The hit rate (84%) is the value that `vllm:prefix_cache_hits / vllm:prefix_cache_queries` gives. The
# inputs are assumptions, so the outputs are estimates. The lab measures the real thing (`vllm-serving-lab`,
# notebook `04_prefix_caching_for_agents`).
#
# ## Exercise 3.1 — build the chain
# Write `chain_names(tokens, block_size)`. It returns the names of the **full** blocks. Each name is
# `hash_block(parent, block_tokens)`, with `parent=None` for the first block.

# %% exercise
def chain_names(tokens, block_size):
    ### BEGIN SOLUTION
    names, parent = [], None
    for i in range(len(tokens) // block_size):
        parent = hash_block(parent, tokens[i * block_size:(i + 1) * block_size])
        names.append(parent)
    return names
    ### END SOLUTION

# %% check
for t in [a, b, c, encode(SYSTEM)]:
    assert chain_names(t, 4) == block_hashes(t, 4)
assert len(chain_names(list(range(15)), 4)) == 3              # the partial block has no name
print("✅ chain_names == block_hashes")

# %% [markdown]
# ## Exercise 3.2 — predict the hit
# A previous request with the prompt `cached` finished, and all its full blocks are still in the cache. A new prompt
# `new` arrives. How many of its tokens come from the cache? Remember two rules:
#
# * Only **full** blocks of the common prefix can hit.
# * The engine always recomputes the last token of `new`, because it needs the logits of that token. Thus at most
#   `(len(new) - 1) // block_size` blocks hit.

# %% exercise
def predicted_hit_tokens(cached, new, block_size):
    ### BEGIN SOLUTION
    common = 0
    while common < min(len(cached), len(new)) and cached[common] == new[common]:
        common += 1
    blocks = min(common // block_size, (len(cached)) // block_size, (len(new) - 1) // block_size)
    return blocks * block_size
    ### END SOLUTION

# %% check
def measured(cached, new, B=4):
    eng = Engine(model, num_blocks=64, block_size=B)
    eng.generate([cached], SamplingParams(max_tokens=1))
    return eng.generate([new], SamplingParams(max_tokens=1))[0].num_cached_tokens
cases = [(a, b), (a, c), (a, a), (encode("0123456789ab"), encode("0123456789abc")), (encode("abcdefgh"), encode("abcdefg"))]
for x, y in cases:
    assert predicted_hit_tokens(x, y, 4) == measured(x, y), (x, y)
print("✅ identical 20-token prompts hit", predicted_hit_tokens(a, a, 4), "tokens, not 20: the last block is recomputed")

# %% [markdown]
# ## Exercise 3.3 — which block is evicted first?
# The free queue is LRU. The engine adds a block at the end of the queue when its refcount decreases to zero, and it
# evicts blocks from the front. A request that finishes frees its blocks **tail first**. Request `a` (blocks
# `a0 a1 a2`, head to tail) finishes, then `b` (`b0 b1`) finishes. Then the engine allocates five new blocks from a
# free queue that has no other blocks. Fill `eviction_order` with the order in which the engine uses those five
# blocks again.

# %% exercise
### BEGIN SOLUTION
eviction_order = ["a2", "a1", "a0", "b1", "b0"]
### END SOLUTION

# %% check
kv = KVCacheManager(5, block_size=2)
names = {}
for rid, toks in [("a", [1, 2, 3, 4, 5, 6]), ("b", [7, 8, 9, 10])]:
    kv.allocate_slots(rid, toks, 0, len(toks))
    kv.cache_blocks(rid, toks, len(toks))
    names.update({bid: f"{rid}{i}" for i, bid in enumerate(kv.tables[rid])})
kv.free("a")
kv.free("b")
kv.allocate_slots("new", list(range(100, 110)), 0, 10)
assert [names[b] for b in kv.tables["new"]] == eviction_order
print("✅ least recently freed first; within a request, the tail first - the shared head survives longest")

# %% [markdown]
# ## Exercise 3.4 — lay out an agent prompt for the cache
# An agent session has four turns. The prompt for each turn must include the current time. Write
# `append_only(transcript, user_msg, clock)` so that the prompt of each turn **extends** the prompt of the previous
# turn plus the reply of the agent. Keep each message, with its own timestamp, exactly as the user sent it.
# `transcript` is a list of `(user_msg, clock, reply)`. Compare with `clock_first`, which puts the time at the start
# of the prompt.

# %%
TURNS = ["Where is my order?", "It was order 4411.", "Can I get a refund?", "Thanks, that is all."]


def clock_first(transcript, user_msg, clock):
    history = "".join(f"User: {u}\nAgent:{r}\n" for u, c, r in transcript)
    return f"[now {clock}] " + SYSTEM + history + f"User: {user_msg}\nAgent:"


def run_session(build):
    eng = Engine(model, num_blocks=256, block_size=16, max_num_batched_tokens=512)
    transcript, hits = [], []
    for i, u in enumerate(TURNS):
        prompt = build(transcript, u, f"10:0{i}")
        out = eng.generate([prompt], SamplingParams(max_tokens=16, seed=i))[0]
        hits.append(out.num_cached_tokens / len(encode(prompt)))
        transcript.append((u, f"10:0{i}", out.text))
    return hits

# %% exercise
def append_only(transcript, user_msg, clock):
    ### BEGIN SOLUTION
    history = "".join(f"User ({c}): {u}\nAgent:{r}\n" for u, c, r in transcript)
    return SYSTEM + history + f"User ({clock}): {user_msg}\nAgent:"
    ### END SOLUTION

# %% check
bad, good = run_session(clock_first), run_session(append_only)
print("hit rate per turn, clock first :", [f"{h:.0%}" for h in bad])
print("hit rate per turn, append-only :", [f"{h:.0%}" for h in good])
assert max(bad) < 0.1 and min(good[1:]) > 0.8
print("✅ stable content first, volatile content last, history append-only")

# %% [markdown]
# ## Exercise 3.5 — block hashing vs a radix tree
# SGLang's RadixAttention keeps cached prefixes in a radix tree at **token** granularity. A block-hash cache uses
# only whole blocks again. Write `radix_reuse(cached, new)`. It returns the tokens that a token-granular cache uses
# again: the whole common prefix, but still at most `len(new) - 1`. The check compares it with block reuse at 16
# tokens per block.

# %% exercise
def radix_reuse(cached, new):
    ### BEGIN SOLUTION
    common = 0
    while common < min(len(cached), len(new)) and cached[common] == new[common]:
        common += 1
    return min(common, len(new) - 1)
    ### END SOLUTION

# %% check
pairs = [(encode(SYSTEM + "User: hi"), encode(SYSTEM + "User: hello")),     # SYSTEM is 183 tokens
         (encode(SYSTEM), encode(SYSTEM + "User: a new question")),
         (encode("ab" * 40), encode("ab" * 39 + "x"))]
extra = [radix_reuse(x, y) - predicted_hit_tokens(x, y, 16) for x, y in pairs]
assert [radix_reuse(x, y) for x, y in pairs] == [190, 183, 78] and extra == [14, 7, 14]
print(f"✅ the radix tree reuses {extra} extra tokens per pair: always less than one block per request")

# %% [markdown]
# The difference is less than a block per request. Thus the choice between the two structures is rarely about hit
# rate. It is about eviction policy and the scheduler. A radix tree makes the question "which cached prefix does
# this request extend?" low-cost to ask. SGLang uses this to put requests in an order for cache locality. Block
# hashes make the cache a flat dictionary that is simple to share, offload and publish as events (a cache-aware
# router uses these events, 05).
#
# ## In a design review
# **The two-minute version.** "The engine gives each full KV block a name: a hash of the name of its parent plus its
# tokens. Thus a name identifies the whole prefix. A new request goes along the chain of its prompt and adopts each
# cached block. The refcount increases, and there is no compute and no new memory. The engine prefills only the
# suffix of the request.
#
# "Finished requests leave their blocks in an LRU free queue, still named, so hits continue until the engine needs
# the memory. The engine evicts tails first, so shared heads stay in the cache.
#
# "For our agents, that means a stable system prompt and tool list at the top and an append-only transcript.
# Anything volatile goes at the end: timestamps and per-request IDs. That layout took our hit rate from near 0% to
# over 80%. We monitor `prefix_cache_hits / prefix_cache_queries`. Across replicas, the router must send a session
# back to the replica that holds its prefix (05)."
#
# **Drill questions**
# 1. *Why include the parent in the block hash?* K/V depend on all earlier tokens. Without the parent, equal blocks
#    after different prefixes collide. Then the engine serves incorrect K/V and gives no warning.
# 2. *Two identical 20-token prompts, block size 4: how many tokens hit?* 16. The engine recomputes the last block,
#    because it needs the logits of the last token.
# 3. *Does prefix caching reduce the memory available to other requests?* No. Cached blocks with no users are free
#    (evictable). The engine uses them again only when nobody needs the memory.
