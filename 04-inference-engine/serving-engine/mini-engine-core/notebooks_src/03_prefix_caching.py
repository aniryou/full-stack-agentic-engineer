# %% [markdown]
# # 03 · Prefix caching
#
# **Tier:** T0 — CPU only, no network, a few seconds. The same effect measured on a real vLLM server (and on
# its `vllm:prefix_cache_queries` / `vllm:prefix_cache_hits` counters) is `vllm-serving-lab` notebook
# `04_prefix_caching_for_agents` (T1).
#
# ## The one-minute version
# Two requests whose prompts start with the same tokens compute the same K/V for that prefix — so compute it once.
# The engine names every **full** KV block by `hash(parent block's name, the block's tokens, extra keys)`. Because
# the parent is part of the name, a name commits to the *entire* prefix, not just the 16 tokens inside. A new
# request walks its prompt's chain of names and adopts every block already cached (refcount + 1) instead of
# recomputing it: its TTFT drops and no new memory is used. When a request finishes, its blocks go to an LRU free
# queue **still named**, so they keep producing hits until memory is actually needed. For agents — long stable
# system prompts, tool schemas, append-only histories — this is often the single largest saving in the engine,
# and prompt layout decides whether you get it.
#
# Primer: §5 *Prefix caching*; §4 *KV cache management revisited* (`../../PRIMER.md`). Background:
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
# `a` and `b` share their first four names (the first 16 tokens are identical), and diverge at the block holding
# `Q1` / `Q2`. `c` has the same tokens as `b` in blocks 2 and 3 — `" be "`, `"brie"` — yet **different names**:
# its first block differs, and the name of every later block depends on it. A block's K/V depends on every token
# before it (attention), so the name must too. The partial last block has no name: only full blocks are cached.
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
# `r1` and `r2` adopt `r0`'s system-prompt blocks: the same physical block ids appear in all three tables with a
# refcount of 3. They only prefill their own question. Hits are counted in tokens, which is exactly what vLLM
# reports as `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries`.
#
# ## Worked example 3 — a burst: three requests in the same step
# An agent fans out three tool calls at once, all behind the same system prompt, so all three are admitted in
# **one** step. A block is published when the scheduler **schedules** the tokens that fill it (vLLM does it in
# `allocate_slots`), so the second and third requests adopt `r0`'s blocks while `r0` is still computing them.

# %%
eng = Engine(model, num_blocks=128, block_size=16, max_num_batched_tokens=1024)
for q in ["User: where is my order?", "User: I want a refund.", "User: my parcel is damaged."]:
    eng.add_request(SYSTEM + q, greedy)
eng.step()
print(eng.trace())
print("from cache:", [eng.requests[r].num_cached_tokens for r in ["r0", "r1", "r2"]], "| blocks in use:",
      eng.kv.num_blocks - eng.kv.num_free_blocks)

# %% [markdown]
# One step: `r0` prefills its whole prompt, `r1` and `r2` only what follows token 176. That is safe because the
# forward pass writes each layer's K/V for the *whole* step before any request attends at that layer — the model
# code does `cache.write(...)` for all tokens, then the per-request attention. It is also why the scheduler
# publishes a running request's blocks only once no request can be preempted in that step any more: a request
# removed from the batch must not publish blocks it will never compute. An engine that published blocks only
# after the step would make a burst compute the shared prefix once per request, and hold one copy per request.
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
# A finished request's blocks sit in the free queue with their names. They count as **free** (the allocator may
# take them: `vllm:kv_cache_usage_perc` does not count them) yet they are still hits — until a new allocation
# pops them from the front of the queue and **evicts** the name. Prefix caching costs no memory; it only uses
# memory nobody else needs yet. Note *which* 32 tokens survived the big request: the **head** of the system
# prompt. Requests free their blocks tail first, so the most widely shared blocks are the last to go.
#
# ## Worked example 5 — why the parent must be in the name
# Replace the hash with one that ignores the parent. Two prompts with the same tokens in their second block —
# after different first blocks — now share a name, and the engine serves K/V computed under the wrong prefix.
# This tiny model's greedy tokens barely depend on context, so compare what the engine *computed*: the logprob of
# every generated token, against the dense reference.

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
# The chained cache reuses only the block whose whole prefix matches and agrees with the reference to rounding
# error. The parentless one reuses more — and computes different probabilities. No error, no crash: just subtly
# wrong outputs (with a real model, visibly wrong text). That is why engines use a chained, collision-resistant hash
# (vLLM defaults to SHA-256 as of Sep 2026, verify) and why tenant isolation adds a salt to the chain's root
# (`cache_salt` in vLLM): without it, a timing side channel could reveal whether someone else sent a prefix.
#
# ## Worked example 6 — what it buys on a real GPU (SIMULATED)
# 60 requests whose 2,000–2,200-token prompts share their first 1,800 tokens (a long system prompt with tool
# schemas), H100 + Llama-3.1-8B, through `perf.simulate` — this package's scheduler and KV manager on a roofline
# clock. First arriving at 6/s, then all at once.

# %%
g, m = perf.GPUS["H100-SXM"], perf.LLMS["llama-3.1-8b"]
for rate in [6, math.inf]:
    w = perf.Workload(n_requests=60, rate=rate, prompt_len=(2000, 2200), output_len=(40, 80), shared_prefix=1800)
    for caching in [True, False]:
        label = f"{'6/s' if rate == 6 else 'burst'}, caching {'on' if caching else 'off'}"
        print(perf.simulate(g, m, w, enable_prefix_caching=caching, label=label).summary())

# %% [markdown]
# At 6/s every request after the first skips 1,792 of its ~2,100 prompt tokens: TTFT p50 drops about 6×, and
# because the shared blocks are held once, not 60 times, peak KV use falls too. In the burst the saving is larger
# still: without caching, 60 copies of the same prefill queue behind one another. The hit rate (84%) is what
# `vllm:prefix_cache_hits / vllm:prefix_cache_queries` would show. Assumptions in, estimates out — the lab
# measures the real thing (`vllm-serving-lab`, notebook `04_prefix_caching_for_agents`).
#
# ## Exercise 3.1 — build the chain
# Write `chain_names(tokens, block_size)`: the names of the **full** blocks, each `hash_block(parent, block_tokens)`
# with `parent=None` for the first block.

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
# A previous request with prompt `cached` finished and all its full blocks are still cached. A new prompt `new`
# arrives. How many of its tokens come from the cache? Remember two rules: only **full** blocks of the common
# prefix can hit, and the last token of `new` is always recomputed (its logits are needed), so at most
# `(len(new) - 1) // block_size` blocks hit.

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
# The free queue is LRU: blocks are appended when their refcount drops to zero and evicted from the front. A
# finishing request frees its blocks **tail first**. Request `a` (blocks `a0 a1 a2`, head to tail) finishes, then
# `b` (`b0 b1`). Five new blocks are then allocated from an otherwise empty free queue. Fill `eviction_order` with
# the order in which those five blocks are reused.

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
# `append_only(transcript, user_msg, clock)` so that each turn's prompt **extends** the previous turn's prompt
# plus the agent's reply — keep every message, with its own timestamp, exactly as it was sent. `transcript` is a
# list of `(user_msg, clock, reply)`. Compare with `clock_first`, which puts the time at the very top.

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
# SGLang's RadixAttention keeps cached prefixes in a radix tree at **token** granularity; a block-hash cache
# reuses only whole blocks. Write `radix_reuse(cached, new)`: the tokens a token-granular cache reuses — the whole
# common prefix, but still at most `len(new) - 1`. The check compares it with block reuse at 16 tokens per block.

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
# Less than a block per request — so the choice between the two structures is rarely about hit rate. It is about
# eviction policy and scheduling: a radix tree makes "which cached prefix does this request extend?" cheap to
# ask, which SGLang uses to order requests for cache locality; block hashes make the cache a flat dictionary that
# is trivial to share, offload and publish as events (what a cache-aware router consumes, 05).
#
# ## In a design review
# **The two-minute version.** "The engine names each full KV block by a hash of its parent's name plus its tokens,
# so a name identifies the whole prefix. A new request walks its prompt's chain and adopts every cached block —
# refcount up, no compute, no new memory; only its suffix is prefilled. Finished requests leave their blocks in an
# LRU free queue, still named, so hits continue until the memory is needed; tails are evicted first so shared
# heads survive. For our agents that means a stable system prompt and tool list at the top, an append-only
# transcript, and anything volatile — timestamps, per-request IDs — at the end: that layout took our hit rate
# from near 0% to over 80%. We watch `prefix_cache_hits / prefix_cache_queries`, and across replicas the router
# must send a session back to the replica that holds its prefix (05)."
#
# **Drill questions**
# 1. *Why include the parent in the block hash?* — K/V depend on all earlier tokens; without the parent, equal
#    blocks after different prefixes collide and the engine silently serves wrong K/V.
# 2. *Two identical 20-token prompts, block size 4 — how many tokens hit?* — 16: the last block is recomputed
#    because the last token's logits are needed.
# 3. *Does prefix caching reduce the memory available to other requests?* — No: cached blocks with no users are
#    free (evictable); they are reused only when nobody needs the memory.
