# %% [markdown]
# # 04 · Prefix caching for agents: measure the hit rate, then design prompts that earn it
#
# **Tier:** T0. The fake vLLM implements the block-hash prefix cache of vLLM, and it gives **simulated** results.
# T1/T3: point `SERVELAB_URL` at a real vLLM. In both cases, the hit rate comes from its `/metrics`. The cached
# tokens of each request appear when the server runs with `--enable-prompt-tokens-details`. This is a
# `vllm serve` flag in v0.30.0, and this statement agrees with its source.
#
# ## The one-minute version
#
# vLLM gives each full KV block (16 tokens) a key: a hash of **its tokens and the hash of the block
# before it**. A new request reuses the longest series of blocks from the start of the prompt with hashes that
# are already in the cache. It computes only the remainder. Thus, the prefill work and TTFT decrease in
# proportion. Two consequences decide everything for agents:
#
# * A hit needs an **identical prefix from token 0**. One changed token near the start (a timestamp in the
#   system prompt, a reordered tool list) makes every block after it invalid.
# * An agent conversation is **append-only**. Thus, turn $n+1$ can reuse all of turn $n$: the prompt,
#   the tool results and the reply of the model. The condition is that the client sends them again byte for byte.
#
# The engine counts the hits: `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` (tokens).
# Concepts: PRIMER §5 "Prefix caching" ([`PRIMER.md`](../../PRIMER.md)). For paging and for blocks that requests share, see
# [`04-inference-engine/paged-attention`](../../../paged-attention/paged-attention-primer.md).
# [`07-application-agent-framework`](../../../../07-application-agent-framework/) builds the agent loops that make
# these prompts: long stable prefixes, and tool results that the loop appends turn after turn.

# %%
import math
from servelab import env, metrics as M, textgen
from servelab.bench import Request, agent_sessions, run_sessions, run_sync
from servelab.bench.client import stream_request
from servelab.bench.runner import _session
from servelab.fake_engine import EngineConfig, FakeEngine, block_hash, profile as engine_profile, simulate, tiny_profile

target = env.connect("t4-qwen2.5-0.5b")
URL, H = target.url, target.headers
LABEL = "SIMULATED" if target.simulated else "MEASURED"
print(target, "|", LABEL)

async def ask(messages, max_tokens=16):
    async with _session() as http:
        from servelab.bench.client import discover_model
        model = await discover_model(http, URL, H)
        return await stream_request(http, URL, model, Request(messages=messages, max_tokens=max_tokens), headers=H)

# %% [markdown]
# ## Worked example: the same 2,000-token system prompt, twice

# %%
SYSTEM = "You are a claims assistant. " + textgen.synthetic_text(2000, 11)
q1 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Where is claim 1042?"}]
q2 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Cancel order 77, please."}]
cold, warm = run_sync(ask(q1)), run_sync(ask(q2))
for name, r in (("first ", cold), ("second", warm)):
    print(f"[{LABEL}] {name}: prompt {r.prompt_tokens} tokens, cached {r.cached_tokens}, TTFT {r.ttft * 1e3:6.1f} ms")

# %% [markdown]
# ## Worked example: the hash chain, block by block
#
# Hash each 16-token block together with the hash of the previous block. Compare the first prompt with
# two other prompts:
#
# - (a) the second prompt, which is different only in the final question,
# - (b) a copy with one changed word in the first sentence of the system prompt.

# %%
def chain(tokens, bs=16):
    out, h = [], None
    for i in range(len(tokens) // bs):
        h = block_hash(h, tuple(tokens[i * bs:(i + 1) * bs]))
        out.append(h)
    return out

def first_diff(x, y):
    return next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))

def reusable_blocks(x, y, bs=16):
    hx, hy = chain(x, bs), chain(y, bs)
    return next((i for i, (u, v) in enumerate(zip(hx, hy)) if u != v), min(len(hx), len(hy)))

q3 = [{"role": "system", "content": SYSTEM.replace("claims", "orders", 1)}, q1[1]]
for name, other in (("(a) different final question", q2), ("(b) one word changed on line 1", q3)):
    x, y = textgen.chat_tokens(q1), textgen.chat_tokens(other)
    same = sum(u == v for u, v in zip(x, y))
    print(f"{name}: {same:,}/{len(x):,} tokens identical, first difference at token {first_diff(x, y):,}, "
          f"reusable full blocks {reusable_blocks(x, y)} of {len(x) // 16}")

# %% [markdown]
# ## Exercise 4.1 — how many tokens will hit?
#
# The previous request processed `prev` (its prompt *and* the generated tokens). Write
# `expected_cached_tokens(prev, new, block_size)` for a request `new` that arrives immediately after it.
# Use the three rules of vLLM:
#
# - vLLM caches only **full** blocks.
# - vLLM never computed the KV of the last token of `prev` (it sampled the token, but did not feed it back).
#   Thus, `prev` left `(len(prev) - 1) // block_size` blocks.
# - vLLM always recomputes at least one token of `new` to get logits. Thus, at most
#   `(len(new) - 1) // block_size` blocks can hit.
#
# Within those limits, the hits are the full blocks of the common prefix.

# %% exercise
def expected_cached_tokens(prev: list, new: list, block_size: int = 16) -> int:
    ### BEGIN SOLUTION
    common = 0
    for x, y in zip(prev, new):
        if x != y:
            break
        common += 1
    blocks = min(common // block_size, (len(prev) - 1) // block_size, (len(new) - 1) // block_size)
    return blocks * block_size
    ### END SOLUTION

# %% check
assert expected_cached_tokens(list(range(64)), list(range(64)), 16) == 48          # identical: last token recomputed
assert expected_cached_tokens(list(range(40)), list(range(40)) + [7] * 30, 16) == 32  # prev left 2 full blocks
assert expected_cached_tokens([1] * 20 + [2] * 20, [1] * 20 + [3] * 20, 8) == 16      # diverges inside block 3
for bs in (4, 16):                                       # against the engine itself: a two-turn conversation
    eng = FakeEngine(tiny_profile(num_blocks=512, block_size=bs, max_model_len=4096))
    (t1,) = simulate(eng, [(0.0, textgen.chat_tokens(q1)[:300], 10)])
    turn2 = t1.prompt + t1.output + [7, 8, 9] * 5
    (t2,) = simulate(eng, [(1.0, turn2, 5)])
    assert t2.cached_tokens == expected_cached_tokens(t1.prompt + t1.output, turn2, bs), (bs, t2.cached_tokens)
print("✅ expected_cached_tokens reproduces the engine's hits, including the model's own previous reply")

# %% [markdown]
# ## Exercise 4.2 — the hit rate from two scrapes
#
# Counters are cumulative. The hit rate of a window is the ratio of the counter *increases*. Write
# `hit_rate(before, after)` from two parsed scrapes. (`scrape.value(name)` sums a metric. The names are
# `M.PREFIX_HITS` and `M.PREFIX_QUERIES`.) If there were no queries, return `nan`.

# %% exercise
def hit_rate(before, after) -> float:
    ### BEGIN SOLUTION
    q = after.value(M.PREFIX_QUERIES, 0.0) - before.value(M.PREFIX_QUERIES, 0.0)
    h = after.value(M.PREFIX_HITS, 0.0) - before.value(M.PREFIX_HITS, 0.0)
    return h / q if q > 0 else math.nan
    ### END SOLUTION

# %% check
s0 = M.scrape(URL, headers=H)
run_sync(ask(q1))
s1 = M.scrape(URL, headers=H)
assert math.isclose(hit_rate(s0, s1), M.snapshot(s1, s0).prefix_hit_rate)
assert math.isnan(hit_rate(s1, s1))
print(f"✅ [{LABEL}] the repeated request hit {hit_rate(s0, s1):.1%} of its prompt tokens in the cache")

# %% [markdown]
# ## Exercise 4.3 — three prompt layouts for the same agent, predicted then measured
#
# The workload has six agent sessions with three turns each. Each session has the request of the user, then two
# tool results. After every turn, the session appends the reply of the model. The content is the same in three
# layouts (see `agent_sessions`):
#
# - `"stable"`: a shared system prompt and tool list, and an append-only history.
# - `"shuffled_tools"`: each session lists the tools in its own order.
# - `"timestamp_first"`: a new timestamp on the first line of every request.
#
# Predict the hit rate of each layout *as a number* before you measure it.
#
# Write `predicted_hit_rate(requests, block_size)` for `requests = [(prompt_ids, output_ids), ...]` in
# arrival order. Each earlier request left its prompt *and* output in the cache. Assume that vLLM evicts
# nothing, because the pool is much larger than this workload. Thus, a request hits the longest prefix that it
# shares with any earlier request, under the rules of exercise 4.1 (reuse `expected_cached_tokens`).
# Return total hits / total prompt tokens. This is the value that `vllm:prefix_cache_hits / queries` measures.

# %% exercise
def predicted_hit_rate(requests: list, block_size: int = 16) -> float:
    ### BEGIN SOLUTION
    seen, hits, total = [], 0, 0
    for prompt, output in requests:
        hits += max((expected_cached_tokens(prev, prompt, block_size) for prev in seen), default=0)
        total += len(prompt)
        seen.append(list(prompt) + list(output))
    return hits / total if total else math.nan
    ### END SOLUTION

# %% check
assert predicted_hit_rate([([1] * 40, [2] * 5), ([1] * 40 + [3] * 8, [])], 16) == 32 / 88

def requests_of(sessions, run):
    """Rebuild what each request sent (toy token ids) and got back, in the order they were sent."""
    got = {(r.session, r.turn): r for r in run.results}
    out = []
    for s in sessions:
        history = []
        for k, content in enumerate(s.turns):
            history.append({"role": "user", "content": content})
            r = got[(s.sid, k)]
            out.append((r.start, textgen.chat_tokens([{"role": "system", "content": s.system(k)}] + history),
                         textgen.tokenize(r.text)))
            history.append({"role": "assistant", "content": r.text})
    return [(prompt, output) for _, prompt, output in sorted(out, key=lambda x: x[0])]

rates, predicted, ttft = {}, {}, {}
for i, layout in enumerate(("stable", "shuffled_tools", "timestamp_first")):
    sessions = agent_sessions(6, turns=3, layout=layout, seed=100 + i)
    before = M.scrape(URL, headers=H)
    run = run_sessions(URL, sessions, session_rate=3.0, headers=H)
    rates[layout] = hit_rate(before, M.scrape(URL, headers=H))
    predicted[layout] = predicted_hit_rate(requests_of(sessions, run))
    ttft[layout] = run.summary().ttft.mean
    print(f"[{LABEL}] {layout:16s} predicted {predicted[layout]:6.1%}  measured {rates[layout]:6.1%}   "
          f"mean TTFT {ttft[layout]:6.1f} ms")
assert predicted["stable"] > predicted["shuffled_tools"] > predicted["timestamp_first"] == 0.0
if target.simulated:        # at T1 the prediction uses toy token counts, the engine real ones: compare by eye
    assert all(abs(predicted[k] - rates[k]) < 0.05 for k in rates), (predicted, rates)
assert rates["timestamp_first"] < 0.05 and ttft["stable"] < ttft["timestamp_first"]
print("✅ the hit rate is predictable from the prompt layout alone: one timestamp on line 1 costs the whole cache; "
      "a reordered tool list costs the cross-session share")

# %% [markdown]
# ## Exercise 4.4 — fix the template
#
# The agent in the next cell renders its prompt with the time on the first line. It puts the tools in the
# order that the registry returns them. Write `render_cache_friendly(system, tools, history, new_message,
# now)`. It must keep every earlier token identical from turn to turn:
#
# - First, the stable system prompt with the tools in a **deterministic** order.
# - Then the history, exactly as the agent sent it before.
# - The dynamic facts (the time) go only on the **new** message.
#
# The caller appends the text that you return for the new message to the history. Thus, that text stays
# verbatim after that.

# %%
def render_naive(system, tools, history, new_message, now):
    sys_msg = {"role": "system", "content": f"Current time: {now}\n{system}\n" + "\n".join(tools)}
    return [sys_msg] + history + [{"role": "user", "content": new_message}]

# %% exercise
def render_cache_friendly(system, tools, history, new_message, now):
    ### BEGIN SOLUTION
    sys_msg = {"role": "system", "content": system + "\n" + "\n".join(sorted(tools))}
    return [sys_msg] + list(history) + [{"role": "user", "content": f"{new_message}\n(current time: {now})"}]
    ### END SOLUTION

# %% check
tools = [f"Tool t{i}: " + textgen.synthetic_text(80, 200 + i) for i in range(5)]
SYS = "You are an operations agent. " + textgen.synthetic_text(400, 5)
REPLY = " restarting now"                      # what the model generated in turn 1
def two_turns(render, tool_order_2):
    turn1 = render(SYS, tools, [], "Restart the ingest job.", "2026-09-26T10:00:00")
    history = turn1[1:] + [{"role": "assistant", "content": REPLY}]              # resent verbatim
    turn2 = render(SYS, tool_order_2, history, "Tool result: ok", "2026-09-26T10:00:07")
    processed = textgen.chat_tokens(turn1) + textgen.tokenize(REPLY)          # what the engine saw in turn 1
    return processed, textgen.chat_tokens(turn2)
prev, new = two_turns(render_cache_friendly, list(reversed(tools)))
assert expected_cached_tokens(prev, new) >= 0.9 * len(prev), "turn 2 should reuse (almost) all of turn 1"
prev_n, new_n = two_turns(render_naive, tools)
assert expected_cached_tokens(prev_n, new_n) == 0
print(f"✅ turn 2 reuses {expected_cached_tokens(prev, new)} of {len(prev)} tokens (naive template: "
      f"{expected_cached_tokens(prev_n, new_n)})")

# %% [markdown]
# ## Exercise 4.5 — what a hit is worth in TTFT
#
# Prefill is compute-bound. Thus, a hit saves approximately the prefill time of the cached tokens. Write
# `ttft_estimate(p, prompt_tokens, cached_tokens)` for an engine that has no other work. The estimate is one step
# that prefills only the uncached tokens (use `p.prefill_s(tokens, context=cached_tokens)`). Then predict
# the TTFT decrease for a 6,000-token agent prompt with 5,000 cached tokens. Compare the prediction with a
# measurement on this server.

# %% exercise
def ttft_estimate(p, prompt_tokens: int, cached_tokens: int) -> float:
    ### BEGIN SOLUTION
    return p.prefill_s(prompt_tokens - cached_tokens, context=cached_tokens)
    ### END SOLUTION

# %% check
P = engine_profile("t4-qwen2.5-0.5b", max_model_len=8192)
assert math.isclose(ttft_estimate(P, 6000, 0), P.prefill_s(6000))
cold_est, warm_est = ttft_estimate(P, 6000, 0), ttft_estimate(P, 6000, 5000)
print(f"model: cold {cold_est * 1e3:.0f} ms -> warm {warm_est * 1e3:.0f} ms ({cold_est / warm_est:.1f}x)")
if target.simulated:
    from servelab.fakeserver import FakeServer
    with FakeServer(P, EngineConfig(max_num_batched_tokens=8192)) as u:
        URL_BIG = u
        long_prompt = textgen.synthetic_text(5000, 21)
        async def ttft_of(text):
            async with _session() as http:
                return (await stream_request(http, URL_BIG, P.model, Request(prompt=text, max_tokens=2))).ttft
        c = run_sync(ttft_of(long_prompt + " " + textgen.synthetic_text(1000, 22)))
        w = run_sync(ttft_of(long_prompt + " " + textgen.synthetic_text(1000, 23)))
    print(f"[SIMULATED] measured: cold {c * 1e3:.0f} ms -> warm {w * 1e3:.0f} ms")
    assert abs(c / cold_est - 1) < 0.5 and w < c / 2
print("✅ a cached prefix turns a compute-bound prefill into a short one: TTFT tracks the *uncached* tokens")

# %%
target.stop()

# %% [markdown]
# ## Capacity: cached prefixes share the pool with running requests
#
# Cached blocks are in the same KV pool as the requests that the engine serves. A block stays cached only
# while it is free and vLLM did not allocate it again yet. (vLLM reallocates the least recently freed block
# first.)
#
# The shared prefix of an agent that uses tools (here ~1,500 tokens = ~94 blocks) is small next to the
# ~66,000 blocks of a T4. Thus, the prefix stays in the cache. Under load, vLLM evicts the per-session
# histories. If the router sends sessions to different replicas, each replica needs its own copy. This is why
# the routers in layer 05 send a session back to the replica that holds its prefix.
#
# ## In a design review
#
# **Two minutes:** "vLLM caches KV per 16-token block. The key is a hash chain through every earlier
# block. Thus, reuse needs an identical prefix from the first token. The system prompt and the tool schemas of
# our agent are ~1,500 tokens on every request, and the conversation only grows.
#
# "Thus, we put the stable parts of the prompt first: the system prompt, then the tools in a fixed order, then
# the history, appended verbatim. We attach timestamps and per-request facts to the newest message. We examine
# the result on the engine with `prefix_cache_hits / prefix_cache_queries` over a window. We expect this ratio to
# be near the fraction of each prompt that repeats.
#
# "A hit saves prefill compute on the cached tokens. Thus, TTFT follows the uncached tail. The first engineer who
# puts `datetime.now()` on line 1 of the system prompt stops this benefit for everyone."
#
# **Drill 1.** *The hit rate dropped from 80% to 3% after a release. Where do you look first?*
# The prompt template. Something dynamic moved into the prefix (a timestamp, a request id, a
# shuffled tool list, a changed chat template). Make a diff of the first blocks of two consecutive requests.
#
# **Drill 2.** *Does caching change the model's output?* Semantically, no. A hit reuses the K/V of
# exactly the same tokens. Also, vLLM adds a salt to the hash: the LoRA ids and an optional per-request
# `cache_salt`. Thus, different adapters or tenants never share blocks (verify).
#
# Bitwise, there is no guarantee. vLLM computed the cached K/V in a different batch and chunk shape than a
# recomputation uses. Floating-point reductions in a different order can be different in the last bits. This
# difference is sufficient to change a near-tie in greedy decoding from time to time. If you need
# bit-reproducible outputs, turn on the batch-invariant mode of vLLM (`VLLM_BATCH_INVARIANT=1` in v0.30.0).
# Examine its cost and coverage for your model (verify).
#
# **Drill 3.** *Why is the hit on a fully repeated prompt one block short?* vLLM always recomputes the last
# token to make logits. Thus, at most `(len - 1) // 16` blocks can hit.
