# %% [markdown]
# # 04 · Prefix caching for agents: measure the hit rate, then design prompts that earn it
#
# **Tier:** T0 — the fake vLLM implements vLLM's block-hash prefix cache (results **simulated**).
# T1/T3: point `SERVELAB_URL` at a real vLLM; the hit rate comes from its `/metrics` either way, and
# per-request cached tokens appear when the server runs with `--enable-prompt-tokens-details` (verify).
#
# ## The one-minute version
#
# vLLM keys each full KV block (16 tokens) by a hash of **its tokens and the hash of the block
# before it**. A new request reuses the longest run of leading blocks whose hashes are already in
# the cache and computes only the rest: prefill work and TTFT drop in proportion. Two consequences
# decide everything for agents:
#
# * a hit needs an **identical prefix from token 0** — one changed token early (a timestamp in the
#   system prompt, a reordered tool list) invalidates every block after it;
# * an agent conversation is **append-only**, so turn *n+1* can reuse all of turn *n* — prompt,
#   tool results and the model's own reply — if the client resends them byte for byte.
#
# The engine counts it: `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` (tokens).
# Concepts: PRIMER §5 "Prefix caching" ([`PRIMER.md`](../../PRIMER.md)); paging and sharing in
# `04-inference-engine/paged-attention/`.

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
# Hash each 16-token block together with the previous block's hash. Compare the first prompt with
# (a) the second one, which differs only in the final question, and (b) a copy with one word
# changed in the first sentence of the system prompt.

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
# The previous request processed `prev` (its prompt *and* generated tokens). Write
# `expected_cached_tokens(prev, new, block_size)` for a request `new` arriving right after, applying
# vLLM's three rules: only **full** blocks are cached; the last token of `prev` never had its KV
# computed (it was sampled, not fed back), so `prev` left `(len(prev) - 1) // block_size` blocks; and
# at least one token of `new` is always recomputed to get logits, so at most
# `(len(new) - 1) // block_size` blocks can hit. Within those limits, hits are the full blocks of
# the common prefix.

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
# Counters are cumulative; the hit rate of a window is the ratio of the counter *increases*. Write
# `hit_rate(before, after)` from two parsed scrapes (`scrape.value(name)` sums a metric; the names are
# `M.PREFIX_HITS` and `M.PREFIX_QUERIES`). Return `nan` when nothing was queried.

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
# ## Exercise 4.3 — three prompt layouts for the same agent
#
# Six agent sessions, three turns each (the user's request, then two tool results; the model's reply
# is appended after every turn). Same content, three layouts (see `agent_sessions`):
# `"stable"` (shared system prompt and tool list, append-only history), `"shuffled_tools"` (each
# session lists the tools in its own order), `"timestamp_first"` (a fresh timestamp on the first
# line of every request). Put the layouts in `ranking`, from highest to lowest expected hit rate.

# %% exercise
### BEGIN SOLUTION
ranking = ["stable", "shuffled_tools", "timestamp_first"]
### END SOLUTION

# %% check
rates, ttft = {}, {}
for i, layout in enumerate(("stable", "shuffled_tools", "timestamp_first")):
    before = M.scrape(URL, headers=H)
    run = run_sessions(URL, agent_sessions(6, turns=3, layout=layout, seed=100 + i), session_rate=3.0, headers=H)
    rates[layout] = hit_rate(before, M.scrape(URL, headers=H))
    ttft[layout] = run.summary().ttft.mean
    print(f"[{LABEL}] {layout:16s} hit rate {rates[layout]:6.1%}   mean TTFT {ttft[layout]:6.1f} ms")
assert sorted(rates, key=rates.get, reverse=True) == ranking
assert rates["timestamp_first"] < 0.05 and ttft["stable"] < ttft["timestamp_first"]
print("✅ one timestamp on line 1 costs the whole cache; a reordered tool list costs the cross-session share")

# %% [markdown]
# ## Exercise 4.4 — fix the template
#
# The agent below renders its prompt with the time on the first line and the tools in whatever
# order the registry returns them. Write `render_cache_friendly(system, tools, history, new_message,
# now)` that keeps every earlier token identical from turn to turn: stable system prompt with the
# tools in a **deterministic** order first, the history exactly as it was sent before, and the
# dynamic facts (the time) attached to the **new** message only. The caller appends what you return
# for the new message to the history, so it stays verbatim afterwards.

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
# Prefill is compute-bound, so a hit saves roughly the prefill time of the cached tokens. Write
# `ttft_estimate(p, prompt_tokens, cached_tokens)` for an otherwise idle engine: one step that
# prefills only the uncached tokens (use `p.prefill_s(tokens, context=cached_tokens)`). Then predict
# the saving for a 6,000-token agent prompt with 5,000 tokens cached, and compare with a
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
# Cached blocks live in the same KV pool as the requests being served: they stay cached only while
# they are free and not yet reallocated (least recently freed goes first). A tool-using agent's
# shared prefix (here ~1,500 tokens = ~94 blocks) is tiny next to a T4's ~66,000 blocks, so it
# survives; per-session histories are what gets evicted under load. When sessions are routed to
# different replicas, each replica needs its own copy — which is why routers in layer 05 send a
# session back to the replica that holds its prefix.
#
# ## In a design review
#
# **Two minutes:** "vLLM caches KV per 16-token block, keyed by a hash chained through every earlier
# block, so reuse needs an identical prefix from the first token. Our agent's system prompt and tool
# schemas are ~1,500 tokens on every request and the conversation only grows, so we lay the prompt
# out stable-first — system prompt, tools in a fixed order, then history appended verbatim — and
# attach timestamps and per-request facts to the newest message. We verify it from the engine:
# `prefix_cache_hits / prefix_cache_queries` over a window, which should sit near the fraction of
# each prompt that repeats. A hit saves prefill compute on the cached tokens, so TTFT follows the
# uncached tail — and the first engineer to put `datetime.now()` on line 1 of the system prompt
# turns that off for everyone."
#
# **Drill 1.** *The hit rate dropped from 80% to 3% after a release. Where do you look first?* —
# The prompt template: something dynamic moved into the prefix (a timestamp, a request id, a
# shuffled tool list, a changed chat template). Diff the first blocks of two consecutive requests.
#
# **Drill 2.** *Does caching change the model's output?* — No: the cached KV is exactly what
# recomputation would produce for the same tokens (vLLM also salts the hash with LoRA ids and an
# optional per-request `cache_salt` so different adapters or tenants never share blocks; verify).
#
# **Drill 3.** *Why is the hit on a fully repeated prompt one block short?* — The last token is
# always recomputed to produce logits, so at most `(len - 1) // 16` blocks can hit.
