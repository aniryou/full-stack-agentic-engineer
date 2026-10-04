# %% [markdown]
# # 03 · The context budget and the prefix cache
#
# **Tier:** T0. It uses only a CPU, it needs no network, and it takes a few seconds. The times are **SIMULATED** (a
# roofline step model), and the prices are dated list prices `(verify)`. The same layouts, measured on the
# `cached_tokens` of a real vLLM server, are in `memory-lab` notebook `03_memory_layouts_and_the_prefix_cache` (T0
# simulated, T1 measured).
#
# ## The one-minute version
# Memory costs tokens two times: the tokens that you inject, and the prefix-cache hits that you destroy. An engine uses
# KV again only for an **exact prefix, in full blocks**. In vLLM, a block is 16 tokens, and a chained hash names each block: the
# hash of a block includes the hash of its parent. The engine always computes the last token of a prompt again.
#
# If the agent retrieves memory again at each turn and puts it **before the history**, the prefix changes immediately
# after the system prompt. Then every history block after that point misses, and the engine prefills the whole
# conversation again at each turn. Two layouts keep the history a stable prefix:
#
# - a **profile pinned once per session**.
# - per-turn memory at the **tail**, immediately before the new user message (ADK's `PreloadMemoryTool` inserts
#   there).
#
# You can predict the hit rate of each layout from the block rules. A roofline model then changes the lost hits into
# TTFT, which is the self-hosted cost. A hosted API is different. It bills the cached rate only when a request gets to
# its caching minimum (4,096 tokens on Gemini 3.x). Below the minimum, the layout does not change the bill at all.
# Also, extraction has a price.
#
# Primer: §5 *The context budget: tokens, the prefix cache and cost per turn*. The block rules are in serving-engine
# PRIMER §5 and mini-engine-core notebook 03 exercise 3.4.

# %%
from memcore import (CACHE_MIN_TOKENS, GPUS, LAYOUTS, LLMS, PrefixCache, Salts, billed_cached, cache_salt, call_cost,
                     expected_cached_tokens, hit_rate, hits_per_turn, prefill_seconds, token_ids, turn_cost)

# %% [markdown]
# ## Worked example 1 — three rules decide every hit
# vllm-serving-lab notebook 04 (exercise 4.1) gives the rules. memcore gives them again as
# `expected_cached_tokens(prev, new)` and as a block-hash `PrefixCache`. Tests make sure that these two agree with each
# other and with `minengine.kv`.

# %%
print(expected_cached_tokens(list(range(64)), list(range(64))), "- identical 64-token prompts: the last block is recomputed")
print(expected_cached_tokens(list(range(40)), list(range(40)) + [7] * 30), "- the earlier request left 2 full blocks")
print(expected_cached_tokens([1] * 20 + [2] * 20, [1] * 20 + [3] * 20, 8), "- divergence inside block 3 (B = 8)")
cache = PrefixCache(16)
cache.serve(list(range(100)), output=[7] * 20)       # a request and the reply it generated
print(cache.lookup(list(range(100)) + [7] * 20 + [1, 2, 3]), "- the next turn reuses the prompt AND the reply")

# %% [markdown]
# ## Worked example 2 — the three layouts, on text
# This is one session with six turns. The system prompt is stable. Each turn retrieves two memories that depend on the
# question, thus these memories change. The history is append-only.
#
# Tokens are 4-character chunks (`token_ids`), thus a shared text prefix is a shared token prefix. Look at the
# **uncached** tokens for each turn. These are the tokens that the engine must prefill.

# %%
SYSTEM = ("You are a travel assistant. Tools: search_flights(origin, dest, date), book(flight_id), "
          "cancel(booking_id). Answer briefly and confirm before booking. ") * 6
QUESTIONS = ["Find me a flight to Oslo on Friday.", "Is there anything earlier?", "Book the 07:10 one.",
             "And a hotel near the station?", "Something quieter, please.", "Send me the receipts."]
MEMORY = [["- The user prefers window seats.", "- The user's home city is Lisbon."],
          ["- The user avoids red-eye flights.", "- The user's home city is Lisbon."],
          ["- The user pays with the company card.", "- The user prefers window seats."],
          ["- The user likes quiet hotels.", "- The user's employer is Acme."],
          ["- The user likes quiet hotels.", "- The user avoids red-eye flights."],
          ["- The user's employer is Acme.", "- The user pays with the company card."]]
PROFILE = "- The user's home city is Lisbon.\n- The user prefers window seats.\n"
REPLY = "Assistant: here are three options, cheapest first; shall I hold the first one for you?\n"


def uncached_per_turn(render):
    """Serve a six-turn session through one prefix cache; return the uncached tokens of each turn."""
    cache, history, out = PrefixCache(16), "", []
    for i, (q, mem) in enumerate(zip(QUESTIONS, MEMORY)):
        ids = token_ids(render(SYSTEM, "\n".join(mem) + "\n", history, q, f"10:0{i}"))
        out.append(len(ids) - cache.serve(ids, token_ids(REPLY)))
        history += f"User: {q}\n{REPLY}"
    return out


LAYOUT_TEXT = {
    "before_history": lambda system, memory, history, q, clock: system + memory + history + f"User: {q}\n",
    "pinned": lambda system, memory, history, q, clock: system + PROFILE + history + f"User: {q}\n",
    "tail": lambda system, memory, history, q, clock: system + history + memory + f"User: {q}\n"}
for layout, render in LAYOUT_TEXT.items():
    print(f"{layout:15} uncached tokens per turn: {uncached_per_turn(render)}")

# %% [markdown]
# With memory before the history, the uncached part **grows every turn**, because it is the whole history. With a
# pinned profile, it is the new message and the rounding of one block. At the tail, it is the last exchange and the new
# memory. That part has a limit, and it does not grow. (Turn 1 is cold for all layouts.)
#
# ## Worked example 3 — the default session: 2,000-token system prompt, 400 tokens of memory

# %%
for layout in LAYOUTS:
    turns = hits_per_turn(layout)                       # 8 turns: 40-token user message, 120-token reply
    print(f"{layout:15} hit rate {hit_rate(turns):6.1%}   cached per turn {[t.cached for t in turns]}")
print("\nover 20 turns:", {l: f"{hit_rate(hits_per_turn(l, turns=20)):.1%}" for l in LAYOUTS})

# %% [markdown]
# `none` is the no-memory baseline. Memory before the history decreases the hit rate from 88% to 58% in 8 turns, and
# to 48% in 20. It becomes **worse** as the conversation grows, because the uncached part is the whole history. A
# pinned profile costs almost nothing in hits. The tail loses one exchange and the memory itself at each turn.
#
# ## Worked example 4 — what the lost hits cost: time on your GPUs (SIMULATED), dollars on a hosted API
# This is turn 8 of the default session. For a self-hosted engine, the cost is time. memcore gives
# `minengine.perf.step_cost` again for one prefill chunk (`prefill_seconds`, roofline:
# $\max\bigl(\tfrac{\text{bytes}}{0.8 \cdot \text{BW}}, \tfrac{\text{FLOPs}}{0.6 \cdot \text{peak}}\bigr) + 2\ \text{ms}$).

# %%
l4, q15, h100, l8 = GPUS["L4"], LLMS["qwen2.5-1.5b"], GPUS["H100-SXM"], LLMS["llama-3.1-8b"]
print(f"{'layout':15} {'uncached':>8} {'L4 1.5B':>9} {'H100 8B':>9}")
for layout in LAYOUTS:
    t = hits_per_turn(layout)[-1]
    print(f"{layout:15} {t.prompt - t.cached:8} {prefill_seconds(l4, q15, t.prompt, t.cached) * 1e3:7.1f}ms "
          f"{prefill_seconds(h100, l8, t.prompt, t.cached) * 1e3:7.1f}ms")

# %% [markdown]
# Memory before the history makes turn 8's prefill 4.5× slower on an L4 (68.4 against 15.3 ms, SIMULATED). The engine
# did the same work all the time. The layout decided how much of that work the engine did again.
#
# Money is a different system. A hosted API has its own cache. It bills the cached rate only when a request gets to
# its **caching minimum**. On Gemini 3.5 Flash, the cached rate is \$0.15 instead of \$1.50 per M (5 Sep 2026,
# verify).
#
# The minimum is 4,096 tokens on Gemini 3.x (scaling primer §5.5). We do not know if the minimum applies to the
# request or to the shared prefix (verify). `turn_cost` applies the minimum through `billed_cached`. It assumes that
# the provider caches the same prefix as vLLM.

# %%
print("caching minimum:", CACHE_MIN_TOKENS["gemini-3.5-flash"], "tokens | turn-8 prompts:",
      {l: hits_per_turn(l)[-1].prompt for l in LAYOUTS})
for system in (2000, 4000):
    cost = {l: sum(turn_cost(t.prompt, t.cached, 120)["total"] for t in hits_per_turn(l, system_tokens=system))
            for l in LAYOUTS}
    print(f"{system:,}-token system prompt, $ per 8-turn session:", {l: round(c, 5) for l, c in cost.items()},
          f"| before_history vs pinned: {cost['before_history'] / cost['pinned'] - 1:+.0%}")

# %% [markdown]
# With the 2,000-token system prompt, every prompt stays below 4,096 tokens. The provider bills no tokens at the cached
# rate, and all three memory layouts cost the same. They cost 12% more than no memory, for the 400 tokens injected.
# With a 4,000-token system prompt (tools and policies), every turn from the second gets to the minimum. (The support
# turn of the scaling primer sends 4–6 k.) Then memory before the history costs 46% more than a pinned profile.
#
# The larger prompt also gives the *lower-cost* session, because it is the prompt that the provider caches. The
# minimum is a threshold, not a slope. Know the threshold of your provider.
#
# ## Worked example 5 — extraction has a price too
# Extraction after every turn is one model call for each turn. Consolidation one time per session reads the whole
# session.

# %%
per_turn = turn_cost(0, 0, 0, extraction_in=600, extraction_out=60)["extraction"]
per_session = turn_cost(0, 0, 0, extraction_in=1600, extraction_out=100, extractions_per_turn=1 / 8)["extraction"]
cached = hits_per_turn("pinned", system_tokens=4000)[-1]
answer = turn_cost(cached.prompt, cached.cached, 120)["answer"]
print(f"answering turn 8 (pinned, 4,000-token system prompt, billed cached): ${answer:.5f} | extraction every turn: "
      f"${per_turn:.5f} | one consolidation per 8-turn session, per turn: ${per_session:.5f}")

# %% [markdown]
# Hot-path extraction at each turn adds 72% to the cost of a cached turn. Batch it. Extract in the background after
# the session (or on a schedule, notebook 04), unless the next turn needs the fact immediately.
#
# ## Worked example 6 — one prefix cache, many tenants
# vLLM lets a request carry a `cache_salt`. The salt goes only into the hash of the first block, and the chain carries
# it to every later block (vllm-internals §4.3). Different salts never share blocks. Thus one tenant cannot learn from
# response times that another tenant sent the same prefix. The salt must be secret and per tenant: `cache_salt(secret,
# tenant)` is an HMAC of the tenant under a server-side secret.
#
# vLLM v0.30.0 accepts at most 128 characters, and it refuses `@`, `/`, `\` and NUL (verify). Thus a salt like
# `acme/alice` is not even valid, and a salt like `acme` is easy to guess.

# %%
prompt = token_ids(SYSTEM + "User: hello\n")
salts = Salts(b"a server-side secret")
cache = PrefixCache(16)
cache.serve(prompt, salt=salts.salt("acme"))
print("acme's salt:", salts.salt("acme"), "| globex's:", salts.salt("globex"))
print("same tenant:", cache.lookup(prompt, salt=salts.salt("acme")), "| other tenant:",
      cache.lookup(prompt, salt=salts.salt("globex")), "| no salt:", cache.lookup(prompt))
salts.rotate("acme")                       # e.g. a deletion request (notebook 04): vLLM cannot evict by salt
print("after rotating acme's salt:", cache.lookup(prompt, salt=salts.salt("acme")), "hits; blocks still resident:",
      cache.count(salts.retired["acme"][0]))

# %% [markdown]
# ## Exercise 3.1 — the three rules
# Write `cached_tokens(prev, new, block_size=16)`. It returns the full blocks of the common prefix, with these limits:
#
# - No more than `(len(prev) - 1) // B` blocks. The engine never computed the KV of the last token of `prev`.
# - No more than `(len(new) - 1) // B` blocks. The engine computes the last token of `new` again for its logits.

# %% exercise
def cached_tokens(prev, new, block_size=16):
    ### BEGIN SOLUTION
    common = 0
    for x, y in zip(prev, new):
        if x != y:
            break
        common += 1
    return max(0, min(common // block_size, (len(prev) - 1) // block_size, (len(new) - 1) // block_size)) * block_size
    ### END SOLUTION

# %% check
import random
rng = random.Random(0)
for _ in range(500):
    a = [rng.randrange(3) for _ in range(rng.randrange(1, 70))]
    b = a[:rng.randrange(0, len(a) + 1)] + [rng.randrange(3) for _ in range(rng.randrange(1, 30))]
    B = rng.choice([4, 8, 16])
    assert cached_tokens(a, b, B) == expected_cached_tokens(a, b, B)
print("✅ the three rules, on 500 random pairs")

# %% [markdown]
# ## Exercise 3.2 — predict the hits without simulating
# The default session has system $S = 2{,}000$, memory $M = 400$, user 40, reply 120 and $B = 16$. For each layout,
# write the cached tokens of turn $t \ge 2$ as a closed form. Hint: find the common prefix with the prompt + reply of
# the previous request. Then round down to a block. Also, remember the last token of the previous request.

# %% exercise
def predicted_cached(layout, t):
    ### BEGIN SOLUTION
    exchange = 160
    if layout == "before_history":
        return 2000                                   # the system prompt; memory differs right after it
    if layout == "tail":
        return 2000 + exchange * (t - 2)              # system + history before the last exchange
    prev_seq = (2040 if layout == "none" else 2440) + exchange * (t - 2) + 120
    return (prev_seq - 1) // 16 * 16                  # everything the last request computed, in full blocks
    ### END SOLUTION

# %% check
for layout in LAYOUTS:
    got = [t.cached for t in hits_per_turn(layout)][1:]
    assert got == [predicted_cached(layout, t) for t in range(2, 9)], (layout, got)
print("✅ the block rules predict every turn of every layout")

# %% [markdown]
# ## Exercise 3.3 — what memory before the history costs per turn
# Write `ttft_lost_ms(t)`. It returns a value for turn `t` of the default session, in milliseconds. The value is the
# SIMULATED prefill time of `before_history` minus the prefill time of `pinned`. Use an L4 with Qwen2.5-1.5B
# (`prefill_seconds`).

# %% exercise
def ttft_lost_ms(t, turns=8):
    ### BEGIN SOLUTION
    a, b = hits_per_turn("before_history", turns=turns)[t - 1], hits_per_turn("pinned", turns=turns)[t - 1]
    return (prefill_seconds(l4, q15, a.prompt, a.cached) - prefill_seconds(l4, q15, b.prompt, b.cached)) * 1e3
    ### END SOLUTION

# %% check
lost = [ttft_lost_ms(t, turns=20) for t in (2, 8, 20)]
assert round(lost[1], 1) == 53.2 and lost[0] < lost[1] < lost[2]
print(f"✅ SIMULATED TTFT lost to memory before the history: turn 2 {lost[0]:.1f} ms, turn 8 {lost[1]:.1f} ms, turn 20 {lost[2]:.1f} ms")

# %% [markdown]
# ## Exercise 3.4 — dollars per session, on a hosted API
# Write `session_cost(layout, system_tokens, model)`. It returns the sum of `call_cost(prompt, 120, billed, model)`
# over the turns of the session (`hits_per_turn(layout, system_tokens=system_tokens)`). If the prompt of the turn is
# at least `CACHE_MIN_TOKENS[model]` tokens, `billed` is the cached tokens of the turn. If not, `billed` is 0. This
# limit is the caching minimum of the provider.
#
# Then set `premium_2k` and `premium_4k`. Each one is how much more `before_history` costs than `pinned`, as a
# fraction (0.5 = 50%). `premium_2k` is for a 2,000-token system prompt, and `premium_4k` is for a 4,000-token system
# prompt.

# %% exercise
def session_cost(layout, system_tokens=2000, model="gemini-3.5-flash"):
    ### BEGIN SOLUTION
    total = 0.0
    for t in hits_per_turn(layout, system_tokens=system_tokens):
        billed = t.cached if t.prompt >= CACHE_MIN_TOKENS[model] else 0
        total += call_cost(t.prompt, 120, billed, model)
    return total
    ### END SOLUTION

### BEGIN SOLUTION
premium_2k = session_cost("before_history", 2000) / session_cost("pinned", 2000) - 1
premium_4k = session_cost("before_history", 4000) / session_cost("pinned", 4000) - 1
### END SOLUTION

# %% check
for layout in LAYOUTS:
    for system in (2000, 3000, 4000):
        want = sum(turn_cost(t.prompt, t.cached, 120)["total"] for t in hits_per_turn(layout, system_tokens=system))
        assert abs(session_cost(layout, system) - want) < 1e-12, (layout, system)
assert premium_2k == 0 and round(premium_4k, 3) == 0.457
print(f"✅ before_history vs pinned: {premium_2k:+.0%} under the caching minimum, {premium_4k:+.0%} above it")

# %% [markdown]
# ## Exercise 3.5 — fix the layout
# `bad_render` puts a timestamp first and per-turn memory before the history. Write `good_render` with the same
# arguments. In `good_render`, the uncached tokens per turn must not grow with the history (the check compares turn 6
# with turn 2). The memory and the clock must both stay in the prompt.

# %%
def bad_render(system, memory, history, question, clock):
    return f"[{clock}] " + system + memory + history + f"User: {question}\n"

# %% exercise
def good_render(system, memory, history, question, clock):
    ### BEGIN SOLUTION
    return system + history + memory + f"User ({clock}): {question}\n"
    ### END SOLUTION

# %% check
bad, good = uncached_per_turn(bad_render), uncached_per_turn(good_render)
assert bad[5] > bad[1] + 60, bad                                   # the bad layout re-prefills everything
assert good[5] <= good[1] + 16 and sum(good) < sum(bad) / 2, good
probe = good_render(SYSTEM, "- MEMORY -\n", "", "hi", "10:03")
assert "10:03" in probe and "- MEMORY -" in probe
print(f"✅ uncached tokens per turn {bad} -> {good}: stable first, history append-only, volatile last")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Memory costs tokens two times: the tokens that we inject and the prefix-cache hits that
# we break. The engine uses KV again only for an exact prefix in full 16-token blocks. Thus the position of memory
# decides the hit rate. If we retrieve memory again at each turn and put it before the history, the prefix changes
# immediately after the system prompt. Then the engine prefills the whole history again at each turn.
#
# "In our 8-turn model, the hit rate decreases from 88% to 58%. On our own engine, turn-8 TTFT on an L4 goes from 15
# to 68 ms (simulated). On a hosted API, the bill changes only when prompts get to the provider's caching minimum
# (4,096 tokens on Gemini 3.x). Then the session costs 46% more than with a pinned profile (a 4,000-token system
# prompt).
#
# "Thus the layout starts with the system prompt and tools. Then comes a profile, retrieved one time per session, and
# then the append-only history. Per-turn memory and anything volatile go at the tail. ADK's preload also puts memory
# there.
#
# "We limit injected memory to the recall knee. We extract in the background, not at each turn. We give the prefix
# cache a salt for each tenant. We monitor `cached_tokens` per turn to find a layout regression."
#
# **Drill questions**
# 1. *TTFT became three times larger after we added long-term memory, for a small recall gain. Why?* The agent
#    retrieves memory again at each turn and injects it above the history. Thus every history block misses the prefix
#    cache, and the engine prefills the conversation again at each turn. Pin a profile per session, or move per-turn
#    memory to the tail. Limit it to the knee.
# 2. *Why does the damage grow with the conversation?* The uncached part is everything after the memory block, that
#    is, the whole history. But the pinned and tail layouts lose a constant quantity per turn.
# 3. *Two tenants share a system prompt on one vLLM. Is there a risk?* Without a per-tenant `cache_salt`, the TTFT of
#    one tenant shows if the other tenant sent the same prefix. Use a salt for each tenant (an HMAC under a server
#    secret, never the tenant's name). Also, include in the budget the shared-prefix hits that you give up.
# 4. *We moved memory to the tail and the hosted-API bill did not change. Why?* Every prompt is below the caching
#    minimum of the provider. Thus the provider billed nothing at the cached rate, before or after the change. The
#    gain shows as TTFT on a self-hosted engine. It shows on the bill only when the prompt gets to the minimum.
