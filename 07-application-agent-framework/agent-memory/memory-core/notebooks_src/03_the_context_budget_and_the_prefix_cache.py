# %% [markdown]
# # 03 · The context budget and the prefix cache
#
# **Tier:** T0 — CPU only, no network, a few seconds. Times are **SIMULATED** (a roofline step model) and prices
# are dated list prices `(verify)`. The same layouts measured on a real vLLM server's `cached_tokens` are
# `memory-lab` notebook `03_memory_layouts_and_the_prefix_cache` (T0 simulated; T1 measured).
#
# ## The one-minute version
# Memory costs tokens twice: the tokens you inject, and the prefix-cache hits you destroy. An engine reuses KV only
# for an **exact prefix, in full blocks** (vLLM: 16 tokens, each block named by a hash chained to its parent; the
# last token of a prompt is always recomputed). If memory is re-retrieved every turn and placed **before the
# history**, the prefix changes right after the system prompt and every history block behind it misses: the whole
# conversation re-prefills on every turn. Two layouts keep the history a stable prefix: a **profile pinned once per
# session**, or per-turn memory at the **tail**, just before the new user message (ADK's `PreloadMemoryTool` inserts
# there). You can predict the hit rate of each layout from the block rules, turn the lost hits into TTFT with a
# roofline model and into dollars with a price table — and extraction has a price too.
#
# Primer: §5 *The context budget: tokens, the prefix cache and cost per turn*; the block rules are
# serving-engine PRIMER §5 and mini-engine-core notebook 03 exercise 3.4.

# %%
from memcore import (GPUS, LAYOUTS, LLMS, PrefixCache, call_cost, expected_cached_tokens, hit_rate, hits_per_turn,
                     prefill_seconds, token_ids, turn_cost)

# %% [markdown]
# ## Worked example 1 — three rules decide every hit
# vllm-serving-lab notebook 04 (exercise 4.1) states them; memcore restates them as `expected_cached_tokens(prev,
# new)` and as a block-hash `PrefixCache` (tested to agree with each other and with `minengine.kv`).

# %%
print(expected_cached_tokens(list(range(64)), list(range(64))), "- identical 64-token prompts: the last block is recomputed")
print(expected_cached_tokens(list(range(40)), list(range(40)) + [7] * 30), "- the earlier request left 2 full blocks")
print(expected_cached_tokens([1] * 20 + [2] * 20, [1] * 20 + [3] * 20, 8), "- divergence inside block 3 (B = 8)")
cache = PrefixCache(16)
cache.serve(list(range(100)), output=[7] * 20)       # a request and the reply it generated
print(cache.lookup(list(range(100)) + [7] * 20 + [1, 2, 3]), "- the next turn reuses the prompt AND the reply")

# %% [markdown]
# ## Worked example 2 — the three layouts, on text
# One session, six turns. The system prompt is stable; each turn retrieves two memories that depend on the question
# (so they change); the history is append-only. Tokens are 4-character chunks (`token_ids`), so a shared text
# prefix is a shared token prefix. Watch the **uncached** tokens per turn — what the engine must prefill.

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
# Before the history, the uncached part **grows every turn** — it is the whole history. Pinned, it is the new
# message and the rounding of one block. At the tail it is the last exchange plus the new memory: bounded, not
# growing. (Turn 1 is cold for everyone.)
#
# ## Worked example 3 — the default session: 2,000-token system prompt, 400 tokens of memory

# %%
for layout in LAYOUTS:
    turns = hits_per_turn(layout)                       # 8 turns: 40-token user message, 120-token reply
    print(f"{layout:15} hit rate {hit_rate(turns):6.1%}   cached per turn {[t.cached for t in turns]}")
print("\nover 20 turns:", {l: f"{hit_rate(hits_per_turn(l, turns=20)):.1%}" for l in LAYOUTS})

# %% [markdown]
# `none` is the no-memory baseline. Memory before the history drops the hit rate from 88% to 58% in 8 turns and
# to 48% in 20 — it gets **worse** as the conversation grows, because the uncached part is the whole history.
# Pinned costs almost nothing in hits; the tail loses one exchange and the memory itself per turn.
#
# ## Worked example 4 — what the lost hits cost (SIMULATED time, dated prices)
# Turn 8 of the default session. Time: memcore's restatement of `minengine.perf.step_cost` for one prefill chunk
# (`prefill_seconds`, roofline: max(bytes / 0.8 BW, FLOPs / 0.6 peak) + 2 ms). Money: `turn_cost` at the scaling
# primer's §3.4 prices (Gemini 3.5 Flash, $1.50 / $0.15 cached / $9.00 per M tokens, 5 Sep 2026, verify).

# %%
l4, q15, h100, l8 = GPUS["L4"], LLMS["qwen2.5-1.5b"], GPUS["H100-SXM"], LLMS["llama-3.1-8b"]
print(f"{'layout':15} {'uncached':>8} {'L4 1.5B':>9} {'H100 8B':>9} {'$ turn 8':>10} {'$ session':>10}")
for layout in LAYOUTS:
    turns = hits_per_turn(layout)
    t = turns[-1]
    print(f"{layout:15} {t.prompt - t.cached:8} {prefill_seconds(l4, q15, t.prompt, t.cached) * 1e3:7.1f}ms "
          f"{prefill_seconds(h100, l8, t.prompt, t.cached) * 1e3:7.1f}ms "
          f"{turn_cost(t.prompt, t.cached, 120)['total']:10.5f} {sum(turn_cost(x.prompt, x.cached, 120)['total'] for x in turns):10.5f}")

# %% [markdown]
# Memory before the history makes turn 8's prefill 4.5× slower on an L4 (68.4 vs 15.3 ms) and the 8-turn session
# 71% dearer than no memory at all ($0.0257 vs $0.0150); pinned costs 7% more than none. The engine did the same
# work the whole time — the layout decided how much of it was repeated.
#
# ## Worked example 5 — extraction has a price too
# Extracting after every turn is a model call per turn. Consolidating once per session reads the whole session.

# %%
per_turn = turn_cost(0, 0, 0, extraction_in=600, extraction_out=60)["extraction"]
per_session = turn_cost(0, 0, 0, extraction_in=1600, extraction_out=100, extractions_per_turn=1 / 8)["extraction"]
answer = turn_cost(3560, 3504, 120)["answer"]
print(f"answering turn 8 (pinned): ${answer:.5f} | extraction every turn: ${per_turn:.5f} | "
      f"one consolidation per 8-turn session, per turn: ${per_session:.5f}")

# %% [markdown]
# Hot-path extraction on every turn nearly doubles the cost of a cached turn. Batch it: extract in the background
# after the session (or on a schedule, notebook 04) unless the next turn needs the fact immediately.
#
# ## Worked example 6 — one prefix cache, many tenants
# vLLM lets a request carry a `cache_salt`, which enters the first block's hash only; the chain carries it to
# every later block (vllm-internals §4.3). Different salts never share blocks, so one tenant cannot learn from
# timing that another sent the same prefix. The salt must be secret, random and per tenant (vLLM v0.30.0 accepts
# at most 128 characters, verify).

# %%
prompt = token_ids(SYSTEM + "User: hello\n")
cache = PrefixCache(16)
cache.serve(prompt, salt="tenant-a-9f3c")
print("same tenant:", cache.lookup(prompt, salt="tenant-a-9f3c"), "| other tenant:", cache.lookup(prompt, salt="tenant-b-77aa"),
      "| no salt:", cache.lookup(prompt))

# %% [markdown]
# ## Exercise 3.1 — the three rules
# Implement `cached_tokens(prev, new, block_size=16)`: full blocks of the common prefix; no more than
# `(len(prev) - 1) // B` blocks (the last token of `prev` never had its KV computed); no more than
# `(len(new) - 1) // B` blocks (the last token of `new` is recomputed for its logits).

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
# For the default session (system S = 2,000, memory M = 400, user 40, reply 120, B = 16), write the cached tokens
# of turn `t ≥ 2` for each layout as a closed form. Hint: work out the common prefix with the previous request's
# prompt + reply, then round down to a block — and remember the previous request's last token.

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
# Write `ttft_lost_ms(t)`: at turn `t` of the default session, the SIMULATED prefill time of `before_history` minus
# that of `pinned`, on an L4 with Qwen2.5-1.5B (`prefill_seconds`), in milliseconds.

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
# ## Exercise 3.4 — dollars per session
# Write `session_cost(layout, model)`: the sum over the default session's turns of `call_cost(prompt, 120, cached,
# model)`. Then compute `premium`, how much dearer `before_history` is than `pinned` at Gemini 3.5 Flash prices, as
# a fraction (0.5 = 50%).

# %% exercise
def session_cost(layout, model="gemini-3.5-flash"):
    ### BEGIN SOLUTION
    return sum(call_cost(t.prompt, 120, t.cached, model) for t in hits_per_turn(layout))
    ### END SOLUTION

### BEGIN SOLUTION
premium = session_cost("before_history") / session_cost("pinned") - 1
### END SOLUTION

# %% check
for layout in LAYOUTS:
    assert abs(session_cost(layout) - sum(turn_cost(t.prompt, t.cached, 120)["total"] for t in hits_per_turn(layout))) < 1e-12
assert round(premium, 3) == 0.602
print(f"✅ memory before the history: {premium:.0%} dearer per session than a pinned profile, same answers")

# %% [markdown]
# ## Exercise 3.5 — fix the layout
# `bad_render` puts a timestamp first and per-turn memory before the history. Write `good_render` with the same
# arguments so that the uncached tokens per turn stop growing with the history (the check compares turn 6 with
# turn 2). The memory and the clock must both stay in the prompt.

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
# **The two-minute version.** "Memory costs tokens twice — what we inject and the prefix-cache hits we break. The
# engine reuses KV only for an exact prefix in full 16-token blocks, so where memory goes decides the hit rate. If
# we re-retrieve memory every turn and put it before the history, the prefix changes right after the system prompt
# and the whole history re-prefills every turn: in our 8-turn model the hit rate falls from 88% to 58%, turn-8 TTFT
# on an L4 goes from 15 to 68 ms (simulated), and the session costs 60% more than with a pinned profile. So the
# layout is: system prompt and tools, then a profile retrieved once per session, then the append-only history, then
# per-turn memory and anything volatile at the tail — which is where ADK's preload puts it too. We cap injected
# memory at the recall knee, extract in the background rather than on every turn, salt the prefix cache per
# tenant, and watch `cached_tokens` per turn to catch a layout regression."
#
# **Drill questions**
# 1. *TTFT tripled after we added long-term memory, for a small recall gain. Why?* — Memory is re-retrieved per
#    turn and injected above the history, so every history block misses the prefix cache and the conversation
#    re-prefills on every turn. Pin a profile per session or move per-turn memory to the tail; cap it at the knee.
# 2. *Why does the damage grow with the conversation?* — The uncached part is everything after the memory block,
#    i.e. the whole history; the pinned and tail layouts lose a constant amount per turn instead.
# 3. *Two tenants share a system prompt on one vLLM. Any risk?* — Without a per-tenant `cache_salt`, one tenant's
#    TTFT reveals whether the other sent the same prefix. Salt per tenant (secret, random), and budget for the
#    shared-prefix hits you give up.
