# %% [markdown]
# # 03 · Memory layouts and the prefix cache: where the block goes decides what every later turn pays
#
# **Tier:** T0 — a fake OpenAI-compatible server counts prefix-cache hits with vLLM's block rules and reports
# them in `usage.prompt_tokens_details.cached_tokens`; its TTFTs come from a roofline model. Every number
# from it is **simulated**. **T1:** set `MEMLAB_LLM_URL` to a real vLLM started with
# `--enable-prompt-tokens-details` (`deploy/any-gpu/`) and the same cells **measure** the cached tokens.
#
# ## The one-minute version
#
# An inference engine reuses the KV of the longest run of full 16-token blocks a prompt shares with an
# earlier one, **from token 0** (serving-engine PRIMER §5 "Prefix caching"; vllm-internals §4.3). Memory
# retrieved again every turn is *different* every turn, so its position decides how much of the prompt
# after it can hit (PRIMER §5 "The context budget: tokens, the prefix cache and cost per turn"):
#
# | layout | what changes each turn | what misses each turn |
# |---|---|---|
# | memory **before the history** | the block right after the system prompt | the block and the *whole history* after it |
# | a **pinned profile** + `recall` | nothing in the prefix (sorted, byte-identical) | the new turn only (plus a tool round) |
# | memory **at the tail**, request-scoped | the end of the prompt | the last reply, the new message, the block |
#
# Missed tokens are prefilled again: TTFT (a roofline estimate here, `minengine.perf.step_cost`'s model)
# and dollars (cached input is billed at a tenth, `scalelab.capacity.cost_per_call`). Primer:
# [`../../PRIMER.md`](../../PRIMER.md). The block rules themselves are exercises in 04.3
# ([`vllm-serving-lab` notebook 04](../../../../04-inference-engine/serving-engine/vllm-serving-lab/notebooks/04_prefix_caching_for_agents.ipynb));
# here they are library code: `cachebench.expected_cached_tokens`.

# %%
import hashlib, hmac, base64, os, math
from memlab import env, cachebench as cb
from memlab.agent import MemoryAgent
from memlab.cachebench import expected_cached_tokens, step_cost, GPUS, LLMS, PRICES, turn_cost
from memlab.fakeserver import FakeLLMServer, render_chat, tokenize, scrape, hit_rate
from memlab.llm import ChatClient, ScriptedModel, to_openai
from memlab.memory import LocalMemory
from memlab.store import SQLiteMemoryStore

print(env.banner())
real = env.llm_url() if env.llm_url() and not env.is_simulated(env.llm_url()) else None
fake = None if real else FakeLLMServer(enable_prompt_tokens_details=True)
URL = real or fake.start()
LABEL = "MEASURED" if real else "SIMULATED"
print("target:", URL, "|", LABEL)

# %% [markdown]
# ## Worked example: one session, three layouts
#
# The same user (a few facts, forty episodic notes), the same eight turns — each a question about the user
# plus ~250 tokens of pasted notes, the way tool results and documents grow an agent's history. Each run
# sends a fresh random `cache_salt`, so it starts cold even on a server that has seen the system prompt.
# `predicted` is `expected_cached_tokens` applied to the fake server's own tokenizer (at T1 the tokenizer is
# the model's, so the prediction is not shown).

# %%
runs = {layout: cb.run_layout(URL, layout, turns=8) for layout in cb.LAYOUTS}
print(runs["before_history"].table(), "\n")
print(runs["tail"].table(), "\n")
print(cb.summary(runs))

# %% [markdown]
# ## Worked example: the engine's own counters agree
#
# `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` over a window of two scrapes is the
# engine's hit rate in tokens — what you would watch on a dashboard. It should equal the per-request
# `cached_tokens` summed over the same window.

# %%
before = scrape(URL)
again = cb.run_layout(URL, "pinned", turns=4)
after = scrape(URL)
print(f"[{LABEL}] /metrics window hit rate {hit_rate(before, after):.1%}; per-request usage {again.hit_rate:.1%}")
assert real or math.isclose(hit_rate(before, after), again.hit_rate, rel_tol=1e-9)

# %% [markdown]
# ## Exercise 3.1 — build the prompt in each layout
#
# Write `build_prompt(layout, system, pinned, block, history, user)` returning the chat messages:
#
# * always `{"role": "system", "content": system}` first, then the pinned profile as a second system message
#   when `pinned` is not None;
# * `"before_history"`: the memory `block` as a system message right after those, then `history`, then the
#   user message;
# * `"tail"`: `history`, then one user message `f"{user}\n\n{block}"` (the block appended);
# * `"tail_before"`: `history`, then `f"{block}\n\n{user}"` (the block in front — where ADK's
#   `PreloadMemoryTool` inserts it).
#
# With `block=None`, every layout is just system (+ pinned) + history + user.

# %% exercise
def build_prompt(layout, system, pinned, block, history, user):
    ### BEGIN SOLUTION
    msgs = [{"role": "system", "content": system}]
    if pinned is not None:
        msgs.append({"role": "system", "content": pinned})
    if block and layout == "before_history":
        msgs.append({"role": "system", "content": block})
    msgs += list(history)
    content = user
    if block and layout == "tail":
        content = f"{user}\n\n{block}"
    elif block and layout == "tail_before":
        content = f"{block}\n\n{user}"
    return msgs + [{"role": "user", "content": content}]
    ### END SOLUTION

# %% check
store = SQLiteMemoryStore(":memory:")
cb.seed_memory(store)
mem = LocalMemory(store, "acme", "u1")
hist = [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": "Hi!"}]
for layout in ("before_history", "tail", "tail_before"):
    agent = MemoryAgent(ScriptedModel(), mem, mode="implicit", layout=layout, instruction=cb.SYSTEM)
    agent.pinned = "PROFILE" if layout == "tail" else None
    block = mem.render(mem.recall("Which city is my home city?", 3))
    assert build_prompt(layout, cb.SYSTEM, agent.pinned, block, hist, "Q?") == agent.build(hist, "Q?", block), layout
assert build_prompt("tail", "S", None, None, [], "Q?") == [{"role": "system", "content": "S"}, {"role": "user", "content": "Q?"}]
print("✅ build_prompt matches the agent's layouts")

# %% [markdown]
# ## Exercise 3.2 — find where the prefix breaks
#
# Write `first_divergence(a, b)`: the index of the first position where token lists `a` and `b` differ
# (`min(len(a), len(b))` if one is a prefix of the other). Then `lost_tokens(prev, new)`: how many tokens of
# `new` must be prefilled again, `len(new) - expected_cached_tokens(prev, new)`. The check renders turn 2 and
# turn 3 of the before-history layout through the fake server's template and tokenizer and asks where they
# part: it must be inside turn 3's memory block — everything after that point is recomputed.

# %% exercise
def first_divergence(a, b):
    ### BEGIN SOLUTION
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    ### END SOLUTION

def lost_tokens(prev, new):
    ### BEGIN SOLUTION
    return len(new) - expected_cached_tokens(prev, new)
    ### END SOLUTION

# %% check
agent = MemoryAgent(ScriptedModel(), mem, mode="implicit", layout="before_history", instruction=cb.SYSTEM)
h, texts = [], []
for t in range(3):
    q = cb.user_turn(t)
    block = mem.render(mem.recall(q, 5, 160))
    msgs = agent.build(h, q, block)
    texts.append((render_chat(to_openai(msgs)), block))
    h += [{"role": "user", "content": q}, {"role": "assistant", "content": "Noted."}]
(t2, _), (t3, block3) = texts[1], texts[2]
i = first_divergence(tokenize(t2), tokenize(t3))
assert t3.index("<<<MEMORY") <= 4 * i + 3 < t3.index("<<<END MEMORY>>>"), (i, t3.index("<<<MEMORY"))
assert first_divergence([1, 2, 3], [1, 2]) == 2 and lost_tokens(list(range(64)), list(range(64))) == 16
lost = lost_tokens(tokenize(t2), tokenize(t3))
print(f"✅ the prompts part at token {i} (inside the memory block, which starts near token "
      f"{t3.index('<<<MEMORY') // 4}); {lost} of {len(tokenize(t3))} tokens of turn 3 are prefilled again")

# %% [markdown]
# ## Exercise 3.3 — what a miss costs in time
#
# Write `ttft_ms(prompt_tokens, cached_tokens, gpu, llm)`: the prefill step of one request on an idle
# engine, `step_cost(GPUS[gpu], LLMS[llm], [(cached, prompt - cached)])["t"]` in milliseconds (at least one
# new token). This is `minengine.perf.step_cost`'s roofline — max(bytes / (BW·0.8), FLOPs / (peak·0.6)) +
# 2 ms — re-implemented in `cachebench`; the check pins it to the numbers the serving engine's model gives.

# %% exercise
def ttft_ms(prompt_tokens, cached_tokens, gpu="L4", llm="qwen2.5-1.5b"):
    ### BEGIN SOLUTION
    new = max(1, prompt_tokens - cached_tokens)
    return step_cost(GPUS[gpu], LLMS[llm], [(cached_tokens, new)])["t"] * 1e3
    ### END SOLUTION

# %% check
pins = {("L4", "qwen2.5-1.5b"): (78.7, 15.1), ("L4", "llama-3.1-8b"): (401.0, 65.6),
        ("H100-SXM", "qwen2.5-1.5b"): (11.4, 3.2), ("H100-SXM", "llama-3.1-8b"): (50.8, 7.7)}
for (g, m), (cold, warm) in pins.items():
    assert round(ttft_ms(2000, 0, g, m), 1) == cold and round(ttft_ms(2000, 1800, g, m), 1) == warm, (g, m)
for r in runs["before_history"].rows:
    assert math.isclose(ttft_ms(r.prompt_tokens, r.cached_tokens or 0), r.prefill_ms)
b, t = runs["before_history"], runs["tail"]
print(f"✅ [SIMULATED roofline] 2,000 tokens cold vs 1,800 cached on an L4: 78.7 vs 15.1 ms; "
      f"this session's prefill: before_history {b.prefill_ms:.0f} ms vs tail {t.prefill_ms:.0f} ms")

# %% [markdown]
# ## Exercise 3.4 — what a miss costs in dollars
#
# Write `session_cost(rows, price)`: the sum over rows of `turn_cost` — (uncached input × input price +
# cached input × cached price + output × output price) / 1e6 — with the row's `prompt_tokens`, its
# `cached_tokens` (0 when unknown) and `completion` tokens taken as 60 per call. (The check also pins the
# scaling primer §3.4 call: 5,000 input tokens with 2,700 cached and 350 output at `gemini-3.5-flash`'s
# prices of 5 Sep 2026 (verify) cost $0.007005.)

# %% exercise
def session_cost(rows, price, completion=60):
    ### BEGIN SOLUTION
    return sum(turn_cost(price, r.prompt_tokens, completion, r.cached_tokens or 0) for r in rows)
    ### END SOLUTION

# %% check
P = PRICES["gemini-3.5-flash"]
assert math.isclose(turn_cost(P, 5000, 350, 2700), 0.007005)
cost = {k: session_cost(r.rows, P) for k, r in runs.items()}
assert cost["before_history"] > cost["tail"], cost
print("✅ $ per 8-turn session at gemini-3.5-flash prices (verify):",
      {k: round(v, 5) for k, v in cost.items()}, f"— before_history costs {cost['before_history'] / cost['tail']:.1f}x tail")

# %% [markdown]
# ## Exercise 3.5 — one salt per tenant
#
# A prefix cache shared by tenants is a side channel: if tenant B's first request hits blocks tenant A
# created, B learns A sent that prefix (TTFT tells). vLLM keys the *first* block by the request's
# `cache_salt`, and every later block chains from it (vllm-internals §4.3), so a per-tenant salt gives each
# tenant its own cache. Write `tenant_salt(secret, tenant)`: an HMAC-SHA256 of the tenant under a server-side
# secret, base64url without padding — 43 characters, within vLLM's limit of 128 and free of `@ / \`.

# %% exercise
def tenant_salt(secret: bytes, tenant: str) -> str:
    ### BEGIN SOLUTION
    mac = hmac.new(secret, tenant.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).rstrip(b"=").decode()
    ### END SOLUTION

# %% check
SECRET = os.urandom(32)
sa, sb = tenant_salt(SECRET, "acme"), tenant_salt(SECRET, "globex")
assert sa == tenant_salt(SECRET, "acme") != sb and len(sa) == 43 and not set("@/\\") & set(sa)
probe = [{"role": "system", "content": cb.SYSTEM}, {"role": "user", "content": "Hello there, what can you do?"}]
with FakeLLMServer(enable_prompt_tokens_details=True) as u:
    def cached(extra):
        return ChatClient(u, extra=extra).generate(probe).usage["cached_tokens"]
    unsalted = (cached({}), cached({}))                              # tenant A, then tenant B, no salt
    salted = (cached({"cache_salt": sa}), cached({"cache_salt": sb}), cached({"cache_salt": sa}))
assert unsalted[1] > 0 and salted[1] == 0 and salted[2] > 0, (unsalted, salted)
print(f"✅ [SIMULATED] unsalted: tenant B's first request hit {unsalted[1]} tokens of tenant A's prefix; "
      f"salted: {salted[1]} (and acme's own repeat still hits {salted[2]})")

# %% [markdown]
# ## T1: measure it on vLLM
#
# Start vLLM with prefix-caching counters per request, then point this notebook at it and re-run from the
# top: the tables above become measurements (and the `predicted` column goes blank — the tokenizer is the
# model's). Keep the salt per run so runs do not warm each other.

# %%
if real:
    print(f"[MEASURED on {real}]")
    print(cb.summary(cb.compare_layouts(real, turns=8, api_key=os.environ.get("MEMLAB_API_KEY"))))
else:
    print("T0 here. On a GPU box (see deploy/any-gpu/README.md), from the repo root:")
    print('   MODEL=Qwen/Qwen2.5-1.5B-Instruct MAX_MODEL_LEN=8192 EXTRA_ARGS="--enable-auto-tool-choice '
          '--tool-call-parser hermes --enable-prompt-tokens-details" '
          '04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/serve.sh')
    print("   export MEMLAB_LLM_URL=http://127.0.0.1:8000   # then re-run this notebook, or: python -m memlab cachebench --url $MEMLAB_LLM_URL")
if fake:
    fake.stop()

# %% [markdown]
# ## In a design review
#
# **Two minutes.** "Memory is re-retrieved every turn, so it is the most volatile part of the prompt, and
# the prefix cache only reuses from token zero. If we inject it after the system prompt, every history block
# after it misses and each turn prefills the whole conversation again — here a 38% hit rate against 77% for
# the same memory appended to the new user message, and about twice the prefill time and dollars per
# session (simulated; the T1 run measures it). So: stable things first — system prompt, tools, a pinned
# per-session profile sorted so it is byte-identical — and per-turn memory at the tail, request-scoped.
# We watch `cached_tokens` per request and `prefix_cache_hits / queries` per replica, and we salt the cache
# per tenant so one tenant's prefix is never another's hit."
#
# **Drill 1.** *After adding memory, TTFT p50 tripled and input cost rose. Why, and what do you change?* —
# The block went in above the history, so the prefix changed every turn and the conversation was re-prefilled
# each time. Move per-turn memory to the tail (or pin a per-session profile), cap it at the recall knee
# (notebook 05), and verify with `cached_tokens`.
#
# **Drill 2.** *The pinned layout had the best hit rate but was not the cheapest. Why?* — Its `recall` tool
# adds a model call per question, each re-sending the prompt; hits make that call cheap, not free. Whether the
# profile beats per-turn retrieval depends on how often the profile alone answers.
#
# **Drill 3.** *Why is `cache_salt` a secret per tenant rather than the tenant id?* — A guessable salt lets
# anyone who can send requests with it probe that tenant's cache; vLLM's guidance is a random, unpredictable
# value (it enforces at most 128 characters in v0.30.0; verify).
