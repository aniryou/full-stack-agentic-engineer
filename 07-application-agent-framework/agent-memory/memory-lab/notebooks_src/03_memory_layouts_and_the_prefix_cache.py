# %% [markdown]
# # 03 · Memory layouts and the prefix cache: where the block goes decides what every later turn pays
#
# **Tier:** T0. A fake OpenAI-compatible server counts prefix-cache hits with the block rules of vLLM and
# reports them in `usage.prompt_tokens_details.cached_tokens`. Its TTFTs come from a roofline model, and
# every number from it is **simulated**. **T1:** set `MEMLAB_LLM_URL` to a real vLLM started with
# `--enable-prompt-tokens-details` (`deploy/any-gpu/`). Then the same cells **measure** the cached tokens, and
# only those. The prefill column stays a roofline model, and the dollars stay a price table, each with a label
# that says so.
#
# ## The one-minute version
#
# A prompt can share a run of full 16-token blocks with an earlier prompt, **from token 0**. An inference
# engine uses the KV of the longest such run again (serving-engine PRIMER §5 "Prefix caching", vllm-internals
# §4.3). Memory that the agent retrieves again every turn is *different* every turn. Thus its position decides
# how much of the prompt after it can get a cache hit. The table shows the three layouts (PRIMER §5 "The
# context budget: tokens, the prefix cache and cost per turn"):
#
# | layout | what changes each turn | what misses each turn |
# |---|---|---|
# | memory **before the history** | the block right after the system prompt | the block and the *whole history* after it |
# | a **pinned profile** + `recall` | nothing in the prefix (sorted, byte-identical) | the new turn only (plus a tool round) |
# | memory **at the tail**, request-scoped | the new user message (block in front of its text) | the previous user message, the last reply, the new message, the block |
#
# ADK's `PreloadMemoryTool` puts preloaded memory at the tail. A variant, `tail_after`, adds the block *after*
# the text of the user, not in front of it. Thus on the next turn, the previous user message is still an exact
# prefix.
#
# The engine must prefill the missed tokens again. On your own engine, this costs TTFT (here a roofline
# estimate, the model of `minengine.perf.step_cost`). On a hosted API, it costs dollars. There, the provider
# bills cached input at a tenth of the price (`scalelab.capacity.cost_per_call`). But it does this only when a
# request reaches the caching minimum of the provider (4,096 tokens on Gemini 3.x, scaling primer §5.5,
# verify).
#
# The session here has a different shape from the model of PRIMER §5 (a shorter system prompt, 250 pasted
# tokens per turn, real retrieval). Thus its rates are different, but the order of the layouts is the
# same. Primer: [`../../PRIMER.md`](../../PRIMER.md). The block rules are exercises in 04.3
# ([`vllm-serving-lab` notebook 04](../../../../04-inference-engine/serving-engine/vllm-serving-lab/notebooks/04_prefix_caching_for_agents.ipynb)).
# Here they are library code: `cachebench.expected_cached_tokens`.

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
# ## Worked example: one session, three layouts and a variant
#
# Each layout gets the same user (a few facts, forty episodic notes) and the same eight turns. Each turn is a
# question about the user plus ~250 tokens of pasted notes. This is how tool results and documents make the
# history of an agent grow. Each run sends a new random `cache_salt`, thus each run starts cold, also on a
# server that has seen the system prompt. `predicted` is `expected_cached_tokens` applied to the tokenizer of
# the fake server. At T1 the tokenizer is that of the model, thus the notebook does not show the prediction.

# %%
runs = {layout: cb.run_layout(URL, layout, turns=8) for layout in cb.LAYOUTS + cb.VARIANTS}
print(runs["before_history"].table(), "\n")
print(runs["tail"].table(), "\n")
print(cb.summary(runs))

# %% [markdown]
# ## Worked example: the engine's own counters agree
#
# `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` over a window of two scrapes is the hit
# rate of the engine in tokens. This is the value that you monitor on a dashboard. If all is correct, it is
# equal to the sum of the per-request `cached_tokens` over the same window.

# %%
before = scrape(URL)
again = cb.run_layout(URL, "pinned", turns=4)
after = scrape(URL)
print(f"[{LABEL}] /metrics window hit rate {hit_rate(before, after):.1%}; per-request usage {again.hit_rate:.1%}")
assert real or math.isclose(hit_rate(before, after), again.hit_rate, rel_tol=1e-9)

# %% [markdown]
# ## Exercise 3.1 — build the prompt in each layout
#
# Write `build_prompt(layout, system, pinned, block, history, user)`. It returns the chat messages:
#
# * Always `{"role": "system", "content": system}` first. Then, when `pinned` is not None, the pinned profile
#   as a second system message.
# * `"before_history"`: the memory `block` as a system message right after those, then `history`, then the
#   user message.
# * `"tail"`: `history`, then one user message `f"{block}\n\n{user}"`. The block is in front of the text of
#   the user. ADK's `PreloadMemoryTool` inserts it there.
# * `"tail_after"`: `history`, then `f"{user}\n\n{block}"` (the block added at the end).
#
# With `block=None`, every layout is only system (+ pinned) + history + user.

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
        content = f"{block}\n\n{user}"
    elif block and layout == "tail_after":
        content = f"{user}\n\n{block}"
    return msgs + [{"role": "user", "content": content}]
    ### END SOLUTION

# %% check
store = SQLiteMemoryStore(":memory:")
cb.seed_memory(store)
mem = LocalMemory(store, "acme", "u1")
hist = [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": "Hi!"}]
for layout in ("before_history", "tail", "tail_after"):
    agent = MemoryAgent(ScriptedModel(), mem, mode="implicit", layout=layout, instruction=cb.SYSTEM)
    agent.pinned = "PROFILE" if layout == "tail" else None
    block = mem.render(mem.recall("Which city is my home city?", 3))
    assert build_prompt(layout, cb.SYSTEM, agent.pinned, block, hist, "Q?") == agent.build(hist, "Q?", block), layout
assert build_prompt("tail", "S", None, None, [], "Q?") == [{"role": "system", "content": "S"}, {"role": "user", "content": "Q?"}]
print("✅ build_prompt matches the agent's layouts")

# %% [markdown]
# ## Exercise 3.2 — find where the prefix breaks
#
# Write `first_divergence(a, b)`. It returns the index of the first position where the token lists `a` and
# `b` are different. If one is a prefix of the other, it returns `min(len(a), len(b))`. Then write
# `lost_tokens(prev, new)`. It returns the number of tokens of `new` that the engine must prefill again:
# `len(new) - expected_cached_tokens(prev, new)`.
#
# The check renders turn 2 and turn 3 of the before-history layout through the template and the tokenizer of
# the fake server. Then it asks where they become different. That point must be inside the memory block of
# turn 3. The engine calculates everything after that point again.

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
# Write `ttft_ms(prompt_tokens, cached_tokens, gpu, llm)`. It returns the time of the prefill step of one
# request on an idle engine, in milliseconds: `step_cost(GPUS[gpu], LLMS[llm], [(cached, prompt - cached)])["t"]`
# (at least one new token). This is the roofline of `minengine.perf.step_cost`, written again in `cachebench`:
# $\max\bigl(\tfrac{\text{bytes}}{0.8 \cdot \text{BW}}, \tfrac{\text{FLOPs}}{0.6 \cdot \text{peak}}\bigr) + 2\ \text{ms}$.
# The check makes sure that it gives the same numbers as the model of the serving engine.

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
# ## Exercise 3.4 — what a miss costs in dollars, on a hosted API
#
# Write `session_cost(rows, price, completion=60)`. It returns the sum over the rows of what a hosted API
# bills: (uncached input × `price.input` + cached input × `price.cached_input` + output × `price.output`) /
# 1e6. Use these values:
#
# * the `prompt_tokens` of the row.
# * `completion` output tokens per call.
# * as cached input, the `cached_tokens` of the row (0 when it is unknown), **only if**
#   `prompt_tokens >= price.min_cached_prompt`. If not, use 0. `price.min_cached_prompt` is the caching
#   minimum of the provider (4,096 on Gemini 3.x).
#
# The check runs it two times. The first run uses the prices of `gemini-3.5-flash` with that minimum. The
# second run uses a provider that bills every engine hit as cached (`min_cached_prompt=0`). (The check also
# reproduces the call of scaling primer §3.4. In that call, 5,000 input tokens with 2,700 cached and 350
# output at the prices of 5 Sep 2026 (verify) cost $0.007005.)

# %% exercise
def session_cost(rows, price, completion=60):
    ### BEGIN SOLUTION
    total = 0.0
    for r in rows:
        cached = (r.cached_tokens or 0) if r.prompt_tokens >= price.min_cached_prompt else 0
        total += ((r.prompt_tokens - cached) * price.input + cached * price.cached_input
                  + completion * price.output) / 1e6
    return total
    ### END SOLUTION

# %% check
P = PRICES["gemini-3.5-flash"]
assert math.isclose(turn_cost(P, 5000, 350, 2700), 0.007005) and P.min_cached_prompt == 4096
every_hit = cb.Price(P.input, P.output, P.cached_input, 0, "every engine hit billed")
cost = {k: session_cost(r.rows, P) for k, r in runs.items()}
ideal = {k: session_cost(r.rows, every_hit) for k, r in runs.items()}
for k, r in runs.items():
    assert math.isclose(cost[k], sum(turn_cost(P, x.prompt_tokens, 60, x.cached_tokens or 0) for x in r.rows)), k
assert abs(cost["before_history"] / cost["tail"] - 1) < 0.01 < ideal["before_history"] / ideal["tail"] - 1
print("✅ $ per 8-turn session [gemini-3.5-flash list prices, 4,096-token minimum (verify)]:",
      {k: round(v, 5) for k, v in cost.items()})
print(f"   below the minimum before_history costs {cost['before_history'] / cost['tail']:.2f}x tail; were every engine "
      f"hit billed cached it would be {ideal['before_history'] / ideal['tail']:.2f}x")

# %% [markdown]
# ## Exercise 3.5 — one salt per tenant
#
# When tenants share a prefix cache, the cache is a side channel. If the first request of tenant B gets a hit
# on blocks that tenant A made, B learns that A sent that prefix. The TTFT shows it. In vLLM, the
# key of the *first* block contains the `cache_salt` of the request. The key of every later block chains
# from it (vllm-internals §4.3). Thus a per-tenant salt gives each tenant its own cache.
#
# Write `tenant_salt(secret, tenant)`. It returns an HMAC-SHA256 of the tenant under a server-side secret, as
# base64url without padding. That is 43 characters, in the vLLM limit of 128, with no `@ / \`.

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
# Start vLLM with prefix-caching counters per request. Then point this notebook at it, and run it again from
# the top. The **hit rates** in the cells before become measurements. The `predicted` column becomes empty,
# because the tokenizer is that of the model. The prefill column stays the roofline model. Pass `gpu=` and
# `llm=` for the card and the model that you serve.
#
# The dollars stay list prices, and the summary gives a label to each column. To measure time, send a
# streaming request and measure the time to its first token. Keep the salt per run, so that one run does not
# make the cache warm for another run.

# %%
if real:
    print(f"[cached tokens MEASURED on {real}; prefill ms and $ modelled]")
    print(cb.summary(cb.compare_layouts(real, turns=8, api_key=os.environ.get("MEMLAB_API_KEY"),
                                        gpu=os.environ.get("MEMLAB_GPU", "L4"),
                                        llm=os.environ.get("MEMLAB_ROOFLINE_LLM", "qwen2.5-1.5b"))))
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
# **Two minutes.** "The agent retrieves memory again every turn, thus memory is the part of the prompt that
# changes most. The prefix cache only uses blocks again from token zero. If we inject memory after the system
# prompt, every history block after it misses, and each turn prefills the whole conversation again.
#
# "In the simulated session of this notebook, that layout gets a 38% hit rate. The same memory at the tail of
# the prompt gets 67% (77% when the block comes after the user's text). The prefill time on our own engine is
# also about two times larger (a roofline estimate, while the T1 run measures the hit rates).
#
# "On a hosted API, these prompts are below the caching minimum of the provider. Thus the bill almost does
# not change until the prompt reaches that minimum. The cost is the latency and the GPU time.
#
# "Thus we put stable things first: the system prompt, the tools, and a pinned per-session profile, sorted so
# that it is byte-identical. Per-turn memory goes at the tail, request-scoped. We monitor `cached_tokens` per
# request and `prefix_cache_hits / queries` per replica. Also, we give the cache a salt per tenant, so that
# the prefix of one tenant is never a hit for another tenant."
#
# **Drill 1.** *After we added memory, TTFT p50 tripled and input cost rose. Why, and what do you change?*
# The block went in before the history. Thus the prefix changed every turn, and the engine prefilled the
# conversation again each time. Do these steps:
#
# * Move per-turn memory to the tail (or use a pinned per-session profile).
# * Limit it to the recall knee (notebook 05).
# * Make sure with `cached_tokens` that the change works.
#
# **Drill 2.** *The pinned layout had the best hit rate but did not have the lowest cost. Why?* Its `recall`
# tool adds a model call per question, and each call sends the prompt again. Hits make that call low-cost,
# but not free. How often the profile alone gives the answer decides if the profile is better than per-turn
# retrieval.
#
# **Drill 3.** *Why is `cache_salt` a secret per tenant, and not the tenant id?* If someone can guess the salt
# and send requests with it, that person can probe the cache of that tenant. The guidance of vLLM is a random
# value that nobody can predict (in v0.30.0 it enforces at most 128 characters, verify).
