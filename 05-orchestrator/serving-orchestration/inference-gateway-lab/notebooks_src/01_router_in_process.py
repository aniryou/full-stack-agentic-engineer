# %% [markdown]
# # 01 · A cache-aware router, in process
#
# **Tier:** T0 — laptop or Colab CPU; no GPU, no Docker, no network. Three vLLM-shaped fake
# backends and the router start inside this notebook on free localhost ports and talk real HTTP.
# Backend *timing* is emulated (`igwlab/fakebackend.py`: prefill cost per uncached token, decode
# cost per token, batch slots), so every latency below is "measured on this machine, emulated
# backend": good for comparing routing policies, not a GPU benchmark.
#
# ## The one-minute version
#
# An LLM replica is a stateful cache with a queue in front of it: a request whose prompt prefix
# is already in *that* replica's KV cache skips most of its prefill. So where a request goes
# changes how much work it is. An LLM-aware router therefore:
#
# 1. **remembers what it sent where** — an approximate prefix index of chained block hashes;
# 2. **reads each replica's load** from its `/metrics` (vLLM's `num_requests_waiting`, `kv_cache_usage_perc`);
# 3. **scores every candidate per signal and adds the scores with weights** (filters → scorers → picker);
# 4. **streams the response back byte for byte**.
#
# After this notebook you can explain each step with numbers, and show on an agentic workload why
# round-robin loses. Background: [PRIMER §1 Why a layer above the engine and §2 Routing signals and
# algorithms](../../PRIMER.md).

# %%
import json
import urllib.request

from igwlab.promtext import Families
from igwlab.router import extract_vllm
from igwlab.stack import LocalStack

stack = LocalStack(n_backends=3, config="round-robin").start()
print("router  :", stack.router_url)
print("backends:", stack.backend_urls)


def chat(url, messages, max_tokens=8, stream=False, headers=None):
    """POST /v1/chat/completions; returns (response headers, raw body bytes)."""
    body = {"model": "lab/llm", "messages": messages, "max_tokens": max_tokens, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return dict(r.headers), r.read()


# %% [markdown]
# ## One request, end to end
#
# The router picks a backend, forwards the unmodified body, and streams the Server-Sent Events back
# as they arrive. It adds one header, `x-gateway-destination-endpoint` — the same header name the
# llm-d EPP uses to tell Envoy which pod to send the request to. The fake backend also names
# itself in `x-inference-pod` (as `llm-d-inference-sim` does).

# %%
hdrs, raw = chat(stack.router_url, [{"role": "user", "content": "Say hello."}], max_tokens=4, stream=True)
print("routed to:", hdrs["x-gateway-destination-endpoint"], "| served by:", hdrs["x-inference-pod"])
print(raw.decode()[:600])

# %% [markdown]
# ## What the router scrapes
#
# Every 50 ms (llm-d's default base tick) the router GETs each backend's `/metrics` and keeps the
# few gauges it routes on. Below: the raw vLLM-named series, then the router's view of them.

# %%
text = stack.backend_metrics()["a"]
print("\n".join(l for l in text.splitlines() if l.startswith(("vllm:num_requests", "vllm:kv_cache_usage",
                                                              "vllm:prefix_cache", "vllm:cache_config"))))
print("\nrouting view:", extract_vllm(Families.from_text(text)))

# %% [markdown]
# ## Remembering prefixes: pseudo-tokens and chained block hashes
#
# The router does not run the model's tokenizer. Like the EPP's default `estimate` token producer
# it packs the request bytes (tool schemas, then role + content of each message) into **4-byte
# pseudo-tokens**, cuts them into blocks of 64 and hashes each block *together with the previous
# block's hash*. Two prompts that agree for the first N blocks therefore share exactly N hashes.

# %%
from igwlab.router import PrefixIndex, block_hashes, estimate_tokens

system = "You are a coding agent. Plan, call tools, verify results, report. " * 40
turn1 = {"model": "lab/llm", "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": "Fix the failing test in utils.py"}]}
turn2 = {"model": "lab/llm", "messages": turn1["messages"] + [
    {"role": "assistant", "content": "I will read utils.py first."},
    {"role": "user", "content": "Tool result: " + "def f(x): return x + 1\n" * 30}]}
t1, t2 = estimate_tokens(turn1), estimate_tokens(turn2)
h1, h2 = block_hashes(t1, 64, "lab/llm"), block_hashes(t2, 64, "lab/llm")
shared = next((i for i, (a, b) in enumerate(zip(h1, h2)) if a != b), min(len(h1), len(h2)))
print(f"turn 1: {len(t1)} pseudo-tokens -> {len(h1)} blocks | turn 2: {len(t2)} -> {len(h2)} blocks | shared: {shared}")

idx = PrefixIndex()
idx.add(h1, "a")                        # "we routed turn 1 to a"
print("turn-2 match per endpoint:", idx.match(h2), f"-> prefix score on a = {idx.match(h2)['a'] / len(h2):.2f}")

# %% [markdown]
# Turn 1's *last* block was partial (fewer than 64 tokens), so in turn 2 — where the conversation
# continues — that block has different content and a different hash. The router can only count
# full-block agreement. That is one of the reasons the index is *approximate*.
#
# ## Exercise 1.1 — the prefix-cache score
#
# Implement the `prefix-cache-scorer` formula (llm-d router, v0.10):
#
# `score = w * min(1, matched_tokens / scale)^2 + (1 - w) * match_blocks / total_blocks`,
# with `matched_tokens = match_blocks * block_size`; a request with `total_blocks == 0` scores 0.
# With the default `w = 0` it is just the fraction of the prompt's blocks already cached there.

# %% exercise
def prefix_score(match_blocks, total_blocks, block_size=64, w=0.0, scale=8192):
    ### BEGIN SOLUTION
    if total_blocks <= 0:
        return 0.0
    ratio = match_blocks / total_blocks
    length = min(1.0, match_blocks * block_size / scale) ** 2
    return w * length + (1.0 - w) * ratio
    ### END SOLUTION

# %% check
from igwlab.router.plugins import PrefixCacheScorer
assert prefix_score(3, 4) == 0.75
assert abs(prefix_score(3, 4, 64, w=0.5, scale=512) - 0.4453125) < 1e-12     # 0.5*(192/512)^2 + 0.5*0.75
assert prefix_score(0, 0) == 0.0
for args in [(10, 16, 64, 0.3, 4096), (16, 16, 64, 1.0, 512), (1, 40, 64, 0.0, 8192)]:
    assert abs(prefix_score(*args) - PrefixCacheScorer.formula(*args)) < 1e-12
print("✅ prefix_score matches the prefix-cache-scorer formula")

# %% [markdown]
# ## Exercise 1.2 — the queue score
#
# `queue-scorer` turns the scraped `vllm:num_requests_waiting` of each endpoint into a score with a
# min-max normalization: `(maxQ - q) / (maxQ - minQ)`. Two details matter:
#
# * if every scraped endpoint has the same queue, they all get the neutral score **1.0**;
# * an endpoint that has never been scraped (`None` here) is **left unscored** — omitted from the
#   result (it then contributes 0 to the weighted total) and excluded from the min/max.

# %% exercise
def queue_scores(waiting: dict) -> dict:
    ### BEGIN SOLUTION
    known = {k: v for k, v in waiting.items() if v is not None}
    if not known:
        return {}
    hi, lo = max(known.values()), min(known.values())
    if hi == lo:
        return {k: 1.0 for k in known}
    return {k: (hi - v) / (hi - lo) for k, v in known.items()}
    ### END SOLUTION

# %% check
assert queue_scores({"a": 0, "b": 5, "c": 10}) == {"a": 1.0, "b": 0.5, "c": 0.0}
assert queue_scores({"a": 3, "b": 3}) == {"a": 1.0, "b": 1.0}
assert queue_scores({"a": 4, "b": None, "c": 8}) == {"a": 1.0, "c": 0.0}
assert queue_scores({"a": None}) == {}
print("✅ queue_scores normalizes like queue-scorer")

# %% [markdown]
# ## Exercise 1.3 — the weighted pick
#
# The scheduler adds, for each endpoint, `weight × clamp(score, 0, 1)` over all scorers (an
# unscored endpoint contributes 0 for that scorer) and the `max-score-picker` takes the highest
# total. Implement `pick(scores, weights)` → the winning endpoint name; break exact ties by the
# alphabetically smallest name (the lab's picker rotates ties round-robin; in the real EPP the
# candidate order is randomized, so ties there land effectively at random).

# %% exercise
def pick(scores: dict, weights: dict) -> str:
    """scores: {scorer: {endpoint: score}}, weights: {scorer: weight}."""
    endpoints = sorted({e for s in scores.values() for e in s})
    ### BEGIN SOLUTION
    total = {e: 0.0 for e in endpoints}
    for scorer, per_ep in scores.items():
        for e, s in per_ep.items():
            total[e] += weights[scorer] * min(1.0, max(0.0, s))
    return min(endpoints, key=lambda e: (-total[e], e))
    ### END SOLUTION

# %% check
s = {"queue": {"a": 1.0, "b": 0.0}, "kv": {"a": 0.1, "b": 0.9}, "prefix": {"a": 0.0, "b": 1.0}}
assert pick(s, {"queue": 2, "kv": 2, "prefix": 3}) == "b"        # a: 2.2, b: 1.8 + 3.0 = 4.8
assert pick(s, {"queue": 2, "kv": 2, "prefix": 0}) == "a"        # a: 2.2 vs b: 1.8
assert pick({"x": {"a": 1.7, "b": 1.0}}, {"x": 1}) == "a"          # clamped to 1.0: tie -> "a"
assert pick({"x": {"a": 0.2}, "y": {"b": 0.5}}, {"x": 1, "y": 1}) == "b"
print("✅ pick reproduces the weighted-sum scheduler")

# %% [markdown]
# ## Round-robin versus the weighted scorer on an agentic workload
#
# 18 agent sessions × 5 turns. Each session belongs to one of 4 agent "programs" (a ~7 KB system
# prompt + 3 tool schemas it re-sends on every call) and every turn re-sends the whole history plus
# a new ~2 KB tool result. Sessions start over the first 0.5 s and run closed-loop (next turn after
# the previous reply + 0–20 ms of tool time). We run the same sessions against two fresh stacks:
#
# * `round-robin` — the preset with no scorers (every total ties at 0; the picker rotates);
# * `default-weighted` — the llm-d Helm chart's default: `prefix-cache-scorer` ×3,
#   `queue-scorer` ×2, `kv-cache-utilization-scorer` ×2.
#
# `hit rate` = prompt tokens the engines reported as cached / all prompt tokens.

# %%
from igwlab.bench import agentic_sessions, ascii_bars, compare

stack.stop()
sessions = agentic_sessions(n_sessions=18, turns=5, n_agents=4, system_words=1200, tool_words=400, seed=0)
results = {}
for cfg in ("round-robin", "default-weighted"):
    with LocalStack(3, cfg) as s:
        results[cfg] = s.bench(sessions, label=cfg)
print(compare(results.values()))
print("\nTTFT p50 (ms, emulated backend):")
print(ascii_bars({k: r.summary()["ttft_p50_ms"] for k, r in results.items()}))

# %% [markdown]
# Two things to notice. The weighted router's **hit rate** is higher because each session's next
# turn goes back to the replica that holds its history (round-robin re-prefills the history ~2/3
# of the time). And **TTFT** falls further than the hit rate suggests, because every avoided
# prefill also shortens the queue of prefills in front of everyone else on that replica. The
# price is imbalance: requests follow the cache, not an even split.
#
# ## Exercise 1.4 — hit rate from the engines' own counters
#
# The benchmark computed the hit rate from each response's `usage.prompt_tokens_details`. An
# operator would read it from Prometheus instead: `vllm:prefix_cache_hits_total` and
# `vllm:prefix_cache_queries_total` are counters **in tokens**, so the hit rate over an interval is
# the ratio of their *increases*, summed over replicas. Implement it from two scrapes of every
# backend (`before`, `after`: `{name: metrics text}`).

# %% exercise
def hit_rate_from_counters(before: dict, after: dict) -> float:
    ### BEGIN SOLUTION
    hits = queries = 0.0
    for name, text in after.items():
        a, b = Families.from_text(text), Families.from_text(before[name])
        hits += a.sum("vllm:prefix_cache_hits_total", 0.0) - b.sum("vllm:prefix_cache_hits_total", 0.0)
        queries += a.sum("vllm:prefix_cache_queries_total", 0.0) - b.sum("vllm:prefix_cache_queries_total", 0.0)
    return hits / queries if queries else 0.0
    ### END SOLUTION

# %% check
with LocalStack(3, "default-weighted") as s:
    before = s.backend_metrics()
    r = s.bench(sessions[:6], label="check", turns=3)
    after = s.backend_metrics()
got, want = hit_rate_from_counters(before, after), r.summary()["hit_rate"]
assert abs(got - want) < 1e-9, (got, want)
print(f"✅ counters and usage agree: hit rate {got:.1%}")

# %% [markdown]
# ## Exercise 1.5 — decision forensics
#
# Every routing decision is recorded (`stack.router.decisions`, and `GET /debug/state`). When you
# are asked "why did this request go to b?", the useful answer names the **decisive scorer**: the
# one whose *weighted* contribution differs most between the winner and the runner-up.
# Implement `decisive_scorer(decision)` using `decision.scores` (`{scorer: {endpoint: score}}`),
# `decision.weights` and `decision.totals`; unscored endpoints count as 0; clamp scores to [0, 1].

# %% exercise
def decisive_scorer(decision) -> str:
    ranked = sorted(decision.totals, key=lambda e: -decision.totals[e])
    winner, runner_up = ranked[0], ranked[1]
    ### BEGIN SOLUTION
    def contrib(scorer, e):
        return decision.weights[scorer] * min(1.0, max(0.0, decision.scores[scorer].get(e, 0.0)))
    return max(decision.scores, key=lambda sc: contrib(sc, winner) - contrib(sc, runner_up))
    ### END SOLUTION

# %% check
from igwlab.router import Decision
d = Decision("r", ["a", "b"], scores={"queue-scorer": {"a": 1.0, "b": 0.0}, "prefix-cache-scorer": {"a": 0.0, "b": 1.0},
                                      "kv-cache-utilization-scorer": {"a": 0.2, "b": 0.9}},
             weights={"queue-scorer": 2, "prefix-cache-scorer": 3, "kv-cache-utilization-scorer": 2},
             totals={"a": 2.4, "b": 4.8}, picked=["b"])
assert decisive_scorer(d) == "prefix-cache-scorer"
with LocalStack(3, "default-weighted") as s:
    s.bench(sessions[:3], label="forensics", turns=2)
    last = s.last_decision()
print(last.table())
print("decisive:", decisive_scorer(last))
print("✅ decisive_scorer explains a decision")

# %% [markdown]
# ## In a design review
#
# **Two-minute walkthrough.** "Our replicas are caches, so we route like a cache-aware load
# balancer, not like a web LB. For each request the router hashes the prompt into 64-token blocks
# (chained hashes, so equal hashes mean an equal prefix), looks up which replica it last sent each
# block to, and turns that into a prefix score per replica. It also scrapes every replica's vLLM
# metrics every 50 ms — waiting queue and KV-cache usage — and scores those. The total is a
# weighted sum (the llm-d default is prefix 3, queue 2, KV 2), the highest total wins, and ties
# rotate. We stream the bytes back untouched. On a multi-turn agent workload that keeps each
# session on the replica that holds its history: fewer prefilled tokens, shorter prefill queues,
# lower TTFT; the cost is an uneven split, which the load scores bound."
#
# **Drill questions**
#
# 1. *Round-robin eventually caches the system prompt everywhere — so why does it still lose?*
#    Each session's growing history is cached only where its last turn ran; round-robin sends the
#    next turn elsewhere ⅔ of the time and re-prefills it. It also stores every prefix on every
#    replica, dividing effective cache capacity by the replica count.
# 2. *Name three ways the prefix index can be wrong.* It is written at routing time (before the
#    engine caches anything); engines evict blocks the index still lists; pseudo-token blocks do not
#    align with the engine's 16-token KV blocks; traffic from another router replica is invisible.
#    Precise alternatives subscribe to the engines' KV-cache events.
# 3. *Why must the router pass SSE bytes through unmodified?* Re-encoding adds latency to every
#    token, can split multi-byte characters or events, and breaks usage chunks and `[DONE]`
#    handling in clients; TTFT/ITL are user-visible.
