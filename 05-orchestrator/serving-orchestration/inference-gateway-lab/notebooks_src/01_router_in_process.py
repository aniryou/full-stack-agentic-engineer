# %% [markdown]
# # 01 · A cache-aware router, in process
#
# **Tier:** T0. It runs on a laptop or Colab CPU, with no GPU, no Docker and no network. Three
# vLLM-shaped fake backends and the router start inside this notebook on free localhost ports, and
# they talk real HTTP. `igwlab/fakebackend.py` emulates the backend *times*: prefill cost per
# uncached token, decode cost per token, batch slots. Thus every latency in this notebook is
# "measured on this machine, emulated backend". It is good for a comparison of routing policies,
# but it is not a GPU benchmark.
#
# ## The one-minute version
#
# An LLM replica is a stateful cache with a queue in front of it. A request whose prompt prefix is
# already in the KV cache of *that* replica skips most of its prefill. Thus the destination of a
# request changes how much work the request is. Thus an LLM-aware router does these steps:
#
# 1. **remembers what it sent where**: it keeps an approximate prefix index of chained block hashes.
# 2. **reads each replica's load** from its `/metrics` (vLLM's `num_requests_waiting`, `kv_cache_usage_perc`).
# 3. **scores every candidate per signal and adds the scores with weights** (first filters, then scorers, then the picker).
# 4. **streams the response back byte for byte**.
#
# After this notebook, you can explain each step with numbers. You can also show on an agentic
# workload why round-robin loses. Background: [PRIMER §1 Why a layer above the engine and §2 Routing signals and
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
# The router selects a backend and forwards the unchanged body. Then it streams the Server-Sent
# Events back when they arrive. It adds one header, `x-gateway-destination-endpoint`. The llm-d EPP
# uses the same header name to tell Envoy which pod must get the request. The fake backend also gives
# its name in `x-inference-pod`, as `llm-d-inference-sim` does.

# %%
hdrs, raw = chat(stack.router_url, [{"role": "user", "content": "Say hello."}], max_tokens=4, stream=True)
print("routed to:", hdrs["x-gateway-destination-endpoint"], "| served by:", hdrs["x-inference-pod"])
print(raw.decode()[:600])

# %% [markdown]
# ## What the router scrapes
#
# Every 50 ms (llm-d's default base tick), the router GETs the `/metrics` of each backend. It keeps
# the few gauges that it uses to route. The next cell prints the raw vLLM-named series, then the
# router's view of them.

# %%
text = stack.backend_metrics()["a"]
print("\n".join(l for l in text.splitlines() if l.startswith(("vllm:num_requests", "vllm:kv_cache_usage",
                                                              "vllm:prefix_cache", "vllm:cache_config"))))
print("\nrouting view:", extract_vllm(Families.from_text(text)))

# %% [markdown]
# ## Remembering prefixes: pseudo-tokens and chained block hashes
#
# The router does not run the model's tokenizer. It does the same as the EPP's default `estimate`
# token producer:
#
# - It packs the request bytes into **4-byte pseudo-tokens**: first the tool schemas, then the role
#   and the content of each message.
# - It divides the pseudo-tokens into blocks of 64.
# - It hashes each block *together with the previous block's hash*.
#
# Thus two prompts that agree for the first $N$ blocks share exactly $N$ hashes.

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
# The *last* block of turn 1 was partial (fewer than 64 tokens). In turn 2, the conversation
# continues, so that block has different content and a different hash. The router can count only
# the agreement of full blocks. That is one of the reasons why the index is *approximate*.
#
# ## Exercise 1.1 — the prefix-cache score
#
# Write the code for the `prefix-cache-scorer` formula (llm-d router, v0.10):
#
# $$
# \text{score} = w \cdot \min\left(1, \frac{\mathrm{matched\_tokens}}{\text{scale}}\right)^2
#   + (1 - w) \cdot \frac{\mathrm{match\_blocks}}{\mathrm{total\_blocks}},
# $$
#
# Here, $\mathrm{matched\_tokens} =$ $\mathrm{match\_blocks} \times \mathrm{block\_size}$. A request
# with `total_blocks == 0` gets the score 0. With the default $w = 0$, the score is only the fraction
# of the prompt's blocks that are already in the cache of that endpoint.

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
# ## Exercise 1.2 — the queue score, and the replica nobody has scraped yet
#
# The `queue-scorer` changes the scraped `vllm:num_requests_waiting` of each endpoint into a score
# with a min-max normalization: $(\mathrm{maxQ} - q) / (\mathrm{maxQ} - \mathrm{minQ})$. If all
# endpoints have the same queue, they all get the neutral score **1.0**.
#
# One detail causes problems in production. The scorer reads the *current* metrics of each endpoint,
# and it does not examine their freshness (llm-d-router v0.10.0 does the same). An endpoint that the
# router has never scraped has all-zero metrics. Such an endpoint is a pod that became ready a moment
# ago (`None` here). Write `queue_scores(waiting)` with exactly that behaviour. Then answer this
# question: what is the score of a newly started replica, and what does that score do to the next burst?

# %% exercise
def queue_scores(waiting: dict) -> dict:
    ### BEGIN SOLUTION
    q = {k: (0 if v is None else v) for k, v in waiting.items()}     # never scraped -> metrics are 0
    if not q:
        return {}
    hi, lo = max(q.values()), min(q.values())
    if hi == lo:
        return {k: 1.0 for k in q}
    return {k: (hi - v) / (hi - lo) for k, v in q.items()}
    ### END SOLUTION

# %% check
from igwlab.router import Endpoint, EndpointMetrics
from igwlab.router.plugins import QueueScorer, RequestCtx
assert queue_scores({"a": 0, "b": 5, "c": 10}) == {"a": 1.0, "b": 0.5, "c": 0.0}
assert queue_scores({"a": 3, "b": 3}) == {"a": 1.0, "b": 1.0}
assert queue_scores({"a": 4, "b": None, "c": 8}) == {"a": 0.5, "b": 1.0, "c": 0.0}
eps = [Endpoint(n, "http://unused", metrics=EndpointMetrics(waiting=w or 0, update_time=0.0 if w is None else 1.0))
       for n, w in (("a", 4), ("b", None), ("c", 8))]
assert queue_scores({"a": 4, "b": None, "c": 8}) == QueueScorer().score(RequestCtx("r", "m", []), eps)
print("✅ a never-scraped replica scores 1.0: it looks idle, so until its first scrape it attracts traffic")

# %% [markdown]
# ## Exercise 1.3 — the weighted pick
#
# For each endpoint, the scheduler adds $\text{weight} \times \operatorname{clamp}(\text{score}, 0, 1)$
# over all scorers. An endpoint without a score from a scorer gets 0 for that scorer. Then the
# `max-score-picker` takes the highest total.
#
# Write `pick(scores, weights)`. It returns the name of the endpoint that wins. If two totals are
# exactly equal, select the name that is first in alphabetical order. The lab's picker rotates ties
# round-robin. The picker of llm-d-router v0.10.0 puts the candidates in a random order before a stable sort by
# score. Thus the winner of its ties is random.

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
# The workload is 18 agent sessions × 5 turns. Each session belongs to one of 4 agent "programs". A
# program is a ~8 KB system prompt and 3 tool schemas, and the session sends them again on every
# call. Every turn also sends the whole history again, plus a new ~3 KB tool result. The cell prints
# the exact sizes.
#
# The sessions start during the first 0.5 s and run closed-loop: the next turn comes after the
# previous reply plus 0–20 ms of tool time. We run the same sessions against two new stacks:
#
# * `round-robin`: the preset with no scorers. All totals are equal at 0, and the picker rotates.
# * `default-weighted`: the default of the llm-d Helm chart, with `prefix-cache-scorer` ×3,
#   `queue-scorer` ×2, `kv-cache-utilization-scorer` ×2.
#
# $$
# \text{hit rate} = \frac{\text{prompt tokens the engines reported as cached}}{\text{all prompt tokens}}.
# $$

# %%
from igwlab.bench import agentic_sessions, ascii_bars, compare

stack.stop()
sessions = agentic_sessions(n_sessions=18, turns=5, n_agents=4, system_words=1200, tool_words=400, seed=0)
print(f"system prompt {len(sessions[0].agent.system.encode()):,} B, tool schemas "
      f"{len(json.dumps(sessions[0].agent.tools)):,} B, one tool result {len(sessions[0].tool_outputs[0].encode()):,} B")
results = {}
for cfg in ("round-robin", "default-weighted"):
    with LocalStack(3, cfg) as s:
        results[cfg] = s.bench(sessions, label=cfg)
print(compare(results.values()))
print("\nTTFT p50 (ms, emulated backend):")
print(ascii_bars({k: r.summary()["ttft_p50_ms"] for k, r in results.items()}))

# %% [markdown]
# Look at two things. First, the **hit rate** of the weighted router is higher. The reason is that
# the next turn of each session goes back to the replica that holds its history. Round-robin does the
# prefill of the history again ~2/3 of the time. Second, **TTFT** decreases more than the hit rate
# suggests. Each prefill that the router prevents also makes the queue of prefills shorter for all
# other requests on that replica.
#
# The fake backend runs one prefill at a time per replica. It is like an engine whose prefills
# compete for the GPU. `llm-d-inference-sim` does not put prefills in a queue. Thus on the notebook-04
# stacks, the TTFT gap is smaller, but the hit-rate gap stays. The cost is imbalance: the requests go
# where the cache is, not in an even split.
#
# ## Exercise 1.4 — hit rate from the engines' own counters
#
# The benchmark calculated the hit rate from the `usage.prompt_tokens_details` of each response. An
# operator reads it from Prometheus instead. `vllm:prefix_cache_hits_total` and
# `vllm:prefix_cache_queries_total` are counters **in tokens**. Thus the hit rate over an interval is
# the ratio of their *increases*, with each increase added over all replicas. Write this calculation
# from two scrapes of every backend (`before`, `after`: `{name: metrics text}`).

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
# The router records every routing decision (`stack.router.decisions`, and `GET /debug/state`).
# Someone can ask you "why did this request go to b?". The useful answer gives the name of the
# **decisive scorer**. That is the scorer whose *weighted* contribution has the largest difference
# between the winner and the runner-up.
#
# Write `decisive_scorer(decision)`. Use `decision.scores` (`{scorer: {endpoint: score}}`),
# `decision.weights` and `decision.totals`. An endpoint without a score counts as 0. Clamp the scores
# to [0, 1].

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
# **Two-minute walkthrough.** "Our replicas are caches. Thus we route like a cache-aware load
# balancer, not like a web LB. For each request, the router hashes the prompt into 64-token blocks.
# The hashes form a chain, so equal hashes mean an equal prefix. The router finds the replica that it
# last sent each block to, and changes that into a prefix score per replica.
#
# "It also scrapes the vLLM metrics of every replica every 50 ms (the wait queue and the KV-cache
# usage) and changes them into scores too. The total is a weighted sum (the llm-d default is prefix
# 3, queue 2, KV 2), and the highest total wins. The picker breaks ties at random. We stream the
# bytes back unchanged.
#
# "On a multi-turn agent workload, this keeps each session on the replica that holds its history.
# The result is fewer prefilled tokens, shorter prefill queues and lower TTFT. The cost is an uneven
# split, and the load scores put a limit on it."
#
# **Drill questions**
#
# 1. *Round-robin caches the system prompt on every replica after some time. Why does it still lose?*
#    The history of each session increases, and only the replica that ran its last turn has it in its
#    cache. Round-robin sends the next turn to a different replica ⅔ of the time and prefills the
#    history again. It also keeps every prefix on every replica. This divides the effective cache
#    capacity by the number of replicas.
# 2. *Name three ways in which the prefix index can be incorrect.* The router writes the index at
#    routing time, before the engine caches anything. The engines evict blocks that the index still
#    lists. The pseudo-token blocks do not align with the engine's 16-token KV blocks. The router
#    cannot see the traffic from another router replica. Precise alternatives subscribe to the
#    KV-cache events of the engines.
# 3. *Why must the router pass SSE bytes through unchanged?* If the router encodes the bytes again,
#    it adds latency to every token. It can also divide multi-byte characters or events. It also
#    breaks how clients process the usage chunks and `[DONE]`. TTFT and ITL are visible to the user.
