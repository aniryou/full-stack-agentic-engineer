# %% [markdown]
# # 02 · Scorer weights, hot prefixes and the herd
#
# **Tier:** T0. It needs only a CPU. The fake backends and the router run in-process. The fake
# backends emulate the backend times, so the latencies are "measured on this machine, emulated backend".
#
# ## The one-minute version
#
# Cache affinity and load balance pull in opposite directions. Pure affinity sends every request for
# a popular prefix to the one replica that has it in its cache. Thus a **hot prefix** overloads that
# replica, and the other replicas are idle. Pure load balance ignores the cache. A weighted router sits
# between the two, and the weights decide exactly *how much* load a cache hit is worth.
#
# Let the prefix weight be $\mathit{wp}$, and let the load weights sum to $W$. Then a fully cached
# replica continues to win until its load scores are more than $\mathit{wp}$ below the load scores
# of a cold, idle replica.
#
# The llm-d optimized baseline replaces the arithmetic with a rule, **sticky until saturated**. First,
# it keeps only the replicas that hold ≥ 80% of the prompt. It does not use this filter if the
# estimated prefill backlog of those replicas is too far behind the best other replica. Then it
# balances the load by tokens in flight.
#
# There are two more traps:
#
# - **scrape lag**: every request between two scrapes sees the same stale numbers. Thus all of these
#   requests go as a herd to the replica that looked the most idle.
# - **thresholds that ignore the workload**: an 80% affinity threshold never applies to a turn whose
#   new tokens are more than 20% of its prompt. On the agents of this lab, that is the first resumed
#   turn of every session.
#
# Background:
# [PRIMER §2 Routing signals and algorithms and §3 Flow control and priorities](../../PRIMER.md).

# %%
import copy
import threading
import time

from igwlab.bench import agentic_sessions, ascii_bars, burst, compare
from igwlab.router import load_config
from igwlab.stack import LocalStack

# %% [markdown]
# ## A hot prefix
#
# The agentic sessions are the same as in notebook 01, but 80% of them belong to one agent program
# (`hot_share=0.8`). Thus one system prompt dominates the traffic. There are four policies, each on a
# new stack:
#
# | preset | what it does |
# |---|---|
# | `prefix-only` | `prefix-cache-scorer` alone |
# | `load-only` | `queue-scorer` + `kv-cache-utilization-scorer` (cache-blind) |
# | `default-weighted` | prefix ×3, queue ×2, KV ×2 (the llm-d chart default) |
# | `sticky-until-saturated` | `prefix-cache-affinity-filter` (≥ 0.8, TTFT gate), then `token-load-scorer` |

# %%
from collections import Counter

hot = agentic_sessions(n_sessions=18, turns=5, n_agents=4, hot_share=0.8, system_words=1200, tool_words=400, seed=0)
print("sessions per agent program:", dict(Counter(s.agent.name for s in hot)))
hot_results = {}
for cfg in ("prefix-only", "load-only", "default-weighted", "sticky-until-saturated"):
    with LocalStack(3, cfg) as s:
        hot_results[cfg] = s.bench(hot, label=cfg)
print(compare(hot_results.values()))
print("\nTTFT p90 (ms, emulated backend):")
print(ascii_bars({k: r.summary()["ttft_p90_ms"] for k, r in hot_results.items()}))
p90 = {k: r.summary()["ttft_p90_ms"] for k, r in hot_results.items()}
assert p90["sticky-until-saturated"] < p90["default-weighted"] < p90["prefix-only"], p90   # what the text below reads

# %% [markdown]
# `prefix-only` sends all requests for the hot program to one replica. The result is a top hit rate
# and by far the worst latency. The reason is that the prefill queue of that replica increases while
# two replicas are idle. `load-only` spreads the requests evenly, and the cost is that the replicas
# prefill the histories again. The weighted and sticky policies both keep most of the hits, and they
# also let the load move traffic off a busy replica. When a second replica has served the hot prefix,
# the index lists it too, and both replicas become "sticky".
#
# **This ranking belongs to this engine and this workload.** Here, `sticky-until-saturated` has a
# clearly shorter tail (p90) than the 3:2:2 chart default. But the primer's simulated fleet
# ([PRIMER §2.5](../../PRIMER.md)) puts the 3:2:2 EPP ahead of it.
#
# One reason is the fake backend. It runs one prefill at a time per replica, so the tokens in the
# prefill queue *are* the delay. Two parts fit this backend well. The first is a filter with a gate on exactly
# that value (in-flight uncached tokens ÷ prefill throughput). The second is a scorer that balances
# tokens. But the queue and KV scores of 3:2:2 see a request count and memory, not prefill work.
#
# On an engine with chunked prefill, other prompt lengths or a hotter prefix, the order can change to
# the opposite. That is why you adjust the weights on a replay of your own traffic.
#
# ## Exercise 2.1 — when does affinity lose?
#
# A *sticky* replica S holds a fraction $r$ of the prompt. It has the load scores $q_s$ (queue) and
# $\mathit{kv}_s$ (KV). A *cold* replica C has no prefix, but it is idle: its queue and KV scores are
# both 1.0. Write `sticky_wins(wp, wq, wkv, r, q_s, kv_s)`. Make it return `True` if the weighted
# total of S is **strictly** higher than the total of C.

# %% exercise
def sticky_wins(wp, wq, wkv, r, q_s, kv_s) -> bool:
    ### BEGIN SOLUTION
    sticky = wp * r + wq * q_s + wkv * kv_s
    cold = wp * 0.0 + wq * 1.0 + wkv * 1.0
    return sticky > cold
    ### END SOLUTION

# %% check
assert sticky_wins(3, 2, 2, 1.0, 0.0, 0.6) is True      # 3 + 0 + 1.2 = 4.2 > 4
assert sticky_wins(3, 2, 2, 1.0, 0.0, 0.4) is False     # 3.8 < 4: the busiest-queue replica loses
assert sticky_wins(3, 2, 2, 0.5, 0.5, 1.0) is True      # 1.5 + 1 + 2 = 4.5 > 4
assert sticky_wins(3, 2, 2, 0.5, 0.0, 1.0) is False     # 1.5 + 0 + 2 = 3.5 < 4
print("✅ sticky_wins: a half-cached replica with a half-full queue still beats a cold idle one (4.5 vs 4.0)")

# %% [markdown]
# It is easy to predict this incorrectly by eye. That is why the router records a per-scorer table
# for every decision.
#
# ## Exercise 2.2 — the weight that makes affinity unconditional
#
# Start from the weighted sum itself. Each load score is in [0, 1]. Thus load scorers whose weights
# sum to $W$ can move a total by $W$ at the most. For this reason, a replica with prefix ratio $r$
# wins against *any* cold replica at any load only if $\mathit{wp} \cdot r > W$.
#
# The `prefix-cache-affinity-filter` README gives the reason why the filter exists: a weighted
# prefix scorer with a max-score picker makes popular prefixes into hot spots. The inequality is
# ours, not the README's.
#
# Write `min_prefix_weight(W, r)`. It returns the smallest weight for which a replica at ratio $r$
# cannot lose to a cold one. Use `>=`. At exact equality, the picker breaks the tie.

# %% exercise
def min_prefix_weight(load_weight_sum: float, r: float) -> float:
    ### BEGIN SOLUTION
    return load_weight_sum / r
    ### END SOLUTION

# %% check
assert min_prefix_weight(4, 1.0) == 4.0
assert min_prefix_weight(4, 0.8) == 5.0
assert abs(min_prefix_weight(2, 0.75) - 8 / 3) < 1e-12
print("✅ with the chart default (3 vs 2+2=4) a fully cached replica CAN lose to a cold idle one: 3 < 4")

# %% [markdown]
# ## The herd: why scraped metrics alone are not enough
#
# The router scrapes the metrics every 50 ms. Between two scrapes, every routing decision sees the
# same numbers.
#
# In the next cell, each replica has **6 batch slots**. Replicas **b** and **c** each get 3 requests
# that do *not* go through this router, and these requests make them busy. These requests are like
# the traffic of another router replica or a batch job. We send them directly to b and c. Thus the
# KV usage of b and c is higher by a small quantity at the last scrape.
#
# Then we keep the view of the router the same: we stop the scraper after one refresh. Then we send a
# burst of 12 requests at the same time. The free slots are a 6, b 3, c 3, which is exactly 12.
# Compare a cache-blind policy on **scraped** signals (`load-only`) with a policy on the router's
# **own in-flight counts** (`active-requests`).

# %%
from igwlab.fakebackend import EngineProfile

def herd_experiment(config, n=12, settings=None):
    with LocalStack(3, config, settings=settings, profile=EngineProfile(max_num_seqs=6)) as s:
        external = [{"model": "lab/llm", "messages": [{"role": "user", "content": f"external job {i}"}],
                     "max_tokens": 300, "stream": True} for i in range(3)]
        load = [threading.Thread(target=burst, args=(s.backend_urls[b], external)) for b in ("b", "c")]
        for t in load:
            t.start()
        time.sleep(0.3)                                     # b and c are now decoding 3 requests each
        s.run(s.router.scraper.refresh())                   # one scrape sees them...
        s.run(s.router.scraper.stop())                      # ...then the view is frozen (a long scrape interval)
        view = s.call(lambda: {e.name: (e.metrics.running, round(e.metrics.kv_usage, 4)) for e in s.router.ds.list()})
        # 64 output tokens (~0.3 s): no burst request can finish while the burst is still being dispatched
        bodies = [{"model": "lab/llm", "messages": [{"role": "user", "content": f"burst {i}"}], "max_tokens": 64,
                   "stream": True} for i in range(n)]
        recs = burst(s.router_url, bodies)
        for t in load:
            t.join()
    counts = {e: sum(r.endpoint == e for r in recs) for e in ("a", "b", "c")}
    return view, counts, sorted(r.ttft * 1e3 for r in recs)

for cfg in ("load-only", "active-requests"):
    view, counts, ttfts = herd_experiment(cfg)
    print(f"{cfg:>16}: view at last scrape (running, kv) {view} -> burst went {counts}; "
          f"burst TTFT p50 {ttfts[len(ttfts) // 2]:.0f} ms, max {ttfts[-1]:.0f} ms")

# %% [markdown]
# ## Exercise 2.3 — predict the burst
#
# Do not examine the numbers from the previous cell too closely yet. First, think about the result.
# Then write the count per replica that each policy gives for the 12-request burst (a dict
# `{"a": .., "b": .., "c": ..}`):
#
# * `load-only`: all queue scores are equal, because no request *waits*. The KV score of **a** is
#   higher, but only by a small quantity. Does the size of the difference matter to `max-score-picker`?
# * `active-requests`: the router counts only the requests that *it* has in flight. It updates the
#   counts when it dispatches a request.

# %% exercise
def predict_burst(config: str) -> dict:
    ### BEGIN SOLUTION
    if config == "load-only":
        return {"a": 12, "b": 0, "c": 0}          # a strictly highest total until the next scrape
    return {"a": 4, "b": 4, "c": 4}               # instant counters spread the burst evenly
    ### END SOLUTION

# %% check
for cfg in ("load-only", "active-requests"):
    _, measured, _ = herd_experiment(cfg)
    assert predict_burst(cfg) == measured, (cfg, measured)
print("✅ predictions match: scraped signals herd, local counters spread (but are blind to b/c's external load)")

# %% [markdown]
# Neither view is correct alone. The scraped view sends all 12 requests as a herd to **a**, which
# has 6 slots. Thus half of the burst waits until the first half finishes. The local view spreads the
# requests 4/4/4 and sends 8 requests to b and c, which have 3 free slots each. On each of b and c,
# one request finds all slots full and waits until one slot becomes free.
#
# Combine the two signals: `running-requests-size-scorer` and `active-request-scorer`. The first uses
# scraped data: it knows about the foreign load on b and c, but its data does not change. The second is local: it is
# instant, but partial. The *ratio* of their weights decides if freshness or knowledge wins:

# %%
def mix(w_running, w_active):
    return {"apiVersion": "llm-d.ai/v1", "kind": "EndpointPickerConfig",
            "plugins": [{"type": "running-requests-size-scorer"}, {"type": "active-request-scorer"}],
            "schedulingProfiles": [{"name": "default", "plugins": [
                {"pluginRef": "running-requests-size-scorer", "weight": w_running},
                {"pluginRef": "active-request-scorer", "weight": w_active}]}]}

mixed = {}
for w in ((2, 1), (1, 1), (1, 2)):
    _, counts, ttfts = herd_experiment(mix(*w))
    mixed[w] = (counts, ttfts)
    print(f"running x{w[0]} + active x{w[1]}: burst went {counts}; TTFT p50 {ttfts[len(ttfts) // 2]:.0f} ms, max {ttfts[-1]:.0f} ms")
alone = {cfg: herd_experiment(cfg)[2][-1] for cfg in ("load-only", "active-requests")}
counts, ttfts = mixed[(1, 2)]
assert counts == {"a": 6, "b": 3, "c": 3}, counts                  # every request gets a free slot
assert ttfts[-1] < 0.5 * min(alone.values()), (ttfts[-1], alone)   # and no one waits for a slot
print(f"✅ running x1 + active x2 fills the free slots exactly: max TTFT {ttfts[-1]:.0f} ms vs "
      f"{alone['load-only']:.0f} ms (scraped only) and {alone['active-requests']:.0f} ms (local only)")

# %% [markdown]
# When the local signal has the higher weight, the replica that is actually idle gets the largest
# share. The busy replicas *also* take exactly what they have space for. Here, 6/3/3 fills the
# 6 + 3 + 3 free slots. Thus no request waits, and the worst TTFT decreases several-fold against
# either signal alone. If you give the stale signal the higher weight, the herd comes back (2:1 sends
# all 12 to a).
#
# The split is this clean only because the burst matches the free slots. The lesson is the
# mechanism. This is why the EPP's recommended load scorers read in-flight state that the router
# keeps itself (`inflight-load-producer`). It is also why a router fleet with several replicas
# depends on scraped signals and flow control. The reason is that each of those router replicas
# cannot see the in-flight requests of the others.

# %% [markdown]
# ## Exercise 2.4 — an affinity threshold that fits the workload
#
# `sticky-until-saturated` keeps only the replicas whose prefix match ratio is ≥ `affinityThreshold`
# (default 0.80). Take an agent session at turn $t$. The ratio on its *own* replica is approximately
# "blocks of turn ${t-1}$'s prompt" / "blocks of turn $t$'s prompt". The reason is that every turn
# adds a reply and a tool result. Calculate those ratios with the router's own hash function. Then
# select the threshold.
#
# 1. `turn_ratios(prompts)`: the input is the request bodies of turns $0 \ldots T-1$ of one session.
#    For each turn $t \ge 1$, return the ratio
#    `leading equal block hashes (turn t-1 vs turn t) / blocks of turn t`.
#    Use `estimate_tokens` and `block_hashes(tokens, 64, model)`.
# 2. `choose_threshold(ratios)`: return the largest multiple of 0.05 that is ≤ the smallest ratio.

# %% exercise
from igwlab.router import block_hashes, estimate_tokens

def turn_ratios(prompts) -> list:
    ### BEGIN SOLUTION
    out = []
    for prev, cur in zip(prompts, prompts[1:]):
        hp = block_hashes(estimate_tokens(prev), 64, prev["model"])
        hc = block_hashes(estimate_tokens(cur), 64, cur["model"])
        n = 0
        while n < min(len(hp), len(hc)) and hp[n] == hc[n]:
            n += 1
        out.append(n / len(hc))
    return out
    ### END SOLUTION

def choose_threshold(ratios) -> float:
    ### BEGIN SOLUTION
    import math
    return math.floor(min(ratios) / 0.05 + 1e-9) * 0.05
    ### END SOLUTION

# %% check
from igwlab.bench import session_messages
uniform = agentic_sessions(n_sessions=18, turns=5, n_agents=4, system_words=1200, tool_words=400, seed=0)
s0 = uniform[0]
replies = ["word " * 24] * 4                       # replies are 24 tokens; their exact words do not matter here
prompts = [{"model": "lab/llm", "tools": s0.agent.tools, "messages": session_messages(s0, replies[:t])} for t in range(5)]
ratios = turn_ratios(prompts)
assert len(ratios) == 4 and all(0.5 < x < 1.0 for x in ratios) and ratios == sorted(ratios)
thr = choose_threshold(ratios)
assert thr <= min(ratios) < thr + 0.05 and abs(thr / 0.05 - round(thr / 0.05)) < 1e-9
below = [t + 1 for t, x in enumerate(ratios) if x < 0.80]
print("ratios by turn:", [round(x, 3) for x in ratios], "-> threshold", round(thr, 2))
print(f"✅ with the default 0.80, turn(s) {below} fall below the threshold and are never sticky "
      "(a turn passes 0.80 only if its new tokens are at most 20% of its prompt)")

# %%
from igwlab.router import RouterSettings

tuned = copy.deepcopy(load_config("sticky-until-saturated").raw)          # the preset as a dict
tuned["plugins"][2]["parameters"]["affinityThreshold"] = round(thr, 2)
res, pinned = {}, {}
for label, cfg in (("sticky 0.80", "sticky-until-saturated"), (f"sticky {thr:.2f}", tuned)):
    with LocalStack(3, cfg, settings=RouterSettings(decisions_kept=1000)) as s:
        res[label] = s.bench(uniform, label=label)
        decisions = s.call(lambda: list(s.router.decisions))
    pinned[label] = sum(len(d.stages[0][1]) < len(d.candidates) for d in decisions)
print(compare(res.values()))
print("decisions the affinity filter narrowed to sticky replicas:", pinned, f"(of {len(decisions)})")

# %% [markdown]
# With the lower threshold, the filter pins many more turns. The first resumed turn of every session
# now counts as sticky. The hit rate changes less than that count suggests, and on a given run the
# two hit rates can even be equal. When the filter keeps every replica, `token-load-scorer` charges
# each replica only the *uncached* tokens of this request. Thus it usually selects the replica that
# holds the history anyway. The threshold is the explicit rule, and the prefix-aware load score is
# the safety net behind it.

# %% [markdown]
# ## Exercise 2.5 — saturation and shedding
#
# With flow control off (the llm-d default), admission is simple. When the pool is saturated, the
# router rejects with HTTP 429 each request whose `InferenceObjective` has **priority < 0**. The
# router routes all other requests. That is the only admission path that the lab router has.
#
# With `featureGates: [flowControl]`, the real EPP instead holds requests in router-side queues by
# priority band (with fairness and TTLs, [PRIMER §3](../../PRIMER.md)). The lab router accepts the
# gate with a warning and continues to shed requests. The saturation comes from the
# `utilization-detector`:
#
# $$
# \text{saturation} = \text{mean over endpoints of }
#   \max\left(\frac{\text{waiting}}{5}, \frac{\mathrm{kv\_usage}}{0.8}\right),
# $$
#
# Here, an endpoint with stale metrics or no metrics counts as 1.0, and an empty pool is 1.0.
#
# Write `pool_saturation(endpoints)` for a list of `(waiting, kv_usage, fresh)` tuples. Also write
# `shed(objectives, saturation)`. It returns the sorted names of the objectives that the router rejects.

# %% exercise
def pool_saturation(endpoints, tq=5, tkv=0.8) -> float:
    ### BEGIN SOLUTION
    if not endpoints:
        return 1.0
    vals = [max(w / tq, kv / tkv) if fresh else 1.0 for w, kv, fresh in endpoints]
    return sum(vals) / len(vals)
    ### END SOLUTION

def shed(objectives: dict, saturation: float) -> list:
    ### BEGIN SOLUTION
    return sorted(n for n, p in objectives.items() if p < 0 and saturation >= 1.0)
    ### END SOLUTION

# %% check
import time as _t
from igwlab.router import Endpoint, EndpointMetrics, Router
eps = [(2, 0.40, True), (10, 0.30, True), (0, 0.0, False)]
router = Router("default-weighted", [Endpoint(f"e{i}", "http://unused",
                                              metrics=EndpointMetrics(waiting=w, kv_usage=kv, update_time=_t.monotonic() if f else 0.0))
                                     for i, (w, kv, f) in enumerate(eps)])
assert abs(pool_saturation(eps) - router.pool_saturation()) < 1e-9          # (0.5 + 2.0 + 1.0) / 3
assert abs(pool_saturation(eps) - 3.5 / 3) < 1e-9 and pool_saturation([]) == 1.0
assert shed({"premium": 100, "standard": 0, "batch": -10, "scavenger": -1}, 1.17) == ["batch", "scavenger"]
assert shed({"batch": -10}, 0.99) == []
print("✅ saturation", round(pool_saturation(eps), 3), "-> sheds", shed({"premium": 100, "batch": -10}, pool_saturation(eps)))

# %% [markdown]
# ## In a design review
#
# **Two-minute walkthrough.** "Affinity and load are one trade-off, and the weights give it a price.
# Take the chart default: prefix 3 against queue 2 plus KV 2. A fully cached replica wins against a
# cold idle one (total 4) until its load scores are more than 3 points worse. It can have the deepest
# queue in the pool (queue score 0) and still win while its KV score stays above 0.5. That is, it
# wins until its KV usage is ~50 points above the KV usage of the idle replica.
#
# "Thus load alone rarely pushes a hot prefix off its replica. The hot prefix spreads because other
# replicas served it too (early ties, partial matches), and then several replicas are sticky. When
# that is not sufficient, we use the TTFT-gated affinity filter of the optimized baseline. Pure
# affinity overloads one replica. Pure load prefills every history again.
#
# "We never route on scraped metrics alone. They are up to one scrape interval stale, so a burst goes
# as a herd to the replica that looked the most idle. Thus we combine them with the router's own
# in-flight counts.
#
# "The thresholds come from the workload. The first resumed turn of our agents shares only ~73% of
# its blocks with the previous turn. Thus we set the affinity threshold to 0.7, not 0.8. The
# prefix-aware token-load scorer covers the turns that the filter lets through.
#
# "Under saturation, the router drops only sheddable (negative-priority) objectives, with a 429."
#
# **Drill questions**
#
# 1. *After you turned on prefix routing, the traffic of a new tenant made p99 TTFT worse. What is the
#    first suspect?* A hot prefix: one system prompt dominates, and all of its traffic is sticky to
#    one replica. Examine the request counts and the queue depth per replica. Increase the load
#    weights, use the TTFT gate of the affinity filter, or add replicas.
# 2. *Why can a KV-usage difference of 0.03 send 100% of a burst to one replica?* `max-score-picker`
#    takes the maximum. Any strict difference wins every decision until the next scrape changes it.
# 3. *What does priority -10 mean for an InferenceObjective?* It is sheddable. Under saturation (with
#    flow control off), the router rejects a request of this priority with 429. It does this before
#    it schedules the request. The router always routes a request with priority ≥ 0. With flow
#    control on, the EPP puts requests in queues by priority band instead. The lab router does not
#    have that path.
