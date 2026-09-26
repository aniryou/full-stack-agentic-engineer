# %% [markdown]
# # 02 · Scorer weights, hot prefixes and the herd
#
# **Tier:** T0 — CPU only; fake backends and the router run in-process. Backend timing is emulated,
# so latencies are "measured on this machine, emulated backend".
#
# ## The one-minute version
#
# Cache affinity and load balance pull in opposite directions. Pure affinity sends every request
# for a popular prefix to the one replica that caches it — a **hot prefix** melts that replica while
# the others idle. Pure load balancing ignores the cache. A weighted router sits in between, and the
# weights decide exactly *how much* load a cache hit is worth: with prefix weight `wp` and load
# weights summing to `W`, a fully cached replica keeps winning until its load scores fall more than
# `wp` below a cold, idle replica's. The llm-d optimized baseline replaces the arithmetic with a
# rule — **sticky until saturated**: filter to replicas holding ≥ 80% of the prompt unless their
# estimated prefill backlog is too far behind the best other replica, then balance by tokens in
# flight. Two more traps: **scrape lag** (every request between two scrapes sees the same stale
# numbers and herds onto the replica that looked idlest) and **thresholds that ignore the workload**
# (an 80% affinity threshold never fires for a turn whose new tokens are more than 20% of its
# prompt — on this lab's agents, the first resumed turn of every session).
# Background: [PRIMER §2 Routing signals and algorithms and §3 Flow control and priorities](../../PRIMER.md).

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
# Same agentic sessions as notebook 01, except that 80% of them belong to one agent program
# (`hot_share=0.8`): one system prompt dominates the traffic. Four policies, each on a fresh stack:
#
# | preset | what it does |
# |---|---|
# | `prefix-only` | `prefix-cache-scorer` alone |
# | `load-only` | `queue-scorer` + `kv-cache-utilization-scorer` (cache-blind) |
# | `default-weighted` | prefix ×3, queue ×2, KV ×2 (the llm-d chart default) |
# | `sticky-until-saturated` | `prefix-cache-affinity-filter` (≥ 0.8, TTFT gate) → `token-load-scorer` |

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

# %% [markdown]
# `prefix-only` sends everything for the hot program to one replica: a top hit rate and by far the
# worst latency, because that replica's prefill queue grows while two replicas idle. `load-only` spreads
# evenly and pays for it in re-prefilled histories. The weighted and sticky policies both keep most
# of the hits while letting load push traffic off a busy replica — once a second replica has served
# the hot prefix, the index lists it too and both become "sticky".
#
# **This ranking belongs to this engine and this workload.** Here `sticky-until-saturated` beats the
# 3:2:2 chart default, while the primer's simulated fleet ([PRIMER §2.5](../../PRIMER.md)) ranks
# the 3:2:2 EPP ahead of it. One reason is the fake backend: it runs one prefill at a time per
# replica, so the tokens queued for prefill *are* the delay, and a filter gated on exactly that
# (in-flight uncached tokens ÷ prefill throughput) plus a scorer that balances tokens fits it
# well; the queue and KV scores of 3:2:2 see a request count and memory, not prefill work. On an
# engine with chunked prefill, other prompt lengths or a hotter prefix the order can flip — which
# is why the weights are tuned on a replay of your own traffic.
#
# ## Exercise 2.1 — when does affinity lose?
#
# A *sticky* replica S holds a fraction `r` of the prompt and has load scores `q_s` (queue) and
# `kv_s` (KV); a *cold* replica C has no prefix but is idle (queue and KV scores both 1.0). Write
# `sticky_wins(wp, wq, wkv, r, q_s, kv_s)` → `True` if S's weighted total is **strictly** higher than C's.

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
# Easy to mis-predict by eye — which is why the router records a per-scorer table for every decision.
#
# ## Exercise 2.2 — the weight that makes affinity unconditional
#
# From the weighted sum itself: each load score lies in [0, 1], so load scorers whose weights sum
# to `W` can move a total by at most `W`. A replica with prefix ratio `r` therefore beats *any*
# cold replica regardless of load only if `wp · r > W`. (The `prefix-cache-affinity-filter` README
# gives the consequence — a weighted prefix scorer with a max-score picker hot-spots popular
# prefixes — as the reason the filter exists; the inequality is ours.) Write
# `min_prefix_weight(W, r)`: the smallest weight for which a replica at ratio `r` cannot lose to a
# cold one (use `>=`; at exact equality the tie is broken by the picker).

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
# Metrics are scraped every 50 ms; between two scrapes every routing decision sees the same numbers.
# Below, each replica has **6 batch slots**, and replicas **b** and **c** are made busy by 3 requests
# each that do *not* go through this router (another router replica, a batch job — we send them
# directly), so their KV usage is a little higher at the last scrape. Then we freeze the router's view
# (stop the scraper after one refresh) and fire a burst of 12 requests at once. Free slots: a 6, b 3,
# c 3 — exactly 12. Compare a cache-blind policy on **scraped** signals (`load-only`) with one on the
# router's **own in-flight counts** (`active-requests`).

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
# Before looking at the numbers above too closely, reason it out and write the per-replica counts
# each policy produces for the 12-request burst (a dict `{"a": .., "b": .., "c": ..}`):
#
# * `load-only`: queue scores all tie (nobody is *waiting*); the KV score of **a** is higher by a
#   hair. Does the size of the difference matter to `max-score-picker`?
# * `active-requests`: the router counts only requests *it* has in flight, updated at dispatch time.

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
# Neither is right on its own. The scraped view herds all 12 onto **a**, which has 6 slots: half the
# burst waits for the first half to finish. The local view spreads 4/4/4 and sends 8 requests to b
# and c, which have 3 free slots each: one request on each waits behind the external jobs. Combine
# them: `running-requests-size-scorer` (scraped: knows about the foreign load on b and c, but frozen)
# plus `active-request-scorer` (local: instant, but partial). The *ratio* of their weights decides
# whether freshness or knowledge wins:

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
# With the local signal weighted higher, the truly idle replica gets the largest share *and* the
# busy ones take exactly what they have room for — here 6/3/3 fills the 6 + 3 + 3 free slots, so no
# request waits and the worst TTFT drops several-fold against either signal alone. Weight the stale
# signal higher and the herd is back (2:1 sends all 12 to a). The split is only this clean because
# the burst matches the free slots; the lesson is the mechanism. This is why the EPP's recommended
# load scorers read in-flight state the router maintains itself (`inflight-load-producer`), and why
# a router fleet with several replicas (each blind to the others' in-flight requests) leans on
# scraped signals and flow control.

# %% [markdown]
# ## Exercise 2.4 — an affinity threshold that fits the workload
#
# `sticky-until-saturated` keeps only replicas whose prefix match ratio is ≥ `affinityThreshold`
# (default 0.80). For an agent session, the ratio its *own* replica achieves at turn `t` is roughly
# "blocks of turn t-1's prompt" / "blocks of turn t's prompt" — every turn adds a reply and a tool
# result. Compute those ratios with the router's own hashing, then pick the threshold.
#
# 1. `turn_ratios(prompts)`: given the request bodies of turns 0..T-1 of one session, return for each
#    turn t ≥ 1 the ratio `leading equal block hashes (turn t-1 vs turn t) / blocks of turn t`
#    (use `estimate_tokens` and `block_hashes(tokens, 64, model)`).
# 2. `choose_threshold(ratios)`: the largest multiple of 0.05 that is ≤ the smallest ratio.

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
tuned = copy.deepcopy(load_config("sticky-until-saturated").raw)          # the preset as a dict
tuned["plugins"][2]["parameters"]["affinityThreshold"] = round(thr, 2)
res = {}
for label, cfg in (("sticky 0.80", "sticky-until-saturated"), (f"sticky {thr:.2f}", tuned)):
    with LocalStack(3, cfg) as s:
        res[label] = s.bench(uniform, label=label)
print(compare(res.values()))

# %% [markdown]
# ## Exercise 2.5 — saturation and shedding
#
# With flow control off (the llm-d default), admission is simple: a request whose
# `InferenceObjective` has **priority < 0** is rejected with HTTP 429 when the pool is saturated;
# everything else is routed. That is the only admission path the lab router implements: with
# `featureGates: [flowControl]` the real EPP instead holds requests in router-side queues by
# priority band (with fairness and TTLs, [PRIMER §3](../../PRIMER.md)); the lab router accepts the
# gate with a warning and keeps shedding. Saturation comes from the `utilization-detector`:
#
# `saturation = mean over endpoints of max(waiting / 5, kv_usage / 0.8)`, where an endpoint with
# stale or missing metrics counts as 1.0 and an empty pool is 1.0.
#
# Write `pool_saturation(endpoints)` for a list of `(waiting, kv_usage, fresh)` tuples, and
# `shed(objectives, saturation)` → sorted names of objectives that would be rejected.

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
# **Two-minute walkthrough.** "Affinity and load are one trade-off, and the weights price it. With
# the chart default — prefix 3 against queue 2 plus KV 2 — a fully cached replica beats a cold idle
# one (total 4) until its load scores are more than 3 points worse: even with the deepest queue in
# the pool (queue score 0) it keeps winning while its KV score stays above 0.5, i.e. until its KV
# usage is ~50 points above the idle replica's. So load alone rarely pushes a hot prefix off its
# replica; it spreads because other replicas served it too (early ties, partial matches), and then
# several are sticky — and when that is not enough we use the optimized baseline's TTFT-gated
# affinity filter. Pure affinity melts one replica; pure load re-prefills every history. We never
# route on scraped metrics alone: they are up to one scrape interval stale, so a burst herds onto
# whoever looked idlest; we combine them with the router's own in-flight counts. Thresholds come
# from the workload: our agents' first resumed turn shares only ~73% of its blocks with the turn
# before, so the affinity threshold is 0.7, not 0.8. Under saturation only sheddable
# (negative-priority) objectives are dropped, with a 429."
#
# **Drill questions**
#
# 1. *A new tenant's traffic made p99 TTFT worse after you enabled prefix routing. First suspect?*
#    A hot prefix: one system prompt dominating, all of it sticky to one replica. Check per-replica
#    request counts and queue depth; raise load weights, use the affinity filter's TTFT gate, or add
#    replicas.
# 2. *Why can a KV-usage difference of 0.03 send 100% of a burst to one replica?* `max-score-picker`
#    takes the maximum; any strict difference wins every decision until the next scrape changes it.
# 3. *What does priority -10 mean for an InferenceObjective?* Sheddable: under saturation (with flow
#    control off) the router rejects it with 429 before scheduling; ≥ 0 is always routed. With flow
#    control on, the EPP queues requests by priority band instead (the lab router does not implement
#    that path).
