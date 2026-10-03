# %% [markdown]
# # 02 · Cache-aware routing and the load trade-off
#
# **Tier:** T0. CPU only, about 15 seconds, no network. `fleetsim` **simulates** every number in this notebook.
#
# ## The one-minute version
# The prefix cache of a replica changes a 6,000-token prefill into a few hundred tokens. But this occurs only on the
# replica that holds the prefix. Thus routing has two goals that pull against each other:
#
# * **Locality** (prefix hash, session affinity): send a request to the replica that has its prefix in the cache.
#   The hit rate and TTFT improve, until a *hot* prefix (one popular agent, one large tenant) overloads the replica
#   that owns it.
# * **Load** (round-robin, power-of-two): spread the work. There are no hot spots. But each replica does the prefill
#   of everything again, and the small cache of each replica must hold the prefixes of everybody.
#
# Routers in production are knobs between the two goals. **Consistent hashing with bounded loads** keeps affinity,
# but it limits each replica to $(1 + \varepsilon) \times \text{the average load}$. The **llm-d EPP** applies
# filters first. Then it adds weighted scores for prefix match, queue depth and KV use. **Sticky until saturated**
# keeps affinity until the estimated TTFT penalty is too high.
#
# In this notebook, you find the optimum of the knob in numbers. You see what each load gate can see and what it
# cannot see. You also route LoRA adapters in the same way. Primer: §2 and §7. §6 tells why long pauses defeat any
# router.

# %%
from fleetsim import (L4_8B, ConsistentHashBoundedLoad, Fleet, HashChain, PowerOfTwo, PrefixAffinityFilter,
                      PrefixHash, RoundRobin, agentic, epp, expand, sticky_until_saturated, table)
import math

p = L4_8B


def agents(zipf=1.0, rate=0.6, seed=3):
    """Agent sessions: 6 agents with 2,000-token system prompts (Zipf-popular); each turn re-sends the history."""
    return agentic(rate, 240, seed=seed, groups=6, zipf=zipf, system=2000, tool=500, output=100, turns=(4, 10),
                   think=(1, 4))


reqs = expand(agents())
print(f"{len(reqs)} requests; mean prompt {sum(r.prompt for r in reqs) / len(reqs):,.0f} tokens; "
      f"KV pool per replica {p.kv_blocks * p.block:,} tokens")

# %% [markdown]
# ## The identity a prefix cache matches on
# vLLM hashes each **full** block of 16 tokens together with the hash of the block before it:
# $h_i = H(h_{i-1}, \text{tokens of block } i)$. Thus block $i$ matches only if the *entire* prefix up to that block
# matches. The same paragraph at a different position is a different block. The engine never shares a partial last
# block.
#
# ## Exercise 2.1 — chain hashes
# Write `block_hashes(tokens, block=16)`. It returns one hash for each **full** block, and the hash of each block
# includes the hash of its parent. Start from `parent = 0`. For each block, calculate
# `parent = hash((parent, tuple(block_tokens)))`. (The Python `hash` of a tuple of ints is deterministic. The Python
# `hash` of a `str` is not.)

# %% exercise
def block_hashes(tokens, block=16):
    ### BEGIN SOLUTION
    out, parent = [], 0
    for i in range(0, len(tokens) - block + 1, block):
        parent = hash((parent, tuple(tokens[i:i + block])))
        out.append(parent)
    return out
    ### END SOLUTION

# %% check
system, a, b = list(range(1000, 1032)), list(range(50)), list(range(100, 150))
ha, hb = block_hashes(system + a), block_hashes(system + b)
assert len(ha) == (32 + 50) // 16 == 5 and ha[:2] == hb[:2] and ha[2] != hb[2]      # shared system prompt only
assert not set(block_hashes([7] * 16 + system)) & set(ha[:2])                        # same text, other offset
assert block_hashes(list(range(15))) == []                                           # partial block: no hash
print("✅ block_hashes works — a hit means the whole prefix matched")

# %% [markdown]
# ## Exercise 2.2 — two EPP scorers
# The `prefix-cache-scorer` of the llm-d Router gives an endpoint the score **matched prefix blocks / total prompt
# blocks**. Its `queue-scorer` applies a min-max normalization to the scraped waiting-queue length:
# **$(\mathrm{maxQ} - q) / (\mathrm{maxQ} - \mathrm{minQ})$**. If all queues are equal, it gives each endpoint a
# neutral **1.0**. Write both scorers.

# %% exercise
def prefix_score(request_hashes, cached):
    """Fraction of the request's blocks covered by the longest prefix found in `cached` (a set of hashes)."""
    ### BEGIN SOLUTION
    n = 0
    for h in request_hashes:
        if h not in cached:
            break
        n += 1
    return n / len(request_hashes) if request_hashes else 0.0
    ### END SOLUTION


def queue_scores(waiting):
    """waiting: {replica: queue length} -> {replica: score in [0, 1]}."""
    ### BEGIN SOLUTION
    lo, hi = min(waiting.values()), max(waiting.values())
    return {k: 1.0 if hi == lo else (hi - v) / (hi - lo) for k, v in waiting.items()}
    ### END SOLUTION

# %% check
assert prefix_score(ha, set(ha[:2])) == 2 / 5
assert prefix_score(ha, set(ha[1:])) == 0.0                     # block 0 missing: nothing after it can be used
assert queue_scores({"a": 0, "b": 2, "c": 4}) == {"a": 1.0, "b": 0.5, "c": 0.0}
assert queue_scores({"a": 3, "b": 3}) == {"a": 1.0, "b": 1.0}
print("✅ prefix_score and queue_scores match the upstream formulas")

# %% [markdown]
# ## Worked example — the whole spectrum on one workload
# Four L4 replicas serve agent sessions from six agents. The agents are Zipf-popular: the top agent sends
# approximately 40 % of sessions. The SLO for the attainment column is TTFT $\le$ 1 s and TPOT $\le$ 150 ms.

# %%
routers = {
    "round-robin": lambda: RoundRobin(),
    "power-of-two": lambda: PowerOfTwo(seed=1),
    "prefix-hash (system prompt)": lambda: PrefixHash(),
    "prefix-hash (session)": lambda: PrefixHash(by="session"),
    "bounded-load hash eps=0.25": lambda: ConsistentHashBoundedLoad(eps=0.25),
    "EPP 3:2:2 (approx index)": lambda: epp((3, 2, 2)),
    "EPP 3:2:2 (precise index)": lambda: epp((3, 2, 2), index="precise"),
    # llm-d's optimized baseline: prefix-cache-affinity-filter + token-load-scorer, its TTFT estimate calibrated to
    # this engine (peak prefill = the L4 profile's 3,781 tok/s); the upstream default gate, then a much tighter one
    "affinity + token load, 18 s gate": lambda: sticky_until_saturated(p.compute_tok_s),
    "affinity + token load, 0.5 s gate": lambda: sticky_until_saturated(p.compute_tok_s, max_ttft_penalty_s=0.5),
}
cols = ["hit_rate", "ttft_p50", "ttft_p95", "imbalance", "preemptions", "slo_attainment"]
rows, gates = [], {}
for name, make in routers.items():
    router = make()
    s = Fleet(p, 4, router).run(agents()).summary(ttft_slo=1.0, tpot_slo=0.15)
    rows.append({"router": name, **{c: s[c] for c in cols}})
    gates.update({name: f.decisions for f in getattr(router, "filters", []) if isinstance(f, PrefixAffinityFilter)})
print(table(rows, ["router"] + cols, title="simulated: agent sessions on 4x L4"))
for name, d in gates.items():
    print(f"{name}: the TTFT gate broke stickiness {d['load_override']} times in "
          f"{d['sticky'] + d['load_override']} decisions that had a sticky endpoint")

# %% [markdown]
# Read the table from left to right. Load-only routers (round-robin, power-of-two) balance perfectly. They get hits
# only on the system prompts, which all replicas cache after some time. A hash of the **system prompt** is locality
# with no brakes. The replica of the most popular agent melts (imbalance well above 2, TTFT in tens of seconds).
#
# A hash of the **session** works here, because the sessions are many and small. But it cannot react when one
# replica is unlucky. Bounded loads repair the melt, at some cost in hits. The EPP gets the history hits of affinity
# *and* reacts to queues.
#
# The last two rows are the current default composition of llm-d. On *this* workload, it is behind the adjusted
# 3:2:2. The reason is what its gate measures: estimated TTFT = the endpoint's uncached prompt tokens in flight /
# peak prefill rate. This is the **prefill backlog**. Here, a hot replica becomes slow because of KV pressure and
# decode residency (see the preemptions). That estimate does not see these causes.
#
# Thus the upstream 18 s gate never acts, and the router is only sticky, with a token-load tie-break. A tighter gate
# of 0.5 s lets it break stickiness 14 times (of ~725 decisions), and it recovers much of the gap. The filter gives
# you one knob in TTFT seconds instead of weights to adjust. Its design is for prefill-bound traffic. The upstream
# project pairs it with `active-request-scorer` for decode-bound traffic.
#
# ## Worked example — the knob: how much should cache locality weigh?
# Keep the queue weight and the KV-utilisation weight at 2. Then sweep the prefix-cache weight.

# %%
sweep = []
for w in (0, 1, 3, 10, 30):
    s = Fleet(p, 4, epp((w, 2, 2))).run(agents()).summary(ttft_slo=1.0, tpot_slo=0.15)
    sweep.append({"prefix weight": w, **{c: s[c] for c in cols}})
print(table(sweep, ["prefix weight"] + cols, title="simulated: EPP weight sweep (queue 2, kv 2)"))

# %% [markdown]
# The hit rate saturates early, but the imbalance continues to increase with the weight. Past the optimum, the
# router continues to send sessions to the replica that has their history, even when its queue is long. Too small a
# weight prefills everything again, and too large a weight builds the prefix-hash hot spot again. The optimum
# depends on the workload, and it moves when the workload moves. This is the reason that the default of llm-d
# replaced adjusted weights with a filter that states its load gate in TTFT seconds. The first table showed the
# price of that robustness: a gate acts only on the load that it can see.
#
# ## Exercise 2.3 — bounded loads
# Write `bounded_pick(order, loads, eps)`. `order` is the list of replicas in clockwise ring order, from the hash of
# the request. `loads` maps each replica to its requests in flight. The capacity is
# $\lceil (1 + \mathrm{eps}) \times (\text{total} + 1) / n \rceil$ (the $+1$ counts the request that the router
# places now). Return the first replica in `order` whose load is **below** capacity.

# %% exercise
def bounded_pick(order, loads, eps):
    ### BEGIN SOLUTION
    cap = math.ceil((1 + eps) * (sum(loads.values()) + 1) / len(loads))
    return next(r for r in order if loads[r] < cap)
    ### END SOLUTION

# %% check
loads = {"A": 3, "B": 1, "C": 0, "D": 0}                       # total 4 -> cap = ceil(1.25 x 5 / 4) = 2
assert bounded_pick(["A", "B", "C", "D"], loads, 0.25) == "B"   # A is full, B is next clockwise
assert bounded_pick(["A", "B", "C", "D"], loads, 1.0) == "B"    # cap = ceil(2 x 5 / 4) = 3: A (3) still full
assert bounded_pick(["A", "B", "C", "D"], loads, 3.0) == "A"    # cap = 5: affinity wins
print("✅ bounded_pick works — eps is the locality/balance knob, with a hard bound on imbalance")

# %%
rows = []
for eps in (0.1, 0.25, 1.0, 4.0):
    s = Fleet(p, 4, ConsistentHashBoundedLoad(eps=eps)).run(agents()).summary(ttft_slo=1.0, tpot_slo=0.15)
    rows.append({"eps": eps, **{c: s[c] for c in cols}})
print(table(rows, ["eps"] + cols, title="simulated: bounded-load hashing on the system prompt"))

# %% [markdown]
# ## Exercise 2.4 — find the band of weights that meets a target
# The earlier sweep samples the prefix weight at only five points. Hold the queue and KV weights at 2. Find the
# **lowest** and the **highest** integer prefix weight from 1 to 12 at which this workload meets all three targets:
#
# - **hit rate $\ge$ 0.85**
# - **p95 TTFT $\le$ 0.6 s**
# - **imbalance $\le$ 1.3**
#
# One run takes less than a second. Set `band = (lowest, highest)`. Be ready to say which target stops the band at
# each end, and why.

# %% exercise
band = None
### BEGIN SOLUTION
def meets_targets(prefix_weight):
    s = Fleet(p, 4, epp((prefix_weight, 2, 2))).run(agents()).summary(ttft_slo=1.0, tpot_slo=0.15)
    return s["hit_rate"] >= 0.85 and s["ttft_p95"] <= 0.6 and s["imbalance"] <= 1.3


ok = [w for w in range(1, 13) if meets_targets(w)]
band = (min(ok), max(ok))   # below it re-prefill costs TTFT; above it the hot replica costs balance
### END SOLUTION

# %% check
fine = []
for w in range(1, 13):
    s = Fleet(p, 4, epp((w, 2, 2))).run(agents()).summary(ttft_slo=1.0, tpot_slo=0.15)
    fine.append({"prefix weight": w, **{c: s[c] for c in cols},
                 "meets": s["hit_rate"] >= 0.85 and s["ttft_p95"] <= 0.6 and s["imbalance"] <= 1.3})
inside = [r["prefix weight"] for r in fine if r["meets"]]
assert band is not None and tuple(band) == (min(inside), max(inside)), "not quite: sweep every integer weight from 1 to 12"
print(table(fine, ["prefix weight"] + cols + ["meets"], title="simulated: EPP weight sweep (queue 2, kv 2), every integer"))
print(f"✅ prefix weights {band[0]}-{band[1]} meet all three targets on this workload; the band moves when the workload does")

# %% [markdown]
# ## Exercise 2.5 — a hotter prefix
# Make one agent dominant (Zipf 2.0: the top agent now sends approximately two thirds of the sessions). **Predict**
# which of these routers gets the largest p95 TTFT. Then let the check run them: `"round-robin"`,
# `"prefix-hash (system prompt)"`, `"prefix-hash (session)"`, `"EPP 3:2:2 (approx index)"`,
# `"affinity + token load, 18 s gate"`.

# %% exercise
worst = None
### BEGIN SOLUTION
worst = "prefix-hash (system prompt)"    # every session of the hot agent lands on one replica
### END SOLUTION

# %% check
hot = {}
for name in ("round-robin", "prefix-hash (system prompt)", "prefix-hash (session)", "EPP 3:2:2 (approx index)",
             "affinity + token load, 18 s gate"):
    hot[name] = Fleet(p, 4, routers[name]()).run(agents(zipf=2.0)).summary(ttft_slo=1.0)["ttft_p95"]
print(table([{"router": k, "ttft_p95": v} for k, v in hot.items()], title="simulated: Zipf 2.0"))
assert worst == max(hot, key=hot.get), (worst, hot)
print("✅ a hot prefix melts pure prefix hashing; the EPP spreads it and keeps the session hits")

# %% [markdown]
# ## Exercise 2.6 — how stale is an approximate index?
# The EPP in the earlier examples never *sees* the cache of a replica. Its approximate index records the blocks of
# each request that it sends, and it forgets them LRU-first.
#
# Write `phantom_share(believed, actual)`. `believed` (a set) holds the blocks that the router believes are in the
# cache. `actual` (a set) holds the blocks that the replica holds. The function returns the fraction of `believed`
# that the replica no longer holds.
#
# The check measures this fraction for an index that is 15x too large for the KV pool of an L4. This is a plausible
# misconfiguration when the default sizes are for larger GPUs. **Predict** first: does the oversized index cost this
# workload more than 0.05 of hit rate (`True` or `False`)?

# %% exercise
def phantom_share(believed, actual):
    ### BEGIN SOLUTION
    return len(believed - actual) / len(believed) if believed else 0.0
    ### END SOLUTION


loses_hit_rate = None     # True or False
### BEGIN SOLUTION
loses_hit_rate = False    # agent turns come back within seconds: the entries the router queries are still fresh
### END SOLUTION

# %% check
from fleetsim import ApproxPrefixIndex, KVCacheUtilizationScorer, PrefixCacheScorer, QueueScorer, WeightedScorer

assert phantom_share({1, 2, 3, 4}, {1, 2}) == 0.5 and phantom_share(set(), {1}) == 0.0
hits = {}
for label, size in (("sized to the pool", p.kv_blocks), ("15x too large", 15 * p.kv_blocks)):
    ix = ApproxPrefixIndex(size)
    fleet = Fleet(p, 4, WeightedScorer([(PrefixCacheScorer(ix), 3), (QueueScorer(), 2), (KVCacheUtilizationScorer(), 2)]))
    s = fleet.run(agents()).summary(ttft_slo=1.0)
    hits[label] = s["hit_rate"]
    ph = sum(phantom_share(set(ix.lru[r.rid]), set(r.pool.cached)) for r in fleet.all) / len(fleet.all)
    print(f"index {label:17s}: {ph:.0%} phantom blocks, hit rate {s['hit_rate']:.2f}, p95 TTFT {s['ttft_p95']:.2f} s")
assert loses_hit_rate == (hits["sized to the pool"] - hits["15x too large"] > 0.05), hits
print("✅ phantom_share works, and your prediction held")

# %% [markdown]
# An index with mostly phantom blocks does almost no damage to *this* workload. Agent turns come back in seconds,
# thus the entries that the router really queries are fresh. The phantom entries belong to sessions that ended.
#
# The index does damage when the reuse is distant: long tool pauses, many tenants, small caches. In that case, also,
# *no* index can help, because the KV is gone (the topic of notebook 05). Precise, event-fed indexes (the
# precise-prefix-cache routing of llm-d) are worth their cost, because they also track offloaded tiers and share one
# view across router replicas.
#
# ## Worked example — adapters are a cache too (multi-LoRA)
# A replica can hold only `max_loras` adapters in a batch (vLLM `--max-loras`, 4 here). When the replica loads an
# adapter, the step stalls (`lora_load_s`, an illustrative 0.2 s). If the adapter of a request has no free slot, the
# request waits until the running requests of a slot drain.
#
# This example has twenty-four Zipf-popular adapters over four replicas. That is more adapters than the 16 slots of
# the fleet. The traffic is 2 req/s of chat, with and without the LoRA affinity filter of the EPP.

# %%
from fleetsim import (ApproxPrefixIndex, KVCacheUtilizationScorer, LoraAffinityFilter, PrefixCacheScorer,
                      QueueScorer, WeightedScorer, chat)

adapters = [f"adapter-{i}" for i in range(24)]


def lora_traffic():
    return chat(2.0, 240, seed=51, system=600, user=300, output=120, loras=adapters, lora_zipf=1.0)


def epp_without_lora_filter():
    return WeightedScorer([(PrefixCacheScorer(ApproxPrefixIndex()), 3), (QueueScorer(), 2),
                           (KVCacheUtilizationScorer(), 2)], name="EPP 3:2:2, no LoRA filter")


lora_rows = []
for name, make in (("power-of-two", lambda: PowerOfTwo(seed=1)), ("EPP 3:2:2, no LoRA filter", epp_without_lora_filter),
                   ("EPP 3:2:2 + LoRA affinity filter", lambda: epp((3, 2, 2)))):
    s = Fleet(p, 4, make()).run(lora_traffic()).summary(ttft_slo=1.0, tpot_slo=0.15)
    lora_rows.append({"router": name, "adapter loads": s["lora_loads"], "ttft_p50": s["ttft_p50"],
                      "ttft_p95": s["ttft_p95"], "imbalance": s["imbalance"], "slo_attainment": s["slo_attainment"]})
print(table(lora_rows, title="simulated: 24 adapters, 4x L4 with 4 adapter slots each, 2 req/s"))

# %% [markdown]
# Without affinity, each replica cycles through most adapters. There are hundreds of loads, and requests wait in a
# queue for a slot. With the filter, each adapter settles on one replica. The loads decrease to a few per adapter,
# and TTFT recovers. The price is imbalance: the replica that owns the hottest adapters does more of the work.
#
# ## Exercise 2.7 — the LoRA affinity filter
# Write `lora_filter(adapter, loaded, max_loras)`. `loaded` maps each replica to a tuple of resident adapters.
#
# Keep the replicas that already have `adapter`. If there are none, keep the replicas with a free slot. If there are
# none, keep all replicas (the engine will evict an idle adapter). If there is no adapter (`None`), keep all
# replicas.
#
# Then **predict**: with the filter, does imbalance go up or down compared with the EPP without it (`"up"` or
# `"down"`)?

# %% exercise
def lora_filter(adapter, loaded, max_loras):
    ### BEGIN SOLUTION
    if adapter is None:
        return list(loaded)
    hot = [r for r, ads in loaded.items() if adapter in ads]
    room = [r for r, ads in loaded.items() if len(ads) < max_loras]
    return hot or room or list(loaded)
    ### END SOLUTION


imbalance_with_filter = None     # "up" or "down"
### BEGIN SOLUTION
imbalance_with_filter = "up"     # a hot adapter pins its replica; the filter trades balance for fewer loads
### END SOLUTION

# %% check
from fleetsim.workload import Request

loaded = {0: ("a",), 1: ("b", "c", "d", "e"), 2: ()}
assert lora_filter("a", loaded, 4) == [0] and lora_filter("z", loaded, 4) == [0, 2]
assert lora_filter("z", {0: ("a", "b"), 1: ("c", "d")}, 2) == [0, 1] and lora_filter(None, loaded, 4) == [0, 1, 2]


class _Stub:
    def __init__(self, rid, ads):
        self.rid, self.p, self._ads = rid, p, ads

    def metrics(self, now):
        return {"loras": self._ads}


for ad in ("a", "b", "z", None):                       # agrees with the library's filter
    reps_ = [_Stub(r, ads) for r, ads in loaded.items()]
    got = [r.rid for r in LoraAffinityFilter().filter(Request(0, 0.0, 16, 1, [], 1, lora=ad), reps_, None, 0)]
    assert got == lora_filter(ad, loaded, p.max_loras), (ad, got)
by = {r["router"]: r for r in lora_rows}
with_f, without = by["EPP 3:2:2 + LoRA affinity filter"], by["EPP 3:2:2, no LoRA filter"]
assert with_f["adapter loads"] < without["adapter loads"] / 2 and with_f["ttft_p95"] < without["ttft_p95"]
assert imbalance_with_filter == ("up" if with_f["imbalance"] > without["imbalance"] else "down")
print(f"✅ loads {without['adapter loads']} -> {with_f['adapter loads']}, p95 TTFT {without['ttft_p95']:.2f} -> "
      f"{with_f['ttft_p95']:.2f} s, imbalance {without['imbalance']:.2f} -> {with_f['imbalance']:.2f}")

# %% [markdown]
# ## In a design review
# **Two-minute version.** "Each replica is a cache, thus routing decides the hit rate. Pure affinity gives the most
# hits, but a popular prefix melts its replica. Pure load balance prefills everything again. My choice is the llm-d
# EPP pattern. It applies filters for adapter and prefix affinity, scores queue depth and KV use, and selects the
# max.
#
# "Then I adjust the prefix weight on a replay of our traffic. I monitor the hit rate *and* the per-replica load
# together, because the failure is a hot spot, not a low hit rate. We can take the affinity filter of llm-d with a
# TTFT gate instead. In that case, I make sure that the estimate of the gate is what really slows our hot replicas.
# That estimate is the prefill backlog. I also count how often the gate breaks stickiness.
#
# "For hard guarantees, bounded-load consistent hashing limits each replica to
# $(1 + \varepsilon) \times \text{average}$."
#
# **Drills**
# 1. *Why not hash on the session id, and stop there?* It ignores load. A few long sessions on one replica wait
#    behind each other, and nothing moves them. The session id is a good *score*, not a policy.
# 2. *Hit rate fell from 0.85 to 0.55 after a deploy. Where do you look?* Look at the prompt layout. A timestamp or
#    a user id in front of the system prompt breaks every block after it. Then look at the block size. After that,
#    look at the router weights and the per-replica load.
# 3. *What does $\varepsilon = 0.25$ give you?* No replica ever holds more than
#    $\lceil (1 + \varepsilon) \times (\text{in-flight} + 1) / n \rceil$ requests. That is approximately 1.25 x the
#    average, for any key skew. The price is that overflow keys move to the next replica on the ring, and they miss
#    there.
# 4. *Sixteen LoRA adapters, four replicas with `--max-loras 4`: what does the router add?* Adapter affinity: keep
#    the replicas that have the adapter, or else the replicas with a free slot. Without it, each replica cycles
#    through adapters, and requests wait for a slot. The price is imbalance, because all the traffic of a hot
#    adapter goes to its replica.
