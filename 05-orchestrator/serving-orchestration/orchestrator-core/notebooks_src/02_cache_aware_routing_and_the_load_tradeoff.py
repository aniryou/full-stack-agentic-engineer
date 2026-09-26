# %% [markdown]
# # 02 · Cache-aware routing and the load trade-off
#
# **Tier:** T0 — CPU only, about 15 seconds, no network. Every number below is **simulated** by `fleetsim`.
#
# ## The one-minute version
# A replica's prefix cache turns a 6,000-token prefill into a few hundred tokens — but only on the replica that holds
# the prefix. So routing has two goals that pull against each other:
#
# * **Locality** (prefix hash, session affinity): send a request where its prefix is cached. Hit rate and TTFT
#   improve — until a *hot* prefix (one popular agent, one big tenant) overloads the replica that owns it.
# * **Load** (round-robin, power-of-two): spread the work. No hot spots, but every replica re-prefills everything,
#   and each replica's small cache has to hold everybody's prefixes.
#
# Production routers are knobs between the two. **Consistent hashing with bounded loads** keeps affinity but caps
# any replica at (1 + ε) x the average load. The **llm-d EPP** filters, then adds weighted scores for prefix match,
# queue depth and KV use. **Sticky until saturated** keeps affinity until the estimated TTFT penalty is too high.
# You will find the knob's optimum in numbers. Primer: §2 (and §6 for why long pauses defeat any router).

# %%
from fleetsim import (L4_8B, ConsistentHashBoundedLoad, Fleet, HashChain, PowerOfTwo, PrefixHash, RoundRobin,
                      agentic, epp, expand, sticky_until_saturated, table)
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
# `h_i = H(h_{i-1}, tokens of block i)`. So block *i* matches only if the *entire* prefix up to it matches — the same
# paragraph at a different position is a different block, and a partial last block is never shared.
#
# ## Exercise 2.1 — chain hashes
# Write `block_hashes(tokens, block=16)` returning one hash per **full** block, chaining each block's hash to its
# parent's: start from `parent = 0`, and for each block compute `parent = hash((parent, tuple(block_tokens)))`.
# (Python's `hash` of a tuple of ints is deterministic, unlike its `str` hash.)

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
# The llm-d Router's `prefix-cache-scorer` scores an endpoint by **matched prefix blocks / total prompt blocks**; its
# `queue-scorer` min-max normalises the scraped waiting-queue length: **(maxQ - q) / (maxQ - minQ)**, and gives every
# endpoint a neutral **1.0** if all queues are equal. Write both.

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
# Four L4 replicas, agent sessions from six agents (Zipf-popular: the top agent sends about 40 % of sessions). SLO
# for the attainment column: TTFT <= 1 s and TPOT <= 150 ms.

# %%
routers = {
    "round-robin": lambda: RoundRobin(),
    "power-of-two": lambda: PowerOfTwo(seed=1),
    "prefix-hash (system prompt)": lambda: PrefixHash(),
    "prefix-hash (session)": lambda: PrefixHash(by="session"),
    "bounded-load hash eps=0.25": lambda: ConsistentHashBoundedLoad(eps=0.25),
    "EPP 3:2:2 (approx index)": lambda: epp((3, 2, 2)),
    "EPP 3:2:2 (precise index)": lambda: epp((3, 2, 2), index="precise"),
    "sticky until saturated": lambda: sticky_until_saturated(p.compute_tok_s, max_ttft_penalty_s=2.0),
}
cols = ["hit_rate", "ttft_p50", "ttft_p95", "imbalance", "preemptions", "slo_attainment"]
rows = []
for name, make in routers.items():
    s = Fleet(p, 4, make()).run(agents()).summary(ttft_slo=1.0, tpot_slo=0.15)
    rows.append({"router": name, **{c: s[c] for c in cols}})
print(table(rows, ["router"] + cols, title="simulated: agent sessions on 4x L4"))

# %% [markdown]
# Read it left to right. Load-only routers (round-robin, power-of-two) balance perfectly and hit only on the system
# prompts every replica ends up caching. Hashing the **system prompt** is locality with no brakes: the most popular
# agent's replica melts (imbalance well above 2, TTFT in tens of seconds). Hashing the **session** works here because
# sessions are many and small — but it cannot react when one replica gets unlucky. Bounded loads fix the melt at some
# cost in hits. The EPP gets the history hits of affinity *and* reacts to queues.
#
# ## Worked example — the knob: how much should cache locality weigh?
# Keep queue and KV-utilisation weights at 2 and sweep the prefix-cache weight.

# %%
sweep = []
for w in (0, 1, 3, 10, 30):
    s = Fleet(p, 4, epp((w, 2, 2))).run(agents()).summary(ttft_slo=1.0, tpot_slo=0.15)
    sweep.append({"prefix weight": w, **{c: s[c] for c in cols}})
print(table(sweep, ["prefix weight"] + cols, title="simulated: EPP weight sweep (queue 2, kv 2)"))

# %% [markdown]
# Hit rate saturates early, but imbalance keeps climbing with the weight: past the optimum, the router keeps sending
# sessions to the replica that has their history even when its queue is long. Too little weight re-prefills
# everything; too much rebuilds the prefix-hash hot spot. The optimum depends on the workload — which is why
# llm-d moved to *filters* that are sticky only until a load gate trips (the last row of the first table).
#
# ## Exercise 2.3 — bounded loads
# Write `bounded_pick(order, loads, eps)`: `order` is the list of replicas in clockwise ring order starting at the
# request's hash; `loads` maps replica -> requests in flight. Capacity is `ceil((1 + eps) * (total + 1) / n)` (the
# `+ 1` counts the request being placed); return the first replica in `order` whose load is **below** capacity.

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
# ## Exercise 2.4 — tune the EPP for a target
# Choose `weights = (prefix, queue, kv)` so that this workload reaches **hit rate >= 0.85**, **p95 TTFT <= 0.6 s**
# and **imbalance <= 1.3** — and be ready to say why your choice sits where it does on the curve above.

# %% exercise
weights = None
### BEGIN SOLUTION
weights = (3, 2, 2)     # enough weight to keep sessions home, enough load weight to leave a busy replica
### END SOLUTION

# %% check
s = Fleet(p, 4, epp(weights)).run(agents()).summary(ttft_slo=1.0, tpot_slo=0.15)
assert s["hit_rate"] >= 0.85 and s["ttft_p95"] <= 0.6 and s["imbalance"] <= 1.3, s
print(f"✅ {weights}: hit {s['hit_rate']:.2f}, p95 TTFT {s['ttft_p95']:.2f} s, imbalance {s['imbalance']:.2f}")

# %% [markdown]
# ## Exercise 2.5 — a hotter prefix
# Make one agent dominant (Zipf 2.0: the top agent now sends about two thirds of the sessions). **Predict** which of
# these routers suffers the largest p95 TTFT, then let the check run them:
# `"round-robin"`, `"prefix-hash (system prompt)"`, `"prefix-hash (session)"`, `"EPP 3:2:2 (approx index)"`.

# %% exercise
worst = None
### BEGIN SOLUTION
worst = "prefix-hash (system prompt)"    # every session of the hot agent lands on one replica
### END SOLUTION

# %% check
hot = {}
for name in ("round-robin", "prefix-hash (system prompt)", "prefix-hash (session)", "EPP 3:2:2 (approx index)"):
    hot[name] = Fleet(p, 4, routers[name]()).run(agents(zipf=2.0)).summary(ttft_slo=1.0)["ttft_p95"]
print(table([{"router": k, "ttft_p95": v} for k, v in hot.items()], title="simulated: Zipf 2.0"))
assert worst == max(hot, key=hot.get), (worst, hot)
print("✅ a hot prefix melts pure prefix hashing; the EPP spreads it and keeps the session hits")

# %% [markdown]
# ## Exercise 2.6 — how stale is an approximate index?
# The EPP above never *sees* a replica's cache: its approximate index records the blocks of every request it sends
# and forgets them LRU-first. Write `phantom_share(believed, actual)`: the fraction of blocks the router believes are
# cached (`believed`, a set) that the replica no longer holds (`actual`, a set). Then measure it for an index sized
# 15x too large for an L4's KV pool — a plausible misconfiguration when defaults are sized for bigger GPUs.

# %% exercise
def phantom_share(believed, actual):
    ### BEGIN SOLUTION
    return len(believed - actual) / len(believed) if believed else 0.0
    ### END SOLUTION

# %% check
from fleetsim import ApproxPrefixIndex, KVCacheUtilizationScorer, PrefixCacheScorer, QueueScorer, WeightedScorer

assert phantom_share({1, 2, 3, 4}, {1, 2}) == 0.5 and phantom_share(set(), {1}) == 0.0
for label, size in (("sized to the pool", p.kv_blocks), ("15x too large", 15 * p.kv_blocks)):
    ix = ApproxPrefixIndex(size)
    fleet = Fleet(p, 4, WeightedScorer([(PrefixCacheScorer(ix), 3), (QueueScorer(), 2), (KVCacheUtilizationScorer(), 2)]))
    s = fleet.run(agents()).summary(ttft_slo=1.0)
    ph = sum(phantom_share(set(ix.lru[r.rid]), set(r.pool.cached)) for r in fleet.all) / len(fleet.all)
    print(f"index {label:17s}: {ph:.0%} phantom blocks, hit rate {s['hit_rate']:.2f}, p95 TTFT {s['ttft_p95']:.2f} s")
print("✅ phantom_share works")

# %% [markdown]
# A mostly-phantom index barely hurts *this* workload: agent turns come back within seconds, so the entries the router
# actually queries are fresh; the phantom ones belong to sessions that ended. It hurts when reuse is distant — long
# tool pauses, many tenants, tiny caches — and that is also when *no* index can help, because the KV is gone
# (notebook 05's topic). Precise, event-fed indexes (llm-d's precise-prefix-cache routing) earn their keep by also
# tracking offloaded tiers and by sharing one view across router replicas.
#
# ## In a design review
# **Two-minute version.** "Each replica is a cache, so routing decides the hit rate. Pure affinity maximises hits but
# a popular prefix melts its replica; pure load balancing re-prefills everything. I would run the llm-d EPP pattern:
# filter for adapter and prefix affinity, score queue depth and KV use, pick the max — and tune the prefix weight on
# a replay of our traffic, watching hit rate *and* per-replica load together, because the failure is a hot spot, not
# a low hit rate. For hard guarantees, bounded-load consistent hashing caps any replica at (1 + ε) x average."
#
# **Drills**
# 1. *Why not hash on the session id and be done?* It ignores load; a few long sessions on one replica queue behind
#    each other, and nothing moves them. It is a good *score*, not a policy.
# 2. *Hit rate fell from 0.85 to 0.55 after a deploy. Where do you look?* The prompt layout (a timestamp or user id
#    ahead of the system prompt breaks every block after it), block size, then router weights and per-replica load.
# 3. *What does ε = 0.25 buy you?* No replica ever holds more than ceil(1.25 x average + 1) in-flight requests,
#    whatever the key skew; the price is that overflow keys move to the next replica on the ring and miss there.
