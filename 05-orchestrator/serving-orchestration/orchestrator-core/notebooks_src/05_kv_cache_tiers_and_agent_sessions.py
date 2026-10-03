# %% [markdown]
# # 05 · KV-cache tiers and agent sessions
#
# **Tier:** T0 (CPU only, a few seconds, no network). Every number in this notebook is a **simulated** number from
# `fleetsim`. The tier bandwidths are assumptions that we state, not measurements.
#
# ## The one-minute version
# An agent session sends its full history again at every turn, and this history becomes longer at each turn. Between
# turns, the session pauses while tools run. Thus its KV cache is the most reusable state in the fleet. It also has
# the highest chance of eviction, because the **working set** ($\text{sessions} \times \text{context} \:\times$
# $\text{KV bytes per token}$) is many times the HBM that a replica can spare. Thus LRU evicts a session during its
# tool call, and the next turn must prefill everything again.
#
# Two changes repair this problem:
#
# * **Offload** evicted KV to host DRAM, local NVMe or a remote store (vLLM's OffloadingConnector, LMCache, Mooncake,
#   SGLang HiCache). A fetch is faster than a recompute when the bandwidth of the tier is more than the rate at which
#   prefill *produces* KV. This rate is $\text{KV bytes/token} \times \text{prefill tokens/s}$: about 0.5 GB/s for an
#   8B model on an L4, and 4 GB/s on an H100.
# * **Keep sessions near their KV** with sticky routing, or make the KV available from all replicas with a
#   **shared** tier. With a shared tier, the router is free to balance load again. Thus memory removes the tension
#   of notebook 02.
#
# Read Primer §6.

# %%
from fleetsim import (H100_8B, L4_8B, LLAMA_8B_KV, Fleet, Tier, TieredKV, agentic, breakeven_gb_s, epp, onload_s,
                      recompute_s, simulate_sessions, table, working_set_gb)

for prof in (L4_8B, H100_8B):
    pool_gb = prof.kv_blocks * prof.block * prof.kv_bytes_per_token / 1e9
    print(f"{prof.name}: KV pool {pool_gb:.1f} GB = {prof.kv_blocks * prof.block:,} tokens")
rows = []
for sessions, ctx in ((50, 8_000), (200, 30_000), (1_000, 30_000)):
    ws = working_set_gb(sessions, ctx, LLAMA_8B_KV)
    rows.append({"sessions": sessions, "context tokens": ctx, "working set GB": ws,
                 "H100 pools to hold it": ws / (H100_8B.kv_blocks * 16 * LLAMA_8B_KV / 1e9)})
print(table(rows, title="KV working set of concurrent agent sessions, 8B model (arithmetic)"))

# %% [markdown]
# Two hundred coding-agent sessions at 30k tokens of context need ~800 GB of KV. That is the KV pool of fifteen
# H100s, before any of the sessions has generated a token. At any instant, most of those sessions are idle, because
# they wait for a tool. Thus the question is not "can HBM hold it" (no). The question is "where do the idle ones
# stay, and what does it cost to bring one back?"
#
# ## Worked example — routing cannot fix a capacity problem
# This example uses the same four-L4 fleet and EPP router as notebook 02. The pauses between turns are short (1–4 s)
# or long (10–40 s).

# %%
rows = []
for think in ((1, 4), (10, 40)):
    reqs = agentic(0.6, 240, seed=3, groups=6, system=2000, tool=500, output=100, turns=(4, 10), think=think)
    s = Fleet(L4_8B, 4, epp()).run(reqs).summary(ttft_slo=1.0)
    rows.append({"tool pause s": f"{think[0]}-{think[1]}", "hit_rate": s["hit_rate"], "ttft_p50": s["ttft_p50"],
                 "ttft_p95": s["ttft_p95"]})
print(table(rows, title="simulated: EPP 3:2:2 on 4x L4, agent sessions"))

# %% [markdown]
# With longer pauses, more sessions are alive at the same time. Thus the replica evicts the KV of each session before
# its next turn. Then the hit rate goes back down to "system prompts only", whatever the router does.
#
# ## Exercise 5.1 — fetch or recompute?
# A recompute of $n$ tokens of KV takes $n / \mathrm{prefill\_tok\_s}$. A fetch of these tokens takes
# $n \times \mathrm{kv\_bytes\_per\_token} / \text{bandwidth}$. Write `fetch_beats_recompute(profile, tiers)`. Return
# the names of the tiers whose read bandwidth is more than the break-even
# $\mathrm{kv\_bytes\_per\_token} \times \mathrm{prefill\_tok\_s}$ (in GB/s).

# %% exercise
tiers = [Tier("host DRAM (PCIe)", 256, 50.0, 0.0005),        # assumed ~50 GB/s over PCIe Gen5 x16 (verify)
         Tier("local NVMe", 2000, 6.0, 0.001),               # assumed one Gen4 drive (verify)
         Tier("remote store, 200G RDMA", 10_000, 20.0, 0.002),
         Tier("remote store, 25 GbE", 10_000, 3.0, 0.002)]


def fetch_beats_recompute(profile, tiers):
    ### BEGIN SOLUTION
    need = profile.kv_bytes_per_token * profile.compute_tok_s / 1e9
    return [t.name for t in tiers if t.read_gb_s > need]
    ### END SOLUTION

# %% check
assert abs(breakeven_gb_s(LLAMA_8B_KV, L4_8B.compute_tok_s) - 0.4956) < 1e-4
assert fetch_beats_recompute(L4_8B, tiers) == [t.name for t in tiers]                   # every tier wins on an L4
assert fetch_beats_recompute(H100_8B, tiers) == [t.name for t in tiers[:3]]             # 25 GbE loses on an H100
n = 10_000
print(f"10k tokens on an H100: recompute {recompute_s(n, H100_8B.compute_tok_s) * 1e3:.0f} ms, "
      f"DRAM {onload_s(n, LLAMA_8B_KV, tiers[0]) * 1e3:.0f} ms, 25 GbE {onload_s(n, LLAMA_8B_KV, tiers[3]) * 1e3:.0f} ms")
print("✅ the faster the GPU, the faster the tier must be to be worth it")

# %% [markdown]
# One caveat about the break-even: `recompute_s` is linear in tokens, as is the prefill of the simulator. Attention
# adds FLOPs that increase with the context. For an 8B model, attention adds about +16 % at 10k tokens and +50 % at
# 30k. Thus, in practice, a long-context recompute is slower than this, and a fetch wins by more. The tier bandwidths
# are assumptions. Measure them.

# %% [markdown]
# ## Exercise 5.2 — two tiers, demote on evict
# Write the core of a cache that offloads:
#
# * `put(key, gb)` puts the entry into the fast tier as the most recently used entry.
# * Then, while the fast tier is over capacity, `put` **demotes** its least recently used entry to the slow tier.
# * While the slow tier is over capacity, `put` **drops** its LRU entry.
# * `where(key)` returns `"fast"`, `"slow"` or `None`.

# %% exercise
from collections import OrderedDict


class TwoTier:
    def __init__(self, fast_gb, slow_gb):
        self.cap = {"fast": fast_gb, "slow": slow_gb}
        self.lru = {"fast": OrderedDict(), "slow": OrderedDict()}

    def where(self, key):
        return next((t for t in ("fast", "slow") if key in self.lru[t]), None)

    def put(self, key, gb):
        ### BEGIN SOLUTION
        for t in ("fast", "slow"):
            self.lru[t].pop(key, None)
        self.lru["fast"][key] = gb
        for tier, down in (("fast", "slow"), ("slow", None)):
            while sum(self.lru[tier].values()) > self.cap[tier]:
                victim, vgb = self.lru[tier].popitem(last=False)
                if down:
                    self.lru[down][victim] = vgb
        ### END SOLUTION

# %% check
c = TwoTier(fast_gb=2, slow_gb=2)
c.put("a", 1.5)
c.put("b", 1.5)
assert c.where("a") == "slow" and c.where("b") == "fast"
c.put("a", 1.5)                              # a returns: promoted to fast, b demoted
assert c.where("a") == "fast" and c.where("b") == "slow"
c.put("c", 1.5)                              # a -> slow pushes b off the end
assert c.where("b") is None and c.where("a") == "slow" and c.where("c") == "fast"
kv = TieredKV([Tier("fast", 2, 1), Tier("slow", 2, 1)], kv_bytes_per_token=10**6)       # the library's version
for k in ("a", "b", "a", "c"):
    kv.put(k, 1500)
assert [kv.lookup(k)[0] for k in ("a", "b", "c")] == [1, None, 0]
print("✅ TwoTier works — the same demotion chain as fleetsim.TieredKV")

# %% [markdown]
# ## Worked example — agent sessions over tiers
# 600 agent sessions arrive at 1/s on four H100 replicas. Each replica keeps ~8 GB of HBM for idle sessions. The rest
# of the HBM serves the requests that run (this is an assumption). `simulate_sessions` replays every turn. For each
# turn, it finds where the context of the session is now, and what it costs to bring the context back. It gives a
# score only to resumed turns, and its floor is the new tool-result tokens that the replica must prefill in all cases.
#
# The tiers of the simulator are exclusive: an evicted session moves down one tier. In practice, offload keeps a copy
# in the lower tier at the time when the engine writes the KV. Thus, count the capacity of a DRAM tier alone, not
# HBM + DRAM.
#
# First, the run with HBM only:

# %%
common = dict(kv_bytes_per_token=LLAMA_8B_KV, prefill_tok_s=H100_8B.compute_tok_s, replicas=4, session_rate=1.0,
              seed=3)
HBM = Tier("HBM", 8, 3350.0)
SHARED = Tier("shared", 1000, 20.0, 0.002)          # a 1 TB store at 20 GB/s (an RDMA-class assumption)


def dram(gb):
    return Tier("DRAM", gb, 50.0, 0.0005)


def replay(gb=0, sticky=True, shared=None):
    return simulate_sessions(600, tiers=[HBM] + ([dram(gb)] if gb else []), sticky=sticky, shared=shared, **common)


cols = ["tiers", "routing", "share_HBM", "share_DRAM", "share_shared", "share_recompute", "recompute_token_share",
        "prefix_cost_p95_s"]
rows = [{"tiers": "HBM only", "routing": "sticky" if st else "random", **replay(0, st)} for st in (True, False)]
print(table([{c: r.get(c, 0.0) for c in cols} for r in rows], cols,
            title="simulated: where each resumed turn found its KV (share of turns)"))

# %% [markdown]
# Even with sticky routing, HBM alone serves a minority of resumed turns.
#
# ## Exercise 5.3 — predict what a tier buys
# Before you run the three cases, **predict** the results. The cases are:
#
# * `"sticky + DRAM"`: 64 GB of DRAM per replica, session-sticky routing.
# * `"random + DRAM"`: the same tiers, random routing.
# * `"random + shared"`: HBM plus the shared 1 TB store, random routing.
#
# Which case recomputes the largest share of resumed prompt tokens? Does the shared tier get within 2 percentage
# points of the recomputed-token share of sticky routing?

# %% exercise
most_recompute = None
shared_matches_sticky = None
### BEGIN SOLUTION
most_recompute = "random + DRAM"      # each replica's DRAM only holds sessions it served itself
shared_matches_sticky = True          # any replica can pull any session's KV
### END SOLUTION

# %% check
runs = {"sticky + DRAM": replay(64, True), "random + DRAM": replay(64, False), "random + shared": replay(0, False, SHARED)}
rows += [{"tiers": f"HBM + {gb} GB DRAM", "routing": "sticky" if st else "random", **replay(gb, st)}
         for gb in (16, 64) for st in (True, False)]
rows.append({"tiers": "HBM + shared 1 TB store", "routing": "random", **runs["random + shared"]})
print(table([{c: r.get(c, 0.0) for c in cols} for r in rows], cols,
            title="simulated: where each resumed turn found its KV (share of turns)"))
got = {k: v["recompute_token_share"] for k, v in runs.items()}
assert most_recompute == max(got, key=got.get), got
assert shared_matches_sticky == (abs(got["random + shared"] - got["sticky + DRAM"]) <= 0.02), got
print("✅", {k: round(v, 3) for k, v in got.items()})

# %% [markdown]
# A DRAM tier of moderate size with sticky routing serves nearly all resumed turns. Each turn costs a few tens of
# milliseconds instead of a full re-prefill. Random routing wastes most of that tier, because each replica holds only
# the sessions that it served by chance. But if the tier is a **shared** tier, the router is free to balance the load
# again.
#
# ## Exercise 5.4 — size the DRAM tier
# The floor is the value that sticky routing gets with unlimited DRAM. Write
# `smallest_dram(candidates, recompute_share, tolerance)`. Return the smallest candidate size whose recomputed-token
# share is within `tolerance` of the floor. `recompute_share(gb)` runs the simulation for one size.

# %% exercise
def smallest_dram(candidates, recompute_share, tolerance=0.01):
    ### BEGIN SOLUTION
    floor = recompute_share(10_000)
    return next(gb for gb in sorted(candidates) if recompute_share(gb) <= floor + tolerance)
    ### END SOLUTION

# %% check
def recompute_share(gb):
    return replay(gb, True)["recompute_token_share"]


pick = smallest_dram([0, 2, 4, 8, 16, 32, 64], recompute_share)
ok = [gb for gb in [0, 2, 4, 8, 16, 32, 64] if recompute_share(gb) <= recompute_share(10_000) + 0.01]
assert pick == min(ok)
print(f"✅ {pick} GB of DRAM per replica reaches the floor for this workload — size from a replay, not a guess")

# %% [markdown]
# ## In a design review
# **Two-minute version.** "Agent sessions send a history again at each turn, and this history becomes longer. The
# sessions are idle during tool calls. Thus their KV is the most reusable, and it is also the first KV that the
# replica evicts. The working set is many times what HBM can spare.
#
# "First, I will add a CPU-memory offload tier on every replica. It is faster than recompute on any GPU, because PCIe
# moves KV faster than prefill creates it. I will also keep the routing session-sticky, with a load gate.
#
# "A shared KV store (LMCache or Mooncake over RDMA) lets any replica resume any session. This is applicable if we
# must balance the load again freely, or continue after the loss of a replica. I will calculate the size of the
# tiers from a replay of session traces from production: the hit shares per tier and the recomputed-token floor."
#
# **Drills**
# 1. *Offload to NVMe or just recompute?* Compare the read bandwidth of the drive with
#    $\text{KV bytes/token} \times \text{prefill tokens/s}$. This rate is ~0.5 GB/s for an 8B model on an L4, and NVMe
#    wins by a large margin. On an H100, the rate is ~4 GB/s, and a single drive is at the limit.
# 2. *Why does sticky routing matter more once you offload?* A local tier helps only the sessions that come back to
#    it. Random routing changes most resumes into misses, unless all the replicas share the tier.
# 3. *What limits a shared tier?* Three things limit it. The first is its network bandwidth and latency against the
#    break-even. The second is its capacity ($\text{sessions} \times \text{context} \times \text{bytes}$). The third
#    is consistency: every replica must use the same chain hashes and model version in the block keys.
