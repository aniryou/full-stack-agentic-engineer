# %% [markdown]
# # 05 · KV-cache tiers and agent sessions
#
# **Tier:** T0 — CPU only, a few seconds, no network. Every number below is **simulated** by `fleetsim`; tier
# bandwidths are stated assumptions, not measurements.
#
# ## The one-minute version
# An agent session re-sends its whole, growing history every turn and pauses between turns while tools run. That
# makes its KV cache the most reusable state in the fleet — and the most evictable: the **working set** (sessions x
# context x KV bytes per token) is many times the HBM a replica can spare, so LRU evicts a session during its tool
# call and the next turn re-prefills everything. Two moves fix it:
#
# * **Offload** evicted KV to host DRAM, local NVMe or a remote store (vLLM's OffloadingConnector, LMCache, Mooncake,
#   SGLang HiCache). Fetching beats recomputing whenever the tier's bandwidth exceeds the rate at which prefill
#   *produces* KV: `KV bytes/token x prefill tokens/s` — about 0.5 GB/s for an 8B model on an L4, 4 GB/s on an H100.
# * **Keep sessions near their KV** — sticky routing — or make the KV reachable from anywhere with a **shared** tier,
#   which frees the router to balance load again (notebook 02's tension, dissolved by memory).
#
# Primer §6.

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
# Two hundred coding-agent sessions at 30k tokens of context need ~800 GB of KV — fifteen H100s' worth of KV pool,
# before any of them has generated a token. Most of those sessions are idle at any instant (waiting on a tool), so
# the question is not "can HBM hold it" (no) but "where should the idle ones live, and what does it cost to bring one
# back?"
#
# ## Worked example — routing cannot fix a capacity problem
# The same four-L4 fleet and EPP router as notebook 02, with short (1–4 s) and long (10–40 s) pauses between turns.

# %%
rows = []
for think in ((1, 4), (10, 40)):
    reqs = agentic(0.6, 240, seed=3, groups=6, system=2000, tool=500, output=100, turns=(4, 10), think=think)
    s = Fleet(L4_8B, 4, epp()).run(reqs).summary(ttft_slo=1.0)
    rows.append({"tool pause s": f"{think[0]}-{think[1]}", "hit_rate": s["hit_rate"], "ttft_p50": s["ttft_p50"],
                 "ttft_p95": s["ttft_p95"]})
print(table(rows, title="simulated: EPP 3:2:2 on 4x L4, agent sessions"))

# %% [markdown]
# Longer pauses mean more sessions are alive at once, so each session's KV is evicted before its next turn: the hit
# rate falls back to "system prompts only", whatever the router does.
#
# ## Exercise 5.1 — fetch or recompute?
# Recomputing `n` tokens of KV takes `n / prefill_tok_s`; fetching them takes `n x kv_bytes_per_token / bandwidth`.
# Write `fetch_beats_recompute(profile, tiers)`: the names of the tiers whose read bandwidth exceeds the break-even
# `kv_bytes_per_token x prefill_tok_s` (in GB/s).

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
# ## Exercise 5.2 — two tiers, demote on evict
# Implement the core of an offloading cache: `put(key, gb)` inserts into the fast tier as most recently used; while
# the fast tier is over capacity, its least recently used entry is **demoted** to the slow tier; while the slow tier
# is over capacity, its LRU entry is **dropped**. `where(key)` returns `"fast"`, `"slow"` or `None`.

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
# 600 agent sessions arrive at 1/s on four H100 replicas; each keeps ~8 GB of HBM for idle sessions (the rest serves
# running requests — an assumption). `simulate_sessions` replays every turn: where is the session's context now,
# and what does bringing it back cost? It scores resumed turns only; its floor is the new tool-result tokens that
# must be prefilled anyway.

# %%
common = dict(kv_bytes_per_token=LLAMA_8B_KV, prefill_tok_s=H100_8B.compute_tok_s, replicas=4, session_rate=1.0,
              seed=3)
HBM = Tier("HBM", 8, 3350.0)


def dram(gb):
    return Tier("DRAM", gb, 50.0, 0.0005)


rows = []
for gb in (0, 16, 64):
    for sticky in (True, False):
        r = simulate_sessions(600, tiers=[HBM] + ([dram(gb)] if gb else []), sticky=sticky, **common)
        rows.append({"tiers": f"HBM + {gb} GB DRAM" if gb else "HBM only", "routing": "sticky" if sticky else "random",
                     **r})
shared = simulate_sessions(600, tiers=[HBM], sticky=False, shared=Tier("shared", 1000, 20.0, 0.002), **common)
rows.append({"tiers": "HBM + shared 1 TB store", "routing": "random", **shared})
cols = ["tiers", "routing", "share_HBM", "share_DRAM", "share_shared", "share_recompute", "recompute_token_share",
        "prefix_cost_p95_s"]
print(table([{c: r.get(c, 0.0) for c in cols} for r in rows], cols,
            title="simulated: where each resumed turn found its KV (share of turns)"))

# %% [markdown]
# HBM alone serves a minority of resumed turns even with sticky routing. A modest DRAM tier with sticky routing
# serves nearly all of them, at a few tens of milliseconds instead of a full re-prefill. Random routing wastes most
# of that tier — each replica only holds the sessions it happened to serve — unless the tier is **shared**, in
# which case routing is free to chase load again.
#
# ## Exercise 5.3 — size the DRAM tier
# The floor is what sticky routing reaches with unlimited DRAM. Write `smallest_dram(candidates, recompute_share,
# tolerance)`: return the smallest candidate size whose recomputed-token share is within `tolerance` of the floor
# (`recompute_share(gb)` runs the simulation for one size).

# %% exercise
def smallest_dram(candidates, recompute_share, tolerance=0.01):
    ### BEGIN SOLUTION
    floor = recompute_share(10_000)
    return next(gb for gb in sorted(candidates) if recompute_share(gb) <= floor + tolerance)
    ### END SOLUTION

# %% check
def recompute_share(gb):
    tiers_ = [HBM] + ([dram(gb)] if gb else [])
    return simulate_sessions(600, tiers=tiers_, sticky=True, **common)["recompute_token_share"]


pick = smallest_dram([0, 2, 4, 8, 16, 32, 64], recompute_share)
ok = [gb for gb in [0, 2, 4, 8, 16, 32, 64] if recompute_share(gb) <= recompute_share(10_000) + 0.01]
assert pick == min(ok)
print(f"✅ {pick} GB of DRAM per replica reaches the floor for this workload — size from a replay, not a guess")

# %% [markdown]
# ## Exercise 5.4 — what does a shared tier buy?
# Answer from the table: which configuration recomputes the most (`"sticky + DRAM"`, `"random + DRAM"`,
# `"random + shared"`)? And does the shared tier get within 2 percentage points of sticky routing's recomputed-token
# share (with 64 GB of DRAM)?

# %% exercise
most_recompute = None
shared_matches_sticky = None
### BEGIN SOLUTION
most_recompute = "random + DRAM"      # each replica's DRAM only holds sessions it served itself
shared_matches_sticky = True          # any replica can pull any session's KV
### END SOLUTION

# %% check
def share(sticky, gb=64, shared_tier=None):
    tiers_ = [HBM] + ([dram(gb)] if gb else [])
    return simulate_sessions(600, tiers=tiers_, sticky=sticky, shared=shared_tier, **common)["recompute_token_share"]


got = {"sticky + DRAM": share(True), "random + DRAM": share(False),
       "random + shared": share(False, gb=0, shared_tier=Tier("shared", 1000, 20.0, 0.002))}
assert most_recompute == max(got, key=got.get), got
assert shared_matches_sticky == (abs(got["random + shared"] - got["sticky + DRAM"]) <= 0.02), got
print("✅", {k: round(v, 3) for k, v in got.items()})

# %% [markdown]
# ## In a design review
# **Two-minute version.** "Agent sessions re-send a growing history and sit idle during tool calls, so their KV is
# both the most reusable and the first to be evicted; the working set is many times what HBM can spare. I'd add a
# CPU-memory offload tier on every replica first — it beats recompute on any GPU because PCIe moves KV faster than
# prefill creates it — and keep routing session-sticky with a load gate. If we need to rebalance freely or survive
# replica loss, a shared KV store (LMCache or Mooncake over RDMA) makes any replica able to resume any session. I'd
# size the tiers from a replay of real session traces: hit shares per tier and the recomputed-token floor."
#
# **Drills**
# 1. *Offload to NVMe or just recompute?* Compare the drive's read bandwidth with KV bytes/token x prefill tokens/s:
#    ~0.5 GB/s for an 8B model on an L4 (NVMe wins easily), ~4 GB/s on an H100 (a single drive is marginal).
# 2. *Why does sticky routing matter more once you offload?* A local tier only helps the sessions that come back to
#    it; random routing turns most resumes into misses unless the tier is shared.
# 3. *What limits a shared tier?* Its network bandwidth and latency against the break-even, its capacity (sessions x
#    context x bytes), and consistency: blocks must be keyed by the same chain hashes and model version everywhere.
