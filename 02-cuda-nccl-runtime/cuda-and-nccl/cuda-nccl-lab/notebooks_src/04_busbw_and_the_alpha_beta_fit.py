# %% [markdown]
# # 04 · busbw and the α-β fit: reading nccl-tests like a performance engineer
#
# **Tier:** T0. It works on the bundled nccl-tests sample (**illustrative**: `tools/make_fixtures.py`
# generated it from a stated α-β model, and it is not a measurement on hardware). It also works on a sweep
# of this machine itself, with the backend of notebook 03. If you bring your own logs from a GPU box
# (`deploy/any-gpu/run_nccl_tests.sh`, `deploy/gke` Job), the same cells read them. That is the T1/T2 use.
#
# ## The one-minute version
#
# * You can calculate each column of nccl-tests again: **$\text{algbw} = \text{size}/\text{time}$**,
#   **$\text{busbw} = \text{algbw} \times \operatorname{factor}(\text{op}, n)$**.
# * A size sweep has two regimes: a **latency floor** (the time does not change with size) and a
#   **bandwidth plateau** (busbw does not change). The α-β fit $t = \alpha + S/B$ gives the two, and the
#   **half-bandwidth size $S_{1/2} = \alpha \cdot B$** is the boundary between them.
# * Compare the plateau busbw with the **link**: NVLink hundreds of GB/s per GPU, PCIe Gen4 x16 about
#   25 GB/s in practice. If the plateau is far below the link, NCCL does not use the path that you think.
# * Inference: the tensor-parallel all-reduces of decode are on the latency floor. $\alpha$, not bandwidth,
#   sets their cost. The all-reduces of prefill are on the plateau.
#
# Concepts: [the primer](../../PRIMER.md) §5 *Collectives* (α-β costs, algbw and busbw, latency-bound
# against bandwidth-bound messages).

# %%
from pathlib import Path

from gpurt import nccltests
from gpurt.dist import alphabeta as ab
from gpurt.dist import busbw as bw

FIX = Path(nccltests.__file__).parent / "fixtures"
text = (FIX / "nccl_all_reduce_8gpu_sample.txt").read_text()
print("\n".join(text.splitlines()[:2]))  # the provenance header: illustrative, and the generating model
res = nccltests.parse(text, op="all_reduce")
print(f"\nop={res.op} ranks={res.nranks} rows={len(res.rows)} header={res.header}")
for r in res.rows[:3] + res.rows[-3:]:
    print(f"{r.size:>12} B  {r.time_us:>9.2f} µs  algbw {r.algbw:7.2f}  busbw {r.busbw:7.2f} GB/s")

# %% [markdown]
# ## Exercise 4.1 — recompute the columns
#
# Write `recompute(size_bytes, time_us, op, n)`. It returns `(algbw, busbw)` in GB/s (1e9). The check
# compares the result with each printed row (each printed value has two decimals after rounding).

# %% exercise
def recompute(size_bytes: int, time_us: float, op: str, n: int) -> tuple[float, float]:
    ### BEGIN SOLUTION
    algbw = size_bytes / (time_us * 1e-6) / 1e9
    return algbw, algbw * bw.bus_factor(op, n)
    ### END SOLUTION

# %% check
for r in res.rows:
    a, b = recompute(r.size, r.time_us, "all_reduce", res.nranks)
    assert abs(a - r.algbw) <= max(0.011, 0.02 * a) and abs(b - r.busbw) <= max(0.011, 0.02 * b), r
print(f"✅ every row: busbw = algbw x 2(n-1)/n = algbw x {bw.bus_factor('all_reduce', res.nranks)} for n = {res.nranks}")

# %% [markdown]
# ## Seeing the two regimes
#
# The next cell shows busbw against size, as a text chart. Small messages are almost not visible. The curve
# bends near $S_{1/2}$ and becomes flat at the link-limited plateau.

# %%
peak = max(r.busbw for r in res.rows)
for r in res.rows[::2]:
    print(f"{r.size:>12} B |{'#' * int(50 * r.busbw / peak):<50}| {r.busbw:7.2f} GB/s")

# %% [markdown]
# ## Exercise 4.2 — fit α and B yourself
#
# Fit $t = \alpha + S/B$ to (size, time) by least squares on the **relative** error. Minimise
# $\sum ((\alpha + \beta \cdot S - t)/t)^2$ with $\beta = 1/B$. The relative error gives a 20 µs point and a 5 ms
# point the same weight. Divide each row of the linear system $[1, S] \cdot [\alpha, \beta] = t$ by $t$. Use
# `np.linalg.lstsq`. Return `(alpha_s, bw_Bps)`.

# %% exercise
import numpy as np  # noqa: E402


def fit_alpha_beta(sizes, times_s) -> tuple[float, float]:
    ### BEGIN SOLUTION
    S, t = np.asarray(sizes, float), np.asarray(times_s, float)
    X = np.column_stack([1 / t, S / t])
    (alpha, beta), *_ = np.linalg.lstsq(X, np.ones_like(t), rcond=None)
    return float(alpha), float(1 / beta)
    ### END SOLUTION

# %% check
sizes = [r.size for r in res.rows]
times = [r.time_us * 1e-6 for r in res.rows]
alpha, B = fit_alpha_beta(sizes, times)
lib = ab.fit(sizes, times)
assert abs(alpha - lib.alpha_s) <= 1e-9 * lib.alpha_s + 1e-12 and abs(B - lib.bw_Bps) <= 1e-6 * lib.bw_Bps
assert abs(alpha - 20e-6) < 2e-6  # the sample's header says it was generated with alpha = 20 us
assert abs(B * bw.bus_factor("all_reduce", 8) / 1e9 - 400) < 20  # ... and busbw -> 400 GB/s
print(f"✅ α = {alpha * 1e6:.1f} µs, algbw -> {B / 1e9:.1f} GB/s, busbw -> {B * 1.75 / 1e9:.0f} GB/s "
      "(recovered the model behind the illustrative sample)")

# %% [markdown]
# ## Exercise 4.3 — the half-bandwidth size, two ways
#
# The $\alpha$ of the fit is the latency of the **whole collective**, and $B$ is its asymptotic **algbw**. The
# primer writes the same ring with a per-hop latency $\alpha_{\text{hop}}$ and a per-link bandwidth
# $B_{\text{link}}$: $t = 2(n-1) \cdot \alpha_{\text{hop}} + 2(n-1)/n \cdot S/B_{\text{link}}$. The two terms
# cross over at $S^{\ast} = n \cdot \alpha_{\text{hop}} \cdot B_{\text{link}}$.
# Write `half_size(alpha_s, bw_Bps)` ($S_{1/2} = \alpha \cdot B$). Also write `per_hop(alpha_s, algbw_Bps, n)`,
# which returns `(alpha_hop, link_Bps)`. Then make sure that the two views give the same size.

# %% exercise
def half_size(alpha_s: float, bw_Bps: float) -> float:
    ### BEGIN SOLUTION
    return alpha_s * bw_Bps
    ### END SOLUTION


def per_hop(alpha_s: float, algbw_Bps: float, n: int) -> tuple[float, float]:
    ### BEGIN SOLUTION
    hops = 2 * (n - 1)
    return alpha_s / hops, algbw_Bps * hops / n
    ### END SOLUTION

# %% check
s_half = half_size(alpha, B)
a_hop, link = per_hop(alpha, B, 8)
assert abs(8 * a_hop * link - s_half) < 1e-6 * s_half
assert abs(link - B * 1.75) < 1e-6 * link  # the per-link rate *is* busbw
print(f"✅ S½ = {s_half / 2**20:.1f} MiB: α_hop = {a_hop * 1e6:.2f} µs per step, link {link / 1e9:.0f} GB/s — "
      "busbw is the fitted link bandwidth")

# %% [markdown]
# ## Regimes, row by row — and an older log format
#
# `AlphaBeta.regime()` gives a label to each size. The second sample (also illustrative) is an `all_gather`
# log in the older nccl-tests layout (no `redop`/`root` columns, a float `error` column). The parser reads
# the two layouts. For this log, the busbw factor is $(n-1)/n$.

# %%
model = ab.AlphaBeta(alpha, B)
for r in res.rows[::3]:
    print(f"{r.size:>12} B  {model.regime(r.size):>15}")
legacy = nccltests.parse((FIX / "nccl_all_gather_4gpu_legacy_sample.txt").read_text(), op="all_gather")
s = nccltests.summarize(legacy)
print(f"\nlegacy all_gather over {s['nranks']} ranks: factor {bw.bus_factor('all_gather', s['nranks'])}, "
      f"α = {s['alpha_us']:.1f} µs, busbw -> {s['busbw_asymptote_gbps']:.1f} GB/s, recheck issues: {nccltests.recheck(legacy)}")

# %% [markdown]
# ## Exercise 4.4 — what tensor parallelism costs a decode step
#
# With the fitted model, one all-reduce of $S$ bytes takes `model.time(S)`. Write
# `tp_comm_us(model, tokens, hidden, layers, dtype_bytes=2)`. It returns the all-reduce time of one decode
# step (two all-reduces of tokens × hidden per layer), in microseconds. Calculate it for a 70B-class model
# (hidden 8192, 80 layers, bf16) at batch 32 on the 8-GPU sample fabric. Compare the result with a 30 ms
# inter-token budget.

# %% exercise
def tp_comm_us(model: ab.AlphaBeta, tokens: int, hidden: int, layers: int, dtype_bytes: int = 2) -> float:
    ### BEGIN SOLUTION
    return 2 * layers * model.time(tokens * hidden * dtype_bytes) * 1e6
    ### END SOLUTION

# %% check
msg = 32 * 8192 * 2
t_us = tp_comm_us(model, 32, 8192, 80)
assert abs(t_us - 160 * model.time(msg) * 1e6) < 1e-6
assert model.regime(msg) == "latency-bound"
share = t_us / 30_000
print(f"✅ {msg // 1024} KiB per all-reduce is latency-bound: {t_us / 1e3:.2f} ms of every step "
      f"({share:.0%} of a 30 ms budget); α alone is {160 * model.alpha_s * 1e3:.2f} ms of it")

# %% [markdown]
# That is why engines work to decrease **$\alpha$** for decode. The methods are:
#
# * fewer, fused all-reduces
# * algorithms with fewer steps (one-shot/two-shot kernels, trees, NVLink SHARP in the switch)
# * an overlap of communication with compute
# * a smaller tensor-parallel degree
#
# ## Exercise 4.5 — is the plateau at the link?
#
# Write `diagnose(plateau_busbw_gbps, link_gbps)`. It returns `"at the link"` when the plateau gets to at
# least 70 % of the per-direction bandwidth of the link. If not, it returns
# `"below the link: check the transport"`. Per-direction values for comparison (datasheet, verify): PCIe
# Gen3 x16 ≈ 15.75 GB/s, Gen4 x16 ≈ 31.5, NVLink on A100 300 GB/s, on H100 450 GB/s.

# %% exercise
def diagnose(plateau_busbw_gbps: float, link_gbps: float) -> str:
    ### BEGIN SOLUTION
    return "at the link" if plateau_busbw_gbps >= 0.7 * link_gbps else "below the link: check the transport"
    ### END SOLUTION

# %% check
assert diagnose(400, 450) == "at the link"  # the sample's plateau vs H100-class NVLink
assert diagnose(5.5, 15.75) == "below the link: check the transport"  # e.g. SHM through host memory on PCIe
print("✅ below the link? NCCL_DEBUG=INFO shows the transport (P2P, SHM, NET); nvidia-smi topo -m shows the path")

# %% [markdown]
# ## Your own numbers
#
# The same parser reads the tables that `gpurt.dist` prints. The next cell does a fast sweep of the CPU
# transport of this machine (the notebook 03 backend). This sweep is a measurement on a much different
# fabric. The fit tells you the $\alpha$ and $B$ of this transport. The fit error tells you how well a straight
# line describes this transport.
#
# Then the cell parses and summarises, in the same way, each nccl-tests or `gpurt.dist` log that you brought
# back from a GPU box. These logs go into the lab's `out/` (`deploy/any-gpu`) or into `deploy/gke/out/`
# (`run.sh nccl`).

# %%
from gpurt import env  # noqa: E402
from gpurt.dist.bench import run  # noqa: E402
from gpurt.dist.sweep import format_table  # noqa: E402

backend = "gloo" if env.has_module("torch") else "pipes"
rows = run(backend, "all_reduce", 2, sizes=[8 * 4 ** k for k in range(11)], iters=5, warmup=1)
mine = nccltests.parse(format_table(rows))
print(format_table(rows).splitlines()[1])
print(ab.fit_rows(rows), f"(backend {backend}, measured on this machine)")
print("recheck:", nccltests.recheck(mine) or "consistent")
# logs you brought back from a GPU box: out/ (deploy/any-gpu recipes) or deploy/gke/out/ (run.sh)
for log in sorted(Path("..").glob("out/*.log")) + sorted(Path("..").glob("deploy/*/out/*.log")):
    for run_ in nccltests.parse_many(log.read_text()):
        print(log.name, "(measured)", nccltests.summarize(run_))

# %% [markdown]
# ## In a design review
#
# **Two minutes.** To read an nccl-tests table, I calculate it again. algbw is size over time. For
# all-reduce, busbw is algbw times $2(n-1)/n$: the per-link rate that an optimal ring needs. Thus busbw
# compares with the link for any number of GPUs.
#
# I fit $t = \alpha + S/B$ to the sweep. $\alpha$ is the latency floor, and $B$ is the plateau.
# $S_{1/2} = \alpha \cdot B$ is the size where the collective gets half the bandwidth. Below that size, a
# faster link does not help, but fewer steps do help.
#
# I compare the plateau with the path (NVLink or PCIe). If the plateau is well below the path, I look at
# the transport that NCCL selected. Then I put the workload on the curve. The tensor-parallel all-reduces
# of decode are hundreds of kilobytes at most, deep in the latency regime. Thus their cost is approximately
# $2 \times \text{layers} \times \alpha$ per token.
#
# **Drill questions**
#
# 1. *busbw is above the NVLink rate. Is the tool incorrect?* No. With NVLink SHARP (NVLS), the switch does
#    the reduction. Thus less data crosses each link than the ring formula assumes. The factor is a
#    convention, not a measurement of wire bytes.
# 2. *Two 8-GPU nodes show 180 GB/s busbw, and one node shows 450. Why?* Across nodes, the ring crosses the
#    NICs. The slowest link (network, per-GPU NIC share) sets the limit. Hierarchical and tree algorithms
#    help, but the plateau is the inter-node bandwidth per GPU.
# 3. *What single number do you give for the performance of a decode all-reduce?* The time at the actual
#    message size (for example, 512 KiB), that is, the latency floor $\alpha$. Do not give the plateau busbw
#    at 1 GiB.
