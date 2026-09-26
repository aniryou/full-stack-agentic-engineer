# %% [markdown]
# # 03 · Collectives from scratch
#
# **Tier:** T0 (CPU, simulated ranks; numpy only). The real thing is in the lab's
# `03_collectives_with_torch_distributed` (gloo on CPU at T0, NCCL at T1/T2) and
# `04_busbw_and_the_alpha_beta_fit` (fits alpha and beta to nccl-tests output).
#
# ## The one-minute version
# * A **collective** is defined by what every rank holds before and after. Broadcast,
#   all-reduce, reduce-scatter, all-gather, all-to-all and send/recv are the ones inference uses.
# * **All-reduce = reduce-scatter + all-gather.** On a ring of p ranks that is 2(p-1) steps, each
#   moving S/p bytes, so `T = 2(p-1)*alpha + 2(p-1)/p * S/B`. That is bandwidth-optimal, but its
#   latency term grows with p.
# * Below a crossover size (`S* = p*alpha*B` for the ring) a collective is **latency-bound**.
#   Tensor-parallel all-reduces during decode are a few hundred KB, which makes them
#   latency-bound. During prefill they are tens of MB and bandwidth-bound. Different algorithms
#   win in each regime.
# * **busbw** (nccl-tests) = algbw x 2(p-1)/p for all-reduce. It turns "bytes per second of
#   buffer" into "bytes per second per link", which you can compare with the NVLink or NIC spec
#   whatever p is.
# * NCCL matches collectives **by issue order**. One rank that issues a different call, or skips
#   one, hangs all the others.
#
# Primer: §5 *Collectives* (`../../PRIMER.md`).

# %%
import numpy as np

from gpusim import collectives as C

p = 4
bufs = [np.arange(4) + 10 * r for r in range(p)]          # rank r holds [10r, 10r+1, 10r+2, 10r+3]
print("before      :", [b.tolist() for b in bufs])
for op in ("broadcast", "reduce", "all_reduce", "reduce_scatter", "all_gather", "all_to_all"):
    print(f"{op:<12}:", [x.tolist() for x in C.reference(op, bufs, root=0)])

# %% [markdown]
# Read each line as a contract:
#
# * **broadcast**: everyone ends with root's buffer. Used for weights or config from rank 0.
# * **reduce**: root ends with the elementwise sum. Rare in inference.
# * **all-reduce**: *everyone* ends with the sum. Tensor parallelism uses it twice per layer.
# * **reduce-scatter**: rank r ends with *chunk r* of the sum. That is the first half of an
#   all-reduce, and sequence parallelism uses it.
# * **all-gather**: everyone ends with the concatenation. That is the second half, and it
#   gathers sharded weights or logits.
# * **all-to-all**: rank r's chunk j goes to rank j, a transpose across ranks. MoE expert
#   parallelism uses it to dispatch and combine tokens.
#
# ## Exercise 3.1: ring reduce-scatter
#
# Implement it with the rules a ring imposes: in every step, **each rank sends exactly one
# chunk to its right neighbour** `(r + 1) % p`, which **adds** it into its own copy of that chunk.
# All sends in a step happen at once, so compute every message from the state at the *start* of
# the step. After the last step, rank r must own the fully reduced chunk r.
#
# Return the chunks and the **schedule**: one list per step of the `(src, dst, chunk)` messages
# sent in that step. The check replays your schedule on fresh buffers, so the schedule, not only
# the final answer, has to be right. (Hint: at step s = 0, 1, ..., p-2, rank r sends chunk
# `(r - s - 1) % p`.)

# %% exercise
def ring_reduce_scatter(bufs):
    p = len(bufs)
    work = [np.array(b, copy=True) for b in bufs]
    idx = np.array_split(np.arange(work[0].size), p)        # chunk c = work[r][idx[c]]
    ### BEGIN SOLUTION
    schedule = []
    for s in range(p - 1):
        msgs = [(r, (r + 1) % p, (r - s - 1) % p) for r in range(p)]
        payload = [(dst, c, work[src][idx[c]].copy()) for src, dst, c in msgs]
        for dst, c, data in payload:
            work[dst][idx[c]] += data
        schedule.append(msgs)
    return [work[r][idx[r]] for r in range(p)], schedule
    ### END SOLUTION

# %% check
def replay(bufs, schedule, op):
    """Run a schedule of (src, dst, chunk) messages; each step's messages are computed from the
    state at the start of the step. op="add" reduces into dst, op="copy" overwrites it."""
    work = [np.array(b, copy=True) for b in bufs]
    idx = np.array_split(np.arange(work[0].size), len(work))
    for step in schedule:
        payload = [(dst, c, work[src][idx[c]].copy()) for src, dst, c in step]
        for dst, c, data in payload:
            if op == "add":
                work[dst][idx[c]] += data
            else:
                work[dst][idx[c]] = data
    return work, idx

def check_ring_schedule(schedule, p):
    assert len(schedule) == p - 1, f"a ring phase takes p-1 = {p - 1} steps, not {len(schedule)}"
    for s, step in enumerate(schedule):
        assert sorted(src for src, _, _ in step) == list(range(p)), f"step {s}: every rank sends exactly once"
        assert all(dst == (src + 1) % p for src, dst, _ in step), f"step {s}: a ring only sends to (r + 1) % p"

rng = np.random.default_rng(0)
for p_ in (2, 3, 4, 5, 8):
    b = [rng.integers(-99, 99, 3 * p_ + 1) for _ in range(p_)]
    chunks, sched = ring_reduce_scatter(b)
    check_ring_schedule(sched, p_)
    want = C.reference("reduce_scatter", b)
    work, idx = replay(b, sched, "add")
    assert all(np.array_equal(work[r][idx[r]], want[r]) for r in range(p_)), "replaying your schedule does not leave chunk r's sum on rank r"
    assert all(np.array_equal(x, y) for x, y in zip(chunks, want))
    sent = [sum(idx[c].size for step in sched for src, _, c in step if src == r) for r in range(p_)]
    assert all(abs(n - (p_ - 1) / p_ * b[0].size) < p_ for n in sent)       # (p-1)/p of the buffer each
print("✅ ring reduce-scatter: p-1 steps, one S/p chunk per rank per step to its right neighbour")

# %% [markdown]
# ## Exercise 3.2: ring all-gather, and all-reduce as the composition
#
# Now each rank r starts with only its own finished chunk. Each step, every rank forwards one
# chunk to its right neighbour, which *stores* it. After p-1 steps everyone has every chunk.
# Write `ring_all_gather(chunks)` returning the full buffers and its schedule, then
# `ring_all_reduce(bufs)` built from the two, returning the full buffers and the combined
# schedule (reduce-scatter steps first).

# %% exercise
def ring_all_gather(chunks):
    p = len(chunks)
    have = [{r: np.array(chunks[r], copy=True)} for r in range(p)]   # rank r's known chunks
    ### BEGIN SOLUTION
    schedule = []
    for s in range(p - 1):
        msgs = [(r, (r + 1) % p, (r - s) % p) for r in range(p)]
        payload = [(dst, c, have[src][c]) for src, dst, c in msgs]
        for dst, c, data in payload:
            have[dst][c] = data.copy()
        schedule.append(msgs)
    full = [np.concatenate([have[r][c] for c in range(p)]) for r in range(p)]
    return full, schedule
    ### END SOLUTION

def ring_all_reduce(bufs):
    ### BEGIN SOLUTION
    chunks, rs = ring_reduce_scatter(bufs)
    full, ag = ring_all_gather(chunks)
    return full, rs + ag
    ### END SOLUTION

# %% check
for p_ in (2, 3, 4, 6, 8):
    b = [rng.integers(-99, 99, 5 * p_ + 3) for _ in range(p_)]
    full, sched = ring_all_reduce(b)
    assert len(sched) == 2 * (p_ - 1)
    check_ring_schedule(sched[: p_ - 1], p_)
    check_ring_schedule(sched[p_ - 1:], p_)
    reduced, _ = replay(b, sched[: p_ - 1], "add")
    gathered, _ = replay(reduced, sched[p_ - 1:], "copy")
    assert all(np.array_equal(g, np.sum(b, axis=0)) for g in gathered), "replaying your schedule does not all-reduce"
    assert all(np.array_equal(x, np.sum(b, axis=0)) for x in full)
print("✅ ring all-reduce = reduce-scatter + all-gather: 2(p-1) steps of S/p bytes")

# %% [markdown]
# ## Reading traces
# `gpusim.collectives` runs the same schedules (and a few more) and records every message. Here
# are the ring and three other ways to all-reduce the same 4 ranks:

# %%
bufs = [rng.integers(0, 9, 8) for _ in range(4)]
for algo in ("ring", "tree", "two_shot", "switch"):
    r = C.all_reduce(bufs, algo, chunks=2)
    assert all(np.array_equal(o, np.sum(bufs, axis=0)) for o in r.out)
    print(r.trace.table(), "\n   bytes sent per rank:", r.trace.sent_bytes(), "\n")

# %% [markdown]
# * **tree** (binomial): halves the active ranks each step, so 2*ceil(log2 p) steps instead of
#   2(p-1). But every message is the *whole* buffer, which makes it latency-cheap and
#   bandwidth-poor. NCCL's real tree is a *pipelined double binary tree*, which keeps the log-p
#   steps and recovers most of the bandwidth.
# * **two_shot**: a direct reduce-scatter then a direct all-gather. That is 2 steps with the same
#   bytes as the ring, but it needs every rank to reach every other at once (NVSwitch). This is
#   what the custom all-reduce kernels in vLLM and TensorRT-LLM do for small and mid-size messages.
# * **switch**: NVLS / SHARP in-network reduction. Each GPU sends its data *once* to the switch,
#   which sums and multicasts the result back, so each GPU sends S instead of 2(p-1)/p * S.
#
# ## The alpha-beta model
# Every step costs `alpha` (launch, synchronisation, hop latency) plus its busiest port's bytes
# divided by `B`. The trace time equals the closed form exactly:

# %%
S, alpha, bw = 8 * 2**20, 2e-6, 100e9                   # assumed parameters, not measurements
for algo in ("ring", "tree", "one_shot", "two_shot", "switch"):
    tr = C.all_reduce([np.ones(S // 4, np.float32)] * 8, algo, chunks=8).trace
    a, c = C.cost_terms("all_reduce", algo, 8, chunks=8)
    print(f"{algo:>9}: {tr.n_steps:>2} steps, T = {a:>2}*alpha + {c:5.3f}*S/B = {tr.time(alpha, bw) * 1e6:7.1f} us "
          f"(closed form {C.model_time('all_reduce', algo, S, 8, alpha, bw, 8) * 1e6:7.1f} us)")

# %% [markdown]
# ## Exercise 3.3: time and crossover of a ring all-reduce
#
# Write `ring_allreduce_time(S, p, alpha, bw)` from the step structure you implemented, and
# `crossover_size(p, alpha, bw)`: the S at which the latency term equals the bandwidth term.
# Then evaluate it for 8 GPUs with layer 01's illustrative NVLink 4 numbers: `alpha = 2 us` per step
# and `B = 450 GB/s` per direction.

# %% exercise
def ring_allreduce_time(S, p, alpha, bw):
    ### BEGIN SOLUTION
    return 2 * (p - 1) * alpha + 2 * (p - 1) / p * S / bw
    ### END SOLUTION

def crossover_size(p, alpha, bw):
    ### BEGIN SOLUTION
    return p * alpha * bw          # 2(p-1)*alpha == 2(p-1)/p * S/B  <=>  S = p*alpha*B
    ### END SOLUTION

# %% check
for p_ in (2, 4, 8, 16, 64):
    for S_ in (8, 2**20, 2**30):
        assert np.isclose(ring_allreduce_time(S_, p_, 2e-6, 450e9), C.model_time("all_reduce", "ring", S_, p_, 2e-6, 450e9))
    assert np.isclose(crossover_size(p_, 2e-6, 450e9), C.crossover_bytes("all_reduce", "ring", p_, 2e-6, 450e9))
print(f"✅ 8 GPUs, alpha=2us, B=450GB/s: ring all-reduce is latency-bound below {crossover_size(8, 2e-6, 450e9) / 1e6:.1f} MB")
print("   and the crossover grows with p: every extra rank adds two alpha-steps to the ring")

# %% [markdown]
# ## algbw vs busbw
# nccl-tests reports two bandwidths. `algbw = S / t` is the buffer size over time. `busbw`
# multiplies by a per-collective factor (2(p-1)/p for all-reduce; (p-1)/p for reduce-scatter,
# all-gather and all-to-all; 1 for broadcast and reduce) so that a bandwidth-optimal algorithm
# reads the per-link bandwidth whatever p is. A simulated sweep in nccl-tests' shape:

# %%
print(C.format_sweep(C.sweep("all_reduce", "ring", p=8, alpha=2e-6, bw=450e9,
                             sizes=[2**e for e in (10, 13, 16, 19, 20, 22, 24, 27, 30)])))

# %% [markdown]
# busbw climbs toward B (450 GB/s here) as messages grow and alpha stops mattering. On a real
# system that plateau is what you compare with the fabric spec: NVLink, PCIe or the NIC line
# rate. The size where busbw reaches half the plateau is about the crossover size.
#
# ## Exercise 3.4: compute busbw like nccl-tests
#
# Each tuple is `(op, ranks, size_bytes, time_us)` in nccl-tests' conventions. These are
# **illustrative numbers in the documented format, not measurements.** Write
# `bandwidths(op, ranks, size, time_us)` returning `(algbw_GBps, busbw_GBps)` with GB = 1e9 bytes.

# %% exercise
rows = [("all_reduce", 8, 134217728, 560.0), ("all_gather", 8, 134217728, 300.0),
        ("reduce_scatter", 4, 67108864, 160.0), ("broadcast", 8, 1073741824, 2450.0)]

def bandwidths(op, ranks, size, time_us):
    ### BEGIN SOLUTION
    factor = {"all_reduce": 2 * (ranks - 1) / ranks, "reduce_scatter": (ranks - 1) / ranks,
              "all_gather": (ranks - 1) / ranks, "all_to_all": (ranks - 1) / ranks,
              "broadcast": 1.0, "reduce": 1.0}[op]
    algbw = size / (time_us * 1e-6) / 1e9
    return algbw, algbw * factor
    ### END SOLUTION

# %% check
for op, n_, s_, t_ in rows:
    a_, b_ = bandwidths(op, n_, s_, t_)
    assert np.isclose(a_, C.algbw(s_, t_ * 1e-6) / 1e9) and np.isclose(b_, C.busbw(op, s_, t_ * 1e-6, n_) / 1e9), op
    print(f"  {op:>14} x{n_}: algbw {a_:6.1f} GB/s  busbw {b_:6.1f} GB/s")
print("✅ busbw, not algbw, is the number to hold against the link spec")

# %% [markdown]
# ## How inference uses collectives
# **Tensor parallelism** (Megatron-style) splits every layer's matmuls across p GPUs and
# all-reduces the activations twice per layer, after attention's output projection and after the
# MLP. The message is `tokens x hidden x 2 bytes`. For a 70B-class dense model (hidden 8192, 80
# layers) at TP=8, with the same illustrative alpha = 2 us and B = 450 GB/s (layer 01 §5.3 prices
# batch 1; here we compare batch 32 with prefill, and the ring with a two-shot all-reduce):

# %%
for phase, tokens in (("decode, batch 32", 32), ("prefill, 8k tokens", 8192)):
    for algo in ("ring", "two_shot"):
        d = C.tp_comm(layers=80, tokens=tokens, hidden=8192, p=8, alpha=2e-6, bw=450e9, algo=algo)
        print(f"{phase:>19} {algo:>8}: {d['message_bytes'] / 2**20:7.2f} MiB x {d['calls']} calls = "
              f"{d['total_s'] * 1e3:7.2f} ms/step, latency share {d['latency_share']:.0%}")

# %% [markdown]
# Two regimes, from one formula. In decode, 93% of the ring's time is alpha. The fix is fewer
# steps (two-shot or one-shot custom all-reduce, NVLS, tree), not more bandwidth, plus CUDA
# graphs so the 160 launches per step cost nothing on the CPU. In prefill the messages are large
# and bandwidth-bound, so the ring's optimal byte count is what matters, and so is overlapping
# the communication with compute.
#
# ## Exercise 3.5: the decode all-reduce you would ship
#
# Same model and TP=8, but a decode batch of **64** tokens. With the same assumed alpha and B,
# compute `message_bytes`, `ring_ms_per_step` (all 160 all-reduces with a ring), and
# `best_algo`, the fastest of `C.best_algorithm` for this message. (`best_algorithm` pipelines the
# in-switch algorithm at its best depth for each size, `C.pipeline_chunks()`: for a 1 MiB message
# that is a single chunk, because extra chunks only add alpha-steps.)

# %% exercise
### BEGIN SOLUTION
message_bytes = 64 * 8192 * 2
ring_ms_per_step = 160 * C.model_time("all_reduce", "ring", message_bytes, 8, 2e-6, 450e9) * 1e3
best_algo = C.best_algorithm("all_reduce", message_bytes, 8, 2e-6, 450e9)[0][1]
### END SOLUTION

# %% check
assert message_bytes == 2**20
assert np.isclose(ring_ms_per_step, C.tp_comm(80, 64, 8192, 8, 2e-6, 450e9, "ring")["total_s"] * 1e3)
assert best_algo == "two_shot"
print(f"✅ 1 MiB per call; ring {ring_ms_per_step:.2f} ms/step; best is {best_algo}: "
      "2 steps, ring-optimal bytes, but it needs all-to-all connectivity (NVSwitch)")

# %% [markdown]
# **Expert parallelism** (MoE) moves tokens rather than partial sums. Each token goes to its
# top-k experts' GPUs (all-to-all *dispatch*) and comes back (all-to-all *combine*). Per GPU and
# direction that is `tokens x top_k x hidden x bytes`, of which (p-1)/p crosses the fabric. For
# a Mixtral-like layer (hidden 4096, top-2) with 256 tokens per GPU in BF16:

# %%
dispatch = 256 * 2 * 4096 * 2
for algo in ("pairwise", "direct"):
    t = C.model_time("all_to_all", algo, dispatch, 8, 2e-6, 450e9)
    print(f"all-to-all {algo:>8}: {dispatch / 2**20:.0f} MiB per GPU, {t * 1e6:.1f} us per dispatch (x2 with combine)")

# %% [markdown]
# Pipeline parallelism sends one `tokens x hidden` activation per stage boundary per micro-batch
# (point-to-point, not per layer). That is tiny, which is why PP tolerates the slower scale-out
# network while TP and EP stay inside the NVLink domain.
#
# ## Exercise 3.6: find the hang
#
# NCCL has no names for collectives. The k-th call on a communicator on every rank is the *same*
# collective. Write `first_bad_call(calls)`, where `calls[rank]` is that rank's list of
# `(op, nbytes)`. Return `(index, sorted list of ranks that disagree with the majority)` for the
# first mismatch (a missing call counts as disagreeing), or `None` if all ranks agree.

# %% exercise
def first_bad_call(calls):
    ### BEGIN SOLUTION
    from collections import Counter
    for i in range(max(len(c) for c in calls.values())):
        seen = {r: (c[i] if i < len(c) else None) for r, c in calls.items()}
        if len(set(seen.values())) > 1:
            majority = Counter(seen.values()).most_common(1)[0][0]
            return i, sorted(r for r, v in seen.items() if v != majority)
    return None
    ### END SOLUTION

# %% check
ar = ("all_reduce", 1 << 20)
logs = {0: [ar, ar, ar, ar], 1: [ar, ar, ar, ar], 2: [ar, ar, ("all_gather", 1 << 20), ar], 3: [ar, ar, ar, ar]}
assert first_bad_call(logs) == (2, [2])
crashed = {0: [ar, ar, ar], 1: [ar, ar, ar], 2: [ar], 3: [ar, ar, ar]}
assert first_bad_call(crashed) == (1, [2])
assert first_bad_call({0: [ar], 1: [ar]}) is None
for case in (logs, crashed):
    m = C.first_mismatch(case)
    assert first_bad_call(case) == (m["call"], m["odd_ranks"])
print("✅ the first call where ranks disagree is where everyone else blocks until the timeout")

# %% [markdown]
# Typical causes: a data-dependent branch that only some ranks take (for example an
# "if loss is nan" path), a rank that crashed or went OOM, different shapes per rank, or ranks
# issuing collectives on different streams or communicators in different orders. With
# `NCCL_DEBUG=INFO` and PyTorch's flight recorder you get each rank's last collective, which is
# exactly the input to this function.
#
# One more thing floats teach you: the ring and the tree add in **different orders**, so their
# float32 results differ in the last bits. Every rank still agrees *within* one algorithm.

# %%
xs = [np.random.default_rng(i).normal(size=4096).astype(np.float32) for i in range(8)]
ring_out, tree_out = C.all_reduce(xs, "ring").out, C.all_reduce(xs, "tree").out
print("ranks agree within the ring:", all(np.array_equal(ring_out[0], o) for o in ring_out))
print("ring vs tree max |diff|    :", float(np.max(np.abs(ring_out[0] - tree_out[0]))))

# %% [markdown]
# ## In a design review
#
# **The two-minute version.** "We serve with TP=8 inside one NVLink domain. Each layer
# all-reduces twice, a message of tokens x hidden x 2 bytes. With alpha and B fitted from our own
# nccl-tests run (here the illustrative 2 us and 450 GB/s), the crossover is about 7 MB
# (S* = p x alpha x B), so decode
# all-reduces, at 0.5 to 1 MB, are
# latency-bound. We use an algorithm with few steps (a one-/two-shot custom all-reduce, or NVLS
# where the switch supports it) and capture it in the decode CUDA graph. Prefill messages are
# 100+ MB and bandwidth-bound, and there NCCL's ring or NVLS is at the link limit, which we
# verify as busbw from nccl-tests against the NVLink spec. EP all-to-alls stay inside the NVLink
# domain too. PP, which moves one activation per stage, is the only parallelism we let cross the
# scale-out network."
#
# **Drill questions**
#
# 1. *Why is all-reduce 2(p-1) steps and not p-1?* It is a reduce-scatter (p-1 steps to finish
#    each chunk's sum) followed by an all-gather (p-1 steps to spread the finished chunks).
#    Each rank sends 2(p-1)/p x S in total, and no algorithm that uses point-to-point sends can
#    do less.
# 2. *Our 8-GPU all-reduce shows busbw of 419 GB/s but algbw of 240 GB/s. Which do we quote?*
#    busbw: it equals algbw x 2(p-1)/p and is what the links actually carried per GPU. algbw
#    depends on p.
# 3. *The job hangs at step 1,000 on one rank's data-dependent branch. Why does NCCL not error?*
#    Collectives are matched only by issue order. The other ranks sit in a call that will never
#    be matched until a watchdog timeout fires. The fix is to make control flow rank-uniform
#    and to set timeouts so a hang becomes an error.
