# %% [markdown]
# # 03 · Collectives from scratch
#
# **Tier:** T0 (CPU, simulated ranks, numpy only). The lab has the real thing:
# `03_collectives_with_torch_distributed` (gloo on CPU at T0, NCCL at T1/T2) and
# `04_busbw_and_the_alpha_beta_fit` (fits $\alpha$ and $\beta$ to nccl-tests output).
#
# ## The one-minute version
# * What each rank holds before and after defines a **collective**. Inference uses these collectives:
#   broadcast, all-reduce, reduce-scatter, all-gather, all-to-all and send/recv.
# * **All-reduce = reduce-scatter + all-gather.** On a ring of $p$ ranks, that is ${2(p-1)}$ steps,
#   and each step moves $S/p$ bytes. Thus $T = 2(p-1) \cdot \alpha + 2(p-1)/p \cdot S/B$. That is
#   bandwidth-optimal, but its latency term increases with $p$.
# * Below a crossover size ($S^{\ast} = p \cdot \alpha \cdot B$ for the ring), a collective is
#   **latency-bound**.
#   Tensor-parallel all-reduces during decode are a few hundred KB. Thus they are latency-bound.
#   During prefill, they are tens of MB and bandwidth-bound. Different algorithms are the best in
#   each regime.
# * **busbw** (nccl-tests) $= \text{algbw} \times 2(p-1)/p$ for all-reduce. It changes "bytes per
#   second of buffer" into "bytes per second per link". You can compare that value with the NVLink
#   or NIC spec at any value of $p$.
# * NCCL matches collectives **by issue order**. If one rank issues a different call, or skips one,
#   all the other ranks hang.
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
# * **broadcast**: each rank ends with the buffer of the root. It sends weights or config from rank 0.
# * **reduce**: the root ends with the elementwise sum. This is rare in inference.
# * **all-reduce**: *all ranks* end with the sum. Tensor parallelism uses it two times per layer.
# * **reduce-scatter**: rank $r$ ends with *chunk $r$* of the sum. That is the first half of an
#   all-reduce, and sequence parallelism uses it.
# * **all-gather**: each rank ends with the concatenation. That is the second half, and it gathers
#   sharded weights or logits.
# * **all-to-all**: chunk $j$ of rank $r$ goes to rank $j$. This is a transpose across ranks. MoE
#   expert parallelism uses it to dispatch and combine tokens.
#
# ## Exercise 3.1: ring reduce-scatter
#
# Write it with the rules of a ring. In each step, **each rank sends exactly one chunk to its right
# neighbour** `(r + 1) % p`. The neighbour **adds** it into its own copy of that chunk. All sends in a
# step occur at the same time. Thus compute each message from the state at the *start* of the step.
# After the last step, rank $r$ must own the fully reduced chunk $r$.
#
# Return the chunks and the **schedule**. The schedule has one list per step, with the
# `(src, dst, chunk)` messages of that step. The check replays your schedule on new buffers. Thus the
# schedule must be correct, not only the final answer. (Hint: at step $s = 0, 1, \dots, p-2$, rank
# $r$ sends chunk `(r - s - 1) % p`.)

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
# Now each rank $r$ starts with only its own finished chunk. In each step, each rank sends one chunk
# forward to its right neighbour, which *stores* it. After ${p-1}$ steps, each rank has all chunks.
#
# Write `ring_all_gather(chunks)`. It returns the full buffers and its schedule. Then write
# `ring_all_reduce(bufs)` from the two functions. It returns the full buffers and the combined
# schedule (the reduce-scatter steps first).

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
# `gpusim.collectives` runs the same schedules (and a few more) and records each message. The next
# cell shows the ring and three other ways to all-reduce the same 4 ranks:

# %%
bufs = [rng.integers(0, 9, 8) for _ in range(4)]
for algo in ("ring", "tree", "two_shot", "switch"):
    r = C.all_reduce(bufs, algo, chunks=2)
    assert all(np.array_equal(o, np.sum(bufs, axis=0)) for o in r.out)
    print(r.trace.table(), "\n   bytes sent per rank:", r.trace.sent_bytes(), "\n")

# %% [markdown]
# * **tree** (binomial): it divides the number of active ranks by two in each step. Thus it needs
#   $2\lceil \log_2 p \rceil$ steps, and not ${2(p-1)}$. But each message is the *full* buffer. Thus
#   the tree has a low latency cost and a high bandwidth cost. NCCL's real tree is a *pipelined double
#   binary tree*. It keeps the $\log p$ steps and gets back most of the bandwidth.
# * **two_shot**: a direct reduce-scatter, then a direct all-gather. That is 2 steps with the same
#   bytes as the ring. But it needs each rank to reach all other ranks at the same time (NVSwitch).
#   The custom all-reduce kernels in vLLM and TensorRT-LLM do this for small and mid-size messages.
# * **switch**: NVLS / SHARP in-network reduction. Each GPU sends its data *one time* to the switch.
#   The switch sums the data and multicasts the result back. Thus each GPU sends $S$, and not
#   $2(p-1)/p \cdot S$.
#
# ## The alpha-beta model
# Each step costs $\alpha$ (launch, synchronisation, hop latency), plus the bytes of its busiest port
# divided by $B$. The trace time is exactly equal to the closed form:

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
# Write `ring_allreduce_time(S, p, alpha, bw)` from the step structure that you wrote. Also write
# `crossover_size(p, alpha, bw)`: the $S$ at which the latency term is equal to the bandwidth term.
# Then calculate it for 8 GPUs. Use the illustrative NVLink 4 numbers of layer 01: $\alpha$ = 2 us
# per step and $B$ = 450 GB/s per direction.

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
# nccl-tests reports two bandwidths. $\text{algbw} = S/t$ is the buffer size divided by the time.
# `busbw` multiplies algbw by a factor for each collective:
#
# * $2(p-1)/p$ for all-reduce
# * $(p-1)/p$ for reduce-scatter, all-gather and all-to-all
# * 1 for broadcast and reduce
#
# With this factor, a bandwidth-optimal algorithm shows the per-link bandwidth at any $p$. The next
# cell shows a simulated sweep in the shape of nccl-tests:

# %%
print(C.format_sweep(C.sweep("all_reduce", "ring", p=8, alpha=2e-6, bw=450e9,
                             sizes=[2**e for e in (10, 13, 16, 19, 20, 22, 24, 27, 30)])))

# %% [markdown]
# busbw increases toward $B$ (450 GB/s here) as messages become larger and $\alpha$ becomes
# unimportant. On a real system, you compare that plateau with the fabric spec: NVLink, PCIe or the NIC
# line rate. The size where busbw gets to half of the plateau is approximately the crossover size.
#
# ## Exercise 3.4: compute busbw like nccl-tests
#
# Each tuple is `(op, ranks, size_bytes, time_us)` in the conventions of nccl-tests. These are
# **illustrative numbers in the documented format, not measurements.** Write
# `bandwidths(op, ranks, size, time_us)`. It returns `(algbw_GBps, busbw_GBps)` with GB = 1e9 bytes.

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
# **Tensor parallelism** (Megatron-style) divides the matmuls of each layer across $p$ GPUs. It
# all-reduces the activations two times per layer: after the output projection of attention, and
# after the MLP. The message is `tokens x hidden x 2 bytes`. The next cell uses a 70B-class dense
# model (hidden 8192, 80 layers) at TP=8, with the same illustrative $\alpha$ = 2 us and
# $B$ = 450 GB/s. Layer 01 §5.3 gives the cost of batch 1. Here we compare batch 32 with prefill,
# and the ring with a two-shot all-reduce.

# %%
for phase, tokens in (("decode, batch 32", 32), ("prefill, 8k tokens", 8192)):
    for algo in ("ring", "two_shot"):
        d = C.tp_comm(layers=80, tokens=tokens, hidden=8192, p=8, alpha=2e-6, bw=450e9, algo=algo)
        print(f"{phase:>19} {algo:>8}: {d['message_bytes'] / 2**20:7.2f} MiB x {d['calls']} calls = "
              f"{d['total_s'] * 1e3:7.2f} ms/step, latency share {d['latency_share']:.0%}")

# %% [markdown]
# One formula gives two regimes. In decode, 93% of the time of the ring is $\alpha$. The solution is
# fewer steps (two-shot or one-shot custom all-reduce, NVLS, tree), not more bandwidth. The solution
# also includes CUDA graphs, so that the 160 launches per step cost nothing on the CPU. In prefill,
# the messages are large and bandwidth-bound. Thus the optimal byte count of the ring is important,
# and so is the overlap of the communication with compute.
#
# ## Exercise 3.5: the decode all-reduce you would ship
#
# Use the same model and TP=8, but a decode batch of **64** tokens. With the same assumed $\alpha$
# and $B$, calculate these values:
#
# * `message_bytes`
# * `ring_ms_per_step`: all 160 all-reduces with a ring
# * `best_algo`: the fastest of `C.best_algorithm` for this message
#
# `best_algorithm` pipelines the in-switch algorithm at its best depth for each size,
# `C.pipeline_chunks()`. For a 1 MiB message, that is a single chunk, because more chunks only add
# $\alpha$-steps.

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
# **Expert parallelism** (MoE) moves tokens, not partial sums. Each token goes to the GPUs of its
# top-k experts (all-to-all *dispatch*), and then comes back (all-to-all *combine*). Per GPU and
# direction, that is `tokens x top_k x hidden x bytes`, and $(p-1)/p$ of it crosses the fabric. The
# next cell uses a Mixtral-like layer (hidden 4096, top-2) with 256 tokens per GPU in BF16:

# %%
dispatch = 256 * 2 * 4096 * 2
for algo in ("pairwise", "direct"):
    t = C.model_time("all_to_all", algo, dispatch, 8, 2e-6, 450e9)
    print(f"all-to-all {algo:>8}: {dispatch / 2**20:.0f} MiB per GPU, {t * 1e6:.1f} us per dispatch (x2 with combine)")

# %% [markdown]
# Pipeline parallelism sends one `tokens x hidden` activation per stage boundary per micro-batch
# (point-to-point, not per layer). That is small. This is why the slower scale-out network is
# sufficient for PP, while TP and EP stay in the NVLink domain.
#
# ## Exercise 3.6: find the hang
#
# NCCL has no names for collectives. The $k$-th call on a communicator is the *same* collective on
# all ranks. Write `first_bad_call(calls)`, where `calls[rank]` is the list of `(op, nbytes)` of
# that rank. For the first mismatch, return
# `(index, sorted list of ranks that disagree with the majority)`. If a rank has no call at that
# index, the absent call counts as a disagreement. If all ranks agree, return `None`.

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
# These are the usual causes:
#
# * a data-dependent branch that only some ranks take (for example an "if loss is nan" path)
# * a rank that crashed or ran out of memory (OOM)
# * different shapes per rank
# * ranks that issue collectives on different streams or communicators in different orders
#
# With `NCCL_DEBUG=INFO` and the flight recorder of PyTorch, you get the last collective of each
# rank. That is exactly the input to this function.
#
# Floats teach you one more thing: the ring and the tree add in **different orders**. Thus their
# float32 results are different in the last bits. But all ranks still agree *within* one algorithm.

# %%
xs = [np.random.default_rng(i).normal(size=4096).astype(np.float32) for i in range(8)]
ring_out, tree_out = C.all_reduce(xs, "ring").out, C.all_reduce(xs, "tree").out
print("ranks agree within the ring:", all(np.array_equal(ring_out[0], o) for o in ring_out))
print("ring vs tree max |diff|    :", float(np.max(np.abs(ring_out[0] - tree_out[0]))))

# %% [markdown]
# ## In a design review
#
# **The two-minute version.** "We serve with TP=8 in one NVLink domain. Each layer all-reduces two
# times, with a message of tokens x hidden x 2 bytes. We fit $\alpha$ and $B$ from our own
# nccl-tests run (here the illustrative 2 us and 450 GB/s). With these values, the crossover is
# approximately 7 MB ($S^{\ast} = p \cdot \alpha \cdot B$). Thus decode all-reduces, at 0.5 to 1 MB,
# are latency-bound.
#
# "We use an algorithm with few steps (a one-/two-shot custom all-reduce, or NVLS where the switch
# supports it). We capture it in the decode CUDA graph. Prefill messages are 100+ MB and
# bandwidth-bound. There, NCCL's ring or NVLS is at the link limit. We confirm this when we compare
# busbw from nccl-tests with the NVLink spec.
#
# "EP all-to-alls also stay in the NVLink domain. PP moves one activation per stage. It is the only
# parallelism that we let cross the scale-out network."
#
# **Drill questions**
#
# 1. *Why is all-reduce ${2(p-1)}$ steps and not ${p-1}$?* It is a reduce-scatter, then an all-gather.
#    The reduce-scatter takes ${p-1}$ steps to finish the sum of each chunk. The all-gather takes
#    ${p-1}$ steps to spread the finished chunks. Each rank sends $2(p-1)/p \cdot S$ in total. No algorithm that uses point-to-point
#    sends can send less.
# 2. *Our 8-GPU all-reduce shows busbw of 419 GB/s but algbw of 240 GB/s. Which do we quote?*
#    busbw. It is equal to $\text{algbw} \times 2(p-1)/p$, and it is what the links actually carried
#    per GPU. algbw changes with $p$.
# 3. *The job hangs at step 1,000 on one rank's data-dependent branch. Why does NCCL not give an
#    error?* NCCL matches collectives only by issue order. The other ranks wait in a call that no
#    rank will match. They wait until a watchdog timeout occurs. The solution is to make control
#    flow rank-uniform, and to set timeouts so that a hang becomes an error.
