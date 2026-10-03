# %% [markdown]
# # 03 · Collectives with torch.distributed: semantics, a real ring, and the NCCL path
#
# **Tier:** T0. With PyTorch (Colab, Kaggle, most laptops), the sweeps use `torch.distributed`
# with the **gloo** backend on CPU. Without PyTorch, they use the **ring over OS pipes** of this lab
# (`gpurt.dist.pipes`), a real multi-process ring all-reduce. **T2**: on two or more GPUs (Kaggle's free
# 2×T4), the same code runs on **NCCL**. Each time measurement in this notebook is a real measurement of the
# backend that ran, and its label names that backend.
#
# ## The one-minute version
#
# * **Which rank has which data at the end** defines a collective: broadcast, reduce, all-reduce,
#   all-gather, reduce-scatter, all-to-all, send/recv.
# * **All-reduce = reduce-scatter + all-gather.** On a ring, each phase takes ${n-1}$ steps, and each link
#   is busy. Each rank sends $2(n-1)/n$ of the buffer. That ratio is the busbw factor.
# * Each parallelism uses a collective:
#   * Tensor parallelism does an all-reduce of the activations (two times per layer).
#   * Data/FSDP parallelism does a reduce-scatter of the gradients and an all-gather of the weights.
#   * Expert parallelism does an all-to-all of the tokens.
#   * Pipeline parallelism sends activations to the next stage.
# * At decode time, tensor-parallel messages are small (kilobytes), thus latency-bound. Prefill messages are
#   megabytes, thus bandwidth-bound. Notebook 04 changes that into numbers.
# * Correctness first: a benchmark does a check of the results (`#wrong`) before it reports speed.
#
# Concepts: [the primer](../../PRIMER.md) §5 *Collectives*.

# %%
import numpy as np

from gpurt import env
from gpurt.dist import busbw as bw
from gpurt.dist import semantics
from gpurt.dist.bench import run
from gpurt.dist.sweep import format_table

TI = env.torch_info()  # imports torch only if it is installed
if TI and TI["cuda_available"] and TI["device_count"] >= 2 and TI["nccl"]:
    BACKEND, NRANKS, MAX_BYTES = "nccl", 2, 64 << 20
elif TI:
    BACKEND, NRANKS, MAX_BYTES = "gloo", 2, 8 << 20
else:
    BACKEND, NRANKS, MAX_BYTES = "pipes", 2, 2 << 20
print(env.describe())
print(f"backend for the sweeps: {BACKEND} with {NRANKS} ranks "
      f"({'T2: NCCL on GPUs' if BACKEND == 'nccl' else 'T0: CPU processes on this machine'})")

# %% [markdown]
# ## What each collective computes
#
# Four ranks each hold a small buffer of integers. `gpurt.dist.semantics` is the reference for the checks
# of the benchmarks.

# %%
xs = [np.arange(4, dtype=np.float32) + 10 * r for r in range(4)]
print("inputs         :", [x.tolist() for x in xs])
print("all_reduce     :", semantics.all_reduce(xs)[0].tolist(), "(on every rank)")
print("reduce_scatter :", [c.tolist() for c in semantics.reduce_scatter(xs)], "(rank r keeps chunk r of the sum)")
print("all_gather     :", semantics.all_gather([x[:1] for x in xs])[0].tolist(), "(every rank gets every chunk)")
print("alltoall       :", [c.tolist() for c in semantics.alltoall(xs)], "(a distributed transpose)")
print("broadcast(r=2) :", semantics.broadcast(xs, root=2)[0].tolist())
print("sendrecv       :", [c.tolist() for c in semantics.sendrecv(xs)], "(ring shift by one)")

# %% [markdown]
# ## Exercise 3.1 — all-reduce is reduce-scatter followed by all-gather
#
# Write the two halves for a list of per-rank arrays (the buffer length is divisible by the number of
# ranks). `my_reduce_scatter(xs)` returns, for rank $r$, chunk $r$ of the element-wise sum.
# `my_all_gather(chunks)` returns, for each rank, the concatenation of all chunks in rank order.

# %% exercise
def my_reduce_scatter(xs: list[np.ndarray]) -> list[np.ndarray]:
    ### BEGIN SOLUTION
    total = np.sum(xs, axis=0)
    return list(np.split(total, len(xs)))
    ### END SOLUTION


def my_all_gather(chunks: list[np.ndarray]) -> list[np.ndarray]:
    ### BEGIN SOLUTION
    full = np.concatenate(chunks)
    return [full.copy() for _ in chunks]
    ### END SOLUTION

# %% check
rng = np.random.default_rng(0)
xs8 = [rng.integers(0, 9, 16).astype(np.float32) for _ in range(8)]
for mine, ref in zip(my_reduce_scatter(xs8), semantics.reduce_scatter(xs8)):
    assert np.array_equal(mine, ref)
for mine, ref in zip(my_all_gather(my_reduce_scatter(xs8)), semantics.all_reduce(xs8)):
    assert np.array_equal(mine, ref)
print("✅ all_gather(reduce_scatter(x)) == all_reduce(x): the decomposition every ring and tree uses")

# %% [markdown]
# ## The ring schedule
#
# Divide the buffer into $n$ chunks. Give the steps numbers from 1, as primer §5.2 does. In reduce-scatter
# step $s = 1, \dots, n-1$, rank $r$ sends chunk $(r-s) \bmod n$ to its right neighbour. It also adds the
# chunk that comes from its left neighbour. After ${n-1}$ steps, rank $r$ has the fully reduced chunk $r$.
#
# The all-gather then moves the finished chunks around the ring for ${n-1}$ more steps. Rank $r$ first sends
# its own finished chunk $r$. Then it sends on the chunk that came in last. `gpurt.dist.pipes` runs this
# schedule (`semantics.ring_chunks`). For 4 ranks, it is the primer's trace, step for step.

# %%
n = 4
print("phase step | rank 0 sends | rank 1 sends | rank 2 sends | rank 3 sends")
for phase in ("rs", "ag"):
    for s in range(1, n):
        sends = [semantics.ring_chunks(r, s, n, phase)[0] for r in range(n)]
        print(f"  {phase}   {s}   |" + "|".join(f"   chunk {c}    " for c in sends))

# %% [markdown]
# ## Exercise 3.2 — write the ring schedule yourself
#
# Write `my_ring_chunks(rank, step, n, phase)`. Do not look at `semantics.ring_chunks`. The function returns
# `(send_chunk, recv_chunk)` for steps $1, \dots, n-1$ of phase `"rs"` (the receiver adds the data that comes
# in) or `"ag"` (the receiver copies it). There are two hints. First, in a step, a rank receives exactly
# what its left neighbour sends in that step. Second, the first all-gather send of a rank must be a chunk
# that the rank has *finished*.
#
# The check runs your schedule on real buffers. `semantics.ring_all_reduce(xs, chunks=my_ring_chunks)`
# refuses a step where the sender and the receiver do not agree. The check then compares the result with
# the all-reduce. It also counts the bytes that each rank sent. This is the busbw factor, and it comes from
# your own schedule.

# %% exercise
def my_ring_chunks(rank: int, step: int, n: int, phase: str) -> tuple[int, int]:
    ### BEGIN SOLUTION
    first = rank if phase == "rs" else rank + 1  # "ag": start from the chunk this rank finished (chunk rank)
    return (first - step) % n, (first - step - 1) % n
    ### END SOLUTION

# %% check
gen = np.random.default_rng(1)
for n_ranks in (2, 3, 4, 8):
    xs_n = [gen.integers(0, 9, 8 * n_ranks).astype(np.float32) for _ in range(n_ranks)]
    out, sent = semantics.ring_all_reduce(xs_n, chunks=my_ring_chunks)
    assert all(np.array_equal(a, b) for a, b in zip(out, semantics.all_reduce(xs_n))), f"wrong sums for n = {n_ranks}"
    ratio = [b / xs_n[0].nbytes for b in sent]
    assert all(abs(x - bw.bus_factor("all_reduce", n_ranks)) < 1e-12 for x in ratio), ratio
    print(f"n = {n_ranks}: correct; every rank sent {ratio[0]:.3f} x the buffer = 2(n-1)/n")
print("✅ your schedule all-reduces exactly, every link busy every step: busbw = algbw x 2(n-1)/n")

# %% [markdown]
# ## A real sweep
#
# `gpurt.dist.bench.run` does the same steps as nccl-tests, on the backend that the notebook selected before:
#
# * It sets the size of the buffers.
# * It does a check of the result one time.
# * It does a warm-up.
# * It measures the time of back-to-back operations between barriers.
# * It calculates the average across ranks.
#
# The pipes and gloo backends move bytes between CPU processes on this machine. Thus expect MB/s to a few
# GB/s, and tens to hundreds of microseconds of latency. That is the cost of the transport, not of the
# algorithm.

# %%
sizes = [8 * 4 ** k for k in range(12) if 8 * 4 ** k <= MAX_BYTES]
ar = run(BACKEND, "all_reduce", NRANKS, sizes=sizes, iters=5, warmup=1)
print(format_table(ar, f"(backend {BACKEND}, measured on this machine)"))
assert all(r.wrong == 0 for r in ar)

# %% [markdown]
# For comparison, the next cell runs a point-to-point ring shift (`sendrecv`, busbw factor 1) over the same
# transport. For a ring all-reduce, the design of busbw makes it read like the bandwidth of one link. Thus,
# at large sizes, we expect it to be of the same order as the send/recv bandwidth.

# %%
sr = run(BACKEND, "sendrecv", NRANKS, sizes=sizes, iters=5, warmup=1)
print(format_table(sr, f"(backend {BACKEND}, measured on this machine)"))

# %% [markdown]
# ## Exercise 3.3 — recompute busbw from size and time
#
# Write `busbw_gbps(size_bytes, time_us, op, n)` from first principles. First calculate
# $\text{algbw} = \text{size}/\text{time}$ (GB = 1e9). Then apply the factor for `op` over `n` ranks. The
# check calculates each row of the two sweeps again.

# %% exercise
def busbw_gbps(size_bytes: int, time_us: float, op: str, n: int) -> float:
    ### BEGIN SOLUTION
    algbw = size_bytes / (time_us * 1e-6) / 1e9
    factor = 2 * (n - 1) / n if op == "all_reduce" else (n - 1) / n if op in ("all_gather", "reduce_scatter", "alltoall") else 1.0
    return algbw * factor
    ### END SOLUTION

# %% check
for row in ar + sr:
    assert abs(busbw_gbps(row.size, row.time_us, row.op, row.nranks) - row.busbw) <= 1e-9 * max(1.0, row.busbw)
big_ar, big_sr = ar[-1], sr[-1]
print(f"✅ at {big_ar.size} B: all_reduce busbw {big_ar.busbw:.3f} GB/s vs sendrecv {big_sr.busbw:.3f} GB/s "
      f"(backend {BACKEND}, measured) — same order: busbw measures the link, whatever the collective")

# %% [markdown]
# ## Which collective each parallelism uses — and how big the messages are
#
# | Parallelism | Collective | Per step | Size |
# |---|---|---|---|
# | tensor (TP) | all-reduce of activations | 2 per layer (after attention, after MLP) | tokens × hidden × bytes |
# | data / FSDP (training) | reduce-scatter grads, all-gather weights | per layer or bucket | parameters |
# | expert (EP, MoE) | all-to-all of tokens | 2 per MoE layer (dispatch, combine) | routed tokens × hidden |
# | pipeline (PP) | send/recv | per micro-batch per stage | micro-batch × hidden |
#
# Decode processes one token per sequence per step. Thus the tensor-parallel all-reduces carry
# *batch × hidden* values: kilobytes. Prefill carries *prompt tokens × hidden*: megabytes.
#
# ## Exercise 3.4 — tensor-parallel all-reduce sizes
#
# Write `tp_allreduce_bytes(tokens, hidden, dtype_bytes=2)` (one all-reduce of the activations). Also write
# `tp_allreduces_per_step(layers)`. Calculate the values for an 8B-class model (hidden 4096, 32 layers, bf16).

# %% exercise
def tp_allreduce_bytes(tokens: int, hidden: int, dtype_bytes: int = 2) -> int:
    ### BEGIN SOLUTION
    return tokens * hidden * dtype_bytes
    ### END SOLUTION


def tp_allreduces_per_step(layers: int) -> int:
    ### BEGIN SOLUTION
    return 2 * layers
    ### END SOLUTION

# %% check
assert tp_allreduce_bytes(8, 4096) == 65_536  # decode, batch 8: 64 KiB
assert tp_allreduce_bytes(4096, 4096) == 32 * 2 ** 20  # prefill of 4096 tokens: 32 MiB
assert tp_allreduces_per_step(32) == 64
print("✅ decode: 64 all-reduces of 64 KiB per step (latency-bound); prefill: 32 MiB each (bandwidth-bound)")

# %% [markdown]
# ## The NCCL path (T2)
#
# On two or more GPUs, the same benchmark runs on NCCL. Run it from a terminal (or from a notebook cell with `!`):
#
# ```bash
# torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl --op all_reduce -e 256M
# NCCL_DEBUG=INFO torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl -b 1M -e 1M -n 5
# ```
#
# In the `NCCL_DEBUG=INFO` output, look for two things: the **transport** of each channel, and the number of
# channels. The transports are:
#
# * `via P2P/IPC`: GPU to GPU over NVLink or PCIe.
# * `via SHM`: through host memory. This is usual when P2P is not available, for example on some
#   virtualised or consumer PCIe setups.
# * `via NET/IB` or `NET/Socket`: across nodes.
#
# `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_ALGO` and `NCCL_PROTO` let you force alternatives and see
# the change in busbw. They are diagnostic tools, not production settings (primer §5).
# `deploy/any-gpu/README.md` has a Kaggle 2×T4 recipe. For comparison, `run_nccl_tests.sh` runs nccl-tests
# itself against the same NCCL library.

# %%
if BACKEND == "nccl":
    print("This run used NCCL — the tables above are GPU collectives (measured).")
else:
    print(f"No 2-GPU NCCL here; the tables above used '{BACKEND}'. Run the commands above on a multi-GPU box.")

# %% [markdown]
# ## In a design review
#
# **Two minutes.** I describe a collective by its result. All-reduce puts the sum on all ranks. All-gather
# puts each shard on all ranks. Reduce-scatter gives each rank one shard of the sum. I build all-reduce from
# the other two.
#
# On a ring, ${n-1}$ reduce-scatter steps and ${n-1}$ all-gather steps keep each link busy. Each rank sends
# $2(n-1)/n$ of the buffer. This is why nccl-tests multiplies algbw by that factor to get busbw: a number
# that you can compare with the link.
#
# Then I show the traffic of each parallelism. Tensor parallelism is two all-reduces per layer of
# batch × hidden values. In decode, this is kilobytes, thus latency is the largest cost. FSDP is
# reduce-scatter and all-gather of parameters. MoE is all-to-all. Before I give any bandwidth, I examine the
# result (`#wrong = 0`) and name the transport that NCCL selected.
#
# **Drill questions**
#
# 1. *Why is the all-reduce busbw factor $2(n-1)/n$ and not 2?* Each rank sends ${n-1}$ chunks of $S/n$ in
#    reduce-scatter and ${n-1}$ more in all-gather: $2(n-1)/n \cdot S$. The factor approaches 2 for large $n$.
# 2. *Tensor parallelism across two nodes over 400 Gb/s NICs, not in one NVLink node: what is the effect on
#    decode?* Two latency-bound all-reduces per layer now cross the network. Each one takes microseconds to
#    tens of microseconds, times 2 × layers per token. Keep TP in the NVLink domain. Use pipeline or data
#    parallelism across nodes.
# 3. *A job hangs in all_reduce. What are the first checks?* Did each rank call the same collectives in the
#    same order, with the same shapes and dtypes? (A skipped call on one rank is the usual cause of a hang.)
#    Did one rank stop? Then use `NCCL_DEBUG=INFO` to see the transport. Also use the collective
#    timeout/watchdog settings of your framework to change a silent hang into an error.
