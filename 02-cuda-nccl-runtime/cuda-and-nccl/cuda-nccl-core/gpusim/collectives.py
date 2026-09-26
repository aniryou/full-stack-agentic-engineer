"""Collectives on simulated ranks. Every collective is a schedule of point-to-point steps, and its
cost is  steps x alpha  +  (bytes through the busiest port) / bandwidth.

A ring all-reduce is a reduce-scatter followed by an all-gather: 2(p-1) steps of S/p bytes, so
T = 2(p-1)*alpha + 2(p-1)/p * S/B. That is bandwidth-optimal, but the latency term grows with p.
Trees, direct (one-/two-shot) algorithms and in-switch reduction (NVLS/SHARP) buy fewer steps or
fewer bytes, and the alpha-beta model says when each one wins.

Every algorithm here really moves numpy data between rank buffers, so you can check its result
against numpy and read its step trace. Port model: each rank has one full-duplex port of
bandwidth B, and the messages in one step run concurrently. Sizes S follow the "size" column of
nccl-tests: the full buffer (all-gather: the gathered output; reduce-scatter: the input;
all-to-all: one rank's send buffer). busbw() applies nccl-tests' correction factors.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

SWITCH = -1   # the NVSwitch (NVLS) / SHARP switch as a pseudo-rank that reduces in the network


@dataclass(frozen=True)
class Send:
    src: int
    dst: int
    lo: int                 # element range [lo, hi) of the source buffer
    hi: int
    op: str = "copy"        # "copy" overwrites the destination range, "add" reduces into it
    at: int | None = None   # where the range lands in dst (default: the same offsets)
    tag: str = ""           # label for printing, e.g. "c2"


@dataclass
class Trace:
    op: str
    algo: str
    p: int
    size: int               # bytes, nccl-tests semantics
    itemsize: int
    steps: list = field(default_factory=list)    # list[list[Send]]
    phases: list = field(default_factory=list)   # phase name per step

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def port_bytes(self, step) -> int:
        """Bytes through the busiest port (egress or ingress of a real rank) in one step."""
        out, inn = Counter(), Counter()
        for s in step:
            n = (s.hi - s.lo) * self.itemsize
            out[s.src] += n
            inn[s.dst] += n
        out.pop(SWITCH, None)
        inn.pop(SWITCH, None)
        return max([*out.values(), *inn.values(), 0])

    def time(self, alpha: float, bw: float) -> float:
        """alpha-beta time: every step costs alpha plus its busiest port's bytes / bandwidth."""
        return sum(alpha + self.port_bytes(st) / bw for st in self.steps)

    def sent_bytes(self) -> dict:
        """Total bytes each real rank sends over the whole collective."""
        c = Counter()
        for st in self.steps:
            for s in st:
                if s.src != SWITCH:
                    c[s.src] += (s.hi - s.lo) * self.itemsize
        return dict(sorted(c.items()))

    def table(self) -> str:
        name = lambda r: "sw" if r == SWITCH else str(r)
        rows = [f"{self.op} / {self.algo}: p={self.p}, S={self.size} B, {self.n_steps} steps"]
        for i, (st, ph) in enumerate(zip(self.steps, self.phases), 1):
            sends = "  ".join(f"{name(s.src)}->{name(s.dst)}:{s.tag}{'+' if s.op == 'add' else ''}"
                              for s in st)
            rows.append(f"  step {i:>2} {ph:<15} {sends}")
        return "\n".join(rows)


@dataclass
class Result:
    out: list          # per-rank output arrays
    trace: Trace


def _bounds(sizes) -> list:
    b = [0]
    for n in sizes:
        b.append(b[-1] + n)
    return b


def _split(n: int, p: int) -> list:
    """np.array_split boundaries: p chunks whose sizes differ by at most one element."""
    q, r = divmod(n, p)
    return _bounds([q + (i < r) for i in range(p)])


def _execute(src: dict, steps, dst: dict | None = None) -> None:
    dst = src if dst is None else dst
    for step in steps:
        payload = [(s, src[s.src][s.lo:s.hi].copy()) for s in step]   # a step is simultaneous
        for s, data in payload:
            at = s.lo if s.at is None else s.at
            view = dst[s.dst][at:at + data.size]
            if s.op == "add":
                view += data
            else:
                view[...] = data


def _prepare(bufs):
    arrs = [np.array(b, copy=True).ravel() for b in bufs]
    if len({a.size for a in arrs}) != 1:
        raise ValueError("every rank must hold a buffer of the same size")
    return {r: a for r, a in enumerate(arrs)}, len(arrs), arrs[0].size, arrs[0].itemsize


def _ring_rs(p, b):     # step s: rank r adds its partial of chunk (r-s-1) into rank r+1
    return [[Send(r, (r + 1) % p, b[c], b[c + 1], "add", tag=f"c{c}")
             for r in range(p) for c in [(r - s - 1) % p]] for s in range(p - 1)]


def _ring_ag(p, b):     # step s: rank r forwards the finished chunk (r-s) to rank r+1
    return [[Send(r, (r + 1) % p, b[c], b[c + 1], "copy", tag=f"c{c}")
             for r in range(p) for c in [(r - s) % p]] for s in range(p - 1)]


def _direct_rs(p, b):   # everyone sends chunk c straight to its owner c
    return [[Send(r, c, b[c], b[c + 1], "add", tag=f"c{c}") for c in range(p) for r in range(p) if r != c]]


def _direct_ag(p, b):   # every owner sends its chunk straight to everyone
    return [[Send(c, r, b[c], b[c + 1], "copy", tag=f"c{c}") for c in range(p) for r in range(p) if r != c]]


def _binomial_reduce(p, n, root=0):
    steps, k = [], 1
    while k < p:
        steps.append([Send((r + root) % p, (r - k + root) % p, 0, n, "add", tag="all")
                      for r in range(k, p, 2 * k)])
        k *= 2
    return steps


def _binomial_bcast(p, n, root=0):
    ks, k = [], 1
    while k < p:
        ks.append(k)
        k *= 2
    return [[Send((r + root) % p, (r + k + root) % p, 0, n, "copy", tag="all")
             for r in range(0, p, 2 * k) if r + k < p] for k in reversed(ks)]


def all_reduce(bufs, algo: str = "ring", chunks: int = 1) -> Result:
    """Every rank ends with the elementwise sum of all buffers.
    ring      reduce-scatter + all-gather around a ring: 2(p-1) steps of S/p
    tree      binomial reduce to rank 0, then binomial broadcast: 2*ceil(log2 p) steps of S
    one_shot  every rank pulls every other rank's whole buffer and sums: 1 step, (p-1)*S per port
    two_shot  direct reduce-scatter, then direct all-gather: 2 steps of (p-1)/p * S per port
    switch    in-network reduction (NVLS/SHARP): send S/k chunks up, the switch sums and multicasts
              them down, pipelined over `chunks`: k+1 steps of S/k"""
    state, p, n, item = _prepare(bufs)
    b = _split(n, p)
    if algo == "ring":
        steps = _ring_rs(p, b) + _ring_ag(p, b)
        phases = ["reduce-scatter"] * (p - 1) + ["all-gather"] * (p - 1)
    elif algo == "tree":
        up, down = _binomial_reduce(p, n), _binomial_bcast(p, n)
        steps, phases = up + down, ["reduce (tree)"] * len(up) + ["broadcast (tree)"] * len(down)
    elif algo == "one_shot":
        steps = [[Send(s, d, 0, n, "add", tag="all") for d in range(p) for s in range(p) if s != d]]
        phases = ["pull+sum"]
    elif algo == "two_shot":
        steps, phases = _direct_rs(p, b) + _direct_ag(p, b), ["reduce-scatter", "all-gather"]
    elif algo == "switch":
        state[SWITCH] = np.zeros_like(state[0])
        c = _split(n, chunks)
        steps, phases = [], []
        for t in range(chunks + 1):
            up = [Send(r, SWITCH, c[t], c[t + 1], "add", tag=f"k{t}") for r in range(p)] if t < chunks else []
            down = [Send(SWITCH, r, c[t - 1], c[t], "copy", tag=f"k{t - 1}") for r in range(p)] if t else []
            steps.append(up + down)
            phases.append("up+down" if up and down else "up (reduce)" if up else "down (multicast)")
    else:
        raise ValueError(f"unknown algo {algo!r}")
    _execute(state, steps)
    trace = Trace("all_reduce", algo, p, n * item, item, steps, phases)
    return Result([state[r] for r in range(p)], trace)


def reduce_scatter(bufs, algo: str = "ring") -> Result:
    """Rank r ends with chunk r (np.array_split order) of the elementwise sum."""
    state, p, n, item = _prepare(bufs)
    b = _split(n, p)
    steps = _ring_rs(p, b) if algo == "ring" else _direct_rs(p, b) if algo == "direct" else None
    if steps is None:
        raise ValueError(f"unknown algo {algo!r}")
    _execute(state, steps)
    trace = Trace("reduce_scatter", algo, p, n * item, item, steps, ["reduce-scatter"] * len(steps))
    return Result([state[r][b[r]:b[r + 1]].copy() for r in range(p)], trace)


def all_gather(pieces, algo: str = "ring") -> Result:
    """Rank r contributes pieces[r]; every rank ends with their concatenation."""
    pieces = [np.asarray(x).ravel() for x in pieces]
    p, b = len(pieces), _bounds([x.size for x in pieces])
    full = np.zeros(b[-1], dtype=np.result_type(*pieces))
    state = {}
    for r, x in enumerate(pieces):
        state[r] = full.copy()
        state[r][b[r]:b[r + 1]] = x
    steps = _ring_ag(p, b) if algo == "ring" else _direct_ag(p, b) if algo == "direct" else None
    if steps is None:
        raise ValueError(f"unknown algo {algo!r}")
    _execute(state, steps)
    trace = Trace("all_gather", algo, p, full.nbytes, full.itemsize, steps, ["all-gather"] * len(steps))
    return Result([state[r] for r in range(p)], trace)


def broadcast(bufs, root: int = 0, algo: str = "ring", chunks: int = 1) -> Result:
    """Every rank ends with the root's buffer.
    ring  a chain root -> root+1 -> ..., pipelined in `chunks` pieces: p+k-2 steps of S/k
    tree  binomial tree: ceil(log2 p) steps of S"""
    state, p, n, item = _prepare(bufs)
    if algo == "tree":
        steps = _binomial_bcast(p, n, root)
    elif algo == "ring":
        c, steps = _split(n, chunks), []
        for t in range(p + chunks - 2 if p > 1 else 0):   # hop i forwards chunk j at step i + j
            steps.append([Send((i + root) % p, (i + 1 + root) % p, c[t - i], c[t - i + 1], tag=f"k{t - i}")
                          for i in range(p - 1) if 0 <= t - i < chunks])
    else:
        raise ValueError(f"unknown algo {algo!r}")
    _execute(state, steps)
    trace = Trace("broadcast", algo, p, n * item, item, steps, ["broadcast"] * len(steps))
    return Result([state[r] for r in range(p)], trace)


def all_to_all(bufs, algo: str = "pairwise") -> Result:
    """Rank r's buffer is p equal chunks, chunk j addressed to rank j; rank r ends with chunk r
    of every rank, in rank order (MoE token dispatch/combine).
    pairwise  step s: rank r sends to r+s: p-1 steps of S/p
    direct    all p-1 sends at once: 1 step of (p-1)/p * S per port"""
    src, p, n, item = _prepare(bufs)
    if n % p:
        raise ValueError("all_to_all needs a buffer size divisible by p")
    q = n // p
    dst = {r: np.zeros_like(src[r]) for r in range(p)}
    for r in range(p):                                    # the chunk a rank keeps is a local copy
        dst[r][r * q:(r + 1) * q] = src[r][r * q:(r + 1) * q]
    sends = lambda s: [Send(r, (r + s) % p, (r + s) % p * q, ((r + s) % p + 1) * q, at=r * q,
                            tag=f"c{(r + s) % p}") for r in range(p)]
    if algo == "pairwise":
        steps = [sends(s) for s in range(1, p)]
    elif algo == "direct":
        steps = [[x for s in range(1, p) for x in sends(s)]]
    else:
        raise ValueError(f"unknown algo {algo!r}")
    _execute(src, steps, dst)
    trace = Trace("all_to_all", algo, p, n * item, item, steps, ["exchange"] * len(steps))
    return Result([dst[r] for r in range(p)], trace)


def reference(op: str, bufs, root: int = 0) -> list:
    """What each rank should hold afterwards, straight from numpy."""
    arrs = [np.asarray(b).ravel() for b in bufs]
    p = len(arrs)
    if op == "all_gather":
        return [np.concatenate(arrs)] * p
    total = np.sum(arrs, axis=0)
    if op == "all_reduce":
        return [total] * p
    if op == "reduce":
        return [total if r == root else arrs[r] for r in range(p)]
    if op == "broadcast":
        return [arrs[root]] * p
    if op == "reduce_scatter":
        return list(np.array_split(total, p))
    if op == "all_to_all":
        return [np.concatenate([np.array_split(a, p)[r] for a in arrs]) for r in range(p)]
    raise ValueError(op)


def cost_terms(op: str, algo: str, p: int, chunks: int = 1) -> tuple:
    """(a, c) such that T = a*alpha + c*S/B for this algorithm (S divisible by p and chunks)."""
    if p < 2:
        return 0, 0.0
    lg, f = math.ceil(math.log2(p)), (p - 1) / p
    k = chunks
    return {
        ("all_reduce", "ring"): (2 * (p - 1), 2 * f),
        ("all_reduce", "tree"): (2 * lg, 2 * lg),
        ("all_reduce", "one_shot"): (1, p - 1),
        ("all_reduce", "two_shot"): (2, 2 * f),
        ("all_reduce", "switch"): (k + 1, (k + 1) / k),
        ("reduce_scatter", "ring"): (p - 1, f), ("reduce_scatter", "direct"): (1, f),
        ("all_gather", "ring"): (p - 1, f), ("all_gather", "direct"): (1, f),
        ("broadcast", "ring"): (p + k - 2, (p + k - 2) / k),
        ("broadcast", "tree"): (lg, lg),
        ("all_to_all", "pairwise"): (p - 1, f), ("all_to_all", "direct"): (1, f),
    }[(op, algo)]


def model_time(op: str, algo: str, size: float, p: int, alpha: float, bw: float, chunks: int = 1) -> float:
    a, c = cost_terms(op, algo, p, chunks)
    return a * alpha + c * size / bw


def crossover_bytes(op: str, algo: str, p: int, alpha: float, bw: float, chunks: int = 1) -> float:
    """The message size where latency and bandwidth terms are equal. Below it the collective is
    latency-bound; above it, bandwidth-bound. Ring all-reduce: S* = p*alpha*B."""
    a, c = cost_terms(op, algo, p, chunks)
    return a * alpha * bw / c


BUSBW_FACTOR = {   # nccl-tests doc/PERFORMANCE.md
    "all_reduce": lambda p: 2 * (p - 1) / p, "reduce_scatter": lambda p: (p - 1) / p,
    "all_gather": lambda p: (p - 1) / p, "all_to_all": lambda p: (p - 1) / p,
    "broadcast": lambda p: 1.0, "reduce": lambda p: 1.0, "send_recv": lambda p: 1.0,
}


def algbw(size: float, seconds: float) -> float:
    return size / seconds


def busbw(op: str, size: float, seconds: float, p: int) -> float:
    """Bus bandwidth as nccl-tests reports it: algbw times a factor that makes a
    bandwidth-optimal algorithm read the per-rank link bandwidth, whatever p is."""
    return algbw(size, seconds) * BUSBW_FACTOR[op](p)


def sweep(op: str = "all_reduce", algo: str = "ring", p: int = 8, alpha: float = 5e-6,
          bw: float = 100e9, sizes=None, chunks: int = 1) -> list:
    """A simulated nccl-tests sweep: time, algbw and busbw from the alpha-beta model."""
    sizes = sizes or [2 ** e for e in range(10, 31, 2)]
    rows = []
    for s in sizes:
        t = model_time(op, algo, s, p, alpha, bw, chunks)
        rows.append({"size": s, "time_us": t * 1e6, "algbw": algbw(s, t) / 1e9, "busbw": busbw(op, s, t, p) / 1e9})
    return rows


def format_sweep(rows) -> str:
    out = ["#  size (B)   time (us)   algbw (GB/s)   busbw (GB/s)   [simulated, alpha-beta model]"]
    out += [f"{r['size']:>11}  {r['time_us']:>10.1f}  {r['algbw']:>13.2f}  {r['busbw']:>13.2f}" for r in rows]
    return "\n".join(out)


def best_algorithm(op: str, size: float, p: int, alpha: float, bw: float, algos=None, chunks: int = 8) -> list:
    """(time, algo) pairs, fastest first: a toy version of what NCCL's tuner decides per call."""
    algos = algos or [a for (o, a) in _ALGOS if o == op]
    return sorted((model_time(op, a, size, p, alpha, bw, chunks), a) for a in algos)


_ALGOS = [("all_reduce", a) for a in ("ring", "tree", "one_shot", "two_shot", "switch")] + \
         [("reduce_scatter", "ring"), ("reduce_scatter", "direct"), ("all_gather", "ring"),
          ("all_gather", "direct"), ("broadcast", "ring"), ("broadcast", "tree"),
          ("all_to_all", "pairwise"), ("all_to_all", "direct")]


def tp_comm(layers: int, tokens: int, hidden: int, p: int, alpha: float, bw: float,
            algo: str = "ring", dtype_bytes: int = 2, per_layer: int = 2, chunks: int = 8) -> dict:
    """Tensor-parallel all-reduce cost of one forward step: `per_layer` all-reduces per layer
    (after attention's output projection and after the MLP), each of tokens x hidden x bytes."""
    size = tokens * hidden * dtype_bytes
    t = model_time("all_reduce", algo, size, p, alpha, bw, chunks)
    a, _ = cost_terms("all_reduce", algo, p, chunks)
    return {"message_bytes": size, "calls": per_layer * layers, "per_call_s": t,
            "total_s": per_layer * layers * t, "latency_share": a * alpha / t}


def first_mismatch(calls: dict):
    """calls[rank] = the (op, nbytes) sequence one rank issued on a communicator. NCCL matches
    collectives purely by issue order, so the first position where ranks disagree is where the job
    hangs (or silently corrupts, if only sizes differ). Returns None when all ranks agree."""
    for i in range(max(len(c) for c in calls.values())):
        seen = {r: (c[i] if i < len(c) else None) for r, c in sorted(calls.items())}
        if len(set(seen.values())) > 1:
            common = Counter(seen.values()).most_common(1)[0][0]
            odd = [r for r, v in seen.items() if v != common]
            return {"call": i, "expected": common, "odd_ranks": odd,
                    "odd_calls": {r: seen[r] for r in odd}}
    return None
