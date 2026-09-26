"""What each collective computes — reference semantics on a list of per-rank NumPy arrays.

The one idea: a collective is defined by *who ends up with what*, independent of the algorithm or
the transport. ``xs[r]`` is rank r's buffer; every function returns the list of buffers after the
operation. Two identities worth memorising, both checked in the tests:

    all_reduce  == all_gather(reduce_scatter(x))        (how ring all-reduce is built)
    all_gather  == every rank broadcasting its chunk     (why its busbw factor is (n-1)/n)

These are the "expected" values the benchmarks validate against (the ``#wrong`` column), and the
vocabulary of parallelism: tensor parallelism all-reduces activations, data/FSDP parallelism
reduce-scatters gradients and all-gathers weights, expert parallelism all-to-alls tokens, pipeline
parallelism send/recvs activations (primer §5).
"""

from __future__ import annotations

import numpy as np


def _stack(xs):
    xs = [np.asarray(x) for x in xs]
    if len({x.shape for x in xs}) != 1:
        raise ValueError("every rank must hold a buffer of the same shape")
    return xs


def all_reduce(xs, op=np.add):
    xs = _stack(xs)
    total = xs[0].copy()
    for x in xs[1:]:
        total = op(total, x)
    return [total.copy() for _ in xs]


def reduce(xs, root: int = 0, op=np.add):
    out = [np.asarray(x).copy() for x in xs]
    out[root] = all_reduce(xs, op)[0]
    return out


def broadcast(xs, root: int = 0):
    xs = _stack(xs)
    return [xs[root].copy() for _ in xs]


def all_gather(xs):
    """Rank r contributes xs[r]; every rank receives the concatenation in rank order."""
    xs = _stack(xs)
    full = np.concatenate(xs)
    return [full.copy() for _ in xs]


def reduce_scatter(xs, op=np.add):
    """Sum the full buffers, then rank r keeps chunk r (buffer length must divide by n)."""
    xs = _stack(xs)
    n = len(xs)
    if xs[0].shape[0] % n:
        raise ValueError("reduce_scatter needs a buffer length divisible by the number of ranks")
    return [c.copy() for c in np.split(all_reduce(xs, op)[0], n)]


def alltoall(xs):
    """Rank r's chunk j goes to rank j, landing in slot r: a distributed transpose."""
    xs = _stack(xs)
    n = len(xs)
    if xs[0].shape[0] % n:
        raise ValueError("alltoall needs a buffer length divisible by the number of ranks")
    chunks = [np.split(x, n) for x in xs]
    return [np.concatenate([chunks[src][dst] for src in range(n)]) for dst in range(n)]


def sendrecv(xs):
    """Ring shift: every rank sends to r+1 and receives from r-1."""
    xs = _stack(xs)
    n = len(xs)
    return [xs[(r - 1) % n].copy() for r in range(n)]


def ring_chunks(rank: int, step: int, n: int, phase: str) -> tuple[int, int]:
    """(chunk sent to rank+1, chunk received from rank-1) at ``step`` = 1 … n-1 of a ring all-reduce.

    Steps are numbered from 1, as in primer §5.2. ``phase`` "rs" is the reduce-scatter (the receiver
    *adds* the incoming chunk into its own copy): rank r sends chunk (r - step) mod n, so after n-1 steps
    it holds the finished chunk r — the reduce-scatter contract. ``phase`` "ag" is the all-gather (the
    receiver *copies*): rank r starts by sending its finished chunk r and then forwards what it last
    received, so it sends chunk (r - step + 1) mod n. ``gpurt.dist.pipes`` runs exactly this schedule,
    and :func:`ring_all_reduce` executes any schedule in NumPy to check it.
    """
    if not 1 <= step <= n - 1:
        raise ValueError(f"ring steps run 1 … {n - 1} for n = {n}, got {step}")
    first = rank if phase == "rs" else rank + 1
    return (first - step) % n, (first - step - 1) % n


def ring_schedule(n: int) -> list[tuple[str, int, int, int, int]]:
    """Every (phase, step, rank, send_chunk, recv_chunk) of a ring all-reduce over n ranks."""
    return [(phase, s, r, *ring_chunks(r, s, n, phase)) for phase in ("rs", "ag")
            for s in range(1, n) for r in range(n)]


def ring_all_reduce(xs, chunks=ring_chunks):
    """Execute a ring schedule step by step on per-rank NumPy buffers: ``(buffers, bytes_sent_per_rank)``.

    ``chunks(rank, step, n, phase) -> (send_chunk, recv_chunk)`` is the schedule (default: the lab's).
    Every step is synchronous — all ranks send from their buffers as they were at the start of the step
    — and consistency is checked on the way: what rank r expects to receive must be the chunk rank r-1
    sends, or ``ValueError``. A wrong schedule that is consistent still fails the final comparison with
    :func:`all_reduce` — which is how a notebook exercise checks a schedule you write.
    """
    bufs = [np.array(x, copy=True) for x in _stack(xs)]
    n = len(bufs)
    bounds = [(i * bufs[0].shape[0]) // n for i in range(n + 1)]
    sl = [slice(bounds[c], bounds[c + 1]) for c in range(n)]
    sent = [0] * n
    for phase in ("rs", "ag"):
        for step in range(1, n):
            plan = [chunks(r, step, n, phase) for r in range(n)]
            payload = [bufs[r][sl[plan[r][0]]].copy() for r in range(n)]  # sends read pre-step state
            for r in range(n):
                left = (r - 1) % n
                send_c, recv_c = plan[left][0], plan[r][1]
                if send_c != recv_c:
                    raise ValueError(f"{phase} step {step}: rank {r} expects chunk {recv_c}, rank {left} sends {send_c}")
                sent[left] += payload[left].nbytes
                if phase == "rs":
                    bufs[r][sl[recv_c]] += payload[left]
                else:
                    bufs[r][sl[recv_c]] = payload[left]
    return bufs, sent


def expected(op: str, xs, root: int = 0):
    """Dispatch by nccl-tests op name (see ``gpurt.dist.busbw.OPS``)."""
    table = {"all_reduce": all_reduce, "all_gather": all_gather, "reduce_scatter": reduce_scatter,
             "alltoall": alltoall, "sendrecv": sendrecv,
             "broadcast": lambda v: broadcast(v, root), "reduce": lambda v: reduce(v, root)}
    return table[op](xs)
