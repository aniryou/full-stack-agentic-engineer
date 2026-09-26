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


def expected(op: str, xs, root: int = 0):
    """Dispatch by nccl-tests op name (see ``gpurt.dist.busbw.OPS``)."""
    table = {"all_reduce": all_reduce, "all_gather": all_gather, "reduce_scatter": reduce_scatter,
             "alltoall": alltoall, "sendrecv": sendrecv,
             "broadcast": lambda v: broadcast(v, root), "reduce": lambda v: reduce(v, root)}
    return table[op](xs)
