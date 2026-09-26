"""Algorithm bandwidth vs bus bandwidth, exactly as nccl-tests defines them.

The one idea: ``algbw = S / t`` answers "how long does a collective of size S take?", but it depends
on the number of ranks, so it cannot be compared with a link's speed. nccl-tests therefore also
prints **busbw = algbw x factor(op, n)** — the per-rank traffic an optimal ring-style algorithm
implies, which *can* be compared with the hardware peak whatever n is (NVIDIA/nccl-tests,
``doc/PERFORMANCE.md``):

    all_reduce                        2(n-1)/n    (reduce-scatter, then all-gather)
    reduce_scatter, all_gather,       (n-1)/n
    alltoall
    broadcast, reduce, sendrecv       1

Why 2(n-1)/n for all-reduce: split the buffer into n chunks of S/n. In a ring every rank sends
n-1 chunks during reduce-scatter and n-1 more during all-gather: 2(n-1)·S/n bytes. If each rank's
link moves B bytes/s, t = 2(n-1)/n · S / B, so B = S/t · 2(n-1)/n = algbw · factor.

``S`` is the ``size`` column nccl-tests prints: the larger of the per-rank send and receive buffers.
For all_gather that is the gathered *output* (n x the per-rank input); for reduce_scatter the
*input*; for all_reduce/broadcast/reduce the buffer. The ``count`` column is the per-rank element
count nccl-tests passes to NCCL (``count / n`` of the size for all_gather, reduce_scatter, alltoall,
rounded down to a 16-byte multiple — see :func:`plan`).
Units: nccl-tests reports GB/s with GB = 1e9 bytes, and time in microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

OPS = ("all_reduce", "all_gather", "reduce_scatter", "alltoall", "broadcast", "reduce", "sendrecv")

# nccl-tests binary name -> op
BINARIES = {"all_reduce_perf": "all_reduce", "all_gather_perf": "all_gather",
            "reduce_scatter_perf": "reduce_scatter", "alltoall_perf": "alltoall",
            "broadcast_perf": "broadcast", "reduce_perf": "reduce", "sendrecv_perf": "sendrecv"}

_ALIASES = {"allreduce": "all_reduce", "allgather": "all_gather", "reducescatter": "reduce_scatter",
            "all_to_all": "alltoall", "a2a": "alltoall", "send_recv": "sendrecv"}


def canonical(op: str) -> str:
    op = op.lower().replace("-", "_").removesuffix("_perf")
    op = _ALIASES.get(op, op)
    if op not in OPS:
        raise ValueError(f"unknown collective {op!r}; expected one of {OPS}")
    return op


def bus_factor(op: str, n: int) -> float:
    """busbw / algbw for a collective over n ranks (nccl-tests' definition)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    op = canonical(op)
    if op == "all_reduce":
        return 2 * (n - 1) / n
    if op in ("all_gather", "reduce_scatter", "alltoall"):
        return (n - 1) / n
    return 1.0


def algbw(size_bytes: float, seconds: float) -> float:
    """GB/s (1e9 bytes)."""
    return size_bytes / seconds / 1e9


def busbw(size_bytes: float, seconds: float, op: str, n: int) -> float:
    return algbw(size_bytes, seconds) * bus_factor(op, n)


def bytes_sent_per_rank(op: str, size_bytes: float, n: int) -> float:
    """What one rank puts on its link in an optimal (ring/pairwise) algorithm: factor x S."""
    return bus_factor(op, n) * size_bytes


@dataclass(frozen=True)
class Plan:
    """Per-rank buffer sizes for one sweep point, mirroring nccl-tests' ``*GetCollByteCount``."""
    op: str
    nranks: int
    itemsize: int
    send_count: int  # elements in each rank's send buffer
    recv_count: int  # elements in each rank's receive buffer
    param_count: int  # the count passed to the collective (nccl-tests' "count" column)

    @property
    def size(self) -> int:
        """Bytes, as in nccl-tests' "size" column: max(send, recv) buffer."""
        return max(self.send_count, self.recv_count) * self.itemsize


ALIGN_BYTES = 16  # nccl-tests >= v2.13.13 (checked in v2.18.0 and v2.20.0): per-rank chunks are 16-byte multiples


def per_rank_count(count: int, n: int, itemsize: int = 4) -> int:
    """nccl-tests' ``(count/nranks) & ~(16/eltSize - 1)``: split, then round down to a 16-byte multiple.

    For 1000 floats over 3 ranks: 1000 // 3 = 333, rounded down to a multiple of 4 floats -> 332.
    (v2.11 and older used plain ``count/nranks``; 8-byte types round to 2 elements, and types of
    16 bytes or more need no rounding.)
    """
    per = count // n
    step = max(ALIGN_BYTES // itemsize, 1)
    return per - per % step


def plan(op: str, nbytes: int, n: int, itemsize: int = 4) -> Plan:
    """Turn a requested byte size into per-rank counts the way nccl-tests (v2.20.0) does.

    ``count = nbytes // itemsize``. all_reduce, broadcast, reduce and sendrecv use it as is. The ops
    that split the buffer across ranks (all_gather, reduce_scatter, alltoall) give each rank
    :func:`per_rank_count` elements: a 4000-byte float all_gather over 3 ranks moves 3 x 332 = 996
    elements (size 3984 B), and a point too small to give every rank 16 bytes has ``param_count == 0``
    (the sweeps skip it).
    """
    op = canonical(op)
    count = nbytes // itemsize
    if op in ("all_gather", "reduce_scatter", "alltoall"):
        per = per_rank_count(count, n, itemsize)
        if op == "all_gather":
            return Plan(op, n, itemsize, per, per * n, per)
        if op == "reduce_scatter":
            return Plan(op, n, itemsize, per * n, per, per)
        return Plan(op, n, itemsize, per * n, per * n, per)
    return Plan(op, n, itemsize, count, count, count)  # all_reduce, broadcast, reduce, sendrecv
