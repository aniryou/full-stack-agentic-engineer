"""A real ring over OS pipes: what NCCL's ring all-reduce does, runnable on any laptop (T0).

The one idea: **ring all-reduce = reduce-scatter + all-gather.** n processes sit on a ring and the
buffer is cut into n chunks. In each of n-1 reduce-scatter steps every rank sends one chunk to its
right neighbour while receiving one from its left and adding it in; afterwards rank r owns the fully
reduced chunk (r+1) mod n. n-1 all-gather steps then circulate the finished chunks. Every link is
busy in every step, and each rank sends 2(n-1)/n x S bytes in total — the busbw factor, observed.

    step (n = 3)      rank 0 sends   rank 1 sends   rank 2 sends
    RS 0              chunk 0        chunk 1        chunk 2      (each receiver adds)
    RS 1              chunk 2        chunk 0        chunk 1      -> rank r owns chunk r+1
    AG 0              chunk 1        chunk 2        chunk 0      (each receiver copies)
    AG 1              chunk 0        chunk 1        chunk 2      -> everyone has everything

The "links" are ``multiprocessing`` pipes (kernel-mediated memory copies) and each rank has a sender
thread, so a step's send and receive overlap (full duplex) instead of deadlocking or serialising.
Broadcast is a pipelined chain, sendrecv a ring shift; alltoall and reduce are left to NCCL/gloo.
This is a toy transport — GB/s and tens of microseconds of latency — but it is a *real measurement
of a real distributed algorithm*, reported with ``backend="pipes"`` and never as NCCL.
"""

from __future__ import annotations

import queue
import threading

import numpy as np

from . import busbw as bw
from .semantics import ring_chunks
from .sweep import Row, default_sizes, spawn, sweep

PIPELINE_BYTES = 256 * 1024  # broadcast chunk: small enough to pipeline, large enough to amortise α


class PipesComm:
    """Ring neighbours only: ``recv_conn`` from rank r-1, ``send_conn`` to rank r+1."""

    backend = "pipes"

    def __init__(self, rank: int, world_size: int, recv_conn, send_conn, barrier):
        self.rank, self.world_size = rank, world_size
        self._recv, self._send, self._barrier = recv_conn, send_conn, barrier
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._sent = threading.Semaphore(0)
        self._tmp = np.empty(0, np.float32)
        self._thread = threading.Thread(target=self._sender, daemon=True)
        self._thread.start()

    # -- the transport -------------------------------------------------------------------------
    def _sender(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            self._send.send_bytes(item)
            self._sent.release()

    def _isend(self, arr: np.ndarray) -> None:
        """Queue a send; ``arr`` must not change until the matching ``_wait_sends``."""
        self._q.put(memoryview(arr).cast("B"))

    def _wait_sends(self, k: int = 1) -> None:
        for _ in range(k):
            self._sent.acquire()

    def _recv_into(self, arr: np.ndarray) -> None:
        got = self._recv.recv_bytes_into(memoryview(arr).cast("B"))
        if got != arr.nbytes:
            raise RuntimeError(f"rank {self.rank}: expected {arr.nbytes} bytes, got {got}")

    def _scratch(self, count: int, dtype) -> np.ndarray:
        if self._tmp.size < count or self._tmp.dtype != dtype:
            self._tmp = np.empty(max(count, 1), dtype)
        return self._tmp[:count]

    # -- ring algorithms -----------------------------------------------------------------------
    @staticmethod
    def _bounds(count: int, n: int) -> list[int]:
        return [(i * count) // n for i in range(n + 1)]

    def _reduce_scatter_ring(self, buf: np.ndarray, bounds, shift: int) -> int:
        """n-1 steps; returns the chunk this rank now owns fully reduced: (rank + 1 + shift) mod n."""
        n, r = self.world_size, self.rank
        for s in range(n - 1):
            send_c, recv_c = ring_chunks(r, s, n, "rs", shift)
            self._isend(buf[bounds[send_c]:bounds[send_c + 1]])
            tmp = self._scratch(bounds[recv_c + 1] - bounds[recv_c], buf.dtype)
            self._recv_into(tmp)
            buf[bounds[recv_c]:bounds[recv_c + 1]] += tmp
            self._wait_sends()
        return (r + 1 + shift) % n

    def _all_gather_ring(self, buf: np.ndarray, bounds, owned: int) -> None:
        """n-1 steps, starting from the one chunk this rank owns."""
        n = self.world_size
        for s in range(n - 1):
            send_c, recv_c = ring_chunks(owned - 1, s, n, "ag")  # "ag" starts from rank + 1 = owned
            self._isend(buf[bounds[send_c]:bounds[send_c + 1]])
            self._recv_into(buf[bounds[recv_c]:bounds[recv_c + 1]])
            self._wait_sends()

    def _broadcast_chain(self, buf: np.ndarray, root: int) -> None:
        n = self.world_size
        pos = (self.rank - root) % n  # 0 = root, n-1 = end of the chain
        step = max(1, PIPELINE_BYTES // buf.itemsize)
        pieces = [buf[i:i + step] for i in range(0, buf.size, step)]
        if pos == 0:
            for p in pieces:
                self._isend(p)
        else:
            for p in pieces:
                self._recv_into(p)
                if pos != n - 1:
                    self._isend(p)  # forward while the next piece is arriving
        if pos != n - 1:
            self._wait_sends(len(pieces))

    # -- the Comm interface used by sweep() ----------------------------------------------------
    def empty(self, count: int) -> np.ndarray:
        return np.empty(count, np.float32)

    def fill(self, buf: np.ndarray, values: np.ndarray) -> None:
        buf[...] = values

    def to_numpy(self, buf: np.ndarray) -> np.ndarray:
        return buf.copy()

    def run(self, op: str, send: np.ndarray, recv: np.ndarray, root: int = 0) -> None:
        n, r = self.world_size, self.rank
        if op == "all_reduce":  # in place
            bounds = self._bounds(recv.size, n)
            owned = self._reduce_scatter_ring(recv, bounds, shift=0)
            self._all_gather_ring(recv, bounds, owned)
        elif op == "reduce_scatter":
            work = send.copy()  # out of place: leave the input untouched
            bounds = self._bounds(work.size, n)
            owned = self._reduce_scatter_ring(work, bounds, shift=-1)  # rank r ends owning chunk r
            recv[...] = work[bounds[owned]:bounds[owned + 1]]
        elif op == "all_gather":
            bounds = self._bounds(recv.size, n)
            recv[bounds[r]:bounds[r + 1]] = send
            self._all_gather_ring(recv, bounds, owned=r)
        elif op == "broadcast":  # in place
            self._broadcast_chain(recv, root)
        elif op == "sendrecv":
            self._isend(send)
            self._recv_into(recv)
            self._wait_sends()
        else:
            raise NotImplementedError(f"{op} needs all-pairs links; use the gloo or nccl backend")

    def barrier(self) -> None:
        self._barrier.wait()

    def synchronize(self) -> None:
        pass  # CPU: operations complete before run() returns

    def allreduce_scalar(self, x: float, how: str = "avg") -> float:
        vals = np.zeros(self.world_size, np.float64)
        vals[self.rank] = x
        self._all_gather_ring(vals, list(range(self.world_size + 1)), owned=self.rank)
        return {"avg": vals.mean(), "sum": vals.sum(), "min": vals.min(), "max": vals.max(),
                "rank0": vals[0]}[how].item()

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=10)


def _rank_main(rank, n, recv_conn, send_conn, barrier, job):
    comm = PipesComm(rank, n, recv_conn, send_conn, barrier)
    try:
        return [row.as_dict() for row in sweep(comm, **job)]
    finally:
        comm.close()


def run(op: str = "all_reduce", nranks: int = 2, sizes=None, iters: int = 10, warmup: int = 2,
        check: bool = True, average: str = "avg", root: int = 0, timeout: float = 300.0) -> list[Row]:
    """Sweep a collective over ``nranks`` local processes connected in a ring by pipes."""
    if nranks < 2:
        raise ValueError("a ring needs at least two ranks")
    import multiprocessing as mp

    op = bw.canonical(op)
    ctx = mp.get_context("spawn")
    links = [ctx.Pipe(duplex=False) for _ in range(nranks)]  # link r: rank r -> rank r+1, (recv_end, send_end)
    barrier = ctx.Barrier(nranks)
    job = {"op": op, "sizes": list(sizes or default_sizes(8, 16 << 20, 4)), "iters": iters,
           "warmup": warmup, "check": check, "average": average, "root": root}
    rows = spawn(_rank_main, nranks,
                 lambda r: (nranks, links[(r - 1) % nranks][0], links[r][1], barrier, job), timeout=timeout)
    return [Row(**d) for d in rows]
