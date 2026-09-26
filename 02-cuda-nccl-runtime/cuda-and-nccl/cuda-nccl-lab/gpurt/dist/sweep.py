"""One benchmark loop for every backend — the part of nccl-tests that is not NCCL.

The one idea: a collective benchmark is six steps, and only the fourth depends on the transport:

  1. size the per-rank buffers the way nccl-tests does (``busbw.plan``);
  2. run the collective once on known inputs and count wrong elements (the ``#wrong`` column);
  3. warm up (connections, allocator, JIT, NCCL's lazy communicator set-up);
  4. time ``iters`` back-to-back operations between a barrier and a device synchronise;
  5. average that time across ranks (nccl-tests' default ``-a 1``);
  6. convert to algbw and busbw.

The transport sits behind a tiny ``Comm`` interface implemented by ``pipes.PipesComm`` (a ring over OS
pipes, T0) and ``bench.TorchComm`` (torch.distributed with gloo on CPU or NCCL on GPUs). All ranks run
the same loop; rank 0's rows are the result.

The time column corresponds to nccl-tests' out-of-place column for all_gather, reduce_scatter,
alltoall and sendrecv, and to its in-place column for all_reduce, broadcast and reduce (those are
in-place in the torch.distributed API).
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from . import busbw as bw
from . import semantics

REDOP = {"all_reduce": "sum", "reduce": "sum", "reduce_scatter": "sum"}
ROOTED = ("broadcast", "reduce")
IN_PLACE = ("all_reduce", "broadcast", "reduce")


class Comm(Protocol):
    rank: int
    world_size: int
    backend: str

    def empty(self, count: int): ...
    def fill(self, buf, values: np.ndarray) -> None: ...
    def to_numpy(self, buf) -> np.ndarray: ...
    def run(self, op: str, send, recv, root: int = 0) -> None: ...
    def barrier(self) -> None: ...
    def synchronize(self) -> None: ...
    def allreduce_scalar(self, x: float, how: str = "avg") -> float: ...


@dataclass
class Row:
    op: str
    nranks: int
    size: int  # bytes, nccl-tests' "size"
    count: int  # elements, nccl-tests' "count"
    dtype: str
    time_us: float
    algbw: float  # GB/s
    busbw: float  # GB/s
    wrong: int  # -1 when not checked
    backend: str
    root: int = -1  # -1 for collectives without a root, as nccl-tests prints

    def as_dict(self) -> dict:
        return asdict(self)


def default_sizes(min_bytes: int = 8, max_bytes: int = 64 << 20, factor: int = 2) -> list[int]:
    sizes, s = [], min_bytes
    while s <= max_bytes:
        sizes.append(s)
        s *= factor
    return sizes


def rank_input(rank: int, count: int) -> np.ndarray:
    """Deterministic inputs: small integers, so float32 sums are exact and checks can use ==."""
    return ((rank + 1) + np.arange(count) % 7).astype(np.float32)


def _check(comm: Comm, op: str, p: bw.Plan, send, recv, root: int) -> int:
    """Run the op once on known inputs; return the number of wrong elements on this rank."""
    n, r = comm.world_size, comm.rank
    inputs = [rank_input(q, p.send_count) for q in range(n)]
    comm.fill(send, inputs[r])
    if recv is not send:
        comm.fill(recv, np.zeros(p.recv_count, np.float32))
    comm.run(op, send, recv, root)
    comm.synchronize()
    if op == "reduce" and r != root:
        return 0  # only the root's buffer is defined
    want = semantics.expected(op, inputs, root)[r]
    got = comm.to_numpy(recv)
    return int(np.count_nonzero(got != want))


def sweep(comm: Comm, op: str, sizes=None, iters: int = 20, warmup: int = 5, check: bool = True,
          average: str = "avg", root: int = 0) -> list[Row]:
    op = bw.canonical(op)
    n = comm.world_size
    rows = []
    for nbytes in sizes or default_sizes():
        p = bw.plan(op, nbytes, n)
        if p.param_count == 0:
            continue  # too small to give every rank an element
        send = comm.empty(p.send_count)
        recv = send if op in IN_PLACE else comm.empty(p.recv_count)
        wrong = _check(comm, op, p, send, recv, root) if check else -1
        for _ in range(warmup):
            comm.run(op, send, recv, root)
        comm.synchronize()
        comm.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            comm.run(op, send, recv, root)
        comm.synchronize()
        dt = comm.allreduce_scalar((time.perf_counter() - t0) / iters, average)
        if check:
            wrong = int(comm.allreduce_scalar(float(wrong), "sum"))
        rows.append(Row(op, n, p.size, p.param_count, "float", dt * 1e6, bw.algbw(p.size, dt),
                        bw.busbw(p.size, dt, op, n), wrong, comm.backend, root if op in ROOTED else -1))
    return rows


def format_table(rows: list[Row], title: str = "") -> str:
    """nccl-tests-like text (one set of columns), parseable by ``gpurt.nccltests``."""
    r0 = rows[0] if rows else None
    lines = [f"# gpurt.dist sweep (not nccl-tests; measured) {title}".rstrip()]
    if r0:
        lines.append(f"# backend: {r0.backend}  op: {r0.op}  nranks: {r0.nranks}")
    lines += ["#",
              f"#{'size':>11} {'count':>13} {'type':>9} {'redop':>7} {'root':>7} {'time':>8} {'algbw':>7} {'busbw':>7} {'#wrong':>7}",
              f"#{'(B)':>11} {'(elements)':>13} {'':>9} {'':>7} {'':>7} {'(us)':>8} {'(GB/s)':>7} {'(GB/s)':>7} {'':>7}"]
    for r in rows:
        wrong = "N/A" if r.wrong < 0 else str(r.wrong)
        lines.append(f"{r.size:>12} {r.count:>13} {r.dtype:>9} {REDOP.get(r.op, 'none'):>7} {r.root:>7} "
                     f"{r.time_us:>8.2f} {r.algbw:>7.3f} {r.busbw:>7.3f} {wrong:>7}")
    if rows:
        lines.append(f"# Avg bus bandwidth    : {sum(r.busbw for r in rows) / len(rows):.4f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- process launcher
def _entry(target, rank, args, results):
    try:
        payload = target(rank, *args)
        if rank == 0:
            results.put(("ok", payload))
    except BaseException:  # report, never hang the parent
        results.put(("error", rank, traceback.format_exc()))


def spawn(target, nranks: int, args_for_rank, timeout: float = 300.0):
    """Start ``nranks`` fresh processes running ``target(rank, *args_for_rank(rank))`` and return rank 0's
    return value. Uses the 'spawn' start method (safe in notebooks and with CUDA); ``target`` must be
    importable (defined at module level). A failure on any rank raises here with its traceback."""
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    procs = [ctx.Process(target=_entry, args=(target, r, args_for_rank(r), results), daemon=True)
             for r in range(nranks)]
    for p in procs:
        p.start()
    try:
        try:
            status = results.get(timeout=timeout)
        except queue_mod.Empty:
            raise TimeoutError(f"no result from rank 0 within {timeout:.0f}s (a rank may have hung)") from None
    finally:
        for p in procs:
            p.join(timeout=10)
        for p in procs:
            if p.is_alive():
                p.terminate()
    if status[0] == "error":
        raise RuntimeError(f"rank {status[1]} failed:\n{status[2]}")
    return status[1]
