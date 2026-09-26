"""torch.distributed collectives benchmark: gloo on CPU (T0 semantics) or NCCL on GPUs (T1/T2).

The one idea: the *same* sweep that nccl-tests runs, driven through the API your framework actually
uses. ``gloo`` moves CPU tensors over TCP/shared memory — right semantics, CPU speed. ``nccl`` moves
CUDA tensors over NVLink, PCIe, shared host memory or the network — whatever NCCL's topology detection
picked (run with ``NCCL_DEBUG=INFO`` to see which transport and how many channels). busbw is computed
exactly as nccl-tests does (``gpurt.dist.busbw``), so a Kaggle 2xT4 number and an 8xH100 number are
on the same scale.

Two ways to launch:

    python -m gpurt.dist.bench --backend gloo --nranks 4 --op all_reduce      # spawns local ranks
    python -m gpurt.dist.bench --backend pipes --nranks 2                       # no torch needed
    torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl --op all_reduce

Under torchrun the rank/world size come from the environment (and it works across nodes).
torch is imported only inside the functions below.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import numpy as np

from . import busbw as bw
from .sweep import Row, default_sizes, format_table, spawn, sweep


class TorchComm:
    """``sweep.Comm`` over an initialised torch.distributed process group."""

    def __init__(self, backend: str, local_rank: int = 0):
        import torch
        import torch.distributed as dist

        self.torch, self.dist, self.backend = torch, dist, backend
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
        else:
            self.device = torch.device("cpu")

    def empty(self, count: int):
        return self.torch.empty(count, dtype=self.torch.float32, device=self.device)

    def fill(self, buf, values: np.ndarray) -> None:
        buf.copy_(self.torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)))

    def to_numpy(self, buf) -> np.ndarray:
        return buf.detach().cpu().numpy()

    def run(self, op: str, send, recv, root: int = 0) -> None:
        d, n = self.dist, self.world_size
        if op == "all_reduce":
            d.all_reduce(recv)
        elif op == "broadcast":
            d.broadcast(recv, src=root)
        elif op == "reduce":
            d.reduce(recv, dst=root)
        elif op == "all_gather":
            if self.backend == "gloo":  # the list form is supported by every gloo build
                d.all_gather(list(recv.chunk(n)), send)
            else:
                d.all_gather_into_tensor(recv, send)
        elif op == "reduce_scatter":
            try:
                d.reduce_scatter_tensor(recv, send)
            except (RuntimeError, NotImplementedError) as e:
                if self.backend == "nccl":
                    raise
                raise NotImplementedError(f"reduce_scatter is not supported by {self.backend} in this "
                                          f"PyTorch build ({e}); use nccl or the pipes backend") from e
        elif op == "alltoall":
            d.all_to_all_single(recv, send)
        elif op == "sendrecv":
            nxt, prv = (self.rank + 1) % n, (self.rank - 1) % n
            if self.backend == "nccl":  # grouped, so a ring of sends cannot deadlock
                reqs = d.batch_isend_irecv([d.P2POp(d.isend, send, nxt), d.P2POp(d.irecv, recv, prv)])
            else:
                reqs = [d.isend(send, nxt), d.irecv(recv, prv)]
            for req in reqs:
                req.wait()
        else:
            raise ValueError(f"unknown op {op!r}")

    def barrier(self) -> None:
        if self.backend == "nccl":
            self.dist.barrier(device_ids=[self.device.index])
        else:
            self.dist.barrier()

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def allreduce_scalar(self, x: float, how: str = "avg") -> float:
        t = self.torch.tensor([x], dtype=self.torch.float64, device=self.device)
        if how == "rank0":
            self.dist.broadcast(t, src=0)
            return float(t.item())
        op = {"avg": self.dist.ReduceOp.SUM, "sum": self.dist.ReduceOp.SUM,
              "min": self.dist.ReduceOp.MIN, "max": self.dist.ReduceOp.MAX}[how]
        self.dist.all_reduce(t, op=op)
        v = float(t.item())
        return v / self.world_size if how == "avg" else v


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rank_main(rank, nranks, port, backend, job):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(nranks), LOCAL_RANK=str(rank))
    import torch
    import torch.distributed as dist

    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=nranks)
    try:
        comm = TorchComm(backend, local_rank=rank)
        return [row.as_dict() for row in sweep(comm, **job)]
    finally:
        dist.destroy_process_group()


def run(backend: str = "gloo", op: str = "all_reduce", nranks: int = 2, sizes=None, iters: int = 20,
        warmup: int = 5, check: bool = True, average: str = "avg", root: int = 0,
        timeout: float = 600.0) -> list[Row]:
    """Spawn ``nranks`` local processes and sweep one collective. ``backend``: gloo | nccl | pipes."""
    op = bw.canonical(op)
    job = {"op": op, "sizes": list(sizes or default_sizes(8, 64 << 20, 4)), "iters": iters,
           "warmup": warmup, "check": check, "average": average, "root": root}
    if backend == "pipes":
        from . import pipes

        return pipes.run(timeout=timeout, nranks=nranks, **job)
    if backend == "nccl":
        import torch

        if torch.cuda.device_count() < nranks:
            raise RuntimeError(f"nccl needs one GPU per rank: {torch.cuda.device_count()} visible, {nranks} requested")
    port = free_port()
    rows = spawn(_rank_main, nranks, lambda r: (nranks, port, backend, job), timeout=timeout)
    return [Row(**d) for d in rows]


def _torchrun_main(a) -> list[Row] | None:
    """Rank/world from torchrun's environment; returns rows on rank 0, None elsewhere."""
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if a.backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=a.backend)
    try:
        rows = sweep(TorchComm(a.backend, local_rank), a.op, _sizes(a), a.iters, a.warmup,
                     not a.no_check, a.average, a.root)
        return rows if dist.get_rank() == 0 else None
    finally:
        dist.destroy_process_group()


def _parse_bytes(s: str) -> int:
    s = s.strip().upper()
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    return int(float(s[:-1]) * mult[s[-1]]) if s[-1] in mult else int(s)


def _sizes(a) -> list[int]:
    return default_sizes(_parse_bytes(a.min_bytes), _parse_bytes(a.max_bytes), a.factor)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Sweep a collective and report algbw/busbw like nccl-tests.")
    p.add_argument("--backend", default="gloo", choices=["gloo", "nccl", "pipes"])
    p.add_argument("--op", default="all_reduce", help=f"one of {', '.join(bw.OPS)}")
    p.add_argument("--nranks", type=int, default=2, help="local ranks to spawn (ignored under torchrun)")
    p.add_argument("-b", "--min-bytes", default="8")
    p.add_argument("-e", "--max-bytes", default="64M")
    p.add_argument("-f", "--factor", type=int, default=2)
    p.add_argument("-n", "--iters", type=int, default=20)
    p.add_argument("-w", "--warmup", type=int, default=5)
    p.add_argument("--average", default="avg", choices=["avg", "min", "max", "rank0"])
    p.add_argument("--root", type=int, default=0)
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--json", help="write rows as JSON (rank 0)")
    a = p.parse_args(argv)
    a.op = bw.canonical(a.op)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and a.backend != "pipes":
        rows = _torchrun_main(a)
    else:
        rows = run(a.backend, a.op, a.nranks, _sizes(a), a.iters, a.warmup, not a.no_check, a.average, a.root)
    if rows:
        print(format_table(rows))
        if a.json:
            Path(a.json).parent.mkdir(parents=True, exist_ok=True)
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump([r.as_dict() for r in rows], f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
