"""The torch backend: the same benchmarks on a CUDA GPU (tiers T1 and T2).

PyTorch is imported when a ``TorchBackend`` is created, never at module import, so the lab
installs and runs without it. What changes on a GPU, and how this module handles it:

* **Work is asynchronous.** A kernel launch returns in microseconds; the GPU finishes later.
  On-device work is timed with CUDA events recorded on the stream around ``inner`` calls;
  transfers and GPU-to-GPU copies (which involve the host or two devices) use the wall
  clock with every device synchronised before and after.
* **Precision is a mode, not just a dtype.** fp32 GEMMs use TF32 tensor cores only when
  ``torch.set_float32_matmul_precision("high")``; the lab measures fp32 ("highest") and tf32
  ("high") separately and restores your setting afterwards.
* **FP8 is guarded.** It needs compute capability 8.9+ (Ada/Hopper/Blackwell) and a PyTorch
  with ``torch._scaled_mm`` (a private API whose signature has moved between releases).
* **Pinned vs pageable host memory.** A pageable buffer must be staged through a pinned
  bounce buffer by the driver; a pinned one is DMA'd directly, and only a pinned copy can
  overlap with compute (``non_blocking=True``).
* **P2P copies.** ``b.copy_(a)`` between devices runs on the *source* device's current stream
  and fences both devices' current streams. A bidirectional test therefore puts each direction
  on its own side stream, or the two copies would serialise and you would measure one.
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager

from ..accounting import canonical_dtype, elementwise_cost, gemm_cost, stream_cost, transfer_cost
from ..measure import Op
from ..timing import Timing, bench
from . import BackendUnavailable


class TorchBackend:
    name = "torch"
    is_gpu = True
    stream_dtype = "float32"

    def __init__(self, device_index: int = 0, seed: int = 0):
        try:
            import torch
        except ImportError as e:
            raise BackendUnavailable("PyTorch is not installed (Colab/Kaggle have it; else pip install -e '.[gpu]')") from e
        if not torch.cuda.is_available():
            raise BackendUnavailable("PyTorch is installed but no CUDA GPU is visible (torch.cuda.is_available() is False)")
        if device_index >= torch.cuda.device_count():
            raise BackendUnavailable(f"cuda:{device_index} requested but only {torch.cuda.device_count()} GPU(s) visible")
        self.torch = torch
        self.index = device_index
        self.dev = torch.device("cuda", device_index)
        self.device = f"cuda:{device_index}"
        self.seed = seed

    # -- facts about the device -------------------------------------------------------------------
    def describe(self) -> dict:
        t = self.torch
        p = t.cuda.get_device_properties(self.index)
        return {"backend": "torch", "torch": t.__version__, "cuda_runtime": t.version.cuda,
                "hip_runtime": getattr(t.version, "hip", None), "device": self.device, "name": p.name,
                "compute_capability": f"{p.major}.{p.minor}", "sm_count": p.multi_processor_count,
                "memory_bytes": p.total_memory, "l2_cache_bytes": getattr(p, "L2_cache_size", None),
                "device_count": t.cuda.device_count()}

    def capability(self) -> tuple:
        return tuple(self.torch.cuda.get_device_capability(self.index))

    def fp8_support(self) -> tuple:
        t = self.torch
        if not (hasattr(t, "float8_e4m3fn") and hasattr(t, "_scaled_mm")):
            return False, "this PyTorch has no float8_e4m3fn / _scaled_mm"
        if self.capability() < (8, 9):
            return False, f"compute capability {self.capability()} < (8, 9): no FP8 tensor cores"
        return True, ""

    def gemm_dtypes(self) -> list:
        dts = ["float32", "tf32", "float16", "bfloat16"]
        if self.capability() < (8, 0):
            dts.remove("tf32")                 # TF32 arrived with Ampere (sm_80)
        if self.fp8_support()[0]:
            dts.append("float8_e4m3fn")
        return dts

    # -- timing ------------------------------------------------------------------------------------
    def sync(self) -> None:
        for i in range(self.torch.cuda.device_count()):
            self.torch.cuda.synchronize(i)

    def time(self, fn, *, method: str = "auto", repeats: int = 5, min_time: float = 0.02, warmup: int = 1,
             setup=None, max_inner: int = 1 << 14) -> Timing:
        """CUDA events around ``inner`` back-to-back calls (``method="auto"``), or the wall clock
        with all devices synchronised (``method="wall"``, for transfers and multi-device work)."""
        if method == "wall" or setup is not None:
            return bench(fn, sync=self.sync, repeats=repeats, min_time=min_time, warmup=warmup,
                         setup=setup, max_inner=max_inner)
        t = self.torch
        with t.cuda.device(self.index):
            for _ in range(max(1, warmup)):
                fn()
            t.cuda.synchronize()
            start, end = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)

            def run(inner: int) -> float:
                start.record()
                for _ in range(inner):
                    fn()
                end.record()
                end.synchronize()
                return start.elapsed_time(end) / 1e3        # elapsed_time is in milliseconds

            inner = 1
            while inner < max_inner:
                dt = run(inner)
                if dt >= min_time:
                    break
                inner = min(max_inner, max(2 * inner, math.ceil(inner * 1.2 * min_time / max(dt, 1e-7))))
            samples = tuple(run(inner) / inner for _ in range(repeats))
        return Timing(samples=samples, inner=inner, warmup=warmup, method="cuda-events")

    # -- compute ----------------------------------------------------------------------------------
    def _tdtype(self, dtype: str):
        t = self.torch
        return {"float32": t.float32, "tf32": t.float32, "float16": t.float16, "bfloat16": t.bfloat16,
                "float64": t.float64}[dtype]

    @contextmanager
    def _matmul_precision(self, precision: str):
        t = self.torch
        previous = t.get_float32_matmul_precision()
        t.set_float32_matmul_precision(precision)
        try:
            yield
        finally:
            t.set_float32_matmul_precision(previous)

    def make_gemm(self, m: int, n: int, k: int, dtype: str = "bfloat16") -> Op:
        t = self.torch
        dtype = canonical_dtype(dtype)
        g = t.Generator(device=self.dev)
        g.manual_seed(self.seed)
        if dtype == "float8_e4m3fn":
            ok, why = self.fp8_support()
            if not ok:
                raise BackendUnavailable(why)
            a = t.randn(m, k, device=self.dev, dtype=t.bfloat16, generator=g).to(t.float8_e4m3fn)
            b_t = t.randn(n, k, device=self.dev, dtype=t.bfloat16, generator=g).to(t.float8_e4m3fn)
            b = b_t.t()                                   # (k, n) column-major, as _scaled_mm requires
            one = t.ones((), device=self.dev, dtype=t.float32)

            def fn():
                t._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=t.bfloat16)

            return Op(fn, gemm_cost(m, n, k, 1, out_bytes=2),
                      note="FP8 e4m3 inputs, bf16 output, per-tensor scales (torch._scaled_mm)")
        tdt = self._tdtype(dtype)
        a = t.randn(m, k, device=self.dev, dtype=tdt, generator=g)
        b = t.randn(k, n, device=self.dev, dtype=tdt, generator=g)
        c = t.empty(m, n, device=self.dev, dtype=tdt)

        def fn():
            t.matmul(a, b, out=c)

        precision = "high" if dtype == "tf32" else "highest"
        note = {"tf32": "fp32 storage, TF32 tensor-core math", "float32": "IEEE fp32 (TF32 off)"}.get(dtype, "")
        if dtype == "bfloat16" and self.capability() < (8, 0):
            note = "no native BF16 before sm_80: expect an emulated, slow path"
        return Op(fn, gemm_cost(m, n, k, a.element_size()), note=note,
                  context=lambda: self._matmul_precision(precision))

    # -- memory -----------------------------------------------------------------------------------
    def make_stream(self, kernel: str, n: int, dtype: str = "float32", threads: int = 1) -> Op:
        """STREAM on the GPU: each kernel is one CUDA kernel, so its traffic is exactly STREAM's."""
        t = self.torch
        tdt = self._tdtype(canonical_dtype(dtype))
        a = t.full((n,), 1.0, device=self.dev, dtype=tdt)
        b = t.full((n,), 2.0, device=self.dev, dtype=tdt)
        c = t.full((n,), 0.5, device=self.dev, dtype=tdt)
        q = 3.0
        fns = {"copy": lambda: c.copy_(a),
               "scale": lambda: t.mul(c, q, out=b),
               "add": lambda: t.add(a, b, out=c),
               "triad": lambda: t.add(b, c, alpha=q, out=a)}      # a = b + q·c in one kernel
        if kernel not in fns:
            raise ValueError(f"unknown STREAM kernel {kernel!r}")
        target, value = {"copy": (c, 1.0), "scale": (b, 1.5), "add": (c, 3.0), "triad": (a, 3.5)}[kernel]
        cost = stream_cost(kernel, n, a.element_size())
        return Op(fns[kernel], cost, extras={"stream_bytes": cost.bytes, "passes": 1},
                  verify=lambda: bool((target == value).all().item()))

    def make_scale_inplace(self, nbytes: int, dtype: str = "float32") -> Op:
        t = self.torch
        tdt = self._tdtype(canonical_dtype(dtype))
        x = t.ones(max(1, nbytes // t.tensor([], dtype=tdt).element_size()), device=self.dev, dtype=tdt)
        return Op(lambda: x.mul_(1.0), elementwise_cost(x.numel(), x.element_size(), 1, 1, 1),
                  extras={"working_set_bytes": x.numel() * x.element_size()})

    # -- transfers ---------------------------------------------------------------------------------
    def make_transfer(self, nbytes: int, direction: str = "h2d", pinned: bool = True) -> Op:
        """Host↔device copy of ``nbytes`` from pinned (page-locked) or pageable host memory."""
        t = self.torch
        host = t.empty(nbytes, dtype=t.uint8, pin_memory=pinned)
        host.fill_(1)                                    # touch every page before timing
        dev = t.empty(nbytes, dtype=t.uint8, device=self.dev)
        dev.fill_(2)
        if direction == "h2d":
            def fn():
                dev.copy_(host, non_blocking=pinned)
        elif direction == "d2h":
            def fn():
                host.copy_(dev, non_blocking=pinned)
        else:
            raise ValueError("direction must be 'h2d' or 'd2h'")
        return Op(fn, transfer_cost(nbytes), timer="wall",
                  note=("pinned" if pinned else "pageable") + " host memory")

    def make_p2p(self, src: int, dst: int, nbytes: int, bidirectional: bool = False) -> Op:
        """GPU ``src`` → GPU ``dst`` copy (both ways at once if ``bidirectional``)."""
        t = self.torch
        a = t.empty(nbytes, dtype=t.uint8, device=f"cuda:{src}")
        a.fill_(1)
        b = t.empty(nbytes, dtype=t.uint8, device=f"cuda:{dst}")
        b.fill_(0)
        peer = bool(t.cuda.can_device_access_peer(src, dst))
        if not bidirectional:
            def fn():
                b.copy_(a, non_blocking=True)       # runs on src's current stream
        else:
            a2 = t.empty(nbytes, dtype=t.uint8, device=f"cuda:{dst}")
            a2.fill_(1)
            b2 = t.empty(nbytes, dtype=t.uint8, device=f"cuda:{src}")
            b2.fill_(0)
            s_src, s_dst = t.cuda.Stream(device=src), t.cuda.Stream(device=dst)

            def fn():
                with t.cuda.stream(s_src):           # src→dst on a side stream of src
                    b.copy_(a, non_blocking=True)
                with t.cuda.stream(s_dst):           # dst→src on a side stream of dst
                    b2.copy_(a2, non_blocking=True)
        note = "direct peer access" if peer else "no peer access: the driver stages through host memory"
        return Op(fn, transfer_cost(nbytes, directions=2 if bidirectional else 1), timer="wall", note=note,
                  extras={"peer_access": peer})

    def make_file_to_device(self, path: str, chunk_bytes: int = 64 << 20, buffers: int = 2,
                            setup=None) -> Op:
        """Disk → pinned host buffer → GPU, double-buffered: while chunk *i* is DMA'd to the GPU,
        chunk *i+1* is read from disk into the other pinned buffer. Throughput approaches the
        slower of the two tiers instead of their sum (primer §6)."""
        t = self.torch
        size = os.path.getsize(path)
        dev = t.empty(size, dtype=t.uint8, device=self.dev)
        pins = [t.empty(chunk_bytes, dtype=t.uint8, pin_memory=True) for _ in range(buffers)]
        views = [memoryview(p.numpy()) for p in pins]     # numpy views share the pinned memory
        stream = t.cuda.Stream(device=self.dev)

        def fn():
            done = [None] * buffers
            with open(path, "rb", buffering=0) as f:
                off, i = 0, 0
                while off < size:
                    slot = i % buffers
                    if done[slot] is not None:
                        done[slot].synchronize()        # the DMA that last used this buffer is finished
                    want = min(chunk_bytes, size - off)
                    got = 0
                    while got < want:
                        r = f.readinto(views[slot][got:want])
                        if not r:
                            raise IOError(f"short read at offset {off + got} of {path}")
                        got += r
                    with t.cuda.stream(stream):
                        dev[off:off + want].copy_(pins[slot][:want], non_blocking=True)
                        ev = t.cuda.Event()
                        ev.record(stream)
                    done[slot] = ev
                    off += want
                    i += 1
            stream.synchronize()

        return Op(fn, transfer_cost(size), timer="wall", setup=setup, max_inner=1 if setup else None,
                  note=f"disk → pinned ({buffers}×{chunk_bytes >> 20} MiB) → {self.device}",
                  extras={"chunk_bytes": chunk_bytes, "buffers": buffers})
