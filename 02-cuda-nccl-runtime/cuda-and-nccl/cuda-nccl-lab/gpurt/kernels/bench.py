"""Timing kernels on a real GPU (T1): CUDA events, device-resident data, and effective bandwidth.

The one idea: **measure the kernel, not the plumbing.** Three mistakes make naive GPU timings
meaningless, and this module avoids each one:

1. *Implicit transfers* — passing NumPy arrays to a Numba kernel copies them host->device and back
   on every launch; here data lives on the device (``cuda.to_device``) before the clock starts.
2. *Asynchrony* — a launch returns before the kernel finishes; timing uses CUDA events recorded on
   the stream around many back-to-back launches, then synchronises on the end event.
3. *Warm-up* — the first launch JIT-compiles the kernel (and Numba caches per argument types); it
   is run and discarded before timing.

Results are real measurements of the GPU you are on. In simulator mode every function here refuses
to run: simulator time is Python-thread time and says nothing about a GPU.

    python -m gpurt.kernels.bench            # full suite, prints a report
    python -m gpurt.kernels.bench --quick --json results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from numba import cuda

from . import SIMULATOR, blocks_for
from . import elementwise as ew
from . import matmul as mm
from . import reduction as red
from . import softmax as sm
from . import traffic
from . import transpose as tr


def require_gpu() -> None:
    if SIMULATOR:
        raise RuntimeError("gpurt.kernels.bench needs a real GPU: the simulator's timings are Python-thread "
                           "times. Run on a GPU box (Colab/Kaggle T4, any rented GPU) with numba-cuda installed.")
    if not cuda.is_available():
        raise RuntimeError("no CUDA device available to Numba (driver missing? numba-cuda not installed? "
                           "see `python -m gpurt.container`)")


def device_info() -> dict:
    require_gpu()
    dev = cuda.get_current_device()
    name = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
    return {"name": name, "cc": tuple(dev.compute_capability), "sms": int(dev.MULTIPROCESSOR_COUNT)}


def time_launches(launch, repeats: int = 20, warmup: int = 3, trials: int = 5) -> float:
    """Seconds per launch: median over ``trials`` of (event time of ``repeats`` back-to-back launches) / repeats."""
    require_gpu()
    for _ in range(warmup):
        launch()
    cuda.synchronize()
    per_launch = []
    for _ in range(trials):
        start, end = cuda.event(timing=True), cuda.event(timing=True)
        start.record()
        for _ in range(repeats):
            launch()
        end.record()
        end.synchronize()
        per_launch.append(cuda.event_elapsed_time(start, end) / 1e3 / repeats)
    return float(np.median(per_launch))


def _dev(shape, fill=None):
    if fill is None:
        return cuda.device_array(shape, dtype=np.float32)
    return cuda.to_device(np.full(shape, fill, dtype=np.float32))


def bandwidth_sweep(kind: str = "vec_add", sizes=None, threads: int = 256) -> list[dict]:
    """Effective bandwidth of a streaming kernel over vectors of 2^10 to 2^26 floats (4 KiB to 256 MiB
    each), i.e. 8 KiB to 512 MiB of traffic for copy (8 B/element) and 12 KiB to 768 MiB for vec_add
    (12 B/element): small sizes are latency (launch)-bound, large ones bandwidth-bound — fit the α-β
    model to see both."""
    require_gpu()
    sizes = sizes or [2 ** k for k in range(10, 27, 2)]
    sms = device_info()["sms"]
    rows = []
    for n in sizes:
        a, b, out = _dev(n, 1.0), _dev(n, 2.0), _dev(n)
        if kind == "vec_add":
            launch = lambda: ew.vec_add[blocks_for(n, threads), threads](a, b, out)  # noqa: E731
        elif kind == "copy":
            blocks = min(blocks_for(n, threads), 32 * sms)  # a few waves; the grid-stride loop does the rest
            launch = lambda: ew.copy[blocks, threads](a, out)  # noqa: E731
        else:
            raise ValueError(f"unknown kind {kind!r}")
        seconds = time_launches(launch)
        nbytes = traffic.elementwise_bytes(kind, n)
        rows.append({"kind": kind, "n": n, "bytes": nbytes, "seconds": seconds,
                     "gbps": traffic.effective_gbps(nbytes, seconds)})
    return rows


def fit_sweep(rows: list[dict]):
    """α-β fit of a bandwidth sweep: α ~ launch + latency, B ~ achievable DRAM bandwidth."""
    from gpurt.dist.alphabeta import fit

    return fit([r["bytes"] for r in rows], [r["seconds"] for r in rows])


def transpose_bench(n: int = 4096) -> dict:
    """copy2d vs naive vs tiled (pad=1) vs tiled without padding (bank conflicts), in GB/s."""
    require_gpu()
    grid, block = tr.launch_config(n, n)
    src, dst = _dev((n, n), 1.0), _dev((n, n))
    nbytes = traffic.transpose_bytes(n, n)
    kernels = {"copy2d": tr.copy2d, "naive": tr.transpose_naive,
               "tiled": tr.make_transpose_tiled(1), "tiled_no_pad": tr.make_transpose_tiled(0)}
    out = {}
    for name, k in kernels.items():
        s = time_launches(lambda k=k: k[grid, block](src, dst))
        out[name] = {"seconds": s, "gbps": traffic.effective_gbps(nbytes, s)}
    return out


def matmul_bench(n: int = 1024, tile: int = 16) -> dict:
    """Naive vs tiled GEMM in GFLOP/s (FP32, CUDA cores). A cuBLAS reference is added when torch+CUDA exist."""
    require_gpu()
    A, B, C = _dev((n, n), 1.0), _dev((n, n), 1.0), _dev((n, n))
    grid, block = mm.launch_config(n, n, tile)
    flops = traffic.matmul_flops(n, n, n)
    out = {}
    for name, k in {"naive": mm.matmul_naive, "tiled": mm.make_matmul_tiled(tile)}.items():
        s = time_launches(lambda k=k: k[grid, block](A, B, C), repeats=5)
        out[name] = {"seconds": s, "gflops": flops / s / 1e9}
    try:
        import torch

        if torch.cuda.is_available():
            ta = torch.ones((n, n), device="cuda", dtype=torch.float32)
            tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False  # compare like with like: FP32 on CUDA cores
            try:
                for _ in range(3):
                    ta @ ta
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    ta @ ta
                torch.cuda.synchronize()
            finally:
                torch.backends.cuda.matmul.allow_tf32 = tf32
            s = (time.perf_counter() - t0) / 10
            out["cublas_fp32 (torch)"] = {"seconds": s, "gflops": flops / s / 1e9}
    except ImportError:
        pass
    return out


def softmax_bench(rows: int = 4096, cols: int = 4096, threads: int = 256) -> dict:
    """Fused (1 launch, 12 B/elem) vs unfused (4 launches, 24 B/elem): time and effective GB/s."""
    require_gpu()
    x = cuda.to_device(np.random.default_rng(0).standard_normal((rows, cols)).astype(np.float32))
    out = _dev((rows, cols))
    fused = sm.make_softmax_fused(threads)
    res = {}
    s = time_launches(lambda: fused[rows, threads](x, out))
    res["fused"] = {"seconds": s, "gbps": traffic.effective_gbps(traffic.softmax_bytes(rows, cols, "fused"), s)}
    bufs = sm.unfused_buffers(rows, cols)
    s = time_launches(lambda: sm.run_softmax_unfused(x, threads, bufs))
    res["unfused"] = {"seconds": s, "gbps": traffic.effective_gbps(traffic.softmax_bytes(rows, cols, "unfused"), s)}
    return res


def reduction_bench(n: int = 2 ** 24, threads: int = 256, runs: int = 5) -> dict:
    """Two-pass vs atomic sum: time, and whether repeated atomic runs agree bit for bit."""
    require_gpu()
    x = np.random.default_rng(0).random(n, dtype=np.float32)
    d_x = cuda.to_device(x)
    block_sum, block_sum_atomic = red.make_block_sum(threads, False), red.make_block_sum(threads, True)
    blocks = red.blocks_for_sum(n, threads)
    partial, acc = _dev(blocks), cuda.to_device(np.zeros(1, np.float32))
    t_first = time_launches(lambda: block_sum[blocks, threads](d_x, partial))
    t_atomic = time_launches(lambda: block_sum_atomic[blocks, threads](d_x, acc))
    atomic_values = {red.run_sum(x, threads, atomic=True) for _ in range(runs)}
    two_pass_values = {red.run_sum(x, threads) for _ in range(runs)}
    return {"first_pass_seconds": t_first, "atomic_seconds": t_atomic,
            "gbps_first_pass": traffic.effective_gbps(traffic.reduction_bytes(n), t_first),
            "distinct_atomic_results": sorted(atomic_values), "distinct_two_pass_results": sorted(two_pass_values),
            "float64_reference": float(x.astype(np.float64).sum())}


@cuda.jit
def _empty():
    pass


def launch_overhead(n: int = 2000) -> float:
    """Microseconds per back-to-back launch of an empty kernel, host clock (Python + Numba + driver)."""
    require_gpu()
    _empty[1, 1]()
    cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        _empty[1, 1]()
    cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def run_all(quick: bool = False) -> dict:
    require_gpu()
    sizes = [2 ** k for k in range(10, 25, 2)] if quick else None
    sweep = bandwidth_sweep("vec_add", sizes) + bandwidth_sweep("copy", sizes)
    ab = fit_sweep([r for r in sweep if r["kind"] == "copy"])
    n = 2048 if quick else 4096
    return {
        "device": device_info(),
        "sweep": sweep,
        "copy_alpha_us": ab.alpha_s * 1e6,
        "copy_bandwidth_gbps": ab.bw_Bps / 1e9,
        "transpose": transpose_bench(n),
        "matmul": matmul_bench(512 if quick else 1024),
        "softmax": softmax_bench(n, n),
        "reduction": reduction_bench(2 ** 22 if quick else 2 ** 24),
        "launch_overhead_us": launch_overhead(500 if quick else 2000),
    }


def format_report(r: dict) -> str:
    d = r["device"]
    lines = [f"# gpurt kernel benchmark — {d['name']} (cc {d['cc'][0]}.{d['cc'][1]}, {d['sms']} SMs) — measured",
             f"copy α-β fit: α = {r['copy_alpha_us']:.1f} µs, B = {r['copy_bandwidth_gbps']:.0f} GB/s",
             "", f"{'kind':>8} {'n':>10} {'bytes':>12} {'µs':>10} {'GB/s':>8}"]
    for row in r["sweep"]:
        lines.append(f"{row['kind']:>8} {row['n']:>10} {row['bytes']:>12} {row['seconds'] * 1e6:>10.1f} {row['gbps']:>8.1f}")
    lines.append("")
    for group, unit in (("transpose", "gbps"), ("matmul", "gflops"), ("softmax", "gbps")):
        for name, v in r[group].items():
            lines.append(f"{group:>9} {name:>20}: {v['seconds'] * 1e3:8.3f} ms  {v[unit]:10.1f} {unit.replace('gbps', 'GB/s').replace('gflops', 'GFLOP/s')}")
    red_r = r["reduction"]
    lines += ["", f"reduction first pass {red_r['gbps_first_pass']:.0f} GB/s; atomic runs gave "
              f"{len(red_r['distinct_atomic_results'])} distinct result(s), two-pass {len(red_r['distinct_two_pass_results'])}",
              f"empty-kernel launch overhead: {r['launch_overhead_us']:.1f} µs"]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--quick", action="store_true", help="smaller sizes (~20 s)")
    p.add_argument("--json", help="write raw results to this file")
    a = p.parse_args(argv)
    if a.json:  # fail before a minutes-long benchmark, not after it
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    r = run_all(quick=a.quick)
    print(format_report(r))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, default=list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
