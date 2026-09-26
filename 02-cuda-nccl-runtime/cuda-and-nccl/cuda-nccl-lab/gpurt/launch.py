"""Launch overhead and CUDA Graphs: why inference engines capture their decode step.

The one idea: issuing a kernel costs the **CPU** a few microseconds (Python, the framework's
dispatcher, the driver). When kernels run longer than that, the CPU races ahead and the GPU never
waits. A decode step at small batch is hundreds of *tiny* kernels, so the GPU finishes each one
before the CPU has issued the next: the step is **launch-bound** — and ``GPU_UTIL`` can still read
100 % (``gpurt.dcgm``). A **CUDA Graph** records the whole sequence once and replays it with a single
launch: the per-kernel CPU cost disappears, leaving only small GPU-side gaps. That is why vLLM, SGLang
and TensorRT-LLM capture decode graphs per batch size, and what ``torch.compile(mode="reduce-overhead")``
does for you (primer §4).

A two-pipeline model (T0) — the slower pipeline sets the pace:

    eager step ~ n * max(k + g, L)      n kernels of k µs, a GPU-side gap g between kernels, L µs of CPU per launch
    graph step ~ G + n * (k + g)        one graph launch G; the GPU-side gaps remain

so graphs help exactly when the step is launch-bound (L > k + g) and change nothing when it is not.

``L``, ``G`` and ``g`` below are *assumptions* of the right order of magnitude, not measurements;
:func:`measure_graph_vs_eager` measures them on a real GPU (T1, torch + CUDA).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LaunchModel:
    launch_us: float = 6.0  # CPU cost per eager launch (assumption — measure yours)
    graph_launch_us: float = 8.0  # one cudaGraphLaunch (assumption)
    node_gap_us: float = 1.0  # GPU-side gap between consecutive kernels (assumption)

    def eager_us(self, n_kernels: int, kernel_us: float) -> float:
        return n_kernels * max(kernel_us + self.node_gap_us, self.launch_us)

    def graph_us(self, n_kernels: int, kernel_us: float) -> float:
        return self.graph_launch_us + n_kernels * (kernel_us + self.node_gap_us)

    def speedup(self, n_kernels: int, kernel_us: float) -> float:
        return self.eager_us(n_kernels, kernel_us) / self.graph_us(n_kernels, kernel_us)

    def launch_bound(self, kernel_us: float) -> bool:
        """True when the GPU would wait for the CPU in eager mode."""
        return kernel_us + self.node_gap_us < self.launch_us

    def gpu_idle_fraction(self, kernel_us: float) -> float:
        """Share of an eager step during which the GPU is not running a kernel."""
        return max(0.0, 1.0 - kernel_us / max(kernel_us + self.node_gap_us, self.launch_us))


def measure_graph_vs_eager(n_kernels: int = 200, numel: int = 1024, iters: int = 50, warmup: int = 5) -> dict:
    """T1: time ``n_kernels`` tiny in-place adds per step, eagerly and replayed from a CUDA Graph.

    Measured with CUDA events on the current stream. Needs torch with CUDA; raises otherwise.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("measure_graph_vs_eager needs torch with a CUDA GPU (T1)")
    x = torch.zeros(numel, device="cuda")

    def step():
        for _ in range(n_kernels):
            x.add_(1.0)

    def timed(fn) -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / iters  # µs per step

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    eager_us = timed(step)

    s = torch.cuda.Stream()  # warm up on a side stream before capture, as the PyTorch docs require
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()  # recorded, not executed
    graph.replay()
    torch.cuda.synchronize()
    graph_us = timed(graph.replay)

    x.zero_()  # check the replay really runs every kernel: one replay adds n_kernels
    graph.replay()
    torch.cuda.synchronize()
    ok = float(x[0].item()) == float(n_kernels)
    return {"n_kernels": n_kernels, "eager_us": eager_us, "graph_us": graph_us,
            "eager_per_kernel_us": eager_us / n_kernels, "graph_per_kernel_us": graph_us / n_kernels,
            "speedup": eager_us / graph_us, "replay_correct": ok, "device": torch.cuda.get_device_name()}
