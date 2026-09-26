"""gpurt — the GPU runtime lab: kernels you can run anywhere, collectives you can measure,
and the plumbing that gets a GPU into a container.

Every module opens with a docstring stating the one idea it teaches:

    gpurt.env        where am I running? (GPU, simulator, torch) — decides Numba's mode before import
    gpurt.kernels    CUDA kernels in Numba: the same source runs on the CPU simulator (T0) and a GPU (T1)
    gpurt.dist       collectives: nccl-tests' algbw/busbw, the alpha-beta fit, a ring over OS pipes,
                     and a torch.distributed (gloo/nccl) benchmark
    gpurt.nccltests  parse nccl-tests output (all_reduce_perf & co.) and re-check its numbers
    gpurt.launch     launch overhead and CUDA Graphs
    gpurt.container  how this container sees its GPU: device nodes, injected driver libs, versions
    gpurt.dcgm       DCGM-exporter metrics -> derived signals, XID triage and alert rules

Importing ``gpurt`` imports nothing heavy. ``gpurt.kernels`` imports numba; torch and triton are
only imported inside the functions that need them.
"""

__version__ = "0.1.0"
