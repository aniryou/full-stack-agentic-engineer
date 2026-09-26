"""roofline -- reading the machine: rooflines, memory, fabrics, and the cost of a token.

    from roofline import specs, llm, fabric, storage, reliability, cost
    from roofline.roofline import attainable, gemm, time_kernel

    h100 = specs.get("h100-sxm")
    step = llm.decode(llm.PRESETS["llama-3.1-8b"], h100, batch=1, context=1024)
    step.bound, step.time        # ('memory', 0.0045...)

Seven small modules, standard library only. Read them in this order: specs,
roofline, llm, fabric, storage, reliability, cost. Every computed number the
topic's PRIMER.md quotes comes from a function here and is pinned by a test.
"""
from . import cost, fabric, llm, reliability, roofline, specs, storage
from .roofline import attainable, ridge_point, time_kernel
from .specs import AS_OF, DEVICES, Device

__all__ = ["specs", "roofline", "llm", "fabric", "storage", "reliability", "cost",
           "attainable", "ridge_point", "time_kernel", "AS_OF", "DEVICES", "Device"]
