"""moelab — watch a mixture-of-experts model route, stream, split and squeeze.

A tiny MoE trained on the CPU, router hooks, decode step time versus batch, expert parallelism on
two GPUs, and fitting a MoE on one small GPU: each at T0 (laptop, simulated or bundled data,
labelled) and at T1/T2 (a real GPU and vLLM, measured). Standalone: it never imports this topic's
``moecore`` or other labs; the few formulas it shares with them are re-implemented and pinned by tests.
"""
__version__ = "0.1.0"
