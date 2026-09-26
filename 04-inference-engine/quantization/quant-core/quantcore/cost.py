"""cost.py - what a scheme buys: bytes for decode, FLOPs for prefill, KV bytes for concurrency.

The one idea: a step costs max(bytes / bandwidth, FLOPs / peak) + overhead (layer 01's roofline). A
quantization scheme changes four inputs: bytes per linear weight (W8, W4), the precision the tensor
cores multiply in - which changes only when activations are quantized too (FP8/INT8 W8A8, FP4 W4A4)
AND the GPU has that datapath - bytes per KV element, and the memory left for KV blocks. `supported`
says what a checkpoint actually runs as on a GPU generation (an FP8 checkpoint on an A100 is a
weight-only FP8 model), `step_cost` prices a step, and `table`/`choose` turn that into a decision.

Every time here is SIMULATED (a roofline model with assumed efficiencies), labelled so wherever it is
printed. GPU numbers are dense datasheet values as in roofline.specs, September 2026 (verify).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    cc: float                 # compute capability
    mem_gb: float
    bw_tbs: float
    tflops: dict              # dense tensor TFLOP/s by precision; T4 has fp16 but no bf16

    def peak(self, precision: str) -> float:
        p = "fp16" if precision == "bf16" and "bf16" not in self.tflops else precision
        return self.tflops[p] * 1e12


# (verify) - the same dense figures as roofline.specs.DEVICES. mem_gb is the round marketing figure, used as
# decimal GB (80 -> 80e9 B), like minengine.perf's round inputs; an "80 GB" H100 really has 80 GiB and its
# driver reports 79.65 GiB (85.5e9 B). So kv_blocks() undercounts on large parts: for Llama-3.1-70B in FP8
# with an FP8 KV cache on an H100 it finds 0 blocks where vLLM's defaults leave 1,047 (4 sessions of 4K).
# vLLM's own budget (0.92 of the reported memory, profiled overheads) is modelled by servelab.sizing and
# quantlab.kv.size; use those for a real deployment.
GPUS = {
    "T4": GPU("T4", 7.5, 16, 0.32, {"fp16": 65, "int8": 130}),
    "A100-80GB": GPU("A100-80GB", 8.0, 80, 2.039, {"bf16": 312, "int8": 624}),
    "L4": GPU("L4", 8.9, 24, 0.30, {"bf16": 121, "fp8": 242.5, "int8": 242.5}),
    "H100-SXM": GPU("H100-SXM", 9.0, 80, 3.35, {"bf16": 989.4, "fp8": 1978.9, "int8": 1978.9}),
    "B200": GPU("B200", 10.0, 180, 8.0, {"bf16": 2250, "fp8": 4500, "fp4": 9000, "int8": 4500}),
    "RTX-PRO-6000": GPU("RTX-PRO-6000", 12.0, 96, 1.6, {"bf16": 500, "fp8": 1000, "fp4": 2000}),
}


@dataclass(frozen=True)
class Scheme:
    name: str
    w_bits: float             # bits per quantized linear weight, scales included
    compute: str              # what the tensor cores multiply in when the GPU supports it natively
    a_bytes: float            # bytes per activation element entering a GEMM
    kernel: str


SCHEMES = {
    "bf16": Scheme("bf16", 16, "bf16", 2, "cuBLAS/CUTLASS"),
    "w8a16-fp8": Scheme("w8a16-fp8", 8, "bf16", 2, "Marlin FP8 (weight-only)"),
    "w4a16": Scheme("w4a16", 4.125, "bf16", 2, "Marlin (Machete on Hopper)"),
    "w8a8-int8": Scheme("w8a8-int8", 8, "int8", 1, "CUTLASS INT8"),
    "w8a8-fp8": Scheme("w8a8-fp8", 8, "fp8", 1, "CUTLASS/FlashInfer FP8"),
    "w4a4-nvfp4": Scheme("w4a4-nvfp4", 4.5, "fp4", 0.5, "CUTLASS/FlashInfer NVFP4"),
}
# Least to most aggressive, by typical accuracy cost - a starting order, not a result (measure, section 8).
ACCURACY_ORDER = ["bf16", "w8a16-fp8", "w8a8-fp8", "w8a8-int8", "w4a16", "w4a4-nvfp4"]


def supported(gpu: GPU, scheme: str, kv_bits: int = 16) -> dict:
    """What `scheme` runs as on `gpu` (vLLM 0.30.0/main: kernels' get_min_capability and dispatch; verify):
    {'runs_as': scheme name or None, 'kernel': str, 'note': str, 'kv': bool}."""
    s, cc = SCHEMES[scheme], gpu.cc
    runs, kernel, note = scheme, s.kernel, ""
    if scheme == "bf16" and "bf16" not in gpu.tflops:
        note = "no bf16 on this GPU: serve fp16 (--dtype half)"
    elif scheme == "w4a16" and cc == 9.0:
        kernel = "Machete"
    elif scheme == "w8a8-fp8" and cc < 8.9:
        runs, kernel, note = "w8a16-fp8", "Marlin FP8", "no FP8 tensor cores: weight-only, memory win only"
    elif scheme == "w8a8-int8" and cc >= 10:
        runs, note = None, "INT8 W8A8 is not supported on compute capability >= 10.0"
    elif scheme == "w4a4-nvfp4" and cc < 10:
        runs, kernel, note = "w4a16", "Marlin NVFP4 (weight-only)", "no FP4 tensor cores: NVFP4 weights run W4A16"
    kv_ok = kv_bits >= 16 or cc >= 8.0
    if not kv_ok:
        note = (note + "; " if note else "") + "no FP8 KV cache below SM80 (Triton needs SM89, FlashInfer SM80)"
    return {"runs_as": runs, "kernel": kernel, "note": note, "kv": kv_ok}


@dataclass(frozen=True)
class Model:
    name: str
    params: float
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    d: int
    tied: bool = False

    @property
    def embed_params(self) -> float:
        return self.vocab * self.d

    @property
    def matmul_params(self) -> float:
        """Parameters every token multiplies through (recipes quantize these; embeddings and LM head stay 16-bit)."""
        return self.params - self.embed_params * (1 if self.tied else 2)

    def weight_bytes(self, w_bits: float = 16) -> float:
        return self.matmul_params * w_bits / 8 + self.embed_params * (1 if self.tied else 2) * 2

    def streamed_bytes(self, w_bits: float = 16) -> float:
        """Read by every step: the linear layers and the LM head; the input embedding is only gathered."""
        return self.matmul_params * w_bits / 8 + self.embed_params * 2

    def kv_bytes_per_token(self, kv_bits: float = 16) -> float:
        return 2 * self.layers * self.kv_heads * self.head_dim * kv_bits / 8


MODELS = {   # from each model's config.json (verify)
    "llama-3.1-8b": Model("llama-3.1-8b", 8.03e9, 32, 32, 8, 128, 128256, 4096),
    "llama-3.1-70b": Model("llama-3.1-70b", 70_553_706_496, 80, 64, 8, 128, 128256, 8192),
    "qwen2.5-0.5b": Model("qwen2.5-0.5b", 494_032_768, 24, 14, 2, 64, 151936, 896, tied=True),
    "qwen2.5-1.5b": Model("qwen2.5-1.5b", 1_543_714_304, 28, 12, 2, 128, 151936, 1536, tied=True),
}


def gemm_time(M: int, K: int, N: int, gpu: GPU, scheme: str = "bf16", w_bits: float | None = None) -> dict:
    """Ideal roofline time of one linear layer for M tokens: FLOPs 2MKN; bytes K N w + M K a + M N 2."""
    s = SCHEMES[supported(gpu, scheme)["runs_as"] or scheme]
    wb, a = (w_bits or SCHEMES[scheme].w_bits) / 8, s.a_bytes
    flops, nbytes = 2 * M * K * N, K * N * wb + M * K * a + M * N * 2
    t_c, t_m = flops / gpu.peak(s.compute), nbytes / (gpu.bw_tbs * 1e12)
    return {"t": max(t_c, t_m), "t_compute": t_c, "t_memory": t_m, "bound": "compute" if t_c > t_m else "memory"}


def crossover_tokens(K: int, N: int, gpu: GPU, scheme: str = "w4a16", w_bits: float | None = None) -> float:
    """Tokens per step above which the GEMM turns compute-bound: 2MKN/P = (K N w + M(K a + 2N))/B."""
    s = SCHEMES[supported(gpu, scheme)["runs_as"] or scheme]
    wb, P, B = (w_bits or SCHEMES[scheme].w_bits) / 8, gpu.peak(s.compute), gpu.bw_tbs * 1e12
    return K * N * wb / B / (2 * K * N / P - (K * s.a_bytes + 2 * N) / B)


def step_cost(gpu: GPU, model: Model, scheme: str, chunks, kv_bits: float = 16, flop_eff: float = 0.6,
              bw_eff: float = 0.8, overhead_s: float = 0.002) -> dict:
    """SIMULATED step time for chunks [(context, new_tokens), ...] - the same model as minengine.perf:
    FLOPs 2 x linear params x tokens + 2 x vocab x d per request (LM head) + 4 L H hd per attended pair;
    bytes = streamed weights + KV read and written. The scheme's compute precision applies to every FLOP
    (attention is ~3% of a 1.8K-token prefill). Defaults: 60% of peak FLOP/s, 80% of bandwidth, 2 ms."""
    sup = supported(gpu, scheme, kv_bits)
    if sup["runs_as"] is None or not sup["kv"]:
        raise ValueError(f"{scheme} (kv {kv_bits}-bit) does not run on {gpu.name}: {sup['note']}")
    compute = SCHEMES[sup["runs_as"]].compute
    tokens = sum(n for _, n in chunks)
    pairs = sum(n * c + n * (n + 1) / 2 for c, n in chunks)
    flops = (2 * model.matmul_params * tokens + 2 * model.embed_params * len(chunks)
             + 4 * model.layers * model.heads * model.head_dim * pairs)
    nbytes = (model.streamed_bytes(SCHEMES[scheme].w_bits)
              + model.kv_bytes_per_token(kv_bits) * (sum(c for c, _ in chunks) + tokens))
    t_m, t_c = nbytes / (gpu.bw_tbs * 1e12 * bw_eff), flops / (gpu.peak(compute) * flop_eff)
    return {"t": max(t_m, t_c) + overhead_s, "bound": "memory" if t_m >= t_c else "compute",
            "flops": flops, "bytes": nbytes}


def kv_blocks(gpu: GPU, model: Model, scheme: str, kv_bits: float = 16, util: float = 0.9,
              block: int = 16, reserve_bytes: float = 1e9) -> int:
    """KV blocks left after weights and a flat activation reserve: util x mem_gb x 1e9 - weights - 1 GB
    (minengine.perf.kv_cache_blocks's round inputs, decimal GB; vLLM's own budget - 0.92 of the
    driver-reported GiB minus profiled overheads - is servelab.sizing / quantlab.kv.size)."""
    free = gpu.mem_gb * 1e9 * util - model.weight_bytes(SCHEMES[scheme].w_bits) - reserve_bytes
    return max(0, int(free // (block * model.kv_bytes_per_token(kv_bits))))


def sessions(gpu: GPU, model: Model, scheme: str, tokens: int, kv_bits: float = 16, block: int = 16, **kw) -> int:
    """Whole sessions of `tokens` that fit: each needs ceil((tokens - 1) / block) blocks (the last token
    generated is never written back)."""
    return kv_blocks(gpu, model, scheme, kv_bits, block=block, **kw) // math.ceil((tokens - 1) / block)


def table(gpu: GPU, model: Model, schemes=None, kv=(16, 8), batch: int = 32, context: int = 1000,
          prompt: int = 1800, session: int = 2000) -> list[dict]:
    """One row per (scheme, KV bits): what it runs as, weights, SIMULATED decode at batch 1 and `batch`,
    prefill of `prompt` tokens, and sessions of `session` tokens; rows that cannot run or fit say why."""
    rows = []
    for name in schemes or SCHEMES:
        for kvb in kv:
            sup = supported(gpu, name, kvb)
            row = {"scheme": name, "kv_bits": kvb, **sup, "weights_gb": model.weight_bytes(SCHEMES[name].w_bits) / 1e9}
            fits = sessions(gpu, model, name, session, kvb) > 0
            if not fits:
                row["note"] = (row["note"] + "; " if row["note"] else "") + "does not fit: no room for one session"
            if sup["runs_as"] is not None and sup["kv"] and fits:
                row.update(decode_1_ms=1e3 * step_cost(gpu, model, name, [(context, 1)], kvb)["t"],
                           decode_b_ms=1e3 * step_cost(gpu, model, name, [(context, 1)] * batch, kvb)["t"],
                           prefill_ms=1e3 * step_cost(gpu, model, name, [(0, prompt)], kvb)["t"],
                           sessions=sessions(gpu, model, name, session, kvb))
            rows.append(row)
    return rows


def choose(rows, min_sessions: int = 0, max_prefill_ms: float = math.inf, max_decode_ms: float = math.inf,
           allowed=None) -> dict | None:
    """The least aggressive runnable row (ACCURACY_ORDER, 16-bit KV before 8-bit) that meets every target."""
    ok = [r for r in rows if "prefill_ms" in r and r["sessions"] >= min_sessions and r["prefill_ms"] <= max_prefill_ms
          and r["decode_b_ms"] <= max_decode_ms and (allowed is None or r["scheme"] in allowed)]
    return min(ok, key=lambda r: (ACCURACY_ORDER.index(r["scheme"]), -r["kv_bits"]), default=None)


def cost_per_million(price_per_hour: float, tokens_per_s: float, utilisation: float = 1.0) -> float:
    """$/M tokens = $/hr / (tokens/s x 3600 x utilisation) x 10^6 (roofline.cost, layer 01 section 8)."""
    return price_per_hour / (tokens_per_s * 3600 * utilisation) * 1e6
