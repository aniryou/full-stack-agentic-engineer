"""offload.py — fit an MoE on one 16 or 24 GB GPU: quantize the experts, or move them off the GPU.

One idea: memory is sized by the TOTAL, so a small GPU runs a MoE by shrinking or relocating the
experts, and each way has a different price per decode step:

* **4-bit experts on the GPU** (GPTQ/AWQ INT4, MXFP4 for gpt-oss): ~3.8x fewer expert bytes, no
  PCIe traffic; costs accuracy (measure it) and needs kernels for your GPU (MXFP4 in vLLM needs
  compute capability 8.0+ and bf16 — not a T4).
* **vLLM UVA offload** (``--cpu-offload-gb N --cpu-offload-params experts``): N GiB of weights live
  in pinned CPU memory and are read over PCIe *in every forward pass*, whichever experts the
  router picks — it adds N GiB / PCIe-bandwidth to every step.
* **llama.cpp ``--n-cpu-moe`` / ``--cpu-moe``**: expert tensors stay in CPU RAM and are *computed on
  the CPU*; only activations cross PCIe, and only the touched experts are read — cheap at batch 1,
  bound by CPU FLOP/s as the batch grows (inferred from the buffer-type override; verify).

Everything left over after weights is KV cache; MoE did not shrink it. ``fit`` and the step costs are
arithmetic (**simulated**); ``parse_startup_log`` reads the lines vLLM prints at start-up, and
``stream.measure_itl`` measures the step, so a T1 run replaces every estimate here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .configs import GPU, GiB, Model, expert_bytes, weight_bytes
from .stream import experts_touched

DEFAULT_UTIL = 0.92            # vLLM v0.30.0's --gpu-memory-utilization default
DEFAULT_OVERHEAD_GIB = 1.5     # activations + CUDA graphs + non-torch memory for a 1-7B model (verify: calibrate)


@dataclass(frozen=True)
class Fit:
    model: str
    gpu: str
    scheme: str
    weights_gib: float
    budget_gib: float
    offload_gib: float
    kv_gib: float
    kv_tokens: int
    max_model_len: int
    runnable: bool
    reason: str

    @property
    def fits(self) -> bool:
        return self.runnable and self.kv_tokens >= self.max_model_len

    @property
    def sessions(self) -> float:
        """Full-length sequences the KV cache holds at once (vLLM's 'Maximum concurrency')."""
        return max(0.0, self.kv_tokens / self.max_model_len)

    def line(self) -> str:
        verdict = "fits" if self.fits else ("no: " + self.reason if self.reason else "no: no room for KV")
        off = f" offload {self.offload_gib:4.1f} GiB" if self.offload_gib else ""
        return (f"{self.model:22s} {self.scheme:12s} on {self.gpu:8s} weights {self.weights_gib:5.1f} GiB{off}  "
                f"KV {max(self.kv_gib, 0):5.1f} GiB = {max(self.kv_tokens, 0):>9,} tokens "
                f"({self.sessions:5.1f} x {self.max_model_len})  {verdict}")


def runnable(model: Model, gpu: GPU, scheme: str) -> tuple[bool, str]:
    """Can vLLM run this precision on this GPU at all? (vLLM v0.30.0 capability rules; verify.)"""
    if scheme == "mxfp4" and gpu.compute_capability < 8.0:
        return False, "MXFP4 needs compute capability 8.0+ and bf16 (Mxfp4Config)"
    if model.native == "mxfp4" and scheme not in ("mxfp4", "bf16"):
        return False, "checkpoint ships MXFP4 experts"
    if scheme == "bf16" and not gpu.bf16:
        return False, "no bf16 on this GPU: serve 16-bit weights with --dtype half (scheme fp16)"
    if scheme == "fp8" and not gpu.fp8:
        return False, "no FP8 tensor cores (verify weight-only FP8 support in your vLLM)"
    return True, ""


def fit(model: Model, gpu: GPU, scheme: str = "fp16", *, max_model_len: int = 4096, util: float = DEFAULT_UTIL,
        overhead_gib: float = DEFAULT_OVERHEAD_GIB, offload_gib: float = 0.0, kv_bytes: float = 2) -> Fit:
    """Weights, KV room and sessions for one GPU: KV = util x memory - overhead - (weights - offload)."""
    ok, why = runnable(model, gpu, scheme)
    w = weight_bytes(model, scheme) / GiB
    budget = util * gpu.memory_gib - overhead_gib
    kv = budget - (w - offload_gib)
    tokens = int(kv * GiB // model.kv_bytes_per_token(kv_bytes)) if kv > 0 else 0
    return Fit(model.name, gpu.name, scheme, w, budget, offload_gib, kv, tokens, max_model_len, ok,
               why if not ok else "")


def min_offload_gib(model: Model, gpu: GPU, scheme: str = "fp16", *, kv_tokens: int = 16_384,
                    util: float = DEFAULT_UTIL, overhead_gib: float = DEFAULT_OVERHEAD_GIB, kv_bytes: float = 2) -> float:
    """The smallest ``--cpu-offload-gb`` (GiB, rounded up to 0.5) that leaves ``kv_tokens`` of KV cache."""
    need = weight_bytes(model, scheme) / GiB + kv_tokens * model.kv_bytes_per_token(kv_bytes) / GiB
    short = need - (util * gpu.memory_gib - overhead_gib)
    return max(0.0, -(-short // 0.5) * 0.5)


def offload_step_s(offload_gib: float, pcie_gbs: float) -> float:
    """UVA offload: every offloaded byte crosses PCIe in every forward pass, independent of routing."""
    return offload_gib * GiB / (pcie_gbs * 1e9)


def cpu_expert_step_s(model: Model, batch: int, *, scheme: str = "int4-experts", cpu_bw_gbs: float = 40.0,
                      cpu_tflops: float = 0.3) -> float:
    """llama.cpp-style: the CPU reads the experts this step touches from DRAM and multiplies them there.
    max(touched bytes / DRAM bandwidth, expert FLOPs / CPU FLOP/s). ``cpu_bw_gbs`` and ``cpu_tflops``
    describe *your* host (assumed defaults: a few server cores; verify with a STREAM run)."""
    per_expert = expert_bytes(model, scheme) / (model.layers * model.n_experts)
    touched = experts_touched(model.n_experts, model.top_k, batch)
    read = model.layers * touched * per_expert
    flops = 2 * model.layers * batch * model.top_k * model.expert_params()
    return max(read / (cpu_bw_gbs * 1e9), flops / (cpu_tflops * 1e12))


# ---- what vLLM prints at start-up ------------------------------------------------------------------------
_LOG_PATTERNS = {   # wording of vLLM v0.30.0 (as verified for layer 04's servelab.sizing.parse_startup_log)
    "model_loading_gib": r"Model loading took ([\d.]+) GiB",
    "available_kv_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "kv_cache_tokens": r"KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
    "max_model_len": r"Maximum concurrency for ([\d,]+) tokens per request",
    "gpu_memory_utilization": r"The current --gpu-memory-utilization=([\d.]+)",
}


def parse_startup_log(text: str) -> dict:
    """Capacity lines plus the two MoE-specific ones: whether vLLM found a tuned fused-MoE config
    (``Using configuration from ... for MoE layer.``) or fell back (``Using default MoE config.
    Performance might be sub-optimal!`` — expected on T4/L4/A10, which have no tuned files)."""
    out: dict = {}
    for key, pat in _LOG_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            v = m.group(1).replace(",", "")
            out[key] = int(v) if key in ("kv_cache_tokens", "max_model_len") else float(v)
    out["moe_default_config"] = "Using default MoE config" in text
    m = re.search(r"Using configuration from (\S+) for MoE layer", text)
    out["moe_config_file"] = m.group(1) if m else None
    return out


def compare_with_log(pred: Fit, log: dict) -> dict:
    """Prediction vs the startup log: weights and KV in GiB, and the overhead the log implies."""
    res = {"pred_weights_gib": pred.weights_gib - pred.offload_gib, "pred_kv_gib": pred.kv_gib}
    if "model_loading_gib" in log:
        res["log_weights_gib"] = log["model_loading_gib"]
    if "available_kv_gib" in log:
        res["log_kv_gib"] = log["available_kv_gib"]
        res["kv_error_gib"] = pred.kv_gib - log["available_kv_gib"]
    return res
