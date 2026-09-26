"""sizing.py — size a vLLM deployment before you start it: weights, KV blocks, concurrency.

One idea: vLLM takes ``gpu_memory_utilization × total GPU memory``, subtracts what the model
needs to run (weights, peak activations of a profiling pass, CUDA graphs, non-torch buffers),
and turns *all* of the rest into fixed-size KV blocks. Capacity follows from four lines:

    kv_bytes_per_token = 2 (K and V) × layers × kv_heads_per_gpu × head_dim × kv_dtype_bytes
    bytes_per_block    = block_size × kv_bytes_per_token
    num_blocks         = floor(kv_budget / bytes_per_block)
    max_concurrency    = num_blocks / ceil(max_model_len / block_size)   # the "Nx" in vLLM's log

The per-token formula is the one in ``00-foundations/gpu-capacity-planning/capacity.py``
(``kv_per_token_kb``); this module adds what the engine adds on top: the utilization cap, block
granularity, tensor-parallel sharding of KV heads, quantized weights and KV, and the check that
one request of ``max_model_len`` tokens fits at all. The overhead terms (activations, CUDA
graphs, non-torch memory) are *estimates* — calibrate them with :func:`parse_startup_log`
against the lines vLLM prints at startup (notebook 01).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

GiB = 1024**3
CONFIG_DIR = Path(__file__).parent / "data" / "configs"

# Bytes per element. "auto" means "whatever the checkpoint is" (usually bfloat16 -> 2).
DTYPE_BYTES = {"float32": 4, "float": 4, "bfloat16": 2, "float16": 2, "half": 2,
               "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "int8": 1}
# Weight bytes per *quantized linear-layer* parameter. Embeddings, lm_head, norms and biases stay
# in 16-bit in typical AWQ/GPTQ/FP8 checkpoints; group-wise int4 also stores one fp16 scale and a
# 4-bit zero point per group (2.5 bytes per group of 128 -> +0.0195 bytes/param).
QUANT_BYTES = {"fp8": 1.0, "int8": 1.0, "w8a8": 1.0, "awq": 0.5, "gptq": 0.5, "int4": 0.5}
INT4_GROUP_OVERHEAD = 2.5 / 128


@dataclass(frozen=True)
class GPU:
    """Datasheet numbers (dense, no sparsity). ``memory_gib`` is the total the driver reports
    (``nvidia-smi --query-gpu=memory.total``). vLLM multiplies the total seen by CUDA
    (``torch.cuda.mem_get_info()[1]``) by ``gpu_memory_utilization``; the two can differ by a few
    hundred MiB, so for exact planning pass ``gpu_memory_bytes=`` measured on your GPU.
    All values (verify) against the vendor datasheet."""
    name: str
    memory_gib: float
    mem_bw_gbs: float          # GB/s (10^9 bytes/s)
    bf16_tflops: float         # dense tensor-core FP16/BF16 TFLOPS
    fp8_tflops: float = 0.0    # 0 = no FP8 tensor cores (T4, A100)
    compute_capability: str = ""
    bf16: bool = True          # False on Turing (T4): serve with --dtype half


GPUS = {
    "T4": GPU("T4", 15.0, 320, 65, 0, "7.5", bf16=False),
    "L4": GPU("L4", 22.49, 300, 121, 242, "8.9"),
    "RTX4090": GPU("RTX4090", 23.99, 1008, 165, 330, "8.9"),
    "L40S": GPU("L40S", 44.99, 864, 362, 733, "8.9"),
    "A100-40GB": GPU("A100-40GB", 40.0, 1555, 312, 0, "8.0"),
    "A100-80GB": GPU("A100-80GB", 80.0, 2039, 312, 0, "8.0"),
    "H100-80GB": GPU("H100-80GB", 79.65, 3352, 989, 1979, "9.0"),
    "RTXPRO6000": GPU("RTXPRO6000", 95.0, 1600, 500, 1000, "12.0"),  # (verify) all numbers
}


def gpu(name_or_obj) -> GPU:
    if isinstance(name_or_obj, GPU):
        return name_or_obj
    key = str(name_or_obj).upper().replace(" ", "")
    for k, v in GPUS.items():
        if k.upper() == key:
            return v
    raise KeyError(f"unknown GPU {name_or_obj!r}; known: {', '.join(GPUS)} (or pass gpu_memory_bytes=)")


# ---------------------------------------------------------------------------------------------
# Model config: the handful of config.json fields that decide memory
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_type: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    num_experts: int = 0            # 0 = dense
    experts_per_token: int = 0
    moe_intermediate_size: int = 0
    qkv_bias: bool = False          # Qwen2 has q/k/v biases
    qk_norm: bool = False           # Qwen3 normalizes q and k per head

    @classmethod
    def from_hf(cls, cfg: dict, name: str | None = None) -> "ModelConfig":
        """Build from a Hugging Face ``config.json`` dict (decoder-only, llama-like families)."""
        c = cfg.get("text_config", cfg)  # multimodal wrappers nest the language model config
        heads = int(c["num_attention_heads"])
        hidden = int(c["hidden_size"])
        mt = str(c.get("model_type", cfg.get("model_type", "")))
        n_exp = int(c.get("num_local_experts") or c.get("num_experts") or 0)
        return cls(
            name=name or cfg.get("_name_or_path") or cfg.get("name") or mt,
            model_type=mt,
            num_layers=int(c["num_hidden_layers"]),
            hidden_size=hidden,
            num_heads=heads,
            num_kv_heads=int(c.get("num_key_value_heads") or heads),
            # ALWAYS prefer an explicit head_dim: Qwen3-0.6B has hidden/heads = 64 but head_dim = 128.
            head_dim=int(c.get("head_dim") or hidden // heads),
            intermediate_size=int(c.get("intermediate_size") or 0),
            vocab_size=int(c["vocab_size"]),
            max_position_embeddings=int(c.get("max_position_embeddings") or 0),
            tie_word_embeddings=bool(c.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False))),
            torch_dtype=str(c.get("torch_dtype") or c.get("dtype") or cfg.get("torch_dtype") or "bfloat16"),
            num_experts=n_exp,
            experts_per_token=int(c.get("num_experts_per_tok") or 0),
            moe_intermediate_size=int(c.get("moe_intermediate_size") or 0),
            qkv_bias=bool(c.get("attention_bias", mt in ("qwen2", "qwen2_moe"))),
            qk_norm=mt in ("qwen3", "qwen3_moe"),
        )


def sample_configs() -> list[str]:
    """Names of the bundled config.json files (key fields of real Hugging Face configs)."""
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def load_config(model: str | dict | ModelConfig) -> ModelConfig:
    """A bundled sample name (``"qwen2.5-0.5b-instruct"``), a path to a config.json, or a dict."""
    if isinstance(model, ModelConfig):
        return model
    if isinstance(model, dict):
        return ModelConfig.from_hf(model)
    p = Path(model)
    if not p.exists():
        key = str(model).lower().split("/")[-1]
        p = CONFIG_DIR / f"{key}.json"
        if not p.exists():
            raise FileNotFoundError(f"no config {model!r}; bundled: {', '.join(sample_configs())}")
    cfg = json.loads(p.read_text())
    return ModelConfig.from_hf(cfg, name=cfg.get("_name_or_path") or p.stem)


# ---------------------------------------------------------------------------------------------
# Parameters and weight bytes
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    total: int         # what must sit in GPU memory
    active: int        # what one token multiplies through (MoE: only the routed experts)
    embedding: int     # embed_tokens + lm_head (counted once if tied)
    linear: int        # attention + MLP projection weights: what weight quantization shrinks


def param_count(m: ModelConfig) -> Params:
    """Count parameters of a llama-style decoder from its config (exact for Llama/Mistral/Qwen2/
    Qwen3/Mixtral: Llama-3.1-8B -> 8,030,261,248). Shared experts and exotic layers are not modelled."""
    q = m.hidden_size * m.num_heads * m.head_dim
    kv = 2 * m.hidden_size * m.num_kv_heads * m.head_dim
    o = m.num_heads * m.head_dim * m.hidden_size
    bias = (m.num_heads + 2 * m.num_kv_heads) * m.head_dim if m.qkv_bias else 0
    qk_norm = 2 * m.head_dim if m.qk_norm else 0
    if m.num_experts:
        expert = 3 * m.hidden_size * (m.moe_intermediate_size or m.intermediate_size)
        router = m.hidden_size * m.num_experts
        mlp_total, mlp_active = m.num_experts * expert + router, m.experts_per_token * expert + router
        mlp_linear = m.num_experts * expert
    else:
        mlp_total = mlp_active = mlp_linear = 3 * m.hidden_size * m.intermediate_size
    norms = 2 * m.hidden_size
    embed = m.vocab_size * m.hidden_size
    embedding = embed if m.tie_word_embeddings else 2 * embed
    total = m.num_layers * (q + kv + o + bias + qk_norm + mlp_total + norms) + m.hidden_size + embedding
    active = m.num_layers * (q + kv + o + bias + qk_norm + mlp_active + norms) + m.hidden_size + embedding
    return Params(total=total, active=active, embedding=embedding, linear=m.num_layers * (q + kv + o + mlp_linear))


def dtype_bytes(dtype: str, m: ModelConfig | None = None) -> float:
    if dtype in (None, "auto"):
        dtype = m.torch_dtype if m else "bfloat16"
    try:
        return DTYPE_BYTES[str(dtype).lower()]
    except KeyError:
        raise KeyError(f"unknown dtype {dtype!r}; known: {', '.join(DTYPE_BYTES)}") from None


def weight_bytes(m: ModelConfig, dtype: str = "auto", quantization: str | None = None,
                 tensor_parallel_size: int = 1) -> float:
    """Bytes of weights **per GPU**. Quantization shrinks only the linear layers:
    ``linear × quant_bytes + everything_else × 16-bit``. TP shards (almost) everything by tp."""
    p = param_count(m)
    base = dtype_bytes(dtype, m)
    if not quantization:
        total = p.total * base
    else:
        qb = QUANT_BYTES[quantization.lower()]
        if qb < 1:
            qb += INT4_GROUP_OVERHEAD
        total = p.linear * qb + (p.total - p.linear) * base
    return total / tensor_parallel_size


def kv_bytes_per_token(m: ModelConfig, kv_cache_dtype: str = "auto", dtype: str = "auto",
                       tensor_parallel_size: int = 1) -> int:
    """KV bytes one token occupies across all layers, **per GPU**.

    ``2 × layers × kv_heads_per_gpu × head_dim × bytes``. Tensor parallelism splits KV heads
    across GPUs; with fewer KV heads than GPUs each GPU keeps one (replicated) head."""
    b = dtype_bytes(dtype, m) if kv_cache_dtype in (None, "auto") else dtype_bytes(kv_cache_dtype)
    kv_heads = max(1, m.num_kv_heads // tensor_parallel_size)
    return int(2 * m.num_layers * kv_heads * m.head_dim * b)


# ---------------------------------------------------------------------------------------------
# The sizing report
# ---------------------------------------------------------------------------------------------
@dataclass
class SizingReport:
    model: str
    gpu: str
    total_bytes: int
    gpu_memory_utilization: float
    requested_bytes: int
    weights_bytes: int
    overhead_bytes: dict
    kv_budget_bytes: int
    kv_bytes_per_token: int
    block_size: int
    bytes_per_block: int
    num_blocks: int
    max_model_len: int
    blocks_per_request: int          # blocks one max_model_len request needs
    max_concurrency: float           # num_blocks / blocks_per_request (vLLM's "Maximum concurrency")
    kv_cache_tokens: int             # int(max_concurrency × max_model_len) (vLLM's "GPU KV cache size")
    fits: bool
    estimated_max_model_len: int     # largest max_model_len that still fits one request
    typical_len: int | None = None
    concurrency_at_typical_len: float | None = None
    notes: list = field(default_factory=list)

    @property
    def kv_capacity_tokens(self) -> int:
        """Raw token slots: num_blocks × block_size."""
        return self.num_blocks * self.block_size

    def error(self) -> str | None:
        """vLLM's startup error text when one max_model_len request does not fit, else None."""
        if self.kv_budget_bytes <= 0:
            return "No available memory for the cache blocks. Try increasing `gpu_memory_utilization`."
        if self.fits:
            return None
        need = self.blocks_per_request * self.bytes_per_block
        return (f"To serve at least one request with the model's max seq len ({self.max_model_len}), "
                f"({need / GiB:.2f} GiB KV cache is needed, which is larger than the available KV cache "
                f"memory ({self.kv_budget_bytes / GiB:.2f} GiB). Based on the available memory, the "
                f"estimated maximum model length is {self.estimated_max_model_len}. Try increasing "
                f"`gpu_memory_utilization` or decreasing `max_model_len`.")

    def summary(self) -> str:
        g = lambda b: f"{b / GiB:6.2f} GiB"  # noqa: E731
        lines = [
            f"{self.model} on {self.gpu}  (gpu_memory_utilization={self.gpu_memory_utilization}, "
            f"max_model_len={self.max_model_len}, block_size={self.block_size})",
            f"  total GPU memory        {g(self.total_bytes)}",
            f"  requested (x util)      {g(self.requested_bytes)}",
            f"  - weights               {g(self.weights_bytes)}",
        ]
        lines += [f"  - {k:<21} {g(v)}  (estimate)" for k, v in self.overhead_bytes.items()]
        lines += [
            f"  = KV cache budget       {g(self.kv_budget_bytes)}",
            f"  KV per token            {self.kv_bytes_per_token:,} B   per block ({self.block_size} tok): "
            f"{self.bytes_per_block:,} B",
            f"  KV blocks               {self.num_blocks:,}  ({self.kv_capacity_tokens:,} token slots)",
            f"  vLLM would log:  GPU KV cache size: {self.kv_cache_tokens:,} tokens, Maximum concurrency "
            f"for {self.max_model_len:,} tokens per request: {self.max_concurrency:.2f}x",
        ]
        if self.typical_len:
            lines.append(f"  concurrency at {self.typical_len:,} tokens/request: "
                         f"{self.concurrency_at_typical_len:.1f}")
        if not self.fits:
            lines.append("  DOES NOT FIT: " + (self.error() or ""))
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["kv_capacity_tokens"] = self.kv_capacity_tokens
        return d


def overhead_estimate(m: ModelConfig, *, dtype: str = "auto", max_num_batched_tokens: int = 2048,
                      max_num_seqs: int = 256, tensor_parallel_size: int = 1,
                      enforce_eager: bool = False) -> dict:
    """Rough non-KV, non-weight memory (bytes). Calibrate against a real startup log.

    * activations: the profiling forward pass runs ``max_num_batched_tokens`` tokens; the MLP's
      gate/up outputs dominate, plus fp32 logits for ``max_num_seqs`` sequences in the sampler;
    * cuda_graphs: captured decode graphs (0 with ``--enforce-eager``);
    * non_torch: cuBLAS/attention workspaces, NCCL buffers when TP > 1."""
    b = dtype_bytes(dtype, m)
    inter = (m.moe_intermediate_size * m.experts_per_token) if m.num_experts else m.intermediate_size
    act = max_num_batched_tokens * (2 * inter / tensor_parallel_size + 4 * m.hidden_size) * b
    logits = max_num_seqs * m.vocab_size * 4 * 2
    big = param_count(m).total >= 3e9
    graphs = 0 if enforce_eager else (0.5 if big else 0.25) * GiB
    non_torch = (0.2 + (0.3 if tensor_parallel_size > 1 else 0.0)) * GiB
    return {"activations": int(act + logits), "cuda_graphs": int(graphs), "non_torch": int(non_torch)}


def size(model, gpu_name=None, *, gpu_memory_utilization: float = 0.92, max_model_len: int | None = None,
         block_size: int = 16, dtype: str = "auto", quantization: str | None = None,
         kv_cache_dtype: str = "auto", tensor_parallel_size: int = 1, max_num_batched_tokens: int = 2048,
         max_num_seqs: int = 256, enforce_eager: bool = False, gpu_memory_bytes: int | None = None,
         overhead_bytes: dict | None = None, kv_budget_bytes: int | None = None,
         typical_len: int | None = None) -> SizingReport:
    """Predict what ``vllm serve`` will report for this model/GPU/flags combination.

    Defaults follow vLLM main as of Sep 2026 (verify): ``gpu_memory_utilization`` 0.92 (0.9 in
    older releases), ``block_size`` 16, ``max_num_batched_tokens`` 2048 and ``max_num_seqs`` 256
    for ``vllm serve`` on GPUs under 70 GiB. ``kv_budget_bytes`` bypasses the memory model (useful
    to pin the block arithmetic, or to replay vLLM's logged "Available KV cache memory")."""
    m = load_config(model)
    g = gpu(gpu_name) if gpu_name is not None else None
    if g is None and gpu_memory_bytes is None:
        raise ValueError("give a GPU name (see sizing.GPUS) or gpu_memory_bytes=")
    total = int(gpu_memory_bytes if gpu_memory_bytes is not None else g.memory_gib * GiB)
    max_len = int(max_model_len or m.max_position_embeddings)
    requested = math.ceil(total * gpu_memory_utilization)
    w = int(weight_bytes(m, dtype, quantization, tensor_parallel_size))
    over = overhead_bytes if overhead_bytes is not None else overhead_estimate(
        m, dtype=dtype, max_num_batched_tokens=max_num_batched_tokens, max_num_seqs=max_num_seqs,
        tensor_parallel_size=tensor_parallel_size, enforce_eager=enforce_eager)
    budget = int(kv_budget_bytes if kv_budget_bytes is not None else requested - w - sum(over.values()))
    per_tok = kv_bytes_per_token(m, kv_cache_dtype, dtype, tensor_parallel_size)
    per_block = block_size * per_tok
    blocks = max(0, budget // per_block)
    per_req = math.ceil(max_len / block_size)
    conc = blocks / per_req
    notes = []
    if g is not None and not g.bf16 and dtype_bytes(dtype, m) == 2 and dtype in ("auto", "bfloat16"):
        notes.append(f"{g.name} has no bfloat16: serve with --dtype half (float16), same 2 bytes")
    if g is not None and (quantization == "fp8" or str(kv_cache_dtype).startswith("fp8")) and not g.fp8_tflops:
        notes.append(f"{g.name} has no FP8 tensor cores: fp8 weights/KV save memory only if the "
                     "engine supports a fallback kernel for this GPU (verify)")
    if m.num_kv_heads < tensor_parallel_size:
        notes.append(f"{m.num_kv_heads} KV heads < TP {tensor_parallel_size}: KV heads are replicated")
    if max_len > m.max_position_embeddings > 0:
        notes.append(f"max_model_len {max_len} exceeds max_position_embeddings {m.max_position_embeddings}")
    return SizingReport(
        model=m.name, gpu=g.name if g else f"{total / GiB:.1f} GiB GPU", total_bytes=total,
        gpu_memory_utilization=gpu_memory_utilization, requested_bytes=requested, weights_bytes=w,
        overhead_bytes=over, kv_budget_bytes=budget, kv_bytes_per_token=per_tok, block_size=block_size,
        bytes_per_block=per_block, num_blocks=blocks, max_model_len=max_len, blocks_per_request=per_req,
        max_concurrency=conc, kv_cache_tokens=int(conc * max_len), fits=blocks >= per_req,
        estimated_max_model_len=min(blocks * block_size, m.max_position_embeddings or blocks * block_size),
        typical_len=typical_len,
        concurrency_at_typical_len=(blocks / math.ceil(typical_len / block_size)) if typical_len else None,
        notes=notes,
    )


def max_model_len_for(model, gpu_name=None, concurrency: int = 1, **kw) -> int:
    """The largest ``max_model_len`` (a multiple of block_size) at which ``concurrency`` requests
    of that length fit in the KV cache at once."""
    r = size(model, gpu_name, **kw)
    return (r.num_blocks // max(1, concurrency)) * r.block_size


# ---------------------------------------------------------------------------------------------
# Calibrate against what vLLM actually printed
# ---------------------------------------------------------------------------------------------
_LOG_PATTERNS = {
    "model_loading_gib": r"Model loading took ([\d.]+) GiB",
    "available_kv_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "kv_cache_tokens": r"KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
    "max_model_len": r"Maximum concurrency for ([\d,]+) tokens per request",
    "init_engine_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s",
}


def parse_startup_log(text: str) -> dict:
    """Pull the capacity lines out of a ``vllm serve`` log (vLLM >= 0.10 wording; verify)."""
    out = {}
    for key, pat in _LOG_PATTERNS.items():
        mt = re.search(pat, text)
        if mt:
            v = mt.group(1).replace(",", "")
            out[key] = int(v) if key in ("kv_cache_tokens", "max_model_len") else float(v)
    return out


def calibrate(report: SizingReport, log: dict) -> dict:
    """Compare a prediction with a parsed log; return the overhead vLLM actually charged.

    ``implied_overhead = requested - logged_weights - logged_available_kv``: plug it back in as
    ``overhead_bytes={"measured": ...}`` and the next prediction for this GPU/model is exact."""
    res = {"predicted_kv_gib": report.kv_budget_bytes / GiB, "predicted_kv_tokens": report.kv_cache_tokens}
    if "available_kv_gib" in log:
        res["logged_kv_gib"] = log["available_kv_gib"]
        res["kv_error_gib"] = res["predicted_kv_gib"] - log["available_kv_gib"]
        weights = log.get("model_loading_gib", report.weights_bytes / GiB)
        res["implied_overhead_gib"] = report.requested_bytes / GiB - weights - log["available_kv_gib"]
    if "kv_cache_tokens" in log:
        res["logged_kv_tokens"] = log["kv_cache_tokens"]
    return res
