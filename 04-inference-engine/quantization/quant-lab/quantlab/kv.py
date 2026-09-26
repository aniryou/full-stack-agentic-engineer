"""kv.py — ``--kv-cache-dtype``: what it buys (sessions, long-context decode), what it needs, what it costs.

One idea: the KV cache is the other big tensor. It costs ``2 x layers x kv_heads x head_dim x bytes``
per token, vLLM turns everything left after weights and overheads into blocks of it, and decode at
long context reads all of it every step. FP8 KV halves the bytes: about twice the sessions in the
same memory and a cheaper KV read per step — *if* the attention backend on your GPU can read FP8
(vLLM v0.30.0 / main, Sep 2026, verify): no backend on a T4 (Triton needs sm_89 for native FP8,
FlashInfer and FlashAttention need sm_80); FlashInfer on an A100 or L4 (FA2 has no FP8 KV, so the
flag *changes the backend*); FlashAttention 3 on an H100 (which also quantizes Q); FlashInfer on
B200. Scales default to 1.0 unless the checkpoint carries calibrated ``k_scale``/``v_scale``.

The sizing reproduces ``servelab.sizing`` (the serving lab's memory model: 0.92 of the driver's
total, profiled activations, CUDA graphs, non-torch memory; ``tests/test_kv.py`` pins its numbers:
2,363 -> 4,727 blocks for Llama-3.1-8B on an L4 with FP8 KV). ``kv_quantizer`` emulates each
KV dtype on the tiny model so the accuracy side can be measured too (PRIMER §6 "KV-cache quantization").
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import numerics as N
from .serve import GPU, gpu as _gpu

GiB = 1024**3
CONFIG_DIR = Path(__file__).parent / "data" / "configs"

KV_BYTES = {"auto": None, "bfloat16": 2, "float16": 2, "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
            # dynamic per-(token, head) scales: 1 or 0.5 byte per element + one fp32 scale per head_dim (verify)
            "int8_per_token_head": 1, "int4_per_token_head": 0.5, "fp8_per_token_head": 1,
            "nvfp4": 0.5 + 1 / 16}
# Weight bytes per *linear-layer* parameter, as the serving lab's sizing: embeddings, lm_head and norms stay 16-bit
WEIGHT_BYTES = {"bf16": None, "fp8": 1.0, "fp8-online": 1.0, "w8a8-int8": 1.0, "int8": 1.0,
                "w4a16": 0.5 + 2.5 / 128, "int4": 0.5 + 2.5 / 128, "nvfp4": 0.5 + 1 / 16}


# ---------------------------------------------------------------------------------------------
# Model shapes (config.json) and parameter counts
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Shape:
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
    qkv_bias: bool = False
    qk_norm: bool = False

    @classmethod
    def from_hf(cls, c: dict, name: str | None = None) -> "Shape":
        heads, hidden, mt = int(c["num_attention_heads"]), int(c["hidden_size"]), str(c.get("model_type", ""))
        return cls(name or c.get("_name_or_path") or mt, mt, int(c["num_hidden_layers"]), hidden, heads,
                   int(c.get("num_key_value_heads") or heads), int(c.get("head_dim") or hidden // heads),
                   int(c.get("intermediate_size") or 0), int(c["vocab_size"]), int(c.get("max_position_embeddings") or 0),
                   bool(c.get("tie_word_embeddings", False)), bool(c.get("attention_bias", mt in ("qwen2",))),
                   mt in ("qwen3",))


def bundled() -> list:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def load_shape(model) -> Shape:
    """A bundled config name (``"llama-3.1-8b-instruct"``), a path to a config.json, a dict or a Shape."""
    if isinstance(model, Shape):
        return model
    if isinstance(model, dict):
        return Shape.from_hf(model)
    p = Path(model)
    if not p.exists():
        p = CONFIG_DIR / f"{str(model).lower().split('/')[-1]}.json"
    cfg = json.loads(p.read_text())
    return Shape.from_hf(cfg, cfg.get("_name_or_path") or p.stem)


@dataclass(frozen=True)
class Params:
    total: int
    embedding: int     # embed_tokens + lm_head (once if tied): kept 16-bit by every recipe here
    linear: int        # q/k/v/o + MLP: what weight quantization shrinks


def params(m: Shape) -> Params:
    """Dense llama-family count (Llama-3.1-8B: 8,030,261,248; Qwen2.5-0.5B: 494,032,768)."""
    q = m.hidden_size * m.num_heads * m.head_dim
    kv = 2 * m.hidden_size * m.num_kv_heads * m.head_dim
    o = m.num_heads * m.head_dim * m.hidden_size
    bias = (m.num_heads + 2 * m.num_kv_heads) * m.head_dim if m.qkv_bias else 0
    qk_norm = 2 * m.head_dim if m.qk_norm else 0
    mlp = 3 * m.hidden_size * m.intermediate_size
    embed = m.vocab_size * m.hidden_size * (1 if m.tie_word_embeddings else 2)
    total = m.num_layers * (q + kv + o + bias + qk_norm + mlp + 2 * m.hidden_size) + m.hidden_size + embed
    return Params(total, embed, m.num_layers * (q + kv + o + mlp))


def weight_bytes(m: Shape, weights: str = "bf16") -> int:
    """Weights in bytes: the linear layers at the scheme's bytes per weight, everything else 16-bit."""
    p, wb = params(m), WEIGHT_BYTES[weights]
    return int(p.total * 2 if wb is None else p.linear * wb + (p.total - p.linear) * 2)


def kv_bytes_per_token(m: Shape, kv_cache_dtype: str = "auto", tensor_parallel_size: int = 1) -> float:
    """``2 x layers x kv_heads_per_gpu x head_dim x bytes`` (Llama-3.1-8B: 131,072 B bf16, 65,536 B fp8)."""
    b = KV_BYTES[kv_cache_dtype] or 2
    return 2 * m.num_layers * max(1, m.num_kv_heads // tensor_parallel_size) * m.head_dim * b


# ---------------------------------------------------------------------------------------------
# Blocks and sessions: the serving lab's memory model, re-derived
# ---------------------------------------------------------------------------------------------
def scheduler_defaults(g: GPU) -> tuple:
    """``(max_num_batched_tokens, max_num_seqs)`` vllm serve picks (v0.30.0): >= 160 GiB 16384/1024;
    >= 70 GiB and not an A100 8192/1024; otherwise 2048/256."""
    mem = g.memory_gib * GiB
    if mem >= 160 * GiB:
        return 16384, 1024
    if mem >= 70 * GiB and "A100" not in g.name.upper():
        return 8192, 1024
    return 2048, 256


def overheads(m: Shape, g: GPU, max_num_batched_tokens: int | None = None, max_num_seqs: int | None = None) -> dict:
    """Non-weight, non-KV memory (estimates): the profiling pass's activations and sampler logits,
    CUDA graphs (0.5 GiB from 3 B parameters, else 0.25), 0.2 GiB of non-torch buffers."""
    t, s = scheduler_defaults(g)
    t, s = max_num_batched_tokens or t, max_num_seqs or s
    act = t * (2 * m.intermediate_size + 4 * m.hidden_size) * 2 + s * m.vocab_size * 4 * 2
    return {"activations": int(act), "cuda_graphs": int((0.5 if params(m).total >= 3e9 else 0.25) * GiB),
            "non_torch": int(0.2 * GiB)}


@dataclass
class KVReport:
    model: str
    gpu: str
    weights: str
    kv_cache_dtype: str
    weight_bytes: int
    kv_budget_bytes: int
    kv_bytes_per_token: float
    num_blocks: int
    block_size: int
    max_model_len: int
    notes: list = field(default_factory=list)
    backend: str | None = None       # the attention backend vLLM would pick; None = no backend reads this KV dtype

    @property
    def kv_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def sessions(self, tokens_per_session: int) -> float:
        """Concurrent sessions of this length the blocks hold (vLLM's "Maximum concurrency" at that length)."""
        return self.num_blocks / math.ceil(tokens_per_session / self.block_size)

    def row(self, typical_len: int = 2000) -> str:
        kvd = self.kv_cache_dtype if self.backend else f"{self.kv_cache_dtype} (no backend on {self.gpu})"
        return (f"| {self.weights} | {kvd} | {self.weight_bytes / 1e9:.2f} GB | "
                f"{self.kv_bytes_per_token:,.0f} B | {self.num_blocks:,} | {self.sessions(typical_len):.1f} |")


def size(model, gpu_name, *, weights: str = "bf16", kv_cache_dtype: str = "auto", gpu_memory_utilization: float = 0.92,
         max_model_len: int | None = None, block_size: int = 16, max_num_batched_tokens: int | None = None,
         max_num_seqs: int | None = None) -> KVReport:
    """KV blocks vLLM would allocate: ``floor((util x total - weights - overheads) / (block x bytes/token))``."""
    m, g = load_shape(model), _gpu(gpu_name)
    total = int(g.memory_gib * GiB)
    wb = weight_bytes(m, weights)
    budget = math.ceil(total * gpu_memory_utilization) - wb - sum(overheads(m, g, max_num_batched_tokens, max_num_seqs).values())
    per_tok = kv_bytes_per_token(m, kv_cache_dtype)
    blocks = max(0, int(budget // (block_size * per_tok)))
    notes = []
    be = attention_backend(g, kv_cache_dtype)
    if be.error:
        notes.append(be.error)
    return KVReport(m.name, g.name, weights, kv_cache_dtype, wb, int(budget), per_tok, blocks, block_size,
                    int(max_model_len or m.max_position_embeddings), notes, be.backend)


def table(model="llama-3.1-8b-instruct", gpu_name="L4", typical_len: int = 2000,
          weights=("bf16", "fp8", "w4a16"), kv=("auto", "fp8")) -> str:
    rows = [f"| weights | kv cache | weight bytes | KV / token | blocks | sessions @ {typical_len:,} |",
            "|---|---|---|---|---|---|"]
    rows += [size(model, gpu_name, weights=w, kv_cache_dtype=k).row(typical_len) for w in weights for k in kv]
    return "\n".join(rows)


# ---------------------------------------------------------------------------------------------
# Which attention backend reads which KV dtype (vLLM v0.30.0 / main, Sep 2026 — verify)
# ---------------------------------------------------------------------------------------------
@dataclass
class Backend:
    backend: str | None
    error: str | None = None


FA_KV = {"auto", "float16", "bfloat16", "fp8", "fp8_e4m3"}                         # flash_attn.py
FLASHINFER_KV = FA_KV | {"fp8_e5m2", "nvfp4"}                                     # flashinfer.py (nvfp4: SM100 family)
TRITON_KV = {"auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2",
             "int8_per_token_head", "int4_per_token_head", "fp8_per_token_head"}   # triton_attn.py


def attention_backend(gpu_name, kv_cache_dtype: str = "auto") -> Backend:
    """The first backend in vLLM's priority list that accepts this KV dtype on this GPU."""
    g = _gpu(gpu_name)
    sm, kv = g.sm, kv_cache_dtype
    fp8 = kv.startswith("fp8")
    if sm < 80:
        order = ["TRITON_ATTN"]                          # FA needs sm_80; FlashInfer is floored at sm_80 (broken on sm_75)
    elif 100 <= sm < 110:
        order = ["FLASHINFER", "FLASH_ATTN", "TRITON_ATTN"]
    else:
        order = ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"]
    for be in order:
        if be == "FLASH_ATTN" and sm >= 80 and kv in FA_KV:
            if fp8 and sm not in (90, 100, 103):        # FP8 KV only with FA3 (sm_90) or FA4 (sm_10x)
                continue
            return Backend("FLASH_ATTN (FA3)" if sm == 90 else "FLASH_ATTN")
        if be == "FLASHINFER" and 80 <= sm <= 121 and kv in FLASHINFER_KV:
            if kv == "nvfp4" and not 100 <= sm < 110:
                continue
            return Backend("FLASHINFER")
        if be == "TRITON_ATTN" and kv in TRITON_KV:
            if kv == "bfloat16" and sm < 80:
                return Backend(None, "bfloat16 KV cache needs sm_80+ (T4: use --dtype half)")
            if fp8 and kv != "fp8_per_token_head" and sm < 89:
                continue
            return Backend("TRITON_ATTN")
    if fp8 and sm < 80:
        return Backend(None, f"FP8 KV cache is not supported on {g.name} (sm_{sm}): Triton needs sm_89 for native "
                             "FP8 (fp8e4nv); FlashAttention and FlashInfer need sm_80")
    return Backend(None, f"no attention backend accepts --kv-cache-dtype {kv} on {g.name} (sm_{sm}) (verify)")


# ---------------------------------------------------------------------------------------------
# Emulating the KV dtypes on the tiny model
# ---------------------------------------------------------------------------------------------
def kv_quantizer(kv_cache_dtype: str, k_scale: float | dict = 1.0, v_scale: float | dict = 1.0):
    """``(kind, layer, x) -> x_hat`` for ``TinyLM.forward(kv_quant=...)``, ``x`` shaped
    ``[batch, kv_heads, tokens, head_dim]``. FP8 divides by the per-tensor scale (vLLM's default is
    1.0; a dict gives per-layer calibrated scales), rounds to E4M3/E5M2 and multiplies back.
    ``*_per_token_head`` computes a symmetric scale per (token, head) at run time."""
    if kv_cache_dtype in ("auto", "bfloat16", "float16"):
        return None
    fmt = "e5m2" if kv_cache_dtype == "fp8_e5m2" else "e4m3"
    top = N.E5M2.max_finite if fmt == "e5m2" else N.E4M3.max_finite

    def per_tensor(kind, layer, x):
        s = k_scale if kind == "k" else v_scale
        s = s[layer] if isinstance(s, dict) else s
        return N.minifloat_round(np.clip(x / s, -top, top), fmt) * s

    def per_token_head(kind, layer, x):
        bits = {"int8_per_token_head": 8, "int4_per_token_head": 4}.get(kv_cache_dtype)
        amax = np.maximum(np.abs(x).max(-1, keepdims=True), 1e-12)
        if bits is None:                                   # fp8_per_token_head
            s = amax / 448.0
            return N.minifloat_round(x / s, "e4m3") * s
        qmax = (1 << (bits - 1)) - 1
        s = amax / qmax
        return np.clip(np.round(x / s), -qmax - 1, qmax) * s
    return per_token_head if kv_cache_dtype.endswith("per_token_head") else per_tensor


def calibrate_kv_scales(model, prompts) -> tuple:
    """Per-layer ``k_scale, v_scale = amax / 448`` over ``prompts`` — what llm-compressor's
    ``kv_cache_scheme`` stores (static, per tensor) so FP8 uses its whole range."""
    amax: dict = {}

    def spy(kind, layer, x):
        amax[(kind, layer)] = max(amax.get((kind, layer), 0.0), float(np.abs(x).max()))
        return x
    model.forward(prompts, kv_quant=spy)
    ks = {i: amax[("k", i)] / 448.0 for i in range(model.layers)}
    vs = {i: amax[("v", i)] / 448.0 for i in range(model.layers)}
    return ks, vs
