"""LLM inference on the roofline: the FLOPs and bytes of one engine step.

The one idea: every step streams the weights from HBM once, whatever the number
of tokens in it, while FLOPs grow with the tokens. So a step's intensity is
roughly its token count (at 2-byte weights): prefill of a 2K prompt is far above
the ridge (compute-bound), a batch-1 decode step is at ~1 FLOP/B (memory-bound),
and batching B decode sequences moves it right -- until each sequence's KV-cache
reads, which grow with context exactly as fast as its FLOPs do, cap the
intensity. Quantization changes bytes; MoE changes which weights are streamed.

Model of one step (all layers, one forward pass):
  FLOPs = 2 x tokens x (per-token matmul params) + 2 x logits_rows x vocab x d
          + attention scores/values 4 x L x heads x head_dim x (positions attended)
  bytes = weights streamed once (+ LM head) + KV read + KV written
Activations are assumed fused/on-chip (not counted); efficiencies default to 1
(the ideal roofline). Numbers are bounds, not predictions.

prefill() and decode() treat the whole step as one kernel, max(sum F / peak,
sum B / BW): the bound if GEMM math overlapped attention's KV streaming perfectly.
An engine runs the kernels one after another, so decode_split() prices the weight
GEMMs (intensity ~ batch) and attention (intensity ~ 2 x GQA group / KV bytes,
never compute-bound) separately and adds them. The two agree while both kernels
are memory-bound and part once the GEMMs cross the ridge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .specs import Device


@dataclass(frozen=True)
class ModelConfig:
    """The handful of config.json fields that set FLOPs and bytes."""
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int                  # MLP width (per expert for MoE)
    vocab: int
    gated_mlp: bool = True     # SwiGLU: up, gate, down = 3 matrices; plain MLP = 2
    tied_embeddings: bool = False
    n_experts: int = 0         # 0 = dense
    top_k: int = 0             # experts per token (MoE)

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    def attn_params(self) -> int:
        """Per layer: W_q (d x h dh), W_k and W_v (d x kv dh each), W_o (h dh x d)."""
        q, kv = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
        return 2 * self.d_model * q + 2 * self.d_model * kv

    def expert_params(self) -> int:
        """Per layer, one MLP (or one expert)."""
        return (3 if self.gated_mlp else 2) * self.d_model * self.d_ff

    def router_params(self) -> int:
        return self.d_model * self.n_experts

    def params(self) -> int:
        """Everything that must sit in memory."""
        mlp = self.expert_params() * max(1, self.n_experts) + self.router_params()
        emb = self.vocab * self.d_model * (1 if self.tied_embeddings else 2)
        return self.n_layers * (self.attn_params() + mlp) + emb

    def active_params(self) -> int:
        """Parameters one token touches (dense: all of them)."""
        mlp = self.expert_params() * (self.top_k if self.is_moe else 1) + self.router_params()
        emb = self.vocab * self.d_model * (1 if self.tied_embeddings else 2)
        return self.n_layers * (self.attn_params() + mlp) + emb

    def matmul_params_per_token(self) -> int:
        """Params in the per-token matmuls, excluding the embedding gather and the LM head."""
        mlp = self.expert_params() * (self.top_k if self.is_moe else 1) + self.router_params()
        return self.n_layers * (self.attn_params() + mlp)

    def lm_head_params(self) -> int:
        return self.vocab * self.d_model

    def kv_bytes_per_token(self, kv_bytes: float = 2) -> float:
        """2 (K and V) x layers x kv_heads x head_dim x bytes. Llama-3.1-8B bf16: 131,072 B."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * kv_bytes


# Published configs (from each model's config.json; verify before relying on them).
PRESETS = {
    "qwen2.5-1.5b": ModelConfig("Qwen2.5-1.5B", 28, 1536, 12, 2, 128, 8960, 151_936, tied_embeddings=True),
    "llama-3.1-8b": ModelConfig("Llama-3.1-8B", 32, 4096, 32, 8, 128, 14_336, 128_256),
    "llama-3.1-70b": ModelConfig("Llama-3.1-70B", 80, 8192, 64, 8, 128, 28_672, 128_256),
    "mixtral-8x7b": ModelConfig("Mixtral-8x7B", 32, 4096, 32, 8, 128, 14_336, 32_000, n_experts=8, top_k=2),
    "qwen3-30b-a3b": ModelConfig("Qwen3-30B-A3B", 48, 2048, 32, 4, 128, 768, 151_936, n_experts=128, top_k=8),
}

# Precision recipes: bytes per weight, bytes per KV element, and which peak runs the math.
# Weight-only schemes (w8a16, w4a16) dequantize to bf16 and use the bf16 peak.
# INT4 group scales add ~2/group_size bytes per weight (~0.016 at group 128); ignored here.
SCHEMES = {
    "bf16": dict(weight_bytes=2, kv_bytes=2, precision="bf16"),
    "fp8": dict(weight_bytes=1, kv_bytes=1, precision="fp8"),
    "w8a16": dict(weight_bytes=1, kv_bytes=2, precision="bf16"),
    "w4a16": dict(weight_bytes=0.5, kv_bytes=2, precision="bf16"),
    "w4a16-kv8": dict(weight_bytes=0.5, kv_bytes=1, precision="bf16"),
}


def experts_touched(n_experts: int, top_k: int, tokens: float) -> float:
    """Expected distinct experts one layer reads for `tokens` tokens, uniform routing.

    E (1 - (1 - k/E)^T): Mixtral (8, top-2) touches 2 at T=1 and 7.92 at T=16.
    """
    if n_experts == 0:
        return 1.0
    return n_experts * (1 - (1 - top_k / n_experts) ** tokens)


def streamed_weight_bytes(model: ModelConfig, tokens: int, weight_bytes: float = 2) -> float:
    """Weight bytes one step reads: every layer once (MoE: only the experts hit) + LM head.

    The input embedding is a gather of `tokens` rows, not a stream of the table.
    """
    experts = experts_touched(model.n_experts, model.top_k, tokens) if model.is_moe else 1
    per_layer = model.attn_params() + experts * model.expert_params() + model.router_params()
    return (model.n_layers * per_layer + model.lm_head_params() + tokens * model.d_model) * weight_bytes


@dataclass(frozen=True)
class Step:
    phase: str
    tokens: int
    flops: float
    bytes: float
    t_compute: float
    t_memory: float

    @property
    def time(self) -> float:
        return max(self.t_compute, self.t_memory)

    @property
    def bound(self) -> str:
        return "compute" if self.t_compute >= self.t_memory else "memory"

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.time


def _step(phase, tokens, flops, bytes_, device, precision, compute_eff, memory_eff) -> Step:
    return Step(phase, tokens, flops, bytes_,
                flops / (device.peak(precision) * compute_eff),
                bytes_ / (device.bandwidth() * memory_eff))


def prefill(model: ModelConfig, device: Device, prompt_len: int, batch: int = 1, *,
            weight_bytes: float = 2, kv_bytes: float = 2, precision: str = "bf16",
            compute_eff: float = 1.0, memory_eff: float = 1.0) -> Step:
    """One step that ingests `batch` prompts of `prompt_len` tokens (no cached prefix).

    Causal attention: token i attends to i positions, so a prompt of S costs
    sum 4 L h dh i = 2 L h dh S (S + 1). Logits only for each prompt's last token.
    """
    t = batch * prompt_len
    attn = batch * 2 * model.n_layers * model.n_heads * model.head_dim * prompt_len * (prompt_len + 1)
    flops = 2 * t * model.matmul_params_per_token() + 2 * batch * model.lm_head_params() + attn
    bytes_ = streamed_weight_bytes(model, t, weight_bytes) + t * model.kv_bytes_per_token(kv_bytes)
    return _step("prefill", t, flops, bytes_, device, precision, compute_eff, memory_eff)


def decode(model: ModelConfig, device: Device, batch: int, context: int, *,
           weight_bytes: float = 2, kv_bytes: float = 2, precision: str = "bf16",
           compute_eff: float = 1.0, memory_eff: float = 1.0) -> Step:
    """One decode step: `batch` sequences, each with `context` tokens already cached.

    Each new token reads its sequence's whole KV cache and writes one entry,
    and attends to context + 1 positions.
    """
    kv = batch * (context + 1) * model.kv_bytes_per_token(kv_bytes)
    bytes_ = streamed_weight_bytes(model, batch, weight_bytes) + kv
    return _step("decode", batch, batch * _per_token_flops(model, context), bytes_, device,
                 precision, compute_eff, memory_eff)


@dataclass(frozen=True)
class SplitStep:
    """A decode step as the two kinds of kernel an engine runs one after another."""
    gemms: Step          # weight GEMMs + LM head: stream the weights, intensity ~ batch
    attention: Step      # each sequence reads its own KV cache: intensity independent of batch

    @property
    def time(self) -> float:
        """Kernels run in sequence: the sum of per-kernel roofline times."""
        return self.gemms.time + self.attention.time


def decode_split(model: ModelConfig, device: Device, batch: int, context: int, *,
                 weight_bytes: float = 2, kv_bytes: float = 2, precision: str = "bf16",
                 compute_eff: float = 1.0, memory_eff: float = 1.0) -> SplitStep:
    """decode() priced per kernel: the same FLOPs and bytes, split in two.

    GEMMs: batch x (2 matmul params + 2 LM head) FLOPs over the streamed weights, so
    their intensity grows with batch at any context and crosses the ridge near
    batch ~ ridge x weight_bytes / 2. Attention: 4 L h dh (c + 1) FLOPs over (c + 1)
    KV entries per sequence, ~2 x (heads / kv_heads) / kv_bytes = 4 FLOP/B for
    Llama-3.1-8B in bf16, whatever the batch. Llama-3.1-8B, H100, FP8, 2K context,
    batch 400: GEMMs 3.03 ms compute-bound + attention 16.03 ms memory-bound = 19.07 ms,
    where the one-kernel decode() says 18.27 ms, memory-bound.
    """
    gemm_flops = batch * (2 * model.matmul_params_per_token() + 2 * model.lm_head_params())
    attn_flops = batch * 4 * model.n_layers * model.n_heads * model.head_dim * (context + 1)
    kv = batch * (context + 1) * model.kv_bytes_per_token(kv_bytes)
    args = (device, precision, compute_eff, memory_eff)
    return SplitStep(_step("decode GEMMs", batch, gemm_flops, streamed_weight_bytes(model, batch, weight_bytes), *args),
                     _step("decode attention", batch, attn_flops, kv, *args))


def gemm_crossover_batch(model: ModelConfig, device: Device, max_batch: int = 1 << 20, **kw) -> int | None:
    """Smallest batch at which decode's weight GEMMs are compute-bound (context does not matter).

    Llama-3.1-8B on an H100: 296 in bf16 and in FP8 (half the bytes, twice the peak).
    This is the planning rule 'decode turns compute-bound at batch ~ ridge'.
    """
    def compute_bound(b):
        return decode_split(model, device, b, 0, **kw).gemms.bound == "compute"

    hi = 1
    while not compute_bound(hi):
        hi *= 2
        if hi > max_batch:
            return None
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if compute_bound(mid) else (mid, hi)
    return hi


def decode_crossover_batch(model: ModelConfig, device: Device, context: int,
                           max_batch: int = 1 << 20, **kw) -> int | None:
    """Smallest batch at which a whole decode step (one-kernel view) is compute-bound, or None.

    This blends GEMMs and attention into one average intensity, so it rises with
    context and vanishes past ~392 tokens for Llama-3.1-8B on an H100; per kernel,
    the GEMMs still cross at gemm_crossover_batch() whatever the context.
    Dense closed form: B* = ridge W / (F_tok - ridge (c+1) kv_tok). When each
    sequence's KV bytes x ridge exceed its FLOPs the denominator is <= 0: batching
    can never reach the ridge. t_compute - t_memory crosses zero once, so bisect.
    """
    def compute_bound(b):
        return decode(model, device, b, context, **kw).bound == "compute"

    hi = 1
    while not compute_bound(hi):
        hi *= 2
        if hi > max_batch:
            return None
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if compute_bound(mid) else (mid, hi)
    return hi


def _per_token_flops(model: ModelConfig, context: int) -> float:
    return (2 * model.matmul_params_per_token() + 2 * model.lm_head_params()
            + 4 * model.n_layers * model.n_heads * model.head_dim * (context + 1))


def decode_intensity_limit(model: ModelConfig, context: int, kv_bytes: float = 2) -> float:
    """Decode intensity as batch -> infinity: weights amortize away, KV reads do not.

    F_tok / ((c + 1) kv_tok). Llama-3.1-8B at 4K context, bf16 KV: ~32 FLOP/B,
    an order of magnitude below an H100's ridge -- no batch size fixes that.
    """
    return _per_token_flops(model, context) / ((context + 1) * model.kv_bytes_per_token(kv_bytes))


def max_context_for_compute_bound(model: ModelConfig, device: Device, *, kv_bytes: float = 2,
                                  precision: str = "bf16") -> float:
    """Largest context at which decode can still reach the ridge with a big enough batch.

    Solve F_tok(c) = ridge (c + 1) kv_tok for c. Returns inf if attention alone is
    above the ridge (never, for GQA models at 1-2 byte KV).
    """
    ridge = device.ridge(precision)
    fixed = 2 * model.matmul_params_per_token() + 2 * model.lm_head_params()
    per_pos = ridge * model.kv_bytes_per_token(kv_bytes) - 4 * model.n_layers * model.n_heads * model.head_dim
    return math.inf if per_pos <= 0 else fixed / per_pos - 1


def max_batch_by_memory(model: ModelConfig, device: Device, context: int, *,
                        weight_bytes: float = 2, kv_bytes: float = 2, reserve: float = 0.10,
                        n_devices: int = 1) -> int:
    """Sequences of `context` tokens whose KV fits beside the weights (reserve = headroom)."""
    free = n_devices * device.memory_gb * 1e9 * (1 - reserve) - model.params() * weight_bytes
    return max(0, math.floor(free / (context * model.kv_bytes_per_token(kv_bytes))))


def best_batch_under_itl(model: ModelConfig, device: Device, context: int, itl_s: float,
                         **kw) -> int:
    """Largest batch whose decode step fits both the ITL budget and memory (0 if none)."""
    mem_kw = {k: v for k, v in kw.items() if k in ("weight_bytes", "kv_bytes")}
    cap = max_batch_by_memory(model, device, context, **mem_kw)
    best = 0
    for b in range(1, cap + 1):
        if decode(model, device, b, context, **kw).time > itl_s:
            break
        best = b
    return best
