"""model.py — a tiny MoE transformer in torch, written the way the real ones are.

One idea: an MoE layer is a router plus E ordinary MLPs; everything interesting is in the router.
``TinyTopKRouter`` scores experts (softmax or sigmoid), picks the top-k — optionally with a
per-expert bias that only *chooses* (DeepSeek-V3's auxiliary-loss-free balancing) — and returns
``(logits, scores, indices)``, the layout of the Hugging Face v5 routers, so ``moelab.hooks`` records
it exactly as it would record OLMoE or Mixtral. ``TinyMoE`` stores its experts as 3D tensors
(``gate_up`` ``[E, 2I, d]``, ``down`` ``[E, d, I]``, as transformers v5 does), sorts the token-expert
assignments by expert, runs one matmul per expert over its rows and scatters the weighted results
back: the gather -> grouped GEMM -> scatter shape of every fused MoE kernel.

Import cost: this module needs torch (CPU is enough); ``moelab.tinymoe`` imports it lazily.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class TinyTopKRouter(nn.Module):
    """Linear scores -> softmax/sigmoid -> top-k -> gate weights.

    ``renormalize``: divide the k weights by their sum (Mixtral always; Qwen3/OLMoE only with
    ``norm_topk_prob``); with k = 1 and no renormalisation the weight is the raw probability p_e —
    the Switch Transformer form, which is what lets the router learn at top-1. ``bias`` (a buffer)
    is added to the scores for the *choice* only; the weights come from the unbiased scores."""

    def __init__(self, d_model: int, n_experts: int, top_k: int, score: str = "softmax",
                 renormalize: bool | None = None, init_std: float = 0.02):
        super().__init__()
        self.n_experts, self.top_k, self.score = n_experts, top_k, score
        self.renormalize = (top_k > 1) if renormalize is None else renormalize
        self.weight = nn.Parameter(torch.randn(n_experts, d_model) * init_std)
        self.register_buffer("bias", torch.zeros(n_experts))

    def forward(self, x: torch.Tensor):
        logits = F.linear(x, self.weight)                                 # [N, E]
        scores = logits.softmax(-1) if self.score == "softmax" else logits.sigmoid()
        indices = (scores + self.bias).topk(self.top_k, dim=-1).indices    # the bias only chooses
        weights = scores.gather(-1, indices)
        if self.renormalize:
            weights = weights / weights.sum(-1, keepdim=True)
        return logits, weights, indices


class TinyMoE(nn.Module):
    """E SwiGLU experts behind a ``TinyTopKRouter``; optional always-on shared expert."""

    def __init__(self, d_model: int, n_experts: int, top_k: int, expert_ff: int, shared_ff: int = 0, **router_kw):
        super().__init__()
        self.n_experts, self.top_k, self.expert_ff = n_experts, top_k, expert_ff
        self.router = TinyTopKRouter(d_model, n_experts, top_k, **router_kw)
        self.gate_up = nn.Parameter(torch.randn(n_experts, 2 * expert_ff, d_model) / math.sqrt(d_model))
        self.down = nn.Parameter(torch.randn(n_experts, d_model, expert_ff) / math.sqrt(expert_ff))
        self.shared = None
        if shared_ff:
            self.shared = nn.ModuleDict({"gate_up": nn.Linear(d_model, 2 * shared_ff, bias=False),
                                         "down": nn.Linear(shared_ff, d_model, bias=False)})
        self.last = {}

    def expert(self, e: int, x: torch.Tensor) -> torch.Tensor:
        g, u = F.linear(x, self.gate_up[e]).chunk(2, dim=-1)
        return F.linear(F.silu(g) * u, self.down[e])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        shape = h.shape
        x = h.reshape(-1, shape[-1])                                       # [N, d]
        logits, weights, indices = self.router(x)
        flat = indices.reshape(-1)                                         # N*k assignments
        order = flat.argsort(stable=True)                                  # sort by expert
        counts = torch.bincount(flat, minlength=self.n_experts)
        token_of = order // self.top_k                                     # which token each row is
        out = torch.zeros_like(x)
        start = 0
        for e, n in enumerate(counts.tolist()):                            # one GEMM per expert
            if n:
                rows = order[start:start + n]
                y = self.expert(e, x[token_of[start:start + n]])
                out.index_add_(0, token_of[start:start + n], y * weights.reshape(-1)[rows, None])
                start += n
        if self.shared is not None:
            g, u = self.shared["gate_up"](x).chunk(2, dim=-1)
            out = out + self.shared["down"](F.silu(g) * u)
        self.last = {"logits": logits, "weights": weights, "indices": indices, "counts": counts}
        return out.reshape(shape)

    def params(self) -> tuple[int, int]:
        """(total, active) parameters of this layer: router + E experts (+ shared) vs router + k."""
        expert = 3 * self.gate_up.shape[2] * self.expert_ff
        shared = sum(p.numel() for p in self.shared.parameters()) if self.shared is not None else 0
        router = self.router.weight.numel()
        return router + self.n_experts * expert + shared, router + self.top_k * expert + shared


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 2):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, d // self.h).transpose(1, 3).unbind(2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(b, t, d))


class TinyMoETransformer(nn.Module):
    """Embedding -> [attention -> MoE] x layers -> LM head, pre-norm residual blocks."""

    def __init__(self, vocab: int, seq_len: int, d_model: int = 32, layers: int = 1, n_experts: int = 8,
                 top_k: int = 1, expert_ff: int = 16, shared_ff: int = 0, **router_kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({"n1": nn.LayerNorm(d_model), "attn": CausalSelfAttention(d_model),
                                              "n2": nn.LayerNorm(d_model),
                                              "moe": TinyMoE(d_model, n_experts, top_k, expert_ff, shared_ff, **router_kw)}))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    @property
    def moes(self) -> list[TinyMoE]:
        return [b["moe"] for b in self.blocks]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.emb(tokens) + self.pos[: tokens.shape[1]]
        for b in self.blocks:
            h = h + b["attn"](b["n1"](h))
            h = h + b["moe"](b["n2"](h))
        return self.head(self.norm(h))


def switch_aux_loss(logits: torch.Tensor, indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    """The Switch/GShard balance loss as transformers' ``load_balancing_loss_func`` computes it:
    E x sum_e f_e P_e, f_e = share of top-k assignments per token (sums to k), P_e = mean router
    probability. Uniform routing gives k (HF's normalisation; Megatron and MegaBlocks divide by k).
    Only P carries gradient — f comes from a top-k."""
    probs = logits.float().softmax(-1)
    f = torch.bincount(indices.reshape(-1), minlength=n_experts).float() / logits.shape[0]
    return n_experts * (f * probs.mean(0)).sum()


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """mean(logsumexp(logits)^2): keeps router logits small (MegaBlocks, Megatron; 1e-3 is typical)."""
    return torch.logsumexp(logits.float(), dim=-1).square().mean()
