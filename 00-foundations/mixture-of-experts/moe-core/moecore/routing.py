"""Routing and load balance: why a learned router collapses, and the ways to stop it.

The one idea: routing is a feedback loop. An expert that wins tokens is trained on them, gets
better at them and wins more; left alone a few experts take everything ("router collapse") and
the rest are dead weight in HBM. Balance must be imposed, and every mechanism here is one way:

  aux loss      E * sum_e f_e * P_e (Switch/GShard): f = share of assignments, P = mean router
                probability; smallest when both are uniform. Two normalisations exist (see below).
  z-loss        mean(logsumexp(logits)^2): keeps logits small so a bf16 softmax stays accurate.
  capacity      at most factor * k * T / E rows per expert; the overflow is dropped (those tokens
                ride the residual). Dropless (MegaBlocks, every inference engine) pads instead.
  bias          a per-expert bias on the *selection* scores, stepped by a fixed rate against the
                load (DeepSeek-V3's auxiliary-loss-free balancing); weights never see it.
  expert choice each expert picks its top-C tokens: balanced by construction, but a token may get
                0 or many experts, and it needs the whole batch (not causal-safe for decode).
"""
from __future__ import annotations

import numpy as np

from .moe import topk


def load(idx: np.ndarray, n_experts: int) -> np.ndarray:
    """Assignments per expert (a token with k experts counts k times in total)."""
    return np.bincount(np.asarray(idx).reshape(-1), minlength=n_experts)


def switch_aux_loss(probs: np.ndarray, idx: np.ndarray, convention: str = "hf") -> float:
    """E * sum_e f_e * P_e over T tokens.

    "hf" (transformers' load_balancing_loss_func): f_e = assignments_e / T, so sum f = k and a
    perfectly uniform router scores k (2 for Mixtral). "megatron" (and MegaBlocks) divides f by k:
    uniform scores 1. Only P carries a gradient; f comes from a top-k.
    """
    t, e = probs.shape
    f = load(idx, e) / t
    if convention == "megatron":
        f = f / idx.shape[1]
    return float(e * np.sum(f * probs.mean(axis=0)))


def switch_aux_grad(probs: np.ndarray, idx: np.ndarray, convention: str = "hf") -> np.ndarray:
    """d(aux)/d(probs[t, e]) = E * f_e / T: the loss pushes down the probability of loaded experts."""
    t, e = probs.shape
    f = load(idx, e) / t / (idx.shape[1] if convention == "megatron" else 1)
    return np.broadcast_to(e * f / t, probs.shape).copy()


def sequence_aux_loss(probs: np.ndarray, idx: np.ndarray, seq_len: int) -> float:
    """The same balance loss scored per sequence and averaged (Megatron "seq_aux_loss", DeepSeek-V2/V3):
    a batch can be balanced on average while each sequence sends all its tokens to one expert."""
    n = probs.shape[0] // seq_len
    return float(np.mean([switch_aux_loss(probs[i * seq_len:(i + 1) * seq_len], idx[i * seq_len:(i + 1) * seq_len],
                                          "megatron") for i in range(n)]))


def z_loss(logits: np.ndarray) -> float:
    """mean over tokens of logsumexp(logits)^2 (MegaBlocks/Megatron; weight ~1e-3)."""
    m = logits.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True)))[:, 0]
    return float(np.mean(lse ** 2))


def capacity(tokens: int, n_experts: int, k: int, factor: float) -> int:
    """Rows one expert may take: int(factor * k * T / E) (MegaBlocks' expert_capacity, one rank)."""
    return int(factor * k * tokens / n_experts)


def apply_capacity(idx: np.ndarray, weights: np.ndarray, cap: int, policy: str = "position") -> np.ndarray:
    """Boolean keep-mask [T, k]. Over-capacity assignments are dropped: "position" drops the tail of
    the batch (later tokens lose), "probs" drops the lowest-weight assignments (Megatron's policies)."""
    t, k = idx.shape
    flat, w = idx.reshape(-1), weights.reshape(-1)
    order = np.arange(t * k) if policy == "position" else np.argsort(-w, kind="stable")
    keep, used = np.zeros(t * k, dtype=bool), np.zeros(idx.max() + 1, dtype=int)
    for a in order:
        if used[flat[a]] < cap:
            keep[a], used[flat[a]] = True, used[flat[a]] + 1
    return keep.reshape(t, k)


def align_block_size(idx: np.ndarray, block: int, n_experts: int):
    """Dropless, as vLLM's moe_align_block_size does it: sort the T*k assignment slots by expert and
    pad each expert's segment to a multiple of `block` with the pad id T*k, so a block GEMM never
    mixes experts. Returns (sorted slot ids, expert id per block, total padded rows)."""
    flat = np.asarray(idx).reshape(-1)
    pad, sorted_ids, block_expert = flat.size, [], []
    for e in range(n_experts):
        slots = list(np.flatnonzero(flat == e))
        n = -(-len(slots) // block) * block
        sorted_ids += slots + [pad] * (n - len(slots))
        block_expert += [e] * (n // block)
    return np.array(sorted_ids), np.array(block_expert), len(sorted_ids)


def update_bias(bias: np.ndarray, tokens_per_expert: np.ndarray, rate: float = 1e-3) -> np.ndarray:
    """Aux-loss-free balancing (Megatron get_updated_expert_bias): step each expert's selection bias
    by `rate` toward the mean load -- sign(total - load * E): under-loaded up, over-loaded down."""
    t = np.asarray(tokens_per_expert)
    return bias + rate * np.sign(t.sum() - t * t.size)


def expert_choice(probs: np.ndarray, cap: int) -> np.ndarray:
    """Expert-choice routing: each expert takes its top-`cap` tokens. Returns a [T, E] 0/1 matrix;
    column sums are exactly `cap`, row sums (experts per token) vary from 0 up."""
    chosen = topk(probs.T, cap)                      # [E, cap] token ids
    a = np.zeros_like(probs, dtype=int)
    np.put_along_axis(a.T, chosen, 1, axis=1)
    return a


def stats(tokens_per_expert: np.ndarray) -> dict:
    """Utilisation of one layer. balancedness = mean / max, the number vLLM's EPLB logs (1 = perfect)."""
    t = np.asarray(tokens_per_expert, dtype=float)
    share = t / t.sum()
    nz = share[share > 0]
    return dict(max_over_mean=t.max() / t.mean(), balancedness=t.mean() / t.max(),
                cv=t.std() / t.mean(), dead=int((t == 0).sum()), top_share=share.max(),
                entropy=float(-(nz * np.log(nz)).sum() / np.log(t.size)))   # 1 = uniform
