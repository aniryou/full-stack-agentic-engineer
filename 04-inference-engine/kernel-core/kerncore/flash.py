"""Tiled attention with the online softmax, the way a FlashAttention-2 forward schedules it.

One idea: attention can be computed one (block_m x block_n) tile of scores at a time, carrying a running
(max m, sum l, unnormalized output o) per query row and rescaling by alpha = exp(m_old - m_new) whenever the
max moves. The result is exact -- the same O and log-sum-exp as materializing the whole score matrix -- and
under a causal mask whole tiles above the diagonal are never loaded.

`flash_attention` counts what it does (tiles visited and masked, bytes a 16-bit kernel with this schedule
would move) so the counts can be checked against ../flash-attention/fa_calculators.py, the cost model the
FlashAttention deep dive uses (imported, not copied: `fa` below). It also exposes the loop-order detail of
the deep dive §11.2: walking the key tiles forward from key 0 means every row's first tile holds a key it
may see, so exp(-inf - -inf) never happens; walking backward from the diagonal (as FA2 does) without the
guard of §2.5 produces NaN rows.

Contents
  fa                            the deep dive's calculators (fa_calculators.py), loaded from the repo
  attention                     reference: materialize S, softmax, P V; returns (O, lse)
  flash_attention               the tiled forward with counters; order, guard and skipping switchable
  online_softmax_steps          the recurrence on one row of scores in blocks (the primers' worked examples)
  cost_model                    fa_calculators' visited/masked tiles and HBM bytes for the same shape

Standard library + numpy. Tier T0.
"""
from __future__ import annotations

import math

import numpy as np

from . import _repo

fa = _repo.load("fa_calculators")


def _allowed(n_q: int, n_k: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Bottom-right-aligned causal mask (the convention of FA2 and fa_calculators): key j is visible to
    query i iff j <= i + (n_k - n_q). For n_q == n_k it is the usual j <= i."""
    return cols[None, :] <= rows[:, None] + (n_k - n_q)


def attention(Q, K, V, causal: bool = False, scale: float | None = None):
    """softmax(scale * Q K^T [masked]) V, with the natural-log LSE per row. Rows with no visible key
    get O = 0 and LSE = -inf."""
    n_q, d = Q.shape
    n_k = K.shape[0]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    S = (Q @ K.T) * scale
    if causal:
        S = np.where(_allowed(n_q, n_k, np.arange(n_q), np.arange(n_k)), S, -np.inf)
    m = S.max(axis=1)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    P = np.exp(S - m_safe[:, None])
    l = P.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        O = np.where(l[:, None] > 0, (P @ V) / np.where(l > 0, l, 1.0)[:, None], 0.0)
        lse = np.where(l > 0, m_safe + np.log(np.where(l > 0, l, 1.0)), -np.inf)
    return O, lse


def flash_attention(Q, K, V, block_m: int = 4, block_n: int = 4, causal: bool = False,
                    order: str = "forward", guard: bool = False, skip_masked_tiles: bool = True,
                    bytes_per: int = 2, scale: float | None = None):
    """The FA2 schedule: outer loop over Q blocks, inner loop over K/V blocks, one tile of S in hand.

    order="forward" walks key tiles 0, 1, ...; "backward" walks from the causal upper bound down (FA2's
    order, diagonal tiles first). guard=True substitutes 0 for a -inf running max before exponentiating
    (FA2's check_inf); without it, a row whose first visited tile is fully masked computes
    exp(-inf - -inf) = NaN. skip_masked_tiles=False visits every tile and relies on the mask alone:
    still exact, just slower. Returns (O, lse, stats).
    """
    if order not in ("forward", "backward"):
        raise ValueError(order)
    n_q, d = Q.shape
    n_k, dv = V.shape
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    t_r, t_c = math.ceil(n_q / block_m), math.ceil(n_k / block_n)
    O, lse = np.zeros((n_q, dv)), np.zeros(n_q)
    visited = masked = 0
    for mb in range(t_r):
        r0, r1 = mb * block_m, min(n_q, (mb + 1) * block_m)
        Qi, rows = Q[r0:r1], np.arange(r0, r1)
        mi, li, Oi = np.full(r1 - r0, -np.inf), np.zeros(r1 - r0), np.zeros((r1 - r0, dv))
        n_max = t_c
        if causal and skip_masked_tiles:          # n_block_max of flash_fwd_kernel.h
            n_max = max(0, min(t_c, math.ceil(((mb + 1) * block_m + n_k - n_q) / block_n)))
        blocks = range(n_max) if order == "forward" else range(n_max - 1, -1, -1)
        for nb in blocks:
            c0, c1 = nb * block_n, min(n_k, (nb + 1) * block_n)
            visited += 1
            S = (Qi @ K[c0:c1].T) * scale
            if causal:
                if (nb + 1) * block_n - 1 > mb * block_m + (n_k - n_q) or (nb + 1) * block_n > n_k:
                    masked += 1                   # same test as fa_calculators.causal_tiles
                S = np.where(_allowed(n_q, n_k, rows, np.arange(c0, c1)), S, -np.inf)
            with np.errstate(invalid="ignore"):
                m_new = np.maximum(mi, S.max(axis=1))
                m_use = np.where(m_new == -np.inf, 0.0, m_new) if guard else m_new
                alpha = np.exp(mi - m_use)        # without the guard: exp(-inf - -inf) = nan
                P = np.exp(S - m_use[:, None])
            li = alpha * li + P.sum(axis=1)
            Oi = alpha[:, None] * Oi + P @ V[c0:c1]
            mi = m_new
        with np.errstate(divide="ignore", invalid="ignore"):
            if guard:
                O[r0:r1] = np.where(li[:, None] > 0, Oi / np.where(li > 0, li, 1.0)[:, None], 0.0)
                lse[r0:r1] = np.where(li > 0, mi + np.log(np.where(li > 0, li, 1.0)), -np.inf)
            else:
                O[r0:r1] = Oi / li[:, None]
                lse[r0:r1] = mi + np.log(li)
    kv_bytes = visited * 2 * block_n * d * bytes_per      # each visited tile loads one K and one V block
    q_o_lse = 2 * n_q * d * bytes_per + 4 * n_q           # Q in once; O and fp32 LSE out once
    stats = {"visited": visited, "masked": masked, "grid": t_r * t_c,
             "bytes": kv_bytes + q_o_lse, "kv_bytes": kv_bytes,
             "nan_rows": int(np.isnan(O).any(axis=1).sum())}
    return O, lse, stats


def cost_model(n: int, d: int, block_m: int, block_n: int, causal: bool = False, bytes_per: int = 2) -> dict:
    """What fa_calculators predicts for a square n x n head: tiles and FA2-schedule HBM bytes (no-L2 model)."""
    if causal:
        visited, masked, grid = fa.causal_tiles(n, n, block_m, block_n)
    else:
        visited, masked, grid = math.ceil(n / block_m) * math.ceil(n / block_n), 0, math.ceil(n / block_m) * math.ceil(n / block_n)
    t = fa.flash_traffic(n, d, block_m, block_n, b=bytes_per, schedule="fa2", causal=causal)
    return {"visited": visited, "masked": masked, "grid": grid, "bytes": t["bytes"],
            "naive_bytes": fa.naive_traffic(n, d, b=bytes_per)["bytes"], "flops": fa.attention_flops(n, n, d, causal)}


def online_softmax_steps(score_blocks, value_blocks=None) -> tuple[list[dict], dict]:
    """The recurrence of the flash-attention primer §5 on one row of (already scaled) scores.

    Returns (steps, final) where each step records m, alpha, l and the unnormalized weights so far, and
    final holds m, l, the normalized weights and, with values, the output."""
    m, l, steps = -math.inf, 0.0, []
    weights = np.zeros(0)
    o = None if value_blocks is None else np.zeros(np.asarray(value_blocks[0]).shape[1])
    for i, s in enumerate(score_blocks):
        s = np.asarray(s, dtype=float)
        m_new = max(m, float(s.max()))
        alpha = math.exp(m - m_new) if m != -math.inf else 0.0
        p = np.exp(s - m_new)
        l = alpha * l + float(p.sum())
        weights = np.concatenate([alpha * weights, p])
        if o is not None:
            o = alpha * o + p @ np.asarray(value_blocks[i], dtype=float)
        steps.append({"m": m_new, "alpha": alpha, "l": l, "p": p, "unnormalized": weights.copy()})
        m = m_new
    final = {"m": m, "l": l, "weights": weights / l, "lse": m + math.log(l)}
    if o is not None:
        final["o"] = o / l
    return steps, final
