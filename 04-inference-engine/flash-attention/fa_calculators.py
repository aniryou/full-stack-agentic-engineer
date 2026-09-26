"""FlashAttention calculators: the derived numbers of flash-attention-deep-dive.md, in code.

One idea: attention cost is a bookkeeping problem. Count the FLOPs, count the bytes that
cross HBM under a given loop schedule, divide by the machine's two peaks, and you know
which wall a kernel hits before you run it. Every number in the deep dive that is derived
rather than cited comes from a function here, and test_fa_calculators.py pins them.

Contents
  DEVICES                      peak dense bf16 FLOP/s and HBM bandwidth (dated; verify)
  attention_flops              the 4*Nq*Nk*d convention, causal and backward factors
  naive_traffic                HBM bytes of the unfused S -> P -> O schedule
  flash_traffic                HBM bytes of the FA1 / FA2 tiled schedules (no-L2 model)
  roofline                     compute time vs memory time vs the bound
  causal_tiles, window_tiles   tiles a causal / sliding-window kernel visits and masks
  num_splits_heuristic         FlashAttention-2's split-KV chooser (ported from flash_api.cpp)
  fa2_decode_splits            ... applied to a decode batch the way mha_fwd_kvcache does
  decode_*                     bytes and arithmetic intensity of decode attention
  mla_decode_intensity         intensity of absorbed-weight MLA decode
  softmax state algebra        (m, l, o) states, merge, LSE form, a traced 2-block example
  round_to_e4m3, hadamard      FP8 emulation and the rotation used by incoherent processing

Standard library + numpy. Tier T0: no GPU needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LOG2E = 1.4426950408889634  # log2(e): exp(x) == exp2(x * LOG2E)


# ---------------------------------------------------------------------------------------
# Devices (dense peaks, no sparsity). Same values as the roofline core in layer 01.
# Datasheet numbers as of 2026-09; treat as (verify).
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Device:
    name: str
    bf16_tflops: float      # dense tensor-core peak, bf16/fp16
    hbm_tbs: float          # memory bandwidth, TB/s
    sms: int                # streaming multiprocessors
    smem_kb_per_sm: int     # max shared memory per SM (KB)
    l2_mb: int              # L2 cache (MB)

    @property
    def ridge(self) -> float:
        """FLOP per byte at which memory time equals compute time."""
        return self.bf16_tflops * 1e12 / (self.hbm_tbs * 1e12)


DEVICES = {
    "T4": Device("NVIDIA T4 (fp16; no bf16)", 65.0, 0.32, 40, 64, 4),
    "L4": Device("NVIDIA L4", 121.0, 0.30, 58, 100, 48),
    "A100": Device("NVIDIA A100 SXM 80GB", 312.0, 2.039, 108, 164, 40),
    "H100": Device("NVIDIA H100 SXM", 989.4, 3.35, 132, 228, 50),
    "B200": Device("NVIDIA B200", 2250.0, 8.0, 148, 228, 126),
}


# ---------------------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------------------
def attention_flops(n_q: int, n_k: int, d: int, causal: bool = False, pass_: str = "fwd",
                    heads: int = 1, batch: int = 1) -> float:
    """FLOPs by the convention every FlashAttention benchmark uses.

    Forward = two GEMMs (Q K^T and P V), each 2*n_q*n_k*d  ->  4*n_q*n_k*d per head.
    Causal halves it (the convention; the exact fraction is causal_fraction()).
    Backward = 5 GEMMs = 2.5x forward ("2.0 bwd + 0.5 recompute" in the Triton tutorial).
    Softmax FLOPs are not counted.
    """
    f = 4.0 * n_q * n_k * d * heads * batch
    if causal:
        f *= 0.5
    factor = {"fwd": 1.0, "bwd": 2.5, "fwd_bwd": 3.5}[pass_]
    return f * factor


def causal_fraction(n_q: int, n_k: int) -> float:
    """Exact fraction of (i, j) pairs kept by a bottom-right-aligned causal mask."""
    kept = 0
    for i in range(n_q):
        kept += max(0, min(n_k, i + (n_k - n_q) + 1))
    return kept / (n_q * n_k)


# ---------------------------------------------------------------------------------------
# HBM traffic
# ---------------------------------------------------------------------------------------
def naive_traffic(n: int, d: int, b: int = 2, b_s: int | None = None,
                  extra_elementwise_passes: int = 0) -> dict:
    """HBM bytes for one head of unfused attention: S = QK^T; P = softmax(S); O = PV.

    b is bytes per element of Q, K, V, O; b_s bytes per element of S and P (default b).
    Each extra unfused elementwise pass over S (scale, mask, dropout) reads and writes N^2.
    """
    b_s = b if b_s is None else b_s
    passes = {
        "S = Q K^T": 2 * n * d * b + n * n * b_s,        # read Q, K; write S
        "P = softmax(S)": 2 * n * n * b_s,               # read S; write P
        "O = P V": n * n * b_s + 2 * n * d * b,          # read P, V; write O
    }
    if extra_elementwise_passes:
        passes["extra elementwise"] = extra_elementwise_passes * 2 * n * n * b_s
    total = sum(passes.values())
    flops = attention_flops(n, n, d)
    return {"passes": passes, "bytes": total, "flops": flops, "intensity": flops / total,
            "s_matrix_bytes": n * n * b_s}


def compulsory_bytes(n: int, d: int, b: int = 2, lse: bool = True) -> int:
    """Read Q, K, V once, write O once (+ fp32 LSE per row): no schedule can do less."""
    return 4 * n * d * b + (4 * n if lse else 0)


def flash_traffic(n: int, d: int, block_m: int, block_n: int, b: int = 2,
                  schedule: str = "fa2", causal: bool = False) -> dict:
    """HBM bytes for one head under the tiled schedules, assuming NO L2 reuse (the paper's model).

    fa1: outer loop over K/V blocks, inner over Q blocks. K, V read once; for every K/V block
         the whole Q and O stream through (O and the fp32 (m, l) are read and written back).
    fa2: outer loop over Q blocks (one CTA each), inner over K/V blocks. Q read once, O and LSE
         written once; K and V are re-read once per Q block (only below-diagonal blocks if causal).
    Real kernels on GPUs with a large L2 see much less DRAM traffic, because CTAs working on the
    same head re-read K/V from L2 -- compare with compulsory_bytes().
    """
    t_r = math.ceil(n / block_m)
    t_c = math.ceil(n / block_n)
    if schedule == "fa1":
        kv = 2 * n * d * b
        per_kv_block = 3 * n * d * b + 4 * n * 4        # read Q, read O, write O; r/w m and l
        total = kv + t_c * per_kv_block
        parts = {"K,V once": kv, "Q,O,m,l per K/V block": t_c * per_kv_block}
    elif schedule == "fa2":
        if causal:
            visited, _, _ = causal_tiles(n, n, block_m, block_n)
        else:
            visited = t_r * t_c
        kv = visited * 2 * block_n * d * b              # each visited tile loads one K and one V block
        q_o_lse = 2 * n * d * b + 4 * n
        total = kv + q_o_lse
        parts = {"K,V re-reads": kv, "Q in, O and LSE out": q_o_lse}
    else:
        raise ValueError(schedule)
    flops = attention_flops(n, n, d, causal=causal)
    return {"bytes": total, "parts": parts, "flops": flops, "intensity": flops / total}


def tile_smem_bytes(block_m: int, block_n: int, d: int, b: int = 2, kv_stages: int = 1) -> int:
    """Shared memory for one Q tile plus kv_stages pairs of K and V tiles."""
    return (block_m + 2 * block_n * kv_stages) * d * b


# ---------------------------------------------------------------------------------------
# Roofline
# ---------------------------------------------------------------------------------------
def roofline(flops: float, bytes_: float, device: str | Device) -> dict:
    """Time floor from each peak; the larger one is the bound."""
    dev = DEVICES[device] if isinstance(device, str) else device
    t_c = flops / (dev.bf16_tflops * 1e12)
    t_m = bytes_ / (dev.hbm_tbs * 1e12)
    t = max(t_c, t_m)
    return {"t_compute_s": t_c, "t_memory_s": t_m, "t_floor_s": t,
            "bound": "memory" if t_m > t_c else "compute",
            "attainable_tflops": flops / t / 1e12,
            "fraction_of_peak": (flops / t) / (dev.bf16_tflops * 1e12)}


# ---------------------------------------------------------------------------------------
# Tile counting (mirrors n_block_min / n_block_max in FA2's flash_fwd_kernel.h)
# ---------------------------------------------------------------------------------------
def causal_tiles(n_q: int, n_k: int, block_m: int, block_n: int) -> tuple[int, int, int]:
    """(tiles visited, tiles that need masking, full grid) for bottom-right-aligned causal."""
    t_r, t_c = math.ceil(n_q / block_m), math.ceil(n_k / block_n)
    visited = masked = 0
    for m in range(t_r):
        n_max = min(t_c, math.ceil(((m + 1) * block_m + n_k - n_q) / block_n))
        for nb in range(max(0, n_max)):
            visited += 1
            # a tile needs the mask if its last key column lies right of its first query row
            first_row_key_limit = m * block_m + (n_k - n_q)     # last allowed key for first row
            if (nb + 1) * block_n - 1 > first_row_key_limit or (nb + 1) * block_n > n_k:
                masked += 1
    return visited, masked, t_r * t_c


def window_tiles(n: int, window_left: int, block_m: int, block_n: int) -> int:
    """Tiles visited by causal sliding-window attention (key j allowed if i - W <= j <= i)."""
    t_c = math.ceil(n / block_n)
    visited = 0
    for m in range(math.ceil(n / block_m)):
        n_min = max(0, (m * block_m - window_left) // block_n)
        n_max = min(t_c, math.ceil((m + 1) * block_m / block_n))
        visited += max(0, n_max - n_min)
    return visited


# ---------------------------------------------------------------------------------------
# Split-KV (Flash-Decoding) heuristic, ported from csrc/flash_attn/flash_api.cpp (fetched 2026-09-26)
# ---------------------------------------------------------------------------------------
def num_splits_heuristic(batch_nheads_mblocks: int, num_sms: int, num_n_blocks: int,
                         max_splits: int) -> int:
    """If the grid already nearly fills the GPU, don't split. Otherwise pick the smallest split
    count whose wave efficiency is >= 85% of the best achievable one."""
    if batch_nheads_mblocks >= 0.8 * num_sms:
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)

    def eligible(s: int) -> bool:   # 11 and 12 splits of 64 blocks are the same partition
        return s == 1 or math.ceil(num_n_blocks / s) != math.ceil(num_n_blocks / (s - 1))

    eff = []
    for s in range(1, max_splits + 1):
        if not eligible(s):
            eff.append(0.0)
            continue
        waves = batch_nheads_mblocks * s / num_sms
        eff.append(waves / math.ceil(waves))
    best = max(eff) if eff else 0.0
    for s in range(1, max_splits + 1):
        if eligible(s) and eff[s - 1] >= 0.85 * best:
            return s
    return 1


def fa2_decode_splits(batch: int, n_heads_q: int, n_heads_kv: int, seqlen_k: int,
                      head_dim: int, num_sms: int) -> dict:
    """What FA2's flash_attn_with_kvcache would launch for one decode step (seqlen_q = 1).

    GQA packing: with seqlen_q == 1 and n_heads_q > n_heads_kv the query heads of a group are
    reshaped into the sequence dimension (seqlenq_ngroups_swapped), so the kernel sees
    n_heads_kv heads with seqlen_q = group. The split-KV kernel uses kBlockM = 64 and
    kBlockN = 256 / 128 / 64 for head_dim <= 64 / <= 128 / larger, and passes 2 * #SMs
    because two 128-thread CTAs fit per SM.
    """
    group = n_heads_q // n_heads_kv
    heads, seqlen_q = (n_heads_kv, group) if group > 1 else (n_heads_q, 1)
    block_n = 256 if head_dim <= 64 else (128 if head_dim <= 128 else 64)
    n_blocks = math.ceil(seqlen_k / block_n)
    m_blocks = math.ceil(seqlen_q / 64)
    ctas = batch * heads * m_blocks
    splits = num_splits_heuristic(ctas, 2 * num_sms, n_blocks, 128)
    return {"group": group, "ctas_without_split": ctas, "block_n": block_n, "n_blocks": n_blocks,
            "splits": splits, "ctas": ctas * splits,
            "blocks_per_split": math.ceil(n_blocks / splits)}


# ---------------------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------------------
def kv_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int, b: int = 2) -> int:
    """K and V for one token across all layers (the kv-cache primer's formula)."""
    return 2 * n_layers * n_kv_heads * head_dim * b


def decode_attention_bytes(tokens_in_batch: int, n_layers: int, n_kv_heads: int,
                           head_dim: int, b: int = 2) -> int:
    """KV bytes one decode step must read: every cached token of every sequence, every layer."""
    return tokens_in_batch * kv_bytes_per_token(n_layers, n_kv_heads, head_dim, b)


def decode_intensity(group: int, b: int = 2) -> float:
    """FLOP/byte of decode attention: 4*L*d*g FLOPs over 2*L*d*b bytes = 2g/b."""
    return 2.0 * group / b


def mla_decode_intensity(n_heads: int = 128, d_latent: int = 512, d_rope: int = 64,
                         b: int = 2) -> float:
    """Absorbed MLA decode: all heads share one cached latent (d_latent + d_rope per token).

    Per cached token: QK = 2*(d_latent + d_rope)*H FLOPs, PV = 2*d_latent*H FLOPs,
    bytes = (d_latent + d_rope) * b.
    """
    flops = 2 * (d_latent + d_rope) * n_heads + 2 * d_latent * n_heads
    return flops / ((d_latent + d_rope) * b)


def prefill_attention_share(context: int, n_heads: int, head_dim: int,
                            linear_params_per_layer: float, causal: bool = True) -> float:
    """Attention FLOPs / linear-layer FLOPs per token per layer, averaged over a prompt.

    Attention per token = 4 * n_ctx * H * d (halved on average for causal); linear = 2 * params.
    """
    attn = 4.0 * context * n_heads * head_dim * (0.5 if causal else 1.0)
    return attn / (2.0 * linear_params_per_layer)


def ring_min_tokens_per_gpu(achieved_tflops: float, link_gbs: float, b: int = 2) -> float:
    """Ring attention hides the K/V hop when per-step compute >= per-step transfer.

    Per head, per step: compute 4*(N/P)^2*d / F, transfer 2*(N/P)*d*b / W  ->  N/P >= b*F/(2W).
    """
    return b * achieved_tflops * 1e12 / (2.0 * link_gbs * 1e9)


# ---------------------------------------------------------------------------------------
# Online-softmax state algebra
# ---------------------------------------------------------------------------------------
@dataclass
class State:
    """Unnormalized attention state for one query row: running max m, sum l, output o."""
    m: float
    l: float
    o: np.ndarray

    def finalize(self) -> tuple[np.ndarray, float]:
        """Normalized output and log-sum-exp (natural log)."""
        if self.l == 0.0:
            return np.zeros_like(self.o), -math.inf
        return self.o / self.l, self.m + math.log(self.l)


def empty_state(dv: int) -> State:
    """Identity of merge(): (-inf, 0, 0)."""
    return State(-math.inf, 0.0, np.zeros(dv))


def block_state(scores: np.ndarray, values: np.ndarray) -> State:
    """State of one block of (already scaled) scores against its values."""
    m = float(np.max(scores))
    p = np.exp(scores - m)
    return State(m, float(p.sum()), p @ values)


def merge(a: State, b: State) -> State:
    """Associative, commutative merge. Guards the (-inf) - (-inf) case the kernels guard."""
    m = max(a.m, b.m)
    if m == -math.inf:
        return empty_state(len(a.o))
    ca = math.exp(a.m - m) if a.m != -math.inf else 0.0
    cb = math.exp(b.m - m) if b.m != -math.inf else 0.0
    return State(m, ca * a.l + cb * b.l, ca * a.o + cb * b.o)


def merge_lse(o_a: np.ndarray, lse_a: float, o_b: np.ndarray, lse_b: float):
    """The same merge on normalized outputs + LSE (FA2 split-KV combine, vLLM
    merge_attn_states, FlashInfer merge_state)."""
    mx = max(lse_a, lse_b)
    if mx == -math.inf:
        return np.zeros_like(o_a), -math.inf
    wa, wb = math.exp(lse_a - mx), math.exp(lse_b - mx)
    lse = mx + math.log(wa + wb)
    return (wa * o_a + wb * o_b) / (wa + wb), lse


def online_trace(score_blocks, value_blocks, use_exp2: bool = False, scale: float = 1.0):
    """Run the online-softmax recurrence over blocks of RAW scores, recording each step.

    With use_exp2, exponentials are computed as exp2(s*c - m*c) with c = scale*log2(e),
    which is how the kernels fold the softmax scale into one FFMA. Returns (steps, o, lse).
    """
    c = scale * LOG2E
    m, l = -math.inf, 0.0
    o = np.zeros(np.asarray(value_blocks[0]).shape[1])
    steps = []
    for s, v in zip(score_blocks, value_blocks):
        s = np.asarray(s, dtype=float)
        v = np.asarray(v, dtype=float)
        m_new = max(m, float(s.max()))                         # max on RAW scores
        if use_exp2:
            alpha = 2.0 ** ((m - m_new) * c) if m != -math.inf else 0.0
            p = 2.0 ** (s * c - m_new * c)
        else:
            alpha = math.exp((m - m_new) * scale) if m != -math.inf else 0.0
            p = np.exp((s - m_new) * scale)
        l = alpha * l + float(p.sum())
        o = alpha * o + p @ v
        steps.append({"m": m_new, "alpha": alpha, "p": p, "l": l, "o_unnorm": o.copy()})
        m = m_new
    lse = m * scale + math.log(l)
    return steps, o / l, lse


def reference_attention_row(scores: np.ndarray, values: np.ndarray, scale: float = 1.0):
    """Plain softmax(scale * s) @ V with its LSE, for checking."""
    z = np.asarray(scores, dtype=float) * scale
    mx = z.max()
    p = np.exp(z - mx)
    return (p / p.sum()) @ np.asarray(values, dtype=float), float(mx + np.log(p.sum()))


def split_kv_decode(q: np.ndarray, k: np.ndarray, v: np.ndarray, n_splits: int,
                    scale: float | None = None):
    """Flash-Decoding for one query vector: each split returns (normalized o, lse);
    a combine step merges them in LSE form. Returns (o, lse, per-split partials)."""
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    bounds = np.linspace(0, k.shape[0], n_splits + 1).astype(int)
    partials = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo:
            continue
        st = block_state((k[lo:hi] @ q) * scale, v[lo:hi])
        partials.append(st.finalize())
    o, lse = partials[0]
    for o_i, lse_i in partials[1:]:
        o, lse = merge_lse(o, lse, o_i, lse_i)
    return o, lse, partials


# ---------------------------------------------------------------------------------------
# FP8 (e4m3fn) emulation and the Hadamard rotation used by incoherent processing
# ---------------------------------------------------------------------------------------
E4M3_MAX = 448.0


def round_to_e4m3(x: np.ndarray) -> np.ndarray:
    """Round to the nearest float8 e4m3fn value (3 mantissa bits, min normal 2^-6,
    subnormal step 2^-9, saturating at +-448). Round-half-to-even."""
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    e = np.maximum(e, -6.0)                          # subnormals share the 2^-6 binade step
    step = np.exp2(e - 3.0)
    q = np.round(a / step) * step
    q = np.minimum(q, E4M3_MAX)
    return np.sign(x) * q


def quantize_fp8(x: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Per-tensor scaled FP8: x ~ round_to_e4m3(x / scale) * scale, scale = amax / 448."""
    scale = float(np.max(np.abs(x))) / E4M3_MAX if scale is None else scale
    scale = scale if scale > 0 else 1.0
    return round_to_e4m3(np.asarray(x) / scale) * scale, scale


def hadamard(n: int) -> np.ndarray:
    """Orthonormal Sylvester-Hadamard matrix (n a power of two): H @ H.T == I."""
    if n < 1 or n & (n - 1):
        raise ValueError("n must be a power of two")
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


def random_hadamard(n: int, seed: int = 0) -> np.ndarray:
    """H @ diag(random signs): still orthonormal, and it mixes a fixed outlier channel into all."""
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=n)
    return hadamard(n) * signs[None, :]
