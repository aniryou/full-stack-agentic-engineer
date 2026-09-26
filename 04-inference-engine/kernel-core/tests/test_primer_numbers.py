"""Every worked number of the kv-cache primer, and those of the paged-attention and flash-attention primers
that kerncore (or fa_calculators, which kerncore imports) can compute, recomputed here and required to
appear on the page verbatim.

If a formula or an input changes, the page fails this test until it is updated. Facts the primers quote
from papers and datasheets are in their verify lists and are inputs here, not outputs.
"""
from pathlib import Path

import numpy as np
import pytest

from kerncore import flash, kv, paged
from kerncore.kv import GB, GiB, KiB

LAYER = Path(__file__).resolve().parents[2]
_norm = lambda t: " ".join(t.split())
KV_PRIMER = _norm((LAYER / "kv-cache" / "kv-cache-primer.md").read_text(encoding="utf-8"))
PAGED_PRIMER = _norm((LAYER / "paged-attention" / "paged-attention-primer.md").read_text(encoding="utf-8"))
FLASH_PRIMER = _norm((LAYER / "flash-attention" / "flash-attention-primer.md").read_text(encoding="utf-8"))
DEEP_DIVE = _norm((LAYER / "flash-attention" / "flash-attention-deep-dive.md").read_text(encoding="utf-8"))

L3, L2 = kv.MODELS["llama-3-8b"], kv.MODELS["llama-2-13b"]


def present(page, *fragments):
    missing = [f for f in fragments if _norm(f) not in page]
    assert not missing, f"the page no longer says: {missing}"


def bracket(b):                       # "1 GiB (1.07 GB)" style, as the primer's table rounds it
    return f"({b / GB:.2f} GB)"


# --- kv-cache primer ---------------------------------------------------------------------------------
def test_kv_s4_per_token_llama3_8b():
    per_tok = kv.kv_bytes_per_token(L3.n_layers, L3.n_kv_heads, L3.head_dim, 2)
    assert per_tok == 131_072 and per_tok / KiB == 128
    present(KV_PRIMER, "per token = 2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB")


def test_kv_s4_table():
    one_8k = kv.kv_cache_bytes(32, 8, 128, 8192)
    assert one_8k == GiB
    present(KV_PRIMER, f"| 1 user, 8K (8,192-token) context | 1 GiB ({one_8k / GB:.2f} GB) |")
    k128 = kv.kv_cache_bytes(32, 8, 128, 131_072)
    k128k = kv.kv_cache_bytes(32, 8, 128, 128_000)
    present(KV_PRIMER, f"16 GiB ({k128 / GB:.1f} GB); the notebooks' 128,000 tokens give "
                       f"{k128k / GiB:.1f} GiB ({k128k / GB:.1f} GB)")
    assert k128 == 16 * GiB
    b32 = kv.kv_cache_bytes(32, 8, 128, 8192, batch=32)
    assert b32 == 32 * GiB
    present(KV_PRIMER, f"| 32 users, 8K context each | 32 GiB ({b32 / GB:.1f} GB) |")
    # the notebooks print two decimals: 1.00 GiB (1.07 GB), 15.62 GiB (16.78 GB), 32.00 GiB (34.36 GB)
    assert [kv.fmt_bytes(x) for x in (one_8k, k128k, b32)] == [
        "1.00 GiB (1.07 GB)", "15.62 GiB (16.78 GB)", "32.00 GiB (34.36 GB)"]


def test_kv_s4_weights_and_sessions():
    w = kv.weight_bytes(L3)
    assert round(w / GB) == 16 and round(w / GiB) == 15
    present(KV_PRIMER, "a fixed ~16 GB (15 GiB) in fp16")
    assert kv.kv_cache_bytes(32, 8, 128, 8192, batch=32) > w            # the batch outweighs the model
    per16 = kv.kv_bytes_per_token(32, 8, 128, kv.DTYPE_BYTES["fp16"])
    per8 = kv.kv_bytes_per_token(32, 8, 128, kv.DTYPE_BYTES["fp8"])
    s16 = kv.sessions_per_gpu(80 * GB, w, 8192, per16)
    s8 = kv.sessions_per_gpu(80 * GB, w, 8192, per8)
    assert round((80 * GB - w) / GB) == 64
    present(KV_PRIMER, f"holds at most **{s16}** sessions of 8K tokens with an fp16 cache, or **{s8}** with an fp8 cache")


def test_kv_s4_llama2_13b_and_the_gqa_gap():
    per_tok = kv.kv_bytes_per_token(L2.n_layers, L2.n_kv_heads, L2.head_dim, 2)
    assert per_tok == 819_200 and per_tok / KiB == 800
    present(KV_PRIMER, "per token = 2 × 40 × 40 × 128 × 2 = 819,200 bytes = 800 KiB")
    ratio = per_tok / kv.kv_bytes_per_token(32, 8, 128, 2)
    assert round(ratio) == 6 and round(L2.params / L3.params, 1) == 1.6
    present(KV_PRIMER, "Six times larger per token, for a model only 1.6× bigger")


def test_kv_s4_long_context_halves_decode_speed():
    w = kv.weight_bytes(L3)
    k128 = kv.kv_cache_bytes(32, 8, 128, 131_072)
    assert round(k128 / GB) == 17
    short = kv.decode_tokens_per_s(3.35e12, w, 0)
    long = kv.decode_tokens_per_s(3.35e12, w, k128)
    assert 0.45 < long / short < 0.55
    present(KV_PRIMER, "decode reads ~16 GB of weights *plus* ~17 GB of cache per token",
            "roughly **half the tokens per second**")


def test_kv_s3_ridge_and_decode_intensity():
    assert round(kv.ridge(1000, 3.35), -2) == 300
    present(KV_PRIMER, "~300 floating-point operations per byte loaded")
    assert kv.decode_intensity(1, 2) == 1.0 and kv.decode_intensity(L3.group, 2) == 4.0
    assert kv.decode_intensity(1, 2) == flash.fa.decode_intensity(1, 2)
    present(KV_PRIMER, "about **1 FLOP per byte**", "GQA with `g = 4` (Llama 3 8B) gets to 4 FLOP/B")


def test_kv_s5_s6_reservation_waste_gqa_and_fp8():
    w = paged.allocation_waste([200], block_size=16, reserve=4096)
    assert round(100 * w["contiguous_waste"]) == 95
    present(KV_PRIMER, "stops after 200 tokens wastes 95% of its reservation")
    assert w["paged_waste"] < 0.04                                     # 13 blocks of 16 = 208 slots
    present(KV_PRIMER, "Waste drops to well under 4%", "fixed-size blocks — say 16 tokens each")
    rows = {(r["kind"], r["dtype"]): r["per_token"] for r in kv.kv_table(L3)}
    assert rows[("MHA", "fp16")] / rows[("GQA", "fp16")] == 4
    present(KV_PRIMER, "Llama 3 8B has 32 query heads and 8 KV heads — a 4× cache reduction")
    assert rows[("GQA", "fp16")] / rows[("GQA", "fp8")] == 2
    present(KV_PRIMER, "Store the cache in fp8 or int8 instead of fp16 — 2× smaller")


# --- paged-attention primer (binary units, GB in brackets, as in the kv-cache primer) ---------------------
def test_paged_llama13b_per_token_and_per_sequence():
    per_tok = kv.kv_bytes_per_token(40, 40, 128, 2)
    assert per_tok / KiB == 800
    seq = 2048 * per_tok
    assert kv.fmt_bytes(seq) == "1.56 GiB (1.68 GB)"
    present(PAGED_PRIMER, "works out to 800 KiB per token", "a single 2,048-token sequence occupies about 1.56 GiB (1.68 GB)",
            "costs 800 KiB of KV cache per token, so ~1.56 GiB (1.68 GB) per 2K-token sequence")
    fit = kv.sessions_per_gpu(40 * GB, 26 * GB, 2048, per_tok)
    assert fit == 8                                                    # "only a handful"
    present(PAGED_PRIMER, "the FP16 weights already take 26 GB, leaving room for only a handful of max-length sequences")


def test_paged_waste_is_at_most_one_partial_block():
    lengths = np.random.default_rng(0).integers(1, 4000, 500)
    w = paged.allocation_waste(lengths, block_size=16)
    assert w["paged_slots"] - w["used"] <= 15 * len(lengths)
    present(PAGED_PRIMER, "waste collapses to at most one partially filled block per sequence",
            "16 by default in vLLM")


# --- flash-attention primer and deep dive (numbers via fa_calculators) --------------------------------
def test_flash_s1_score_matrix_size():
    s = 8192 * 8192 * 2
    assert s == 128 * 1024**2 and s * 32 * 8 == 32 * GiB
    present(FLASH_PRIMER, "`S` in fp16 is 128 MB **per head, per sequence in the batch**",
            "32 heads and a batch of 8, that's 32 GB")


def test_flash_s2_s3_naive_is_memory_bound():
    assert round(flash.fa.DEVICES["H100"].ridge, -2) == 300 and round(990 / 3.3) == 300
    t = flash.fa.naive_traffic(4096, 64)
    assert round(t["flops"] / 1e9, 1) == 4.3
    assert round(3 * 4096 * 64 * 2 / 1e6, 1) == 1.6 and round(t["s_matrix_bytes"] / 1e6, 1) == 33.6
    assert round(4 * t["s_matrix_bytes"] / 1e6) == 134 and round(t["intensity"]) == 32
    present(FLASH_PRIMER, "≈ **4.3 GFLOP**", "`Q,K,V` are only 1.6 MB total", "roughly **134 MB** of HBM traffic",
            "≈ **32 FLOPs per byte**")


def test_flash_s5_worked_online_softmax():
    steps, final = flash.online_softmax_steps([[1, 3], [5, 2]])
    assert f"{steps[0]['l']:.4f}" == "1.1353" and f"{steps[1]['alpha']:.4f}" == "0.1353"
    assert f"{final['l']:.4f}" == "1.2034"
    present(FLASH_PRIMER, "`ℓ = exp(1−3) + exp(3−3) = 0.1353 + 1 = 1.1353`", "`α = exp(3−5) = 0.1353`")
    w = "[" + ", ".join(f"{x:.4f}" for x in final["weights"]) + "]"
    present(FLASH_PRIMER, f"So the true weights are `{w}`", "summing to `1.2034`", f"you recover `{w}` exactly")
    assert np.allclose(final["weights"], flash.attention(np.array([[1.0]]), np.array([[1.0], [3], [5], [2]]),
                                                         np.eye(4), scale=1.0)[0][0])


def test_flash_s8_matmul_vs_fp32_ratio_and_causal_half():
    assert round(312 / 19.5) == 16
    present(FLASH_PRIMER, "a non-matmul FLOP costs ~16× a matmul FLOP")
    v, _, g = flash.fa.causal_tiles(32768, 32768, 128, 64)
    assert abs(v / g - 0.5) < 0.01
    present(FLASH_PRIMER, "Causal masking is worth roughly 2×")


def test_deep_dive_s4_4_tiles_from_the_tiled_kernel():
    rng = np.random.default_rng(0)
    Q, K, V = (rng.standard_normal((512, 8)) for _ in range(3))
    _, _, st = flash.flash_attention(Q, K, V, 128, 64, causal=True)
    assert (st["visited"], st["masked"], st["grid"]) == (20, 8, 32)
    present(DEEP_DIVE, "20 of 32 tiles visited, 8 masked")
    v, m, g = flash.fa.causal_tiles(4096, 4096, 128, 64)
    present(DEEP_DIVE, f"visits {v:,} of {g:,} tiles ({100 * v / g:.1f}%, {m} masked)")


def test_deep_dive_s2_6_two_block_example():
    steps, final = flash.online_softmax_steps([np.array([4, 8]) * 0.5, np.array([6, 12]) * 0.5],
                                              [[[1, 0], [0, 1]], [[1, 1], [2, -1]]])
    o = "[" + ", ".join(f"{x:.6f}" for x in final["o"]).replace("-", "−") + "]"
    present(DEEP_DIVE, f"`O = {o}`; `LSE = {final['lse']:.6f}`")
    present(DEEP_DIVE, f"`l = {steps[0]['l']:.6f}`", f"`α = exp(0.5·(8 − 12)) = e^−2 = {steps[1]['alpha']:.6f}`")
