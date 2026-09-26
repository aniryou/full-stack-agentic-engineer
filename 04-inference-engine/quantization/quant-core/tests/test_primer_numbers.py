"""Every number ../PRIMER.md computes with quantcore is recomputed here and must appear in it verbatim.

If a formula, a seed or a default changes, the primer fails this test until it is updated. Numbers the primer
quotes from sources (papers, READMEs, vLLM docs) or from other repo modules are marked (verify) or linked there
and are not pinned here, except where quantcore reproduces them (tests/test_repo_numbers.py).
"""
from pathlib import Path

import numpy as np
import pytest

from quantcore import TinyModel, awq, cost as C, eval as E, formats as F, granularity as G, gptq, kvquant as K
from quantcore import quantize_model, smoothquant as S, w8a8


def _norm(text: str) -> str:
    return " ".join(text.split())


PRIMER = _norm((Path(__file__).resolve().parents[2] / "PRIMER.md").read_text(encoding="utf-8"))
L4, H100, T4 = C.GPUS["L4"], C.GPUS["H100-SXM"], C.GPUS["T4"]
M8, M70 = C.MODELS["llama-3.1-8b"], C.MODELS["llama-3.1-70b"]
pct = lambda v, d=1: f"{100 * v:.{d}f}%"


def present(*fragments):
    missing = [f for f in fragments if _norm(f) not in PRIMER]
    assert not missing, f"PRIMER.md no longer says: {missing}"


@pytest.fixture(scope="module")
def tiny():
    m = TinyModel()
    X, y = m.sample(4000, "test")
    Xc, _ = m.sample(256, "calib")
    return m, X, y, Xc, m.forward(X), m.calibration_inputs(Xc)


def test_s1_crossovers_and_bytes():
    bf, f8, w4 = (C.crossover_tokens(14336, 4096, L4, s) for s in ("bf16", "w8a8-fp8", "w4a16"))
    present(f"BF16 {bf:.0f} tokens FP8 W8A8 {f8:.0f} W4A16 {w4:.0f}")
    t = lambda M, s: C.gemm_time(M, 14336, 4096, L4, s, w_bits=4.16 if s == "w4a16" else None)["t"] * 1e6
    present(f"W4A16 at {t(1, 'w4a16'):.0f} µs against BF16's {t(1, 'bf16'):.0f} µs")
    assert abs(t(2048, "w4a16") - t(2048, "bf16")) < 1e-9
    present(f"it is {2 * M8.embed_params / 1e9:.2f} GB of the {M8.weight_bytes(4.125) / 1e9:.2f} GB an INT4 checkpoint")
    present(f"16/4.125 = **{16 / 4.125:.2f}×**")


def test_s2_integers_and_floats():
    w = np.random.default_rng(0).standard_normal((128, 256)) * 0.02
    sq = lambda conv: G.error(w, G.fake_quant(w, fmt="int4", granularity="group", group_size=32, convention=conv))["sqnr_db"]
    present(f"{sq('full') - sq('restricted'):.2f} dB better on Gaussian INT4 g32")
    scale, zero = F.asym_params(-0.3, 1.2, 4)
    present(f"gives scale {float(scale):.1f} and zero point {int(zero)}")
    e4, e5 = F.E4M3, F.E5M2
    assert (e4.grid()[-1], e5.grid()[-1], 2 * len(e4.grid()) - 1) == (448, 57344, 253)
    present("| FP8 **E4M3** (`fn`) | 1-4-3, 7 | **448** | 2⁻⁶ | 2⁻⁹ | 6.25% |",
            "| FP8 **E5M2** | 1-5-2, 15 | **57,344** | 2⁻¹⁴ | 2⁻¹⁶ | 12.5% |",
            f"2^{np.log2(e4.max_value / e4.min_normal):.1f} |", f"2^{np.log2(e5.max_value / e5.min_normal):.1f} |",
            "254 finite codes and 253 distinct values")
    rng = np.random.default_rng(0)                          # notebook 01's draws, in order
    rng.standard_normal((128, 256)), rng.standard_normal((128, 256))
    x = rng.standard_normal(10000)
    rel = lambda f: np.abs(F.to_float(x, f) - x) / np.abs(x)
    m4, m5 = rel(e4), rel(e5)
    present(f"E4M3 rounds with a {pct(np.median(m4))} median and {pct(m4[np.abs(x) > e4.min_normal].max())} maximum "
            f"relative error, E5M2 with {pct(np.median(m5))} and {pct(m5[np.abs(x) > e5.min_normal].max())}")


def test_s2_block_formats():
    codes = F.mxfp4(np.r_[np.full(32, 5.0), np.full(32, 0.75)])[1].ravel()
    assert list(codes) == [127, 124]
    present("gets exponent 0 (byte 127)", "gets −3 (byte 124)")
    rng = np.random.default_rng(0)
    Wg, Wt = rng.standard_normal((256, 512)) * 0.02, rng.standard_t(3, (256, 512)) * 0.02
    rows = {"INT4 g32, fp16 scale (full convention) | 4.5":
            lambda W: G.fake_quant(W, fmt="int4", granularity="group", group_size=32, convention="full"),
            "FP4 E2M1 g32, fp16 scale | 4.5": lambda W: G.fake_quant(W, fmt="fp4", granularity="group", group_size=32),
            "MXFP4 | 4.25": lambda W: F.mxfp4(W)[2], "NVFP4 | 4.5": lambda W: F.nvfp4(W)[3]}
    for label, fn in rows.items():
        present(f"| {label} | {G.error(Wg, fn(Wg))['rel']:.4f} | {G.error(Wt, fn(Wt))['rel']:.4f} |")


def test_s2_error_model_and_outliers():
    w = np.random.default_rng(0).standard_normal((128, 256))
    crest = np.mean(np.abs(w).max(1) / np.sqrt((w ** 2).mean(1)))
    meas = {b: G.error(w, G.fake_quant(w, fmt=f"int{b}", convention="full"))["sqnr_db"] for b in (4, 8)}
    present(f"(crest factor {crest:.2f}), the rule predicts {F.sqnr_rule_db(8, crest):.1f} dB at INT8 and the grid "
            f"measures {meas[8]:.1f}; at INT4 it predicts {F.sqnr_rule_db(4, crest):.1f} and the grid measures {meas[4]:.1f}")
    row = np.random.default_rng(3).standard_normal(4096) * 0.02
    out = row.copy()
    out[100] = 0.8
    c0, c1 = np.abs(row).max() / np.sqrt(np.mean(row ** 2)), 0.8 / np.sqrt(np.mean(out ** 2))
    s0 = G.error(row[None], G.fake_quant(row[None], fmt="int8", convention="full"))["sqnr_db"]
    s1 = G.error(out[None], G.fake_quant(out[None], fmt="int8", convention="full"))["sqnr_db"]
    present(f"crest factor of {c0:.1f}", f"it becomes {c1:.1f}", f"falls from {s0:.1f} dB to {s1:.1f} dB (the rule "
            f"predicts {F.sqnr_rule_db(8, c1):.1f})", f"**{20 * np.log10(c1 / c0) / 6.02:.1f} bits**")


def test_s3_columns_activations_and_budget(tiny):
    rng = np.random.default_rng(1)                          # notebook 02's draws, in order
    rng.standard_normal((32, 64))
    Wc = rng.standard_normal((256, 512)) * 0.02
    Wc[:, 7] *= 20
    rest = [c for c in range(512) if c != 7]
    e = lambda gran, g: G.error(Wc[:, rest], G.fake_quant(Wc, fmt="int4", granularity=gran, group_size=g,
                                                          convention="full")[:, rest])["rel"]
    present(f"is {e('channel', 128):.3f} per channel, {e('group', 128):.3f} with groups of 128, "
            f"{e('group', 64):.3f} with 64 and {e('group', 32):.3f} with 32")
    m, *_ = tiny
    A = m.calibration_inputs(m.sample(256, "calib")[0])["blocks.0.up"]
    amax = np.abs(A).max(0)
    out = np.argsort(-amax)[:4]
    normal = np.setdiff1d(np.arange(64), out)
    present(f"absmax {amax[out].min():.0f}–{amax[out].max():.0f} against a median channel absmax of {np.median(amax[normal]):.1f}")
    tok, ten, fp8 = (G.quantize_activations(A, f, p) for f, p in (("int8", "token"), ("int8", "tensor"), ("fp8", "token")))
    present(f"Per-token INT8 gives {pct(G.error(A, tok)['rel'])} error",
            f"but **{pct(G.error(A[:, normal], tok[:, normal])['rel'])}** on the 60 ordinary channels",
            f"Per tensor gives {pct(G.error(A[:, normal], ten[:, normal])['rel'])} on them, and per-token FP8 "
            f"**{pct(G.error(A[:, normal], fp8[:, normal])['rel'])}**")
    Xt, _ = m.sample(2000, "test")
    At, Wu = m.calibration_inputs(Xt)["blocks.0.up"], m.weights["blocks.0.up"]
    ref = At @ Wu.T
    err = lambda **kw: np.linalg.norm(w8a8.w8a8_matmul(At, Wu, "int8", **kw)[0] - ref) / np.linalg.norm(ref)
    st = lambda a: pct(err(act="tensor", static_amax=a), 2)
    present(f"the calibration max gives {st(np.abs(A).max())} error", f"The 99.99th percentile gives "
            f"{st(np.percentile(np.abs(A), 99.99))}", f"gives {pct(err(act='tensor', static_amax=np.percentile(np.abs(A), 99)))}. "
            f"Dynamic per-token scales give **{pct(err(), 2)}**")
    present(f"INT8 per channel, 4,096 inputs, fp16 scale {F.bits_per_weight(8, 4096):.3f}",
            f"{F.bits_per_weight(4, 128)}", f"{F.bits_per_weight(4, 128, zero_point_bits=4)}",
            f"{F.bits_per_weight(4, 32)} / {F.bits_per_weight(4, 32, zero_point_bits=4)}",
            f"MXFP4 (E8M0 per 32) {F.bits_per_weight(4, 32, scale_bits=8)}", f"FP8, FP32 scale per 128 × 128 block "
            f"{F.bits_per_weight(8, 128 * 128, scale_bits=32):.3f}")
    for name, label in (("qwen2.5-0.5b", "Qwen2.5-0.5B (tied embedding)"), ("llama-3.1-8b", "Llama-3.1-8B (untied)"),
                        ("llama-3.1-70b", "Llama-3.1-70B")):
        mm = C.MODELS[name]
        d = 3 if name.startswith("qwen") else (2 if name == "llama-3.1-8b" else 1)
        gb = [f"{mm.weight_bytes(b) / 1e9:.{d}f} GB" for b in (16, 8, 4.125)]
        present(f"| {label} | {gb[0]} | {gb[1]} | {gb[2]} | {mm.weight_bytes(16) / mm.weight_bytes(4.125):.2f}× | "
                f"{pct(mm.embed_params * (1 if mm.tied else 2) / mm.params)}")


def test_s4_gptq_awq_and_calibration(tiny):
    m, X, y, Xc, ref, cap = tiny
    acc = lambda q: float(np.mean(q.forward(X).argmax(1) == y))
    fp = float(np.mean(ref.argmax(1) == y))
    rtn4, rtn3 = acc(quantize_model(m, "rtn", 4, 32)), acc(quantize_model(m, "rtn", 3, 32))
    present(f"INT4 g32 costs {100 * (fp - rtn4):.1f} points ({pct(fp)} → {pct(rtn4)}) and INT3 {100 * (fp - rtn3):.0f} points")
    ks = []
    for name in ("blocks.0.down", "blocks.1.down"):
        ev = np.sort(np.linalg.eigvalsh(gptq.hessian(cap[name])))[::-1]
        ks.append(int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.99)) + 1)
    present(f"lies in {min(ks)}–{max(ks)} directions")
    W, Xn = m.weights["blocks.0.down"], cap["blocks.0.down"]
    Xh = m.calibration_inputs(X[:1000])["blocks.0.down"]
    g = gptq.gptq(W, Xn, bits=4, group_size=32).w_hat
    present(f"({G.output_error(Xn, W, gptq.rtn(W, 4, 32).w_hat):.4f} → {G.output_error(Xn, W, g):.4f}, "
            f"{G.output_error(Xh, W, g):.4f} on held-out data)")
    r4, g4 = (E.compare(ref, quantize_model(m, meth, 4, 32, calib=Xc).forward(X), y) for meth in ("rtn", "gptq"))
    g3 = E.compare(ref, quantize_model(m, "gptq", 3, 32, calib=Xc).forward(X), y)
    present(f"INT4 g32 recovers to **{pct(g4['acc'])}** (KL {r4['kl']:.3f} → {g4['kl']:.3f}) and INT3 to **{pct(g3['acc'])}**")
    s, alpha, losses = awq.search_scale(m.weights["blocks.0.up"], cap["blocks.0.up"], 4, 32, symmetric=True)
    present(f"minimum at α = {alpha:.2f}, {min(losses) / losses[0]:.2f} of RTN's loss")
    Wu, Xu = m.weights["blocks.0.up"], cap["blocks.0.up"]
    q, s_ = awq.awq(Wu, Xu, bits=4, group_size=32, symmetric=True)
    present(f"({G.output_error(Xu, Wu, gptq.rtn(Wu, 4, 32).w_hat):.4f} → {G.output_error(Xu / s_, Wu * s_, q.w_hat):.4f})")
    ups = ["blocks.0.up", "blocks.1.up"]
    a3, g3u = (acc(quantize_model(m, meth, 3, 32, calib=Xc, targets=ups)) for meth in ("awq", "gptq"))
    present(f"it beats GPTQ: {pct(a3)} against {pct(g3u)}")
    present(f"for the lowest KL of all: {E.compare(ref, quantize_model(m, 'awq+gptq', 4, 32, calib=Xc).forward(X), y)['kl']:.4f}")
    by_n = {n: pct(acc(quantize_model(m, "gptq", 3, 32, calib=m.sample(n, "calib")[0]))) for n in (16, 64, 256, 1024)}
    present(f"GPTQ INT3 gets {by_n[16]} with 16 samples, {by_n[64]} with 64, {by_n[256]} with 256 and {by_n[1024]} with 1,024")
    Xa, ya = m.sample(4096, "calib")
    narrow = pct(acc(quantize_model(m, "gptq", 3, 32, calib=Xa[ya < 2][:256])))
    noise = pct(acc(quantize_model(m, "gptq", 3, 32, calib=np.random.default_rng(1).standard_normal((256, 64)) * 1.6)))
    present(f"it still gets {narrow}, and with 256 samples of pure noise {noise}")


def test_s5_epilogue_smoothquant_fp8_and_head(tiny):
    m, X, y, Xc, ref, cap = tiny
    rng = np.random.default_rng(0)                          # notebook 04's draws, in order
    Xa, Wa = rng.standard_normal((16, 4096)), rng.standard_normal((1024, 4096)) * 0.02
    fake = G.quantize_activations(Xa, "int8") @ G.fake_quant(Wa, fmt="int8").T
    assert np.abs(w8a8.w8a8_matmul(Xa, Wa, "int8")[0] - fake).max() < 1e-14
    present(f"up to K = {(2 ** 31 - 1) // (127 * 127):,}")
    Xu, Wu = cap["blocks.0.up"], m.weights["blocks.0.up"]
    present(f"falls from {pct(S.w8a8_error(Xu, Wu), 2)} to **{pct(S.w8a8_error(Xu, Wu, alpha=0.5), 2)}** at α = 0.5")

    def w8a8_logits(model, fmt):
        wq = {n: G.fake_quant(model.weights[n], fmt=fmt, granularity="channel") for n in model.linears()}
        aq = lambda name, x: x if name == "head" else G.quantize_activations(x, fmt, "token")
        return model.forward(X, act_quant=aq, weights=wq)

    sm = m
    for name in ("blocks.0.up", "blocks.1.up"):
        A = sm.calibration_inputs(Xc)[name]
        sm = sm.with_weights(sm.fold(name, S.smooth_scales(np.abs(A).max(0), np.abs(sm.weights[name]).max(0), 0.5)))
    k_int8, k_sm, k_fp8 = E.kl(ref, w8a8_logits(m, "int8")), E.kl(ref, w8a8_logits(sm, "int8")), E.kl(ref, w8a8_logits(m, "fp8"))
    present(f"KL falls {k_int8 / k_sm:.1f}× ({k_int8:.5f} → {k_sm:.5f})",
            f"FP8 W8A8 has {k_fp8 / k_int8:.0f}× the KL of INT8 W8A8 ({k_fp8:.4f} against {k_int8:.5f})")
    Xb, Wb = rng.standard_normal((16, 512)), rng.standard_normal((512, 512))
    Wb[:128, :128] *= 30
    Xb[:, :128] *= 20
    rb = Xb @ Wb.T
    rel = lambda Yb: np.linalg.norm(Yb - rb) / np.linalg.norm(rb)
    fp8s = [rel(w8a8.block_fp8_matmul(Xb, Wb)), rel(w8a8.w8a8_matmul(Xb, Wb, "fp8")[0]),
            rel(w8a8.w8a8_matmul(Xb, Wb, "fp8", act="tensor", weight="tensor")[0])]
    present(f"all give {100 * min(fp8s):.1f}–{100 * max(fp8s):.1f}% output error, while INT8 per token and channel gives "
            f"{pct(rel(w8a8.w8a8_matmul(Xb, Wb, 'int8')[0]))}")
    b200 = {r["scheme"]: r for r in C.table(C.GPUS["B200"], M8, kv=(8,))}
    present(f"prefill of {b200['bf16']['prefill_ms']:.0f} ms in BF16, {b200['w8a8-fp8']['prefill_ms']:.0f} ms in FP8 and "
            f"{b200['w4a4-nvfp4']['prefill_ms']:.0f} ms in NVFP4")
    fp = float(np.mean(ref.argmax(1) == y))
    head = float(np.mean(quantize_model(m, "rtn", 4, None, targets=["head"]).forward(X).argmax(1) == y))
    hidden = float(np.mean(quantize_model(m, "rtn", 4, None).forward(X).argmax(1) == y))
    present(f"costs {100 * (fp - head):.1f} points, against {100 * (fp - hidden):.1f} for all four hidden linears")


def test_s6_kv_cache():
    per_tok = [M8.kv_bytes_per_token(b) for b in (16, 8, 5, 3)]
    present(f"is {per_tok[0]:,.0f} B per token in BF16 and {per_tok[1]:,.0f} in FP8", f"({per_tok[2]:,.0f} B per token)",
            f"costs {K.kv_bits_per_element(2, 32, 16, 16):.0f} ({per_tok[3]:,.0f})")
    sess = [C.sessions(L4, M8, "w8a8-fp8", 2000, b) for b in (16, 8, 5, 3)]
    present("sessions go " + " → ".join(str(s) for s in sess))
    t = [C.step_cost(H100, M8, "w8a8-fp8", [(8000, 1)] * 32, kv)["t"] * 1e3 for kv in (16, 8)]
    present(f"takes a decode step from {t[0]:.1f} ms to {t[1]:.1f} ms")
    Q, Kc, V = K.synthetic_qkv()
    err = lambda Kh, Vh, Vr=V: K.attention_error(Q, Kc, Vr, Kh, Vh)
    present(f"| FP8 K and V, calibrated per-tensor scales | {pct(err(K.fp8_kv(Kc), K.fp8_kv(V)), 2)} |",
            f"| … keys only / values only | {pct(err(K.fp8_kv(Kc), V), 2)} / {pct(err(Kc, K.fp8_kv(V)), 2)} |",
            f"| FP8, scale 1.0 (vLLM's default), values of order 1 | {pct(err(K.fp8_kv(Kc, 1.0), K.fp8_kv(V, 1.0)), 2)} |")
    for f, label in ((1e-3, "values ×10⁻³ | "), (1e3, "values ×10³ | ")):
        Vf = V * f
        d = 1 if f < 1 else 0
        present(f"{label}{pct(err(Kc, K.fp8_kv(Vf, 1.0), Vf), d)}", f"calibrated: {pct(err(Kc, K.fp8_kv(Vf, 'tensor'), Vf), 2)} |")
    kv = lambda b, ax: pct(K.attention_error(Q, Kc, V, *K.kivi(Kc, V, b, 32, 32, key_axis=ax)), 2 if b == 4 else 1)
    present(f"4-bit keys per channel give {kv(4, 0)} error against {kv(4, 1)} per token, and 2-bit {kv(2, 0)} against {kv(2, 1)}")


def test_s7_s9_bits_and_packing():
    present(f"costs {F.bits_per_weight(4, 64, scale_bits=32)} bits per weight",
            f"it costs {F.bits_per_weight(4, 64, scale_bits=8 + 32 / 256):.3f}")
    assert int(F.pack_int4(np.array([[-8, -7, 0, 1, 2, 3, 4, 7]])).view(np.uint32)[0, 0]) == 0xFCBA9810
    present("[−8,−7,0,1,2,3,4,7] → 0xfcba9810", "[4096, 896] INT4 weight → weight_packed [4096, 112]")
    assert F.pack_int4(np.zeros((4096, 896), int)).shape == (4096, 112)
    assert [int(b) for b in F.pack_fp4(np.array([0.5, -6.0, 1.5, 0.0]))] == [0xF1, 0x03]
    present("[0.5, −6, 1.5, 0] → 0xf1, 0x03")
    present(f"(`q4_0` 18 B per 32 = {F.bits_per_weight(4, 32)} bits, `q8_0` {F.bits_per_weight(8, 32)}")
    assert 18 * 8 / 32 == F.bits_per_weight(4, 32)


def test_s8_eval_numbers(tiny):
    m, X, y, Xc, ref, _ = tiny
    r = E.compare(ref, quantize_model(m, "rtn", 4, 32).forward(X), y)
    g = E.compare(ref, quantize_model(m, "gptq", 4, 32, calib=Xc).forward(X), y)
    present(f"INT4 RTN has KL {r['kl']:.3f}, top-1 agreement {pct(r['top1'])} and accuracy {pct(r['acc'])}: it lost "
            f"{r['lost']} right answers and gained {r['gained']}",
            f"GPTQ has KL {g['kl']:.3f}, {pct(g['top1'])} and {pct(g['acc'])}, losing {g['lost']} and gaining {g['gained']}")
    present(f"±{E.accuracy_stderr(0.768, 250):.4f}", f"~{100 * 2 * E.accuracy_stderr(0.768, 250):.0f}")
    n = next(n for n in range(100, 100000) if 2 * E.accuracy_stderr(0.77, n) <= 0.01)
    present(f"takes {n:,} items")
    i8 = E.compare(ref, quantize_model(m, "rtn", 8, None).forward(X), y)
    assert E.within_budget(i8, max_kl=0.05) and E.within_budget(g, max_kl=0.05) and not E.within_budget(r, max_kl=0.05)


def test_s10_decisions_and_cost():
    pick = C.choose(C.table(L4, M8), min_sessions=48, max_prefill_ms=300)
    present(f"FP8 W8A8 with FP8 KV: {pick['sessions']} sessions, {pick['prefill_ms']:.0f} ms")
    t4 = {r["scheme"]: r for r in C.table(T4, M8, kv=(16,))}
    assert C.choose(list(t4.values()), min_sessions=20, max_prefill_ms=700)["scheme"] == "w4a16"
    present(f"weights leave room for {t4['w8a16-fp8']['sessions']} sessions, INT4 for {t4['w4a16']['sessions']}")
    present(f"{M70.weight_bytes(16) / 1e9:.1f} GB in BF16 and {M70.weight_bytes(8) / 1e9:.1f} GB in FP8",
            f"({M70.weight_bytes(4.125) / 1e9:.1f} GB) serves {C.sessions(H100, M70, 'w4a16', 4000, 8)} of them")
    assert C.sessions(H100, M70, "w8a8-fp8", 4000, 8) == 0
    rows = {}
    for s, kv in (("bf16", 16), ("w8a8-fp8", 8)):
        b = C.sessions(L4, M8, s, 2000, kv)
        tps = b / C.step_cost(L4, M8, s, [(1000, 1)] * b, kv)["t"]
        rows[s] = (b, tps, C.cost_per_million(0.70, tps))
    (b0, t0, c0), (b1, t1, c1) = rows["bf16"], rows["w8a8-fp8"]
    present(f"BF16 fits {b0} sessions, {t0:,.0f} tokens/s, **${c0:.3f}/M**",
            f"FP8 weights plus FP8 KV fit {b1} sessions, {t1:,.0f} tokens/s, **${c1:.3f}/M**",
            f"That is {c0 / c1:.1f}× cheaper", f"a {b1 / b0:.0f}× larger batch")
