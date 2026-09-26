"""Every computed number the topic's PRIMER.md quotes is recomputed here and must appear verbatim.

If a formula, a config or a seed changes, the primer fails this test until it is updated. (Inputs and cited
facts -- 675B / 41B for Mistral Large 3, DeepEP's 77 us, published totals -- are quoted, not computed.)
"""
import math
from pathlib import Path

import numpy as np
import pytest

from moecore import ep as E
from moecore import routing as R
from moecore import sizing as S
from moecore import touched as T
from moecore import train as TR
from moecore.moe import ROUTERS, route

PRIMER_PATH = Path(__file__).resolve().parents[2] / "PRIMER.md"
PRIMER = " ".join(PRIMER_PATH.read_text(encoding="utf-8").split()) if PRIMER_PATH.exists() else ""
M, D, L = S.MODELS, T.DEVICES, E.LINKS
MIX, Q3, DS, L8 = M["mixtral-8x7b"], M["qwen3-30b-a3b"], M["deepseek-v3"], M["llama-3.1-8b"]
H200, H100 = D["h200"], D["h100-sxm"]
B = lambda x: f"{x / 1e9:.2f}B"

pytestmark = pytest.mark.skipif(not PRIMER, reason="PRIMER.md not beside this core")


def present(*fragments):
    missing = [f for f in fragments if " ".join(f.split()) not in PRIMER]
    assert not missing, f"PRIMER.md no longer says: {missing}"


def test_s1_why_sparsity():
    pf = S.prefill_flops(41e9, 2048)
    present(f"= {pf / 1e14:.2f} × 10¹⁴ FLOPs", f"{675 / 41:.1f}× less",
            f"**{S.decode_floor(675e9, 8, H200) * 1e3:.1f} ms** per step across 8 H200s")
    present(f"| HBM for weights | total parameters | {B(MIX.total())} → {S.weight_bytes(MIX) / 1e9:.1f} GB bf16 | "
            f"{B(Q3.total())} → {S.weight_bytes(Q3) / 1e9:.1f} GB bf16 | {B(DS.total())} → {S.weight_bytes(DS, 8) / 1e9:.1f} GB FP8 |",
            f"| {B(MIX.active())} | {B(Q3.active())} | {B(DS.active())} |",
            f"| {MIX.total() / MIX.active():.1f} | {Q3.total() / Q3.active():.1f} | {DS.total() / DS.active():.1f} |",
            f"| {MIX.kv_bytes_per_token():,.0f} B | {Q3.kv_bytes_per_token():,.0f} B | {DS.kv_bytes_per_token():,.0f} B (MLA latent) |",
            f"both cache exactly 2 × 32 × 8 × 128 × 2 = {L8.kv_bytes_per_token():,.0f} bytes")


def test_s2_the_layer():
    logits = np.array([[2.0, 1.2, 0.4, 0.3, -0.5, -1.0, -1.1, -2.0]])
    w = {f: route(logits, 1 if f == "llama4" else 2, **o).weights[0] for f, o in ROUTERS.items()}
    fmt = lambda a, n: "[" + ", ".join(f"{x:.{n}f}" for x in a) + "]"
    present(f"| {fmt(w['mixtral'], 2)} |", f"| {fmt(w['qwen3'], 3)}, sum {w['qwen3'].sum():.3f} |",
            f"| {fmt(w['deepseek-v3'], 3)}, sum {w['deepseek-v3'].sum():.1f} |", f"| {fmt(w['llama4'], 3)} (k = 1) |")
    assert np.allclose(w["gpt-oss"], w["mixtral"])
    rb = route(logits, 2, select_bias=np.array([0, 0, 0, 0, 0, 0, 0, 1.0]), **ROUTERS["deepseek-v3"])
    present(f"{rb.weights[0][list(rb.idx[0]).index(7)]:.3f} of the 2.5")
    present(f"= {MIX.expert_params():,}", f"= {MIX.attn_params():,}", f"= {MIX.router_params():,}",
            f"= {MIX.total():,} ({B(MIX.total())})", f"= {MIX.active():,} ({B(MIX.active())})",
            f"MLA per layer is {DS.attn_params():,} parameters", f"one expert {DS.expert_params():,}",
            f"Total **{B(DS.total())}** and active **{B(DS.active())}**")
    for key, extra in (("mixtral-8x7b", "12.9B"), ("qwen3-30b-a3b", "3.3B"), ("deepseek-v3", "37B"), ("olmoe-1b-7b", "1.3B / 6.9B")):
        c = M[key]
        present(f"| {B(c.total())} | {B(c.active())} | {B(c.active('head'))} | {B(c.active('none'))} | {extra} |")
    oss = M["gpt-oss-120b"]
    present(f"| {B(oss.total())} | {B(oss.active())} | **{B(oss.active('head'))}** | {B(oss.active('none'))} | 5.1B |",
            f"is {oss.vocab * oss.d_model / 1e9:.2f}B per table — {oss.vocab * oss.d_model / oss.active('head'):.0%} of")
    present(f"from {math.comb(8, 2)} to {math.comb(64, 16) / 1e14:.2f} × 10¹⁴")


def test_s3_balance():
    t, e = 64, 8
    spread = np.array([[(i * 2 + j) % e for j in range(2)] for i in range(t)])
    lumped_p = np.zeros((t, e)); lumped_p[:, :2] = 0.5
    lumped = np.tile([0, 1], (t, 1))
    present(f"| uniform | {R.switch_aux_loss(np.full((t, e), 1 / e), spread):.2f} | "
            f"{R.switch_aux_loss(np.full((t, e), 1 / e), spread, 'megatron'):.2f} |",
            f"| two experts take everything | {R.switch_aux_loss(lumped_p, lumped):.2f} | "
            f"{R.switch_aux_loss(lumped_p, lumped, 'megatron'):.2f} |")
    present(f"(ln 8)² = {R.z_loss(np.zeros((4, 8))):.3f}", f"raises it to {R.z_loss(np.full((4, 8), 20.0)):.0f}")
    logits = np.random.default_rng(1).standard_normal((256, 8)) + np.array([1.5, 0.8, 0, 0, 0, 0, 0, 0])
    rr = route(logits, 2, **ROUTERS["mixtral"])
    loads = R.load(rr.idx, 8)
    present(f"(hottest expert {loads.max()} assignments against a mean of {loads.mean():.0f})",
            f"({loads.max()} / {loads.mean():.0f} = {loads.max() / loads.mean():.2f}; 3.00 in steps of 0.25)")
    for f in (1.0, 1.25, 2.0):
        cap = R.capacity(256, 8, 2, f)
        present(f"| {f:.2f} | {cap} | {1 - R.apply_capacity(rr.idx, rr.weights, cap).mean():.1%} |")
    padded = R.align_block_size(rr.idx, 16, 8)[2]
    present(f"512 assignments become {padded} rows at block 16, {padded / 512 - 1:.1%} padding")
    scores = np.random.default_rng(2).standard_normal((512, 8)) + np.array([2.0, 1.0, 0, 0, 0, 0, 0, 0])
    scores = np.exp(scores) / np.exp(scores).sum(1, keepdims=True)
    before = R.load(np.argsort(-scores, axis=1)[:, :2], 8)
    bias = np.zeros(8)
    for _ in range(400):
        c = R.load(np.argsort(-(scores + bias), axis=1)[:, :2], 8)
        bias = R.update_bias(bias, c, 0.002)
    present(f"[{before[0]}, {before[1]}, {before[2]}, …] over 8 experts",
            f"bring max/mean load to {R.stats(c)['max_over_mean']:.2f}")
    touched, hot = T.touched_mc(128, 8, 16, s=1.0)
    present(f"s = 1.0 touches {touched:.1f} experts instead of {T.experts_touched(128, 8, 16):.1f} and loads the "
            f"hottest one {hot:.1f}× the mean")


def test_s3_the_toy_trainer():
    task = TR.make_task(seed=6)
    runs = {b: TR.train(task, balance=b) for b in ("none", "aux", "bias")}
    s0 = sorted(runs["none"].share[0], reverse=True)
    present(f"split {s0[0] * 100:.0f}/{s0[1] * 100:.0f}", f"one expert carries {runs['none'].final_share.max():.0%} of the tokens",
            f"task loss stays at {runs['none'].loss[-1]:.3f}", f"lowers the task loss to {runs['aux'].loss[-1]:.3f} at seed 6",
            f"({runs['bias'].loss[-1]:.3f} at seed 6)")


def test_s5_which_experts_a_step_touches():
    s1 = T.decode_step(MIX, H200, 1, 1024)
    present(f"{s1.bytes:,.0f} bytes at batch 1", f"of which {T.streamed_weight_bytes(MIX, 1):,.0f} are weights")
    t_mix = math.ceil(math.log(1 - 7.5 / 8) / math.log(0.75))
    t_q = math.ceil(math.log(1 - 120 / 128) / math.log(1 - 8 / 128))
    present(f"from batch {t_mix}, Qwen3 120 of 128 from batch {t_q}")
    ds = [f"{T.experts_touched(256, 8, n):.1f}" for n in (1, 8, 32, 128, 256)]
    present(f"touches {ds[0]}, {ds[1]}, {ds[2]}, {ds[3]} and {ds[4]} experts")
    skew, _ = T.touched_mc(128, 8, 16, s=1.0)
    sp = T.decode_step(Q3, H200, 16, 1024).time / T.decode_step(Q3, H200, 16, 1024, touched=skew).time
    present(f"reads {skew:.1f} experts per layer instead of 82.4, a {sp:.2f}× faster step")
    x = {m.name: T.decode_crossover_batch(m, H200, 0) for m in (L8, MIX, Q3)}
    ratio = lambda m: (m.total() - m.vocab * m.d_model) / (m.active() - m.vocab * m.d_model)
    present(f"| Mixtral-8x7B | {MIX.total() / MIX.active():.1f} | {ratio(MIX):.1f} | {x['Mixtral-8x7B']} |",
            f"| Qwen3-30B-A3B | {Q3.total() / Q3.active():.1f} | {ratio(Q3):.1f} | {x['Qwen3-30B-A3B']:,} |",
            f"| 1.0 | 1.0 | {x['Llama-3.1-8B']} |", f"207 × 9.9 ≈ {207 * ratio(Q3):,.0f}")
    rows = [256 * k / e for e, k in ((8, 2), (128, 8), (256, 8))]
    need = [f"{math.ceil(206 * e / k):,}" for e, k in ((8, 2), (128, 8), (256, 8))]
    present(f"that is {rows[0]:.0f} rows for Mixtral, {rows[1]:.0f} for Qwen3 and {rows[2]:.0f} for DeepSeek-V3",
            f"batches of {need[0]}, {need[1]} and {need[2]} tokens")
    present(f"the KV cache is {T.kv_share(MIX, 64, 1024):.1%} of a Mixtral step but {T.kv_share(L8, 64, 1024):.1%} of a "
            f"Llama-3.1-8B step; at 32K context, {T.kv_share(MIX, 64, 32768):.1%} against {T.kv_share(L8, 64, 32768):.1%}")


def test_s6_running_moe_on_gpus():
    nv, ib = L["nvlink4"], L["ib-ndr"]
    s = E.dispatch_bytes(256, 2, 4096)
    present(f"{s // 2 ** 20} MiB per direction, **{E.a2a_time(s, 8, nv, 'pairwise') * 1e6:.0f} µs pairwise or "
            f"{E.a2a_time(s, 8, nv) * 1e6:.0f} µs direct**")
    for tok, lk, label in ((8, nv, "NVLink 4"), (8, ib, "400 Gb/s IB"), (4096, nv, "NVLink 4"), (4096, ib, "400 Gb/s IB")):
        d, dd, dc = E.dispatch_bytes(tok, 2, 4096), E.dispatch_bytes(tok, 8, 7168, 1, 128), E.dispatch_bytes(tok, 8, 7168, 2)
        mib = lambda b: f"{b / 2 ** 20:.2f}" if tok == 8 else f"{b / 2 ** 20:.0f}"
        us = lambda b: f"{E.a2a_time(b, 8, lk) * 1e6:,.1f}"
        present(f"| {label} | {mib(d)} MiB, {us(d)} µs | {mib(dd)} MiB, {us(dd)} µs | {mib(dc)} MiB, {us(dc)} µs |")
    d, c = E.dispatch_bytes(128, 8, 7168, 1, 128), E.dispatch_bytes(128, 8, 7168, 2)
    present(f"= {d:,} B in 77 µs is **{d / 77e-6 / 1e9:.1f} GB/s**", f"{c:,} B in 114 µs, is {c / 114e-6 / 1e9:.1f} GB/s")
    rng = np.random.default_rng(0)
    pop = T.zipf_popularity(128, 1.0)[rng.permutation(128)]
    origin = np.repeat(np.arange(8), 128)
    uni = T.sample_routes(128, 8, 1024, None, np.random.default_rng(1))
    skew = T.sample_routes(128, 8, 1024, pop, np.random.default_rng(1))
    lt = {}
    for name, idx in (("uni", uni), ("skew", skew)):
        for strat in ("linear", "round_robin"):
            lt[name, strat] = E.layer_time(E.exchange(idx, origin, E.placement(128, 8, strat), 8), Q3.expert_params(), 2048, H100, nv)
    present(f"| uniform | linear | {lt['uni', 'linear'].imbalance:.2f} | {lt['uni', 'linear'].total * 1e6:.1f} µs |",
            f"| {lt['skew', 'linear'].imbalance:.2f} | {lt['skew', 'linear'].total * 1e6:.1f} µs |",
            f"| round-robin | {lt['skew', 'round_robin'].imbalance:.2f} | {lt['skew', 'round_robin'].total * 1e6:.1f} µs |")
    sl = lt["skew", "linear"]
    present(f"the layer {sl.total / (2 * sl.comm + sl.compute.mean()):.2f}× slower than a balanced one")
    load = np.bincount(skew.ravel(), minlength=128)
    r0, r8 = E.rebalance(load, 8), E.rebalance(load, 8, 8)
    present(f"from {r0.max() / r0.mean():.3f} to {r8.max() / r8.mean():.3f} max/mean with 8 redundant experts")
    extra = E.wide_ep_weights(DS, 32, 1, redundant=32) - E.wide_ep_weights(DS, 32, 1)
    present(f"58 MoE layers × {DS.expert_params():,} B in FP8 = {extra / 2 ** 30:.2f} GiB")
    tp, epc = E.moe_comm(64, 2, 4096, 8, nv, "tp"), E.moe_comm(8, 2, 4096, 8, nv, "ep")
    present(f"sends {tp[0] / 1024:.0f} KiB per GPU in {tp[1] * 1e6:.1f} µs", f"sends {epc[0] / 1024:.0f} KiB per GPU in {epc[1] * 1e6:.1f} µs")
    for n in (8, 16, 32, 64):
        w = E.wide_ep_weights(DS, n, 1)
        present(f"| {n} | {w / 1e9:.1f} GB | {w / 141e9:.0%} | {S.sessions(DS, H200, n, 4096, 8, layout='ep'):,} |")
    replicated = DS.total() - DS.moe_layers * DS.n_experts * DS.expert_params()
    present(f"{replicated / 1e9:.1f}B parameters are replicated on every rank")
    a = E.decode_on(DS, H200, 512, 4096, 16, "ep", nv, 1)
    b = E.decode_on(DS, H200, 512, 4096, 16, "ep", ib, 1)
    present(f"spends {a['comm'] * 1e3:.1f} ms per step in all-to-alls on NVLink but {b['comm'] * 1e3:.1f} ms over 400 Gb/s "
            f"InfiniBand, {b['comm'] / b['time']:.0%} of the step")
    o = M["olmoe-1b-7b"]
    gib = (4 * 4096 * o.kv_bytes_per_token() - (0.9 * 16e9 - S.weight_bytes(o))) / 2 ** 30
    pen, step = S.offload_step_s(gib, 12) * 1e3, T.decode_step(o, D["t4"], 1, 4096, precision="fp16").time * 1e3
    present(f"fp16 ({S.weight_bytes(o) / 1e9:.1f} GB)", f"offloading {gib:.2f} GiB to fit four adds ~{pen:.0f} ms per step",
            f"to a {step:.0f} ms step — about {(step + pen) / step:.0f}× slower")
    present(f"gpt-oss-120b **{S.weight_bytes(M['gpt-oss-120b'], 16, 4.25) / 1e9:.1f} GB**",
            f"gpt-oss-20b **{S.weight_bytes(M['gpt-oss-20b'], 16, 4.25) / 1e9:.1f} GB**")


def test_s7_sizing_and_cost():
    present(f"{S.weight_bytes(Q3) / 1e9:.1f} GB in bf16, {S.weight_bytes(Q3, 8) / 1e9:.1f} GB in FP8, "
            f"{S.weight_bytes(Q3, 16, 4.25) / 1e9:.1f} GB with 4.25-bit experts",
            f"with room for {S.sessions(Q3, D['l4'], 1, 4096, 16, 4.25)} sequences of 4K tokens")
    ratio = S.prefill_flops(M["llama-3.1-70b"].active(), 8192) / S.prefill_flops(MIX.active(), 8192)
    ms = S.prefill_flops(MIX.active(), 8192) / (H100.peak() * 0.5) * 1e3
    present(f"{ratio:.2f}× the FLOPs it costs Mixtral-8x7B", f"Mixtral prefills it in {ms:.0f} ms")
    cases = [(MIX, H100, 64, 50, "nvlink4", 16, 3.0, "ep"), (M["llama-3.1-70b"], H100, 64, 50, "nvlink4", 16, 3.0, "tp"),
             (DS, H200, 256, 50, "nvlink4", 8, 3.0, "ep"), (Q3, D["l4"], 32, 60, "pcie4x16", 8, 0.7, "ep")]
    for cfg, dev, b, itl, lk, bits, usd, lay in cases:
        p = S.plan(cfg, dev, b, 4096, itl, L[lk], bits, usd_per_gpu_hr=usd, layout=lay)
        present(f"| {b} | {itl} ms | {p.gpus} | {p.step_ms:.1f} ms | {p.tok_s:,.0f} | {p.usd_per_mtok:.2f} at ${usd:g}/GPU-h |")
    present(f"its all-experts floor ({S.decode_floor(S.weight_bytes(DS, 8), 8, H200) * 1e3:.1f} ms")
    comm = E.decode_on(DS, H200, 256, 4096, 8, "ep", L["nvlink4"], 1)["comm"]
    present(f"and {comm * 1e3:.1f} ms of all-to-alls")
    out = []
    for c in (4096, 32_768, 131_072):
        fit = S.sessions(MIX, H100, 2, c, layout="ep")
        b = min(16, fit)
        out.append((fit, b, b / E.decode_on(MIX, H100, b, c, 2, "ep", L["nvlink4"])["time"]))
    present(f"at 4K context {out[0][0]} sequences fit and batch 16 runs at {out[0][2]:,.0f} tokens/s",
            f"at 32K only {out[1][0]} fit ({out[1][2]:.0f} tokens/s at batch {out[1][1]})",
            f"at 128K, {out[2][0]} ({out[2][2]:.0f} tokens/s)")


def test_s8_failure_modes():
    a, b = T.decode_step(MIX, H200, 1, 1024).time, T.decode_step(L8, H200, 1, 1024).time
    present(f"{a * 1e3:.2f} ms vs Llama-3.1-8B's {b * 1e3:.2f} ms ({a / b:.2f}×)")
