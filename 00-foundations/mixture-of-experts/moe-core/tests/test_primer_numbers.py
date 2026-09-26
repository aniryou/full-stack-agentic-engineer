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
    present(f"| {fmt(w['mixtral'], 2)} |", f"| {fmt(w['olmoe'], 3)}, sum {w['olmoe'].sum():.3f} |",
            f"| {fmt(w['deepseek-v3'], 3)}, sum {w['deepseek-v3'].sum():.1f} |", f"| {fmt(w['llama4'], 3)} (k = 1) |")
    assert np.allclose(w["gpt-oss"], w["mixtral"]) and np.allclose(w["qwen3-moe"], w["mixtral"])
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
    s1, s64 = T.decode_step(MIX, H200, 1, 1024), T.decode_step(MIX, H200, 64, 1024)
    present(f"at batch 1 Mixtral reads {T.experts_touched(8, 2, 1):.2f} of 8 experts per layer, {s1.bytes:,.0f} bytes of which "
            f"{T.streamed_weight_bytes(MIX, 1):,.0f} are weights, in {s1.time * 1e3:.2f} ms; at batch 64 it reads "
            f"{T.experts_touched(8, 2, 64):.2f} of 8, {s64.bytes / 1e9:.1f} GB, in {s64.time * 1e3:.2f} ms. Qwen3 reads "
            f"{T.experts_touched(128, 8, 1):.1f} and {T.experts_touched(128, 8, 64):.1f} of 128")
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
    present(f"gives {x['Llama-3.1-8B']} for Llama-3.1-8B, {x['Mixtral-8x7B']} for Mixtral-8x7B and {x['Qwen3-30B-A3B']:,} for Qwen3-30B-A3B",
            f"streamed ÷ multiplied is {ratio(MIX):.1f} for Mixtral and {ratio(Q3):.1f} for Qwen3, against total ÷ active of "
            f"{MIX.total() / MIX.active():.1f} and {Q3.total() / Q3.active():.1f}",
            f"207 × 9.9 ≈ {207 * ratio(Q3):,.0f}", f"207 × 9.1 would be {1 - 207 * Q3.total() / Q3.active() / x['Qwen3-30B-A3B']:.0%} low")
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
    pop = T.zipf_popularity(128, 1.0)[np.random.default_rng(0).permutation(128)]
    lt, loads = {}, {}
    for tpg in (128, 4096):
        origin = np.repeat(np.arange(8), tpg)
        for name, popularity in (("uni", None), ("skew", pop)):
            idx = T.sample_routes(128, 8, 8 * tpg, popularity, np.random.default_rng(1))
            loads[tpg, name] = np.bincount(idx.ravel(), minlength=128)
            for strat in ("linear", "round_robin"):
                lt[tpg, name, strat] = E.layer_time(idx, origin, E.placement(128, 8, strat), 8, Q3.expert_params(),
                                                    2048, H100, nv)
    by = {"memory": "weight reads", "compute": "FLOPs"}
    row = lambda x: f"| {x.imbalance:.2f} | {by[x.bound]} | {x.total * 1e6:,.1f} µs | {x.penalty:.2f}× |"
    present(f"| 128 (decode) | uniform | linear {row(lt[128, 'uni', 'linear'])}",
            f"| 128 (decode) | Zipf s = 1.0 | linear (contiguous blocks, vLLM's default) {row(lt[128, 'skew', 'linear'])}",
            f"| 4,096 (prefill chunk) | uniform | linear {row(lt[4096, 'uni', 'linear'])}",
            f"| 4,096 (prefill chunk) | Zipf s = 1.0 | linear {row(lt[4096, 'skew', 'linear'])}",
            f"| 4,096 (prefill chunk) | Zipf s = 1.0 | round-robin (not applied to this model; below) {row(lt[4096, 'skew', 'round_robin'])}")
    d, pf = lt[128, "skew", "linear"], lt[4096, "skew", "linear"]
    assert d.bound == "memory" and np.ptp(d.rank) == 0 and pf.bound == "compute"
    present(f"every rank spends {d.memory.max() * 1e6:.1f} µs streaming its 16 experts' weights",
            f"({d.compute.max() * 1e6:.1f} µs of FLOPs against a mean of {d.compute.mean() * 1e6:.1f})",
            f"({d.comm * 1e6:.1f} µs per all-to-all against {d.comm_even * 1e6:.1f}), so the layer is {d.penalty:.2f}× slower",
            f"the busiest rank's {pf.imbalance:.2f}× rows make the layer {pf.penalty:.2f}× slower",
            f"move the hot spot ({pf.imbalance:.2f} → {lt[4096, 'skew', 'round_robin'].imbalance:.2f} rows)",
            f"the H100's ridge of {H100.ridge():.0f}")
    r0, r8 = E.rebalance(loads[128, "skew"], 8), E.rebalance(loads[128, "skew"], 8, 8)
    present(f"from {d.imbalance:.2f} to {r0.max() / r0.mean():.2f} max/mean; eight redundant copies take it to {r8.max() / r8.mean():.2f}")
    extra = E.wide_ep_weights(DS, 32, 1, redundant=32) - E.wide_ep_weights(DS, 32, 1)
    present(f"58 MoE layers × {DS.expert_params():,} B in FP8 = {extra / 2 ** 30:.2f} GiB")
    tp, a2a, agrs = (E.moe_comm(64, 2, 4096, 8, nv, "tp"), E.moe_comm(8, 2, 4096, 8, nv, "a2a"),
                     E.moe_comm(8, 2, 4096, 8, nv, "agrs"))
    present(f"sends {tp[0] / 1024:.0f} KiB per GPU in {tp[1] * 1e6:.1f} µs", f"sends {a2a[0] / 1024:.0f} KiB per GPU in {a2a[1] * 1e6:.1f} µs",
            f"the same {agrs[0] / 1024:.0f} KiB and {agrs[1] * 1e6:.1f} µs as TP")
    assert agrs[0] == tp[0]
    for n in (8, 16, 32, 64):
        w = E.wide_ep_weights(DS, n, 1)
        present(f"| {n} | {w / 1e9:.1f} GB | {w / 141e9:.0%} | {S.sessions(DS, H200, n, 4096, 8, layout='ep'):,} |")
    replicated = DS.total() - DS.moe_layers * DS.n_experts * DS.expert_params()
    present(f"{replicated / 1e9:.1f}B parameters are replicated on every rank")
    fp8 = dict(dispatch_elem=1, scale_block=128)
    one = E.decode_on(DS, H200, 512, 4096, 16, "ep", nv, 1, **fp8)
    two = E.decode_on(DS, H200, 512, 4096, 16, "ep", ib, 1, **fp8, per_node=8, intra=nv)
    upper = E.decode_on(DS, H200, 512, 4096, 16, "ep", ib, 1, **fp8)
    present(f"{one['comm'] * 1e3:.1f} ms of all-to-alls per step if all 16 GPUs shared one NVLink domain",
            f"{two['comm'] * 1e3:.1f} ms as two 8-GPU nodes", f"which is {two['comm'] / two['time']:.0%} of a {two['time'] * 1e3:.1f} ms step",
            f"the upper bound, gives {upper['comm'] * 1e3:.1f} ms")
    o, t4 = M["olmoe-1b-7b"], D["t4"]
    budget = S.kv_room_gib(o, t4) + S.weight_bytes(o) / 2 ** 30
    off = S.min_offload_gib(o, t4, 4 * 4096)
    pen, step4 = S.offload_step_s(off, 12) * 1e3, T.decode_step(o, t4, 4, 4096, precision="fp16").time * 1e3
    assert S.kv_tokens(o, t4) == 0 and S.kv_tokens(o, t4, offload_gib=off) >= 4 * 4096
    present(f"minus ~1.5 GiB of activations and CUDA graphs (verify), is {budget:.1f} GiB",
            f"OLMoE-1B-7B in fp16 is {S.weight_bytes(o) / 1e9:.1f} GB = {S.weight_bytes(o) / 2 ** 30:.1f} GiB of weights",
            f"four 4K-token sequences is {off:.1f} GiB", f"`--cpu-offload-gb {off:g}`",
            f"adds ~{pen:.0f} ms to every step, against a {step4:.0f} ms step at batch 4 — about {(step4 + pen) / step4:.0f}× slower",
            f"with 4-bit experts OLMoE is {S.weight_bytes(o, 16, 4.25) / 1e9:.1f} GB and leaves room for {S.kv_tokens(o, t4, 16, 4.25):,} tokens")
    present(f"gpt-oss-120b **{S.weight_bytes(M['gpt-oss-120b'], 16, 4.25) / 1e9:.1f} GB**",
            f"gpt-oss-20b **{S.weight_bytes(M['gpt-oss-20b'], 16, 4.25) / 1e9:.1f} GB**")


def test_s7_sizing_and_cost():
    l4 = D["l4"]
    present(f"{S.weight_bytes(Q3) / 1e9:.1f} GB in bf16, {S.weight_bytes(Q3, 8) / 1e9:.1f} GB in FP8, "
            f"{S.weight_bytes(Q3, 16, 4.25) / 1e9:.1f} GB with 4.25-bit experts",
            f"then leaves {S.kv_room_gib(Q3, l4, 16, 4.25):.1f} GiB for KV: {S.kv_tokens(Q3, l4, 16, 4.25):,} tokens, five sequences",
            f"budget would say {S.sessions(Q3, l4, 1, 4096, 16, 4.25)};")
    assert S.kv_tokens(Q3, l4, 16, 4.25) // 4096 == 5
    ratio = S.prefill_flops(M["llama-3.1-70b"].active(), 8192) / S.prefill_flops(MIX.active(), 8192)
    ms = S.prefill_flops(MIX.active(), 8192) / (H100.peak() * 0.5) * 1e3
    present(f"{ratio:.2f}× the FLOPs it costs Mixtral-8x7B", f"Mixtral prefills it in {ms:.0f} ms")
    fp8 = dict(dispatch_elem=1, scale_block=128)
    cases = [(MIX, H100, 64, 50, "nvlink4", 16, 3.0, "ep", {}), (M["llama-3.1-70b"], H100, 64, 50, "nvlink4", 16, 3.0, "tp", {}),
             (DS, H200, 256, 50, "nvlink4", 8, 3.0, "ep", fp8), (Q3, l4, 32, 60, "pcie-l4", 8, 0.7, "ep", dict(exchange="agrs"))]
    for cfg, dev, b, itl, lk, bits, usd, lay, kw in cases:
        p = S.plan(cfg, dev, b, 4096, itl, L[lk], bits, usd_per_gpu_hr=usd, layout=lay, **kw)
        present(f"| {b} | {itl} ms | {p.gpus} | {p.step_ms:.1f} ms | {p.tok_s:,.0f} | {p.usd_per_mtok:.2f} at ${usd:g}/GPU-h |")
    lk = L["pcie-l4"]
    present(f"link of {lk.gbs:g} GB/s and α = {lk.alpha_us:g} µs (`ep.LINKS[\"pcie-l4\"]`")
    ds = S.plan(DS, H200, 256, 4096, 50, L["nvlink4"], 8, **fp8)
    present(f"DeepSeek-V3's step ({ds.step_ms:.1f} ms)",
            f"its all-experts floor ({S.decode_floor(S.weight_bytes(DS, 8), 8, H200) * 1e3:.1f} ms")
    comm = E.decode_on(DS, H200, 256, 4096, 8, "ep", L["nvlink4"], 1, **fp8)["comm"]
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
    dense = S.dense_equivalent(MIX)
    step = lambda cfg, b: T.decode_step(cfg, H200, b, 1024)
    assert step(MIX, 1).time == pytest.approx(step(dense, 1).time, rel=1e-3)
    r4, r16 = step(MIX, 4).time / step(dense, 4).time, step(MIX, 16).time / step(dense, 16).time
    present(f"Mixtral on an H200 at batch 1: {step(MIX, 1).time * 1e3:.2f} ms, the same as a dense model of its "
            f"{dense.total() / 1e9:.1f}B active size",
            f"but {S.weight_bytes(MIX) / 1e9:.0f} GB of weights resident against {S.weight_bytes(dense) / 1e9:.0f} GB",
            f"at batch 4 and 16 it streams {step(MIX, 4).bytes / 1e9:.1f} and {step(MIX, 16).bytes / 1e9:.1f} GB against "
            f"{step(dense, 4).bytes / 1e9:.1f} and {step(dense, 16).bytes / 1e9:.1f} GB, {r4:.1f}× and {r16:.1f}× slower",
            f"on {MIX.total() / dense.total():.1f}× the memory")
    # the other rows, and drills 1 and 5, repeat numbers computed in §6-§7: keep the copies in step
    nv, ib, fp8 = L["nvlink4"], L["ib-ndr"], dict(dispatch_elem=1, scale_block=128)
    one = E.decode_on(DS, H200, 512, 4096, 16, "ep", nv, 1, **fp8)
    two = E.decode_on(DS, H200, 512, 4096, 16, "ep", ib, 1, **fp8, per_node=8, intra=nv)
    present(f"all-to-alls {two['comm'] * 1e3:.1f} ms per step ({two['comm'] / two['time']:.0%} of the step) against "
            f"{one['comm'] * 1e3:.1f} ms in one NVLink domain",
            f"batch 512: {two['comm'] * 1e3:.1f} ms of all-to-alls per step across two nodes on 400 Gb/s vs {one['comm'] * 1e3:.1f} ms")
    pop = T.zipf_popularity(128, 1.0)[np.random.default_rng(0).permutation(128)]
    lt = {tpg: E.layer_time(T.sample_routes(128, 8, 8 * tpg, pop, np.random.default_rng(1)), np.repeat(np.arange(8), tpg),
                            E.placement(128, 8), 8, Q3.expert_params(), 2048, H100, nv) for tpg in (128, 4096)}
    present(f"{lt[4096].imbalance:.2f}× rows on the busiest rank make a prefill layer {lt[4096].penalty:.2f}× slower; in decode "
            f"the weight reads stay balanced and the hot rank's port makes the layer {lt[128].penalty:.2f}× slower")
    n4k = S.kv_tokens(Q3, D["l4"], 16, 4.25) // 4096
    present(f"18.5 GB with 4-bit experts ({n4k} sequences of 4K at vLLM's defaults)",
            f"leaving room for about {n4k} sequences of 4K tokens at vLLM's defaults ({S.kv_tokens(Q3, D['l4'], 16, 4.25):,} tokens")
    o, t4 = M["olmoe-1b-7b"], D["t4"]
    off = S.min_offload_gib(o, t4, 4 * 4096)
    step4 = T.decode_step(o, t4, 4, 4096, precision="fp16").time
    present(f"(OLMoE fp16 on a T4: {off:g} GiB offloaded, ~{(step4 + S.offload_step_s(off, 12)) / step4:.0f}× slower at batch 4)")
