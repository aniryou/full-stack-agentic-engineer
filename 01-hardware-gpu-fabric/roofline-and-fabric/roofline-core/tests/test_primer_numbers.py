"""Every computed number the topic's PRIMER.md quotes is recomputed here and must appear verbatim.

If a formula or a catalogue entry changes, the primer fails this test until it is updated.
(Inputs and datasheet constants -- 80 GB, 3.35 TB/s, 16,384 GPUs -- are quoted, not computed.)
"""
import math
from pathlib import Path

import pytest

from roofline import cost, fabric, llm, reliability, specs, storage
from roofline import roofline as rl

def _norm(text: str) -> str:
    return " ".join(text.split())          # line wrapping and alignment spaces do not matter


PRIMER = _norm((Path(__file__).resolve().parents[2] / "PRIMER.md").read_text(encoding="utf-8"))
D, L, P = specs.DEVICES, fabric.LINKS, llm.PRESETS
H100, L4, H200 = D["h100-sxm"], D["l4"], D["h200"]
M8, M70, MIX, Q3 = P["llama-3.1-8b"], P["llama-3.1-70b"], P["mixtral-8x7b"], P["qwen3-30b-a3b"]
GIB = 1 << 30


def present(*fragments):
    missing = [f for f in fragments if _norm(f) not in PRIMER]
    assert not missing, f"PRIMER.md no longer says: {missing}"


def test_s1_spec_sheet_numbers():
    present(*(f"{specs.peak_from_clock(*specs.CLOCKS[k]):.1f} TFLOP/s" for k in specs.CLOCKS))
    present(f"{H100.tflops['bf16'] / H100.tflops['fp32']:.1f}× of the chip",
            f"80 GiB = {80 * 1024 ** 3 / 1e9:.1f} × 10⁹ bytes",
            f'"900 GB/s" is {H100.scaleup_gbs_per_dir:.0f} each way',
            f'"128 GB/s" is {L["pcie5x16"].gbs:.0f} each way',
            f"400 Gb/s NIC moves {L['ib-ndr'].gbs:.0f} GB/s",
            f"three hundred times the H100's L2")
    assert round(llm.streamed_weight_bytes(M8, 1) / (H100.l2_mb * 1e6)) == 300


def test_s2_roofline_numbers():
    present(f"**{H100.ridge():.0f} FLOP/B** (fp8: {H100.ridge('fp8'):.0f})",
            f"= {L4.ridge():.0f} (fp8: {L4.ridge('fp8'):.0f})", f"T4 fp16 65 / 0.32 = {D['t4'].ridge('fp16'):.0f}",
            f"A100 80GB {D['a100-80gb'].ridge():.0f}", f"H200 {H200.ridge():.0f}", f"B200 {D['b200'].ridge():.0f}",
            f"MI300X {D['mi300x'].ridge():.0f}")
    n = 1 << 26
    add, red = rl.elementwise(n, 2), rl.reduction(n, 4)
    gemv, g64, g4k = rl.gemm(1, 4096, 4096), rl.gemm(64, 4096, 4096), rl.gemm(4096, 4096, 4096)
    pct = lambda k: rl.attainable(k.intensity, H100) / H100.peak()
    present(f"{rl.attainable(add.intensity, H100) / 1e12:.2f} TFLOP/s, {pct(add):.2%} of peak",
            f"{rl.attainable(red.intensity, H100) / 1e12:.2f} TFLOP/s",
            f"{rl.attainable(gemv.intensity, H100) / 1e12:.2f} TFLOP/s, {pct(gemv):.2%} of peak",
            f"| {g64.intensity:.0f} | {rl.attainable(g64.intensity, H100) / 1e12:.0f} TFLOP/s, memory-bound",
            f"| {g4k.intensity:,.0f} | {rl.attainable(g4k.intensity, H100) / 1e12:.0f} TFLOP/s, compute-bound")
    n_min = math.ceil(1.5 * 2 * H100.ridge())
    assert rl.gemm(n_min, n_min, n_min).intensity >= H100.ridge() > rl.gemm(n_min - 1, n_min - 1, n_min - 1).intensity
    present(f"n ≥ {n_min}", f"{rl.time_kernel(rl.gemm(8192, 8192, 8192), H100).time * 1e3:.2f} ms on an H100")
    big, small = rl.gemm(8192, 8192, 8192), rl.gemm(128, 128, 128)
    t = rl.time_kernel(small, H100)
    present(f"does {big.flops / 1e12:.1f} × 10¹² FLOPs",
            f"a 128³ GEMM is {small.flops / 1e6:.0f} MFLOP and {small.bytes / 1e3:.0f} KB: {t.t_compute * 1e9:.0f} ns\n"
            f"of math at peak, {t.time * 1e9:.0f} ns on the roofline")
    assert t.bound == "memory"                                             # 43 FLOP/B < 295


def test_s3_llm_numbers():
    present(f"{M8.params() / 1e9:.2f} B parameters", f"{M8.matmul_params_per_token() / 1e9:.2f} B per-token",
            f"{M8.lm_head_params() / 1e6:.0f} M-parameter LM head", f"**{llm.streamed_weight_bytes(M8, 1) / 1e9:.1f} GB**",
            f"{M8.kv_bytes_per_token():,.0f} B (128 KiB)")
    pf = llm.prefill(M8, H100, 2048)
    present(f"FLOPs = {pf.flops / 1e13:.2f} × 10¹³", f"bytes = {pf.bytes / 1e9:.1f} GB",
            f"intensity = {pf.intensity:,.0f} FLOP/B", f"compute time {pf.t_compute * 1e3:.1f} ms",
            f"memory time {pf.t_memory * 1e3:.2f} ms", f"(L4: {llm.prefill(M8, L4, 2048).t_compute * 1e3:.0f} ms)",
            f"8K tokens: {llm.prefill(M8, H100, 8192).time * 1e3:.0f} ms, "
            f"{llm.prefill(M8, H100, 8192).time / pf.time:.1f}× the 2K figure")
    assert llm.prefill(M8, H100, 300).bound == "memory" and llm.prefill(M8, H100, 320).bound == "compute"
    d1, d1l4 = llm.decode(M8, H100, 1, 1024), llm.decode(M8, L4, 1, 1024)
    present(f"**{d1.time * 1e3:.2f} ms on an H100 ({d1.tokens_per_s:.0f} tokens/s), "
            f"{d1l4.time * 1e3:.1f} ms on an L4 ({d1l4.tokens_per_s:.1f} tokens/s)")
    present(f"re-reads 15 GB for ~{d1.flops / 1e9:.0f} GFLOPs")
    cap = llm.max_batch_by_memory(M8, H100, 2048)
    for b in (1, 8, 32, 64, 128, cap):
        s = llm.decode(M8, H100, b, 2048)
        kv = b * 2049 * M8.kv_bytes_per_token() / s.bytes
        label = f"{b} (HBM limit)" if b == cap else f"{b}"
        present(f"| {label} | {s.intensity:.1f} | {s.time * 1e3:.2f} ms | {s.tokens_per_s:,.0f} | {1 / s.time:.0f} | {kv:.0%} |")
    assert llm.max_batch_by_memory(M8, H100, 2048) < 256                   # no infeasible rows in the table
    present(f"stops where HBM does: {cap} sequences of 2K tokens fit",
            f"the largest batch is {llm.best_batch_under_itl(M8, H100, 4096, 0.020)}",
            f"HBM capacity allows {llm.max_batch_by_memory(M8, H100, 4096)}")


def test_s3_kv_quantization_moe_numbers():
    lim = lambda c: f"{llm.decode_intensity_limit(M8, c):.1f}"
    present(f"1K context → {lim(1024)} FLOP/B    4K → {lim(4096)}    32K → {lim(32768)}",
            f"{llm.decode_crossover_batch(M8, H100, 0)} at c = 0, {llm.decode_crossover_batch(M8, H100, 128)} at c = 128, "
            f"{llm.decode_crossover_batch(M8, H100, 256)} at c = 256",
            f"~{llm.max_context_for_compute_bound(M8, H100):.0f} tokens",
            f"({llm.max_batch_by_memory(M8, H100, 4096)} sequences of 4K")
    fp8 = llm.SCHEMES["fp8"]
    g_bf16, g_fp8 = llm.gemm_crossover_batch(M8, H100), llm.gemm_crossover_batch(M8, H100, **fp8)
    assert g_bf16 == g_fp8
    attn = llm.decode_split(M8, H100, 64, 4096).attention
    present(f"compute-bound from batch {g_bf16} (`gemm_crossover_batch()`; FP8 halves the\nbytes and doubles the peak, so also {g_fp8})",
            f"(heads / kv_heads) / kv_bytes = {attn.intensity:.0f} FLOP/B")
    assert attn.bound == "memory" and llm.decode_split(M8, H100, 1 << 16, 4096).attention.bound == "memory"
    fp8_mem = {k: fp8[k] for k in ("weight_bytes", "kv_bytes")}
    sp, one = llm.decode_split(M8, H100, 400, 2048, **fp8), llm.decode(M8, H100, 400, 2048, **fp8)
    assert sp.gemms.bound == "compute" and sp.attention.bound == "memory" and one.bound == "memory"
    present(f"batch 400 ({llm.max_batch_by_memory(M8, H100, 2048, **fp8_mem)} fit)",
            f"one kernel    {one.time * 1e3:.2f} ms, memory-bound",
            f"GEMMs {sp.gemms.time * 1e3:.2f} ms (compute-bound) + attention {sp.attention.time * 1e3:.2f} ms "
            f"(memory-bound) = {sp.time * 1e3:.2f} ms",
            f"{400 / sp.time:,.0f} tokens/s here at any larger batch")
    assert 600 / llm.decode_split(M8, H100, 600, 2048, **fp8).time == pytest.approx(400 / sp.time)
    # "the two agree ... every decode row in 3.3-3.6 and 8": all of those rows have memory-bound GEMMs
    rows = [(M8, H100, b, 2048, {}) for b in (1, 8, 32, 64, 128, 208)] + [(M8, H100, 96, 4096, {})]
    rows += [(M8, H100, b, 2048, llm.SCHEMES[k]) for b in (1, 64) for k in llm.SCHEMES]
    rows += [(MIX, H200, b, 1024, {}) for b in (1, 4, 16, 64)] + [(M8, H100, 68, 2048, {}), (M8, H100, 193, 2048, fp8)]
    rows += [(M8, L4, 20, 2048, {})]
    for m, dev, b, c, kw in rows:
        assert llm.decode_split(m, dev, b, c, **kw).time == pytest.approx(llm.decode(m, dev, b, c, **kw).time), (m.name, b, c, kw)
    b1, b64 = llm.decode(M8, H100, 1, 2048), llm.decode(M8, H100, 64, 2048)
    for name in llm.SCHEMES:
        kw = llm.SCHEMES[name]
        s1, s64 = llm.decode(M8, H100, 1, 2048, **kw), llm.decode(M8, H100, 64, 2048, **kw)
        present(f"| {s1.time * 1e3:.2f} ms | {b1.time / s1.time:.2f}× | {s64.time * 1e3:.2f} ms | "
                f"{b64.time / s64.time:.2f}× | {'batch ' if name == 'bf16' else ''}"
                f"{llm.decode_crossover_batch(M8, H100, 0, **kw)} |")
    fp8_mem = {k: llm.SCHEMES["fp8"][k] for k in ("weight_bytes", "kv_bytes")}
    present(f"{llm.max_batch_by_memory(M8, H100, 2048, **fp8_mem)} sequences of 2K tokens fit on the H100 "
            f"instead of {llm.max_batch_by_memory(M8, H100, 2048)}")
    for b in (1, 4, 16, 64):
        s = llm.decode(MIX, H200, b, 1024)
        e = llm.experts_touched(8, 2, b)
        mix = f"{e:.2f} experts/layer" if b == 1 else f"{e:.2f}"
        q = f"{llm.experts_touched(128, 8, b):.1f}" + (" / 128" if b == 1 else "")
        present(f"| {b} | {mix} | {s.bytes / 1e9:.1f} GB | {s.time * 1e3:.2f} ms | {q} |")
    present(f"Mixtral ({MIX.params() / 1e9:.1f} B total, {MIX.active_params() / 1e9:.1f} B active)",
            f"nearly all {MIX.params() * 2 / 1e9:.0f} GB")
    x = {m.name: llm.decode_crossover_batch(m, H200, 0) for m in (M8, MIX, Q3)}
    ratio = lambda m: m.params() / m.active_params()
    # streamed weights / multiplied weights: total / active without the input embedding (a gather, in neither)
    ratio_mm = lambda m: (m.params() - m.vocab * m.d_model) / (m.active_params() - m.vocab * m.d_model)
    present(f"c = 0 on an H200 (ridge {H200.ridge():.0f}; `decode_crossover_batch()`): {x[M8.name]} for Llama-3.1-8B, "
            f"{x[MIX.name]} for Mixtral-8x7B\n(total/active {ratio(MIX):.1f}) and {x[Q3.name]:,} for Qwen3-30B-A3B "
            f"({ratio(Q3):.1f}, or {ratio_mm(Q3):.1f} without the input embedding")
    for m in (MIX, Q3):                                                    # crossover ~ ridge x that ratio
        assert x[m.name] / x[M8.name] == pytest.approx(ratio_mm(m), rel=0.01)


def test_s4_hierarchy_numbers():
    present(*(f"{t}×{t2} → {rl.tile_intensity(t, t2):.0f}" for t, t2 in ((16, 16), (64, 64), (128, 128), (128, 256), (256, 256))))
    smem = rl.tile_smem_bytes(128, 256, 64, 2, 4)
    present(f"256 KB per SM ({256 * 132 / 1024:.0f} MB total)")                  # 132 SMs x 256 KiB
    present(f"= {smem // 1024} KiB of shared memory", f"= {128 * 256 * 4 // 1024} KiB of registers",
            f"{smem / (228 * 1024):.0%} of an H100 SM's shared memory")      # 228 KB here means KiB
    n = 4096
    tiled, comp = rl.tiled_gemm_bytes(n, n, n, 128, 128), rl.gemm(n, n, n)
    present(f"move {tiled / 1e9:.2f} GB — {tiled / comp.bytes:.1f}× the compulsory {comp.bytes / 1e6:.0f} MB — "
            f"at {comp.flops / tiled:.0f} FLOP/B", f"sit at {comp.flops / rl.tiled_gemm_bytes(n, n, n, 1, 1):.1f} FLOP/B")
    ne = 4096 * 8192
    uf, f = rl.fusion_bytes(ne, 4, 2, False, extra_inputs=1), rl.fusion_bytes(ne, 4, 2, True, extra_inputs=1)
    assert (uf, f) == (9 * ne * 2, 3 * ne * 2)                            # 4 ops x (in + out) + residual; x, residual, out
    present(f"unfused {uf / 1e6:.0f} MB → {uf / H100.bandwidth() * 1e6:.0f} µs",
            f"fused {f / 1e6:.0f} MB → {f / H100.bandwidth() * 1e6:.0f} µs", f"removes {1 - f / uf:.0%} of the bytes")


def test_s5_fabric_numbers():
    for key in ("nvlink5", "nvlink4", "nvlink3", "pcie5x16", "pcie4x16", "ib-xdr", "eth-100g-tcp"):
        lk = L[key]
        present(f"| {lk.gbs:g} | {lk.alpha_us:g} µs |")
    t = fabric.ring_allreduce_time(GIB, 8, L["nvlink4"])
    present(f"**{fabric.allreduce_crossover_bytes(8, L['nvlink4']) / 1e6:.1f} MB**",
            f"(over a 400 Gb/s NIC: {fabric.allreduce_crossover_bytes(8, L['ib-ndr']) / 1e6:.1f} MB)",
            f"takes {t * 1e3:.2f} ms — algbw {fabric.algbw(GIB, t) / 1e9:.0f} GB/s",
            f"= {fabric.busbw(GIB, t, 8) / 1e9:.0f} GB/s**")
    dec, pf = llm.decode(M70, H100, 1, 1024), llm.prefill(M70, H100, 4096)
    ratio = fabric.allreduce_crossover_bytes(8, L["nvlink4"]) / fabric.tp_allreduce_bytes(M70, 1)
    present(f"**{fabric.tp_allreduces_per_step(M70)}\nall-reduces per step**",
            f"= {fabric.tp_allreduce_bytes(M70, 1) // 1024} KiB", f"({round(ratio, -1):.0f}× below",
            f"shard in {dec.t_memory / 8 * 1e3:.2f} ms",
            f"= {fabric.tp_comm_time(M70, 1, 8, L['nvlink4']) * 1e3:.2f} ms",
            f"(latency-optimal algorithm: {fabric.tp_comm_time(M70, 1, 8, L['nvlink4'], algo='recursive-doubling') * 1e3:.2f} ms)",
            f"= {fabric.tp_allreduce_bytes(M70, 4096) / 1e6:.0f} MB", f"= {pf.t_compute / 8 * 1e3:.1f} ms per GPU",
            f"= {fabric.tp_comm_time(M70, 4096, 8, L['nvlink4']) * 1e3:.1f} ms over NVLink 4",
            f"({fabric.tp_comm_time(M70, 4096, 8, L['ib-ndr']) * 1e3:.1f} ms if every ring hop crossed a 400 Gb/s NIC")
    nv, ib = L["nvlink4"], L["ib-ndr"]
    rails = fabric.tp_comm_time_across_nodes(M70, 4096, 8, 2, nv, ib)
    one = fabric.tp_comm_time_across_nodes(M70, 4096, 8, 2, nv, ib, nics_per_node=1)
    flat = fabric.tp_comm_time(M70, 4096, 16, ib)
    c8, c16, t8 = pf.t_compute / 8, pf.t_compute / 16, fabric.tp_comm_time(M70, 4096, 8, nv)
    present(f"compute   = {c16 * 1e3:.1f} ms per GPU", f"comm/step = {rails * 1e3:.1f} ms with rails",
            f"{one * 1e3:.1f} ms with one NIC per node", f"{flat * 1e3:.1f} ms as one flat ring through the NICs",
            f"TP=8 already spends {t8 * 1e3:.0f} ms communicating per {c8 * 1e3:.0f} ms of compute",
            f"spends ~{rails * 1e3:.0f} ms per 4K-token step in\nall-reduce against ~{c16 * 1e3:.0f} ms of compute per GPU",
            f"{(c16 + rails) * 1e3:.0f} ms per step against TP=8's {(c8 + t8) * 1e3:.0f} ms (no overlap), for "
            f"{16 * (c16 + rails) / (8 * (c8 + t8)):.1f}× the\nGPU-seconds",
            f"({one * 1e3:.0f} ms with one NIC per node, {flat * 1e3:.0f} ms as a flat ring)")
    assert rails > c16 and rails < one < flat                              # still 2x compute even on rails
    lat = lambda p: dec.t_memory / p + fabric.tp_comm_time(M70, 1, p, L["nvlink4"])
    present(f"{1 / (lat(2) * 2):.1f} at TP=2, {1 / (lat(8) * 8):.1f} at TP=8")
    rails = fabric.hierarchical_allreduce_time(GIB, 8, 4, L["nvlink4"], L["ib-ndr"])
    one = fabric.hierarchical_allreduce_time(GIB, 8, 4, L["nvlink4"], L["ib-ndr"], nics_per_node=1)
    present(f"**{rails * 1e3:.1f} ms with 8 NICs per node and {one * 1e3:.1f} ms with one**")
    a, b = fabric.leaf_spine(2048, 64), fabric.leaf_spine(3072, 64, 48)
    present(f"2,048\nendpoints need {a.leaves} leaves and {a.spines} spines",
            f"with {b.leaves}\nleaves and {b.spines} spines — at **{b.oversubscription:.0f}:1 oversubscription**",
            f"from {fabric.bisection_gbs(3072, 400):,.0f} GB/s to {fabric.bisection_gbs(3072, 400, 3):,.0f} GB/s")
    present(f"store-and-forward    {fabric.staged_transfer_time(GIB, [63, 50, 63]) * 1e3:.1f} ms",
            f"pipelined (1 MiB)  {fabric.staged_transfer_time(GIB, [63, 50, 63], 1 << 20) * 1e3:.1f} ms")


def test_s6_storage_numbers():
    b8, b70 = storage.checkpoint_bytes(M8.params()), storage.checkpoint_bytes(M70.params())
    present(f"Llama-3.1-8B is\n{b8 / 1e9:.1f} GB in bf16, Llama-3.1-70B {b70 / 1e9:.1f} GB")
    one = storage.read_time(b70, 0.1)
    present(f"| {one:,.0f} s = {one / 60:.1f} min |", f"(1.2) | {storage.read_time(b70, 1.2):.0f} s |",
            f"(7) | {storage.read_time(b70, 7):.1f} s |",
            f"(12.5) | {storage.read_time(b70, storage.parallel_gbs(128, 0.1, 12.5)):.1f} s |",
            f"(20) | {storage.read_time(b70, 20):.1f} s |", f"(8 × 50) | {storage.read_time(b70, 400):.2f} s |")
    fetch = storage.parallel_gbs(128, 0.1, 12.5)
    rows = [storage.cold_start(b70, fetch_gbs=0.1, h2d_gbs=50, gpus=8, provision_s=120, image_s=60, init_s=90),
            storage.cold_start(b70, fetch_gbs=fetch, h2d_gbs=50, gpus=8, provision_s=120, image_s=60, init_s=90, streamed=True),
            storage.cold_start(b70, fetch_gbs=20, h2d_gbs=50, gpus=8, init_s=90, streamed=True)]
    labels = ["new node, one stream", "new node, parallel + streamed weights", "warm node, checkpoint in page cache"]
    for label, cs in zip(labels, rows):
        name, s = max(cs.stages, key=lambda st: st[1])
        present(f"| {label} | {cs.total:,.0f} s | {name}, {s / cs.total:.0%} |")
    need = b70 / 40 / 1e9
    present(f"= **{need:.2f} GB/s** of fetch, i.e. {math.ceil(need / 0.1)} parallel streams")


def test_s7_reliability_numbers():
    m = reliability.component_mtbf_from_observation(419, 54 * 24, 16384)
    present(f"= {m:,.0f} GPU-hours ≈ {m / 8760:.1f} years",
            f"every {reliability.cluster_mtbf(m, 8):,.0f} h       P(clean 24 h) = {reliability.p_survive(24, 8, m):.3f}",
            f"every {reliability.cluster_mtbf(m, 1024):.1f} h", f"P(clean 24 h) = {reliability.p_survive(24, 1024, m):.3f}",
            f"every {reliability.cluster_mtbf(m, 16384):.2f} h ({reliability.failures_per_day(16384, m):.2f} per day)")
    ms = reliability.cluster_mtbf(m, 16384) * 3600
    t60, t10 = reliability.young_daly_interval(60, ms), reliability.young_daly_interval(10, ms)
    w = reliability.wasted_fraction
    present(f"M = {ms / 3600:.2f} h = {ms:,.0f} s",
            f"τ* = {t60 / 60:.1f} min (Daly {reliability.young_daly_interval(60, ms, True) / 60:.1f}), waste {w(t60, 60, ms):.1%}",
            f"(checkpointing hourly: {w(3600, 60, ms):.1%})", f"τ* =  {t10 / 60:.1f} min,             waste  {w(t10, 10, ms):.1%}",
            f"(hourly: {w(3600, 10, ms):.1%})")
    m4k = reliability.cluster_mtbf(m, 4096) * 3600
    t30 = reliability.young_daly_interval(30, m4k)
    present(f"τ* = {t30 / 60:.1f} min, waste {w(t30, 30, m4k):.1%}")
    r = 600                                                                # assumed 10-minute restart
    present("restart cost R = 0 (ignored)", f"R = 10 min (reschedule, reload, re-initialise): R/M = {r / ms:.1%} more",
            f"δ = 60 s → waste {w(t60, 60, ms, r):.1%}        δ = 10 s → waste {w(t10, 10, ms, r):.1%}",
            f"a 10-minute restart ({r / ms:.1%}) costs more than everything a 10 s checkpoint\nwastes ({w(t10, 10, ms):.1%})")
    assert w(t60 * 1.2, 60, ms, r) > w(t60, 60, ms, r) < w(t60 / 1.2, 60, ms, r)   # R does not move tau*
    a8 = reliability.replica_availability(m, 8, 48)
    present(f"= M/8 = {reliability.cluster_mtbf(m, 8):,.0f} h", f"= {a8:.5f}",
            f"P(≥ 8 up) = {reliability.p_at_least(8, 8, a8):.3f}", f"deploy  9 → {reliability.p_at_least(8, 9, a8):.3f}",
            f"deploy 10 → {reliability.p_at_least(8, 10, a8):.5f}",
            f"needs {reliability.replicas_for(8, a8, 0.999)} replicas = 80 GPUs",
            f"→ {reliability.replicas_for(16, reliability.replica_availability(m, 4, 48), 0.999)} replicas = 72 GPUs",
            f"→ {reliability.replicas_for(64, reliability.replica_availability(m, 1, 48), 0.999)} replicas = 66 GPUs")


def test_s8_cost_numbers():
    slo, fp8 = 0.010, llm.SCHEMES["fp8"]
    fp8_mem = {k: fp8[k] for k in ("weight_bytes", "kv_bytes")}
    b16, b8 = llm.best_batch_under_itl(M8, H100, 2048, slo), llm.best_batch_under_itl(M8, H100, 2048, slo, **fp8)
    assert b16 < llm.max_batch_by_memory(M8, H100, 2048) and b8 < llm.max_batch_by_memory(M8, H100, 2048, **fp8_mem)
    assert llm.best_batch_under_itl(M8, L4, 2048, slo) == 0                 # the L4 misses the SLO at any batch
    b_l4 = llm.max_batch_by_memory(M8, L4, 2048)
    present(f"ITL of {slo * 1e3:.0f} ms** ({1 / slo:.0f} tokens/s per user)",
            f"at batch 1 it needs {llm.decode(M8, L4, 1, 2048).time * 1e3:.1f} ms")
    rows = [("H100 on demand, bf16", 11.0, llm.decode(M8, H100, b16, 2048), f"{b16} (ITL-bound)"),
            ("H100 Spot, bf16", 3.7, llm.decode(M8, H100, b16, 2048), f"{b16} (ITL-bound)"),
            ("H100 on demand, FP8", 11.0, llm.decode(M8, H100, b8, 2048, **fp8), f"{b8} (ITL-bound)"),
            ("L4 on demand, bf16 — misses the SLO", 0.70, llm.decode(M8, L4, b_l4, 2048), f"{b_l4} (HBM-bound)")]
    costs, tps = {}, {}
    for name, price, s, batch in rows:
        c1 = cost.cost_per_million_tokens(price, s.tokens_per_s)
        c6 = cost.cost_per_million_tokens(price, s.tokens_per_s, 0.6)
        costs[name], tps[name] = c1, s.tokens_per_s
        per_user = f"{1 / s.time:.1f}" if 1 / s.time < 100 else f"{1 / s.time:.0f}"
        c6s = f"${c6:.3f}" if c6 < 1 else f"${c6:.2f}"
        present(f"| {name} | {batch} | {s.time * 1e3:.2f} ms | {s.tokens_per_s:,.0f} | {per_user} | ${c1:.3f} | {c6s} |"
                if s.time < 0.02 else
                f"| {name} | {batch} | {s.time * 1e3:.1f} ms | {s.tokens_per_s:,.0f} | {per_user} | ${c1:.3f} | {c6s} |")
    h16, h8, l4 = "H100 on demand, bf16", "H100 on demand, FP8", "L4 on demand, bf16 — misses the SLO"
    same_batch = cost.cost_per_million_tokens(11.0, llm.decode(M8, H100, b16, 2048, **fp8).tokens_per_s)
    present(f"deliver {tps[h16] / tps[l4]:.0f}× the L4's tokens/s for {11.0 / 0.70:.1f}× the price",
            f"another\n{costs[h16] / costs[h8]:.1f}×: exactly {costs[h16] / same_batch:.1f}× from halving every byte, "
            f"and {same_batch / costs[h8]:.1f}× more from the bigger batch ({b8} vs {b16})",
            f"~{round(llm.prefill(M8, H100, 2048).tokens_per_s, -3):,.0f} tokens/s")
    u = cost.utilisation([0.2] * 8 + [1.0] * 8 + [0.6] * 8)
    present(f"averages {u:.0%}", f"{1 / u:.2f}× higher")
    fixed, energy = cost.owned_cost_per_hour(300_000, 4, 10.2, pue=1.3, usd_per_kwh=0.10, opex_per_year=30_000)
    be = lambda r: cost.breakeven_utilisation(r, fixed / 8, energy / 8)
    present(f"${fixed / 8:.2f} per GPU-hour fixed + ${energy / 8:.3f} per busy GPU-hour",
            f"$11/GPU-hr: {be(11):.1%}", f"Spot at $3.7: {be(3.7):.1%}", f"$2.0 rental: {be(2.0):.1%}")


def test_s9_landscape_table_is_the_catalogue():
    keys = ["t4", "l4", "a100-80gb", "h100-sxm", "h200", "b200", "gb200", "gb300", "rtx-pro-6000",
            "mi300x", "mi325x", "mi355x", "tpu-v5e", "tpu-v6e", "tpu-v7"]
    present(specs.table(keys))
    present(f"{D['b200'].memory_tbs / L4.memory_tbs:.0f}× range")


def test_design_review_numbers():
    m = reliability.component_mtbf_from_observation(419, 54 * 24, 16384)
    ms = reliability.cluster_mtbf(m, 16384) * 3600
    present(f"({D['b200'].memory_tbs:g} vs {H100.memory_tbs} TB/s,\n   {D['b200'].memory_tbs / H100.memory_tbs:.1f}×)",
            f"not its {D['b200'].tflops['bf16'] / H100.tflops['bf16']:.1f}× bf16 peak",
            f"every √(2 × 60 × {ms:,.0f}) s ≈ {reliability.young_daly_interval(60, ms) / 60:.0f} min",
            f"a NIC {L['nvlink4'].gbs / L['ib-ndr'].gbs:.0f}× slower than NVLink")
    present(f"{llm.decode(M8, H100, 1, 1024).time * 1e3:.1f} ms at batch 1",
            f"the GEMMs reach the ridge near batch {round(llm.gemm_crossover_batch(M8, H100), -2)}",
            f"averages at most {llm.decode_intensity_limit(M8, 4096):.0f} FLOP/B")
    b70 = storage.checkpoint_bytes(M70.params())
    present(f"(NIC-bound at ~{storage.read_time(b70, storage.parallel_gbs(128, 0.1, 12.5)):.0f} s)")
    t10, r = reliability.young_daly_interval(10, ms), 600
    w = reliability.wasted_fraction
    present(f"losing ~{w(reliability.young_daly_interval(60, ms), 60, ms):.0%} plus R/M for restarts "
            f"(~{w(reliability.young_daly_interval(60, ms), 60, ms, r):.0%} with a 10-minute restart)",
            f"interval to ~{t10 / 60:.0f} min and the checkpoint waste to ~{w(t10, 10, ms):.0%}")
    pf, nv, ib = llm.prefill(M70, H100, 4096), L["nvlink4"], L["ib-ndr"]
    present(f"4K-token prefill step spends ~{fabric.tp_comm_time_across_nodes(M70, 4096, 8, 2, nv, ib) * 1e3:.0f} ms "
            f"in all-reduce against ~{pf.t_compute / 16 * 1e3:.0f} ms of compute per GPU",
            f"with one NIC per node it is {fabric.tp_comm_time_across_nodes(M70, 4096, 8, 2, nv, ib, 1) * 1e3:.0f} ms")


def test_rounded_restatements_match_their_sources():
    """Numbers the primer repeats in prose (one-minute version, lessons, drills) are pinned too."""
    ridge = H100.ridge()
    n = 1 << 26
    add, red, gemv = rl.elementwise(n, 2), rl.reduction(n, 4), rl.gemm(1, 4096, 4096)
    present(f"about {ridge:.0f} FLOP per byte for an H100", f"the ridge: {ridge:.0f} FLOP per\nbyte on an H100",
            f"β per GPU falls ~{L['nvlink4'].gbs / L['ib-ndr'].gbs:.0f}× from NVLink",
            f"(A100 {D['a100-80gb'].ridge():.0f} → H100 {ridge:.0f})", f"with more bandwidth: {H200.ridge():.0f})",
            f"| 1/6 = {add.intensity:.2f} |", f"| 4n | {red.intensity:.2f} |", f"| ≈ dk × 2 | {gemv.intensity:.1f} |",
            f"from {D['a100-80gb'].ridge():.0f} (A100 80GB) to {D['tpu-v6e'].ridge():.0f} (TPU v6e)",
            f"~{llm.prefill(M8, H100, 2048).t_compute * 1e3:.0f} ms ideal for an 8B model")
    assert add.intensity < 0.2 < red.intensity                             # "~0.2 FLOP/B"
    assert llm.prefill(M8, H100, 300).bound == "memory" and llm.prefill(M8, H100, 320).bound == "compute"
    present("the bound flips between 300 and 320")
    b64 = llm.decode(M8, H100, 64, 2048)
    sp = lambda k: b64.time / llm.decode(M8, H100, 64, 2048, **llm.SCHEMES[k]).time
    present(f"only {sp('w4a16'):.1f}× at batch 64", f"crossover to ~{llm.decode_crossover_batch(M8, H100, 0, **llm.SCHEMES['w4a16'])}",
            f"but {sp('w8a16'):.1f}× at batch 64 × 2K context")
    b70 = storage.checkpoint_bytes(M70.params())
    present(f"leaves {120 - 20 - 60} s for streamed weights: {b70 / 1e9:.1f} GB / {120 - 20 - 60} s",
            f"is {storage.read_time(b70, 0.1) / 60:.1f} of those minutes")
    m = reliability.component_mtbf_from_observation(419, 54 * 24, 16384)
    a4 = reliability.replica_availability(m, 4, 48)
    present(f"interrupted every ~{reliability.cluster_mtbf(m, 16384):.1f} h",
            f"{reliability.replicas_for(16, a4, 0.999)} replicas ({4 * reliability.replicas_for(16, a4, 0.999)} GPUs)",
            f"TP makes {fabric.tp_allreduces_per_step(M70)} all-reduces per step for a 70B model")
