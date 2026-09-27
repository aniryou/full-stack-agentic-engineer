"""Numbers this lab re-implements must match their home in the repo (SPEC §6b/§6c): minengine.spec (speculative
decoding), roofline.cost and roofline.llm (cost of a token, decode step), capacity.py (the capacity primer),
quantcore.eval (agreement), the Wilson interval of memory-core and the 07 agent lab, servelab.sizing (parameter
counts) and the 06 scaling lab's price model. Each source is loaded by path from the repo (labs never import each
other); a test skips when its source is not in the checkout."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from distillab import agreement as A
from distillab import cost as C
from distillab import draft as DR
from distillab import teacher as TE
from distillab.hf import memory as H

REPO = Path(__file__).resolve().parents[4]
PRIMER_P = [0.5, 0.3, 0.15, 0.05]     # serving-engine PRIMER §7's example target distribution
PRIMER_Q = [0.2, 0.2, 0.2, 0.4]       # ... and its draft: alpha = 0.6


@pytest.fixture(autouse=True)
def _no_bytecode_outside_this_lab(monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)      # never write __pycache__ into other labs


def _load(path: Path, name: str):
    if not path.exists():
        pytest.skip(f"{path} not in this checkout")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _package(root: Path, pkg: str):
    """Import a sibling package (roofline, minengine, quantcore, ...) from its lab directory, then forget it."""
    if not (root / pkg.split(".")[0]).is_dir():
        pytest.skip(f"{root / pkg.split('.')[0]} not in this checkout")
    sys.path.insert(0, str(root))
    try:
        return __import__(pkg, fromlist=["*"])
    finally:
        sys.path.remove(str(root))


def test_speculative_decoding_matches_minengine_spec_on_the_serving_primers_example():
    """serving-engine PRIMER §7: p = (0.5, 0.3, 0.15, 0.05), q = (0.2, 0.2, 0.2, 0.4) -> alpha = 0.6."""
    spec = _package(REPO / "04-inference-engine/serving-engine/mini-engine-core", "minengine.spec")
    spec = sys.modules["minengine.spec"]
    p, q = np.array(PRIMER_P), np.array(PRIMER_Q)
    assert DR.acceptance_rate(PRIMER_P, PRIMER_Q) == pytest.approx(spec.acceptance_rate(p, q)) == pytest.approx(0.6)
    for alpha in (0.0, 0.5, 0.6, 0.8, 0.95, 1.0):
        for k in (1, 3, 4, 5, 8):
            assert DR.expected_tokens(alpha, k) == pytest.approx(spec.expected_tokens(alpha, k))
            for c in (0.0, 0.05, 0.1, 0.15):
                assert DR.speedup(alpha, k, c) == pytest.approx(spec.speedup(alpha, k, c))
        for c in (0.05, 0.1, 0.2):
            assert DR.best_k(alpha, c) == spec.best_k(alpha, c)
    assert DR.expected_tokens(0.8, 4) == pytest.approx(3.3616)              # the primer's "alpha 0.8, k 4 -> 3.36"
    assert DR.best_k(0.8, 0.1) == 6 and DR.speedup(0.8, 6, 0.1) == pytest.approx(2.47, abs=0.005)
    # vLLM's counters for alpha = 0.6 (fact sheet): mean acceptance length = E, "acceptance rate" = (E - 1) / k
    v = DR.vllm_views(0.6, 5)
    assert v["mean_acceptance_length"] == pytest.approx(2.3834, abs=1e-4) and v["acceptance_rate"] == pytest.approx(0.2767, abs=1e-4)
    assert v["per_position"] == pytest.approx([0.6, 0.36, 0.216, 0.1296, 0.07776])
    assert DR.speedup(0.6, 3, 0.1) == pytest.approx(1.6738, abs=1e-4) == pytest.approx(spec.speedup(0.6, spec.best_k(0.6, 0.1), 0.1), abs=1e-4)
    # a greedy draft proposes argmax q = token 3, which the target keeps with p(3) = 0.05, not 0.6
    assert DR.greedy_acceptance(PRIMER_P, PRIMER_Q) == pytest.approx(0.05)


def test_cost_per_million_tokens_matches_roofline_cost():
    cost = _package(REPO / "01-hardware-gpu-fabric/roofline-and-fabric/roofline-core", "roofline.cost")
    cost = sys.modules["roofline.cost"]
    for price, tps, util, n in ((11.0, 6847, 1.0, 1), (11.0, 6847, 0.6, 1), (3.7, 571, 1.0, 1), (0.70, 1200, 0.5, 2)):
        assert C.cost_per_million_tokens(price, tps, util, n) == pytest.approx(cost.cost_per_million_tokens(price, tps, util, n))
    # 01 PRIMER §8.1: Llama-3.1-8B on an H100 at 2K context under a 10 ms ITL -> batch 68, 6,847 tok/s, $0.446/M
    assert C.cost_per_million_tokens(11, 6847) == pytest.approx(0.446, abs=5e-4)
    assert C.cost_per_million_tokens(11, 6847, 0.6) == pytest.approx(0.744, abs=5e-4)
    with pytest.raises(ValueError):
        C.cost_per_million_tokens(11, 0)                              # no batch meets the ITL: guarded, not a ZeroDivision


def test_decode_step_matches_roofline_llm():
    rl = _package(REPO / "01-hardware-gpu-fabric/roofline-and-fabric/roofline-core", "roofline.llm")
    llm, specs = sys.modules["roofline.llm"], sys.modules["roofline.specs"]
    pairs = {"t4": ("T4", "fp16"), "l4": ("L4", "bf16"), "h100-sxm": ("H100", "bf16")}
    for preset in ("qwen2.5-1.5b", "llama-3.1-8b", "llama-3.1-70b"):
        mc = llm.PRESETS[preset]
        m = C.Shape(preset, mc.n_layers, mc.d_model, mc.n_heads, mc.n_kv_heads, mc.head_dim, mc.d_ff, mc.vocab, mc.tied_embeddings)
        assert m.params() == mc.params() and m.kv_bytes_per_token() == mc.kv_bytes_per_token()
        for key, (gpu, prec) in pairs.items():
            dev = specs.get(key)
            assert (C.GPUS[gpu].bw_tbs, C.GPUS[gpu].tflops[prec], C.GPUS[gpu].memory_gb) == (dev.memory_tbs, dev.tflops[prec], dev.memory_gb)
            for batch, ctx in ((1, 1024), (16, 2048), (64, 4096)):
                assert C.decode_step(m, C.GPUS[gpu], batch, ctx, precision=prec) == pytest.approx(
                    llm.decode(mc, dev, batch, ctx, precision=prec).time)
            assert C.max_batch_by_memory(m, C.GPUS[gpu], 2048) == llm.max_batch_by_memory(mc, dev, 2048)
    mc, dev = llm.PRESETS["llama-3.1-8b"], specs.get("h100-sxm")
    m = C.Shape("llama", mc.n_layers, mc.d_model, mc.n_heads, mc.n_kv_heads, mc.head_dim, mc.d_ff, mc.vocab)
    assert C.best_batch(m, C.GPUS["H100"], 2048, 0.010) == llm.best_batch_under_itl(mc, dev, 2048, 0.010) == 68
    # the bundled Qwen2.5-1.5B config gives the roofline preset's shape
    assert C.shape("qwen2.5-1.5b-instruct").params() == llm.PRESETS["qwen2.5-1.5b"].params() == 1_543_569_408


def test_capacity_primer_formulas():
    cap = _load(REPO / "00-foundations/gpu-capacity-planning/capacity.py", "capacity_primer")
    ms, h100 = cap.MISTRAL_SMALL, cap.GPUS["H100"]
    shape = C.Shape("mistral-small", ms.layers, 5120, ms.q_heads, ms.kv_heads, ms.head_dim, 32768, 131072)
    assert shape.kv_bytes_per_token(2) == pytest.approx(cap.kv_per_token_kb(ms, "bf16") * 1e3)       # 163,840 B
    assert shape.kv_bytes_per_token(1) == pytest.approx(cap.kv_per_token_kb(ms, "fp8") * 1e3)
    # one forward pass = 2 x params x tokens (capacity.prefill_flops without attention; the teacher's scoring cost)
    for p_b, s in ((24, 2048), (1.5, 512), (32, 100)):
        assert C.teacher_flops(p_b * 1e9, s) == pytest.approx(cap.prefill_flops(p_b, s))
    # training is 3x a forward pass: 6 N D (transformer primer §6)
    assert C.train_flops(1.5e9, 2e8) == pytest.approx(3 * cap.prefill_flops(1.5, 2e8)) == pytest.approx(1.8e18)
    # the single-stream decode ceiling: weights / bandwidth, as capacity.decode_tok_s_single (GB = 1e9)
    w = cap.weight_memory_gb(24, "bf16")
    assert 1 / (w * 1e9 / (C.GPUS["H100"].bw_tbs * 1e12)) == pytest.approx(cap.decode_tok_s_single(w, h100))


def test_agreement_matches_quantcore_eval():
    qroot = REPO / "04-inference-engine/quantization/quant-core"
    _package(qroot, "quantcore.eval")
    ev, gran = sys.modules["quantcore.eval"], sys.modules["quantcore.granularity"]
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(64, 50)) * 3
    test = ref + rng.normal(size=(64, 50))
    assert A.kl(ref, test) == pytest.approx(ev.kl(ref, test))
    assert A.argmax_agreement(ref, test) == pytest.approx(gran.argmax_agreement(ref, test))
    assert A.kl(ref, ref) == pytest.approx(0.0, abs=1e-12)


def _function(path: Path, name: str):
    """One top-level function's source, executed alone (its module imports its own package)."""
    src = path.read_text()
    start = src.index(f"def {name}")
    ns: dict = {}
    exec("import math\n" + src[start:src.index("\n\n\n", start)], ns)
    return ns[name]


def test_wilson_interval_matches_memory_core_and_the_agent_lab():
    cases = ((0, 10), (7, 10), (30, 60), (170, 200), (0, 20), (60, 60), (0, 0))
    homes = [REPO / "07-application-agent-framework/agent-memory/memory-core/memcore/harness.py",
             REPO / "07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/agentlab/evals/gate.py"]
    found = [h for h in homes if h.exists()]
    if not found:
        pytest.skip("neither memory-core nor the 07 agent lab is in this checkout")
    for home in found:
        theirs = _function(home, "wilson_interval")
        for passes, n in cases:
            assert A.wilson_interval(passes, n) == pytest.approx(theirs(passes, n)), (home.name, passes, n)
    # the fact sheet's worked values
    assert A.wilson_interval(30, 60) == pytest.approx((0.3773, 0.6227), abs=1e-4)
    assert A.wilson_interval(170, 200) == pytest.approx((0.7939, 0.8929), abs=1e-4)
    assert A.wilson_interval(0, 20) == pytest.approx((0.0, 0.1611), abs=1e-4)


def test_param_counts_match_servelab_sizing():
    lab = REPO / "04-inference-engine/serving-engine/vllm-serving-lab"
    _package(lab, "servelab.sizing")
    S = sys.modules["servelab.sizing"]
    for name in ("qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct", "qwen2.5-7b-instruct", "qwen3-0.6b", "qwen3-8b",
                 "qwen3-1.7b", "qwen3-4b", "deepseek-r1-distill-qwen-1.5b", "qwen2.5-32b-instruct"):
        cfg = C.load_config(name)
        assert H.param_count(cfg).total == S.param_count(S.ModelConfig.from_hf(cfg, name)).total, name
    # the fact sheet's table (§7): exact counts from the bundled configs
    assert H.param_count(C.load_config("qwen2.5-0.5b-instruct")).total == 494_032_768
    assert H.param_count(C.load_config("qwen3-0.6b")).total == 596_049_920
    assert H.param_count(C.load_config("qwen2.5-1.5b-instruct")).total == 1_543_714_304


def test_api_cost_matches_the_scaling_lab():
    root = REPO / "06-gateway/scaling-admission-cost/agentic-scaling-lab"
    _package(root, "scalelab.capacity")
    cap = sys.modules["scalelab.capacity"]
    for model, (inp, out, cached) in TE.PRICES.items():
        assert cap.PRICES[model][:3] == (inp, out, cached), model
        for i, o, c in ((1000, 200, 0), (20_000, 2_000, 15_000), (0, 1e6, 0)):
            assert TE.api_cost(model, i, o, c) == pytest.approx(cap.cost_per_call(model, i, o, c))
    # the fact sheet's fixed cost: 100k prompts x 2,000 teacher tokens at $9/M output = $1,800
    assert TE.api_cost("gemini-3.5-flash", 0, 2e8) == pytest.approx(1800.0)
    assert math.isclose(C.fixed_cost(teacher_tokens=2e8, teacher_price_per_m=9.0, student_params=1.5e9,
                                     train_tokens=2e8, gpu=C.GPUS["H100"], mfu=0.4)["training GPU-hours"], 1.2634, rel_tol=1e-4)
