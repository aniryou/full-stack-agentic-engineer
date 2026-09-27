"""distillcore restates formulas that already have a home in this repo. Each test reproduces the other topic's
own worked numbers with distillcore's code, then — when that topic is in the checkout — compares function by
function with the original, loaded by path (tests/conftest.py: repo_module). The test names say whose numbers."""
import importlib.util
import math
import sys

import numpy as np
import pytest

from distillcore import ThinkToy, cost as K, draft as Dr, eval as E, onpolicy as op, reasoning as R
from tests.conftest import REPO, repo_module

MINENGINE = "04-inference-engine/serving-engine/mini-engine-core/minengine"
ROOFLINE = "01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/roofline"
QUANTCORE = "04-inference-engine/quantization/quant-core/quantcore"
RLCORE = "00-foundations/rl-and-thinking-models/rl-core/rlcore"
MEMCORE = "07-application-agent-framework/agent-memory/memory-core/memcore"
SCALELAB = "06-gateway/scaling-admission-cost/agentic-scaling-lab/scalelab"


def _capacity():
    path = REPO / "00-foundations/gpu-capacity-planning/capacity.py"
    if not path.is_file():
        pytest.skip("capacity.py not in this checkout")
    spec = importlib.util.spec_from_file_location("_repo_capacity", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_repo_capacity"] = mod                            # dataclasses look their module up here
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = saved
    return mod


# -- serving-engine PRIMER §7 and minengine.spec ------------------------------------------------------------
def test_reproduces_minengine_spec_serving_engine_primer_section_7():
    """p = (0.5, 0.3, 0.15, 0.05), q = (0.2, 0.2, 0.2, 0.4) → α = 0.6; α = 0.8, k = 4 → 3.36 tokens per pass;
    α = 0.8, c = 0.1 → best k = 6 at 2.47× (k = 4, 5, 6, 7: 2.40, 2.46, 2.47, 2.45)."""
    p, q = np.array([0.5, 0.3, 0.15, 0.05]), np.array([0.2, 0.2, 0.2, 0.4])
    assert round(Dr.acceptance_rate(p, q), 10) == 0.6 and round(Dr.expected_tokens(0.8, 4), 4) == 3.3616
    assert Dr.best_k(0.8, 0.1) == 6 and [round(Dr.speedup(0.8, k, 0.1), 2) for k in (4, 5, 6, 7)] == [2.40, 2.46, 2.47, 2.45]
    spec = repo_module(MINENGINE, "spec")
    for a in (0.3, 0.6, 0.72, 0.8, 0.95, 1.0):
        for k in (1, 3, 4, 8):
            assert Dr.expected_tokens(a, k) == spec.expected_tokens(a, k)
            assert Dr.speedup(a, k, 0.15) == spec.speedup(a, k, 0.15)
        assert Dr.best_k(a, 0.1) == spec.best_k(a, 0.1)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, y = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))
        assert Dr.acceptance_rate(x, y) == spec.acceptance_rate(x, y)


# -- roofline PRIMER §3.3, §8.1 and roofline.llm / roofline.cost ----------------------------------------------
def test_reproduces_roofline_cost_and_decode_roofline_primer_sections_3_and_8():
    """§8.1: Llama-3.1-8B on an H100, 2K context, 10 ms ITL → batch 68, 9.93 ms, 6,847 tok/s, $0.446/M at $11/GPU-h,
    $0.744 at 60%; FP8 → 193, 9.98 ms, 19,345 tok/s, $0.158. §3.3: batch 1 at 1K → 4.52 ms (221 tok/s) on an H100,
    50.5 ms (19.8 tok/s) on an L4. §3.1: 8.03 B parameters, 131,072 B of KV per token."""
    m, h = K.SHAPES["llama-3.1-8b"], K.GPUS["h100"]
    assert round(m.params() / 1e9, 2) == 8.03 and m.kv_bytes_per_token() == 131_072
    b = K.best_batch(m, h, 2048, 0.010)
    t = K.decode_step(m, h, b, 2048)
    assert b == 68 and round(t * 1e3, 2) == 9.93 and round(b / t) == 6847
    assert round(K.cost_per_million_tokens(11, b / t), 3) == 0.446 and round(K.cost_per_million_tokens(11, b / t, 0.6), 3) == 0.744
    kw = dict(wbytes=1, kvbytes=1, precision="fp8")
    b8 = K.best_batch(m, h, 2048, 0.010, **kw)
    t8 = K.decode_step(m, h, b8, 2048, **kw)
    assert b8 == 193 and round(t8 * 1e3, 2) == 9.98 and round(b8 / t8) == 19345 and round(K.cost_per_million_tokens(11, b8 / t8), 3) == 0.158
    t1, tl4 = K.decode_step(m, h, 1, 1024), K.decode_step(m, K.GPUS["l4"], 1, 1024)
    assert round(t1 * 1e3, 2) == 4.52 and round(1 / t1) == 221 and round(tl4 * 1e3, 1) == 50.5 and round(1 / tl4, 1) == 19.8
    cost, llm, specs = repo_module(ROOFLINE, "cost"), repo_module(ROOFLINE, "llm"), repo_module(ROOFLINE, "specs")
    for price, tps, u, n in ((11, 6847, 1.0, 1), (3.7, 571, 0.6, 1), (0.7, 294, 1.0, 8)):
        assert K.cost_per_million_tokens(price, tps, u, n) == cost.cost_per_million_tokens(price, tps, u, n)
    assert K.SHAPES["qwen2.5-1.5b"].params() == llm.PRESETS["qwen2.5-1.5b"].params()
    assert K.SHAPES["llama-3.1-8b"].params() == llm.PRESETS["llama-3.1-8b"].params()
    for batch, ctx in ((1, 1024), (68, 2048), (300, 4096)):
        assert math.isclose(K.decode_step(m, h, batch, ctx), llm.decode(llm.PRESETS["llama-3.1-8b"], specs.get("h100-sxm"), batch, ctx).time)
    assert K.best_batch(m, h, 2048, 0.010) == llm.best_batch_under_itl(llm.PRESETS["llama-3.1-8b"], specs.get("h100-sxm"), 2048, 0.010)
    assert K.max_batch(m, h, 2048) == llm.max_batch_by_memory(llm.PRESETS["llama-3.1-8b"], specs.get("h100-sxm"), 2048)


# -- the capacity primer and capacity.py ----------------------------------------------------------------------
def test_reproduces_capacity_py_and_the_capacity_primers_bank_example():
    """24B → 48 GB bf16; Mistral Small 163.84 kB of KV per token; 48 GB on an H100 → ~70 tok/s single-stream; the
    bank at 1,650 tokens of context → 88.8 sessions per GPU in bf16, 355.1 in fp8; 24B × 2K prompt ≈ 100 TFLOP."""
    assert K.weight_gb(24e9) == 48 and K.weight_gb(24e9, 1) == 24
    kv, kv8 = K.kv_per_token_kb(40, 8, 128), K.kv_per_token_kb(40, 8, 128, 1)
    assert kv == 163.84 and kv8 == 81.92 and round(K.decode_tok_s_single(48, 3.35), 1) == 69.8
    assert round(K.sessions_per_gpu(80, 48, kv, 1650), 1) == 88.8 and round(K.sessions_per_gpu(80, 24, kv8, 1650), 1) == 355.1
    assert round(K.training_flops(24e9, 2048, per_token=2) / 1e12) == 98                  # forward only: 2·P·S
    cap = _capacity()
    assert K.weight_gb(24e9) == cap.weight_memory_gb(24) and kv == cap.kv_per_token_kb(cap.MISTRAL_SMALL)
    assert kv8 == cap.kv_per_token_kb(cap.MISTRAL_SMALL, "fp8")
    spare = cap.usable_hbm_gb(cap.GPUS["H100"]) - cap.weight_memory_gb(24)
    assert math.isclose(K.sessions_per_gpu(80, 48, kv, 1650), cap.max_concurrent_sessions(spare, cap.MISTRAL_SMALL, 1650))
    assert math.isclose(K.decode_tok_s_single(48, 3.35), cap.decode_tok_s_single(48, cap.GPUS["H100"]))
    assert K.training_flops(24e9, 2048, per_token=2) == cap.prefill_flops(24, 2048)


# -- quantization PRIMER §8 and quantcore.eval -----------------------------------------------------------------
def test_reproduces_quantcore_eval_quantization_primer_section_8():
    """250 GSM8K items at 76.8% → ±0.0268; 85 lost against 51 gained → z = −2.9; and `compare`'s flips, accuracies
    and paired z from the same items."""
    assert round(E.accuracy_stderr(0.768, 250), 4) == 0.0268 and round(E.paired_z(85, 51), 1) == -2.9
    ev = repo_module(QUANTCORE, "eval")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(50, 17)), rng.normal(size=(50, 17))
    assert math.isclose(E.kl(a, b), ev.kl(a, b)) and E.argmax_agreement(a, b) == ev.argmax_agreement(a, b)
    assert E.accuracy_stderr(0.768, 250) == ev.accuracy_stderr(0.768, 250) and E.paired_z(85, 51) == ev.paired_z(85, 51)
    labels = rng.integers(0, 17, 50)
    theirs = ev.compare(a, b, labels)                              # logits in; correctness per item is what matters
    ours = E.compare(np.argmax(a, -1) == labels, np.argmax(b, -1) == labels)
    assert (theirs["lost"], theirs["gained"], theirs["n"]) == (ours["lost"], ours["gained"], ours["n"])
    assert theirs["acc_ref"] == ours["teacher"] and theirs["acc"] == ours["student"] and theirs["paired_z"] == ours["paired_z"]
    assert math.isclose(theirs["kl"], E.kl(a, b)) and theirs["top1"] == E.argmax_agreement(a, b)


# -- rl-and-thinking-models PRIMER §2, §5 and rlcore -------------------------------------------------------------
def test_reproduces_rlcore_thinktask_rl_primer_section_2():
    """ThinkTask(e0 = 0.8, q = 0.1): L* = 20.23 at c = 0.01; ThinkTask(q = 0.15, max_think = 16) under a policy
    that answers at once half the time: accuracy 0.304, a mean of 1.0 thinking token."""
    task = ThinkToy(0.8, 0.1, 64)
    assert round(task.optimal_length(0.01), 2) == 20.23
    e = R.LengthPolicy(16).expected(ThinkToy(0.8, 0.15, 16))
    assert round(e["accuracy"], 3) == 0.304 and round(e["length"], 1) == 1.0
    tasks = repo_module(RLCORE, "tasks")
    for e0, q, c in ((0.8, 0.1, 0.01), (0.8, 0.15, 0.02), (0.6, 0.05, 0.001)):
        theirs = tasks.ThinkTask(e0=e0, q=q, max_think=64, cost=c)
        assert ThinkToy(e0, q, 64).optimal_length(c) == theirs.optimal_length()
        assert np.allclose(ThinkToy(e0, q, 64).accuracy(np.arange(64)), theirs.accuracy(np.arange(64)))
    theirs = tasks.ThinkTask(e0=0.8, q=0.15, max_think=16).expected(np.full(16, 0.5))
    assert math.isclose(theirs["accuracy"], e["accuracy"]) and math.isclose(theirs["length"], e["length"])


def test_matches_rlcore_pg_reinforce_grad_convention_with_the_teacher_as_reference():
    """On-policy distillation's advantages are rlcore.pg.reinforce_grad's reward shaping with ref = teacher,
    beta = 1 and zero task reward: the same sign, the same batch-mean baseline, the same 1/N average."""
    pg, policy, tasks = repo_module(RLCORE, "pg"), repo_module(RLCORE, "policy"), repo_module(RLCORE, "tasks")
    task = tasks.SeqTask("brackets", 8)
    rng = np.random.default_rng(0)
    student = policy.Policy.for_task(task, rng.normal(0, 0.5, (task.n_states, task.n_actions)))
    teacher = policy.Policy.for_task(task, rng.normal(0, 1.0, (task.n_states, task.n_actions)))
    trajs = student.sample(task, rng, 16)
    for t in trajs:
        t.reward = 0.0                                             # distillation has no task reward
    theirs = pg.reinforce_grad(student, trajs, "mean", ref=teacher, beta=1.0)
    s_lp = np.array([student.token_logprobs(t) for t in trajs])
    t_lp = np.array([teacher.token_logprobs(t) for t in trajs])
    w = op.advantages(s_lp, t_lp)
    ours = sum(student.grad_logprob(t, w[i]) for i, t in enumerate(trajs))
    assert np.allclose(ours, theirs, atol=1e-12)


# -- Wilson intervals: memory-core and the 07 platform lab --------------------------------------------------------
def test_reproduces_the_wilson_intervals_of_memory_core_and_the_platform_labs_evals_notebook():
    """memory PRIMER: 45/50 → (0.7864, 0.9565), 0/20 → (0.0, 0.1611), 12/13 → 67%–99%, 360/390 → 89.2%–94.6%;
    platform lab notebook 08: 9/10 → (0.596, 0.982), 10/10 → a lower bound of 0.722."""
    r4 = lambda t: tuple(round(x, 4) for x in t)
    assert r4(E.wilson_interval(45, 50)) == (0.7864, 0.9565) and r4(E.wilson_interval(0, 20)) == (0.0, 0.1611)
    lo, hi = E.wilson_interval(12, 13)
    assert (round(lo * 100), round(hi * 100)) == (67, 99)
    lo, hi = E.wilson_interval(360, 390)
    assert (round(lo * 100, 1), round(hi * 100, 1)) == (89.2, 94.6)
    assert tuple(round(x, 3) for x in E.wilson_interval(9, 10)) == (0.596, 0.982)
    assert round(E.wilson_interval(10, 10)[0], 3) == 0.722 and E.wilson_interval(10, 10)[1] == 1.0
    harness = repo_module(MEMCORE, "harness")
    for k, n in ((0, 20), (9, 10), (45, 50), (360, 390), (13, 13)):
        assert E.wilson_interval(k, n) == harness.wilson_interval(k, n)


# -- the 06 scaling lab's price model --------------------------------------------------------------------------
def test_reproduces_scalelab_cost_per_call_scaling_primer_section_3_4():
    """A 3.5 Flash call with 2,300 uncached + 2,700 cached input and 350 output tokens → $0.0070 ($0.007005);
    the same call on 3.5 Flash-Lite → $0.00165."""
    assert K.api_cost("gemini-3.5-flash", 5000, 350, 2700) == pytest.approx(0.007005)
    assert round(K.api_cost("gemini-3.5-flash-lite", 5000, 350, 2700), 5) == 0.00165
    cap = repo_module(SCALELAB, "capacity")
    for m in K.PRICES:
        assert K.PRICES[m] == cap.PRICES[m][:3]
        assert K.api_cost(m, 5000, 350, 2700) == cap.cost_per_call(m, 5000, 350, 2700)
