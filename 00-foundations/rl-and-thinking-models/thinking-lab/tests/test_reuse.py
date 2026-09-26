"""Numbers this lab re-implements must match their home in the repo (SPEC §6b): the capacity primer's
capacity.py, the 04 lab's servelab.sizing, and the 07 agent lab's Wilson interval. Each source is loaded
by path from the repo (labs never import each other); the test skips if the repo is not around it."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from thinklab import engine as E
from thinklab import workload as W
from thinklab.thinking import ttc

REPO = Path(__file__).resolve().parents[4]


def _load(path: Path, name: str):
    if not path.exists():
        pytest.skip(f"{path} not in this checkout")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_capacity_primer_reproduced_from_capacity_py():
    cap = _load(REPO / "00-foundations/gpu-capacity-planning/capacity.py", "capacity_primer")
    h100, model = cap.GPUS["H100"], cap.MISTRAL_SMALL
    rps = 10_000 * 0.10 * 0.5 / 60
    for out in (300, 3000):
        mine = W.capacity_primer_view(rps, 1500, out)
        dur = cap.request_duration_s(model.active_b, 1500, out, h100, tpot_ms=40, dtype="fp8")
        conc = cap.concurrency(rps, dur)
        spare = cap.usable_hbm_gb(h100) - cap.weight_memory_gb(model.params_b, "fp8")
        sessions = cap.max_concurrent_sessions(spare, model, 1500 + out // 2, "fp8")
        assert mine["duration_s"] == pytest.approx(dur) and mine["concurrency"] == pytest.approx(conc)
        assert mine["kv_per_session_gb"] == pytest.approx(cap.kv_per_session_gb(model, 1500 + out // 2, "fp8"))
        assert mine["sessions_per_gpu"] == pytest.approx(sessions)
        assert mine["gpus_for_memory"] == pytest.approx(conc / sessions)


def _servelab():
    lab = REPO / "04-inference-engine/serving-engine/vllm-serving-lab"
    if not (lab / "servelab/sizing.py").exists():
        pytest.skip("04 serving lab not in this checkout")
    sys.path.insert(0, str(lab))
    try:
        from servelab import sizing
    finally:
        sys.path.remove(str(lab))
    return sizing, lab


def test_profiles_reproduce_servelab_sizing():
    S, lab = _servelab()
    base = json.loads((lab / "servelab/data/configs/qwen3-0.6b.json").read_text())
    shapes = {"t4-qwen3-0.6b": (base, "T4", "half", 8192),
              "t4-qwen3-1.7b": (dict(base, hidden_size=2048, intermediate_size=6144), "T4", "half", 8192),
              "l4-qwen3-4b": (dict(base, hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                                   num_attention_heads=32), "L4", "auto", 16384)}
    for name, (cfg, gpu, dtype, mml) in shapes.items():
        m = S.ModelConfig.from_hf(cfg, name=name)
        p = E.PROFILES[name]
        assert S.param_count(m).total == p["params"], name
        assert S.kv_bytes_per_token(m) == p["kv_bytes_per_token"], name
        assert S.size(m, S.gpu(gpu), dtype=dtype, max_model_len=mml).num_blocks == p["num_blocks"], name
        g = S.gpu(gpu)
        assert E._gpu(gpu) == (g.mem_bw_gbs, g.bf16_tflops), name


def test_wilson_interval_matches_the_agent_lab():
    path = REPO / "07-application-agent-framework/agent-fundamentals/gcp-agent-platform-lab/agentlab/evals/gate.py"
    if not path.exists():
        pytest.skip("07 agent lab not in this checkout")
    src = path.read_text()
    start = src.index("def wilson_interval")
    ns: dict = {}
    exec("import math\n" + src[start:src.index("\n\n\n", start)], ns)   # just the function: the module imports its package
    for passes, n in ((0, 10), (7, 10), (30, 60), (60, 60), (0, 0)):
        assert ttc.wilson_interval(passes, n) == pytest.approx(ns["wilson_interval"](passes, n))
