"""KV sizing reproduces the serving lab (servelab.sizing); backend rules follow vLLM's source."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from quantlab import kv

# servelab.sizing numbers for Llama-3.1-8B-Instruct on an L4 at vLLM v0.30.0 defaults (facts sheet, re-run
# 2026-09-26): weights, KV budget, blocks, sessions of 2,000 tokens. Serving-engine PRIMER §8 and
# vllm-internals §4.7/§8.2 quote the same rows.
SERVELAB = {("bf16", "auto"): (16_060_522_496, 4_957_217_896, 2_363, 18.9),
            ("fp8", "auto"): (9_081_200_640, 11_936_539_752, 5_691, 45.5),
            ("w4a16", "auto"): (5_727_854_592, 15_289_885_800, 7_290, 58.3),
            ("bf16", "fp8"): (16_060_522_496, 4_957_217_896, 4_727, 37.8),
            ("fp8", "fp8"): (9_081_200_640, 11_936_539_752, 11_383, 91.1),
            ("w4a16", "fp8"): (5_727_854_592, 15_289_885_800, 14_581, 116.6)}


@pytest.mark.parametrize("weights,kvd", list(SERVELAB))
def test_reproduces_servelab_sizing(weights, kvd):
    r = kv.size("llama-3.1-8b-instruct", "L4", weights=weights, kv_cache_dtype=kvd)
    w, budget, blocks, sessions = SERVELAB[(weights, kvd)]
    assert (r.weight_bytes, r.kv_budget_bytes, r.num_blocks) == (w, budget, blocks)
    assert round(r.sessions(2000), 1) == sessions


def test_parameter_counts_and_kv_bytes():
    llama, qwen = kv.load_shape("llama-3.1-8b-instruct"), kv.load_shape("qwen2.5-0.5b-instruct")
    assert kv.params(llama).total == 8_030_261_248 and kv.params(llama).linear == 6_979_321_856
    assert kv.params(qwen) == kv.Params(494_032_768, 136_134_656, 357_826_560)
    assert kv.kv_bytes_per_token(llama) == 131_072 and kv.kv_bytes_per_token(llama, "fp8") == 65_536
    assert kv.kv_bytes_per_token(qwen) == 12_288
    assert kv.size("qwen2.5-0.5b-instruct", "T4").num_blocks == 66_023          # servelab: T4 66,023, L4 103,656
    assert kv.size("qwen2.5-0.5b-instruct", "L4").num_blocks == 103_656


def test_live_cross_check_with_servelab_when_the_repo_has_it():
    """Load the serving lab's sizing.py by path (no package import) and compare on a grid."""
    path = Path(__file__).resolve().parents[3] / "serving-engine/vllm-serving-lab/servelab/sizing.py"
    if not path.exists():
        pytest.skip("serving lab not in this checkout")
    mod = types.ModuleType("servelab_sizing_crosscheck")   # compiled from source: no bytecode written there
    mod.__file__ = str(path)
    sys.modules[mod.__name__] = mod                        # dataclasses look their module up here
    try:
        exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)  # noqa: S102 — the repo's own file
    finally:
        sys.modules.pop(mod.__name__, None)
    qmap = {"bf16": None, "fp8": "fp8", "w4a16": "int4"}
    for model in ("llama-3.1-8b-instruct", "qwen2.5-1.5b-instruct", "qwen2.5-0.5b-instruct"):
        for gpu in ("T4", "L4", "H100-80GB"):
            for w, q in qmap.items():
                for kvd in ("auto", "fp8"):
                    theirs = mod.size(model, gpu, quantization=q, kv_cache_dtype=kvd, max_model_len=4096)
                    ours = kv.size(model, gpu, weights=w, kv_cache_dtype=kvd, max_model_len=4096)
                    assert ours.num_blocks == theirs.num_blocks, (model, gpu, w, kvd)


def test_attention_backend_rules():
    be = kv.attention_backend
    assert be("T4", "auto").backend == "TRITON_ATTN"
    assert be("T4", "fp8").error and "sm_89" in be("T4", "fp8").error             # no FP8 KV on a T4, any backend
    assert be("A100-80GB", "auto").backend == "FLASH_ATTN" and be("A100-80GB", "fp8").backend == "FLASHINFER"
    assert be("L4", "fp8").backend == "FLASHINFER"                                  # the flag changes the backend
    assert be("H100-80GB", "fp8").backend.startswith("FLASH_ATTN") and be("H100-80GB", "fp8_e5m2").backend == "FLASHINFER"
    assert be("B200", "nvfp4").backend == "FLASHINFER" and be("L4", "nvfp4").error
    assert be("T4", "int8_per_token_head").backend == "TRITON_ATTN"


def test_kv_quantizers_on_a_tensor():
    x = np.random.default_rng(0).standard_normal((2, 2, 5, 32)) * 3
    e4 = kv.kv_quantizer("fp8")("k", 0, x)
    e5 = kv.kv_quantizer("fp8_e5m2")("k", 0, x)
    i8 = kv.kv_quantizer("int8_per_token_head")("k", 0, x)
    err = lambda y: np.linalg.norm(y - x) / np.linalg.norm(x)  # noqa: E731
    assert err(i8) < err(e4) < err(e5) < 0.1                                        # 7 vs 3 vs 2 mantissa bits
    tiny = kv.kv_quantizer("fp8", k_scale=1e-3)("k", 0, x)                         # a stale scale saturates at 448e-3
    assert np.abs(tiny).max() <= 0.448 + 1e-9
    assert kv.kv_quantizer("auto") is None


def test_llama_70b_on_one_h100_at_vllm_defaults():
    """PRIMER §10 and drill 6: FP8 Llama-3.1-70B fits one H100 with room for only 2 (BF16 KV) or 4 (FP8 KV)
    sessions of 4,000 tokens at vLLM's defaults (0.92 of the 79.65 GiB the driver reports); INT4 + FP8 KV
    fits 54. quant-core's round inputs (0.9 x 80e9 B - 1 GB) leave 0 for FP8 (its test_primer_numbers)."""
    cfg = {"model_type": "llama", "num_hidden_layers": 80, "hidden_size": 8192, "num_attention_heads": 64,
           "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 28672, "vocab_size": 128256,
           "max_position_embeddings": 131072, "tie_word_embeddings": False}
    assert kv.params(kv.load_shape(cfg)).total == 70_553_706_496
    got = {(w, d): kv.size(cfg, "H100-80GB", weights=w, kv_cache_dtype=d).num_blocks
           for w, d in (("fp8", "auto"), ("fp8", "fp8"), ("w4a16", "fp8"))}
    assert got == {("fp8", "auto"): 523, ("fp8", "fp8"): 1047, ("w4a16", "fp8"): 13593}
    assert [b * 16 // 4000 for b in got.values()] == [2, 4, 54]
