"""Recipes -> compressed-tensors checkpoints that load back; GPTQ/AWQ/SmoothQuant do their jobs."""
import json

import numpy as np
import pytest

from quantlab import compress as C, stio, tinymodel as tm


@pytest.fixture(scope="module")
def model():
    return tm.load()


@pytest.fixture(scope="module")
def calib(model):
    return C.calibration_inputs(model, 64)


def test_presets_match_compressed_tensors():
    w4 = C.SCHEMES["W4A16"]["weights"]
    assert (w4["num_bits"], w4["type"], w4["strategy"], w4["group_size"], w4["symmetric"]) == (4, "int", "group", 128, True)
    f8 = C.SCHEMES["FP8_DYNAMIC"]
    assert f8["weights"]["strategy"] == "channel" and f8["input_activations"]["dynamic"] is True
    assert [C.compression_format(C.SCHEMES[s]) for s in ("W4A16", "W8A8", "FP8_DYNAMIC", "NVFP4A16")] == \
        ["pack-quantized", "int-quantized", "float-quantized", "nvfp4-pack-quantized"]
    assert not C.Recipe("FP8_DYNAMIC").needs_calibration and C.Recipe("FP8").needs_calibration
    assert C.Recipe("W4A16", ("gptq",)).needs_calibration


def test_int_grid_conventions():
    s, zp = C.qparams(np.array([[-1.5, 0.3, 0.75]]), C.SCHEMES["W4A16"]["weights"])
    assert s[0, 0] == pytest.approx(1.5 / 7.5) and zp is None                     # amax / 7.5
    q = C.quantize_block(np.array([[-1.5, 1.5]]), s, None, C.SCHEMES["W4A16"]["weights"])
    assert q.tolist() == [[-8, 7]]                                                  # +amax clips to 7
    s, zp = C.qparams(np.array([[0.0, 1.0, 3.0]]), C.SCHEMES["W4A16_ASYM"]["weights"])
    assert s[0, 0] == pytest.approx(3 / 15) and zp[0, 0] == -8


@pytest.mark.parametrize("scheme,algos", [("FP8_DYNAMIC", ()), ("W4A16", ("gptq",)), ("W4A16_ASYM", ()),
                                          ("W8A8", ("smoothquant",)), ("NVFP4A16", ()), ("FP8", ())])
def test_checkpoint_roundtrip(tmp_path, model, calib, scheme, algos):
    qm = C.quantize_model(model, C.Recipe(scheme, algos), calib)
    path = C.save_checkpoint(qm, tmp_path / scheme)
    assert C.validate_checkpoint(path) == []
    cfg = json.loads((path / "config.json").read_text())["quantization_config"]
    assert cfg["quant_method"] == "compressed-tensors" and cfg["ignore"] == ["lm_head"]
    loaded, act = C.load_checkpoint(path)
    for name, lr in qm.layers.items():               # what was written is what was computed, up to bf16 scales
        np.testing.assert_allclose(loaded.weights[name + ".weight"], lr.w_hat, rtol=2 ** -7, atol=1e-6)
    p, a = tm.make_task("add", 64, 1000)
    mine = qm.model().answer_logits(p, a, act_quant=qm.act_quant()).argmax(-1)
    theirs = loaded.answer_logits(p, a, act_quant=act).argmax(-1)
    assert (mine == theirs).mean() >= 0.99
    assert loaded.weights["lm_head.weight"].dtype == np.float32


def test_w4a16_tensor_names_dtypes_and_bytes(tmp_path, model, calib):
    path = C.save_checkpoint(C.quantize_model(model, C.Recipe("W4A16"), calib), tmp_path / "w4")
    t = stio.load(path / "model.safetensors")
    n = "model.layers.0.mlp.down_proj"
    assert t[n + ".weight_packed"].dtype == "I32" and t[n + ".weight_packed"].shape == (128, 32)   # 256 * 4 / 32
    assert t[n + ".weight_scale"].dtype == "BF16" and t[n + ".weight_scale"].shape == (128, 2)
    assert t[n + ".weight_shape"].numpy().tolist() == [128, 256] and n + ".weight" not in t
    b = C.checkpoint_bytes(path)
    linear = sum(model.weights[x + ".weight"].size for x in model.linear_names())
    assert b["codes"] == linear // 2 and b["scales"] == linear // 128 * 2 + 14 * 16  # bf16 scale per 128 + weight_shape


def test_gptq_beats_rtn_where_it_matters(model, calib):
    rtn = C.quantize_model(model, C.Recipe("W4A16"), calib)
    gptq = C.quantize_model(model, C.Recipe("W4A16", ("gptq",)), calib)
    n = "model.layers.0.self_attn.q_proj"                                           # its input has the outlier channels
    assert gptq.layers[n].out_err < rtn.layers[n].out_err / 3


def test_transforms_preserve_the_function(model, calib):
    ids = tm.make_task("add", 8, 1000)[0]
    ref = model.forward(ids)
    for w in (C.smoothquant(model, calib, 0.8), C.awq(model, calib, C.SCHEMES["W4A16"]["weights"], n_grid=5)):
        np.testing.assert_allclose(tm.TinyLM(model.config, w).forward(ids), ref, atol=2e-3)
    sq = C.smoothquant(model, calib, 0.8)
    chans = tm.outlier_channels(model)
    assert sq["model.layers.0.input_layernorm.weight"][chans].max() < model.weights["model.layers.0.input_layernorm.weight"][chans].min()


def test_group_size_must_divide_in_features():
    with pytest.raises(ValueError, match="divisible"):
        C.rtn(np.zeros((4, 96)), C.Recipe("W4A16").scheme_args()["weights"])


def test_llmcompressor_script_is_valid_python():
    import ast
    for r in (C.Recipe("FP8_DYNAMIC"), C.Recipe("W4A16", ("gptq",)), C.Recipe("W8A8", ("smoothquant", "gptq")),
              C.Recipe("W4A16_ASYM", ("awq",)), C.Recipe("NVFP4")):
        src = C.llmcompressor_script(r)
        ast.parse(src)
        assert "oneshot(" in src and "ignore=['lm_head']" in src and "save_compressed=True" in src
    assert "dataset=" not in C.llmcompressor_script(C.Recipe("FP8_DYNAMIC"))
    ast.parse(C.gptqmodel_script())
