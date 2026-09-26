"""Which scheme each GPU runs natively, the vLLM kernel, the flags; checkpoints -> vLLM schemes."""
from pathlib import Path

import pytest

from quantlab import compress, serve

SAMPLES = Path(__file__).resolve().parents[1] / "quantlab/data/samples"


def test_the_gpu_table_encodes_the_source_rules():
    t4 = serve.plan("fp8", "T4")
    assert t4.supported and "weight-only" in t4.compute and t4.kernel == "MarlinFP8ScaledMMLinearKernel"
    assert "--dtype" in t4.flags and t4.flags[t4.flags.index("--dtype") + 1] == "half"
    assert "FP8 x FP8" in serve.plan("fp8", "L4").compute
    assert serve.plan("w4a16", "H100-80GB").kernel == "MacheteLinearKernel"       # Hopper only
    assert serve.plan("w4a16", "B200").kernel == "MarlinLinearKernel"
    assert not serve.plan("w8a8-int8", "B200").supported                          # INT8 W8A8 refused on sm_100+
    assert "W4A4" in serve.plan("nvfp4", "B200").compute and "weight-only" in serve.plan("nvfp4", "H100-80GB").compute
    assert not serve.plan("fp8", "T4", kv_cache_dtype="fp8").supported


def test_online_fp8_uses_the_name_that_works_on_main():
    p = serve.plan("fp8-online", "L4")
    assert p.flags[:2] == ["--quantization", "fp8_per_tensor"] and "--quantization fp8 " not in p.command() + " "


def test_checkpoint_to_vllm_scheme():
    ct = lambda s: {"quantization_config": compress.quantization_config(compress.Recipe(s))}  # noqa: E731
    assert serve.vllm_scheme(ct("W4A16")["quantization_config"])["scheme"] == "CompressedTensorsWNA16"
    assert serve.check_checkpoint(ct("W4A16"), "T4")[0]
    ok, msg = serve.check_checkpoint(ct("FP8_DYNAMIC"), "A100-80GB")
    assert ok and "weight-only" in msg                                             # W8A16 fallback below sm_89
    ok, msg = serve.check_checkpoint(ct("W8A8"), "B200")
    assert not ok and "10.0" in msg
    assert serve.vllm_scheme(ct("NVFP4A16")["quantization_config"])["scheme"].startswith("CompressedTensorsW4A4Fp4")


def test_deploy_renderers():
    p = serve.plan("fp8-online", "L4", kv_cache_dtype="fp8")
    tf = serve.cloud_run_tfvars(p)
    assert 'extra_args             = ["--quantization", "fp8_per_tensor", "--kv-cache-dtype", "fp8"]' in tf
    assert "max_model_len          = 4096" in tf
    args = serve.gke_container_args(p)
    assert args[:2] == ["Qwen/Qwen2.5-1.5B-Instruct", "--port=8000"] and "--kv-cache-dtype=fp8" in args
    assert serve.plan("w4a16", "T4").docker().startswith("docker run --rm --gpus all") and serve.IMAGE in serve.plan("w4a16", "T4").docker()


def test_startup_log_parser_on_the_samples():
    a = serve.parse_startup_log((SAMPLES / "vllm_startup_w4a16_t4.log").read_text())
    assert a["kernels"] == [("MarlinLinearKernel", "CompressedTensorsWNA16")] and a["attention_backend"] == "TRITON_ATTN"
    b = serve.parse_startup_log((SAMPLES / "vllm_startup_fp8_fp8kv_l4.log").read_text())
    assert b["kv_cache_dtype"] == "fp8" and b["attention_backend"] == "FLASHINFER" and b["kv_cache_tokens"] == 182_128


def test_quantcore_names_map_onto_the_lab_keys():
    """The core (quantcore.cost) and the lab name the same GPUs and schemes differently; both spellings work."""
    assert serve.gpu("H100-SXM") is serve.gpu("H100-80GB") and serve.gpu("RTX-PRO-6000") is serve.gpu("RTXPRO6000")
    assert serve.plan("w8a8-fp8", "L4").scheme == "fp8" and serve.plan("w4a4-nvfp4", "B200").scheme == "nvfp4"
    with pytest.raises(KeyError, match="runs as below sm_89"):
        serve.plan("w8a16-fp8", "L4")


def test_cache_config_info_from_metrics():
    """vLLM exports its CacheConfig as the labels of vllm:cache_config_info; the fake server does the same."""
    line = ('vllm:cache_config_info{block_size="16",cache_dtype="fp8",enable_prefix_caching="True",'
            'gpu_memory_utilization="0.92",num_gpu_blocks="11383"} 1.0\n')
    got = serve.parse_cache_config_info("# HELP x\n" + line)
    assert (got["cache_dtype"], got["block_size"], got["num_gpu_blocks"]) == ("fp8", 16, 11383)
    assert serve.parse_cache_config_info("vllm:num_requests_running 0\n") == {}
    from quantlab import bench, env
    from quantlab.fakeserver import FakeServer
    prof = bench.profile("qwen2.5-0.5b-instruct", "L4", "fp8", kv_cache_dtype="fp8")
    with FakeServer(prof) as url:
        info = serve.parse_cache_config_info(env.get_text(url, "/metrics"))
    assert (info["cache_dtype"], info["num_gpu_blocks"]) == ("fp8", prof.num_blocks)
