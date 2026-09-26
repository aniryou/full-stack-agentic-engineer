"""The tuner: flag rendering, grids, choosing under an SLO, rate search, and a fake-backend sweep."""
import math

import pytest

from servelab.bench import SLO, Lengths, random_requests
from servelab.bench.summary import Stat, Summary
from servelab.tune import FakeBackend, Trial, VLLMBackend, best, grid, max_rate_under_slo, sweep, to_cli_flags


def test_cli_flags_render_like_vllm_expects():
    flags = to_cli_flags({"max_num_seqs": 64, "enable_prefix_caching": False, "enforce_eager": True,
                          "speculative_config": {"method": "ngram", "num_speculative_tokens": 4}, "quantization": None})
    assert flags == ["--max-num-seqs", "64", "--no-enable-prefix-caching", "--enforce-eager",
                     "--speculative-config", '{"method":"ngram","num_speculative_tokens":4}']
    cmd = VLLMBackend("Qwen/Qwen2.5-0.5B-Instruct", base_flags={"dtype": "half"}, port=8001).command({"max_num_seqs": 8})
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen2.5-0.5B-Instruct"] and cmd[-4:] == ["--dtype", "half", "--max-num-seqs", "8"]


def test_grid_is_a_cartesian_product():
    g = grid(max_num_seqs=[8, 32], enable_prefix_caching=[True, False])
    assert len(g) == 4 and {"max_num_seqs": 32, "enable_prefix_caching": False} in g


def _trial(cfg, attainment, throughput):
    st = Stat(1, 1.0, 1.0, 0.0, {99: 1.0})
    s = Summary(10, 0, 1.0, 100, 100, 10.0, throughput, throughput, 10 * attainment, attainment, st, st, st, st)
    return Trial(cfg, s, None, None)


def test_best_is_the_fastest_config_that_meets_the_slo():
    trials = [_trial({"a": 1}, 0.99, 500.0), _trial({"a": 2}, 0.80, 900.0), _trial({"a": 3}, 0.95, 700.0)]
    assert best(trials, 0.9).config == {"a": 3}          # the 900 tok/s config misses the SLO
    assert best(trials, 0.999) is None


def test_max_rate_bisection_converges():
    knee = 7.3
    rate, hist = max_rate_under_slo(lambda r: 1.0 if r <= knee else 0.5, lo=1.0, hi=64.0, iters=8)
    assert knee / 1.04 <= rate <= knee and len(hist) == 10
    assert math.isnan(max_rate_under_slo(lambda r: 0.0, 1.0, 8.0)[0])


def test_fake_backend_rejects_knobs_it_does_not_model():
    with pytest.raises(ValueError):
        FakeBackend("tiny").build({"tensor_parallel_size": 2})
    prof, cfg = FakeBackend("tiny", spec_acceptance=0.5).build(
        {"max_num_seqs": 4, "speculative_config": {"method": "eagle", "num_speculative_tokens": 3}})
    assert cfg.max_num_seqs == 4 and cfg.num_speculative_tokens == 3 and cfg.spec_acceptance == 0.5
    with pytest.raises(ValueError, match="spec_acceptance"):     # not a vLLM SpeculativeConfig field
        FakeBackend("tiny").build({"speculative_config": {"method": "ngram", "num_speculative_tokens": 4,
                                                          "acceptance": 0.7}})


def test_num_gpu_blocks_override_sets_the_block_pool():
    prof, _ = FakeBackend("tiny").build({"num_gpu_blocks_override": 96})
    assert prof.num_blocks == 96
    assert to_cli_flags({"num_gpu_blocks_override": 96}) == ["--num-gpu-blocks-override", "96"]


def test_sweep_on_the_fake_backend():
    trials = sweep(FakeBackend("tiny"), grid(max_num_seqs=[1, 8]),
                   lambda: random_requests(8, Lengths.fixed(32), Lengths.fixed(8), seed=2), rate=math.inf,
                   slo=SLO(ttft_ms=10_000), warmup=1)
    one, eight = trials
    assert one.summary.completed == eight.summary.completed == 8
    # with a burst of 8, one-at-a-time batching makes later requests wait: TTFT tail grows
    assert one.summary.ttft.p[99] > eight.summary.ttft.p[99]
    assert one.snapshot is not None and one.snapshot.requests_finished == 8
