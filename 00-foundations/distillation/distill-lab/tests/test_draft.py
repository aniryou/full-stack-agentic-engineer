"""The distilled draft: vLLM's counters and config, and the tiny models' acceptance (torch-free parts)."""
import json
import math
from importlib import resources

import pytest

from distillab import cost as C
from distillab import draft as DR
from distillab import metrics as M


def _sample(name):
    return resources.files("distillab.assets").joinpath("samples", name).read_text()


def test_counters_give_back_alpha_and_the_mean_acceptance_length():
    c = DR.simulate_counters([0.7] * 4, 4, 40000, seed=1)
    r = M.spec_decode(M.parse(DR.counters_text({"drafts": 0, "draft_tokens": 0, "accepted": 0, "per_position": [0] * 4})),
                      M.parse(DR.counters_text(c)))
    v = DR.vllm_views(0.7, 4)
    assert r["mean_acceptance_length"] == pytest.approx(v["mean_acceptance_length"], abs=0.03)
    assert r["acceptance_rate"] == pytest.approx(v["acceptance_rate"], abs=0.01)
    assert r["alpha_pos0"] == pytest.approx(0.7, abs=0.01) and r["k"] == 4
    assert r["per_position"] == pytest.approx(v["per_position"], abs=0.01)


def test_bundled_spec_metrics_are_current_and_parse():
    before, after = M.parse(_sample("spec_decode_metrics_before.txt")), M.parse(_sample("spec_decode_metrics_after.txt"))
    assert _sample("spec_decode_metrics_after.txt") == DR.counters_text(DR.simulate_counters([0.7] * 4, 4, 20000, seed=0))
    r = M.spec_decode(before, after)
    assert r["drafts"] == 20000 and 0.68 < r["alpha_pos0"] < 0.72


def test_speculative_config_catches_the_known_mistakes():
    cfg = DR.speculative_config("Qwen/Qwen3-0.6B", 4)
    assert cfg == {"method": "draft_model", "num_speculative_tokens": 4, "model": "Qwen/Qwen3-0.6B"}
    for bad in ({"tensor_parallel_size": 2}, {"speculative_token_tree": "[]"}):
        with pytest.raises(ValueError):
            DR.speculative_config("m", 4, **bad)
    with pytest.raises(ValueError):
        DR.speculative_config("m", 0)
    args = DR.serve_args("Qwen/Qwen3-4B", cfg)
    assert json.loads(args[args.index("--speculative-config") + 1]) == cfg


def test_vocab_check_and_cost_ratio():
    ok, _ = DR.check_vocab(C.load_config("qwen3-4b"), C.load_config("qwen3-0.6b"))
    bad, msg = DR.check_vocab(C.load_config("qwen2.5-7b-instruct"), C.load_config("qwen2.5-0.5b-instruct"))
    assert ok and not bad and "same vocabulary size" in msg
    c = DR.cost_ratio(C.shape("qwen3-0.6b").params(), C.shape("qwen3-4b").params())
    assert 0.14 < c < 0.16 and DR.best_k(0.8, c) >= 2


def test_topk_lower_bound():
    top_p = [("a", math.log(0.6)), ("b", math.log(0.3))]
    top_q = [("a", math.log(0.5)), ("c", math.log(0.4))]
    assert DR.topk_overlap_acceptance(top_p, top_q) == pytest.approx(0.5)
    assert DR.topk_overlap_acceptance(top_p, top_q) <= DR.acceptance_rate([0.6, 0.3, 0.05, 0.05], [0.5, 0.05, 0.4, 0.05])
