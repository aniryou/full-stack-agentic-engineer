"""The MoE layer: sparse == dense, the router variants, and parameter counts from configs."""
import numpy as np
import pytest

from moecore.moe import ROUTERS, Expert, MoELayer, route, softmax
from moecore.sizing import MODELS

RNG = np.random.default_rng(0)


@pytest.mark.parametrize("family", list(ROUTERS))
def test_sparse_forward_equals_every_expert_on_every_token(family):
    """The grouped (sort -> per-expert GEMM -> scatter-add) forward equals the dense masked reference."""
    rng = np.random.default_rng(1)
    k = 1 if family == "llama4" else 2
    layer = MoELayer.init(rng, d=16, ff=32, n_experts=8, k=k, n_shared=1 if family == "llama4" else 0,
                          **ROUTERS[family])
    if family == "gpt-oss":
        layer.router_bias = rng.standard_normal(8)
    if family == "deepseek-v3":
        layer.select_bias = rng.standard_normal(8) * 0.1
        layer.router.update(groups=4, topk_groups=2)
    if family == "llama4":
        layer.scale_input = True
    x = rng.standard_normal((37, 16))
    np.testing.assert_allclose(layer.forward(x), layer.forward_dense(x), atol=1e-12)


def test_one_expert_top_one_is_the_dense_mlp():
    rng = np.random.default_rng(2)
    layer = MoELayer.init(rng, d=8, ff=16, n_experts=1, k=1, **ROUTERS["mixtral"])
    x = rng.standard_normal((5, 8))
    np.testing.assert_allclose(layer.forward(x), layer.experts[0](x), atol=1e-12)


def test_renormalisation_conventions_hand_computed():
    logits = np.array([[2.0, 1.0, 0.0, -1.0]])
    p = softmax(logits)[0]                                     # 0.6439, 0.2369, 0.0871, 0.0321
    mix = route(logits, 2, **ROUTERS["mixtral"])
    qwen = route(logits, 2, **ROUTERS["qwen3"])
    oss = route(logits, 2, **ROUTERS["gpt-oss"])
    assert mix.idx.tolist() == [[0, 1]]
    np.testing.assert_allclose(mix.weights[0], [p[0] / (p[0] + p[1]), p[1] / (p[0] + p[1])])   # 0.731, 0.269
    np.testing.assert_allclose(qwen.weights[0], p[:2])                                        # 0.644, 0.237
    np.testing.assert_allclose(oss.weights[0], mix.weights[0])   # softmax of the top-k logits = renormalised top-k
    assert round(mix.weights[0, 0], 3) == 0.731 and round(qwen.weights[0].sum(), 3) == 0.881


def test_deepseek_bias_picks_but_never_weights():
    logits = np.array([[1.0, 0.5, 0.0, -0.5]])
    bias = np.array([0.0, 0.0, 0.0, 10.0])                     # force expert 3 in
    r = route(logits, 2, select_bias=bias, **ROUTERS["deepseek-v3"])
    s = 1 / (1 + np.exp(-logits[0]))
    assert sorted(r.idx[0].tolist()) == [0, 3]
    w = dict(zip(r.idx[0].tolist(), r.weights[0]))
    assert w[3] == pytest.approx(2.5 * s[3] / (s[0] + s[3]))  # its UNbiased score, renormalised, x 2.5
    assert sum(w.values()) == pytest.approx(2.5)


def test_group_limited_routing_stays_inside_the_kept_groups():
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((200, 256))
    r = route(logits, 8, groups=8, topk_groups=4, **ROUTERS["deepseek-v3"])
    groups_used = [len(set((row // 32).tolist())) for row in r.idx]
    assert max(groups_used) <= 4                                 # a token talks to at most 4 of 8 groups
    free = route(logits, 8, **ROUTERS["deepseek-v3"])
    assert max(len(set((row // 32).tolist())) for row in free.idx) > 4


def test_unchosen_experts_do_no_work():
    calls = []
    class Counting(Expert):
        def __call__(self, x):
            calls.append(len(x))
            return super().__call__(x)
    rng = np.random.default_rng(4)
    layer = MoELayer.init(rng, d=8, ff=8, n_experts=16, k=2, **ROUTERS["mixtral"])
    layer.experts = [Counting(e.w_up, e.w_down, e.w_gate) for e in layer.experts]
    x = rng.standard_normal((3, 8))
    layer.forward(x)
    assert sum(calls) == 3 * 2 and len(calls) <= 6              # T x k rows, at most T x k experts run


def test_param_counts_match_the_published_totals():
    """The facts sheet recomputed every family from its config; so does MoEConfig."""
    m = MODELS
    assert m["mixtral-8x7b"].total() == 46_702_526_464 and m["mixtral-8x7b"].active() == 12_879_659_008
    assert m["qwen3-30b-a3b"].total() == 30_531_911_680 and m["qwen3-30b-a3b"].active() == 3_352_821_760
    assert m["llama-3.1-8b"].total() == 8_029_995_008
    expect = {"qwen3-235b-a22b": (235.09, 22.19), "qwen1.5-moe-a2.7b": (14.32, 2.69), "deepseek-v3": (671.03, 37.55),
              "gpt-oss-120b": (116.83, 5.71), "gpt-oss-20b": (20.91, 4.19), "llama4-scout": (107.77, 17.17),
              "llama4-maverick": (400.71, 17.18), "olmoe-1b-7b": (6.92, 1.28), "granite-3.0-1b-a400m": (1.33, 0.43),
              "granite-3.0-3b-a800m": (3.30, 0.88)}
    for name, (total, active) in expect.items():
        assert round(m[name].total() / 1e9, 2) == total, name
        assert round(m[name].active() / 1e9, 2) == active, name


def test_active_parameter_conventions():
    """OpenAI counts the LM head only (5.1B / 3.6B); DeepSeek's 37B and roofline.llm count both tables."""
    assert round(MODELS["gpt-oss-120b"].active("head") / 1e9, 2) == 5.13
    assert round(MODELS["gpt-oss-20b"].active("head") / 1e9, 2) == 3.61
    ds = MODELS["deepseek-v3"]
    assert ds.attn_params() == 187_107_328 and ds.expert_params() == 44_040_192
    assert round(ds.active("both") / 1e9, 2) == 37.55 and round(ds.active("none") / 1e9, 2) == 35.70
    mix = MODELS["mixtral-8x7b"]
    assert mix.expert_params() == 176_160_768 and mix.attn_params() == 41_943_040 and mix.router_params() == 32_768


def test_moe_does_not_change_the_kv_cache():
    """KV follows attention: Mixtral's attention is a Llama-3.1-8B-shaped GQA, so the same 128 KiB per token."""
    assert MODELS["mixtral-8x7b"].kv_bytes_per_token() == MODELS["llama-3.1-8b"].kv_bytes_per_token() == 131_072
    assert MODELS["deepseek-v3"].kv_bytes_per_token() == 70_272               # MLA latent: (512 + 64) x 61 x 2
    assert MODELS["qwen3-30b-a3b"].kv_bytes_per_token() == 98_304
