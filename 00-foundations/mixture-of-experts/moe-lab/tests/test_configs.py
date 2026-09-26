"""Parameter counts from configs: every model reproduces its published total/active, under a named convention."""
import pytest

from moelab import configs as C

G = C.get


@pytest.mark.parametrize("key,total,active_both,active_head", [
    ("mixtral-8x7b", 46.70, 12.88, 12.75),          # roofline.llm: 46,702,526,464 / 12,879,659,008
    ("qwen3-30b-a3b", 30.53, 3.35, 3.04),
    ("olmoe-1b-7b", 6.92, 1.28, 1.18),               # OLMoE README: 6.9B / 1.3B
    ("qwen1.5-moe-a2.7b", 14.32, 2.69, 2.38),
    ("gpt-oss-20b", 20.91, 4.19, 3.61),              # OpenAI counts the LM head only: 21B / 3.6B
    ("gpt-oss-120b", 116.83, 5.71, 5.13),            # 116.8B / 5.1B
    ("granite-1b-a400m", 1.33, 0.43, 0.43),
    ("granite-3b-a800m", 3.30, 0.88, 0.88),
    ("llama-3.1-8b", 8.03, 8.03, 7.50),
])
def test_totals_and_active_under_both_conventions(key, total, active_both, active_head):
    m = G(key)
    assert round(m.total_params() / 1e9, 2) == total
    assert round(m.active_params("both") / 1e9, 2) == active_both
    assert round(m.active_params("head") / 1e9, 2) == active_head
    assert m.active_params("matmul") < m.active_params("head") <= m.active_params("both")


def test_exact_roofline_numbers():
    """The same integers as layer 01's roofline.llm (fact sheet §1)."""
    assert G("mixtral-8x7b").total_params() == 46_702_526_464
    assert G("mixtral-8x7b").active_params() == 12_879_659_008
    assert G("qwen3-30b-a3b").total_params() == 30_531_911_680
    assert G("qwen3-30b-a3b").active_params() == 3_352_821_760
    assert G("llama-3.1-8b").total_params() == 8_029_995_008


def test_per_layer_pieces_worked_in_the_fact_sheet():
    assert G("mixtral-8x7b").expert_params() == 176_160_768
    assert G("mixtral-8x7b").attn_params() == 41_943_040
    assert G("qwen3-30b-a3b").expert_params() == 4_718_592
    assert G("gpt-oss-20b").expert_params() == 24_891_840          # mlp1 [2I, d] + bias, mlp2 [d, I] + bias
    assert G("gpt-oss-20b").attn_params() == 26_550_144            # qkv + bias + o + bias + sinks
    q = G("qwen1.5-moe-a2.7b")
    assert q.shared_params() == 4 * q.expert_params() + q.d_model  # the shared expert = 4 routed ones + its gate


@pytest.mark.parametrize("key,kv", [("mixtral-8x7b", 131_072), ("qwen3-30b-a3b", 98_304), ("olmoe-1b-7b", 131_072),
                                    ("qwen1.5-moe-a2.7b", 196_608), ("granite-1b-a400m", 49_152)])
def test_kv_bytes_per_token_do_not_depend_on_experts(key, kv):
    assert G(key).kv_bytes_per_token() == kv


@pytest.mark.parametrize("key,scheme,gb", [("gpt-oss-20b", "mxfp4", 13.8), ("gpt-oss-120b", "mxfp4", 65.2),
                                           ("qwen3-30b-a3b", "int4", 17.1), ("qwen1.5-moe-a2.7b", "int4", 8.5),
                                           ("qwen3-30b-a3b", "fp8", 31.2), ("mixtral-8x7b", "bf16", 93.4),
                                           ("olmoe-1b-7b", "fp16", 13.8)])
def test_checkpoint_sizes(key, scheme, gb):
    assert round(C.weight_bytes(G(key), scheme) / 1e9, 1) == gb


def test_expert_bytes_are_most_of_an_moe():
    o = G("olmoe-1b-7b")
    assert 0.9 < C.expert_bytes(o) / C.weight_bytes(o) < 0.95
    assert C.expert_bytes(G("llama-3.1-8b")) == 0


def test_unknown_names_fail_clearly():
    with pytest.raises(KeyError, match="known"):
        G("nope")
    with pytest.raises(KeyError, match="known"):
        C.gpu("V100")
