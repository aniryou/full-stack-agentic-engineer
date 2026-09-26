"""Bundled samples parse, carry their illustrative label, and the simulated records regenerate exactly."""
import json
from importlib import resources

from thinklab import parsers
from thinklab.thinking import budget as B
from thinklab.thinking import recorded
from thinklab.thinking.client import parse_response

S = resources.files("thinklab.data").joinpath("samples")


def test_json_samples_are_labelled_and_parse():
    for name in ("vllm_chat_qwen3_thinking.json", "vllm_chat_qwen3_nothinking.json", "vllm_chat_truncated.json",
                 "sglang_chat_reasoning_content.json"):
        d = json.loads(S.joinpath(name).read_text())
        assert "illustrative" in d["_thinklab_note"]
        c = parse_response(d)[0]
        assert c.completion_tokens > 0
    t = parse_response(json.loads(S.joinpath("vllm_chat_truncated.json").read_text()))[0]
    assert B.classify(t) == "cut_in_thinking"
    on = parse_response(json.loads(S.joinpath("vllm_chat_qwen3_thinking.json").read_text()))[0]
    assert on.answer == "291" and on.answer_tokens == 29


def test_raw_samples():
    assert parsers.split_deepseek_r1(S.joinpath("deepseek_r1_raw.txt").read_text()).content.strip().endswith("\\boxed{4}.")
    assert parsers.split_harmony(S.joinpath("gptoss_harmony_raw.txt").read_text()).content.endswith("\\boxed{dee}")
    assert S.joinpath("vllm_stream_qwen3.sse").read_text().startswith(": sample output in the documented format (illustrative)")


def test_simulated_records_regenerate_exactly():
    assert recorded.load() == recorded.generate_records()
    assert "illustrative" in S.joinpath(recorded.FILE).read_text().splitlines()[0]
