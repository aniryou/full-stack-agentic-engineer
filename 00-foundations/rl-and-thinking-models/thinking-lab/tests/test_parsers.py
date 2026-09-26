"""Reasoning parsers: the server-side semantics (vLLM's deepseek_r1 / qwen3, gpt-oss Harmony), both field
names, streaming with tags split across chunks, and answer extraction."""
import random

import pytest

from thinklab import parsers as P


@pytest.mark.parametrize("text,reasoning,content", [
    ("<think>a</think>b", "a", "b"),
    ("a</think>b", "a", "b"),                       # R1: the template opened <think> in the prompt
    ("still thinking", "still thinking", None),     # cut off inside the thinking
    ("x</think>", "x", None),                       # nothing after the end tag -> content None
    ("<think>\n\n</think>\n\nhi", "\n\n", "\n\nhi"),
])
def test_deepseek_r1_semantics(text, reasoning, content):
    s = P.split_deepseek_r1(text)
    assert (s.reasoning, s.content) == (reasoning, content)


def test_qwen3_thinking_off_is_all_content():
    assert P.split_qwen3("<think>a</think>b", enable_thinking=False) == P.Split(None, "<think>a</think>b")
    assert P.split_qwen3("a</think>b").reasoning == "a"


def test_harmony_channels():
    s = P.split_harmony("<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>42<|return|>")
    assert (s.reasoning, s.content) == ("think", "42")
    assert P.split("<|channel|>final<|message|>x<|return|>", "openai_gptoss").reasoning is None


def test_reasoning_field_names():
    assert P.reasoning_of({"reasoning": "r"}) == "r"
    assert P.reasoning_of({"reasoning_content": "r2", "reasoning": None}) == "r2"
    assert P.reasoning_of({"reasoning_content": ""}) is None
    class Msg:
        reasoning = "obj"
    assert P.reasoning_of(Msg()) == "obj"


def _stream(t, chunks):
    sp = P.StreamSplitter(starts_in_reasoning=True)
    for c in chunks:
        sp.feed(c)
    return sp.close()


def test_stream_splitter_is_chunking_invariant_and_matches_batch_parse():
    rng = random.Random(0)
    for t in ["<think>abc</think>\n\nanswer", "abc</think>tail", "no tags at all", "a</think>b</think>c", "x<th"]:
        whole = _stream(t, [t])
        for _ in range(100):
            cuts = sorted(rng.sample(range(1, len(t)), k=min(len(t) - 1, rng.randrange(0, 6))))
            assert _stream(t, [t[i:j] for i, j in zip([0] + cuts, cuts + [len(t)])]) == whole
    for t in ["<think>abc</think>\n\nanswer", "abc</think>tail", "no tags at all"]:
        ref = P.split_deepseek_r1(t)
        assert _stream(t, [t]) == P.Split(ref.reasoning, ref.content)
    assert _stream("a</think>b</think>c", ["a</think>b</think>c"]).content == "bc"   # duplicate end tag absorbed


def test_stream_splitter_holds_back_partial_tags():
    sp = P.StreamSplitter()
    assert sp.feed("abc</th") == ("abc", "")
    assert sp.feed("ink>ok") == ("", "ok")


@pytest.mark.parametrize("content,expected", [
    ("so \\boxed{ 42 }", "42"), ("Answer: Tuesday.", "tuesday"), ("\\boxed{1} then \\boxed{2}", "2"),
    ("no answer here", None), (None, None), ("The answer is **391**", "391")])
def test_extract_answer(content, expected):
    assert P.extract_answer(content) == expected


def test_unknown_parser():
    with pytest.raises(ValueError):
        P.split("x", "nope")
