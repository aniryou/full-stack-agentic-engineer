"""Qwen3-style rendering: reasoning dropped before the last real query, kept in the current tool loop;
the cached-prefix rule (whole blocks, never the last prompt token)."""
from thinklab import templates as T


def test_reasoning_dropped_from_earlier_turns():
    msgs = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "<think>\nsecret\n</think>\n\nA1"},
            {"role": "user", "content": "q2"}]
    out = T.render_qwen3(msgs)
    assert "secret" not in out and "<|im_start|>assistant\nA1<|im_end|>" in out
    assert out.endswith("<|im_start|>assistant\n")


def test_reasoning_kept_inside_tool_loop():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "calling", "reasoning": "plan"},
            {"role": "tool", "content": "result"}]
    out = T.render_qwen3(msgs)
    assert "<think>\nplan\n</think>" in out and "<tool_response>\nresult\n</tool_response>" in out


def test_nonthinking_generation_prompt():
    assert T.render_qwen3([{"role": "user", "content": "q"}], enable_thinking=False).endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_cached_tokens_rules():
    prev = list(range(100))
    assert T.cached_tokens(prev, list(range(40)) + [-1]) == 32
    assert T.cached_tokens(prev, list(range(32))) == 16          # the last prompt token is always computed
    assert T.cached_tokens(prev, [7]) == 0


def test_turn_reuse_dropped_vs_interleaved():
    hist = [{"role": "system", "content": "sys " * 40}, {"role": "user", "content": "q"}]
    out = {"reasoning": "step " * 100, "content": "ans"}
    dropped = T.turn_reuse(hist, out, "next")
    kept = T.turn_reuse(hist, out, "next", keep_all_reasoning=True)
    prompt_n = len(T.tokens(T.render_qwen3(hist)))
    assert dropped["common_prefix"] == prompt_n                  # the match ends right after turn 1's header
    assert dropped["cached"] == prompt_n // 16 * 16
    assert kept["cached"] > dropped["cached"] + 90 and kept["prefilled"] < 32
