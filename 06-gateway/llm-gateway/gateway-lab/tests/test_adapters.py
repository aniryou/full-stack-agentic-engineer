"""The adapter table: usage, finish reasons and errors normalised; requests and Anthropic streams translated."""
import json

import pytest

from gwlab.gateway import adapters as A
from gwlab.gateway.metering import PRICES, cost_usd


def test_usage_normalised_from_three_dialects_on_the_same_call():
    oa = A.normalise_usage("openai", A.payload("openai_usage")["usage"])
    an = A.normalise_usage("anthropic", A.payload("anthropic_message")["usage"])
    ge = A.normalise_usage("gemini", A.payload("gemini_response")["usageMetadata"])
    # the primer's §1.5 call in three spellings: Anthropic's input_tokens (2,300) excludes the 2,700 cache reads;
    # Gemini's 1,200 thoughts sit outside its 350 candidates
    for u in (oa, an, ge):
        assert (u.prompt_tokens, u.cached_tokens, u.completion_tokens, u.reasoning_tokens) == (5000, 2700, 1550, 1200)


def test_the_same_call_priced_per_row():
    an = A.normalise_usage("anthropic", A.payload("anthropic_message")["usage"])
    visible = A.Usage(5000, 350, 2700)                     # the scaling primer's §3.4 call: the same, without thinking
    assert cost_usd(visible, PRICES["claude-haiku-4-5"]) == pytest.approx(0.00432)
    assert cost_usd(visible, PRICES["gemini-3.5-flash"]) == pytest.approx(0.007005)
    assert cost_usd(visible, PRICES["gpt-5.4-mini"]) == pytest.approx(0.0035025)
    assert cost_usd(an, PRICES["claude-haiku-4-5"]) == pytest.approx(0.00432 + 1200 * 5.00 / 1e6)     # thinking = output
    assert cost_usd(an, PRICES["gemini-3.5-flash"]) == pytest.approx(0.017805)                       # the core's number
    naive = A.Usage(2300, 1550, 2700)      # feeding Anthropic's input_tokens straight in double-discounts the cache
    assert cost_usd(naive, PRICES["claude-haiku-4-5"]) < cost_usd(an, PRICES["claude-haiku-4-5"])


def test_finish_reasons_map_to_openai_values():
    assert A.normalise_finish("anthropic", "tool_use") == "tool_calls"
    assert A.normalise_finish("anthropic", "max_tokens") == "length"
    assert A.normalise_finish("anthropic", "refusal") == "content_filter"
    assert A.normalise_finish("gemini", "SAFETY") == "content_filter"
    assert A.normalise_finish("gemini", "MAX_TOKENS") == "length"
    for dialect, reason in (("anthropic", "pause_turn"), ("gemini", "MALFORMED_FUNCTION_CALL"), ("gemini", "OTHER")):
        assert A.normalise_finish(dialect, reason) == reason != "stop"          # never guessed to be complete


def test_errors_get_string_codes():
    vllm = A.normalise_error("openai", 400, {"error": {"message": "This model's maximum context length is 4096 tokens.",
                                                       "type": "BadRequestError", "code": 400}})
    assert vllm.code == "context_length_exceeded" and isinstance(vllm.code, str)
    oa = A.normalise_error("openai", 429, {"error": {"message": "slow down", "type": "rate_limit_error", "code": "slow_down"}})
    assert oa.code == "slow_down"
    an = A.normalise_error("anthropic", 529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    assert (an.status, an.code, an.type) == (529, "overloaded", "overloaded_error")
    assert A.normalise_error("openai", 503, "not json").code == "service_unavailable"
    assert A.normalise_error("openai", 400, {"error": {"message": "x", "code": "content_policy_violation"}}).code == "content_filter"


def test_openai_upstream_body():
    body = {"model": "chat", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5000,
            "stream_options": {"include_usage": False}}
    non = A.to_upstream("openai", body, "fast-1", stream=False, output_cap=2048)
    assert "stream_options" not in non and non["stream"] is False            # vLLM: 400 otherwise
    assert non["max_completion_tokens"] == 2048 and "max_tokens" not in non and non["model"] == "fast-1"
    st = A.to_upstream("openai", body, "fast-1", stream=True, cache_salt="abc")
    assert st["stream_options"] == {"include_usage": True} and st["cache_salt"] == "abc"
    routed = A.to_upstream("openai", {**body, "metadata": {"cache_class": "account", "user": "alice"}}, "fast-1", stream=False)
    assert "metadata" not in routed                                          # route metadata never leaves the gateway
    assert body["model"] == "chat"                                          # the client's body is untouched


def test_anthropic_upstream_body():
    body = {"model": "chat", "temperature": 0, "stop": "END",
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "weather?"},
                         {"role": "assistant", "content": None, "tool_calls": [
                             {"id": "t1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]},
                         {"role": "tool", "tool_call_id": "t1", "content": "18C"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
            "tool_choice": "required"}
    up = A.to_upstream("anthropic", body, "claude-haiku-4-5", stream=True)
    assert up["system"] == "Be brief." and up["max_tokens"] == A.ANTHROPIC_DEFAULT_MAX_TOKENS
    assert [m["role"] for m in up["messages"]] == ["user", "assistant", "user"]
    assert up["messages"][1]["content"][0] == {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}}
    assert up["messages"][2]["content"][0]["type"] == "tool_result" and up["tool_choice"] == {"type": "any"}
    assert up["tools"][0]["input_schema"] == {"type": "object"} and up["stop_sequences"] == ["END"]
    with pytest.raises(ValueError):
        A.to_upstream("gemini", body, "g", stream=False)


def test_anthropic_stream_translates_to_chat_chunks():
    tr = A.AnthropicStreamTranslator("claude-haiku-4-5")
    chunks = [c for name, data in A.payload("anthropic_stream")["events"] for c in tr.feed(name, data)]
    from gwlab.sse import StreamAccumulator
    acc = StreamAccumulator()
    for c in chunks:
        acc.add(c)
    assert tr.done and acc.content == "Checking the weather."
    assert acc.calls[0]["function"]["name"] == "get_weather"
    assert json.loads(acc.calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert acc.finish_reason == "tool_calls" and (tr.usage.prompt_tokens, tr.usage.completion_tokens) == (40, 27)


def test_anthropic_message_to_completion():
    comp, usage = A.completion_from_anthropic(A.payload("anthropic_message"), "claude-haiku-4-5")
    msg = comp["choices"][0]["message"]
    assert msg["content"].startswith("Your invoice") and msg["tool_calls"][0]["function"]["name"] == "get_invoice"
    assert comp["choices"][0]["finish_reason"] == "tool_calls" and comp["usage"]["prompt_tokens"] == 5000
    assert comp["usage"]["prompt_tokens_details"]["cached_tokens"] == 2700
    assert comp["usage"]["completion_tokens_details"]["reasoning_tokens"] == 1200


def test_payloads_are_labelled_illustrative():
    for name in ("openai_usage", "anthropic_message", "anthropic_stream", "gemini_response", "vllm_midstream_error"):
        assert "illustrative" in A.payload(name)["_label"]
