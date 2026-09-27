"""§1: the lingua franca and adapters as data. Every dialect's sample is the same call spelled four ways."""
import pytest

from gwcore import api
from gwcore.providers import ADAPTERS, PAYLOADS, Clock, FakeProvider, normalize_finish, normalize_stream, normalize_usage

SAME_CALL = {"prompt_tokens": 5000, "completion_tokens": 1550, "total_tokens": 6550, "cached_tokens": 2700, "reasoning_tokens": 1200}


@pytest.mark.parametrize("dialect,field", [("openai", "usage"), ("anthropic", "usage"), ("gemini", "usageMetadata"), ("vllm", "usage")])
def test_four_spellings_of_one_usage(dialect, field):
    assert normalize_usage(dialect, PAYLOADS[dialect]["response"][field]) == SAME_CALL


def test_anthropic_input_excludes_cache_and_gemini_output_excludes_thoughts():
    a = PAYLOADS["anthropic"]["response"]["usage"]
    assert a["input_tokens"] == 2300 and normalize_usage("anthropic", a)["prompt_tokens"] == 2300 + 2700
    g = PAYLOADS["gemini"]["response"]["usageMetadata"]
    assert g["candidatesTokenCount"] == 350 and normalize_usage("gemini", g)["completion_tokens"] == 350 + 1200


def test_finish_reasons_map_to_the_canonical_four():
    assert [normalize_finish("anthropic", r) for r in ("end_turn", "max_tokens", "tool_use", "refusal")] == \
        ["stop", "length", "tool_calls", "content_filter"]
    assert normalize_finish("gemini", "SAFETY") == "content_filter" and normalize_finish("gemini", "MAX_TOKENS") == "length"
    assert ADAPTERS["anthropic"]["auth"][0] == "x-api-key" and ADAPTERS["gemini"]["auth"][0] == "x-goog-api-key"


def test_sse_round_trip_and_done():
    ev = api.chunk("r", "m", 0, content="hi")
    text = api.sse(ev) + ": keep-alive\n\n" + api.sse(api.DONE)
    assert text.endswith("data: [DONE]\n\n") and api.parse_sse(text) == [ev, api.DONE]


def test_tool_call_deltas_accumulate_by_index():
    acc = api.StreamAccumulator()
    for ev in PAYLOADS["openai"]["stream"]:
        acc.add(ev)
    calls = acc.tool_calls()
    assert [(c["id"], c["name"], c["arguments"]) for c in calls] == [
        ("call_a", "get_weather", {"city": "Paris"}), ("call_b", "get_time", {"tz": "Europe/Paris"})]
    assert acc.finish_reason == "tool_calls" and normalize_usage("openai", acc.usage) == SAME_CALL


def test_a_cut_tool_call_is_reported_not_guessed():
    acc = api.StreamAccumulator()
    for ev in PAYLOADS["openai"]["stream"][:4]:
        acc.add(ev)
    assert all(not c["valid_json"] for c in acc.tool_calls()) and acc.usage is None


@pytest.mark.parametrize("dialect", ["anthropic", "gemini"])
def test_other_dialects_stream_to_canonical_chunks(dialect):
    acc = api.StreamAccumulator()
    for ev in normalize_stream(dialect, PAYLOADS[dialect]["stream"]):
        acc.add(ev)
    (call,) = acc.tool_calls()
    assert call["name"] == "get_weather" and call["arguments"] == {"city": "Paris"} and call["valid_json"]
    assert acc.text() == "Checking the weather." and "".join(acc.reasoning) == "The user wants the weather."
    assert acc.finish_reason == "tool_calls" and acc.usage["prompt_tokens"] == 5000
    # Anthropic's stream carries no thinking-token count (only the non-streamed Usage does): it does not normalise
    assert acc.usage["reasoning_tokens"] == (0 if dialect == "anthropic" else 1200)


def test_vllm_mid_stream_error_arrives_inside_a_200():
    acc = api.StreamAccumulator()
    for ev in PAYLOADS["vllm"]["stream"]:
        acc.add(ev)
    assert acc.error["code"] == 500 and acc.text() == "It is sunny"
    assert api.normalize_error({"error": acc.error}, 500)["error"]["code"] == "500"    # int -> string, as OpenAI


def test_fake_provider_timing_on_the_virtual_clock():
    clock = Clock()
    p = FakeProvider("p", clock, ttft=0.3, itl=0.02, output_tokens=10)
    events = list(p.chat(api.chat_request("m", [{"role": "user", "content": "hi"}], stream=True, include_usage=True)).events)
    assert clock.now() == pytest.approx(0.3 + 9 * 0.02)                 # TTFT + ITL x (tokens - 1)
    assert events[-1] == api.DONE and events[-2]["choices"] == [] and events[-2]["usage"]["completion_tokens"] == 10


def test_fake_provider_429_retry_after_and_outages():
    clock = Clock()
    p = FakeProvider("p", clock, rpm=2, outages=[(100, 130, 503)])
    req = api.chat_request("m", [{"role": "user", "content": "hi"}], stream=True)   # not consumed: the clock stays at 0
    assert [p.chat(req).status for _ in range(3)] == [200, 200, 429]
    clock.sleep(20)
    r = p.chat(req)
    assert r.status == 429 and r.headers["retry-after"] == "40"          # the oldest call leaves the window at t=60
    clock.t = 110
    r = p.chat(req)
    assert r.status == 503 and r.headers["retry-after"] == "20"


def test_stream_options_without_stream_is_a_400_as_in_vllm():
    r = FakeProvider("p", Clock()).chat({"model": "m", "messages": [], "stream_options": {"include_usage": True}})
    assert r.status == 400 and "stream=True" in r.body["error"]["message"]
