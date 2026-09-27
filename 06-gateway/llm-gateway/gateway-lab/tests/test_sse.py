"""SSE parsing and stream accumulation: split bytes, tool-call deltas by index, usage, errors inside a 200."""
import json

from gwlab import sse


def test_parser_handles_splits_crlf_comments_and_events():
    raw = ('event: message_start\r\ndata: {"a": 1}\r\n\r\n: keep-alive\n\ndata: {"t": "caf' + "é" + '"}\n\n'
           'data: line1\ndata: line2\n\ndata: [DONE]\n\n').encode()
    p, events = sse.SSEParser(), []
    for i in range(0, len(raw), 3):                      # three bytes at a time: splits lines and the UTF-8 "é"
        events += p.feed(raw[i:i + 3])
    assert [e.event for e in events] == ["message_start", None, None, None]
    assert events[0].json() == {"a": 1} and events[1].json() == {"t": "café"}
    assert events[2].data == "line1\nline2" and events[3].data == "[DONE]"


def test_encode_roundtrip():
    chunks, done = sse.parse_stream(sse.encode({"x": 1}) + sse.encode({"y": 2}, event="e") + sse.encode_done())
    assert chunks == [{"x": 1}, {"y": 2}] and done


def _chunk(delta=None, finish=None, **kw):
    return {"id": "c1", "model": "m", "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}], **kw}


def test_accumulator_rebuilds_tool_calls_by_index():
    acc = sse.StreamAccumulator()
    acc.add(_chunk({"role": "assistant", "content": ""}))
    acc.add(_chunk({"tool_calls": [{"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}))
    acc.add(_chunk({"tool_calls": [{"index": 1, "id": "call_b", "type": "function", "function": {"name": "get_time", "arguments": "{\"tz\""}}]}))
    acc.add(_chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\": \"Pa"}}]}))
    acc.add(_chunk({"tool_calls": [{"index": 0, "function": {"arguments": "ris\"}"}}]}))
    acc.add(_chunk({"tool_calls": [{"index": 1, "function": {"arguments": ": \"UTC\"}"}}]}))
    acc.add(_chunk({}, "tool_calls"))
    calls = acc.completion()["choices"][0]["message"]["tool_calls"]
    assert [c["id"] for c in calls] == ["call_a", "call_b"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert json.loads(calls[1]["function"]["arguments"]) == {"tz": "UTC"}
    assert acc.finish_reason == "tool_calls"


def test_accumulator_keeps_the_last_usage_and_counts_text_chunks():
    acc = sse.StreamAccumulator()
    for i, word in enumerate(["a", " b", " c"]):         # continuous usage: a running total on every chunk
        acc.add(_chunk({"content": word}, usage={"prompt_tokens": 5, "completion_tokens": i + 1}))
    acc.add({"id": "c1", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}})
    assert acc.usage == {"prompt_tokens": 5, "completion_tokens": 3}       # not 1 + 2 + 3 + 3
    assert acc.content == "a b c" and acc.token_chunks == 3 and acc.output_estimate() == 3


def test_accumulator_records_a_midstream_error_and_reasoning():
    acc = sse.StreamAccumulator()
    acc.add(_chunk({"reasoning": "think "}))
    acc.add(_chunk({"reasoning_content": "more"}))
    acc.add({"error": {"message": "EngineCore encountered an issue", "code": 500}})
    assert acc.reasoning == "think more" and acc.error["code"] == 500
