"""The fake providers behave like their dialects, including the parts that trip gateways up."""
import json

from gwlab import client, promtext, sse

from .conftest import make_stack


def _post(url, path, body, headers):
    return client.request("POST", url + path, body, headers)


def test_openai_fake_quirks(stack):
    url = stack.fake_url("acme")
    auth = {"Authorization": "Bearer acme-key-1"}
    msg = [{"role": "user", "content": "hi"}]
    assert _post(url, "/v1/chat/completions", {"model": "fast-1", "messages": msg}, {"Authorization": "Bearer nope"}).status == 401
    r = _post(url, "/v1/chat/completions", {"model": "fast-1", "messages": msg, "stream_options": {"include_usage": True}}, auth)
    assert r.status == 400 and "stream=True" in r.json["error"]["message"] and r.json["error"]["code"] == 400   # vLLM: int code
    assert _post(url, "/v1/chat/completions", {"model": "fast-1", "messages": msg, "cache_salt": "a/b"}, auth).status == 400
    assert _post(url, "/v1/chat/completions", {"model": "nope", "messages": msg}, auth).status == 404
    r = _post(url, "/v1/chat/completions", {"model": "fast-1", "messages": msg, "stream": True,
                                            "stream_options": {"include_usage": True}}, auth)
    chunks, done = sse.parse_stream(r.raw)
    assert done and chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 12
    assert all(c["usage"] is None for c in chunks[:-1])          # OpenAI: usage: null on every other chunk
    assert r.header("x-gwlab-simulated") == "true"


def test_prefix_cache_counts_full_blocks_and_honours_the_salt(stack):
    url, auth = stack.fake_url("acme"), {"Authorization": "Bearer acme-key-1"}
    system = "You are a careful assistant. " * 12                      # 61 words + markers
    body = {"model": "fast-1", "messages": [{"role": "system", "content": system}, {"role": "user", "content": "q1"}]}
    first = _post(url, "/v1/chat/completions", body, auth).json["usage"]
    again = _post(url, "/v1/chat/completions", body, auth).json["usage"]
    n = first["prompt_tokens"]
    assert first["prompt_tokens_details"]["cached_tokens"] == 0
    assert again["prompt_tokens_details"]["cached_tokens"] == (n - 1) // 16 * 16         # the last token is computed
    salted = _post(url, "/v1/chat/completions", {**body, "cache_salt": "tenant-b-salt"}, auth).json["usage"]
    assert salted["prompt_tokens_details"]["cached_tokens"] == 0                          # the salt isolates


def test_anthropic_fake_speaks_messages():
    with make_stack() as s:
        url = s.fake_url("bolt")
        base = {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]}
        assert _post(url, "/v1/messages", {**base, "max_tokens": 50}, {"x-api-key": "bolt-key-1"}).status == 400  # no version
        hdr = {"x-api-key": "bolt-key-1", "anthropic-version": "2023-06-01"}
        r = _post(url, "/v1/messages", base, hdr)
        assert r.status == 400 and r.json["type"] == "error"                            # max_tokens is required
        r = _post(url, "/v1/messages", {**base, "max_tokens": 50, "stream": True}, hdr)
        p = sse.SSEParser()
        names = [e.event for e in p.feed(r.raw)]
        assert names[0] == "message_start" and names[-2:] == ["message_delta", "message_stop"] and "content_block_delta" in names
        last = [json.loads(e.data) for e in sse.SSEParser().feed(r.raw) if e.event == "message_delta"][0]
        assert last["usage"]["output_tokens"] == 12


def test_limits_faults_and_metrics():
    from gwlab.fakes import FakeSpec
    fakes = {"acme": FakeSpec(name="acme", rpm=2, minute_s=60, ttft_s=0.005, itl_s=0.001, output_tokens=4),
             "bolt": FakeSpec.anthropic("bolt", ttft_s=0.005, itl_s=0.001, output_tokens=4)}
    with make_stack(fakes=fakes) as s:
        url, auth = s.fake_url("acme"), {"Authorization": "Bearer acme-key-1"}
        body = {"model": "fast-1", "messages": [{"role": "user", "content": "x"}]}
        assert [_post(url, "/v1/chat/completions", body, auth).status for _ in range(3)] == [200, 200, 429]
        r = _post(url, "/v1/chat/completions", body, auth)
        assert r.status == 429 and int(r.header("Retry-After")) >= 1 and r.header("x-ratelimit-limit-requests") == "2"
        m = promtext.parse(s.provider_metrics("acme"))
        assert promtext.total(m, "vllm:generation_tokens_total") == 8 and promtext.total(m, "fake_http_responses_total", status="429") == 2
        url_b, hb = s.fake_url("bolt"), {"x-api-key": "bolt-key-1", "anthropic-version": "2023-06-01"}
        s.fault("bolt", "503", count=1)
        r = _post(url_b, "/v1/messages", {"model": "claude-haiku-4-5", "max_tokens": 8, "messages": body["messages"]}, hb)
        assert r.status == 529 and r.json["error"]["type"] == "overloaded_error"
        s.fault("bolt", "midstream", count=1, after_chunks=2)
        r = _post(url_b, "/v1/messages", {"model": "claude-haiku-4-5", "max_tokens": 8, "stream": True,
                                          "messages": body["messages"]}, hb)
        events = sse.SSEParser().feed(r.raw)
        assert r.status == 200 and events[-1].event == "error"                         # an error inside the 200
