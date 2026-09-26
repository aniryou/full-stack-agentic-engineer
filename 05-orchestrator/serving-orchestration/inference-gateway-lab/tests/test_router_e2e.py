"""The router as a real HTTP proxy: streaming fidelity, affinity, admission, errors, cleanup."""
import asyncio
import json
import time
import urllib.error
import urllib.request

import aiohttp
import pytest
from aiohttp import web

from igwlab.promtext import Families
from igwlab.router import Endpoint, Router, RouterSettings
from igwlab.stack import LocalStack

CHUNKS = [b'data: {"choices":[{"delta":{"content":"caf', "é".encode()[:1], "é".encode()[1:] + b'"}}]}\n',
          b"\ndata: {\"choices\":[{\"delta\":{\"content\":\" ok\"}}]}\n\n", b"data: [DONE]\n\n"]


async def _odd_upstream():
    """An upstream that streams CHUNKS with a pause between them, splitting lines and a UTF-8 char."""
    async def handle(request):
        await request.read()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "X-Upstream": "odd"})
        await resp.prepare(request)
        for i, c in enumerate(CHUNKS):
            await resp.write(c)
            await asyncio.sleep(0.25 if i == 0 else 0.01)
        await resp.write_eof()
        return resp

    async def metrics(request):
        return web.Response(text="vllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0.0\n")
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    app.router.add_get("/metrics", metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


def test_streaming_is_forwarded_byte_for_byte_and_without_buffering():
    async def go():
        runner, url = await _odd_upstream()
        router = Router("round-robin", [Endpoint("odd", url)])
        rurl = await router.start()
        try:
            async with aiohttp.ClientSession() as http:
                t0 = time.perf_counter()
                async with http.post(rurl + "/v1/chat/completions", json={"model": "m", "messages": [], "stream": True}) as r:
                    first = await r.content.readany()
                    t_first = time.perf_counter() - t0
                    rest = await r.read()
                    return r.status, dict(r.headers), first + rest, t_first
        finally:
            await router.stop()
            await runner.cleanup()
    status, headers, body, t_first = asyncio.run(go())
    assert status == 200 and body == b"".join(CHUNKS)                 # identical bytes, split UTF-8 intact
    assert headers["X-Upstream"] == "odd" and headers["Content-Type"] == "text/event-stream"
    assert headers["x-gateway-destination-endpoint"] == "odd"
    assert t_first < 0.2                                               # first chunk not held for the 250 ms pause


def _post(url, body, headers=None):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read())


SYSTEM = "You are a careful agent. Use the tools, check every result, report what you did. " * 60


def test_second_turn_follows_the_cached_prefix():
    with LocalStack(3, "default-weighted") as s:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "task one"}]
        st, h1, r1 = _post(s.router_url, {"model": "lab/llm", "messages": msgs, "max_tokens": 4})
        msgs += [{"role": "assistant", "content": r1["choices"][0]["message"]["content"]},
                 {"role": "user", "content": "tool output " * 50}]
        st2, h2, r2 = _post(s.router_url, {"model": "lab/llm", "messages": msgs, "max_tokens": 4})
        assert st == st2 == 200
        assert h1["x-gateway-destination-endpoint"] == h2["x-gateway-destination-endpoint"] == h2["x-inference-pod"]
        assert r1["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] >= 0.8 * r1["usage"]["prompt_tokens"]
        d = s.last_decision()
        assert d.scores["prefix-cache-scorer"][d.endpoint] > 0.5


def test_round_robin_preset_spreads_evenly():
    with LocalStack(3, "round-robin") as s:
        for i in range(6):
            _post(s.router_url, {"model": "lab/llm", "messages": [{"role": "user", "content": f"q{i}"}], "max_tokens": 2})
        assert s.routed() == {"a": 2, "b": 2, "c": 2}


def test_sheddable_requests_are_dropped_when_saturated_and_others_pass():
    # a zero staleness threshold makes every endpoint read "stale" -> utilization detector says 1.0
    settings = RouterSettings(objectives={"batch": -10, "premium": 100}, metrics_staleness_s=0.0)
    with LocalStack(2, "default-weighted", settings=settings) as s:
        body = {"model": "lab/llm", "messages": [{"role": "user", "content": "x"}], "max_tokens": 2}
        st, h, err = _post(s.router_url, body, {"x-llm-d-inference-objective": "batch"})
        assert st == 429 and h["x-llm-d-request-dropped-reason"] == "rejected-saturated"
        assert _post(s.router_url, body, {"x-llm-d-inference-objective": "premium"})[0] == 200
        assert _post(s.router_url, body)[0] == 200                     # no objective -> priority 0, never shed


def test_no_endpoints_is_503_and_bad_json_is_400_and_unknown_model_passes_through():
    with LocalStack(0, "default-weighted", names=[]) as s:
        st, h, err = _post(s.router_url, {"model": "lab/llm", "messages": []})
        assert st == 503 and h["x-llm-d-request-dropped-reason"] == "rejected-no-endpoints"
    with LocalStack(1, "round-robin") as s:
        req = urllib.request.Request(s.router_url + "/v1/chat/completions", data=b"{not json", method="POST")
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req, timeout=30)
        assert ei.value.code == 400
        assert _post(s.router_url, {"model": "no-such-model", "messages": []})[0] == 404


def test_metrics_endpoints_speak_prometheus_with_vllm_names():
    with LocalStack(2, "default-weighted") as s:
        _post(s.router_url, {"model": "lab/llm", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3})
        rm = Families.from_text(s.router_metrics())
        assert rm.sum("igw_request_total") == 1.0 and rm.value("igw_ready_endpoints") == 2.0
        bm = Families.from_text(next(iter(s.backend_metrics().values())))
        for name in ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
                     "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:prompt_tokens_total",
                     "vllm:generation_tokens_total", "vllm:time_to_first_token_seconds_bucket",
                     "vllm:inter_token_latency_seconds_bucket", "vllm:cache_config_info"):
            assert bm.has(name), name


def test_client_disconnect_mid_stream_releases_router_and_backend_state():
    with LocalStack(1, "active-requests") as s:
        async def go():
            async with aiohttp.ClientSession() as http:
                body = {"model": "lab/llm", "messages": [{"role": "user", "content": "long"}], "max_tokens": 400, "stream": True}
                async with http.post(s.router_url + "/v1/chat/completions", json=body) as r:
                    await r.content.readany()          # first chunk, then hang up
        asyncio.run(go())
        deadline = time.time() + 5
        while time.time() < deadline:
            state = s.call(lambda: (s.router.ds.get("a").inflight_requests, s.router.ds.get("a").inflight_tokens,
                                    len(s.backends[0].engine.running)))
            if state == (0, 0, 0):
                break
            time.sleep(0.05)
        assert state == (0, 0, 0)


def test_a_failed_start_stops_what_it_started():
    import threading
    with pytest.raises(Exception):
        LocalStack(3, "no-such-preset").start()           # backends start before the config is rejected
    time.sleep(0.1)
    assert not any(t.name == "igwlab-stack" for t in threading.enumerate())


def test_router_in_front_of_already_running_backends():
    """The T1 shape: the in-process router in front of servers it did not start (here: fakes)."""
    with LocalStack(2, "round-robin") as servers:
        with LocalStack(config="round-robin", backends=servers.backend_urls) as s:
            assert s.backends == [] and s.names == ["a", "b"]
            for i in range(4):
                assert _post(s.router_url, {"model": "lab/llm", "messages": [{"role": "user", "content": f"q{i}"}],
                                            "max_tokens": 2})[0] == 200
            assert s.routed() == {"a": 2, "b": 2}
            m = s.backend_metrics()                                   # fetched over HTTP
            assert set(m) == {"a", "b"} and all("vllm:num_requests_running" in v for v in m.values())
