"""The fake vLLM over HTTP: OpenAI streaming shape, usage, vLLM metric names, cache-hit TTFT, errors."""
import json
import urllib.error
import urllib.request

import pytest

from servelab import metrics as M
from servelab.bench import Lengths, Request, random_requests, run_open_loop, run_sync
from servelab.bench.client import stream_request
from servelab.bench.runner import _session
from servelab.fake_engine import EngineConfig, tiny_profile
from servelab.fakeserver import FakeServer
from servelab.textgen import synthetic_text


@pytest.fixture(scope="module")
def server():
    s = FakeServer(tiny_profile(num_blocks=1024, max_model_len=4096), EngineConfig(max_num_batched_tokens=512),
                   model="tiny-model")
    url = s.start()
    yield url
    s.stop()


def send(url, req):
    async def go():
        async with _session(60) as http:
            return await stream_request(http, url, "tiny-model", req)
    return run_sync(go())


def test_streaming_completion_timings_and_usage(server):
    r = send(server, Request(prompt=synthetic_text(40, 1), max_tokens=12, prompt_tokens=40))
    assert r.ok and r.output_tokens == 12 and r.prompt_tokens == 40
    assert r.ttft > 0 and len(r.itl) == 11 and r.latency > r.ttft
    assert len(r.text.split()) == 12                                    # one toy token per chunk


def test_chat_streaming_and_non_streaming(server):
    msgs = [{"role": "system", "content": synthetic_text(30, 2)}, {"role": "user", "content": "hello there"}]
    r = send(server, Request(messages=msgs, max_tokens=5))
    assert r.ok and r.output_tokens == 5 and len(r.itl) == 4              # role chunk + 5 token chunks
    body = json.dumps({"model": "tiny-model", "messages": msgs, "max_tokens": 3}).encode()
    req = urllib.request.Request(server + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        out = json.loads(resp.read())
        assert resp.headers["x-servelab-simulated"] == "true"
    assert out["usage"]["completion_tokens"] == 3 and out["choices"][0]["finish_reason"] == "length"


def test_prefix_cache_hit_is_reported_and_faster(server):
    text = synthetic_text(1600, 7)
    cold = send(server, Request(prompt=text, max_tokens=2))
    warm = send(server, Request(prompt=text, max_tokens=2))
    assert cold.cached_tokens == 0 and warm.cached_tokens == (1600 - 1) // 4 * 4
    assert warm.ttft < cold.ttft


def test_metrics_page_has_vllm_names(server):
    send(server, Request(prompt=synthetic_text(20, 3), max_tokens=3))
    text = urllib.request.urlopen(server + "/metrics", timeout=10).read().decode()
    for name in (M.RUNNING, M.WAITING, M.KV_USAGE, M.PREFIX_QUERIES + "_total", M.PREFIX_HITS + "_total",
                 M.PREEMPTIONS + "_total", M.PROMPT_TOKENS + "_total", M.GENERATION_TOKENS + "_total",
                 M.TTFT + "_bucket", M.ITL + "_bucket", M.TPOT + "_bucket", M.QUEUE + "_bucket",
                 M.PREFILL_TIME + "_bucket", M.DECODE_TIME + "_bucket", M.REQUEST_SUCCESS + "_total",
                 M.ITERATION_TOKENS + "_bucket", M.CACHE_CONFIG_INFO):
        assert name in text, name
    snap = M.snapshot(M.parse(text))
    assert snap.requests_finished >= 1 and snap.generation_tokens >= 3


def test_too_long_request_is_rejected_like_vllm(server):
    body = json.dumps({"model": "tiny-model", "prompt": synthetic_text(4090, 1), "max_tokens": 64}).encode()
    req = urllib.request.Request(server + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req, timeout=10)
    assert err.value.code == 400 and "maximum context length" in err.value.read().decode()


def test_open_loop_run_is_labelled_simulated(server):
    run = run_open_loop(server, random_requests(12, Lengths.fixed(64), Lengths.fixed(8), seed=5), rate=40.0)
    s = run.summary()
    assert run.simulated and "SIMULATED" in run.source
    assert s.completed == 12 and s.total_output == 96 and s.request_throughput > 0
