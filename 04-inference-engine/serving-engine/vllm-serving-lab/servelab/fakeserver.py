"""fakeserver.py — a fake vLLM: OpenAI-compatible streaming API + vLLM-named ``/metrics``.

One idea: to learn to *measure* an engine you need something that behaves like one. This server
runs :class:`servelab.fake_engine.FakeEngine` in real time — every step sleeps for the time the
roofline model gives — behind the same endpoints as ``vllm serve``: ``/v1/completions`` and
``/v1/chat/completions`` (SSE streaming, ``stream_options.include_usage``), ``/v1/models``,
``/health`` and a Prometheus ``/metrics`` with vLLM's metric names and histogram buckets. The
benchmark, metrics parser, sizing and tuner in this lab therefore run unchanged against it (T0)
and against a real vLLM (T1/T3).

Everything it reports is **simulated**: ``/version`` says so and every response carries the
header ``x-servelab-simulated: true``. The upstream tool with the same purpose, used by llm-d for
routing experiments, is ``llm-d-inference-sim`` (github.com/llm-d/llm-d-inference-sim).

    python -m servelab fake --profile t4-qwen2.5-0.5b --port 8000
    with FakeServer("t4-qwen2.5-0.5b") as url: ...            # in-process, background thread
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import traceback
import uuid

from aiohttp import web
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from . import metrics as M
from .fake_engine import EngineConfig, EngineProfile, FakeEngine, profile as named_profile
from .textgen import chat_tokens, tokenize

# Histogram bucket edges, copied from vllm/v1/metrics/buckets.py (vLLM v0.30.0; same on main, Sep 2026).
BUCKETS = {
    "request_latency": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                        120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0],
    "time_to_first_token": [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0,
                            7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0],
    "inter_token_latency": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0,
                            7.5, 10.0, 20.0, 40.0, 80.0],
    "iteration_tokens": [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
}


def request_token_buckets(max_model_len: int) -> list:
    """vLLM's 1-2-5 series capped at max_model_len (``build_1_2_5_buckets``)."""
    out, e = [], 0
    while True:
        for m in (1, 2, 5):
            v = m * 10**e
            if v > max_model_len:
                return out
            out.append(v)
        e += 1


class _Metrics:
    """The subset of vLLM's Prometheus metrics this lab reads, registered per server."""

    def __init__(self, model: str, engine: FakeEngine):
        r = self.registry = CollectorRegistry()
        lab, lv = ["model_name", "engine"], (model, "0")

        def g(name, doc):
            return Gauge(name, doc, lab, registry=r).labels(*lv)

        def c(name, doc):
            return Counter(name, doc, lab, registry=r).labels(*lv)

        def h(name, doc, buckets):
            return Histogram(name, doc, lab, buckets=buckets, registry=r).labels(*lv)

        self.running = g(M.RUNNING, "Number of requests in model execution batches.")
        self.waiting = g(M.WAITING, "Number of requests waiting to be processed.")
        self.kv_usage = g(M.KV_USAGE, "KV-cache usage. 1 means 100 percent usage.")
        self.prefix_queries = c(M.PREFIX_QUERIES, "Prefix cache queries, in terms of number of queried tokens.")
        self.prefix_hits = c(M.PREFIX_HITS, "Prefix cache hits, in terms of number of cached tokens.")
        self.preemptions = c(M.PREEMPTIONS, "Cumulative number of preemption from the engine.")
        self.prompt_tokens = c(M.PROMPT_TOKENS, "Number of prefill tokens processed.")
        self.generation_tokens = c(M.GENERATION_TOKENS, "Number of generation tokens processed.")
        success = Counter(M.REQUEST_SUCCESS, "Count of successfully processed requests.", lab + ["finished_reason"],
                          registry=r)
        self.success = {k: success.labels(model, "0", k) for k in ("stop", "length", "abort")}
        lat, ttft, itl = BUCKETS["request_latency"], BUCKETS["time_to_first_token"], BUCKETS["inter_token_latency"]
        self.ttft = h(M.TTFT, "Histogram of time to first token in seconds.", ttft)
        self.itl = h(M.ITL, "Histogram of inter-token latency in seconds.", itl)
        self.tpot = h(M.TPOT, "Histogram of time_per_output_token_seconds per request.", itl)
        self.e2e = h(M.E2E, "Histogram of e2e request latency in seconds.", lat)
        self.queue = h(M.QUEUE, "Histogram of time spent in WAITING phase for request.", lat)
        self.inference = h(M.INFERENCE_TIME, "Histogram of time spent in RUNNING phase for request.", lat)
        self.prefill = h(M.PREFILL_TIME, "Histogram of time spent in PREFILL phase for request.", lat)
        self.decode = h(M.DECODE_TIME, "Histogram of time spent in DECODE phase for request.", lat)
        self.iteration_tokens = h(M.ITERATION_TOKENS, "Histogram of number of tokens per engine_step.",
                                  BUCKETS["iteration_tokens"])
        req_tok = request_token_buckets(engine.p.max_model_len)
        self.req_prompt = h(M.REQUEST_PROMPT_TOKENS, "Number of prefill tokens processed.", req_tok)
        self.req_gen = h(M.REQUEST_GENERATION_TOKENS, "Number of generation tokens processed.", req_tok)
        c_cfg = engine.c
        info = {"block_size": engine.p.block_size, "enable_prefix_caching": c_cfg.enable_prefix_caching,
                "num_gpu_blocks": engine.p.num_blocks, "engine": "0", "simulated": True}
        Gauge(M.CACHE_CONFIG_INFO, "Information of the LLMEngine CacheConfig", list(info), registry=r) \
            .labels(**{k: str(v) for k, v in info.items()}).set(1)
        self.spec = None
        if c_cfg.num_speculative_tokens:
            self.spec = (c(M.SPEC_DRAFTS, "Number of spec decoding drafts."),
                         c(M.SPEC_DRAFT_TOKENS, "Number of draft tokens."),
                         c(M.SPEC_ACCEPTED, "Number of accepted tokens."))
            pos = Counter(M.SPEC_ACCEPTED_PER_POS, "Accepted tokens per draft position.", lab + ["position"], registry=r)
            self.spec_pos = [pos.labels(model, "0", str(i)) for i in range(c_cfg.num_speculative_tokens)]

    def on_step(self, plan, res) -> None:
        self.prefix_queries.inc(plan.prefix_queries)
        self.prefix_hits.inc(plan.prefix_hits)
        self.preemptions.inc(len(plan.preempted))
        self.prompt_tokens.inc(res.prompt_tokens_done)
        self.generation_tokens.inc(res.generated)
        self.iteration_tokens.observe(plan.prefill_tokens + res.generated)
        if self.spec and res.spec_drafts:
            self.spec[0].inc(res.spec_drafts)
            self.spec[1].inc(res.spec_draft_tokens)
            self.spec[2].inc(res.spec_accepted)
            for c, v in zip(self.spec_pos, res.spec_accepted_per_pos):
                c.inc(v)
        for ev in res.events:
            s = ev.seq
            if ev.first:
                self.ttft.observe(s.first_token - s.arrival)
            elif ev.itl is not None:
                self.itl.observe(ev.itl)
            if ev.finished:
                self.on_finish(s)

    def on_finish(self, s) -> None:
        n = len(s.output)
        self.success[s.finish_reason or "length"].inc()
        self.e2e.observe(s.finished_at - s.arrival)
        if s.first_scheduled is not None:
            self.queue.observe(s.first_scheduled - s.arrival)
        if s.first_token is not None:
            decode = s.last_token - s.first_token
            self.prefill.observe(s.first_token - s.first_scheduled)
            self.inference.observe(s.last_token - s.first_scheduled)
            self.decode.observe(decode)
            self.tpot.observe(decode / (n - 1) if n > 1 else 0.0)
        self.req_prompt.observe(len(s.prompt))
        self.req_gen.observe(n)


class FakeServer:
    """The fake engine behind an aiohttp app. ``start()`` runs it in a background thread and
    returns the base URL; ``stop()`` shuts it down. Also a context manager."""

    def __init__(self, profile: EngineProfile | str = "t4-qwen2.5-0.5b", config: EngineConfig | None = None, *,
                 model: str | None = None, host: str = "127.0.0.1", port: int = 0, time_scale: float = 1.0):
        prof = named_profile(profile) if isinstance(profile, str) else profile
        self.engine = FakeEngine(prof, config)
        self.model = model or prof.model
        self.host, self.port, self.time_scale = host, port, time_scale
        self.metrics = _Metrics(self.model, self.engine)
        self._streams: dict = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopping: asyncio.Event | None = None
        self._runner: web.AppRunner | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # -- HTTP -----------------------------------------------------------------------------------
    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/version", self._version)
        app.router.add_get("/v1/models", self._models)
        app.router.add_get("/metrics", self._metrics)
        app.router.add_post("/v1/completions", self._completions)
        app.router.add_post("/v1/chat/completions", self._chat)
        return app

    async def _health(self, request):
        return web.Response(status=200, headers={"x-servelab-simulated": "true"})

    async def _version(self, request):
        return web.json_response({"version": "servelab-fakeserver", "simulated": True})

    async def _models(self, request):
        return web.json_response({"object": "list", "data": [{
            "id": self.model, "object": "model", "created": int(time.time()), "owned_by": "servelab (simulated)",
            "root": self.model, "max_model_len": self.engine.p.max_model_len}]},
            headers={"x-servelab-simulated": "true"})

    async def _metrics(self, request):
        e = self.engine
        self.metrics.running.set(len(e.running))
        self.metrics.waiting.set(len(e.waiting))
        self.metrics.kv_usage.set(e.pool.usage())
        return web.Response(body=generate_latest(self.metrics.registry),
                            headers={"Content-Type": CONTENT_TYPE_LATEST})

    async def _completions(self, request):
        body = await request.json()
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
            prompt = prompt[0]
        if not isinstance(prompt, str):
            return _error(400, "this fake server accepts one string prompt per request")
        return await self._generate(request, body, tokenize(prompt), chat=False)

    async def _chat(self, request):
        body = await request.json()
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return _error(400, "messages must be a non-empty list")
        return await self._generate(request, body, chat_tokens(msgs), chat=True)

    async def _generate(self, request, body, prompt_ids, chat: bool):
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 16)
        try:
            seq = self.engine.add_request(prompt_ids, max_tokens, time.perf_counter())
        except ValueError as e:
            return _error(400, str(e))
        q: asyncio.Queue = asyncio.Queue()
        self._streams[seq.rid] = q
        self._wake.set()
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:16]
        created, obj = int(time.time()), ("chat.completion.chunk" if chat else "text_completion")
        usage = lambda: {"prompt_tokens": len(seq.prompt), "completion_tokens": len(seq.output),  # noqa: E731
                         "total_tokens": seq.num_tokens, "prompt_tokens_details": {"cached_tokens": seq.cached_tokens}}
        try:
            if not body.get("stream"):
                text = []
                while True:
                    texts, finished = await q.get()
                    if texts is None:
                        return _error(500, "fake engine failed; see the server's stderr")
                    text += texts
                    if finished:
                        break
                choice = ({"index": 0, "message": {"role": "assistant", "content": "".join(text)},
                           "finish_reason": seq.finish_reason} if chat else
                          {"index": 0, "text": "".join(text), "finish_reason": seq.finish_reason})
                return web.json_response({"id": rid, "object": "chat.completion" if chat else "text_completion",
                                          "created": created, "model": self.model, "choices": [choice],
                                          "usage": usage()}, headers={"x-servelab-simulated": "true"})
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                                               "x-servelab-simulated": "true"})
            await resp.prepare(request)
            first = True
            while True:
                texts, finished = await q.get()
                if texts is None:                  # the engine loop died: end the stream visibly
                    await resp.write(b'data: {"error": "fake engine failed"}\n\n')
                    break
                reason = seq.finish_reason if finished else None
                if chat:
                    if first:  # vLLM's first chat chunk carries the role, at the first token
                        await _sse(resp, {"id": rid, "object": obj, "created": created, "model": self.model,
                                          "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                                       "logprobs": None, "finish_reason": None}]})
                    choice = {"index": 0, "delta": {"content": "".join(texts)}, "logprobs": None, "finish_reason": reason}
                else:
                    choice = {"index": 0, "text": "".join(texts), "logprobs": None, "finish_reason": reason,
                              "stop_reason": None}
                await _sse(resp, {"id": rid, "object": obj, "created": created, "model": self.model, "choices": [choice]})
                first = False
                if finished:
                    break
            if (body.get("stream_options") or {}).get("include_usage"):
                await _sse(resp, {"id": rid, "object": obj, "created": created, "model": self.model, "choices": [],
                                  "usage": usage()})
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        except (ConnectionResetError, asyncio.CancelledError):
            self.engine.abort(seq, time.perf_counter())   # client went away: free its KV blocks
            raise
        finally:
            self._streams.pop(seq.rid, None)

    # -- the engine loop ---------------------------------------------------------------------------
    async def _engine_loop(self):
        try:
            await self._run_engine()
        except asyncio.CancelledError:
            raise
        except Exception:                          # a bug must be loud, not a silent hang
            traceback.print_exc()
            for q in list(self._streams.values()):
                q.put_nowait((None, True))
            raise

    async def _run_engine(self):
        e = self.engine
        while True:
            if not e.has_work():
                self._wake.clear()
                await self._wake.wait()
                continue
            plan = e.schedule(time.perf_counter())
            if plan.empty:
                await asyncio.sleep(0.001)
                continue
            await asyncio.sleep(e.step_time(plan) * self.time_scale)
            res = e.commit(plan, time.perf_counter())
            self.metrics.on_step(plan, res)
            for ev in res.events:   # (a preempted request just waits again: nothing to send)
                q = self._streams.get(ev.seq.rid)
                if q is not None:
                    q.put_nowait((ev.texts, ev.finished))

    async def _serve(self):
        self._wake = asyncio.Event()
        self._stopping = asyncio.Event()
        self._runner = web.AppRunner(self.app(), access_log=None)
        await self._runner.setup()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        self.port = sock.getsockname()[1]
        await web.SockSite(self._runner, sock).start()
        loop_task = asyncio.create_task(self._engine_loop())
        self._ready.set()
        try:
            await self._stopping.wait()
        finally:
            loop_task.cancel()
            await self._runner.cleanup()

    # -- lifecycle -------------------------------------------------------------------------------
    def start(self) -> str:
        """Run the server in a daemon thread; return its base URL once it is accepting requests."""
        def run():
            self._loop = asyncio.new_event_loop()
            self._loop.run_until_complete(self._serve())
            self._loop.close()
        self._thread = threading.Thread(target=run, name="servelab-fakeserver", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("fake server did not start")
        return self.url

    def stop(self) -> None:
        if self._loop and self._stopping and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stopping.set)
        if self._thread:
            self._thread.join(timeout=10)

    def serve_forever(self) -> None:
        """Foreground mode for the CLI (Ctrl-C to stop)."""
        print(f"fake vLLM (simulated) serving {self.model} on http://{self.host}:{self.port or '<auto>'}")
        print(self.engine.p.describe())
        try:
            asyncio.run(self._serve())
        except KeyboardInterrupt:
            pass

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


async def _sse(resp: web.StreamResponse, obj: dict) -> None:
    await resp.write(b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n")


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"object": "error", "message": message, "type": "BadRequestError", "code": status},
                             status=status)
