"""An OpenAI-compatible fake engine: vLLM's timing *shape* and metric *names*, no model (T0).

The one idea: an engine replica is a stateful cache with a queue in front of it. For every
request

    TTFT = wait for a batch slot + KV blocks        (vllm:num_requests_waiting > 0 means this)
         + wait for the replica's prefill "GPU"     (prefills are serialized per replica here)
         + prefill_overhead + uncached_prompt_tokens x prefill_s_per_token
    ITL  = itl_s x (1 + itl_batch_slope x (running - 1))

and "uncached" is decided by a vLLM-style automatic prefix cache: the prompt's *full* 16-token
blocks get chained hashes; blocks already in this replica's cache are reused (up to
prompt_len - 1 tokens, as vLLM always recomputes at least one token); finished requests
release their blocks into an LRU of free-but-cached blocks (tail blocks first), evicted only
when a new request needs room. `vllm:kv_cache_usage_perc` counts blocks held by running
requests, so a replica full of *reusable* cache still reports low usage — exactly as vLLM does.

Everything that makes LLM load balancing different from web load balancing is in those
formulas: prompt-length-dependent cost, cache locality, and head-of-line queueing. All timing
is *emulated* (asyncio sleeps calibrated by EngineProfile), so any latency you measure against
this backend is "measured on this machine, emulated backend" — real HTTP, fake GPU.

Upstream equivalent: `llm-d-inference-sim` (ghcr.io/llm-d/llm-d-inference-sim), which the
deploy/local and deploy/kind stacks use; `EngineProfile.sim_args()` prints the matching flags.
What carries over is the *per-request* model: TTFT = overhead + uncached tokens x per-token
time, ITL per output token, 16-token prefix-cache blocks, max_num_seqs slots. What does not:
this fake serializes prefills per replica (a request waits for the prefills ahead of it), the
simulator runs each request's prefill independently and only stretches prefill and ITL by
`--time-factor-under-load` as the batch fills (sim_args sets it to match the ITL slope). So
cache-aware routing wins less TTFT on the simulator than here -- compare hit rates and the
per-endpoint split there, not TTFT alone.

Not modelled (documented simplifications): preemption, chunked prefill interleaving prefill
with decode, caching of generated-token blocks, speculative decoding, real tokenization. KV
blocks for all `max_tokens` output tokens are reserved at admission (vLLM allocates decode
blocks as tokens are generated), so `vllm:kv_cache_usage_perc` and KV-limited admission are
pessimistic compared with vLLM, most visibly for requests with a large max_tokens.
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import random
import re
import struct
import time
import uuid
from dataclasses import dataclass, field

from .promtext import Registry

__all__ = ["EngineProfile", "KVCache", "EmulatedEngine", "FakeBackend", "tokenize", "chat_template",
           "TTFT_BUCKETS", "ITL_BUCKETS", "REQUEST_LATENCY_BUCKETS"]

# vLLM's histogram buckets as mirrored by llm-d-inference-sim (verify: vllm/v1/metrics/buckets.py)
TTFT_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5,
                10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0)
ITL_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
               20.0, 40.0, 80.0)
REQUEST_LATENCY_BUCKETS = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                           120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0)

_PIECE = re.compile(r"\w+|[^\w\s]")
_VOCAB = ("the plan uses a tool to read the file and then checks the result before it writes the "
          "next step so each call returns data that the agent can verify quickly with one more "
          "query against the index while the cache stays warm for the following turn").split()


def tokenize(text: str) -> list[int]:
    """Word/punctuation pieces hashed to 32-bit ids (FNV-1a), like llm-d-inference-sim's
    simulated tokenizer. Deterministic, so equal prompt prefixes give equal token prefixes."""
    out = []
    for piece in _PIECE.findall(text):
        h = 0x811C9DC5
        for b in piece.encode():
            h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
        out.append(h)
    return out


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def chat_template(body: dict) -> str:
    """A minimal chat template: tools first, then one tagged section per message."""
    parts = []
    if body.get("tools"):
        parts.append("<|tools|>\n" + json.dumps(body["tools"], sort_keys=True) + "\n")
    for m in body.get("messages", []):
        parts.append(f"<|{m.get('role', 'user')}|>\n{_text(m.get('content'))}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


@dataclass
class EngineProfile:
    """Emulation knobs. Defaults: ~20k prompt tokens/s of prefill, 4 ms/token decode, 8 slots,
    2,048 x 16-token KV blocks (32k tokens). Scale them together to make runs faster/slower."""
    max_num_seqs: int = 8
    num_gpu_blocks: int = 2048
    block_size: int = 16
    prefill_overhead_s: float = 0.002
    prefill_s_per_token: float = 0.00005
    itl_s: float = 0.004
    itl_batch_slope: float = 0.02
    max_loras: int = 0
    lora_load_s: float = 0.02
    default_max_tokens: int = 32
    max_model_len: int = 65536

    def prefill_seconds(self, prompt_tokens: int, cached_tokens: int) -> float:
        return self.prefill_overhead_s + max(0, prompt_tokens - cached_tokens) * self.prefill_s_per_token

    def itl_seconds(self, running: int) -> float:
        return self.itl_s * (1.0 + self.itl_batch_slope * max(0, running - 1))

    def time_factor_under_load(self) -> float:
        """llm-d-inference-sim scales latencies by 1 + (f - 1)(n - 1)/(max_num_seqs - 1) with n running;
        this f makes its ITL slope equal ours at a full batch: 1 + slope x (max_num_seqs - 1)."""
        return 1.0 + self.itl_batch_slope * max(0, self.max_num_seqs - 1)

    def sim_args(self, model: str = "lab/llm", port: int = 8000) -> list[str]:
        """The llm-d-inference-sim flags (v0.11) that approximate this profile."""
        ms = lambda s: f"{s * 1e3:g}ms"
        us = lambda s: f"{s * 1e6:g}us"
        return ["--model", model, "--port", str(port), "--max-num-seqs", str(self.max_num_seqs),
                "--max-model-len", str(self.max_model_len), "--latency-calculator", "per-token",
                "--prefill-overhead", ms(self.prefill_overhead_s), "--prefill-time-per-token", us(self.prefill_s_per_token),
                "--inter-token-latency", ms(self.itl_s), "--time-factor-under-load", f"{self.time_factor_under_load():g}",
                "--enable-kvcache", "--kv-cache-size", str(self.num_gpu_blocks), "--block-size", str(self.block_size)]


class KVCache:
    """Block pool with vLLM-style automatic prefix caching on full blocks."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.ref: dict[int, int] = {}                          # cached block hash -> refcount
        self.free_cached: collections.OrderedDict = collections.OrderedDict()   # refcount-0 blocks, LRU first
        self.in_use_cached = 0
        self.anon_used = 0                                     # partial/decode blocks (never cached)
        self.evictions = 0

    def hashes(self, tokens) -> list[int]:
        out, prev = [], b"\x00" * 8
        for i in range(0, len(tokens) - len(tokens) % self.block_size, self.block_size):
            blk = struct.pack(f"<{self.block_size}I", *tokens[i:i + self.block_size])
            prev = hashlib.blake2b(prev + blk, digest_size=8).digest()
            out.append(int.from_bytes(prev, "little"))
        return out

    @property
    def used(self) -> int:
        return self.in_use_cached + self.anon_used

    def usage(self) -> float:
        return self.used / self.num_blocks

    def lookup(self, hashes) -> int:
        n = 0
        for h in hashes:
            if h not in self.ref:
                break
            n += 1
        return n

    def can_admit(self, hashes, blocks_needed: int, max_hit: int) -> bool:
        hit = min(self.lookup(hashes), max_hit)
        reactivated = sum(1 for h in hashes[:hit] if self.ref[h] == 0)
        return self.used + reactivated + (blocks_needed - hit) <= self.num_blocks

    def _make_room(self, n: int) -> None:
        while self.num_blocks - self.used - len(self.free_cached) < n and self.free_cached:
            h, _ = self.free_cached.popitem(last=False)
            del self.ref[h]
            self.evictions += 1

    def _take(self, h: int) -> None:
        if self.ref.get(h, 0) == 0:
            self.free_cached.pop(h, None)
            self.in_use_cached += 1
        self.ref[h] = self.ref.get(h, 0) + 1

    def allocate(self, hashes, blocks_needed: int, max_hit: int) -> tuple[int, tuple]:
        """Reuse the cached prefix, allocate the rest; returns (hit blocks, allocation record)."""
        hit = min(self.lookup(hashes), max_hit)
        taken = []
        for h in hashes[:hit]:
            self._take(h)
            taken.append(h)
        for h in hashes[hit:]:
            if h not in self.ref:
                self._make_room(1)
            self._take(h)
            taken.append(h)
        anon = max(0, blocks_needed - len(hashes))
        self._make_room(anon)
        self.anon_used += anon
        return hit, (taken, anon)

    def free(self, record) -> None:
        taken, anon = record
        for h in reversed(taken):                   # tail blocks become LRU-first, as in vLLM
            self.ref[h] -= 1
            if self.ref[h] == 0:
                self.in_use_cached -= 1
                self.free_cached[h] = None
        self.anon_used -= anon


@dataclass(eq=False)
class _Req:
    tokens: list
    max_tokens: int
    lora: str | None
    arrival: float
    hashes: list = field(default_factory=list)
    blocks_needed: int = 0
    max_hit: int = 0
    alloc: tuple | None = None


class EmulatedEngine:
    def __init__(self, profile: EngineProfile | None = None, model: str = "lab/llm", lora_modules=(), clock=time.perf_counter):
        self.p = profile or EngineProfile()
        self.model, self.loras, self.clock = model, list(lora_modules), clock
        self.kv = KVCache(self.p.num_gpu_blocks, self.p.block_size)
        self.waiting: collections.deque = collections.deque()
        self.running: set = set()
        self._cond: asyncio.Condition | None = None
        self._prefill: asyncio.Lock | None = None
        self.resident_loras: collections.OrderedDict = collections.OrderedDict()
        self._init_metrics()

    # ---------------------------------------------------------------- metrics (vLLM names)
    def _init_metrics(self):
        r = self.registry = Registry()
        L = ["model_name", "engine"]
        self.lv = {"model_name": self.model, "engine": "0"}
        self.g_running = r.gauge("vllm:num_requests_running", "Number of requests in model execution batches.", L)
        self.g_waiting = r.gauge("vllm:num_requests_waiting", "Number of requests waiting to be processed.", L)
        self.g_kv = r.gauge("vllm:kv_cache_usage_perc", "KV-cache usage. 1 means 100 percent usage.", L)
        self.c_pq = r.counter("vllm:prefix_cache_queries", "Prefix cache queries, in terms of number of queried tokens.", L)
        self.c_ph = r.counter("vllm:prefix_cache_hits", "Prefix cache hits, in terms of number of cached tokens.", L)
        self.c_pre = r.counter("vllm:num_preemptions", "Cumulative number of preemption from the engine.", L)
        self.c_prompt = r.counter("vllm:prompt_tokens", "Number of prefill tokens processed.", L)
        self.c_gen = r.counter("vllm:generation_tokens", "Number of generation tokens processed.", L)
        self.c_ok = r.counter("vllm:request_success", "Count of successfully processed requests.", L + ["finished_reason"])
        self.h_ttft = r.histogram("vllm:time_to_first_token_seconds", "Histogram of time to first token in seconds.", TTFT_BUCKETS, L)
        self.h_itl = r.histogram("vllm:inter_token_latency_seconds", "Histogram of inter-token latency in seconds.", ITL_BUCKETS, L)
        self.h_tpot = r.histogram("vllm:request_time_per_output_token_seconds", "Histogram of time_per_output_token_seconds per request.", ITL_BUCKETS, L)
        self.h_e2e = r.histogram("vllm:e2e_request_latency_seconds", "Histogram of e2e request latency in seconds.", REQUEST_LATENCY_BUCKETS, L)
        self.h_queue = r.histogram("vllm:request_queue_time_seconds", "Histogram of time spent in WAITING phase for request.", REQUEST_LATENCY_BUCKETS, L)
        self.g_cache_info = r.gauge("vllm:cache_config_info", "Information of the LLMEngine CacheConfig",
                                    ["block_size", "num_gpu_blocks", "enable_prefix_caching", "engine"])
        self.g_cache_info.labels(block_size=self.p.block_size, num_gpu_blocks=self.p.num_gpu_blocks,
                                 enable_prefix_caching="True", engine="0").set(1)
        for g in (self.g_running, self.g_waiting, self.g_kv):
            g.labels(**self.lv).set(0)
        for c in (self.c_pq, self.c_ph, self.c_pre, self.c_prompt, self.c_gen):
            c.labels(**self.lv).inc(0)
        self.g_lora = None
        if self.p.max_loras > 0:
            self.g_lora = r.gauge("vllm:lora_requests_info", "Running stats on lora requests.",
                                  ["max_lora", "waiting_lora_adapters", "running_lora_adapters"])

    def _update_gauges(self):
        self.g_running.labels(**self.lv).set(len(self.running))
        self.g_waiting.labels(**self.lv).set(len(self.waiting))
        self.g_kv.labels(**self.lv).set(self.kv.usage())
        if self.g_lora is not None:
            run = ",".join(sorted({r.lora for r in self.running if r.lora}))
            wait = ",".join(sorted({r.lora for r in self.waiting if r.lora}))
            self.g_lora.labels(max_lora=self.p.max_loras, waiting_lora_adapters=wait, running_lora_adapters=run).set(time.time())
            if len(self.g_lora._children) > 16:          # vLLM leaves old label sets behind; keep a few
                for k in sorted(self.g_lora._children, key=lambda k: self.g_lora._children[k].v)[:-8]:
                    del self.g_lora._children[k]

    def metrics_text(self) -> str:
        return self.registry.render()

    # ---------------------------------------------------------------- scheduling
    def _loop_objects(self):
        if self._cond is None:
            self._cond = asyncio.Condition()
            self._prefill = asyncio.Lock()
        return self._cond, self._prefill

    def _can_admit(self, r: _Req) -> bool:
        if len(self.running) >= self.p.max_num_seqs:
            return False
        if r.lora and self.p.max_loras > 0:
            active = {x.lora for x in self.running if x.lora}
            if r.lora not in active and len(active) >= self.p.max_loras:
                return False
        return self.kv.can_admit(r.hashes, r.blocks_needed, r.max_hit)

    def validate(self, n_prompt: int, max_tokens: int) -> str | None:
        if n_prompt + max_tokens > self.p.max_model_len:
            return (f"This model's maximum context length is {self.p.max_model_len} tokens. However, you requested "
                    f"{n_prompt + max_tokens} tokens ({n_prompt} in the messages, {max_tokens} in the completion).")
        if -(-(n_prompt + max_tokens) // self.p.block_size) > self.p.num_gpu_blocks:
            return "request needs more KV blocks than the engine has"
        return None

    async def generate(self, tokens, max_tokens: int, lora: str | None = None):
        """Async generator of events: ('first', cached_tokens), ('token', i), ...; always ends by
        releasing its batch slot and KV blocks, even if the consumer stops early."""
        cond, prefill = self._loop_objects()
        p = self.p
        r = _Req(tokens=tokens, max_tokens=max_tokens, lora=lora, arrival=self.clock())
        r.hashes = self.kv.hashes(tokens)
        r.blocks_needed = -(-(len(tokens) + max_tokens) // p.block_size)
        r.max_hit = max(0, (len(tokens) - 1) // p.block_size)
        admitted = False
        try:
            async with cond:
                self.waiting.append(r)
                self._update_gauges()
                await cond.wait_for(lambda: self.waiting[0] is r and self._can_admit(r))
                self.waiting.popleft()
                self.running.add(r)
                admitted = True
                hit, r.alloc = self.kv.allocate(r.hashes, r.blocks_needed, r.max_hit)
                self._update_gauges()
                cond.notify_all()
            cached = hit * p.block_size
            t_admit = self.clock()
            self.h_queue.labels(**self.lv).observe(t_admit - r.arrival)
            self.c_pq.labels(**self.lv).inc(len(tokens))
            self.c_ph.labels(**self.lv).inc(cached)
            self.c_prompt.labels(**self.lv).inc(len(tokens))
            async with prefill:
                delay = p.prefill_seconds(len(tokens), cached)
                if lora:
                    if lora in self.resident_loras:
                        self.resident_loras.move_to_end(lora)
                    else:
                        delay += p.lora_load_s
                        self.resident_loras[lora] = None
                        while len(self.resident_loras) > max(1, p.max_loras):
                            self.resident_loras.popitem(last=False)
                await asyncio.sleep(delay)
            t_first = self.clock()
            self.h_ttft.labels(**self.lv).observe(t_first - r.arrival)
            yield ("first", cached)
            prev = t_first
            for i in range(1, max_tokens):
                await asyncio.sleep(p.itl_seconds(len(self.running)))
                now = self.clock()
                self.h_itl.labels(**self.lv).observe(now - prev)
                prev = now
                yield ("token", i)
            end = self.clock()
            if max_tokens > 1:
                self.h_tpot.labels(**self.lv).observe((end - t_first) / (max_tokens - 1))
            self.h_e2e.labels(**self.lv).observe(end - r.arrival)
            self.c_gen.labels(**self.lv).inc(max_tokens)
            self.c_ok.labels(finished_reason="length", **self.lv).inc()
        finally:
            async with cond:
                if admitted:
                    self.running.discard(r)
                    if r.alloc is not None:
                        self.kv.free(r.alloc)
                elif r in self.waiting:
                    self.waiting.remove(r)
                self._update_gauges()
                cond.notify_all()


def _reply_words(prompt: str, n: int) -> list[str]:
    seed = int.from_bytes(hashlib.blake2b(prompt.encode(), digest_size=8).digest(), "little")
    rng = random.Random(seed)
    return [_VOCAB[rng.randrange(len(_VOCAB))] for _ in range(n)]


class FakeBackend:
    """An aiohttp app serving /v1/chat/completions, /v1/completions, /v1/models, /metrics, /health."""

    def __init__(self, name: str = "fake", profile: EngineProfile | None = None, model: str = "lab/llm",
                 lora_modules=()):
        self.name = name
        self.engine = EmulatedEngine(profile, model, lora_modules)
        self._runner = None
        self.url: str | None = None

    @property
    def profile(self) -> EngineProfile:
        return self.engine.p

    def make_app(self):
        from aiohttp import web
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/chat/completions", self.handle)
        app.router.add_post("/v1/completions", self.handle)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_get("/metrics", self.handle_metrics)
        app.router.add_get("/health", self.handle_health)
        return app

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        from aiohttp import web
        self._runner = web.AppRunner(self.make_app(), access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()
        self.url = f"http://{host}:{site._server.sockets[0].getsockname()[1]}"
        return self.url

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()

    async def handle_metrics(self, request):
        from aiohttp import web
        return web.Response(text=self.engine.metrics_text(), content_type="text/plain")

    async def handle_health(self, request):
        from aiohttp import web
        return web.Response(text="")

    async def handle_models(self, request):
        from aiohttp import web
        e = self.engine
        data = [{"id": e.model, "object": "model", "owned_by": "igwlab", "root": e.model, "parent": None}]
        data += [{"id": a, "object": "model", "owned_by": "igwlab", "root": a, "parent": e.model} for a in e.loras]
        return web.json_response({"object": "list", "data": data})

    @staticmethod
    def _err(status, msg, typ):
        from aiohttp import web
        return web.json_response({"error": {"message": msg, "type": typ, "code": status}}, status=status)

    async def handle(self, request):
        from aiohttp import web
        try:
            body = await request.json()
        except Exception:
            return self._err(400, "invalid JSON body", "BadRequestError")
        e = self.engine
        model = body.get("model") or e.model
        if model != e.model and model not in e.loras:
            return self._err(404, f"The model `{model}` does not exist.", "NotFoundError")
        chat = request.path.endswith("/chat/completions")
        prompt = chat_template(body) if chat else str(body.get("prompt", ""))
        tokens = tokenize(prompt)
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or e.p.default_max_tokens)
        problem = e.validate(len(tokens), max_tokens)
        if problem:
            return self._err(400, problem, "BadRequestError")
        words = _reply_words(prompt, max_tokens)
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        lora = model if model != e.model else None
        headers = {"x-inference-pod": self.name, "x-request-id": request.headers.get("x-request-id", rid)}
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        gen = e.generate(tokens, max_tokens, lora)
        cached = 0

        def usage():
            return {"prompt_tokens": len(tokens), "completion_tokens": max_tokens,
                    "total_tokens": len(tokens) + max_tokens, "prompt_tokens_details": {"cached_tokens": cached}}

        def chunk(i=None, finish=None, role=False):
            if chat:
                delta = {} if i is None else ({"role": "assistant", "content": ""} if role else {"content": words[i] + " "})
                ch = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
                obj = "chat.completion.chunk"
            else:
                ch = {"index": 0, "text": "" if i is None else words[i] + " ", "logprobs": None, "finish_reason": finish}
                obj = "text_completion"
            return {"id": rid, "object": obj, "created": created, "model": model, "choices": [ch]}

        if not stream:
            try:
                async for kind, val in gen:
                    if kind == "first":
                        cached = val
            finally:
                await gen.aclose()
            text = " ".join(words) + " "
            choice = ({"index": 0, "message": {"role": "assistant", "content": text}, "logprobs": None,
                       "finish_reason": "length"} if chat else
                      {"index": 0, "text": text, "logprobs": None, "finish_reason": "length"})
            return web.json_response({"id": rid, "object": "chat.completion" if chat else "text_completion",
                                      "created": created, "model": model, "choices": [choice], "usage": usage()},
                                     headers=headers)

        resp = web.StreamResponse(headers={**headers, "Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)

        def sse(obj) -> bytes:
            return b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"

        try:
            async for kind, val in gen:
                if kind == "first":
                    cached = val
                    first = (sse(chunk(0, role=True)) if chat else b"") + sse(chunk(0))
                    await resp.write(first)
                else:
                    await resp.write(sse(chunk(val)))
            await resp.write(sse(chunk(None, finish="length")))
            if include_usage:
                await resp.write(sse({"id": rid, "object": "chat.completion.chunk" if chat else "text_completion",
                                      "created": created, "model": model, "choices": [], "usage": usage()}))
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
        except (ConnectionError, OSError):
            pass                                          # client went away; the generator releases its slot
        finally:
            await gen.aclose()
        return resp


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m igwlab.fakebackend", description="Emulated vLLM-shaped backend (T0).")
    ap.add_argument("--name", default="fake")
    ap.add_argument("--model", default="lab/llm")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-num-seqs", type=int, default=EngineProfile.max_num_seqs)
    ap.add_argument("--num-gpu-blocks", type=int, default=EngineProfile.num_gpu_blocks)
    ap.add_argument("--prefill-us-per-token", type=float, default=EngineProfile.prefill_s_per_token * 1e6)
    ap.add_argument("--itl-ms", type=float, default=EngineProfile.itl_s * 1e3)
    ap.add_argument("--max-loras", type=int, default=0)
    ap.add_argument("--lora", action="append", default=[], help="LoRA adapter name (repeatable)")
    a = ap.parse_args(argv)
    prof = EngineProfile(max_num_seqs=a.max_num_seqs, num_gpu_blocks=a.num_gpu_blocks,
                         prefill_s_per_token=a.prefill_us_per_token / 1e6, itl_s=a.itl_ms / 1e3, max_loras=a.max_loras)
    fb = FakeBackend(a.name, prof, a.model, a.lora)

    async def run():
        url = await fb.start(a.host, a.port)
        print(f"fake backend {a.name!r} ({a.model}) on {url}", flush=True)
        await asyncio.Event().wait()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
