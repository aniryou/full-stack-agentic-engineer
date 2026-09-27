"""fakeserver.py — a fake OpenAI-compatible server: chat with tool calls, embeddings, and vLLM's prefix-cache accounting.

The one idea: to see what a memory layout does to the prefix cache you need a server that *counts* the
way vLLM counts. This one renders each chat request through a Hermes/Qwen-style chat template, splits
the text into tokens (4 characters each, so a token is ``len(text) // 4`` as the rest of the lab counts),
names every full 16-token block by a hash chained through its parent — the request's ``cache_salt``
enters the first block only, as in vLLM v0.30.0 (vllm-internals §4.3) — and reports the longest run of
leading blocks it has seen before, capped so the last prompt token is always recomputed. When a
request finishes, its prompt *and* output blocks are cached (the last token excepted). The answers come
from ``llm.ScriptedModel``; embeddings from the hashing embedder.

Like vLLM, per-request ``usage.prompt_tokens_details.cached_tokens`` appears only when the server runs
with ``enable_prompt_tokens_details`` (``--enable-prompt-tokens-details``); ``/metrics`` always has
``vllm:prefix_cache_queries_total`` / ``vllm:prefix_cache_hits_total`` (tokens), ``vllm:prompt_tokens_total``
and a ``vllm:time_to_first_token_seconds`` histogram with vLLM's bucket edges. **Everything it reports is
simulated**: ``/version`` says so, responses carry ``x-memlab-simulated: true``, and the TTFT is the
roofline prefill time of ``cachebench.step_cost`` for (cached, new) — not a measurement. One process
serves both chat and embeddings for convenience; vLLM serves one model per process.

    python -m memlab fake --port 8000 --prompt-tokens-details
    with FakeLLMServer(enable_prompt_tokens_details=True) as url: ...
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
import uuid
import zlib
from collections import OrderedDict

from aiohttp import web

from .cachebench import BLOCK_SIZE, prefill_ms
from .embedders import HashingEmbedder
from .llm import ScriptedModel, from_openai

# vLLM's TTFT histogram bucket edges (vllm/v1/metrics/buckets.py, v0.30.0), as the serving lab's fake has them.
TTFT_BUCKETS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0,
                40.0, 80.0, 160.0, 640.0, 2560.0]
MAX_SALT = 128          # vLLM validate_cache_salt: <= 128 characters, no '@', '/', '\\', NUL (facts sheet §11)


# ------------------------------------------------------------------------------------- template + tokens
def _render_call(tc: dict) -> str:
    args = tc.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return "<tool_call>\n" + json.dumps({"name": tc.get("name"), "arguments": args}, sort_keys=True) + "\n</tool_call>"


def output_text(text: str | None, tool_calls: list[dict] | None = None) -> str:
    """What the engine generates for one reply: the text or the tool calls, then the end-of-turn token."""
    body = "\n".join(_render_call(tc) for tc in tool_calls) if tool_calls else (text or "")
    return body + "<|im_end|>"


def render_chat(messages: list[dict], tools: list[dict] | None = None) -> str:
    """A Hermes/Qwen-style chat template over OpenAI messages: tools in the first system turn, tool results
    as ``<tool_response>`` user turns, then the generation prompt."""
    parts = []
    tool_block = ""
    if tools:
        tool_block = "\n\n# Tools\n" + "\n".join(json.dumps(t.get("function", t), sort_keys=True) for t in tools)
    first_system = True
    for m in messages:
        role, content = m.get("role"), m.get("content") or ""
        if role == "system" and first_system:
            parts.append(f"<|im_start|>system\n{content}{tool_block}<|im_end|>\n")
            first_system = False
            continue
        if role == "assistant":
            calls = [{"name": tc["function"]["name"], "arguments": tc["function"].get("arguments", "{}")}
                     for tc in m.get("tool_calls") or []]
            parts.append("<|im_start|>assistant\n" + output_text(content, calls) + "\n")
        elif role == "tool":
            parts.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n")
        else:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if first_system and tool_block:
        parts.insert(0, f"<|im_start|>system\n{tool_block.strip()}<|im_end|>\n")
    return "".join(parts) + "<|im_start|>assistant\n"


def tokenize(text: str) -> list[int]:
    """A toy tokenizer: 4-character pieces, ids by crc32. Same text, same ids; ``len`` ≈ ``len(text) // 4``."""
    return [zlib.crc32(text[i:i + 4].encode()) for i in range(0, len(text), 4)]


# ------------------------------------------------------------------------------------- the prefix cache
def block_hash(parent, tokens, extra=None) -> str:
    """``minengine.kv.hash_block``'s recipe: sha256 over (parent hash, the block's token ids, extra keys)."""
    return hashlib.sha256(repr((parent, tuple(tokens), extra)).encode()).hexdigest()


def block_hashes(tokens: list[int], salt: str | None = None, block_size: int = BLOCK_SIZE) -> list[str]:
    out, parent = [], None
    for i in range(len(tokens) // block_size):
        extra = (salt,) if (i == 0 and salt) else None           # vLLM: cache_salt on the first block only
        parent = block_hash(parent, tokens[i * block_size:(i + 1) * block_size], extra)
        out.append(parent)
    return out


class PrefixCache:
    """Full-block hashes with LRU eviction. ``lookup`` stops at the first miss and never covers the last
    prompt token; ``insert`` publishes the full blocks of a finished sequence minus its last token."""

    def __init__(self, num_blocks: int = 200_000, block_size: int = BLOCK_SIZE):
        self.blocks: OrderedDict[str, None] = OrderedDict()
        self.num_blocks, self.block_size = num_blocks, block_size

    def lookup(self, tokens: list[int], salt: str | None = None) -> int:
        hashes = block_hashes(tokens, salt, self.block_size)[: (len(tokens) - 1) // self.block_size]
        n = 0
        for h in hashes:
            if h not in self.blocks:
                break
            self.blocks.move_to_end(h)
            n += 1
        return n * self.block_size

    def insert(self, tokens: list[int], salt: str | None = None) -> None:
        for h in block_hashes(tokens[:-1], salt, self.block_size):
            self.blocks[h] = None
            self.blocks.move_to_end(h)
        while len(self.blocks) > self.num_blocks:
            self.blocks.popitem(last=False)

    def reset(self) -> int:
        n = len(self.blocks)
        self.blocks.clear()
        return n


# ------------------------------------------------------------------------------------- the server
class FakeLLMServer:
    def __init__(self, *, host: str = "127.0.0.1", port: int = 0, model: str = "memlab/scripted-chat",
                 embed_model: str = "memlab/hashing-1024", enable_prompt_tokens_details: bool = False,
                 gpu: str = "L4", llm: str = "qwen2.5-1.5b", time_scale: float = 0.0, num_blocks: int = 200_000):
        self.host, self.port, self.model, self.embed_model = host, port, model, embed_model
        self.details, self.gpu, self.llm, self.time_scale = enable_prompt_tokens_details, gpu, llm, time_scale
        self.cache = PrefixCache(num_blocks)
        self.embedder = HashingEmbedder(1024)
        self.scripted = ScriptedModel()
        self.m = {"queries": 0, "hits": 0, "prompt": 0, "generation": 0, "success": 0, "ttft_sum": 0.0,
                  "ttft_count": 0, "ttft_buckets": [0] * len(TTFT_BUCKETS), "resets": 0}
        self._lock = threading.Lock()
        self._thread = None
        self._loop = None
        self._ready = threading.Event()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/version", self._version)
        app.router.add_get("/v1/models", self._models)
        app.router.add_get("/metrics", self._metrics)
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_post("/v1/embeddings", self._embeddings)
        app.router.add_post("/reset_prefix_cache", self._reset)
        return app

    async def _health(self, request):
        return web.Response(status=200, headers={"x-memlab-simulated": "true"})

    async def _version(self, request):
        return web.json_response({"version": "memlab-fakeserver", "simulated": True})

    async def _models(self, request):
        data = [{"id": m, "object": "model", "owned_by": "memlab (simulated)", "created": 0}
                for m in (self.model, self.embed_model)]
        return web.json_response({"object": "list", "data": data})

    async def _reset(self, request):
        with self._lock:
            n = self.cache.reset()
            self.m["resets"] += 1
        return web.json_response({"ok": True, "blocks_evicted": n})

    def handle_chat(self, body: dict) -> tuple[dict, dict]:
        """The request logic without HTTP (the tests and ``cachebench`` call it through HTTP; notebooks can
        call it directly). Returns (response JSON, extra headers)."""
        messages, tools = body.get("messages") or [], body.get("tools")
        salt = body.get("cache_salt")
        if salt is not None and (not 1 <= len(salt) <= MAX_SALT or any(c in salt for c in "@/\\\x00")):
            raise ValueError("cache_salt must be 1-128 characters without '@', '/', '\\\\' or NUL")
        prompt = tokenize(render_chat(messages, tools))
        internal = from_openai(messages)
        schemas = [t.get("function", t) for t in tools] if tools else None
        reply = self.scripted.generate(internal, schemas)
        calls = [{"name": tc.name, "arguments": tc.args} for tc in reply.tool_calls]
        out = tokenize(output_text(reply.text, calls))
        with self._lock:
            cached = self.cache.lookup(prompt, salt)
            self.cache.insert(prompt + out, salt)
            ttft_s = prefill_ms(len(prompt), cached, self.gpu, self.llm) / 1e3
            self.m["queries"] += len(prompt)
            self.m["hits"] += cached
            self.m["prompt"] += len(prompt)
            self.m["generation"] += len(out)
            self.m["success"] += 1
            self.m["ttft_sum"] += ttft_s
            self.m["ttft_count"] += 1
            for i, edge in enumerate(TTFT_BUCKETS):
                if ttft_s <= edge:
                    self.m["ttft_buckets"][i] += 1
        usage = {"prompt_tokens": len(prompt), "completion_tokens": len(out), "total_tokens": len(prompt) + len(out)}
        if self.details:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        if calls:
            message = {"role": "assistant", "content": None,
                       "tool_calls": [{"id": tc.id, "type": "function",
                                       "function": {"name": tc.name, "arguments": json.dumps(tc.args)}}
                                      for tc in reply.tool_calls]}
            finish = "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": reply.text}, "stop"
        resp = {"id": "chatcmpl-" + uuid.uuid4().hex[:16], "object": "chat.completion", "created": int(time.time()),
                "model": self.model, "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": usage}
        return resp, {"x-memlab-simulated": "true", "x-memlab-simulated-ttft-ms": f"{ttft_s * 1e3:.3f}",
                      "x-memlab-cached-tokens": str(cached)}

    async def _chat(self, request):
        body = await request.json()
        if body.get("stream"):
            return web.json_response({"error": {"message": "this fake server does not stream"}}, status=400)
        try:
            resp, headers = self.handle_chat(body)
        except ValueError as e:
            return web.json_response({"error": {"message": str(e), "type": "BadRequestError"}}, status=400)
        if self.time_scale:
            await asyncio.sleep(float(headers["x-memlab-simulated-ttft-ms"]) / 1e3 * self.time_scale)
        return web.json_response(resp, headers=headers)

    async def _embeddings(self, request):
        body = await request.json()
        inp = body.get("input")
        texts = [inp] if isinstance(inp, str) else list(inp or [])
        vecs = self.embedder.encode(texts) if texts else []
        n = sum(len(tokenize(t)) for t in texts)
        return web.json_response({"object": "list", "model": self.embed_model,
                                  "data": [{"object": "embedding", "index": i, "embedding": [float(x) for x in v]}
                                           for i, v in enumerate(vecs)],
                                  "usage": {"prompt_tokens": n, "total_tokens": n}},
                                 headers={"x-memlab-simulated": "true"})

    async def _metrics(self, request):
        lab = f'engine="0",model_name="{self.model}"'
        m = self.m
        lines = []

        def metric(name, kind, value, help_):
            lines.extend([f"# HELP {name} {help_}", f"# TYPE {name} {kind}", f"{name}{{{lab}}} {value}"])
        metric("vllm:num_requests_running", "gauge", 0, "Number of requests in model execution batches.")
        metric("vllm:num_requests_waiting", "gauge", 0, "Number of requests waiting to be processed.")
        metric("vllm:kv_cache_usage_perc", "gauge", round(len(self.cache.blocks) / self.cache.num_blocks, 6),
               "KV-cache usage. 1 means 100 percent usage.")
        metric("vllm:prefix_cache_queries_total", "counter", m["queries"],
               "Prefix cache queries, in terms of number of queried tokens.")
        metric("vllm:prefix_cache_hits_total", "counter", m["hits"],
               "Prefix cache hits, in terms of number of cached tokens.")
        metric("vllm:prompt_tokens_total", "counter", m["prompt"], "Number of prefill tokens processed.")
        metric("vllm:generation_tokens_total", "counter", m["generation"], "Number of generation tokens processed.")
        lines += ["# HELP vllm:request_success_total Count of successfully processed requests.",
                  "# TYPE vllm:request_success_total counter",
                  f'vllm:request_success_total{{{lab},finished_reason="stop"}} {m["success"]}',
                  "# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds (SIMULATED).",
                  "# TYPE vllm:time_to_first_token_seconds histogram"]
        for edge, cumulative in zip(TTFT_BUCKETS, m["ttft_buckets"]):     # each observation counted in every le >= it
            lines.append(f'vllm:time_to_first_token_seconds_bucket{{{lab},le="{edge}"}} {cumulative}')
        lines += [f'vllm:time_to_first_token_seconds_bucket{{{lab},le="+Inf"}} {m["ttft_count"]}',
                  f"vllm:time_to_first_token_seconds_sum{{{lab}}} {m['ttft_sum']:.6f}",
                  f"vllm:time_to_first_token_seconds_count{{{lab}}} {m['ttft_count']}",
                  "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig",
                  "# TYPE vllm:cache_config_info gauge",
                  f'vllm:cache_config_info{{block_size="{BLOCK_SIZE}",enable_prefix_caching="True",'
                  f'num_gpu_blocks="{self.cache.num_blocks}",simulated="True"}} 1']
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")

    # -- lifecycle -------------------------------------------------------------------------------
    def start(self) -> str:
        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            runner = web.AppRunner(self.app(), access_log=None)
            self._loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, self.host, self.port)
            self._loop.run_until_complete(site.start())
            self.port = site._server.sockets[0].getsockname()[1]
            self._ready.set()
            self._loop.run_forever()
            self._loop.run_until_complete(runner.cleanup())
            self._loop.close()
        self._thread = threading.Thread(target=run, daemon=True, name="memlab-fakeserver")
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("fake server did not start")
        return self.url

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def parse_metrics(text: str) -> dict[str, float]:
    """Sum each metric over its label sets (``_total`` suffixes kept): enough for two-scrape deltas."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_labels, _, value = line.rpartition(" ")
        name = name_labels.split("{", 1)[0]
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return out


def scrape(url: str, headers: dict | None = None) -> dict[str, float]:
    import urllib.request
    req = urllib.request.Request(url.rstrip("/") + "/metrics", headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as r:
        return parse_metrics(r.read().decode())


def hit_rate(before: dict, after: dict) -> float:
    """``vllm:prefix_cache_hits / queries`` over the window between two scrapes (counters are cumulative)."""
    q = after.get("vllm:prefix_cache_queries_total", 0) - before.get("vllm:prefix_cache_queries_total", 0)
    h = after.get("vllm:prefix_cache_hits_total", 0) - before.get("vllm:prefix_cache_hits_total", 0)
    return h / q if q > 0 else float("nan")
