"""Fake LLM providers over HTTP, in two dialects, with the failure modes a gateway exists to handle.

The one idea: to learn what a gateway does you need upstreams that behave like real ones — and misbehave like
them. `FakeProvider` is an aiohttp server modelled on `servelab/fakeserver.py` (04 serving lab) that speaks

* the **OpenAI** dialect (`POST /v1/chat/completions`, `Authorization: Bearer`), with vLLM 0.30.0's frontend
  quirks: `stream_options` without `stream` is a 400, `cache_salt` is validated and salts the prefix cache,
  a mid-stream failure is an `error` object inside the HTTP 200 followed by `[DONE]`, and error codes are the
  HTTP status as an int (`error_code_style="vllm"`; `"openai"` gives OpenAI's string codes); or
* the **Anthropic** dialect (`POST /v1/messages`, `x-api-key` + `anthropic-version`), with named stream events,
  `input_tokens` that exclude cache reads, a cumulative `output_tokens` in `message_delta`, and 529 overloaded.

Each has its own RPM and TPM limits over a sliding "minute" (429 + `Retry-After`), a prefix cache of 16-token
blocks (so `cached_tokens` means something), heavy-tailed output lengths, tool calls whose arguments stream as
fragments, optional reasoning tokens, a scripted leak of a stand-in secret when asked (for the guardrail
notebook; harmless by construction), and faults you can switch on over HTTP (`POST /admin/fault`): 503/529,
429, a stall before the first byte, an error mid-stream, a connection reset mid-stream. `/metrics` uses vLLM's
names so the gateway's ledger can be reconciled against it.

Every number it produces is **simulated**: timings come from sleeps (`ttft_s`, `prefill_s_per_token`, `itl_s`),
token counts from a word tokenizer, not a model's. Responses carry `x-gwlab-simulated: true`.

    fake = FakeProvider(FakeSpec(name="acme", dialect="openai")); url = await fake.start()     # on a loop
    python -m gwlab fake --name acme --dialect openai --port 8101                              # a process
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import math
import random
import re
import time
import uuid
import zlib
from dataclasses import dataclass, field

from aiohttp import web

from .promtext import Registry

SIMULATED = {"x-gwlab-simulated": "true"}
LEAK_TRIGGER = re.compile(r"print (the |your )?(api )?key", re.I)
STAND_IN_SECRET = "sk-FAKEFAKEFAKEFAKE0000"            # matches the output screener; not a credential anywhere
_WORD = re.compile(r"\w+|[^\w\s]")
VOCAB = ("the gateway routes each request to a target and meters every token as it streams while the engine "
         "batches work on the accelerator so latency stays low and cost stays visible per tenant").split()


def tokenize(text: str) -> list[int]:
    """The fake's tokenizer: words and punctuation, each mapped to a stable id (crc32). Not a model's."""
    return [zlib.crc32(w.encode()) & 0xFFFFFF for w in _WORD.findall(text)]


def prompt_ids(messages: list, system: str | None = None) -> list[int]:
    ids = []
    if system:
        ids += [1] + tokenize(system)
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(str(p.get("text") or p.get("content") or json.dumps(p.get("input", ""))) for p in content
                               if isinstance(p, dict))
        ids += [2 + ("system", "user", "assistant", "tool").index(m.get("role", "user")) if m.get("role") in
                ("system", "user", "assistant", "tool") else 7] + tokenize(str(content or ""))
        for tc in m.get("tool_calls") or []:
            ids += tokenize(json.dumps(tc.get("function", {})))
    return ids


@dataclass
class FakeSpec:
    name: str = "acme"
    dialect: str = "openai"                       # openai | anthropic
    models: dict = field(default_factory=lambda: {"fast-1": {"context_window": 8192},
                                                  "strong-1": {"context_window": 32768}})
    keys: tuple = ("acme-key-1",)                 # API keys it accepts (two during a rotation)
    ttft_s: float = 0.03                          # base latency to the first token
    prefill_s_per_token: float = 2e-5             # plus this per *uncached* prompt token
    itl_s: float = 0.006                          # between output chunks
    output_tokens: int | None = None              # fixed output length (else lognormal below)
    output_median: int = 40
    output_sigma: float = 0.8
    output_max: int = 4096
    rpm: int = 0                                  # provider limits over a sliding minute (0 = none)
    tpm: int = 0
    minute_s: float = 60.0
    block_size: int = 16
    cache_blocks: int = 8192
    fail_rate: float = 0.0                        # seeded share of requests answered 503/529 before the first byte
    error_code_style: str = "vllm"                # openai dialect: vllm (int codes) | openai (string codes)
    seed: int = 0
    time_scale: float = 1.0

    @classmethod
    def anthropic(cls, name: str = "bolt", **kw) -> "FakeSpec":
        kw.setdefault("models", {"claude-haiku-4-5": {"context_window": 200000}})
        kw.setdefault("keys", ("bolt-key-1",))
        return cls(name=name, dialect="anthropic", **kw)


@dataclass
class Fault:
    mode: str = "ok"             # ok | 503 | 429 | timeout | midstream | reset
    until: float = 0.0           # monotonic deadline (0 = use `remaining`)
    remaining: int = 0           # requests left to fail (when until == 0)
    after_chunks: int = 3        # midstream / reset: after this many content chunks
    stall_s: float = 30.0        # timeout: how long to hold the request before answering

    def active(self, now: float) -> bool:
        if self.mode == "ok":
            return False
        return now < self.until if self.until else self.remaining > 0


class _PrefixCache:
    """Chained block hashes with LRU eviction; `cache_salt` enters the first block only (vLLM's rule)."""

    def __init__(self, block_size: int, capacity: int):
        self.B, self.capacity = block_size, capacity
        self.blocks: collections.OrderedDict = collections.OrderedDict()

    def hashes(self, ids: list, salt: str | None) -> list[str]:
        out, parent = [], "root"
        for i in range(len(ids) // self.B):
            extra = salt if (i == 0 and salt) else ""
            parent = hashlib.sha256(f"{parent}|{extra}|{ids[i * self.B:(i + 1) * self.B]}".encode()).hexdigest()[:16]
            out.append(parent)
        return out

    def lookup(self, ids: list, salt: str | None) -> int:
        """Cached tokens: full blocks found in order, capped at (len − 1) // B blocks (the last token is computed)."""
        hs = self.hashes(ids, salt)[: max(0, (len(ids) - 1) // self.B)]
        hit = 0
        for h in hs:
            if h not in self.blocks:
                break
            self.blocks.move_to_end(h)
            hit += 1
        return hit * self.B

    def insert(self, ids: list, salt: str | None) -> None:
        for h in self.hashes(ids, salt):
            self.blocks[h] = True
            self.blocks.move_to_end(h)
        while len(self.blocks) > self.capacity:
            self.blocks.popitem(last=False)


class FakeProvider:
    def __init__(self, spec: FakeSpec | None = None, host: str = "127.0.0.1", port: int = 0):
        self.spec = spec or FakeSpec()
        self.host, self.port = host, port
        self.keys = set(self.spec.keys)
        self.rng = random.Random(self.spec.seed)
        self.cache = _PrefixCache(self.spec.block_size, self.spec.cache_blocks)
        self.fault = Fault()
        self.window: collections.deque = collections.deque()      # (t, requests, tokens) for the provider's limits
        self.running = 0
        self.stats = collections.Counter()
        self.request_log: collections.deque = collections.deque(maxlen=500)
        self._runner = None
        self._init_metrics()

    # ------------------------------------------------------------------------------------------ metrics
    def _init_metrics(self):
        r = self.registry = Registry()
        lab = ["model_name"]
        self.m_prompt = r.counter("vllm:prompt_tokens", "Number of prefill tokens processed.", lab)
        self.m_gen = r.counter("vllm:generation_tokens", "Number of generation tokens processed.", lab)
        self.m_pq = r.counter("vllm:prefix_cache_queries", "Prefix cache queries, in tokens.", lab)
        self.m_ph = r.counter("vllm:prefix_cache_hits", "Prefix cache hits, in tokens.", lab)
        self.m_ok = r.counter("vllm:request_success", "Successfully processed requests.", lab + ["finished_reason"])
        self.g_running = r.gauge("vllm:num_requests_running", "Requests in flight.", lab)
        self.m_http = r.counter("fake_http_responses", "HTTP responses by status (simulated provider).", ["status"])

    def metrics_text(self) -> str:
        self.g_running.labels(model_name=self.spec.name).set(self.running)
        return "# simulated provider: " + self.spec.name + "\n" + self.registry.render()

    # ------------------------------------------------------------------------------------------ lifecycle
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def app(self) -> web.Application:
        app = web.Application(client_max_size=32 * 1024 * 1024)
        path = "/v1/chat/completions" if self.spec.dialect == "openai" else "/v1/messages"
        app.router.add_post(path, self.handle)
        app.router.add_get("/v1/models", self.models)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_get("/health", self.health)
        app.router.add_post("/admin/fault", self.set_fault)
        app.router.add_post("/admin/keys", self.set_keys)
        app.router.add_get("/admin/stats", self.admin_stats)
        return app

    async def health(self, request):
        return web.json_response({"status": "ok", "simulated": True})

    async def admin_stats(self, request):
        return web.json_response(dict(self.stats))

    async def start(self) -> str:
        self._runner = web.AppRunner(self.app(), access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return self.url

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    def serve_forever(self) -> None:
        print(f"fake provider {self.spec.name!r} ({self.spec.dialect} dialect, SIMULATED) on http://{self.host}:{self.port}")
        web.run_app(self.app(), host=self.host, port=self.port, access_log=None, print=None)

    # ------------------------------------------------------------------------------------------ admin
    async def set_fault(self, request):
        d = await request.json()
        now = time.monotonic()
        self.fault = Fault(mode=d.get("mode", "ok"), until=now + float(d["for_s"]) if d.get("for_s") else 0.0,
                           remaining=int(d.get("count", 0)), after_chunks=int(d.get("after_chunks", 3)),
                           stall_s=float(d.get("stall_s", 30.0)))
        return web.json_response({"fault": self.fault.mode, "simulated": True})

    async def set_keys(self, request):
        """Key rotation at the provider: {"add": "...", "remove": "..."}."""
        d = await request.json()
        if d.get("add"):
            self.keys.add(d["add"])
        if d.get("remove"):
            self.keys.discard(d["remove"])
        return web.json_response({"keys": len(self.keys)})

    async def models(self, request):
        return web.json_response({"object": "list", "data": [
            {"id": m, "object": "model", "owned_by": f"{self.spec.name} (simulated)"} for m in self.spec.models]},
            headers=SIMULATED)

    async def metrics(self, request):
        return web.Response(text=self.metrics_text(), content_type="text/plain")

    # ------------------------------------------------------------------------------------------ errors
    def _error(self, status: int, message: str, etype: str | None = None, code: str | None = None,
               headers: dict | None = None) -> web.Response:
        self.m_http.labels(status=str(status)).inc()
        self.stats[f"http_{status}"] += 1
        if self.spec.dialect == "anthropic":
            etype = etype or {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error",
                              429: "rate_limit_error", 529: "overloaded_error"}.get(status, "api_error")
            body = {"type": "error", "error": {"type": etype, "message": message}}
        elif self.spec.error_code_style == "vllm":
            body = {"error": {"message": message, "type": etype or "BadRequestError", "param": None, "code": status}}
        else:
            body = {"error": {"message": message, "type": etype or "invalid_request_error", "param": None,
                              "code": code or {401: "invalid_api_key", 404: "model_not_found", 429: "rate_limit_exceeded",
                                               503: "server_is_overloaded"}.get(status, "bad_request")}}
        return web.json_response(body, status=status, headers={**SIMULATED, **(headers or {})})

    def _limits(self, now: float, prompt_tokens: int) -> tuple[bool, float, dict]:
        s = self.spec
        while self.window and self.window[0][0] < now - s.minute_s:
            self.window.popleft()
        reqs = sum(w[1] for w in self.window)
        toks = sum(w[2] for w in self.window)
        headers = {}
        if s.rpm:
            headers |= {"x-ratelimit-limit-requests": str(s.rpm), "x-ratelimit-remaining-requests": str(max(0, s.rpm - reqs))}
        if s.tpm:
            headers |= {"x-ratelimit-limit-tokens": str(s.tpm), "x-ratelimit-remaining-tokens": str(max(0, s.tpm - toks))}
        if (s.rpm and reqs >= s.rpm) or (s.tpm and toks >= s.tpm):
            retry = (self.window[0][0] + s.minute_s - now) if self.window else s.minute_s
            return False, max(retry, 0.001), headers
        return True, 0.0, headers

    # ------------------------------------------------------------------------------------------ the request
    async def handle(self, request):
        s, now = self.spec, time.monotonic()
        self.stats["requests"] += 1
        if s.dialect == "openai":
            scheme, _, key = request.headers.get("Authorization", "").partition(" ")
            if scheme.lower() != "bearer" or key not in self.keys:
                return self._error(401, "Incorrect API key provided", "AuthenticationError", "invalid_api_key")
        else:
            if request.headers.get("x-api-key") not in self.keys:
                return self._error(401, "invalid x-api-key")
            if not request.headers.get("anthropic-version"):
                return self._error(400, "anthropic-version header is required")
        try:
            body = await request.json()
        except ValueError:
            return self._error(400, "request body is not JSON")
        model = body.get("model")
        if model not in s.models:
            return self._error(404, f"The model `{model}` does not exist.", "NotFoundError", "model_not_found")
        stream = bool(body.get("stream"))
        if s.dialect == "openai" and body.get("stream_options") is not None and not stream:
            return self._error(400, "Stream options can only be defined when `stream=True`.")
        salt = body.get("cache_salt") if s.dialect == "openai" else None
        if salt is not None and not (isinstance(salt, str) and 1 <= len(salt) <= 128 and not any(c in salt for c in "@/\\\x00")):
            return self._error(400, "Parameter 'cache_salt' must be a non-empty string of at most 128 characters "
                                    "without '@', '/', '\\' or NUL.")
        if s.dialect == "anthropic" and not isinstance(body.get("max_tokens"), int):
            return self._error(400, "max_tokens: Field required")
        ids = prompt_ids(body.get("messages") or [], body.get("system"))
        cap = body.get("max_completion_tokens") or body.get("max_tokens") or s.output_max
        ctx = s.models[model].get("context_window", 8192)
        if len(ids) + min(cap, ctx) > ctx and len(ids) >= ctx:
            return self._error(400, f"This model's maximum context length is {ctx} tokens. However, you requested "
                                    f"{len(ids)} tokens in the messages.")
        # faults before the first byte
        f = self.fault
        if f.active(now) and f.mode in ("503", "429", "timeout"):
            if not f.until:
                f.remaining -= 1
            if f.mode == "timeout":
                self.stats["stalled"] += 1
                await asyncio.sleep(f.stall_s)
                return self._error(504, "stalled (simulated)")
            if f.mode == "429":
                return self._error(429, "Rate limit reached (simulated outage).", code="slow_down",
                                   headers={"Retry-After": "1"})
            return self._error(529 if s.dialect == "anthropic" else 503, "Overloaded (simulated outage).",
                               "overloaded_error" if s.dialect == "anthropic" else "ServiceUnavailableError")
        if s.fail_rate and self.rng.random() < s.fail_rate:
            return self._error(529 if s.dialect == "anthropic" else 503, "Overloaded (simulated random failure).",
                               "overloaded_error" if s.dialect == "anthropic" else "ServiceUnavailableError")
        ok, retry, rl_headers = self._limits(now, len(ids))
        if not ok:
            self.stats["rate_limited"] += 1
            return self._error(429, f"Rate limit reached for {s.name} (simulated).", "RateLimitError", "rate_limit_exceeded",
                               headers={**rl_headers, "Retry-After": str(max(1, math.ceil(retry)))})
        entry = [now, 1, len(ids)]
        self.window.append(entry)
        return await self._generate(request, body, ids, salt, int(cap), stream, rl_headers, entry)

    def _plan_output(self, body: dict, cap: int) -> tuple[list, list, dict | None, str]:
        """(reasoning words, answer words, tool call or None, finish_reason) for this request."""
        s = self.spec
        text = ""
        for m in reversed(body.get("messages") or []):
            if m.get("role") == "user":
                c = m.get("content")
                text = c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or [] if isinstance(p, dict))
                break
        rng = random.Random(zlib.crc32(json.dumps(body.get("messages"), sort_keys=True).encode()) ^ self.rng.getrandbits(16))
        n = s.output_tokens if s.output_tokens is not None else min(
            s.output_max, max(1, int(round(s.output_median * math.exp(s.output_sigma * rng.gauss(0, 1))))))
        tools = body.get("tools") if body.get("tool_choice") not in ("none", {"type": "none"}) else None
        if tools:
            t = tools[0]
            name = t["function"]["name"] if "function" in t else t["name"]
            args = json.dumps({"query": " ".join(text.split()[:6])})
            return [], [], {"name": name, "arguments": args}, "tool_calls"
        reasoning = []
        if s.models[body["model"]].get("thinking"):
            reasoning = [VOCAB[(i * 7) % len(VOCAB)] for i in range(max(1, n // 2))]
        words = ["Answer:"] + [VOCAB[(i + len(text)) % len(VOCAB)] for i in range(n - 1)]
        if LEAK_TRIGGER.search(text):
            k = min(len(words) - 1, 5)
            words[k:k] = ["the", "key", "is", STAND_IN_SECRET]
            words = words[:max(n, 10)]
        finish = "length" if len(reasoning) + len(words) > cap else "stop"
        budget = cap
        reasoning = reasoning[:budget]
        words = words[:max(0, budget - len(reasoning))]
        return reasoning, words, None, finish

    async def _generate(self, request, body, ids, salt, cap, stream, rl_headers, entry):
        s = self.spec
        cached = self.cache.lookup(ids, salt)
        self.m_pq.labels(model_name=s.name).inc(len(ids))
        self.m_ph.labels(model_name=s.name).inc(cached)
        self.m_prompt.labels(model_name=s.name).inc(len(ids))
        reasoning, words, call, finish = self._plan_output(body, cap)
        ttft = (s.ttft_s + s.prefill_s_per_token * (len(ids) - cached)) * s.time_scale
        itl = s.itl_s * s.time_scale
        self.running += 1
        generated = 0
        rid = uuid.uuid4().hex[:16]
        state = {"generated": 0}

        def usage(n_out: int, n_reason: int) -> dict:
            if s.dialect == "anthropic":
                return {"input_tokens": len(ids) - cached, "cache_read_input_tokens": cached,
                        "cache_creation_input_tokens": 0, "output_tokens": n_out,
                        "output_tokens_details": {"thinking_tokens": n_reason}}
            u = {"prompt_tokens": len(ids), "completion_tokens": n_out, "total_tokens": len(ids) + n_out,
                 "prompt_tokens_details": {"cached_tokens": cached}}
            if n_reason:
                u["completion_tokens_details"] = {"reasoning_tokens": n_reason}
            return u

        pieces: list = []                 # (kind, text) in stream order
        pieces += [("reasoning", w + " ") for w in reasoning]
        pieces += [("content", (w if i == 0 else " " + w)) for i, w in enumerate(words)]
        if call:
            a = call["arguments"]
            cut = [0, len(a) // 3, 2 * len(a) // 3, len(a)]
            pieces += [("tool_args", a[cut[i]:cut[i + 1]]) for i in range(3)]
        n_tool = len(tokenize(call["arguments"])) if call else 0
        n_out = len(reasoning) + len(words) + n_tool
        self.request_log.append({"id": rid, "prompt_tokens": len(ids), "cached": cached, "output": n_out,
                                 "salt": bool(salt), "stream": stream, "t": time.time()})
        try:
            await asyncio.sleep(ttft)
            if not stream:
                await asyncio.sleep(itl * max(0, len(pieces) - 1))
                state["generated"] = n_out
                self.cache.insert(ids, salt)
                self._finish(finish, n_out, entry)
                return web.json_response(self._complete_body(body, rid, reasoning, words, call, finish, usage(n_out, len(reasoning))),
                                         headers={**SIMULATED, **rl_headers})
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                                               **SIMULATED, **rl_headers})
            await resp.prepare(request)
            self.cache.insert(ids, salt)          # blocks are published when prefill is scheduled (serving-engine §5)
            f = self.fault
            fault_mid = f.active(time.monotonic()) and f.mode in ("midstream", "reset")
            if fault_mid and not f.until:
                f.remaining -= 1
            include_usage = s.dialect == "openai" and bool((body.get("stream_options") or {}).get("include_usage"))
            await self._stream(request, resp, body, rid, pieces, call, finish, n_out, len(reasoning), usage, itl,
                               include_usage, fault_mid, state, entry)
            if state.get("aborted"):
                return resp
            self._finish(finish, n_out, entry)
            await resp.write_eof()
            return resp
        except (ConnectionResetError, asyncio.CancelledError) as e:
            self.stats["client_disconnects"] += 1
            if not state.get("aborted"):
                self._finish("abort", state["generated"], entry)  # the engine frees the request; tokens so far count
            if isinstance(e, asyncio.CancelledError):
                raise
            return web.Response(status=499)                         # nobody is listening; nothing to log
        finally:
            self.running -= 1

    def _finish(self, reason: str, n_out: int, entry) -> None:
        if entry is not None:
            entry[2] += n_out                      # the provider's TPM window counts output as it is produced
        self.m_gen.labels(model_name=self.spec.name).inc(n_out)
        self.m_ok.labels(model_name=self.spec.name, finished_reason=reason).inc()
        self.m_http.labels(status="200").inc()

    def _complete_body(self, body, rid, reasoning, words, call, finish, usage):
        s = self.spec
        text = "".join(w if i == 0 else " " + w for i, w in enumerate(words))
        if s.dialect == "anthropic":
            content = [{"type": "text", "text": text}] if text else []
            if call:
                content.append({"type": "tool_use", "id": "toolu_" + rid, "name": call["name"],
                                "input": json.loads(call["arguments"])})
            return {"id": "msg_" + rid, "type": "message", "role": "assistant", "model": body["model"], "content": content,
                    "stop_reason": {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}[finish],
                    "stop_sequence": None, "usage": usage}
        msg = {"role": "assistant", "content": text or None}
        if reasoning:
            msg["reasoning"] = " ".join(reasoning)
        if call:
            msg["tool_calls"] = [{"id": "call_" + rid, "type": "function", "function": call}]
        return {"id": "chatcmpl-" + rid, "object": "chat.completion", "created": int(time.time()), "model": body["model"],
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}

    async def _stream(self, request, resp, body, rid, pieces, call, finish, n_out, n_reason, usage, itl,
                      include_usage, fault_mid, state, entry):
        s, f = self.spec, self.fault
        model = body["model"]
        created = int(time.time())

        async def send(obj, event=None):
            data = json.dumps(obj, separators=(",", ":")).encode()
            await resp.write((f"event: {event}\n".encode() if event else b"") + b"data: " + data + b"\n\n")

        def chunk(delta, fin=None):
            c = {"id": "chatcmpl-" + rid, "object": "chat.completion.chunk", "created": created, "model": model,
                 "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": fin}]}
            if include_usage:
                c["usage"] = None          # OpenAI: with include_usage every other chunk carries usage: null
            return c

        if s.dialect == "anthropic":
            await send({"type": "message_start", "message": {"id": "msg_" + rid, "type": "message", "role": "assistant",
                                                             "model": model, "content": [], "stop_reason": None,
                                                             "usage": usage(1, 0)}}, "message_start")
        else:
            await send(chunk({"role": "assistant", "content": ""}))
        n_content, gen = 0, 0
        block_kind, block_index = None, -1            # anthropic: the open content block
        tool_started = False
        for kind, text in pieces:
            if fault_mid and n_content >= f.after_chunks:
                self.stats["midstream_faults"] += 1
                state["generated"], state["aborted"] = gen, True
                self._finish("abort", gen, entry)
                if f.mode == "reset":                  # drop the TCP connection without another byte
                    if request.transport is not None:
                        request.transport.close()
                    return
                if s.dialect == "anthropic":
                    await send({"type": "error", "error": {"type": "overloaded_error",
                                                           "message": "Overloaded (simulated, mid-stream)"}}, "error")
                else:                                   # vLLM: an error object inside the 200, then [DONE]
                    await send({"error": {"message": "EngineCore encountered an issue (simulated)",
                                          "type": "InternalServerError", "param": None, "code": 500}})
                    await resp.write(b"data: [DONE]\n\n")
                await resp.write_eof()
                return
            await asyncio.sleep(itl)
            if kind != "tool_args":
                gen += 1
            state["generated"] = gen
            if s.dialect == "anthropic":
                want = {"reasoning": "thinking", "content": "text", "tool_args": "tool_use"}[kind]
                if block_kind != want:
                    if block_kind is not None:
                        await send({"type": "content_block_stop", "index": block_index}, "content_block_stop")
                    block_index += 1
                    block_kind = want
                    cb = {"thinking": {"type": "thinking", "thinking": ""}, "text": {"type": "text", "text": ""},
                          "tool_use": {"type": "tool_use", "id": "toolu_" + rid, "name": call["name"] if call else "",
                                       "input": {}}}[want]
                    await send({"type": "content_block_start", "index": block_index, "content_block": cb},
                               "content_block_start")
                delta = {"reasoning": {"type": "thinking_delta", "thinking": text},
                         "content": {"type": "text_delta", "text": text},
                         "tool_args": {"type": "input_json_delta", "partial_json": text}}[kind]
                await send({"type": "content_block_delta", "index": block_index, "delta": delta}, "content_block_delta")
            else:
                if kind == "reasoning":
                    await send(chunk({"reasoning": text}))
                elif kind == "content":
                    await send(chunk({"content": text}))
                else:
                    tc = {"index": 0, "function": {"arguments": text}}
                    if not tool_started:
                        tc = {"index": 0, "id": "call_" + rid, "type": "function",
                              "function": {"name": call["name"], "arguments": text}}
                    tool_started = True
                    await send(chunk({"tool_calls": [tc]}))
            if kind != "reasoning":
                n_content += 1
        state["generated"] = n_out
        if s.dialect == "anthropic":
            if block_kind is not None:
                await send({"type": "content_block_stop", "index": block_index}, "content_block_stop")
            stop = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}[finish]
            await send({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                        "usage": {"output_tokens": n_out, "output_tokens_details": {"thinking_tokens": n_reason}}},
                       "message_delta")
            await send({"type": "message_stop"}, "message_stop")
        else:
            await send(chunk({}, finish))
            if include_usage:
                await send({"id": "chatcmpl-" + rid, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [], "usage": usage(n_out, n_reason)})
            await resp.write(b"data: [DONE]\n\n")
