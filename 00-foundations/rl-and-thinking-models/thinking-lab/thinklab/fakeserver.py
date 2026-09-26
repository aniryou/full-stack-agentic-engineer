"""fakeserver.py — a fake vLLM serving a *thinking* model: ``reasoning`` deltas, heavy-tailed lengths.

One idea: to learn how a thinking model behaves behind an OpenAI-compatible API — the extra
``reasoning`` field, the thinking switch, budgets, the ``max_tokens`` trap, the long decode it puts on
the engine — you need a server that does all of it without a GPU. This one answers like
``vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3`` (v0.30.0 request and response shapes):

* ``POST /v1/chat/completions`` — streaming or not; ``message.reasoning`` / ``delta.reasoning`` (set
  ``reasoning_field="reasoning_content"`` to answer like SGLang or the DeepSeek API);
  ``chat_template_kwargs.enable_thinking`` (Qwen3's hard switch, default on), the ``/think`` and
  ``/no_think`` soft switches, ``reasoning_effort`` (``"none"`` = off, anything else = on — for Qwen3 it
  is a switch, not a length control), ``thinking_token_budget`` (−1 = unlimited), ``include_reasoning``,
  ``max_tokens``/``max_completion_tokens`` (counts reasoning tokens: a small value ends inside the
  thinking with ``content: null`` and ``finish_reason: "length"``), ``n``, ``seed``, and
  ``continue_final_message`` (the second call of Qwen's two-call budget recipe);
  ``usage.completion_tokens_details.reasoning_tokens`` and ``usage.prompt_tokens_details.cached_tokens``.
* ``GET /metrics`` with vLLM's metric names and buckets (no reasoning-specific metric, as in v0.30.0),
  ``/health``, ``/v1/models`` and ``/version`` — which says ``"simulated": true``.

*What* it says comes from :mod:`thinklab.fakemodel` (a simulated model that knows the answers to the
generated eval questions and gets them right with a probability that grows with its thinking);
*when* it says it comes from :mod:`thinklab.engine` run in real time (``time_scale`` < 1 runs faster;
latency metrics are then in simulated seconds). A prefix index over rendered prompts reports
``cached_tokens`` so the multi-turn prefix-cache effect of dropped thinking is visible (it is not
tied to the block pool — a simplification). Everything it reports is **simulated**; every response
carries ``x-thinklab-simulated: true``. The upstream tool with the same purpose (no reasoning
emulation) is ``llm-d-inference-sim``; the 04 lab's ``servelab.fakeserver`` is the non-thinking sibling.
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import traceback
import uuid
from collections import OrderedDict

from aiohttp import web

from . import fakemodel, metrics as M, templates
from .engine import Engine, EngineConfig, Profile, profile as named_profile

SIM = {"x-thinklab-simulated": "true"}


class PrefixIndex:
    """Block-hash prefix index (chained hashes over ``block_size`` toy tokens, LRU-bounded)."""

    def __init__(self, block_size: int = 16, capacity: int = 50_000):
        self.bs, self.cap = block_size, capacity
        self.blocks: OrderedDict = OrderedDict()

    def _hashes(self, toks: list):
        h, out = None, []
        for i in range(0, len(toks) - self.bs + 1, self.bs):
            h = hash((h, tuple(toks[i:i + self.bs])))
            out.append(h)
        return out

    def match(self, toks: list) -> int:
        n = 0
        for h in self._hashes(toks[: max(0, len(toks) - 1)]):   # the last prompt token is always computed
            if h not in self.blocks:
                break
            self.blocks.move_to_end(h)
            n += self.bs
        return n

    def insert(self, toks: list) -> None:
        for h in self._hashes(toks):
            self.blocks[h] = True
            self.blocks.move_to_end(h)
        while len(self.blocks) > self.cap:
            self.blocks.popitem(last=False)


class _Choice:
    """One generation: the pieces to emit, and where the reasoning ends."""

    def __init__(self, sample: fakemodel.Sample, thinking: bool, parser: str | None, max_tokens: int):
        pieces = []                                   # (kind, text); kind: start|reasoning|end|content
        if thinking:
            pieces.append(("start", "<think>\n"))
            pieces += [("reasoning", w + " ") for w in sample.reasoning_tokens]
            if sample.forced_stop:
                pieces += [("reasoning", w + " ") for w in fakemodel.EARLY_STOP.split()]
            pieces.append(("end", "\n</think>\n\n"))
        pieces += [("content", w + (" " if i < len(sample.content_tokens) - 1 else "")) for i, w in enumerate(sample.content_tokens)]
        self.truncated = len(pieces) > max_tokens
        self.pieces = pieces[:max_tokens]
        self.parser = parser
        self.finish_reason = "length" if self.truncated else "stop"
        self.reasoning_count = sum(k == "reasoning" for k, _ in self.pieces)
        self.sample = sample

    def split(self, upto: int | None = None) -> tuple:
        """(reasoning text or None, content text or None) of the first ``upto`` pieces, as a server
        with (or without) a reasoning parser would return them."""
        ps = self.pieces[:upto]
        if self.parser is None:
            return None, "".join(t for _, t in ps) or None
        r = "".join(t for k, t in ps if k == "reasoning").strip() or None
        c = "".join(t for k, t in ps if k == "content") or None
        return r, c

    def generated_text(self) -> str:
        r, c = self.split()
        think = f"<think>\n{r}\n</think>\n\n" if any(k == "start" for k, _ in self.pieces) else ""
        return f"{think}{c or ''}{templates.IM_END}\n"


def _thinking_switch(body: dict, messages: list, default: bool) -> bool:
    ctk = body.get("chat_template_kwargs") or {}
    if "enable_thinking" in ctk:
        return bool(ctk["enable_thinking"])
    effort = body.get("reasoning_effort")
    if effort is not None:
        return effort != "none"            # vLLM: enable_thinking = (reasoning_effort != "none")
    last_user = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
    if isinstance(last_user, str):          # Qwen3 soft switches: the latest instruction wins
        if "/no_think" in last_user:
            return False
        if "/think" in last_user:
            return True
    return default


class FakeServer:
    """The simulated thinking model behind an aiohttp app. ``start()`` runs it in a daemon thread and
    returns the base URL; also a context manager."""

    def __init__(self, profile: Profile | str = "t4-qwen3-0.6b", card: fakemodel.ModelCard = fakemodel.SMALL, *,
                 config: EngineConfig | None = None, reasoning_parser: str | None = "qwen3",
                 reasoning_field: str = "reasoning", default_enable_thinking: bool = True,
                 model: str | None = None, host: str = "127.0.0.1", port: int = 0, time_scale: float = 1.0):
        self.p = named_profile(profile) if isinstance(profile, str) else profile
        self.engine = Engine(self.p, config, keep_itl=False)
        self.card, self.parser, self.field = card, reasoning_parser, reasoning_field
        self.default_thinking = default_enable_thinking
        self.model = model or self.p.model
        self.host, self.port, self.time_scale = host, port, time_scale
        self.prefix = PrefixIndex(self.p.block_size, self.p.num_blocks)
        self.reg = M.Registry({"model_name": self.model, "engine": "0"})
        self.reg.gauge(M.CACHE_CONFIG_INFO, "Information of the LLMEngine CacheConfig", 1,
                       {"block_size": str(self.p.block_size), "num_gpu_blocks": str(self.p.num_blocks),
                        "simulated": "true"})
        self._streams: dict = {}
        self._counter = 0
        self._loop = self._thread = self._stopping = self._runner = None
        self._ready = threading.Event()
        self._last_preemptions = 0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def vnow(self) -> float:
        """The engine's clock in simulated seconds (real seconds when time_scale == 1)."""
        return time.perf_counter() / self.time_scale

    # -- HTTP -----------------------------------------------------------------------------------
    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/health", lambda r: web.Response(status=200, headers=SIM))
        app.router.add_get("/version", lambda r: web.json_response(
            {"version": "thinklab-fakeserver", "simulated": True, "time_scale": self.time_scale,
             "reasoning_parser": self.parser, "profile": self.p.name}))
        app.router.add_get("/v1/models", self._models)
        app.router.add_get("/metrics", self._metrics)
        app.router.add_post("/v1/chat/completions", self._chat)
        return app

    async def _models(self, request):
        return web.json_response({"object": "list", "data": [{
            "id": self.model, "object": "model", "created": int(time.time()), "owned_by": "thinklab (simulated)",
            "root": self.model, "max_model_len": self.p.max_model_len}]}, headers=SIM)

    async def _metrics(self, request):
        e = self.engine
        self.reg.gauge(M.RUNNING, "Number of requests in model execution batches.", len(e.running))
        self.reg.gauge(M.WAITING, "Number of requests waiting to be processed.", len(e.waiting))
        self.reg.gauge(M.KV_USAGE, "KV-cache usage. 1 means 100 percent usage.", e.usage())
        return web.Response(text=self.reg.render(), content_type="text/plain", headers=SIM)

    async def _chat(self, request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error(400, "body is not JSON")
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return _error(400, "messages must be a non-empty list")
        stream, n = bool(body.get("stream")), int(body.get("n") or 1)
        if stream and n != 1:
            return _error(400, "this fake server streams one choice (n=1)")
        if body.get("continue_final_message") and body.get("add_generation_prompt"):
            return _error(400, "Cannot set both `continue_final_message` and `add_generation_prompt` to True.")
        cont = bool(body.get("continue_final_message")) and msgs[-1].get("role") == "assistant"
        thinking = _thinking_switch(body, msgs, self.default_thinking)
        question = next((m.get("content") or "" for m in reversed(msgs) if m.get("role") == "user"), "")
        prefilled = None
        if cont:
            prefix_text = templates.render_qwen3(msgs[:-1], add_generation_prompt=True) + (msgs[-1].get("content") or "")
            r_part = (msgs[-1].get("content") or "").split("</think>")[0].split("<think>")[-1]
            prefilled = len(templates.tokens(r_part))
            thinking = False                      # the thinking was supplied; only the answer is generated
        else:
            prefix_text = templates.render_qwen3(msgs, enable_thinking=thinking)
        prompt_toks = templates.tokens(prefix_text)
        if len(prompt_toks) >= self.p.max_model_len:
            return _error(400, f"prompt ({len(prompt_toks)} tokens) is longer than max_model_len {self.p.max_model_len}")
        room = self.p.max_model_len - len(prompt_toks)
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
        if max_tokens is not None and int(max_tokens) > room:
            return _error(400, f"max_tokens ({max_tokens}) + prompt ({len(prompt_toks)}) exceeds max_model_len "
                               f"{self.p.max_model_len}")
        max_tokens = int(max_tokens) if max_tokens is not None else room
        budget = body.get("thinking_token_budget")
        if budget is not None and (not isinstance(budget, int) or budget < -1):
            return _error(400, "thinking_token_budget must be a non-negative integer or -1")
        if self.parser is None:
            budget = None                         # budgets need --reasoning-parser (vLLM derives the end token from it)
        seed = body.get("seed")
        if seed is None:
            self._counter += 1
            seed = f"auto-{self._counter}"
        choices = []
        for i in range(n):
            s = fakemodel.generate(self.card, question, thinking=thinking, budget=budget, seed=seed, sample_index=i,
                                   prefilled_think=prefilled)
            choices.append(_Choice(s, thinking, self.parser, max_tokens))
        cached = self.prefix.match(prompt_toks)
        self.reg.counter(M.PREFIX_QUERIES, "Prefix cache queries, in terms of number of queried tokens.", len(prompt_toks) * n)
        self.reg.counter(M.PREFIX_HITS, "Prefix cache hits, in terms of number of cached tokens.", cached * n)
        include_reasoning = body.get("include_reasoning", True)
        rid = "chatcmpl-" + uuid.uuid4().hex[:16]
        created = int(time.time())
        queues = []
        for c in choices:
            q: asyncio.Queue = asyncio.Queue()
            req = self.engine.add(len(prompt_toks), max(1, len(c.pieces)), self.vnow(), c.reasoning_count, cached)
            req.tag = "thinking" if thinking else "direct"
            self._streams[req.rid] = (q, c, prompt_toks)
            queues.append((req, q, c))
        self._wake.set()

        def usage():
            comp = sum(len(c.pieces) for c in choices)
            u = {"prompt_tokens": len(prompt_toks), "total_tokens": len(prompt_toks) + comp, "completion_tokens": comp,
                 "prompt_tokens_details": {"cached_tokens": cached}}
            if self.parser is not None:
                u["completion_tokens_details"] = {"reasoning_tokens": sum(c.reasoning_count for c in choices)}
            return u

        try:
            if not stream:
                out = []
                for i, (req, q, c) in enumerate(queues):
                    while True:
                        idx, done = await q.get()
                        if idx is None:
                            return _error(500, "fake engine failed; see the server's stderr")
                        if done:
                            break
                    r, content = c.split()
                    msg = {"role": "assistant", "content": content, self.field: r if include_reasoning else None,
                           "tool_calls": []}
                    out.append({"index": i, "message": msg, "logprobs": None, "finish_reason": c.finish_reason,
                                "stop_reason": None})
                return web.json_response({"id": rid, "object": "chat.completion", "created": created, "model": self.model,
                                          "choices": out, "usage": usage()}, headers=SIM)
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", **SIM})
            await resp.prepare(request)
            req, q, c = queues[0]
            base = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": self.model}
            first = True
            while True:
                idx, done = await q.get()
                if idx is None:
                    await resp.write(b'data: {"error": "fake engine failed"}\n\n')
                    break
                if first:
                    await _sse(resp, {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                                           "logprobs": None, "finish_reason": None}]})
                    first = False
                kind, text = c.pieces[idx] if idx < len(c.pieces) else ("content", "")
                delta = {}
                if self.parser is None:
                    delta = {"content": text}
                elif kind == "reasoning" and include_reasoning:
                    delta = {self.field: text}
                elif kind == "content":
                    delta = {"content": text}
                if delta or done:
                    await _sse(resp, {**base, "choices": [{"index": 0, "delta": delta, "logprobs": None,
                                                           "finish_reason": c.finish_reason if done else None}]})
                if done:
                    break
            if (body.get("stream_options") or {}).get("include_usage"):
                await _sse(resp, {**base, "choices": [], "usage": usage()})
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        finally:
            for req, _, _ in queues:
                self._streams.pop(req.rid, None)

    # -- the engine loop ---------------------------------------------------------------------------
    async def _run_engine(self):
        e = self.engine
        while True:
            if not e.has_work():
                self._wake.clear()
                await self._wake.wait()
                continue
            plan = e.schedule(self.vnow())
            if not plan["decode"] and not plan["prefill"]:
                await asyncio.sleep(0.001)
                continue
            dt = e.step_time(plan)
            await asyncio.sleep(dt * self.time_scale)
            events = e.commit(plan, self.vnow())
            self._on_step(events)

    def _on_step(self, events) -> None:
        e, reg = self.engine, self.reg
        if e.preemptions > self._last_preemptions:
            reg.counter(M.PREEMPTIONS, "Cumulative number of preemption from the engine.", e.preemptions - self._last_preemptions)
            self._last_preemptions = e.preemptions
        reg.counter(M.GENERATION_TOKENS, "Number of generation tokens processed.", len(events))
        for r, idx, done in events:
            entry = self._streams.get(r.rid)
            if idx == 0:
                reg.observe(M.TTFT, "Histogram of time to first token in seconds.", r.first_token - r.arrival, M.BUCKETS["ttft"])
                reg.counter(M.PROMPT_TOKENS, "Number of prefill tokens processed.", r.prompt_tokens - r.cached_tokens)
            elif r.last_token is not None and getattr(r, "_prev", None) is not None:
                reg.observe(M.ITL, "Histogram of inter-token latency in seconds.", r.last_token - r._prev, M.BUCKETS["itl"])
            r._prev = r.last_token
            if done:
                self._on_finish(r, entry)
            if entry is not None:
                entry[0].put_nowait((idx, done))

    def _on_finish(self, r, entry) -> None:
        reg = self.reg
        n = r.generated
        reason = entry[1].finish_reason if entry else "stop"
        reg.counter(M.REQUEST_SUCCESS, "Count of successfully processed requests.", 1, {"finished_reason": reason})
        reg.observe(M.E2E, "Histogram of e2e request latency in seconds.", r.finished_at - r.arrival, M.BUCKETS["latency"])
        reg.observe(M.QUEUE, "Histogram of time spent in WAITING phase for request.", r.admitted_at - r.arrival, M.BUCKETS["latency"])
        reg.observe(M.TPOT, "Histogram of time_per_output_token_seconds per request.",
                    (r.last_token - r.first_token) / (n - 1) if n > 1 else 0.0, M.BUCKETS["itl"])
        buckets = M.token_buckets(self.p.max_model_len)
        reg.observe(M.REQUEST_PROMPT_TOKENS, "Number of prefill tokens processed.", r.prompt_tokens, buckets)
        reg.observe(M.REQUEST_GENERATION_TOKENS, "Number of generation tokens processed.", n, buckets)
        if entry is not None:
            _, choice, prompt_toks = entry
            self.prefix.insert(prompt_toks + templates.tokens(choice.generated_text()))

    async def _engine_loop(self):
        try:
            await self._run_engine()
        except asyncio.CancelledError:
            raise
        except Exception:                          # a bug must be loud, not a silent hang
            traceback.print_exc()
            for q, _, _ in list(self._streams.values()):
                q.put_nowait((None, True))
            raise

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
        task = asyncio.create_task(self._engine_loop())
        self._ready.set()
        try:
            await self._stopping.wait()
        finally:
            task.cancel()
            await self._runner.cleanup()

    # -- lifecycle -------------------------------------------------------------------------------
    def start(self) -> str:
        def run():
            self._loop = asyncio.new_event_loop()
            self._loop.run_until_complete(self._serve())
            self._loop.close()
        self._thread = threading.Thread(target=run, name="thinklab-fakeserver", daemon=True)
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
        print(f"fake vLLM (simulated thinking model) serving {self.model} on http://{self.host}:{self.port or '<auto>'}")
        print(self.p.describe())
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
                             status=status, headers=SIM)
