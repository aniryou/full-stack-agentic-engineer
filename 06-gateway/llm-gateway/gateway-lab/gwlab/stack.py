"""The whole lab stack in one Python process: fake providers + the gateway, on free ports.

The one idea: to learn a gateway you need real HTTP between real processes' worth of state — keys, buckets,
breakers, a cache, a ledger — without Docker. This starts everything on a private asyncio loop in a background
thread, so the same code runs from a notebook (where a loop is already running), a test or a script:

    with LocalStack() as s:                          # acme (OpenAI dialect) + bolt (Anthropic dialect) + gateway
        key = s.issue_key("team-a")
        r = s.chat(key, "Hello", stream=True)       # through the gateway, over HTTP
        s.ledger()[-1]                               # the row it wrote

T1: pass `upstreams={"local": "http://127.0.0.1:8000"}` with `config="vllm"` to put a real `vllm serve` behind
the gateway (deploy/any-gpu); the fakes named in the config but not in `upstreams` still start (the fallback).
"""
from __future__ import annotations

import asyncio
import threading

from . import client as _client
from .fakes import FakeProvider, FakeSpec
from .gateway.config import load_config
from .gateway.server import Gateway

DEFAULT_FAKES = {"acme": FakeSpec(name="acme", dialect="openai"), "bolt": FakeSpec.anthropic("bolt")}


class LocalStack:
    def __init__(self, config="lab", fakes: dict | None = None, upstreams: dict | None = None,
                 overrides: dict | None = None, gateway_port: int = 0, fake_ports: dict | None = None, mcp=None):
        self.config_src, self.overrides = config, overrides or {}
        self.fake_specs = dict(DEFAULT_FAKES if fakes is None else fakes)
        self.upstreams = dict(upstreams or {})
        self.gateway_port, self.fake_ports = gateway_port, dict(fake_ports or {})
        self.fakes: dict[str, FakeProvider] = {}
        self.mcp_options = mcp                     # an AsOptions (or True) starts the fake AS + MCP server too
        self.mcp = None
        self.gateway: Gateway | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "LocalStack":
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run():
            asyncio.set_event_loop(self.loop)
            self.loop.call_soon(ready.set)
            self.loop.run_forever()

        self._thread = threading.Thread(target=run, name="gwlab-stack", daemon=True)
        self._thread.start()
        ready.wait()
        try:
            self.run(self._astart())
        except BaseException:
            self.stop()
            raise
        return self

    async def _astart(self):
        cfg0 = load_config(self.config_src, self.overrides)
        prov_over = {}
        for name in cfg0.providers:
            if name in self.upstreams:
                prov_over[name] = {"base_url": self.upstreams[name]}
            elif name in self.fake_specs:
                fake = FakeProvider(self.fake_specs[name], port=self.fake_ports.get(name, 0))
                await fake.start()
                self.fakes[name] = fake
                prov_over[name] = {"base_url": fake.url, "key": self.fake_specs[name].keys[0]}
        if self.mcp_options:
            from .mcp.server import AsOptions, McpServers
            self.mcp = McpServers(self.mcp_options if isinstance(self.mcp_options, AsOptions) else AsOptions())
            await self.mcp.start()
            self.overrides = {**self.overrides, "mcp_servers": {"notes": self.mcp.resource}}
        merged = {**self.overrides, "providers": {**prov_over, **{k: {**prov_over.get(k, {}), **v} for k, v in
                                                                  (self.overrides.get("providers") or {}).items()}}}
        self.cfg = load_config(self.config_src, merged)
        self.gateway = Gateway(self.cfg)
        await self.gateway.start(port=self.gateway_port)

    async def _astop(self):
        if self.gateway:
            await self.gateway.stop()
        if self.mcp:
            await self.mcp.stop()
        for f in self.fakes.values():
            await f.stop()

    def stop(self) -> None:
        if self.loop is None:
            return
        try:
            self.run(self._astop(), timeout=10)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(timeout=5)
            self.loop.close()
            self.loop = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ------------------------------------------------------------------ helpers
    def run(self, coro, timeout: float | None = 120):
        """Run a coroutine on the stack's loop from any thread and return its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def call(self, fn, *args):
        """Run a plain function on the loop thread (safe access to live state)."""
        async def _f():
            return fn(*args)
        return self.run(_f())

    @property
    def url(self) -> str:
        return self.gateway.url

    @property
    def admin_token(self) -> str:
        return self.cfg.admin_token

    def fake_url(self, name: str) -> str:
        return self.fakes[name].url

    def issue_key(self, tenant: str, **kw) -> str:
        """A virtual key via the admin API (over HTTP, as an operator would)."""
        r = _client.post_json(self.url + "/admin/keys", {"tenant": tenant, **kw}, token=self.admin_token)
        assert r.status == 200, r.json
        return r.json["key"]

    def chat(self, key: str, content: str | list, *, model: str = "chat", stream: bool = False, **kw) -> "_client.Result":
        """One chat completion through the gateway. `content` is the user message, or a full messages list."""
        messages = content if isinstance(content, list) else [{"role": "user", "content": content}]
        return _client.chat(self.url, key, {"model": model, "messages": messages, **kw}, stream=stream)

    def fault(self, provider: str, mode: str, **kw) -> dict:
        """Switch on a fault at a fake provider: 503 | 429 | timeout | midstream | reset | ok."""
        return _client.post_json(self.fake_url(provider) + "/admin/fault", {"mode": mode, **kw}).json

    def ledger(self, **where) -> list[dict]:
        rows = self.gateway.store.query("SELECT * FROM ledger ORDER BY ts")
        return [r for r in rows if all(r.get(k) == v for k, v in where.items())]

    def decisions(self) -> list[dict]:
        return self.call(lambda: list(self.gateway.decisions))

    def last_decision(self) -> dict | None:
        d = self.decisions()
        return d[-1] if d else None

    def spans(self, trace_id: str | None = None) -> list:
        return self.call(lambda: [s for s in self.gateway.tracer.spans if trace_id is None or s.trace_id == trace_id])

    def metrics(self) -> str:
        return _client.get_text(self.url + "/metrics")

    def provider_metrics(self, name: str) -> str:
        return _client.get_text(self.fake_url(name) + "/metrics")
