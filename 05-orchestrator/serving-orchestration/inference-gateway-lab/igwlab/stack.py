"""A whole serving stack in one Python process: N fake backends + the router, on free ports.

The one idea: to learn routing you need *real* requests hitting *stateful* replicas. This
starts everything on a private asyncio event loop in a background thread, so the same code
works from a notebook (where a loop is already running), a test, or a script:

    with LocalStack(n_backends=3, config="default-weighted") as s:
        print(s.router_url)            # http://127.0.0.1:<port>  (OpenAI-compatible)
        s.router.decisions[-1].table() # why the last request went where it went

Each LocalStack is fresh (empty caches, zeroed counters), which is what you want when
comparing two routing policies on the same workload.
"""
from __future__ import annotations

import asyncio
import threading

from .fakebackend import EngineProfile, FakeBackend
from .router.datalayer import Endpoint
from .router.server import Router, RouterSettings

__all__ = ["LocalStack"]


class LocalStack:
    def __init__(self, n_backends: int = 3, config="default-weighted", profile: EngineProfile | None = None,
                 model: str = "lab/llm", settings: RouterSettings | None = None, lora_modules=(),
                 names=None):
        self.n, self.config, self.model = n_backends, config, model
        self.profile = profile or EngineProfile()
        self.settings = settings or RouterSettings()
        self.lora_modules = list(lora_modules)
        self.names = list(names) if names else [chr(ord("a") + i) for i in range(n_backends)]
        self.backends: list[FakeBackend] = []
        self.router: Router | None = None
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

        self._thread = threading.Thread(target=run, name="igwlab-stack", daemon=True)
        self._thread.start()
        ready.wait()
        self.run(self._astart())
        return self

    async def _astart(self):
        for name in self.names:
            b = FakeBackend(name, self.profile, self.model, self.lora_modules)
            await b.start()
            self.backends.append(b)
        eps = [Endpoint(name=b.name, url=b.url) for b in self.backends]
        self.router = Router(self.config, eps, self.settings)
        await self.router.start()

    async def _astop(self):
        if self.router:
            await self.router.stop()
        for b in self.backends:
            await b.stop()

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
        """Run a plain function on the stack's loop thread (safe access to live state)."""
        async def _f():
            return fn(*args)
        return self.run(_f())

    @property
    def router_url(self) -> str:
        return self.router.url

    @property
    def backend_urls(self) -> dict:
        return {b.name: b.url for b in self.backends}

    def backend_metrics(self) -> dict:
        """name -> current /metrics text of each backend (what the router scrapes)."""
        return self.call(lambda: {b.name: b.engine.metrics_text() for b in self.backends})

    def router_metrics(self) -> str:
        return self.call(lambda: (self.router._refresh_gauges(), self.router.registry.render())[1])

    def last_decision(self):
        return self.call(lambda: self.router.decisions[-1] if self.router.decisions else None)
