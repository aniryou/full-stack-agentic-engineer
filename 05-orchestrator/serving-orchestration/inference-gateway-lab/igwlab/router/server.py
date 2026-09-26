"""The router process: an OpenAI-compatible front door that picks an endpoint per request and
streams the response back untouched.

The one idea: in production the work is split — a proxy (Envoy, or a cloud L7 load balancer
in Gateway mode) carries the bytes and, for every request, asks the Endpoint Picker (EPP)
over the ext-proc gRPC protocol "which pod?"; the EPP answers with the header
`x-gateway-destination-endpoint: <pod-ip:port>`. Here one asyncio process plays both roles so
the whole decision path fits on a screen:

    client --POST /v1/chat/completions--> Router
        1. read body, estimate tokens, look up the InferenceObjective priority (header
           x-llm-d-inference-objective)
        2. admission: a sheddable request (priority < 0) is rejected with 429 when the pool
           is saturated (llm-d's default "legacy" admission, utilization detector)
        3. Scheduler.schedule(): producers -> filters -> weighted scorers -> picker
        4. forward the *unmodified* body to the chosen backend; stream every byte back as it
           arrives (SSE chunks are never re-encoded); record first-byte time and completion
        5. update in-flight counters and the prefix index; export metrics

Its own `/metrics` mirror a subset of the EPP's `llm_d_epp_*` series under an `igw_` prefix.
"""
from __future__ import annotations

import asyncio
import collections
import json
import time
import uuid
from dataclasses import dataclass, field

from ..promtext import Registry
from .config import PickerConfig, load_config
from .datalayer import Datastore, Endpoint, Scraper
from .plugins import RequestCtx
from .scheduler import Scheduler
from .tokens import estimate_tokens

__all__ = ["RouterSettings", "Router", "DESTINATION_HEADER", "OBJECTIVE_HEADER", "DROPPED_REASON_HEADER"]

DESTINATION_HEADER = "x-gateway-destination-endpoint"
OBJECTIVE_HEADER = "x-llm-d-inference-objective"
OLD_OBJECTIVE_HEADER = "x-gateway-inference-objective"
DROPPED_REASON_HEADER = "x-llm-d-request-dropped-reason"
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "host", "content-length"}

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)


@dataclass
class RouterSettings:
    scrape_interval_s: float = 0.05            # llm-d default base tick; the config's metrics-data-source may override
    upstream_timeout_s: float = 600.0          # per-read timeout while streaming
    objectives: dict = field(default_factory=dict)   # InferenceObjective name -> priority (what the CRDs declare)
    queue_depth_threshold: int = 5             # utilization-detector defaults
    kv_cache_util_threshold: float = 0.8
    metrics_staleness_s: float = 0.2
    decisions_kept: int = 256
    seed: int = 0


class Router:
    def __init__(self, config, endpoints, settings: RouterSettings | None = None):
        self.settings = settings or RouterSettings()
        self.config: PickerConfig = config if isinstance(config, PickerConfig) else load_config(config, seed=self.settings.seed)
        self.ds = Datastore(endpoints)
        self.scheduler = Scheduler(self.config, self.ds)
        self.decisions: collections.deque = collections.deque(maxlen=self.settings.decisions_kept)
        self.session = None
        self.scraper: Scraper | None = None
        self._runner = None
        self.url: str | None = None
        self._init_metrics()

    # ------------------------------------------------------------------ metrics
    def _init_metrics(self):
        r = self.registry = Registry()
        lab = ["model_name"]
        self.m_requests = r.counter("igw_request_total", "Requests routed (llm_d_epp_request_total).", lab + ["endpoint"])
        self.m_errors = r.counter("igw_request_error_total", "Errored or rejected requests.", lab + ["error_code"])
        self.m_duration = r.histogram("igw_request_duration_seconds", "End-to-end latency seen by the router.", LATENCY_BUCKETS, lab)
        self.m_ttft = r.histogram("igw_request_ttft_seconds", "Router-observed time to first byte from the backend.", LATENCY_BUCKETS, lab)
        self.m_sched = r.histogram("igw_scheduler_e2e_duration_seconds", "Scheduling decision latency.",
                                   (1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2))
        self.m_attempts = r.counter("igw_scheduler_attempts_total", "Scheduling attempts by outcome.", ["status", "endpoint_name"])
        self.m_hit = r.histogram("igw_prefix_indexer_hit_ratio", "Prefix match ratio of the picked endpoint.",
                                 (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
        self.g_running = r.gauge("igw_endpoint_inflight_requests", "Requests in flight per endpoint (router-local).", ["endpoint"])
        self.g_avg_queue = r.gauge("igw_average_queue_size", "Mean scraped waiting queue (llm_d_epp_average_queue_size).")
        self.g_avg_kv = r.gauge("igw_average_kv_cache_utilization", "Mean scraped KV usage (llm_d_epp_average_kv_cache_utilization).")
        self.g_ready = r.gauge("igw_ready_endpoints", "Endpoints in the pool (llm_d_epp_ready_endpoints).")
        self.g_saturation = r.gauge("igw_pool_saturation", "Utilization-detector saturation (>= 1.0 sheds priority < 0).")
        self.g_index = r.gauge("igw_prefix_indexer_size", "Entries in the approximate prefix index.")

    def _refresh_gauges(self):
        eps = self.ds.list()
        fresh = [e for e in eps if e.metrics.fresh]
        self.g_ready.set(len(eps))
        self.g_avg_queue.set(sum(e.metrics.waiting for e in fresh) / len(fresh) if fresh else 0.0)
        self.g_avg_kv.set(sum(e.metrics.kv_usage for e in fresh) / len(fresh) if fresh else 0.0)
        self.g_saturation.set(self.pool_saturation())
        for e in eps:
            self.g_running.labels(endpoint=e.name).set(e.inflight_requests)
        idx = self.prefix_index
        self.g_index.set(idx.size() if idx else 0)

    @property
    def prefix_index(self):
        for p in self.config.producers:
            if hasattr(p, "index"):
                return p.index
        return None

    def pool_saturation(self, now: float | None = None) -> float:
        """utilization-detector: mean over endpoints of max(queue/Tq, kv/Tkv); stale -> 1.0; empty -> 1.0."""
        eps = self.ds.list()
        if not eps:
            return 1.0
        now = time.monotonic() if now is None else now
        s = self.settings
        vals = []
        for e in eps:
            m = e.metrics
            if not m.fresh or now - m.update_time > s.metrics_staleness_s:
                vals.append(1.0)
            else:
                vals.append(max(m.waiting / s.queue_depth_threshold, m.kv_usage / s.kv_cache_util_threshold))
        return sum(vals) / len(vals)

    # ------------------------------------------------------------------ lifecycle
    async def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        import aiohttp
        from aiohttp import web
        self.session = aiohttp.ClientSession(auto_decompress=False,
                                             timeout=aiohttp.ClientTimeout(total=None, sock_read=self.settings.upstream_timeout_s),
                                             connector=aiohttp.TCPConnector(limit=0))
        interval = self.config.scrape_interval_s or self.settings.scrape_interval_s
        self.scraper = Scraper(self.ds, self.session, interval_s=interval, path=self.config.metrics_path)
        await self.scraper.refresh()                 # one synchronous scrape so the first request has data
        self.scraper.start()
        self._runner = web.AppRunner(self.make_app(), access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()
        actual = site._server.sockets[0].getsockname()[1]
        self.url = f"http://{host}:{actual}"
        return self.url

    async def stop(self):
        if self.scraper:
            await self.scraper.stop()
        if self._runner:
            await self._runner.cleanup()
        if self.session:
            await self.session.close()

    def make_app(self):
        from aiohttp import web
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/chat/completions", self.handle_inference)
        app.router.add_post("/v1/completions", self.handle_inference)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_get("/metrics", self.handle_metrics)
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/debug/state", self.handle_state)
        return app

    # ------------------------------------------------------------------ handlers
    async def handle_health(self, request):
        from aiohttp import web
        return web.json_response({"status": "ok"})

    async def handle_metrics(self, request):
        from aiohttp import web
        self._refresh_gauges()
        return web.Response(text=self.registry.render(), content_type="text/plain")

    async def handle_models(self, request):
        from aiohttp import web
        for ep in self.ds.list():
            try:
                async with self.session.get(ep.url + "/v1/models") as r:
                    return web.Response(body=await r.read(), status=r.status, content_type="application/json")
            except Exception:
                continue
        return web.json_response({"object": "list", "data": []})

    async def handle_state(self, request):
        from aiohttp import web
        idx = self.prefix_index
        eps = [{"name": e.name, "url": e.url, "inflight_requests": e.inflight_requests,
                "inflight_tokens": e.inflight_tokens, "routed_total": e.routed_total,
                "waiting": e.metrics.waiting, "running": e.metrics.running, "kv_usage": e.metrics.kv_usage,
                "scrape_errors": e.scrape_errors} for e in self.ds.list()]
        return web.json_response({"endpoints": eps, "pool_saturation": self.pool_saturation(),
                                  "prefix_index_blocks": idx.block_counts() if idx else {},
                                  "decisions": [d.as_dict() for d in list(self.decisions)[-20:]]})

    def _error(self, status: int, msg: str, model: str, code: str, reason: str | None = None):
        from aiohttp import web
        self.m_errors.labels(model_name=model, error_code=code).inc()
        headers = {DROPPED_REASON_HEADER: reason} if reason else None
        return web.json_response({"error": {"message": msg, "type": code, "code": status}}, status=status, headers=headers)

    def build_ctx(self, body: dict, headers) -> RequestCtx:
        objective = headers.get(OBJECTIVE_HEADER) or headers.get(OLD_OBJECTIVE_HEADER)
        return RequestCtx(request_id=headers.get("x-request-id") or uuid.uuid4().hex,
                          model=str(body.get("model", "")), tokens=estimate_tokens(body), body=body,
                          headers=dict(headers), cache_salt=str(body.get("cache_salt") or ""),
                          priority=int(self.settings.objectives.get(objective, 0)) if objective else 0)

    async def handle_inference(self, request):
        from aiohttp import web
        t0 = time.perf_counter()
        raw = await request.read()
        try:
            body = json.loads(raw)
            assert isinstance(body, dict)
        except Exception:
            return self._error(400, "request body must be a JSON object", "", "invalid_request")
        ctx = self.build_ctx(body, request.headers)
        model = ctx.model

        # ---- admission (llm-d's legacy controller: only sheddable traffic is ever rejected here)
        if ctx.priority < 0 and self.pool_saturation() >= 1.0:
            return self._error(429, "system saturated, sheddable request dropped", model, "rejected_saturated",
                               "rejected-saturated")

        # ---- scheduling
        decision = self.scheduler.schedule(ctx)
        self.m_sched.observe(decision.duration_s)
        self.decisions.append(decision)
        if decision.endpoint is None:
            self.m_attempts.labels(status="failure", endpoint_name="").inc()
            return self._error(503, f"no endpoints available: {decision.reason}", model, "service_unavailable",
                               "rejected-no-endpoints")
        ep: Endpoint = self.ds.get(decision.endpoint)
        self.m_attempts.labels(status="success", endpoint_name=ep.name).inc()
        pm = ctx.data.get("approx-prefix-cache-producer", {}).get(ep.name)
        if pm is not None:
            self.m_hit.observe(pm.ratio)

        # ---- dispatch
        self.scheduler.pre_request(ctx, ep)
        ep.inflight_requests += 1
        ep.routed_total += 1
        first = True
        fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        resp = None
        try:
            async with self.session.post(ep.url + request.path, data=raw, headers=fwd_headers) as up:
                out_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP_BY_HOP}
                out_headers[DESTINATION_HEADER] = ep.name
                resp = web.StreamResponse(status=up.status, headers=out_headers)
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    if first:
                        first = False
                        self.m_ttft.labels(model_name=model).observe(time.perf_counter() - t0)
                        self.scheduler.on_first_token(ctx, ep)
                    await resp.write(chunk)
                await resp.write_eof()
            self.m_requests.labels(model_name=model, endpoint=ep.name).inc()
            return resp
        except (ConnectionError, asyncio.TimeoutError, OSError) as e:
            if resp is not None and resp.prepared:        # client went away or backend died mid-stream
                self.m_errors.labels(model_name=model, error_code="stream_interrupted").inc()
                return resp
            return self._error(502, f"backend {ep.name} failed: {type(e).__name__}", model, "bad_gateway")
        except Exception as e:                            # aiohttp.ClientError and friends
            if resp is not None and resp.prepared:
                self.m_errors.labels(model_name=model, error_code="stream_interrupted").inc()
                return resp
            return self._error(502, f"backend {ep.name} failed: {type(e).__name__}", model, "bad_gateway")
        finally:
            ep.inflight_requests -= 1
            self.scheduler.on_complete(ctx, ep)
            self.m_duration.labels(model_name=model).observe(time.perf_counter() - t0)
