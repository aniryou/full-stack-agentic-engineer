"""The gateway process: one OpenAI-compatible front door in front of many providers (PRIMER §1).

For every `POST /v1/chat/completions` it runs the same pipeline, in this order:

    1. authenticate   the virtual key -> tenant (never a header); alias allowed? budget left?     401/403/404/429
    2. screen input   guardrail placement: inline | parallel | shadow                              400 content_filter
    3. cache          exact, then semantic (single-turn, cacheable classes only)                   served, cost 0
    4. admit          RPM + TPM per tenant (+ a global TPM): reserve prompt + output cap           429 + Retry-After
    5. route          alias -> filtered, ordered chain; a breaker per target
    6. call           adapter -> provider (its key, never the caller's); inject include_usage on streams, cache_salt
    7. relay          parse every chunk: translate the dialect, strip the usage chunk nobody asked for, meter as it
                      streams, apply the output guardrail, fall back only before the first byte
    8. account        reconcile the reservation with `usage`; one ledger row; spans; metrics; cache the answer

Unlike the 05 router (`igwlab/router/server.py`), which forwards bytes untouched and lets the endpoint picker
choose a *replica*, this process parses the stream because it owns the bill, and chooses a *model, provider
or pool* (05 PRIMER §1.3, §3.3, §7). Everything here runs in one asyncio process with sqlite3; `gwlab.stack`
starts it next to the fake providers for the notebooks.
"""
from __future__ import annotations

import asyncio
import collections
import json
import time
import uuid

import aiohttp
from aiohttp import web

from .. import sse
from ..promtext import Registry
from ..tokens import estimate_prompt, last_user_text, message_text, requested_output
from . import adapters, otel
from .adapters import DIALECTS, AnthropicStreamTranslator, Usage, UpstreamError, normalise_error, normalise_usage
from .cache import GatewayCache, Lookup, cache_salt
from .config import GatewayConfig
from .guardrails import RegexScreener
from .keys import KeyError401, KeyStore, ProviderKeys, bearer
from .metering import LedgerRow, cost_usd
from .ratelimit import Limiter
from .routing import FAIL, FALLTHROUGH, Needs, Router, classify
from .store import Store

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
TENANT_HEADERS = ("x-tenant", "x-tenant-id", "x-org-id")          # read only to show they are ignored


class Outcome:
    """What one attempt against one target produced."""

    def __init__(self, kind: str, *, error: UpstreamError | None = None, response=None, usage: Usage | None = None,
                 acc=None, ttft_s: float | None = None, sent: bool = False, finish: str | None = None,
                 blocked: bool = False, disconnected: bool = False, reason: str = "", tail: list | None = None):
        self.kind, self.error, self.response, self.usage, self.acc = kind, error, response, usage, acc
        self.tail = tail or []           # stream bytes written after the ledger row, so a client that has seen
                                         # [DONE] can already read its own row
        self.ttft_s, self.sent, self.finish, self.blocked, self.disconnected = ttft_s, sent, finish, blocked, disconnected
        self.reason = reason or (error.code if error else "")


class Gateway:
    def __init__(self, cfg: GatewayConfig, *, screener: RegexScreener | None = None, clock=time.monotonic):
        self.cfg = cfg
        self.store = Store(cfg.db)
        self.keys = KeyStore(self.store)
        self.provider_keys = ProviderKeys({n: p.key for n, p in cfg.providers.items()})
        self.limiter = Limiter(cfg.limits, cfg.tenants, clock)
        self.router = Router(cfg, clock)
        self.cache = GatewayCache(self.store, cfg.cache)
        self.screener = screener or RegexScreener(cfg.guardrails.check_ms / 1000)
        self.tracer = otel.Tracer(cfg.spans)
        self.decisions: collections.deque = collections.deque(maxlen=500)
        self.audit: collections.deque = collections.deque(maxlen=1000)
        self.session: aiohttp.ClientSession | None = None
        self.mcp = None
        self._runner = None
        self.url: str | None = None
        self._init_metrics()

    # ------------------------------------------------------------------------------------------ metrics
    def _init_metrics(self):
        r = self.registry = Registry()
        self.m_req = r.counter("gwlab_requests", "Requests by tenant, alias, serving target and status.",
                               ["tenant", "alias", "target", "status"])
        self.m_att = r.counter("gwlab_attempts", "Upstream attempts by target and outcome.", ["target", "outcome"])
        self.m_tok = r.counter("gwlab_tokens", "Tokens by tenant and kind (from usage, or estimated).", ["tenant", "kind"])
        self.m_cost = r.counter("gwlab_cost_usd", "Dollars by tenant, priced from usage.", ["tenant"])
        self.m_cache = r.counter("gwlab_cache", "Cache lookups by result.", ["result"])
        self.m_rl = r.counter("gwlab_ratelimited", "Requests refused by a limit.", ["tenant", "limit"])
        self.m_guard = r.counter("gwlab_guardrail_findings", "Guardrail findings.", ["hook", "rule", "action"])
        self.m_disc = r.counter("gwlab_client_disconnects", "Streams the client cut before the end.")
        self.m_ttft = r.histogram("gwlab_ttft_seconds", "Time to first relayed token, as the client sees it.",
                                  LATENCY_BUCKETS, ["alias"])
        self.m_e2e = r.histogram("gwlab_request_duration_seconds", "End-to-end latency at the gateway.",
                                 LATENCY_BUCKETS, ["alias"])
        self.g_breaker = r.gauge("gwlab_breaker_state", "Breaker per target: 0 closed, 0.5 half-open, 1 open.", ["target"])

    def metrics_text(self) -> str:
        for mid, st in self.router.state().items():
            self.g_breaker.labels(target=mid).set({"closed": 0.0, "half_open": 0.5, "open": 1.0}[st["breaker"]])
        return self.registry.render()

    # ------------------------------------------------------------------------------------------ lifecycle
    async def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=5),
                                             connector=aiohttp.TCPConnector(limit=0), auto_decompress=True)
        if self.cfg.mcp_servers:
            from ..mcp.client import McpClient
            self.mcp = McpClient(self.session, client_id=None)
        self._runner = web.AppRunner(self.make_app(), access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()
        self.url = f"http://{host}:{site._server.sockets[0].getsockname()[1]}"
        if self.mcp:
            self.mcp.client_id = self.url + "/oauth/client-metadata.json"
        return self.url

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
        if self.session:
            await self.session.close()

    def make_app(self) -> web.Application:
        app = web.Application(client_max_size=32 * 1024 * 1024)
        app.router.add_post("/v1/chat/completions", self.handle_chat)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_get("/metrics", self.handle_metrics)
        app.router.add_get("/health", self.handle_health)
        app.router.add_post("/admin/keys", self.admin_issue)
        app.router.add_get("/admin/keys", self.admin_list)
        app.router.add_delete("/admin/keys/{key_id}", self.admin_revoke)
        app.router.add_post("/admin/providers/{name}/key", self.admin_rotate)
        app.router.add_delete("/admin/providers/{name}/key", self.admin_retire)
        app.router.add_delete("/admin/cache", self.admin_cache_clear)
        app.router.add_get("/debug/state", self.admin_state)
        app.router.add_get("/oauth/client-metadata.json", self.client_metadata)
        app.router.add_post("/mcp/{server}", self.handle_mcp)
        return app

    async def handle_metrics(self, request):
        return web.Response(text=self.metrics_text(), content_type="text/plain")

    async def handle_health(self, request):
        return web.json_response({"status": "ok"})

    # ------------------------------------------------------------------------------------------ helpers
    @staticmethod
    def error(status: int, code: str, message: str, etype: str | None = None, headers: dict | None = None) -> web.Response:
        etype = etype or {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
                          404: "not_found_error", 429: "rate_limit_error"}.get(status, "api_error")
        return web.json_response({"error": {"message": message, "type": etype, "param": None, "code": code}},
                                 status=status, headers=headers)

    def _admin(self, request) -> bool:
        return bearer(request.headers) == self.cfg.admin_token

    def _rl_headers(self, key) -> dict:
        return self.limiter.headers(key.tenant, key.key_id if (key.rpm or key.tpm) else None)

    def tenant_salt(self, tenant: str) -> str:
        return cache_salt(tenant, self.cfg.salt_secret)

    # ------------------------------------------------------------------------------------------ admin API
    async def admin_issue(self, request):
        if not self._admin(request):
            return self.error(401, "invalid_admin_token", "admin token required")
        d = await request.json()
        if not d.get("tenant"):
            return self.error(400, "bad_request", "tenant is required")
        secret, k = self.keys.issue(d["tenant"], d.get("aliases"), d.get("rpm"), d.get("tpm"), d.get("budget_usd"),
                                    d.get("ttl_s"))
        self.audit.append({"event": "key.issue", "key_id": k.key_id, "tenant": k.tenant, "ts": time.time()})
        return web.json_response({"key": secret, **k.public(), "note": "shown once; the gateway stores only its SHA-256"})

    async def admin_list(self, request):
        if not self._admin(request):
            return self.error(401, "invalid_admin_token", "admin token required")
        return web.json_response({"keys": [k.public() for k in self.keys.list(request.query.get("tenant"))]})

    async def admin_revoke(self, request):
        if not self._admin(request):
            return self.error(401, "invalid_admin_token", "admin token required")
        ok = self.keys.revoke(request.match_info["key_id"])
        self.audit.append({"event": "key.revoke", "key_id": request.match_info["key_id"], "ok": ok, "ts": time.time()})
        return web.json_response({"revoked": ok}, status=200 if ok else 404)

    async def admin_rotate(self, request):
        if not self._admin(request):
            return self.error(401, "invalid_admin_token", "admin token required")
        name = request.match_info["name"]
        if name not in self.cfg.providers:
            return self.error(404, "not_found", f"unknown provider {name}")
        self.provider_keys.rotate(name, (await request.json())["key"])
        self.audit.append({"event": "provider_key.rotate", "provider": name, "ts": time.time(),
                           "fingerprint": self.provider_keys.fingerprint(name)})
        return web.json_response({"provider": name, "fingerprint": self.provider_keys.fingerprint(name)})

    async def admin_retire(self, request):
        if not self._admin(request):
            return self.error(401, "invalid_admin_token", "admin token required")
        old = self.provider_keys.retire(request.match_info["name"])
        return web.json_response({"retired": old is not None})

    async def admin_cache_clear(self, request):
        if not self._admin(request):
            return self.error(401, "invalid_admin_token", "admin token required")
        return web.json_response({"deleted": self.cache.invalidate(request.query.get("tenant"))})

    async def admin_state(self, request):
        if not self._admin(request):
            return self.error(401, "invalid_admin_token", "admin token required")
        return web.json_response({"router": self.router.state(), "limits": self.limiter.state(),
                                  "cache": self.cache.stats, "decisions": list(self.decisions)[-20:],
                                  "provider_key_fingerprints": {p: self.provider_keys.fingerprint(p) for p in self.cfg.providers}})

    async def client_metadata(self, request):
        """The gateway's Client ID Metadata Document (PRIMER §8): its `client_id` is this URL."""
        me = f"{request.scheme}://{request.host}"
        return web.json_response({"client_id": me + "/oauth/client-metadata.json", "client_name": "gwlab gateway",
                                  "redirect_uris": [me + "/oauth/callback"], "grant_types": ["authorization_code", "refresh_token"],
                                  "response_types": ["code"], "token_endpoint_auth_method": "none"})

    # ------------------------------------------------------------------------------------------ /v1/models
    async def handle_models(self, request):
        try:
            key = self.keys.verify(bearer(request.headers))
        except KeyError401 as e:
            return self.error(401, "invalid_api_key", str(e))
        tenant = self.cfg.tenants.get(key.tenant)
        names = [a for a in self.cfg.aliases if key.allows(a) and (tenant is None or a in tenant.aliases)]
        return web.json_response({"object": "list", "data": [
            {"id": a, "object": "model", "owned_by": "gwlab", "targets": self.cfg.aliases[a].targets} for a in names]})

    # ------------------------------------------------------------------------------------------ the pipeline
    async def handle_chat(self, request):
        t0 = time.perf_counter()
        rid = request.headers.get("x-request-id") or "req_" + uuid.uuid4().hex[:16]
        span = self.tracer.start("POST /v1/chat/completions", "server", **{"http.request.method": "POST",
                                                                          "url.path": "/v1/chat/completions"})
        dec = {"request_id": rid, "trace_id": span.trace_id, "attempts": [], "ts": time.time()}
        try:
            key = self.keys.verify(bearer(request.headers))
        except KeyError401 as e:
            return self._finish_error(span, dec, self.error(401, "invalid_api_key", str(e)))
        dec["tenant"], dec["key_id"] = key.tenant, key.key_id
        for h in TENANT_HEADERS:
            if request.headers.get(h) and request.headers[h] != key.tenant:
                dec["ignored_header"] = f"{h}: {request.headers[h]}"
        try:
            body = await request.json()
            assert isinstance(body, dict) and isinstance(body.get("messages"), list) and body["messages"]
        except (ValueError, AssertionError):
            return self._finish_error(span, dec, self.error(400, "bad_request", "body must be JSON with a non-empty messages list"))
        alias = str(body.get("model", ""))
        dec["alias"] = alias
        tenant = self.cfg.tenants.get(key.tenant)
        if alias not in self.cfg.aliases:
            return self._finish_error(span, dec, self.error(404, "model_not_found", f"unknown model alias {alias!r}"))
        if not key.allows(alias) or (tenant is not None and alias not in tenant.aliases):
            return self._finish_error(span, dec, self.error(403, "model_not_allowed", f"{alias!r} is not allowed for this key"))
        if key.budget_usd is not None and key.spent_usd >= key.budget_usd:
            return self._finish_error(span, dec, self.error(429, "budget_exceeded",
                                                            f"key budget ${key.budget_usd:.4f} spent", "insufficient_quota"))
        stream = bool(body.get("stream"))
        want_usage = stream and bool((body.get("stream_options") or {}).get("include_usage"))
        span.set(**{otel.REQUEST_MODEL: alias, otel.GW_TENANT: key.tenant, otel.GW_ALIAS: alias})
        if stream:
            span.set(**{otel.REQUEST_STREAM: True})

        # 2. input guardrail
        g = self.cfg.guardrails
        input_check = None
        if g.input != "off":
            text = " ".join(message_text(m) for m in body["messages"] if m.get("role") in ("user", "tool"))
            if g.input == "inline":
                findings = await self._screen("input", text, "block")
                if findings:
                    return self._finish_error(span, dec, self.error(400, "content_filter",
                                                                    f"blocked by the gateway's input guardrail: {findings[0].rule}"))
            else:
                input_check = asyncio.ensure_future(self._screen("input", text, "block" if g.input == "parallel" else "log"))
                if g.input == "shadow":
                    input_check = None

        # 3. cache
        look = self.cache.lookup(key.tenant, alias, body) if (tenant is None or tenant.cache) else Lookup(None, reason="tenant opted out")
        self.m_cache.labels(result=look.kind or ("bypass" if look.reason not in ("miss", "below threshold", "entity guard") else "miss")).inc()
        dec["cache"] = {"result": look.kind or "miss", "reason": look.reason, "score": round(look.score, 4), "matched": look.matched}
        if look.kind:
            return await self._serve_cached(request, span, dec, key, alias, body, look, t0, stream, want_usage)

        # 4. admit
        prompt_est, requested = estimate_prompt(body), requested_output(body)
        adm = self.limiter.admit(key.tenant, prompt_est, requested, key_id=key.key_id, rpm=key.rpm, tpm=key.tpm)
        if not adm.ok:
            self.m_rl.labels(tenant=key.tenant, limit=adm.limit).inc()
            return self._finish_error(span, dec, self.error(429, "rate_limit_exceeded", f"tenant {key.tenant}: {adm.limit} limit",
                                                            headers=adm.headers))
        res = adm.reservation
        dec["reserved_tokens"] = res.charged

        # 5. route
        plan = self.router.plan(alias, tenant, Needs(prompt_est, res.reserved_output, bool(body.get("tools")), rid))
        dec["plan"], dec["skipped"] = plan.ids(), plan.skipped
        last_err, outcome, served = None, None, None
        for n, model in enumerate(plan.targets, 1):
            br = self.router.breakers[model.id]
            if not br.allow():
                dec["attempts"].append({"target": model.id, "outcome": "breaker_open"})
                self.m_att.labels(target=model.id, outcome="breaker_open").inc()
                continue
            prov = self.cfg.providers[model.provider]
            cspan = self.tracer.start(f"chat {model.upstream}", "client", parent=span, **{
                otel.OPERATION: "chat", otel.PROVIDER: prov.otel_name or DIALECTS[prov.dialect]["otel_provider"],
                otel.REQUEST_MODEL: model.upstream, otel.SERVER_ADDRESS: prov.base_url.split("//")[-1].split(":")[0],
                otel.GW_TARGET: model.id, otel.GW_ATTEMPT: n})
            if stream:
                cspan.set(**{otel.REQUEST_STREAM: True})
            ta = time.perf_counter()
            try:
                outcome = await self._attempt(request, model, prov, body, stream, want_usage, res, key, cspan, dec, n,
                                              input_check, t0)
            except Exception as e:                      # a gateway bug must not look like a provider failure
                br.release()
                self.tracer.end(cspan, error=type(e).__name__)
                raise
            att = {"target": model.id, "outcome": outcome.kind, "ms": round((time.perf_counter() - ta) * 1e3, 1)}
            if outcome.error:
                att.update(status=outcome.error.status, code=outcome.error.code)
            dec["attempts"].append(att)
            self.m_att.labels(target=model.id, outcome=outcome.kind).inc()
            if outcome.kind == FALLTHROUGH:
                self.router.record(model.id, False)
                self.tracer.end(cspan, error=outcome.reason)
                last_err = outcome.error
                continue
            if outcome.kind == FAIL:
                br.release()
                self.tracer.end(cspan, error=outcome.reason)
                last_err = outcome.error
                break
            served = model                               # "done": answered, or failed after the first byte
            self.router.record(model.id, outcome.error is None, outcome.ttft_s)
            self._close_client_span(cspan, outcome, model)
            break

        # 8. account
        if served is None:
            self.limiter.reconcile(res, 0, 0)            # nothing ran: refund the whole reservation
            resp = self._no_answer(dec, outcome, last_err)
            err_code = json.loads(resp.body)["error"]["code"]
            self._ledger(dec, key, alias, None, Usage(0, 0, source="none"), resp.status, stream, t0, None, err_code, None)
            return self._finish_error(span, dec, resp)
        return await self._account(request, span, dec, key, alias, served, outcome, res, body, stream, t0)

    def _no_answer(self, dec, outcome, last_err) -> web.Response:
        """The response when no target answered: a non-fallthrough error keeps its status (a provider 401/403 is
        the gateway's credential problem, so the client sees 502); a chain of 429s is a 429; otherwise 503."""
        if outcome is not None and outcome.kind == FAIL and last_err is not None:
            if last_err.status in (401, 403):
                return self.error(502, "upstream_auth_error", f"the provider rejected the gateway's credential: {last_err.message}")
            return self.error(last_err.status, last_err.code, last_err.message, last_err.type)
        tried = [a for a in dec["attempts"] if a["outcome"] != "breaker_open"]
        if not tried:
            wait = min((self.router.breakers[a["target"]].recovery_s for a in dec["attempts"]), default=1.0)
            return self.error(503, "no_target_available", "every target is unavailable (breakers open or filtered out)",
                              headers={"Retry-After": str(max(1, int(wait)))})
        if all(a.get("status") == 429 for a in tried):
            return self.error(429, "rate_limit_exceeded", f"every target is rate-limited ({len(tried)} tried)",
                              headers={"Retry-After": str(max(1, int((last_err.retry_after if last_err else 1) or 1)))})
        msg = last_err.message if last_err else "no answer"
        return self.error(503, "all_targets_failed", f"{len(tried)} target(s) failed; last: {msg}", "api_error",
                          headers={"Retry-After": "1"})

    async def _screen(self, hook: str, text: str, action: str):
        await asyncio.sleep(self.screener.check_s)            # the classifier call a regex stands in for (simulated)
        findings = self.screener.check_input(text) if hook == "input" else self.screener.check_output(text)
        for f in findings:
            self.m_guard.labels(hook=hook, rule=f.rule, action=action).inc()
            self.audit.append({"event": f"guardrail.{hook}", "rule": f.rule, "action": action, "ts": time.time()})
        return findings

    # ------------------------------------------------------------------------------------------ one attempt
    async def _attempt(self, request, model, prov, body, stream, want_usage, res, key, cspan, dec, n, input_check, t0):
        salt = self.tenant_salt(key.tenant) if prov.cache_salt else None
        up = adapters.to_upstream(prov.dialect, body, model.upstream, stream=stream, want_usage=True, cache_salt=salt,
                                  output_cap=self.cfg.limits.hard_output_cap)
        url = prov.base_url.rstrip("/") + DIALECTS[prov.dialect]["path"]
        headers = {**adapters.auth_headers(prov.dialect, self.provider_keys.get(prov.name)),
                   "traceparent": f"00-{cspan.trace_id}-{cspan.span_id}-01", "x-request-id": dec["request_id"]}
        t_send = time.perf_counter()
        timeout = prov.first_byte_timeout_s if stream else prov.total_timeout_s
        try:
            resp = await asyncio.wait_for(self.session.post(url, json=up, headers=headers), timeout)
        except asyncio.TimeoutError:
            return Outcome(FALLTHROUGH, error=UpstreamError(504, "timeout", "timeout", f"no response headers in {timeout:g}s"))
        except aiohttp.ClientError as e:
            return Outcome(FALLTHROUGH, error=UpstreamError(502, "connection_error", "connect_error", type(e).__name__))
        try:
            if resp.status != 200:
                raw = await resp.read()
                try:
                    data = json.loads(raw)
                except ValueError:
                    data = raw.decode(errors="replace")
                ra = resp.headers.get("Retry-After")
                err = normalise_error(prov.dialect, resp.status, data, float(ra) if ra and ra.replace(".", "").isdigit() else None)
                return Outcome(classify(err.status, err.code), error=err)
            if not stream:
                return await self._nonstream(resp, model, prov, input_check, t_send)
            return await self._relay(request, resp, model, prov, want_usage, res, key, input_check, dec, n, t0, t_send)
        finally:
            resp.release()

    async def _nonstream(self, resp, model, prov, input_check, t_send):
        try:
            data = await asyncio.wait_for(resp.json(), prov.total_timeout_s)
        except (asyncio.TimeoutError, aiohttp.ClientError, ValueError) as e:
            return Outcome(FALLTHROUGH, error=UpstreamError(502, "upstream_error", "bad_upstream_body", type(e).__name__))
        if prov.dialect == "anthropic":
            completion, usage = adapters.completion_from_anthropic(data, model.upstream)
        else:
            completion, usage = data, normalise_usage("openai", data.get("usage"))
            completion.setdefault("model", model.upstream)
        if input_check is not None and await input_check:
            return Outcome(FAIL, error=UpstreamError(400, "invalid_request_error", "content_filter", "blocked by the input guardrail"),
                           usage=usage)
        blocked = False
        g = self.cfg.guardrails
        if g.output not in ("off",):
            text = (completion["choices"][0]["message"].get("content") or "")
            findings = await self._screen("output", text, "log" if g.output == "shadow" else "block")
            if findings and g.output != "shadow":
                completion["choices"][0]["message"]["content"] = "[withheld by the gateway's output guardrail]"
                completion["choices"][0]["finish_reason"] = "content_filter"
                blocked = True
        ttft = time.perf_counter() - t_send
        return Outcome("done", response=completion, usage=usage, ttft_s=ttft,
                       finish=completion["choices"][0].get("finish_reason"), blocked=blocked)

    async def _relay(self, request, resp, model, prov, want_usage, res, key, input_check, dec, n, t0, t_send):
        """Relay a stream: nothing reaches the client until the first real chunk, so any failure before it can
        still fall through to the next target; after it, failures are surfaced in the stream, never spliced."""
        g = self.cfg.guardrails
        parser, acc = sse.SSEParser(), sse.StreamAccumulator()
        tr = AnthropicStreamTranslator(model.upstream) if prov.dialect == "anthropic" else None
        upstream_usage: dict | None = None
        out: web.StreamResponse | None = None
        held: list = []                      # chunks not yet released (the role chunk; a guardrail window)
        st = {"ttft": None, "window_start": 0, "sent": False, "blocked": False, "cut": False, "check": None}
        done = False
        first_deadline = time.perf_counter() + prov.first_byte_timeout_s

        async def write(chunks: list):
            nonlocal out
            if not chunks:
                return
            if input_check is not None and not st["sent"] and await input_check:
                raise _Blocked("input")
            if out is None:
                out = web.StreamResponse(status=200, headers={
                    "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "x-request-id": dec["request_id"],
                    "x-gwlab-target": model.id, "x-gwlab-provider": prov.name, "x-gwlab-attempts": str(n),
                    "x-gwlab-cache": "miss", **self._rl_headers(key)})
                await out.prepare(request)
            try:
                for c in chunks:
                    await out.write(sse.encode(c))
            except ConnectionResetError as e:          # aiohttp's ClientConnectionResetError is one of these
                raise _ClientGone() from e
            if not st["sent"]:
                st["sent"], st["ttft"] = True, time.perf_counter() - t0
                self.m_ttft.labels(alias=dec.get("alias", "")).observe(st["ttft"])

        async def release(final: bool = False):
            """Apply the output guardrail placement to the held chunks (PRIMER §7)."""
            new = acc.content_chunks - st["window_start"]
            if g.output in ("window", "full"):
                if not (final or (g.output == "window" and new >= g.window_tokens)):
                    return
                if held and await self._screen("output", acc.content, "block"):
                    st["blocked"] = True
                    return
                st["window_start"] = acc.content_chunks
            elif g.output == "parallel":
                t = st["check"]
                if t is not None and t.done() and t.result():
                    st["blocked"] = True
                    return
                if (new >= g.window_tokens or final) and (t is None or t.done()):
                    st["window_start"] = acc.content_chunks
                    st["check"] = asyncio.ensure_future(self._screen("output", acc.content, "cut"))
            batch = held[:]
            held.clear()
            await write(batch)
            if final and g.output == "shadow":
                asyncio.ensure_future(self._screen("output", acc.content, "log"))

        try:
            while not done:
                timeout = max(0.001, first_deadline - time.perf_counter()) if not st["sent"] else prov.total_timeout_s
                try:
                    raw = await asyncio.wait_for(resp.content.readany(), timeout)
                except asyncio.TimeoutError:
                    if not st["sent"]:
                        resp.close()
                        return Outcome(FALLTHROUGH, error=UpstreamError(504, "timeout", "timeout", "no first chunk in time"))
                    raise _UpstreamGone("idle timeout mid-stream")
                if not raw:
                    break                                                  # the upstream closed the connection
                for ev in parser.feed(raw):
                    if ev.data.strip() == sse.DONE:
                        done = True
                        continue
                    obj = json.loads(ev.data)
                    chunks = tr.feed(ev.event, obj) if tr else [obj]
                    if tr and tr.done:
                        done = True
                    for ch in chunks:
                        if ch.get("error"):
                            e = ch["error"] if isinstance(ch["error"], dict) else {"message": str(ch["error"])}
                            status = e.get("status") or (e["code"] if isinstance(e.get("code"), int) else 500)
                            err = normalise_error("openai", int(status), {"error": e})
                            if not st["sent"]:
                                resp.close()
                                return Outcome(classify(err.status, err.code), error=err)
                            raise _MidStream(err)
                        acc.add(ch)
                        if not ch.get("choices") and ch.get("usage"):
                            upstream_usage = ch["usage"]                  # the usage chunk: kept for the end
                            continue
                        if not want_usage:
                            ch.pop("usage", None)                         # the client did not ask for usage
                        held.append(ch)
                        if not self.limiter.on_output(res, acc.output_estimate()):
                            st["cut"] = done = True
                            break
                    if acc.token_chunks or acc.tool_calls or acc.finish_reason:
                        await release()
                    if st["blocked"]:
                        done = True
                        break
            if not done and not (tr is None and acc.finish_reason):
                if not st["sent"]:
                    return Outcome(FALLTHROUGH, error=UpstreamError(502, "upstream_error", "connection_reset",
                                                                    "upstream closed before the first chunk"))
                raise _UpstreamGone("upstream closed mid-stream")
            if st["blocked"] or st["cut"]:
                resp.close()                                              # stop the provider generating
            if not st["blocked"]:
                await release(final=True)
            usage = normalise_usage("openai", upstream_usage) if not tr else tr.usage
            if st["blocked"]:
                held.clear()
                await write([{"id": acc.id, "object": "chat.completion.chunk", "model": model.upstream,
                              "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}]}])
                acc.finish_reason = "content_filter"
                usage = None                                              # cut short: the bill is estimated
            if want_usage and usage:
                await write([{"id": acc.id or "chatcmpl-" + dec["request_id"], "object": "chat.completion.chunk",
                              "model": model.upstream, "choices": [], "usage": usage.to_openai()}])
            if out is None:                                               # an empty answer: still a valid stream
                await write([{"id": acc.id, "object": "chat.completion.chunk", "model": model.upstream,
                              "choices": [{"index": 0, "delta": {}, "finish_reason": acc.finish_reason or "stop"}]}])
            return Outcome("done", response=out, usage=usage, acc=acc, ttft_s=st["ttft"], sent=True,
                           finish=acc.finish_reason, blocked=st["blocked"], reason="output_cap" if st["cut"] else "",
                           tail=[sse.encode_done()])
        except _ClientGone:                                    # the client went away: stop the upstream, meter what ran
            self.m_disc.inc()
            resp.close()
            return Outcome("done", response=out, acc=acc, ttft_s=st["ttft"], sent=True, disconnected=True,
                           finish=acc.finish_reason)
        except _Blocked:
            resp.close()
            return Outcome(FAIL, error=UpstreamError(400, "invalid_request_error", "content_filter",
                                                     "blocked by the input guardrail"), acc=acc)
        except (_MidStream, _UpstreamGone, aiohttp.ClientPayloadError, aiohttp.ClientConnectionError) as e:
            err = e.err if isinstance(e, _MidStream) else UpstreamError(502, "upstream_error", "stream_interrupted", str(e))
            if not st["sent"]:
                return Outcome(FALLTHROUGH, error=err, acc=acc)
            # after the first byte: surface the error in the stream, never splice another model's output
            return Outcome("done", response=out, acc=acc, ttft_s=st["ttft"], sent=True, error=err, finish=acc.finish_reason,
                           tail=[sse.encode({"error": err.body()["error"]}), sse.encode_done()])

    # ------------------------------------------------------------------------------------------ accounting
    def _close_client_span(self, cspan, outcome, model):
        u = outcome.usage
        if u:
            cspan.set(**{otel.INPUT_TOKENS: u.prompt_tokens, otel.OUTPUT_TOKENS: u.completion_tokens,
                         otel.CACHE_READ: u.cached_tokens, otel.REASONING: u.reasoning_tokens})
        cspan.set(**{otel.RESPONSE_MODEL: model.upstream, otel.FINISH_REASONS: [outcome.finish or "error"],
                     otel.TIME_TO_FIRST_CHUNK: outcome.ttft_s})
        self.tracer.end(cspan, error=outcome.error.code if outcome.error else ("client_disconnect" if outcome.disconnected else None))

    async def _account(self, request, span, dec, key, alias, model, o, res, body, stream, t0):
        usage = o.usage
        if usage is None:                                 # a cut stream: no usage chunk arrived -- never bill it as 0
            usage = Usage(res.prompt_estimate, o.acc.output_estimate() if o.acc is not None else 0, source="estimate")
        self.limiter.reconcile(res, usage.prompt_tokens, usage.completion_tokens)
        cost = cost_usd(usage, model.price)
        self.keys.add_spend(key.key_id, cost)
        status = 200
        err = o.error.code if o.error else ("client_disconnect" if o.disconnected else ("guardrail_blocked" if o.blocked else ""))
        self._ledger(dec, key, alias, model, usage, status, stream, t0, o.ttft_s, err, o.finish, cost)
        for kind, v in (("prompt", usage.prompt_tokens), ("completion", usage.completion_tokens),
                        ("cached", usage.cached_tokens), ("reasoning", usage.reasoning_tokens)):
            self.m_tok.labels(tenant=key.tenant, kind=kind).inc(v)
        self.m_cost.labels(tenant=key.tenant).inc(cost)
        self.m_req.labels(tenant=key.tenant, alias=alias, target=model.id, status=str(status)).inc()
        self.m_e2e.labels(alias=alias).observe(time.perf_counter() - t0)
        completion = o.response if not stream else (o.acc.completion() if o.acc else None)
        if not o.error and not o.blocked and not o.disconnected and o.finish == "stop" and completion:
            cached = json.loads(json.dumps(completion))
            cached["usage"] = usage.to_openai()
            self.cache.put(key.tenant, alias, body, cached)
        span.set(**{otel.GW_TARGET: model.id, otel.GW_COST: round(cost, 8), otel.GW_CACHE: "miss",
                    "http.response.status_code": status})
        self.tracer.end(span, error=err or None)
        dec.update(served_by=model.id, status=status, cost_usd=cost, usage=usage.as_dict(),
                   ttft_ms=round(o.ttft_s * 1e3, 1) if o.ttft_s else None, e2e_ms=round((time.perf_counter() - t0) * 1e3, 1),
                   error=err, finish_reason=o.finish)
        self.decisions.append(dec)
        if stream:
            if not o.disconnected:
                try:
                    for b in o.tail:
                        await o.response.write(b)
                    await o.response.write_eof()
                except ConnectionResetError:
                    pass
            return o.response
        resp = web.json_response(o.response, headers={"x-request-id": dec["request_id"], "x-gwlab-target": model.id,
                                                      "x-gwlab-provider": model.provider, "x-gwlab-cache": "miss",
                                                      "x-gwlab-attempts": str(len(dec["attempts"])),
                                                      **self._rl_headers(key)})
        return resp

    def _ledger(self, dec, key, alias, model, usage, status, stream, t0, ttft_s, error, finish, cost=0.0, cache="miss"):
        row = LedgerRow(request_id=dec["request_id"], ts=time.time(), tenant=key.tenant, key_id=key.key_id, alias=alias,
                        target=model.id if model else "", provider=model.provider if model else "", status=status,
                        stream=stream, prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                        cached_tokens=usage.cached_tokens, reasoning_tokens=usage.reasoning_tokens,
                        usage_source=usage.source, cost_usd=cost, ttft_ms=ttft_s * 1e3 if ttft_s else None,
                        e2e_ms=(time.perf_counter() - t0) * 1e3, cache=cache, attempts=len(dec["attempts"]),
                        error=error or "", finish_reason=finish, trace_id=dec["trace_id"])
        d = row.as_dict()
        self.store.execute(f"INSERT OR REPLACE INTO ledger ({','.join(d)}) VALUES ({','.join('?' * len(d))})",
                           tuple(int(v) if isinstance(v, bool) else v for v in d.values()))

    def _finish_error(self, span, dec, resp: web.Response) -> web.Response:
        try:
            code = json.loads(resp.body)["error"]["code"]
        except (ValueError, KeyError, TypeError):
            code = str(resp.status)
        span.set(**{"http.response.status_code": resp.status})
        self.tracer.end(span, error=code)
        dec.update(status=resp.status, error=code)
        self.decisions.append(dec)
        self.m_req.labels(tenant=dec.get("tenant", ""), alias=dec.get("alias", ""), target="", status=str(resp.status)).inc()
        return resp

    async def _serve_cached(self, request, span, dec, key, alias, body, look, t0, stream, want_usage):
        completion = dict(look.response)
        headers = {"x-request-id": dec["request_id"], "x-gwlab-cache": look.kind, "x-gwlab-cache-score": f"{look.score:.4f}"}
        usage_d = completion.get("usage") or {}
        self._ledger(dec, key, alias, None, Usage(0, 0, source="none"), 200, stream, t0, None, "", "stop", 0.0, look.kind)
        span.set(**{otel.GW_CACHE: look.kind, "http.response.status_code": 200})
        self.tracer.end(span)
        dec.update(served_by="cache:" + look.kind, status=200, cost_usd=0.0,
                   saved_usage=usage_d, e2e_ms=round((time.perf_counter() - t0) * 1e3, 1))
        self.decisions.append(dec)
        self.m_req.labels(tenant=key.tenant, alias=alias, target="cache", status="200").inc()
        if not stream:
            return web.json_response(completion, headers=headers)
        out = web.StreamResponse(headers={"Content-Type": "text/event-stream", **headers})
        await out.prepare(request)
        msg = completion["choices"][0]["message"]
        base = {"id": completion.get("id"), "object": "chat.completion.chunk", "model": completion.get("model")}
        words = (msg.get("content") or "").split(" ")
        chunks = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
        for i in range(0, len(words), 8):
            chunks.append({**base, "choices": [{"index": 0, "delta": {"content": (" " if i else "") + " ".join(words[i:i + 8])},
                                                "finish_reason": None}]})
        chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        if want_usage:
            chunks.append({**base, "choices": [], "usage": usage_d})
        for c in chunks:
            await out.write(sse.encode(c))
        await out.write(sse.encode_done())
        await out.write_eof()
        return out

    # ------------------------------------------------------------------------------------------ MCP (PRIMER §8)
    async def handle_mcp(self, request):
        """Agents reach MCP servers through the gateway: the virtual key names the principal; the gateway holds
        (and obtains, with the OAuth flow) the MCP token for (principal, resource, scopes). Never passthrough."""
        try:
            key = self.keys.verify(bearer(request.headers))
        except KeyError401 as e:
            return self.error(401, "invalid_api_key", str(e))
        server = request.match_info["server"]
        if self.mcp is None or server not in self.cfg.mcp_servers:
            return self.error(404, "unknown_mcp_server", f"no MCP server {server!r} configured")
        rpc = await request.json()
        principal = request.headers.get("x-gwlab-user") and f"{key.tenant}/{request.headers['x-gwlab-user']}" or key.tenant
        status, body = await self.mcp.call(principal, self.cfg.mcp_servers[server], rpc)
        self.audit.append({"event": "mcp.call", "principal": principal, "server": server, "method": rpc.get("method"),
                           "status": status, "ts": time.time()})
        return web.json_response(body, status=status)


class _Blocked(Exception):
    pass


class _ClientGone(Exception):
    pass


class _UpstreamGone(Exception):
    pass


class _MidStream(Exception):
    def __init__(self, err: UpstreamError):
        super().__init__(str(err))
        self.err = err


__all__ = ["Gateway", "Outcome", "last_user_text"]
