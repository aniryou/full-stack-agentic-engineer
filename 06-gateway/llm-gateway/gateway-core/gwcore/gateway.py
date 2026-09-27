"""The pipeline: one front door (PRIMER §1).

The one idea: every request crosses the same ordered stages -- authenticate the key (the tenant comes
from it), screen the input, look up the cache, reserve tokens, walk the fallback chain, relay and meter
the stream, reconcile, record -- and any stage can end the request with a well-formed OpenAI error.
The gateway decides *whether* a request runs and *which* model, provider, region or pool serves it;
the 05 router picks the replica inside a pool, and the engine batches.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from . import api, cache as cachemod, keys as keymod, otel
from .metering import Ledger, row_from_usage
from .providers import CATALOGUE, normalize_usage
from .ratelimit import ReserveLimiter
from .routing import falls_through


@dataclass
class Result:
    status: int
    headers: dict = field(default_factory=dict)
    body: dict | None = None                       # a non-streamed answer, or an error
    frames: list = field(default_factory=list)     # the SSE frames the client received
    served_by: object = None
    attempts: list = field(default_factory=list)   # (target, outcome) per target tried or skipped
    row: object = None                             # the ledger row
    cache: str = "miss"


class Gateway:
    def __init__(self, *, keys, router, providers: dict, clock, tracer=None, ledger=None, cache=None, screener=None,
                 output_cap: int = 4096, timeout: float = 10.0, salt_secret: bytes = b"gateway-salt-secret"):
        self.keys, self.router, self.providers, self.clock = keys, router, providers, clock
        self.tracer, self.ledger = tracer or otel.Tracer(clock), ledger or Ledger()
        self.cache, self.screener, self.output_cap, self.timeout, self.salt_secret = cache, screener, output_cap, timeout, salt_secret
        self.limiters: dict = {}
        self.ids = itertools.count(1)

    def _fail(self, span, status: int, message: str, type_: str, code: str | None = None, headers: dict | None = None,
              attempts: list | None = None) -> Result:
        self.tracer.end(span, error=code or str(status), **{"http.response.status_code": status})
        return Result(status, headers or {}, api.error_body(status, message, type_, code), attempts=attempts or [])

    def _limiter(self, vk) -> ReserveLimiter | None:
        if vk.tpm is None:
            return None
        return self.limiters.setdefault(vk.tenant, ReserveLimiter(vk.tpm))

    def handle(self, api_key: str, request: dict) -> Result:
        rid, t_start = f"req-{next(self.ids)}", self.clock.now()
        stream, wants_usage = bool(request.get("stream")), bool((request.get("stream_options") or {}).get("include_usage"))
        server = self.tracer.start("POST /v1/chat/completions", "SERVER", **{otel.REQUEST_MODEL: request["model"]})
        try:                                                        # 1. the tenant comes from the verified key
            vk = self.keys.verify(api_key)
        except PermissionError as e:
            return self._fail(server, 401, str(e), "authentication_error", "invalid_api_key")
        ok, why = self.keys.authorize(vk, request["model"])
        if not ok:
            return self._fail(server, 403, why, "permission_error", "key_not_allowed")
        server["attrs"]["gw.tenant"] = vk.tenant
        text = api.prompt_text(request)                            # 2. screen the input
        if self.screener and (verdict := self.screener.check(text, "input")).block:
            return self._fail(server, 400, f"blocked by input guardrail: {verdict.findings}", "invalid_request_error", "content_policy")
        ckey = None                                                 # 3. exact cache, namespaced by tenant
        if self.cache is not None and cachemod.cacheable(request)[0]:
            ckey = cachemod.exact_key(vk.tenant, request)
            if (hit := self.cache.get(ckey, t_start)) is not None:
                self.tracer.end(server, **{"gw.cache": "hit"})
                return self._replay(rid, request, hit, stream, wants_usage)
        cap = min(request.get("max_completion_tokens") or self.output_cap, self.output_cap)
        prompt_est, lim = api.estimate_tokens(text), self._limiter(vk)
        if lim and not lim.admit(rid, prompt_est + cap, t_start):   # 4. reserve prompt + output cap
            return self._fail(server, 429, "tenant tokens-per-minute limit", "rate_limit_error", "tokens",
                              {"retry-after": str(lim.retry_after(t_start))})
        upstream = {k: v for k, v in request.items() if k != "metadata"} | {"max_completion_tokens": cap}
        if stream:
            upstream["stream_options"] = {"include_usage": True}   # always meter; strip the chunk if not asked
        attempts, debited = [], 0
        for target in self.router.candidates(request["model"], upstream, now=t_start, tier=vk.tier, regions=vk.regions,
                                             prompt_tokens=prompt_est, request_id=rid):   # 5. walk the chain
            if not self.router.allow(target, self.clock.now()):
                attempts.append((target, "breaker open"))
                continue
            provider, dialect = self.providers[target.provider], CATALOGUE[target.model].dialect
            wire = getattr(provider, "dialect", "openai")         # the format on the wire: what usage is normalised from
            body = {**upstream, "model": target.model}
            if dialect == "vllm":                                   # engine prefix-cache isolation per tenant
                body["cache_salt"] = keymod.cache_salt(vk.tenant, self.salt_secret)
            span = self.tracer.start(f"chat {target.model}", "CLIENT", server, **{
                otel.OPERATION_NAME: "chat", otel.REQUEST_MODEL: target.model, otel.PROVIDER_NAME: otel.PROVIDER_VALUE[dialect],
                otel.SERVER_ADDRESS: target.provider, **({otel.REQUEST_STREAM: True} if stream else {})})
            t0, resp = self.clock.now(), provider.chat(body)
            if resp.status != 200:
                if resp.status is None:
                    self.clock.sleep(self.timeout)                  # no answer: our timeout decides
                code = ((resp.body or {}).get("error") or {}).get("code")
                if resp.status is None or resp.status == 429 or resp.status >= 500:
                    self.router.observe(target, False, self.clock.now())   # only the provider's health trips its breaker
                self.tracer.end(span, error=str(code or resp.status or "timeout"))
                attempts.append((target, resp.status or "timeout"))
                if falls_through(resp.status, code):
                    continue
                if lim:
                    lim.finish(rid)
                if resp.status == 400:                              # the client's request: the same everywhere
                    return self._fail(server, 400, resp.body["error"].get("message", ""), "invalid_request_error", code,
                                      attempts=attempts)
                return self._fail(server, 502, f"upstream {target.provider} refused our credentials or model id",
                                  "api_error", "upstream_configuration", attempts=attempts)   # ours, not the caller's
            frames, first, acc = [], None, api.StreamAccumulator()  # 6. relay and meter
            if stream:
                for ev in resp.events:
                    acc.add(ev)
                    if ev == api.DONE or "error" in ev:
                        break
                    if ev.get("usage") is not None and not ev.get("choices"):
                        frames += [api.sse(ev)] if wants_usage else []   # the usage chunk nobody asked for is stripped
                        continue
                    first = self.clock.now() if first is None else first
                    frames.append(api.sse(ev))
                    delta = ev["choices"][0]["delta"] if ev.get("choices") else {}
                    n = max(1, api.estimate_tokens((delta.get("content") or "") + (delta.get("reasoning") or "")))
                    if lim:
                        lim.debit(rid, n, self.clock.now())
                        debited += n
                if acc.error and first is None:                     # failed before the first byte: fall through
                    self.router.observe(target, False, self.clock.now())
                    self.tracer.end(span, error="stream_error")
                    attempts.append((target, "error before first byte"))
                    continue
                if acc.error:                                       # after it: surface the error, never splice
                    frames.append(api.sse(api.normalize_error({"error": acc.error}, 500)))
                frames.append(api.sse(api.DONE))
                raw_usage, finish = acc.usage, ("error" if acc.error else acc.finish_reason)
            else:
                first, raw_usage = self.clock.now(), resp.body.get("usage")
                finish = resp.body["choices"][0]["finish_reason"]
            usage = normalize_usage(wire, raw_usage) if raw_usage else {      # 7. reconcile: usage is the bill
                "prompt_tokens": prompt_est, "completion_tokens": acc.output_estimate(), "cached_tokens": 0, "reasoning_tokens": 0}
            if lim:
                correction = usage["prompt_tokens"] + usage["completion_tokens"] - debited
                if correction > 0:
                    lim.debit(rid, correction, self.clock.now())
                else:                                               # we over-estimated while streaming: give it back
                    lim.meter.add(self.clock.now(), correction)
                lim.finish(rid)
            self.router.observe(target, not acc.error, self.clock.now(), first - t0)
            now = self.clock.now()
            row = row_from_usage(rid, vk.tenant, vk.key_id, target.model, usage, estimated=raw_usage is None,
                                 status="cut" if acc.error else "ok", ttft=first - t_start, duration=now - t_start,
                                 trace_id=server["traceId"])
            self.ledger.add(row)
            self.keys.charge(vk, row.cost)
            gen = {otel.RESPONSE_MODEL: target.model, otel.INPUT_TOKENS: usage["prompt_tokens"],
                   otel.OUTPUT_TOKENS: usage["completion_tokens"], otel.CACHE_READ: usage.get("cached_tokens", 0),
                   otel.REASONING: usage.get("reasoning_tokens", 0), otel.FINISH_REASONS: [finish]}
            self.tracer.end(span, error="stream_error" if acc.error else None, **gen, **{otel.TIME_TO_FIRST_CHUNK: first - t0})
            self.tracer.end(server, error="stream_error" if acc.error else None, **gen, **{"gw.attempts": len(attempts) + 1})
            answer = acc.text() if stream else resp.body["choices"][0]["message"]["content"]
            if ckey and finish == "stop":                               # 8. cache only complete answers
                self.cache.put(ckey, answer, now)
            attempts.append((target, "ok" if not acc.error else "cut after first byte"))
            headers = {"x-gateway-target": f"{target.provider}/{target.model}"}
            return Result(200, headers, None if stream else resp.body, frames, target, attempts, row)
        if lim:
            lim.finish(rid)
        if not attempts:                                            # filters left nothing: capabilities, context, region
            return self._fail(server, 400, "no target in the chain can serve this request", "invalid_request_error",
                              "no_capable_target")
        return self._fail(server, 503, "every target in the chain failed", "service_unavailable_error", "no_healthy_target",
                          {"retry-after": "1"}, attempts)

    def _replay(self, rid: str, request: dict, answer: str, stream: bool, wants_usage: bool) -> Result:
        zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not stream:
            body = {"id": rid, "object": "chat.completion", "model": request["model"], "usage": zero,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]}
            return Result(200, {"x-cache": "hit"}, body, cache="hit")
        frames = [api.sse(api.chunk(rid, request["model"], 0, content=answer, finish_reason="stop"))]
        frames += [api.sse(api.chunk(rid, request["model"], 0, usage=zero))] if wants_usage else []
        return Result(200, {"x-cache": "hit"}, None, frames + [api.sse(api.DONE)], cache="hit")
