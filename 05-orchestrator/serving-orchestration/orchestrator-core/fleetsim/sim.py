"""The event loop: arrivals -> router -> replica steps -> completions, plus the autoscaler's clock.

Simulated time only, deterministic under the seeds you pass in. Each replica runs one engine step at a time
(duration from the roofline step model). The router decides at arrival from what it can see: its own dispatch
counters (fresh) and replica metrics scraped at most `metrics_age` s ago (llm-d's EPP refreshes every 50 ms).
Staleness is modelled as that maximum age: a snapshot is re-read on the first look after it expires, so its real
age varies between 0 and `metrics_age`, with no scrape jitter or per-replica phase.

With `flow_control=FlowControl(...)` the router holds requests in its own queue instead of committing them to an
engine (llm-d flow control, §3 of the primer): see FlowControl.

With `prefill=k`, the first k replicas are a prefill-only pool (P/D disaggregation, llm-d style): the router
first picks a decode replica; if that replica would have to prefill >= `pd_threshold` uncached tokens (upstream:
the `prefix-based-pd-decider`'s `nonCachedTokens`), a prefill replica computes the prompt, the KV the decode
replica lacks crosses the link (`disagg.transfer_s`), and decoding continues on the chosen decode replica.
Shorter prompts are served locally (conditional disaggregation). P/D fleets are not autoscaled here.
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

from .disagg import transfer_s
from .replica import EngineProfile, Replica
from .routers import LeastOutstanding, RoundRobin
from .workload import expand

ARRIVE, STEP, READY, XFER, TICK = range(5)


@dataclass(frozen=True)
class FlowControl:
    """Router-side queueing, the shape of the llm-d EPP's flow control with a `concurrency-detector` (requests
    mode, no headroom): a request is dispatched only to an endpoint with fewer than `max_concurrency` requests in
    flight from this router; otherwise it waits in the router — highest `Request.priority` first, FCFS within a
    band (the default global-strict fairness + fcfs ordering) — until a request finishes somewhere. A request that
    waited longer than `ttl_s` is rejected when it reaches the head (defaultRequestTTL); one that finds `max_queue`
    requests queued is rejected on arrival (maxRequests). Rejections are counted as `shed` (a 429/503 upstream)."""
    max_concurrency: int = 100              # per endpoint (upstream default 100; tune it with Little's law)
    ttl_s: float = math.inf
    max_queue: int | None = None


@dataclass
class Result:
    requests: list
    replicas: list
    itl: list = field(repr=False)
    timeline: list = field(repr=False)
    duration: float = 0.0
    label: str = "SIMULATED by fleetsim (a model of an engine, not a measurement)"

    def summary(self, **slo) -> dict:
        from .metrics import summarize
        return summarize(self, **slo)


class Fleet:
    """`replicas` engines of one `profile` behind `router`. Optional: an `autoscaler` (ticks every sync period;
    new replicas serve after `cold_start_s`), `metrics_age` (max age of scraped metrics), a prefill pool of
    `prefill` replicas joined to the rest by a `link_gbps` link, and `sample_s` (timeline rows without an
    autoscaler). One `run()` per Fleet."""

    def __init__(self, profile: EngineProfile, replicas=4, router=None, *, autoscaler=None, cold_start_s=0.0,
                 metrics_age=0.05, prefill=0, link_gbps=100.0, link_latency_s=0.001, pd_threshold=0,
                 prefill_router=None, sample_s=0.0, flow_control: FlowControl | None = None):
        if prefill and (autoscaler or flow_control):
            raise ValueError("autoscaling or flow control of a P/D fleet is out of scope for this core")
        if prefill and prefill >= replicas:
            raise ValueError("a P/D fleet needs at least one decode replica (prefill < replicas)")
        if flow_control and flow_control.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.p, self.router = profile, router or RoundRobin()
        self.fc, self.queue = flow_control, []         # the router's own queue: (-priority, arrival, rid, request)
        self.prefill_router = prefill_router or LeastOutstanding()
        self.autoscaler, self.cold_start_s, self.metrics_age = autoscaler, cold_start_s, metrics_age
        self.link_gbps, self.link_latency_s, self.pd_threshold = link_gbps, link_latency_s, pd_threshold
        self.sample_s = autoscaler.hpa.sync_s if autoscaler else sample_s
        self._ids = itertools.count()
        self.all = [self._new(0.0, True, "prefill" if i < prefill else "both") for i in range(replicas)]
        self.replicas, self.pending, self.timeline = list(self.all), [], []
        if autoscaler:                                 # warm start: as if it had been recommending this size
            autoscaler.hpa.recs.append((0.0, replicas))

    def _new(self, now, ready, role="both"):
        return Replica(next(self._ids), self.p, role=role, now=now, ready=ready, max_age=self.metrics_age)

    def _push(self, t, kind, obj):
        heapq.heappush(self._heap, (t, next(self._seq), kind, obj))

    def run(self, requests, horizon: float | None = None) -> Result:
        """Simulate until every request (and every later session turn) finishes, or past `horizon` for ticks."""
        self._heap, self._seq, self.horizon = [], itertools.count(), horizon
        self.total, self.done = len(expand(requests)), 0
        for r in requests:
            self._push(r.arrival, ARRIVE, r)
        if self.sample_s:
            self._push(0.0, TICK, None)
        t = 0.0
        while self._heap:
            t, _, kind, obj = heapq.heappop(self._heap)
            (self._arrive, self._step_end, self._ready, self._xfer_done, self._tick)[kind](t, obj)
        for r in self.all:
            r.died = t if r.died is None else r.died
        return Result(expand(requests), self.all, [g for r in self.all for g in r.itl], self.timeline, t)

    # -- events ----------------------------------------------------------------------------------------------
    def _arrive(self, t, req):
        pool = [r for r in self.replicas if r.ready and not r.draining]
        decode = [r for r in pool if r.role != "prefill"]
        if not decode:
            self.pending.append(req)                   # nothing ready (scaled to zero / cold start): hold it
            return
        if self.fc:
            decode = self._room(decode)
            if not decode:                             # every endpoint at its cap: wait in the router
                if self.fc.max_queue is not None and len(self.queue) >= self.fc.max_queue:
                    return self._shed(req)
                heapq.heappush(self.queue, (-req.priority, req.arrival, req.rid, req))
                return
        self._dispatch(t, req, pool, decode)

    def _room(self, decode):
        return [r for r in decode if self.router.outstanding[r.rid] < self.fc.max_concurrency]

    def _drain(self, t):
        """A slot freed up: dispatch queued requests, highest priority first, while an endpoint has room."""
        while self.queue:
            room = self._room([r for r in self.replicas if r.ready and not r.draining and r.role != "prefill"])
            if not room:
                return
            req = heapq.heappop(self.queue)[-1]
            if t - req.arrival > self.fc.ttl_s:
                self._shed(req)
            else:
                self._dispatch(t, req, room, room)

    def _shed(self, req):
        req.shed = True
        self.done += len(expand([req]))                # a rejected agent turn ends its session

    def _dispatch(self, t, req, pool, decode):
        d = self.router.pick(req, decode, t)
        self.router.on_dispatch(req, d, t)
        req.t_dispatch, req.replica = t, d.rid
        prefill = [r for r in pool if r.role == "prefill"]
        if prefill and req.prompt - d.cached_tokens(req) >= self.pd_threshold:
            req._decode = d
            p = self.prefill_router.pick(req, prefill, t)
            self.prefill_router.on_dispatch(req, p, t)
            p.enqueue(req, t)
            self._kick(t, p)
        else:
            d.enqueue(req, t)
            self._kick(t, d)

    def _kick(self, t, r):
        if r.busy:
            return
        dur = r.start_step(t)
        if dur is None:
            if r.draining and r.idle():
                self._retire(t, r)
            return
        r.busy = True
        self._push(t + dur, STEP, r)

    def _step_end(self, t, r):
        r.busy = False
        router = self.prefill_router if r.role == "prefill" else self.router
        for event, req in r.end_step(t):
            if event == "first":
                router.on_first_token(req, r)
            elif event == "handoff":
                router.on_first_token(req, r)
                router.on_complete(req, r)             # only the blocks the decode replica lacks cross the link
                missing = req.prompt - req._decode.cached_tokens(req)
                self._push(t + transfer_s(missing, self.p.kv_bytes_per_token, self.link_gbps,
                                          self.link_latency_s), XFER, req)
            else:
                router.on_complete(req, r)
                self.done += 1
                if req.next is not None:               # closed loop: the session's next turn
                    req.next.arrival = t + req.think
                    self._push(req.next.arrival, ARRIVE, req.next)
                if self.fc:
                    self._drain(t)
        self._kick(t, r)

    def _xfer_done(self, t, req):
        d = req._decode
        req.t_first = t                                # first token deliverable once the KV has landed
        self.router.on_first_token(req, d)
        d.enqueue(req, t, kv_ready=True)
        self._kick(t, d)

    def _ready(self, t, r):
        if r.died is not None:
            return                                     # cancelled while it was still starting
        r.ready, r.ready_at = True, t
        held, self.pending = self.pending, []
        for req in held:
            self._arrive(t, req)
        if self.fc:
            self._drain(t)

    def _retire(self, t, r):
        r.died = t
        self.replicas.remove(r)

    def _tick(self, t, _):
        live = [r for r in self.replicas if not r.draining]
        ready = [r for r in live if r.ready]
        util = {}
        for r in ready:                                # "GPU util": fraction of the interval a step was running
            util[r.rid] = min(1.0, (r.busy_time - r._busy_mark) / self.sample_s)
            r._busy_mark = r.busy_time
        row = dict(t=t, replicas=len(live), ready=len(ready), pending=len(self.pending),
                   queued=len(self.queue), waiting=sum(len(r.waiting) for r in ready),
                   running=sum(len(r.running) for r in ready),
                   gpu_util=sum(util.values()) / max(1, len(ready)),
                   kv=sum(r.pool.usage() for r in ready) / max(1, len(ready)))
        if self.autoscaler:
            vals = [self.autoscaler.value(r, util[r.rid]) for r in ready]
            held = self.autoscaler.held(self.pending + [q[-1] for q in self.queue], self.p)   # at the gateway/router
            desired = self.autoscaler.decide(t, len(live), vals, len(live) - len(ready), sum(vals) + held)
            self._resize(t, desired)
            row["desired"] = desired
        self.timeline.append(row)
        unfinished = self.done < self.total and (self._heap or self.autoscaler)    # else nothing can finish them
        if unfinished or (self.horizon and t + self.sample_s <= self.horizon):
            self._push(t + self.sample_s, TICK, None)

    def _resize(self, t, desired):
        live = [r for r in self.replicas if not r.draining]
        for _ in range(desired - len(live)):
            r = self._new(t, False)
            self.all.append(r)
            self.replicas.append(r)
            self._push(t + self.cold_start_s, READY, r)
        for r in sorted(live, key=lambda r: (r.ready, -r.born))[:max(0, len(live) - desired)]:
            r.draining = True                          # stop routing to it; finish what it has, then go
            if not r.ready or r.idle():
                self._retire(t, r)
