"""The scheduling cycle of an endpoint picker, and a record of why it chose what it chose.

The one idea: one routing decision =

    1. producers   compute per-request data for every candidate (prefix matches, in-flight load)
    2. filters     run in order; each narrows the set (an empty set means HTTP 503)
    3. scorers     each returns scores in [0,1] for the survivors; total = sum(weight * score)
    4. picker      turns totals into a choice (max score, ties broken round-robin)
    5. pre-request producers record the choice (the prefix index learns "E has this prefix now")

`Decision.table()` prints the per-scorer columns so you can see *which* signal won — the most
useful debugging view of a cache-aware router (the real EPP can emit the same scores as
Envoy metadata with `--emit-endpoint-scores`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .plugins import RequestCtx, clamp

__all__ = ["Decision", "Scheduler"]


@dataclass
class Decision:
    request_id: str
    candidates: list
    stages: list = field(default_factory=list)          # [(filter name, [survivor names])]
    scores: dict = field(default_factory=dict)          # scorer name -> {endpoint: raw score}
    weights: dict = field(default_factory=dict)         # scorer name -> weight
    totals: dict = field(default_factory=dict)          # endpoint -> weighted total
    picked: list = field(default_factory=list)          # endpoint names, best first
    reason: str = ""
    duration_s: float = 0.0

    @property
    def endpoint(self) -> str | None:
        return self.picked[0] if self.picked else None

    def table(self) -> str:
        names = list(self.totals) or self.candidates
        cols = list(self.scores)
        head = f"{'endpoint':<10}" + "".join(f"{c[:22]:>24}" for c in cols) + f"{'total':>9}"
        lines = [head, "-" * len(head)]
        for n in names:
            cells = "".join(
                f"{(f'{self.scores[c][n]:.3f} x{self.weights[c]:g}' if n in self.scores[c] else 'unscored'):>24}"
                for c in cols)
            mark = "  <- picked" if self.picked and n == self.picked[0] else ""
            lines.append(f"{n:<10}{cells}{self.totals.get(n, 0.0):>9.3f}{mark}")
        for f, surv in self.stages:
            lines.append(f"filter {f}: kept {surv}")
        if self.reason:
            lines.append(self.reason)
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {"request_id": self.request_id, "candidates": self.candidates, "stages": self.stages,
                "scores": self.scores, "weights": self.weights, "totals": self.totals, "picked": self.picked,
                "reason": self.reason, "duration_ms": round(self.duration_s * 1e3, 3)}


class Scheduler:
    """Runs one PickerConfig against a Datastore."""

    def __init__(self, config, datastore):
        self.config, self.ds = config, datastore

    def schedule(self, ctx: RequestCtx, endpoints=None) -> Decision:
        t0 = time.perf_counter()
        eps = list(endpoints) if endpoints is not None else self.ds.list()
        d = Decision(request_id=ctx.request_id, candidates=[e.name for e in eps])
        if not eps:
            d.reason = "no endpoints in the pool"
            return d
        for p in self.config.producers:
            p.produce(ctx, eps)
        prof = self.config.profile
        survivors = eps
        for f in prof.filters:
            survivors = f.filter(ctx, survivors)
            d.stages.append((f.name, [e.name for e in survivors]))
            if not survivors:
                d.reason = f"filter {f.name} eliminated every endpoint"
                d.duration_s = time.perf_counter() - t0
                return d
        totals = {e.name: 0.0 for e in survivors}
        for scorer, w in prof.scorers:
            raw = scorer.score(ctx, survivors)
            d.scores[scorer.name] = raw
            d.weights[scorer.name] = w
            for n, s in raw.items():
                if n in totals:
                    totals[n] += clamp(s) * w
        d.totals = totals
        by_name = {e.name: e for e in survivors}
        picked = prof.picker.pick(ctx, [(by_name[n], t) for n, t in totals.items()])
        d.picked = [e.name for e in picked]
        d.duration_s = time.perf_counter() - t0
        return d

    # lifecycle hooks, called by the proxy
    def pre_request(self, ctx, ep):
        for p in self.config.producers:
            p.pre_request(ctx, ep)

    def on_first_token(self, ctx, ep):
        for p in self.config.producers:
            p.on_first_token(ctx, ep)

    def on_complete(self, ctx, ep):
        for p in self.config.producers:
            p.on_complete(ctx, ep)
