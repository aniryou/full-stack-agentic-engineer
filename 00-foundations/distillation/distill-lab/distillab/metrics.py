"""metrics.py — vLLM's Prometheus metrics that a distillation decision reads: throughput and speculation.

One idea: the two numbers this topic needs from a running engine are both *counters*, so they are read as a
difference between two scrapes, never as a single value:

    output tokens per second   Δ vllm:generation_tokens_total / Δt        → cost per million tokens (PRIMER §9)
    draft acceptance           Δ vllm:spec_decode_num_accepted_tokens_total / Δ ..._num_draft_tokens_total
                               mean acceptance length = 1 + Δ accepted / Δ vllm:spec_decode_num_drafts_total
                               per position i: Δ ..._accepted_tokens_per_pos_total{position="i"} / Δ drafts
                                                                                   (PRIMER §7)

Names are those of vLLM v0.30.0 (``vllm/v1/metrics/loggers.py`` and ``vllm/v1/spec_decode/metrics.py``;
counters carry a ``_total`` suffix on the wire). A tiny writer lets the fake teacher export the same names.
Standard library only.
"""
from __future__ import annotations

import math
import re
import urllib.request
from dataclasses import dataclass, field

RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
KV_USAGE = "vllm:kv_cache_usage_perc"
PROMPT_TOKENS = "vllm:prompt_tokens"
GENERATION_TOKENS = "vllm:generation_tokens"
REQUEST_SUCCESS = "vllm:request_success"
SPEC_DRAFTS = "vllm:spec_decode_num_drafts"
SPEC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens"
SPEC_ACCEPTED_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"
COUNTERS = (PROMPT_TOKENS, GENERATION_TOKENS, REQUEST_SUCCESS, SPEC_DRAFTS, SPEC_DRAFT_TOKENS, SPEC_ACCEPTED,
            SPEC_ACCEPTED_PER_POS)

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+([-+0-9.eEinfINFNaN]+)(\s+\d+)?$')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


@dataclass
class Scrape:
    """Parsed exposition text: ``samples[(name, frozenset(labels))] = value``; ``t`` = when it was taken."""
    samples: dict = field(default_factory=dict)
    t: float = math.nan

    def value(self, name: str, default: float = 0.0, **labels) -> float:
        """Sum of every series of ``name`` (``_total`` added for counters) whose labels include ``labels``."""
        names = {name, name + "_total"}
        want = set(labels.items())
        vals = [v for (n, ls), v in self.samples.items() if n in names and want <= set(ls)]
        return sum(vals) if vals else default

    def by_label(self, name: str, label: str) -> dict:
        names = {name, name + "_total"}
        out = {}
        for (n, ls), v in self.samples.items():
            if n in names:
                d = dict(ls)
                if label in d:
                    out[d[label]] = out.get(d[label], 0.0) + v
        return out


def parse(text: str, t: float = math.nan) -> Scrape:
    s = Scrape(t=t)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        labels = frozenset(_LABEL.findall(m.group(3) or ""))
        s.samples[(m.group(1), labels)] = float(m.group(4))
    return s


def scrape(url: str, headers: dict | None = None, timeout: float = 10.0) -> Scrape:
    import time
    req = urllib.request.Request(url.rstrip("/") + "/metrics", headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310
        return parse(r.read().decode(), time.time())


def throughput(before: Scrape, after: Scrape, seconds: float | None = None) -> dict:
    """Tokens per second between two scrapes (``seconds`` overrides the scrape timestamps)."""
    dt = seconds if seconds is not None else after.t - before.t
    gen = after.value(GENERATION_TOKENS) - before.value(GENERATION_TOKENS)
    prompt = after.value(PROMPT_TOKENS) - before.value(PROMPT_TOKENS)
    req = after.value(REQUEST_SUCCESS) - before.value(REQUEST_SUCCESS)
    return {"seconds": dt, "generation_tokens": gen, "prompt_tokens": prompt, "requests": req,
            "output_tok_s": gen / dt if dt and dt > 0 else math.nan}


def spec_decode(before: Scrape, after: Scrape) -> dict:
    """vLLM's own definitions (``SpecDecodingProm`` docstring): acceptance rate = accepted / draft tokens,
    mean acceptance length = 1 + accepted / drafts (it counts the target's bonus token), and the per-position
    rates accepted_per_pos[i] / drafts. With independent per-token acceptance α the per-position rate is α^(i+1),
    so α is the rate at position 0 — not the overall "acceptance rate", which is (mean length − 1) / k."""
    d = lambda n: after.value(n) - before.value(n)  # noqa: E731
    drafts, draft_toks, acc = d(SPEC_DRAFTS), d(SPEC_DRAFT_TOKENS), d(SPEC_ACCEPTED)
    if drafts <= 0:
        return {"drafts": 0.0}
    pos_a, pos_b = after.by_label(SPEC_ACCEPTED_PER_POS, "position"), before.by_label(SPEC_ACCEPTED_PER_POS, "position")
    per_pos = [(pos_a[k] - pos_b.get(k, 0.0)) / drafts for k in sorted(pos_a, key=int)]
    return {"drafts": drafts, "draft_tokens": draft_toks, "accepted": acc, "k": draft_toks / drafts,
            "acceptance_rate": acc / draft_toks if draft_toks else math.nan,
            "mean_acceptance_length": 1 + acc / drafts, "per_position": per_pos,
            "alpha_pos0": per_pos[0] if per_pos else math.nan}


# --- writer (the fake teacher's /metrics) ---------------------------------------------------------------

class Registry:
    """Counters and gauges rendered in Prometheus text format (enough for vLLM's names)."""

    def __init__(self, base_labels: dict | None = None):
        self.base = base_labels or {}
        self.series: dict = {}
        self.kinds: dict = {}

    def _key(self, name, labels):
        return name, tuple(sorted({**self.base, **(labels or {})}.items()))

    def counter(self, name: str, value: float = 1.0, labels: dict | None = None) -> None:
        self.kinds[name] = "counter"
        k = self._key(name, labels)
        self.series[k] = self.series.get(k, 0.0) + value

    def gauge(self, name: str, value: float, labels: dict | None = None) -> None:
        self.kinds[name] = "gauge"
        self.series[self._key(name, labels)] = float(value)

    def render(self) -> str:
        out = []
        for name in sorted(self.kinds):
            kind = self.kinds[name]
            out.append(f"# TYPE {name}{'_total' if kind == 'counter' else ''} {kind}")
            for (n, labels), v in sorted(self.series.items()):
                if n != name:
                    continue
                lab = ",".join(f'{k}="{val}"' for k, val in labels)
                out.append(f"{n}{'_total' if kind == 'counter' else ''}{{{lab}}} {v}")
        return "\n".join(out) + "\n"
