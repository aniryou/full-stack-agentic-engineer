"""workload.py — from measured output lengths to the serving shape of a thinking workload.

One idea: thinking turns a serving workload *output-heavy*. A request that reads P prompt tokens and
writes L output tokens holds KV for P + t tokens at decode step t, so over its life it occupies

    KV token-steps = Σ_{t=1..L} (P + t)  =  P·L + L(L+1)/2        (≈ P·L + L²/2)

— quadratic in L. Ten times the output is more than ten times the KV × time: the working set per
request grows, fewer requests fit the KV pool, each decode step reads more KV bytes (ITL rises with
the *context* of the batch, not just its size), and requests live ten times longer, so Little's law
(concurrency = rate × duration) multiplies the concurrency a rate needs. TTFT barely moves; ITL and
KV capacity become the binding constraints (PRIMER §7 "What thinking does to serving").

This module (1) summarises length distributions measured from a server (``length_stats``,
``fit_lognormal``) — thinking lengths are heavy-tailed, so p99 ≫ p50 and the mean is not enough;
(2) derives the serving shape analytically (``derive_shape``) with the same roofline step model as
:mod:`thinklab.engine`; (3) reproduces the capacity primer's arithmetic (``capacity_primer_view``,
00-foundations/gpu-capacity-planning/PRIMER.md and ``capacity.py``: Little's law, KV per session, sessions
per GPU) for a no-thinking baseline and a 10× thinking output; (4) drives open-loop load with long
outputs against any OpenAI-compatible server (``run_open_loop``), timing TTFT, time to first *content*
token and ITL from the stream; and (5) compares no-thinking / thinking / budgeted thinking in virtual
time on the engine emulator (``simulate_modes``, labelled simulated).
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import fakemodel
from .engine import EngineConfig, Profile, simulate
from .parsers import REASONING_FIELDS
from .thinking.client import Completion, request_body


# --- 1. length distributions -------------------------------------------------------------------
def quantile(values: list, q: float) -> float:
    """Linear-interpolated quantile (numpy's default method)."""
    if not values:
        return math.nan
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass
class LengthStats:
    n: int
    mean: float
    p50: float
    p90: float
    p99: float
    max: float

    @property
    def tail_ratio(self) -> float:
        """p99 / p50: ~1 for fixed lengths, e^(2.33σ) for a log-normal with spread σ."""
        return self.p99 / self.p50 if self.p50 else math.inf

    def row(self, label: str) -> dict:
        return {"": label, "n": self.n, "mean": round(self.mean), "p50": round(self.p50), "p90": round(self.p90),
                "p99": round(self.p99), "max": round(self.max), "p99/p50": round(self.tail_ratio, 1)}


def length_stats(values: list) -> LengthStats:
    v = [float(x) for x in values]
    return LengthStats(len(v), statistics.fmean(v) if v else math.nan, quantile(v, 0.5), quantile(v, 0.9),
                       quantile(v, 0.99), max(v) if v else math.nan)


def fit_lognormal(values: list) -> tuple:
    """(median, σ) of a log-normal fitted by the moments of log(length) — the usual model of thinking lengths."""
    logs = [math.log(x) for x in values if x > 0]
    mu = statistics.fmean(logs)
    return math.exp(mu), statistics.pstdev(logs)


def from_completions(comps: list) -> dict:
    """Reasoning, answer and total output lengths of measured completions (usage-based)."""
    ok = [c for c in comps if c.ok]
    r = [c.reasoning_tokens or 0 for c in ok]
    return {"reasoning": length_stats(r), "answer": length_stats([c.completion_tokens - x for c, x in zip(ok, r)]),
            "total": length_stats([c.completion_tokens for c in ok])}


# --- 2. the serving shape ----------------------------------------------------------------------
def kv_token_steps(prompt: int, output: int) -> int:
    """Σ_{t=1..L} (P + t) = P·L + L(L+1)/2 — the KV a request holds, summed over its decode steps."""
    return prompt * output + output * (output + 1) // 2


@dataclass
class Shape:
    label: str
    prompt: float
    mean_output: float
    p99_output: float
    kv_peak_mean_gb: float           # (P + L) × bytes, averaged over requests
    kv_peak_p99_gb: float            # (P + L_p99) × bytes: what max_model_len must admit
    kv_time_avg_gb: float            # KV held on average over a request's life
    batch_by_memory: int             # concurrent requests the KV pool holds (time-averaged working sets)
    itl_ms: float                    # decode step at that batch (simulated roofline)
    duration_s: float                # prefill + mean output × ITL
    req_per_s: float                 # Little's law: batch / duration
    out_tok_per_s: float
    bound: str                       # "memory (KV)" or "max_num_seqs"

    def row(self) -> dict:
        return {"": self.label, "out mean": round(self.mean_output), "out p99": round(self.p99_output),
                "KV/req GB": round(self.kv_time_avg_gb, 3), "batch": self.batch_by_memory,
                "ITL ms": round(self.itl_ms, 1), "req s": round(self.duration_s, 1),
                "req/s/GPU": round(self.req_per_s, 2), "tok/s/GPU": round(self.out_tok_per_s)}


def derive_shape(prof: Profile, prompt_tokens: float, outputs: list, label: str = "", max_num_seqs: int = 256) -> Shape:
    """Steady-state shape of one engine serving requests with these output lengths (simulated
    roofline: see :class:`thinklab.engine.Profile`). The batch is whatever the KV pool holds when
    every running request sits at its *time-averaged* context P + L/2, capped by ``max_num_seqs``."""
    b = prof.kv_bytes_per_token
    mean_l = statistics.fmean(outputs)
    p99 = quantile(outputs, 0.99)
    avg_ctx = prompt_tokens + mean_l / 2
    by_mem = int(prof.kv_capacity_tokens // avg_ctx)
    batch = max(1, min(by_mem, max_num_seqs))
    itl = prof.decode_step_s(batch, batch * avg_ctx)
    ttft = prof.step_s(int(prompt_tokens), 0, prompt_tokens)
    duration = ttft + mean_l * itl
    return Shape(label, prompt_tokens, mean_l, p99, (prompt_tokens + mean_l) * b / 1e9, (prompt_tokens + p99) * b / 1e9,
                 avg_ctx * b / 1e9, batch, itl * 1e3, duration, batch / duration, batch / itl,
                 "memory (KV)" if by_mem < max_num_seqs else "max_num_seqs")


def capacity_primer_view(rps: float, in_tokens: int, out_tokens: int, *, params_b: float = 24, layers: int = 40,
                         kv_heads: int = 8, head_dim: int = 128, hbm_gb: float = 80, fp8_tflops: float = 1979,
                         tpot_ms: float = 40, mfu: float = 0.5, overhead: float = 0.10) -> dict:
    """The capacity primer's worked arithmetic (``00-foundations/gpu-capacity-planning/capacity.py``,
    its "Singapore bank" example: Mistral Small 3 24B on H100, fp8), re-implemented so this lab stays
    standalone — ``tests/test_reuse.py`` checks it against ``capacity.py`` itself. Units as there:
    weights in 1e9 bytes, KV per session in KB/1024² (the primer's mixed convention, kept on purpose)."""
    ttft = 2 * params_b * 1e9 * in_tokens / (fp8_tflops * 1e12 * mfu)
    duration = ttft + out_tokens * tpot_ms / 1000
    conc = rps * duration
    avg_ctx = in_tokens + out_tokens // 2
    kv_session_gb = (2 * layers * kv_heads * head_dim * 1.0 / 1024) * avg_ctx / (1024 * 1024)
    spare = hbm_gb * (1 - overhead) - params_b * 1.0
    sessions = spare / kv_session_gb
    return {"ttft_s": ttft, "duration_s": duration, "concurrency": conc, "avg_ctx": avg_ctx,
            "kv_per_session_gb": kv_session_gb, "sessions_per_gpu": sessions, "gpus_for_memory": conc / sessions,
            "decode_tok_s_needed": rps * out_tokens}


# --- 3. open-loop load against a server ----------------------------------------------------------
@dataclass
class Result:
    completion: Completion
    start: float
    tag: str = ""


@dataclass
class Run:
    results: list
    duration_s: float
    rate: float
    simulated: bool
    target: str = ""
    params: dict = field(default_factory=dict)

    def summary(self, tag: str | None = None) -> dict:
        rs = [r for r in self.results if r.completion.ok and (tag is None or r.tag == tag)]
        c = [r.completion for r in rs]
        itl = [x for cc in c for x in cc.itl]
        ms = lambda v: round(1e3 * v, 1)                                    # noqa: E731
        return {"label": ("SIMULATED" if self.simulated else "MEASURED") + (f" {tag}" if tag else ""),
                "completed": len(c), "failed": sum(not r.completion.ok for r in self.results if tag is None or r.tag == tag),
                "out_tok_per_s": round(sum(x.completion_tokens for x in c) / self.duration_s, 1),
                "reasoning_share": round(sum(x.reasoning_tokens or 0 for x in c) / max(1, sum(x.completion_tokens for x in c)), 3),
                "ttft_p50_ms": ms(quantile([x.ttft for x in c], .5)), "ttfc_p50_ms": ms(quantile([x.ttfc for x in c if x.ttfc == x.ttfc], .5)),
                "ttfc_p99_ms": ms(quantile([x.ttfc for x in c if x.ttfc == x.ttfc], .99)),
                "itl_p50_ms": ms(quantile(itl, .5)), "itl_p99_ms": ms(quantile(itl, .99)),
                "e2e_p50_s": round(quantile([x.latency for x in c], .5), 2), "e2e_p99_s": round(quantile([x.latency for x in c], .99), 2)}


async def _stream(session, url: str, body: dict, headers: dict) -> Completion:
    c, last, t0 = Completion(), None, time.perf_counter()
    reasoning, content, buf = [], [], b""
    try:
        async with session.post(url.rstrip("/") + "/v1/chat/completions", json=body, headers=headers) as resp:
            if resp.status != 200:
                c.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return c
            async for chunk in resp.content.iter_any():
                buf += chunk
                while b"\n\n" in buf:
                    event, buf = buf.split(b"\n\n", 1)
                    line = event.decode().strip()
                    if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                        continue
                    data = json.loads(line[5:])
                    if data.get("usage"):
                        u = data["usage"]
                        c.prompt_tokens, c.completion_tokens = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                        c.reasoning_tokens = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
                        c.cached_tokens = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                    for ch in data.get("choices") or []:
                        d = ch.get("delta") or {}
                        r_piece = next((d[f] for f in REASONING_FIELDS if d.get(f)), None)
                        c_piece = d.get("content") or None
                        now = time.perf_counter()
                        if r_piece or c_piece:
                            if math.isnan(c.ttft):
                                c.ttft = now - t0
                            elif last is not None:
                                c.itl.append(now - last)
                            last = now
                        if r_piece:
                            reasoning.append(r_piece)
                        if c_piece:
                            if math.isnan(c.ttfc):
                                c.ttfc = now - t0
                            content.append(c_piece)
                        if ch.get("finish_reason"):
                            c.finish_reason = ch["finish_reason"]
    except Exception as e:  # noqa: BLE001 — a failed request is data
        c.error = f"{type(e).__name__}: {e}"
    c.latency = time.perf_counter() - t0
    c.reasoning, c.content = "".join(reasoning).strip() or None, "".join(content) or None
    return c


async def open_loop(url: str, requests: list, rate: float, seed: int = 0, headers: dict | None = None,
                    model: str | None = None) -> Run:
    """Send ``requests`` = [(messages, request_body kwargs, tag)] with Poisson arrivals at ``rate``/s."""
    import aiohttp
    rng = random.Random(seed)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s:
        if model is None:
            async with s.get(url.rstrip("/") + "/v1/models", headers=headers or {}) as r:
                model = (await r.json())["data"][0]["id"]
        simulated = False
        try:
            async with s.get(url.rstrip("/") + "/version") as r:
                simulated = bool((await r.json(content_type=None)).get("simulated"))
        except Exception:  # noqa: BLE001
            pass
        t0 = time.perf_counter()
        tasks, t = [], 0.0
        for msgs, kw, tag in requests:
            await asyncio.sleep(max(0.0, t0 + t - time.perf_counter()))
            body = request_body(msgs, model=model, stream=True, **kw)
            start = time.perf_counter()
            tasks.append((asyncio.create_task(_stream(s, url, body, headers or {})), start, tag))
            t += rng.expovariate(rate) if rate != math.inf else 0.0
        results = [Result(await task, start - t0, tag) for task, start, tag in tasks]
    return Run(results, time.perf_counter() - t0, rate, simulated, url, {"n": len(requests), "seed": seed})


def run_open_loop(url: str, requests: list, rate: float, **kw) -> Run:
    """Blocking wrapper that also works inside Jupyter (runs the loop in a worker thread)."""
    with ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, open_loop(url, requests, rate, **kw)).result()


# --- 4. modes compared in virtual time ------------------------------------------------------------
def sample_lengths(questions: list, *, thinking: bool = True, budget: int | None = None, max_tokens: int | None = None,
                   card: fakemodel.ModelCard = fakemodel.SMALL, seed: int = 0) -> list:
    """(reasoning tokens, answer tokens, correct) per question from the simulated model — no server."""
    out = []
    for i, q in enumerate(questions):
        s = fakemodel.generate(card, q, thinking=thinking, budget=budget, max_tokens=max_tokens, seed=seed, sample_index=i)
        extra = len(fakemodel.EARLY_STOP.split()) if s.forced_stop else 0
        out.append((len(s.reasoning_tokens) + extra, len(s.content_tokens), bool(s.correct)))
    return out


def simulate_modes(prof: Profile, questions: list, modes: dict, rate: float, prompt_tokens: int = 60,
                   cfg: EngineConfig | None = None, seed: int = 0) -> list:
    """Run the same arrival process through the engine emulator once per mode.

    ``modes``: name → kwargs for :func:`sample_lengths` (e.g. {"thinking": False}). Returns one summary
    row per mode — all **simulated**: TTFT, time to first content token, ITL, E2E, KV usage,
    preemptions, accuracy and output tokens per correct answer."""
    rows = []
    for name, kw in modes.items():
        kw = {"max_tokens": prof.max_model_len - prompt_tokens, **kw}    # a server caps unbounded requests there
        lens = sample_lengths(questions, seed=seed, **kw)
        rng = random.Random(seed)
        t, reqs = 0.0, []
        for r, a, _ in lens:
            reqs.append((t, prompt_tokens, max(1, r + a), r))
            t += rng.expovariate(rate)
        eng = simulate(prof, reqs, cfg)
        fin = eng.finished
        itl = [x for q in fin for x in q.itls]
        acc = sum(c for _, _, c in lens) / len(lens)
        out_tokens = statistics.fmean(r + a for r, a, _ in lens)
        rows.append({"mode": name, "accuracy": round(acc, 3), "out mean": round(out_tokens),
                     "TTFT p50 ms": round(1e3 * quantile([q.first_token - q.arrival for q in fin], .5)),
                     "answer starts p50 s": round(quantile([q.first_content - q.arrival for q in fin
                                                            if q.first_content is not None], .5), 2),
                     "ITL p50 ms": round(1e3 * quantile(itl, .5), 1), "ITL p99 ms": round(1e3 * quantile(itl, .99), 1),
                     "E2E p99 s": round(quantile([q.finished_at - q.arrival for q in fin], .99), 1),
                     "KV peak": round(max(u for _, u, _, _ in eng.kv_samples), 2),
                     "max running": max(r for _, _, r, _ in eng.kv_samples),
                     "preemptions": eng.preemptions,
                     "tokens/correct": round(out_tokens / acc) if acc else math.inf})
    return rows


def run_with_gauges(url: str, requests: list, rate: float, interval_s: float = 0.05, headers: dict | None = None,
                    **kw) -> tuple:
    """:func:`run_open_loop` while a thread scrapes ``/metrics`` every ``interval_s``: returns
    ``(run, samples)`` with samples = [(t, kv_usage, running, waiting)] — the gauges a dashboard would plot."""
    import threading

    from . import metrics as M
    samples, stop = [], threading.Event()

    def poll():
        t0 = time.perf_counter()
        while not stop.is_set():
            try:
                s = M.scrape(url, headers=headers, timeout=5)
                samples.append((time.perf_counter() - t0, M.value(s, M.KV_USAGE), M.value(s, M.RUNNING), M.value(s, M.WAITING)))
            except Exception:  # noqa: BLE001 — a missed scrape is not a failed run
                pass
            stop.wait(interval_s)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    try:
        run = run_open_loop(url, requests, rate, headers=headers, **kw)
    finally:
        stop.set()
        th.join(timeout=10)
    return run, samples
