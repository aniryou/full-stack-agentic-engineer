"""bench.py — TTFT, ITL and throughput per quantization scheme: predicted, simulated, then measured.

One idea: a scheme changes two numbers — the bytes a step must read and the FLOP/s its GEMMs run
at — and everything a user feels follows from where each step sits on the roofline. One decode
step reads (nearly) all the weights plus every running sequence's KV; one prefill chunk does
``2 x params x tokens`` FLOPs. So weight-only INT4 cuts decode time (bytes) but not prefill time
(it still multiplies in 16-bit, and the dequantization is not free), FP8 W8A8 halves both on GPUs
with FP8 tensor cores, and FP8 KV cuts the per-sequence part of every decode step:

    step_time = overhead + max(FLOPs / (peak x compute_eff), bytes / (bandwidth x memory_eff))

``gemm_time`` is the same law for one GEMM (it reproduces vllm-internals §8.1's ``down_proj``
table); ``Engine`` is a small continuous-batching emulator (FCFS, token budget, chunked prefill,
KV blocks) driven by that step time — :func:`simulate` runs it on a virtual clock, and
:mod:`quantlab.fakeserver` runs it in real time behind an OpenAI-style API. :func:`run_http`
measures any OpenAI-compatible server (the fake one, or a real ``vllm serve``) from the client,
the way ``vllm bench serve`` does. Everything from the emulator is **simulated**; the efficiencies
are assumptions you can change, not measurements.
"""
from __future__ import annotations

import http.client
import json
import math
import statistics
import threading
import time
import urllib.parse
from dataclasses import dataclass, field

import numpy as np

from . import kv as K
from .serve import GPU, gpu as _gpu


# ---------------------------------------------------------------------------------------------
# One GEMM on the roofline
# ---------------------------------------------------------------------------------------------
GEMM_PATH = {  # scheme -> (weight bytes per weight, activation bytes, precision of the multiply)
    "bf16": (2.0, 2.0, "bf16"), "fp8": (1.0, 1.0, "fp8"), "w8a8-int8": (1.0, 1.0, "int8"),
    "w4a16": ((4 + 16 / 128 + 4 / 128) / 8, 2.0, "bf16"),       # 4.16 bits: group-128 scale + zero point
    "w4a4-nvfp4": (4.5 / 8, 4.5 / 8, "fp4"), "w4a16-nvfp4": (4.5 / 8, 2.0, "bf16"),
}


def gemm_time(M: int, K_: int, N_: int, gpu_name, scheme: str = "bf16") -> float:
    """Seconds for ``[M, K] x [K, N]`` at 100% of peak: ``max(2MKN / peak, (KN w + MK a + MN 2) / bw)``.
    Llama-3.1-8B ``down_proj`` (K 14,336, N 4,096) on an L4: 392 / 102 / 196 us at M = 1 for BF16 /
    W4A16 / FP8 (vllm-internals §8.1)."""
    g = _gpu(gpu_name)
    w, a, prec = GEMM_PATH[scheme]
    flops = 2.0 * M * K_ * N_
    byts = K_ * N_ * w + M * K_ * a + M * N_ * 2
    return max(flops / g.peak(prec), byts / (g.mem_bw_gbs * 1e9))


def crossover_tokens(K_: int, N_: int, gpu_name, fast: str = "w4a16", base: str = "bf16", max_m: int = 1 << 14) -> int:
    """Smallest M at which ``fast`` stops beating ``base`` (W4A16 vs BF16: ~120 tokens on an L4, ~85 on an H100)."""
    for m in range(1, max_m):
        if gemm_time(m, K_, N_, gpu_name, fast) >= gemm_time(m, K_, N_, gpu_name, base) * (1 - 1e-9):
            return m
    return max_m


# ---------------------------------------------------------------------------------------------
# A model + GPU + scheme as a step-time model
# ---------------------------------------------------------------------------------------------
@dataclass
class Profile:
    name: str
    scheme: str
    kv_cache_dtype: str
    params_per_token: float        # matmul parameters one token multiplies through (linear + lm_head)
    weight_bytes: float            # resident weights
    streamed_bytes: float          # weights read by one decode step (all but the input-embedding gather)
    kv_bytes_per_token: float
    peak_flops: float              # of the GEMM path this scheme takes on this GPU
    mem_bw: float                  # bytes/s
    num_blocks: int
    block_size: int = 16
    compute_eff: float = 0.6
    memory_eff: float = 0.8
    overhead_s: float = 0.002      # per step: launches, scheduling, sampling (assumption)
    notes: list = field(default_factory=list)

    def step_time(self, decode_contexts, prefill_tokens: int = 0, prefill_context: int = 0) -> float:
        """One engine step: ``decode_contexts`` is the context length of each decoding sequence."""
        n_dec = len(decode_contexts)
        tokens = n_dec + prefill_tokens
        if tokens == 0:
            return 0.0
        flops = 2.0 * self.params_per_token * tokens
        kv_read = (sum(decode_contexts) + prefill_context) * self.kv_bytes_per_token
        byts = self.streamed_bytes + kv_read
        return self.overhead_s + max(flops / (self.peak_flops * self.compute_eff), byts / (self.mem_bw * self.memory_eff))

    def describe(self) -> str:
        return (f"{self.name} [{self.scheme}, kv {self.kv_cache_dtype}] weights {self.weight_bytes / 1e9:.2f} GB, "
                f"{self.peak_flops / 1e12:.0f} TFLOP/s path, {self.mem_bw / 1e9:.0f} GB/s, {self.num_blocks:,} KV blocks "
                f"(SIMULATED; efficiencies {self.compute_eff:.0%} / {self.memory_eff:.0%} are assumptions)")


SCHEME_WEIGHTS = {"bf16": "bf16", "fp8": "fp8", "fp8-online": "fp8", "w8a8-int8": "w8a8-int8", "w4a16": "w4a16",
                  "nvfp4": "nvfp4"}


def gemm_precision(scheme: str, g: GPU) -> str:
    """The precision the GEMMs run in: FP8 needs sm_89, FP4 needs sm_100; else the weights are dequantized to 16-bit."""
    if scheme in ("fp8", "fp8-online"):
        return "fp8" if g.sm >= 89 else "bf16"
    if scheme == "w8a8-int8":
        if g.sm >= 100:
            raise ValueError("INT8 W8A8 is not supported on compute capability >= 10.0 (vLLM docs)")
        return "int8"
    if scheme == "nvfp4":
        return "fp4" if g.sm >= 100 else "bf16"
    return "bf16"


def profile(model: str = "llama-3.1-8b-instruct", gpu_name="L4", scheme: str = "bf16", *, kv_cache_dtype: str = "auto",
            compute_eff: float = 0.6, memory_eff: float = 0.8, overhead_s: float = 0.002,
            weight_only_eff: float = 1.0, gpu_memory_utilization: float = 0.92) -> Profile:
    """``weight_only_eff`` scales the compute ceiling of dequantizing kernels (W4A16, weight-only FP8/FP4)
    relative to a plain 16-bit GEMM: 1.0 is the pure roofline; below 1 models the dequantization cost."""
    m, g = K.load_shape(model), _gpu(gpu_name)
    p = K.params(m)
    wb = K.weight_bytes(m, SCHEME_WEIGHTS[scheme])
    input_embed = 0 if m.tie_word_embeddings else m.vocab_size * m.hidden_size * 2
    lm_head = m.vocab_size * m.hidden_size
    prec = gemm_precision(scheme, g)
    peak = g.peak(prec)
    notes = []
    dequant = scheme in ("w4a16",) or (scheme != "bf16" and prec == "bf16")
    if dequant:
        peak *= weight_only_eff
        notes.append(f"{scheme} on {g.name}: dequantize to 16-bit, multiply in 16-bit")
    rep = K.size(m, g, weights=SCHEME_WEIGHTS[scheme], kv_cache_dtype=kv_cache_dtype,
                 gpu_memory_utilization=gpu_memory_utilization)
    notes += rep.notes
    return Profile(f"{m.name} on {g.name}", scheme, kv_cache_dtype, p.linear + lm_head, wb, wb - input_embed,
                   K.kv_bytes_per_token(m, kv_cache_dtype), peak, g.mem_bw_gbs * 1e9, rep.num_blocks,
                   compute_eff=compute_eff, memory_eff=memory_eff, overhead_s=overhead_s, notes=notes)


def decode_floor_ms(model: str, gpu_name, scheme: str) -> float:
    """Streamed weight bytes / bandwidth at 100%: the ITL floor at batch 1 (vllm-internals §8.2:
    Llama-3.1-8B on an L4 50.0 / 26.8 / 15.6 ms for BF16 / FP8 / INT4)."""
    p = profile(model, gpu_name, scheme)
    return p.streamed_bytes / p.mem_bw * 1e3


# ---------------------------------------------------------------------------------------------
# A continuous-batching engine emulator
# ---------------------------------------------------------------------------------------------
@dataclass
class Seq:
    rid: int
    arrival: float
    prompt_len: int
    output_len: int
    prefilled: int = 0
    generated: int = 0
    blocks: int = 0
    first_token: float | None = None
    token_times: list = field(default_factory=list)
    finished: float | None = None

    @property
    def context(self) -> int:
        return self.prefilled + self.generated


@dataclass
class Plan:
    decode: list
    prefill: list                  # [(seq, n_tokens)]
    time: float = 0.0


class Engine:
    """FCFS continuous batching: decodes first, then prefill chunks within ``max_num_batched_tokens``;
    a request is admitted only when blocks for its whole prompt + output are free (no preemption)."""

    def __init__(self, prof: Profile, max_num_seqs: int = 256, max_num_batched_tokens: int = 2048):
        self.p, self.max_seqs, self.budget = prof, max_num_seqs, max_num_batched_tokens
        self.waiting: list = []
        self.running: list = []
        self.free_blocks = prof.num_blocks

    def add(self, seq: Seq) -> None:
        need = math.ceil((seq.prompt_len + seq.output_len) / self.p.block_size)
        if need > self.p.num_blocks:
            raise ValueError(f"request needs {need} blocks, the cache has {self.p.num_blocks}")
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> Plan:
        budget = self.budget
        decode = [s for s in self.running if s.prefilled == s.prompt_len][: budget]
        budget -= len(decode)
        prefill = []
        for s in self.running:
            if s.prefilled < s.prompt_len and budget > 0:
                n = min(budget, s.prompt_len - s.prefilled)
                prefill.append((s, n))
                budget -= n
        while self.waiting and budget > 0 and len(self.running) < self.max_seqs:
            s = self.waiting[0]
            need = math.ceil((s.prompt_len + s.output_len) / self.p.block_size)
            if need > self.free_blocks:
                break
            self.waiting.pop(0)
            s.blocks, self.free_blocks = need, self.free_blocks - need
            self.running.append(s)
            n = min(budget, s.prompt_len)
            prefill.append((s, n))
            budget -= n
        t = self.p.step_time([s.context for s in decode], sum(n for _, n in prefill),
                             sum(s.prefilled for s, _ in prefill))
        return Plan(decode, prefill, t)

    def commit(self, plan: Plan, now: float) -> list:
        """Apply a step that ended at ``now``; returns ``[(seq, emitted_token: bool, finished: bool)]``."""
        events = []
        for s in plan.decode:
            s.generated += 1
            s.token_times.append(now)
            events.append(s)
        for s, n in plan.prefill:
            s.prefilled += n
            if s.prefilled == s.prompt_len:            # the last prefill chunk samples the first token
                s.generated, s.first_token = 1, now
                s.token_times.append(now)
                events.append(s)
        out = []
        for s in events:
            done = s.generated >= s.output_len
            if done:
                s.finished = now
                self.running.remove(s)
                self.free_blocks += s.blocks
            out.append((s, True, done))
        return out


def simulate(prof: Profile, requests: list, *, max_num_seqs: int = 256, max_num_batched_tokens: int = 2048) -> list:
    """Run ``requests`` = ``[(arrival_s, prompt_len, output_len)]`` on a virtual clock -> finished Seqs."""
    eng = Engine(prof, max_num_seqs, max_num_batched_tokens)
    pending = sorted((a, i, p, o) for i, (a, p, o) in enumerate(requests))
    now, done, k = 0.0, [], 0
    while k < len(pending) or eng.has_work():
        while k < len(pending) and pending[k][0] <= now:
            a, i, p, o = pending[k]
            eng.add(Seq(i, a, p, o))
            k += 1
        if not eng.has_work():
            now = pending[k][0]
            continue
        plan = eng.schedule()
        if not plan.decode and not plan.prefill:        # blocked on memory: wait for the next arrival
            now = pending[k][0] if k < len(pending) else now + prof.overhead_s
            continue
        now += plan.time
        done += [s for s, _, fin in eng.commit(plan, now) if fin]
    return sorted(done, key=lambda s: s.rid)


def closed_loop(n_users: int, n_requests: int, prompt_len: int, output_len: int, prof: Profile, **kw) -> list:
    """``n_users`` each send one request, then the next when it finishes (simulated)."""
    eng = Engine(prof, **kw)
    now, sent, done = 0.0, 0, []
    for _ in range(min(n_users, n_requests)):
        eng.add(Seq(sent, now, prompt_len, output_len))
        sent += 1
    while eng.has_work():
        plan = eng.schedule()
        now += plan.time
        for s, _, fin in eng.commit(plan, now):
            if fin:
                done.append(s)
                if sent < n_requests:
                    eng.add(Seq(sent, now, prompt_len, output_len))
                    sent += 1
    return done


# ---------------------------------------------------------------------------------------------
# Results, from the emulator or from the wire
# ---------------------------------------------------------------------------------------------
@dataclass
class Result:
    ttft: float
    itl: list
    e2e: float
    output_tokens: int
    start: float = 0.0
    end: float = 0.0

    @property
    def tpot(self) -> float:
        return (self.e2e - self.ttft) / (self.output_tokens - 1) if self.output_tokens > 1 else 0.0


def from_seqs(seqs) -> list:
    return [Result(s.first_token - s.arrival, list(np.diff(s.token_times)), s.finished - s.arrival, s.generated,
                   s.arrival, s.finished) for s in seqs]


def summarize(results: list, label: str, duration: float | None = None) -> dict:
    """Means and percentiles as ``vllm bench serve`` reports them (TTFT, TPOT, ITL, throughput)."""
    ok = [r for r in results if r.output_tokens > 0]
    dur = duration or (max(r.end for r in ok) - min(r.start for r in ok))
    itl = [x for r in ok for x in r.itl]
    pct = lambda xs, q: float(np.percentile(xs, q)) * 1e3 if xs else float("nan")  # noqa: E731
    return {"source": label, "requests": len(ok), "duration_s": dur,
            "ttft_ms_mean": statistics.fmean(r.ttft for r in ok) * 1e3, "ttft_ms_p99": pct([r.ttft for r in ok], 99),
            "tpot_ms_mean": statistics.fmean(r.tpot for r in ok) * 1e3,
            "itl_ms_mean": statistics.fmean(itl) * 1e3 if itl else float("nan"), "itl_ms_p99": pct(itl, 99),
            "output_tok_s": sum(r.output_tokens for r in ok) / dur, "req_s": len(ok) / dur}


def compare(schemes, model="llama-3.1-8b-instruct", gpu_name="L4", *, users: int = 8, n_requests: int = 32,
            prompt_len: int = 1024, output_len: int = 128, kv_cache_dtype: str = "auto", **pkw) -> list:
    """Closed-loop simulation of each scheme on the same workload -> summaries (SIMULATED)."""
    out = []
    for sch in schemes:
        prof = profile(model, gpu_name, sch, kv_cache_dtype=kv_cache_dtype, **pkw)
        res = from_seqs(closed_loop(users, n_requests, prompt_len, output_len, prof))
        out.append({"scheme": sch, **summarize(res, f"SIMULATED {prof.name}")})
    return out


# ---------------------------------------------------------------------------------------------
# Measuring a server over HTTP (the fake one at T0, vllm serve at T1)
# ---------------------------------------------------------------------------------------------
def synthetic_prompt(n_tokens: int, seed: int = 0) -> str:
    """~``n_tokens`` tokens of distinct words (the fake server counts words; real tokenizers vary: verify)."""
    rng = np.random.default_rng(seed)
    words = ["alpha", "beta", "gamma", "delta", "kappa", "sigma", "omega", "theta", "lambda", "zeta"]
    return " ".join(words[i] for i in rng.integers(0, len(words), n_tokens))


def stream_completion(url: str, model: str, prompt: str, max_tokens: int, headers: dict | None = None,
                      timeout: float = 600) -> Result:
    """One streaming ``/v1/completions`` request; chunk arrival times give TTFT and ITL."""
    u = urllib.parse.urlparse(url)
    conn = (http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection)(
        u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout)
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "stream": True,
                       "ignore_eos": True, "temperature": 0.0, "stream_options": {"include_usage": True}})
    t0 = time.perf_counter()
    conn.request("POST", (u.path.rstrip("/") or "") + "/v1/completions", body,
                 {"Content-Type": "application/json", **(headers or {})})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:300]!r}")
    times, n_out, buf = [], 0, b""
    while True:
        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(1)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            line = ev.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            obj = json.loads(line[5:])
            if obj.get("usage"):
                n_out = obj["usage"].get("completion_tokens", n_out)
            if obj.get("choices") and obj["choices"][0].get("text"):
                times.append(time.perf_counter())
    conn.close()
    t_end = times[-1] if times else time.perf_counter()
    return Result(times[0] - t0 if times else float("nan"), list(np.diff(times)), t_end - t0, n_out or len(times),
                  t0, t_end)


def is_simulated(url: str) -> bool:
    """The fake server says so on ``/version``; a real vLLM answers ``{"version": ...}`` only."""
    try:
        u = urllib.parse.urlparse(url)
        c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=5)
        c.request("GET", "/version")
        return bool(json.loads(c.getresponse().read() or b"{}").get("simulated"))
    except Exception:  # noqa: BLE001
        return False


def run_http(url: str, model: str, *, users: int = 4, n_requests: int = 16, prompt_len: int = 256,
             output_len: int = 32, headers: dict | None = None) -> dict:
    """Closed loop over HTTP: ``users`` threads, each sending requests back to back."""
    results, lock, counter = [], threading.Lock(), iter(range(n_requests))

    def user():
        while True:
            with lock:
                i = next(counter, None)
            if i is None:
                return
            r = stream_completion(url, model, synthetic_prompt(prompt_len, seed=i), output_len, headers)
            with lock:
                results.append(r)
    ths = [threading.Thread(target=user) for _ in range(users)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    label = f"SIMULATED (fake server at {url})" if is_simulated(url) else f"MEASURED ({url})"
    return summarize(results, label, time.perf_counter() - t0)
