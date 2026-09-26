"""engine.py — a timing emulator of a vLLM-style engine, for long outputs (no model, no GPU).

One idea: a thinking request is a *long decode*, and an engine's decode step costs

    step_time = overhead + max( 2 × params × tokens_in_step / FLOP/s ,
                                (weight_bytes + Σ_running context_tokens × kv_bytes_per_token) / bandwidth )

— every step streams the weights once *plus the KV cache of every running request*. With short
answers the weights dominate; with thousands of thinking tokens per request the KV term grows
until it does, the KV pool fills, and the scheduler starts preempting. This emulator runs that
loop: FCFS admission under ``max_num_seqs`` and a per-step token budget, one token per running
request per step, KV blocks allocated as sequences grow, and preemption *by recompute* (the
newest running request is evicted, its blocks freed, and it prefills again later) when the pool is
empty — the policy vLLM V1 uses. :mod:`thinklab.fakeserver` runs it in real time behind an
OpenAI-compatible API; :func:`simulate` runs it in virtual time (instantly) for what-if analysis.

Everything it produces is **simulated**. Simplifications: attention FLOPs ignored, prefill of a
prompt is one chunk (a prefill larger than the per-step token budget, e.g. a long request recomputed
after preemption, runs alone as one long step where vLLM would chunk it over several), no CUDA-graph
padding, no prefix cache inside the block pool (the fake server keeps a separate prefix index), all
blocks usable. The profiles' block counts are the 04 lab's
sizing predictions (``servelab.sizing.size``; reproduced in ``tests/test_reuse.py``); the
efficiencies (mfu 0.5, 80% of peak bandwidth, 4 ms overhead) are that lab's defaults — assumptions.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    """Hardware + model numbers that set step time and KV capacity."""
    name: str
    model: str
    gpu: str
    weight_bytes: float
    params: float
    kv_bytes_per_token: int
    num_blocks: int
    block_size: int = 16
    mem_bw: float = 320e9 * 0.8        # effective bytes/s
    flops: float = 65e12 * 0.5         # effective FLOP/s
    overhead_s: float = 0.004
    max_model_len: int = 8192

    @property
    def kv_capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def decode_step_s(self, batch: int, context_tokens: float) -> float:
        """One decode step for ``batch`` sequences holding ``context_tokens`` KV tokens *in total*."""
        compute = 2 * self.params * batch / self.flops
        memory = (self.weight_bytes + context_tokens * self.kv_bytes_per_token) / self.mem_bw
        return self.overhead_s + max(compute, memory)

    def step_s(self, prefill_tokens: int, decode_seqs: int, context_tokens: float) -> float:
        compute = 2 * self.params * (prefill_tokens + decode_seqs) / self.flops
        memory = (self.weight_bytes + context_tokens * self.kv_bytes_per_token) / self.mem_bw
        return self.overhead_s + max(compute, memory)

    def describe(self) -> str:
        return (f"[simulated profile {self.name}] {self.model} on {self.gpu}: weights {self.weight_bytes / 1e9:.2f} GB, "
                f"KV {self.kv_bytes_per_token:,} B/token, {self.num_blocks:,} blocks x {self.block_size} = "
                f"{self.kv_capacity_tokens:,} KV tokens, effective {self.mem_bw / 1e9:.0f} GB/s and "
                f"{self.flops / 1e12:.1f} TFLOP/s, {self.overhead_s * 1e3:.0f} ms/step overhead")


def _gpu(name):   # (peak bandwidth GB/s, dense 16-bit TFLOPS) — datasheet numbers as in servelab.sizing.GPUS (verify)
    return {"T4": (320, 65), "L4": (300, 121), "H100": (3350, 989)}[name]


def make_profile(name, model, gpu, params, kv_bytes_per_token, num_blocks, bytes_per_param=2, max_model_len=8192,
                 mfu=0.5, bw_efficiency=0.8, overhead_ms=4.0, block_size=16) -> Profile:
    bw, tf = _gpu(gpu)
    return Profile(name, model, gpu, params * bytes_per_param, params, kv_bytes_per_token, num_blocks, block_size,
                   bw * 1e9 * bw_efficiency, tf * 1e12 * mfu, overhead_ms / 1e3, max_model_len)


# Parameter counts and KV bytes/token: servelab.sizing.param_count / kv_bytes_per_token on the Qwen3
# configs (28 layers x 8 KV heads x head_dim 128 x 2 (K,V) x 2 bytes = 114,688 B for 0.6B and 1.7B;
# 36 layers for 4B = 147,456 B). Block counts: servelab.sizing.size(model, gpu, dtype, max_model_len)
# with vLLM v0.30.0's default gpu_memory_utilization 0.92 — predictions, calibrate on a real startup log.
PROFILES = {
    "t4-qwen3-0.6b": dict(model="Qwen/Qwen3-0.6B", gpu="T4", params=596_049_920, kv_bytes_per_token=114_688,
                          num_blocks=6_969, max_model_len=8192),
    "t4-qwen3-1.7b": dict(model="Qwen/Qwen3-1.7B", gpu="T4", params=1_720_574_976, kv_bytes_per_token=114_688,
                          num_blocks=5_721, max_model_len=8192),
    "l4-qwen3-4b": dict(model="Qwen/Qwen3-4B", gpu="L4", params=4_022_468_096, kv_bytes_per_token=147_456,
                        num_blocks=5_504, max_model_len=16384),
}


def profile(name: str = "t4-qwen3-0.6b", **overrides) -> Profile:
    if name == "tiny":
        return Profile("tiny", "tiny", "none", 1e8, 5e7, 1000, overrides.pop("num_blocks", 64), 4, 1e11, 5e12,
                       0.0005, overrides.pop("max_model_len", 4096))
    return make_profile(name, **{**PROFILES[name], **overrides})


@dataclass
class EngineConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048


@dataclass(eq=False)
class Req:
    rid: int
    prompt_tokens: int
    output_tokens: int                    # what the model will emit (reasoning + answer), capped by max_tokens
    arrival: float
    reasoning_tokens: int = 0             # first `reasoning_tokens` of the output are thinking
    cached_tokens: int = 0                # prompt tokens served by a prefix cache (prefill skipped)
    generated: int = 0
    computed: int = 0                     # tokens whose KV is resident (0 after a preemption)
    blocks: int = 0
    state: str = "waiting"
    admitted_at: float | None = None
    first_token: float | None = None
    first_content: float | None = None
    last_token: float | None = None
    finished_at: float | None = None
    preemptions: int = 0
    itls: list = field(default_factory=list)
    tag: str = ""

    @property
    def context(self) -> int:
        return self.prompt_tokens + self.generated


class Engine:
    """FCFS continuous batching over a fixed KV block pool, with recompute preemption."""

    def __init__(self, prof: Profile, cfg: EngineConfig | None = None, keep_itl: bool = True):
        self.p, self.c, self.keep_itl = prof, cfg or EngineConfig(), keep_itl
        self.free = prof.num_blocks
        self.waiting: deque = deque()
        self.running: list = []
        self.finished: list = []
        self.preemptions = 0
        self.steps = 0
        self.busy_s = 0.0
        self.kv_samples: list = []        # (time, kv usage 0-1, running, waiting) after each step
        self._next = 0

    # -- requests ------------------------------------------------------------------------------
    def add(self, prompt_tokens: int, output_tokens: int, arrival: float, reasoning_tokens: int = 0,
            cached_tokens: int = 0, tag: str = "") -> Req:
        if prompt_tokens + output_tokens > self.p.max_model_len:
            raise ValueError(f"prompt ({prompt_tokens}) + max output ({output_tokens}) exceeds max_model_len "
                             f"{self.p.max_model_len}: vLLM rejects this request with HTTP 400")
        r = Req(self._next, prompt_tokens, output_tokens, arrival, reasoning_tokens, cached_tokens, tag=tag)
        self._next += 1
        self.waiting.append(r)
        return r

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _blocks_for(self, tokens: int) -> int:
        return math.ceil(tokens / self.p.block_size)

    def usage(self) -> float:
        return 1 - self.free / self.p.num_blocks

    # -- one step ------------------------------------------------------------------------------
    def schedule(self, now: float) -> dict:
        """Decide this step: decode every running request (preempting the newest when a block is
        missing), then admit waiting requests FCFS while sequences, tokens and blocks allow."""
        decode = []
        for r in list(self.running):
            if r not in self.running:
                continue
            need = self._blocks_for(r.context + 1) - r.blocks
            while need > self.free and self.running:
                victim = self.running[-1]                     # newest first, as vLLM
                self._preempt(victim)
                if victim is r:
                    break
            if r in self.running:
                self.free -= need
                r.blocks += need
                decode.append(r)
        budget = self.c.max_num_batched_tokens - len(decode)
        prefill = []
        while self.waiting and len(self.running) < self.c.max_num_seqs:
            r = self.waiting[0]
            todo = r.context - (r.cached_tokens if r.preemptions == 0 else 0)   # a preempted request recomputes everything
            need = self._blocks_for(r.context + 1)
            if (todo > budget and prefill) or need > self.free:
                break                                  # an oversized prefill is admitted alone (see the docstring)
            self.waiting.popleft()
            self.free -= need
            r.blocks, r.state = need, "running"
            if r.admitted_at is None:
                r.admitted_at = now
            budget -= todo
            prefill.append((r, todo))
            self.running.append(r)
        return {"decode": decode, "prefill": prefill}

    def abort(self, r: Req, now: float) -> None:
        """The client went away: drop the request and free its blocks (vLLM aborts it the same way)."""
        if r in self.running:
            self.running.remove(r)
            self.free += r.blocks
            r.blocks = 0
        elif r in self.waiting:
            self.waiting.remove(r)
        r.state, r.finished_at = "aborted", now

    def _preempt(self, r: Req) -> None:
        self.running.remove(r)
        self.free += r.blocks
        r.blocks, r.computed, r.state = 0, 0, "waiting"
        r.preemptions += 1
        self.preemptions += 1
        self.waiting.appendleft(r)

    def step_time(self, plan: dict) -> float:
        prefill_tokens = sum(t for _, t in plan["prefill"])
        ctx = sum(r.context for r in plan["decode"]) + sum(r.context for r, _ in plan["prefill"])
        return self.p.step_s(prefill_tokens, len(plan["decode"]), ctx)

    def commit(self, plan: dict, now: float) -> list:
        """Apply a step that ended at ``now``: every scheduled request emits one token.
        Returns ``[(req, token_index, finished)]``."""
        events = []
        self.steps += 1
        for r in plan["decode"] + [r for r, _ in plan["prefill"]]:
            if r.state != "running":
                continue
            idx = r.generated
            r.generated += 1
            r.computed = r.context
            if r.first_token is None:
                r.first_token = now
            elif self.keep_itl and r.last_token is not None:
                r.itls.append(now - r.last_token)
            if r.first_content is None and r.generated > r.reasoning_tokens:
                r.first_content = now
            r.last_token = now
            done = r.generated >= r.output_tokens
            if done:
                r.state, r.finished_at = "finished", now
                self.running.remove(r)
                self.free += r.blocks
                r.blocks = 0
                self.finished.append(r)
            events.append((r, idx, done))
        self.kv_samples.append((now, self.usage(), len(self.running), len(self.waiting)))
        return events


def simulate(prof: Profile, requests: list, cfg: EngineConfig | None = None, keep_itl: bool = True,
             max_steps: int = 5_000_000) -> Engine:
    """Run ``requests`` = [(arrival_s, prompt_tokens, output_tokens, reasoning_tokens[, tag])] in virtual
    time and return the engine (``.finished`` holds the requests with their timings, in seconds)."""
    eng = Engine(prof, cfg, keep_itl)
    pending = deque(sorted(requests, key=lambda r: r[0]))
    now = 0.0
    while pending or eng.has_work():
        while pending and pending[0][0] <= now:
            a, p, o, rtok, *tag = pending.popleft()
            eng.add(p, o, a, rtok, tag=tag[0] if tag else "")
        plan = eng.schedule(now)
        if not plan["decode"] and not plan["prefill"]:
            if pending:
                now = max(now, pending[0][0])
                continue
            raise RuntimeError("engine stalled: a request cannot fit the KV pool even alone")
        dt = eng.step_time(plan)
        now += dt
        eng.busy_s += dt
        eng.commit(plan, now)
        if eng.steps > max_steps:
            raise RuntimeError("simulation did not finish")
    return eng
