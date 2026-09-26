"""fake_engine.py — a timing emulator of a vLLM-style engine: scheduler, KV blocks, prefix cache
and a roofline step-time model. No model and no GPU: tokens are made up, times are computed.

One idea: an engine is a loop of *steps*. Each step the scheduler picks which tokens to compute
under two budgets (``max_num_seqs`` sequences, ``max_num_batched_tokens`` tokens), the KV manager
finds blocks for them (reusing cached prefix blocks found by a chained block hash), and the step
lasts as long as the roofline says:

    step_time = overhead + max( 2 × active_params × tokens / FLOP/s,
                                (weight_bytes + KV bytes read) / memory bandwidth ) + draft time

Everything a serving benchmark observes falls out of those pieces: TTFT that grows with
queueing, ITL that grows with batch size and whenever a prefill chunk shares the step, prefix
hits that shorten prefill, preemptions when blocks run out, speculation that pays at low batch.
:mod:`servelab.fakeserver` runs this loop in real time behind an OpenAI-compatible API;
:func:`simulate` runs it in virtual time for tests.

Simplifications (outputs are always labelled "simulated"): FCFS only, attention FLOPs ignored,
no CUDA-graph padding, preemption always by recompute, every block usable (vLLM keeps one back as
its null block), speculative acceptance as independent coin flips with probability
``spec_acceptance``. The real scheduler is
``vllm/v1/core/sched/scheduler.py``; a from-scratch engine with a real model is this topic's
``mini-engine-core``.
"""
from __future__ import annotations

import itertools
import math
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from . import sizing
from .textgen import generated_piece


# ---------------------------------------------------------------------------------------------
# Hardware + model timing profile
# ---------------------------------------------------------------------------------------------
@dataclass
class EngineProfile:
    """Numbers that set step time and KV capacity. Built from sizing + a GPU datasheet, so every
    field is traceable; the efficiencies (mfu, bandwidth efficiency, overhead) are assumptions."""
    name: str
    model: str
    weight_bytes: float
    active_params: float
    kv_bytes_per_token: int
    num_blocks: int
    block_size: int
    mem_bw: float            # effective bytes/s
    flops: float             # effective FLOP/s
    overhead_s: float        # fixed per step: scheduling, kernel launches, sampling
    max_model_len: int

    def decode_step_s(self, batch: int, context: int) -> float:
        """One decode step for ``batch`` sequences of ``context`` tokens each."""
        compute = 2 * self.active_params * batch / self.flops
        memory = (self.weight_bytes + batch * context * self.kv_bytes_per_token) / self.mem_bw
        return self.overhead_s + max(compute, memory)

    def prefill_s(self, tokens: int, context: int = 0) -> float:
        """One step that prefills ``tokens`` prompt tokens (and nothing else)."""
        compute = 2 * self.active_params * tokens / self.flops
        memory = (self.weight_bytes + (context + tokens) * self.kv_bytes_per_token) / self.mem_bw
        return self.overhead_s + max(compute, memory)

    def describe(self) -> str:
        return (f"[simulated profile {self.name}] {self.model}: weights {self.weight_bytes / 1e9:.2f} GB, "
                f"{self.active_params / 1e9:.2f} B active params, KV {self.kv_bytes_per_token:,} B/token, "
                f"{self.num_blocks:,} blocks x {self.block_size} tokens, effective {self.mem_bw / 1e9:.0f} GB/s "
                f"and {self.flops / 1e12:.1f} TFLOP/s, {self.overhead_s * 1e3:.1f} ms/step overhead; "
                f"decode floor {self.decode_step_s(1, 512) * 1e3:.1f} ms/token at batch 1")


def build_profile(model: str = "qwen2.5-0.5b-instruct", gpu: str = "T4", *, dtype: str = "auto",
                  quantization: str | None = None, kv_cache_dtype: str = "auto",
                  gpu_memory_utilization: float = 0.92, max_model_len: int = 4096, block_size: int = 16,
                  mfu: float = 0.5, bw_efficiency: float = 0.8, overhead_ms: float = 4.0,
                  int4_compute_efficiency: float = 1.0,
                  num_blocks: int | None = None, name: str | None = None) -> EngineProfile:
    """A profile from a model config and a GPU datasheet, via :func:`servelab.sizing.size`.

    FP8 (W8A8) doubles tensor-core FLOP/s on GPUs that have FP8 units; int4 weight-only formats
    (AWQ/GPTQ) shrink bytes but still compute in 16-bit. What dequantization costs a prefill is
    kernel-dependent (Marlin-style kernels hide most of it at large batch; verify on yours), so
    it is an explicit assumption: ``int4_compute_efficiency`` (1.0 = no penalty, the default)."""
    m = sizing.load_config(model)
    g = sizing.gpu(gpu)
    rep = sizing.size(m, g, dtype=dtype, quantization=quantization, kv_cache_dtype=kv_cache_dtype,
                      gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                      block_size=block_size)
    tflops = g.bf16_tflops
    if quantization in ("fp8", "w8a8") and g.fp8_tflops:
        tflops = g.fp8_tflops
    elif quantization in ("awq", "gptq", "int4"):
        tflops *= int4_compute_efficiency
    return EngineProfile(
        name=name or f"{g.name.lower()}-{m.name.split('/')[-1].lower()}", model=m.name,
        weight_bytes=rep.weights_bytes, active_params=float(sizing.param_count(m).active),
        kv_bytes_per_token=rep.kv_bytes_per_token, num_blocks=num_blocks or rep.num_blocks,
        block_size=block_size, mem_bw=g.mem_bw_gbs * 1e9 * bw_efficiency, flops=tflops * 1e12 * mfu,
        overhead_s=overhead_ms / 1e3, max_model_len=max_model_len)


PROFILES = {
    # Colab/Kaggle T4, the smallest useful chat model, fp16 (T4 has no bf16).
    "t4-qwen2.5-0.5b": dict(model="qwen2.5-0.5b-instruct", gpu="T4", dtype="half", max_model_len=4096),
    # One L4 (Cloud Run GPU / GKE g2): a 1.5B model in bf16.
    "l4-qwen2.5-1.5b": dict(model="qwen2.5-1.5b-instruct", gpu="L4", max_model_len=8192),
    # One L4 with an 8B model in FP8 weights and FP8 KV.
    "l4-llama3.1-8b-fp8": dict(model="llama-3.1-8b-instruct", gpu="L4", quantization="fp8",
                               kv_cache_dtype="fp8", max_model_len=16384, overhead_ms=5.0),
    # One H100 with the same model in bf16.
    "h100-llama3.1-8b": dict(model="llama-3.1-8b-instruct", gpu="H100-80GB", max_model_len=32768,
                             overhead_ms=3.0),
}


def profile(name: str = "t4-qwen2.5-0.5b", **overrides) -> EngineProfile:
    """A named profile from :data:`PROFILES` (or ``"tiny"``), with keyword overrides."""
    if name == "tiny":
        return tiny_profile(**overrides)
    return build_profile(**{**PROFILES[name], **overrides}, name=name)


def tiny_profile(num_blocks: int = 64, block_size: int = 4, overhead_ms: float = 0.5,
                 max_model_len: int = 512) -> EngineProfile:
    """A toy engine for tests: a small KV cache (64 × 4 = 256 token slots) and ~1.5 ms steps."""
    return EngineProfile(name="tiny", model="tiny", weight_bytes=1e8, active_params=5e7, kv_bytes_per_token=1000,
                         num_blocks=num_blocks, block_size=block_size, mem_bw=1e11, flops=5e12,
                         overhead_s=overhead_ms / 1e3, max_model_len=max_model_len)


# ---------------------------------------------------------------------------------------------
# Engine knobs (vLLM flag names)
# ---------------------------------------------------------------------------------------------
@dataclass
class EngineConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    long_prefill_token_threshold: int = 0     # 0 = no per-request cap on a prefill chunk
    num_speculative_tokens: int = 0           # k draft tokens verified per decode step
    spec_acceptance: float = 0.0              # P(a draft token is accepted): workload-dependent
    spec_draft_cost: float = 0.0              # draft time per token / target weight-read time
    seed: int = 0


# ---------------------------------------------------------------------------------------------
# Requests and the block pool
# ---------------------------------------------------------------------------------------------
@dataclass(eq=False)
class Seq:
    rid: str
    prompt: list
    max_tokens: int
    arrival: float
    output: list = field(default_factory=list)
    texts: list = field(default_factory=list)
    num_computed: int = 0            # tokens whose KV is in the cache (hits + computed)
    blocks: list = field(default_factory=list)
    hashes: list = field(default_factory=list)
    num_registered: int = 0          # full blocks already offered to the prefix cache
    status: str = "waiting"
    first_scheduled: float | None = None
    first_token: float | None = None
    last_token: float | None = None
    finished_at: float | None = None
    finish_reason: str | None = None
    cached_tokens: int = 0           # prefix-cache hit at first admission
    preemptions: int = 0
    emit_times: list = field(default_factory=list)   # one entry per step that emitted tokens

    @property
    def tokens(self) -> list:
        return self.prompt + self.output

    @property
    def num_tokens(self) -> int:
        return len(self.prompt) + len(self.output)


def block_hash(parent: int | None, tokens: tuple) -> int:
    """A block's identity is its tokens *and everything before them* (the parent's hash), so equal
    hashes mean equal prefixes. vLLM hashes (parent, token ids, extra keys such as LoRA id or a
    cache salt) with sha256 by default; Python's tuple hash is enough for a simulator."""
    return hash((parent, tokens))


class BlockPool:
    """Fixed pool of KV blocks with reference counts and a hash -> block prefix cache.

    Free blocks sit in an LRU queue; a free block may still hold a cached prefix and is only
    forgotten when it is reallocated (evicted) — so "free" and "cached" overlap, as in vLLM."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.ref = [0] * num_blocks
        self.hash_of: list = [None] * num_blocks
        self.cached: dict = {}
        self.free: OrderedDict = OrderedDict((i, None) for i in range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self.free)

    def usage(self) -> float:
        """Fraction of blocks referenced by live requests (vllm:kv_cache_usage_perc)."""
        return 1.0 - self.num_free / self.num_blocks if self.num_blocks else 0.0

    def lookup(self, hashes: list) -> list:
        """Block ids for the longest prefix of ``hashes`` that is cached (no side effects)."""
        out = []
        for h in hashes:
            b = self.cached.get(h)
            if b is None:
                break
            out.append(b)
        return out

    def touch(self, ids: list) -> None:
        for b in ids:
            if self.ref[b] == 0:
                del self.free[b]
            self.ref[b] += 1

    def allocate(self, n: int) -> list:
        if n > self.num_free:
            raise RuntimeError("out of KV blocks")
        out = []
        for _ in range(n):
            b, _ = self.free.popitem(last=False)      # least recently freed first
            h = self.hash_of[b]
            if h is not None:                          # evict the prefix it was caching
                if self.cached.get(h) == b:
                    del self.cached[h]
                self.hash_of[b] = None
            self.ref[b] = 1
            out.append(b)
        return out

    def release(self, ids: list) -> None:
        """Drop one reference; blocks that reach zero go to the back of the LRU queue. Callers pass
        a request's blocks tail-first, so its prefix blocks are the last to be evicted."""
        for b in ids:
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free[b] = None

    def register(self, block: int, h: int) -> None:
        if h not in self.cached:
            self.cached[h] = block
            self.hash_of[block] = h


# ---------------------------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------------------------
@dataclass
class Scheduled:
    seq: Seq
    num_new: int          # tokens computed this step
    is_prefill: bool
    num_spec: int = 0     # draft tokens verified this step


@dataclass
class Plan:
    items: list
    preempted: list
    prefix_queries: int = 0          # first admissions only (vllm:prefix_cache_queries)
    prefix_hits: int = 0
    preempted_prefix_queries: int = 0   # re-admissions after preemption (kept apart, as vLLM does)
    preempted_prefix_hits: int = 0

    @property
    def empty(self) -> bool:
        return not self.items

    @property
    def num_tokens(self) -> int:
        return sum(i.num_new for i in self.items)

    @property
    def prefill_tokens(self) -> int:
        return sum(i.num_new for i in self.items if i.is_prefill)


@dataclass
class Event:
    seq: Seq
    token_ids: list
    texts: list
    first: bool
    finished: bool
    itl: float | None     # time since this request's previous emission (None for the first)


@dataclass
class StepResult:
    events: list
    generated: int
    prompt_tokens_done: int       # full prompt lengths of requests that got their first token
    spec_drafts: int = 0
    spec_draft_tokens: int = 0
    spec_accepted: int = 0
    spec_accepted_per_pos: list = field(default_factory=list)


class FakeEngine:
    def __init__(self, profile: EngineProfile, config: EngineConfig | None = None):
        self.p, self.c = profile, config or EngineConfig()
        if not self.c.enable_chunked_prefill and self.c.max_num_batched_tokens < profile.max_model_len:
            raise ValueError(f"max_num_batched_tokens ({self.c.max_num_batched_tokens}) is smaller than "
                             f"max_model_len ({profile.max_model_len}): without chunked prefill a whole "
                             "prompt must fit in one step's token budget")
        self.pool = BlockPool(profile.num_blocks)
        self.waiting: deque = deque()
        self.running: list = []
        self.rng = random.Random(self.c.seed)
        self._ids = itertools.count()
        self.total_preemptions = 0

    # -- requests ------------------------------------------------------------------------------
    def add_request(self, prompt: list, max_tokens: int, arrival: float, rid: str | None = None) -> Seq:
        if not prompt:
            raise ValueError("empty prompt")
        if len(prompt) + max_tokens > self.p.max_model_len:
            raise ValueError(f"This model's maximum context length is {self.p.max_model_len} tokens. However, "
                             f"you requested {len(prompt) + max_tokens} tokens ({len(prompt)} in the messages, "
                             f"{max_tokens} in the completion).")
        if math.ceil((len(prompt) + max_tokens) / self.p.block_size) > self.p.num_blocks:
            raise ValueError("request can never fit in the KV cache")
        seq = Seq(rid or f"req-{next(self._ids)}", list(prompt), max(1, int(max_tokens)), arrival)
        self.waiting.append(seq)
        return seq

    def abort(self, seq: Seq, now: float) -> None:
        if seq.status == "finished":
            return
        if seq.status == "running":
            self.running.remove(seq)
            self.pool.release(list(reversed(seq.blocks)))
        elif seq.status == "waiting" and seq in self.waiting:
            self.waiting.remove(seq)
        seq.blocks, seq.status, seq.finish_reason, seq.finished_at = [], "finished", "abort", now

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    # -- hashing -------------------------------------------------------------------------------
    def _update_hashes(self, seq: Seq, upto_blocks: int) -> None:
        bs, toks = self.p.block_size, seq.tokens
        while len(seq.hashes) < upto_blocks:
            i = len(seq.hashes)
            seq.hashes.append(block_hash(seq.hashes[-1] if seq.hashes else None, tuple(toks[i * bs:(i + 1) * bs])))

    def _preempt(self, seq: Seq) -> None:
        """Recompute-style preemption: drop the KV, requeue at the front, keep the output."""
        self.pool.release(list(reversed(seq.blocks)))
        seq.blocks, seq.num_computed, seq.num_registered, seq.status = [], 0, 0, "waiting"
        seq.preemptions += 1
        self.total_preemptions += 1
        self.waiting.appendleft(seq)

    # -- one step --------------------------------------------------------------------------------
    def schedule(self, now: float) -> Plan:
        """Pick this step's work: running requests first (decode, or the next prefill chunk),
        then waiting ones in arrival order while both budgets and the KV blocks allow."""
        c, bs = self.c, self.p.block_size
        budget = c.max_num_batched_tokens
        plan = Plan([], [])
        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            remaining = seq.num_tokens - seq.num_computed
            prefill = remaining > 1 or seq.first_token is None
            k = 0
            if prefill:
                n = min(remaining, budget)
                if c.long_prefill_token_threshold:
                    n = min(n, c.long_prefill_token_threshold)
            else:
                k = min(c.num_speculative_tokens, max(0, seq.max_tokens - len(seq.output) - 1), budget - 1)
                n = 1 + k
            need = math.ceil((seq.num_computed + n) / bs) - len(seq.blocks)
            while need > self.pool.num_free:
                victim = self.running.pop()           # the most recently admitted request loses
                self._preempt(victim)
                plan.preempted.append(victim)
                if victim is seq:
                    break
            if seq.status != "running":
                break
            if need > 0:
                seq.blocks += self.pool.allocate(need)
            plan.items.append(Scheduled(seq, n, prefill, k))
            budget -= n
            i += 1

        while not plan.preempted and self.waiting and budget > 0 and len(self.running) < c.max_num_seqs:
            seq = self.waiting[0]
            hits = []
            if c.enable_prefix_caching:
                max_blocks = (seq.num_tokens - 1) // bs          # always compute >= 1 token (logits)
                self._update_hashes(seq, max_blocks)
                hits = self.pool.lookup(seq.hashes[:max_blocks])
            hit_tokens = len(hits) * bs
            remaining = seq.num_tokens - hit_tokens
            if not c.enable_chunked_prefill and remaining > budget:
                break
            n = min(remaining, budget)
            if c.long_prefill_token_threshold:
                n = min(n, c.long_prefill_token_threshold)
            # admit only if the whole input fits (vLLM's scheduler_reserve_full_isl)
            need_total = math.ceil(seq.num_tokens / bs) - len(hits)
            if need_total > self.pool.num_free - sum(1 for b in hits if self.pool.ref[b] == 0):
                break
            self.waiting.popleft()
            self.pool.touch(hits)
            seq.blocks = list(hits)
            seq.blocks += self.pool.allocate(math.ceil((hit_tokens + n) / bs) - len(seq.blocks))
            seq.num_computed, seq.num_registered, seq.status = hit_tokens, len(hits), "running"
            if c.enable_prefix_caching:
                # vLLM counts a re-admitted (preempted) request separately, not in
                # vllm:prefix_cache_queries/hits: it would mostly re-hit its own blocks
                if seq.preemptions == 0:
                    plan.prefix_queries += seq.num_tokens
                    plan.prefix_hits += hit_tokens
                else:
                    plan.preempted_prefix_queries += seq.num_tokens
                    plan.preempted_prefix_hits += hit_tokens
            if seq.first_scheduled is None:
                seq.first_scheduled, seq.cached_tokens = now, hit_tokens
            self.running.append(seq)
            plan.items.append(Scheduled(seq, n, True, 0))
            budget -= n
        return plan

    def step_time(self, plan: Plan) -> float:
        """The roofline: a step is compute-bound (big prefill chunks) or memory-bound (decode)."""
        if plan.empty:
            return 0.0
        p = self.p
        tokens = plan.num_tokens
        context = sum(it.seq.num_computed + it.num_new for it in plan.items)
        compute = 2.0 * p.active_params * tokens / p.flops
        memory = (p.weight_bytes + context * p.kv_bytes_per_token) / p.mem_bw
        k = max((it.num_spec for it in plan.items), default=0)
        draft = k * self.c.spec_draft_cost * p.weight_bytes / p.mem_bw
        return p.overhead_s + max(compute, memory) + draft

    def _accepted(self, k: int) -> int:
        a = 0
        while a < k and self.rng.random() < self.c.spec_acceptance:
            a += 1
        return a

    def commit(self, plan: Plan, now: float) -> StepResult:
        """Apply a finished step: advance computed tokens, sample tokens, finish requests."""
        res = StepResult([], 0, 0, spec_accepted_per_pos=[0] * self.c.num_speculative_tokens)
        for it in plan.items:
            seq = it.seq
            if seq.status != "running":
                continue                                 # aborted while the step ran
            if it.is_prefill:
                seq.num_computed += it.num_new
                if seq.num_computed < seq.num_tokens:
                    self._register(seq)
                    continue                             # mid-prompt chunk: no token yet
                n_out = 1
            else:
                a = self._accepted(it.num_spec)
                if it.num_spec:
                    res.spec_drafts += 1
                    res.spec_draft_tokens += it.num_spec
                    res.spec_accepted += a
                    for pos in range(a):
                        res.spec_accepted_per_pos[pos] += 1
                seq.num_computed += 1 + a
                n_out = 1 + a
            ids, texts = [], []
            for _ in range(n_out):   # same prompt -> same continuation, like greedy decoding
                tid, text = generated_piece(hash((tuple(seq.prompt[-16:]), len(seq.prompt), len(seq.output))) & 0x7FFFFFFF)
                seq.output.append(tid)
                seq.texts.append(text)
                ids.append(tid)
                texts.append(text)
            first = seq.first_token is None
            itl = None if first else now - seq.last_token
            if first:
                seq.first_token = now
                res.prompt_tokens_done += len(seq.prompt)
            seq.last_token = now
            seq.emit_times.append(now)
            res.generated += n_out
            done = len(seq.output) >= seq.max_tokens or seq.num_tokens >= self.p.max_model_len
            self._register(seq)
            if done:
                seq.status, seq.finish_reason, seq.finished_at = "finished", "length", now
                self.running.remove(seq)
                self.pool.release(list(reversed(seq.blocks)))
                seq.blocks = []
            res.events.append(Event(seq, ids, texts, first, done, itl))
        return res

    def _register(self, seq: Seq) -> None:
        """Offer every newly *full and computed* block to the prefix cache."""
        if not self.c.enable_prefix_caching:
            return
        full = min(seq.num_computed, seq.num_tokens) // self.p.block_size
        self._update_hashes(seq, full)
        for i in range(seq.num_registered, min(full, len(seq.blocks))):
            self.pool.register(seq.blocks[i], seq.hashes[i])
        seq.num_registered = max(seq.num_registered, min(full, len(seq.blocks)))


def simulate(engine: FakeEngine, requests: list, max_steps: int = 1_000_000) -> list:
    """Run the engine in *virtual* time (no sleeping): ``requests`` are
    ``(arrival_s, prompt_token_ids, max_tokens)``. Returns the finished :class:`Seq` objects."""
    pending = sorted(requests, key=lambda r: r[0])
    seqs, t, j = [], 0.0, 0
    for _ in range(max_steps):
        while j < len(pending) and pending[j][0] <= t:
            seqs.append(engine.add_request(pending[j][1], pending[j][2], pending[j][0]))
            j += 1
        if not engine.has_work():
            if j >= len(pending):
                break
            t = pending[j][0]
            continue
        plan = engine.schedule(t)
        if plan.empty:                     # blocked until something arrives (cannot happen when idle)
            if j >= len(pending):
                raise RuntimeError("engine stalled")
            t = pending[j][0]
            continue
        t += engine.step_time(plan)
        engine.commit(plan, t)
    return seqs


def seq_timings(seq: Seq) -> dict:
    """Server-side view of one finished request, with vLLM's definitions."""
    n = len(seq.output)
    ttft = seq.first_token - seq.arrival
    decode = seq.last_token - seq.first_token
    return {"ttft": ttft, "e2e": seq.finished_at - seq.arrival, "queue": seq.first_scheduled - seq.arrival,
            "tpot": decode / (n - 1) if n > 1 else 0.0, "itl": [b - a for a, b in zip(seq.emit_times, seq.emit_times[1:])],
            "cached_tokens": seq.cached_tokens, "preemptions": seq.preemptions, "output_tokens": n}
