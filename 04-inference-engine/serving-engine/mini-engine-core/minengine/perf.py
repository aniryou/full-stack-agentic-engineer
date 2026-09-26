"""perf.py - a roofline step-time model driving the REAL scheduler: SIMULATED TTFT / ITL under load.

The one idea: one forward pass must (a) stream every weight byte, plus the KV of every scheduled
request, from HBM and (b) do about 2 x params FLOPs per token plus attention. Its time is whichever
is slower, plus a fixed per-step overhead:
    t_step = max(bytes / (HBM_BW x bw_eff), flops / (PEAK x flop_eff)) + overhead
Small batches are memory-bound: the weight read dominates, so extra decode tokens ride along almost
free. Past the knee, tokens ~ PEAK / BW x bytes_per_param / 2, every extra token costs compute.
`simulate` runs this package's actual Scheduler and KVCacheManager on a virtual clock with that step
time, under Poisson arrivals. Every number it produces is SIMULATED - assumptions in, estimates out.
Device and model figures are spec-sheet values as of Sep 2026 (verify); the efficiencies and the
overhead are assumptions to replace with measurements (vllm-serving-lab measures the real thing).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .kv import KVCacheManager
from .sampler import SamplingParams
from .scheduler import Request, Scheduler, SchedulerConfig
from .spec import expected_tokens


@dataclass(frozen=True)
class GPU:
    name: str
    peak_flops: float        # dense 16-bit tensor-core FLOP/s (no sparsity)
    hbm_bw: float            # bytes/s
    hbm_bytes: float
    fp8: bool = False        # FP8 tensor cores (Ada, Hopper, Blackwell)


GPUS = {   # datasheet values (verify): dense BF16/FP16 tensor FLOP/s, HBM/GDDR bandwidth, capacity
    "T4": GPU("T4", 65e12, 320e9, 16e9),
    "L4": GPU("L4", 121e12, 300e9, 24e9, fp8=True),
    "A100-80GB": GPU("A100-80GB", 312e12, 2.039e12, 80e9),
    "H100-SXM": GPU("H100-SXM", 989e12, 3.35e12, 80e9, fp8=True),
}


@dataclass(frozen=True)
class LLM:
    name: str
    params: float
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    embed_params: float = 0.0        # vocab x d_model: the embedding table (and LM head, if tied)
    tied: bool = True                # LM head shares the embedding table
    bytes_per_param: float = 2.0     # the matmul weights: bf16 = 2, fp8/int8 = 1, int4 (g128) ~ 0.52
    embed_bytes_per_param: float = 2.0   # embedding + LM head: GPTQ/AWQ/FP8 checkpoints keep them in 16-bit
    kv_bytes_per_value: float = 2.0  # bf16 = 2, fp8 = 1
    compute_scale: float = 1.0       # 2.0 when weights AND activations are FP8 (W8A8) on FP8 hardware
                                     # (applied to every FLOP - a simplification; attention is <5% of a prefill)

    @property
    def weight_bytes(self) -> float:
        """HBM the weights occupy: matmul weights at bytes_per_param, the embedding table and LM head
        (one table if tied, two if not) at embed_bytes_per_param."""
        return (self.matmul_params * self.bytes_per_param
                + self.embed_params * (1 if self.tied else 2) * self.embed_bytes_per_param)

    @property
    def streamed_bytes(self) -> float:
        """Bytes one step reads: every matmul weight and the LM head; an untied input embedding is a
        gather of a few rows, not a stream."""
        return self.matmul_params * self.bytes_per_param + self.embed_params * self.embed_bytes_per_param

    @property
    def matmul_params(self) -> float:
        """Parameters every token multiplies through (the LM head is charged per sampled row instead)."""
        return self.params - self.embed_params * (1 if self.tied else 2)

    @property
    def kv_bytes_per_token(self) -> float:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.kv_bytes_per_value


LLMS = {   # from each model's config.json (verify)
    "qwen2.5-0.5b": LLM("qwen2.5-0.5b", 0.494e9, 24, 14, 2, 64, 151936 * 896),
    "qwen3-0.6b": LLM("qwen3-0.6b", 0.596e9, 28, 16, 8, 128, 151936 * 1024),
    "qwen2.5-1.5b": LLM("qwen2.5-1.5b", 1.54e9, 28, 12, 2, 128, 151936 * 1536),
    "llama-3.2-1b": LLM("llama-3.2-1b", 1.24e9, 16, 32, 8, 64, 128256 * 2048),   # a draft for llama-3.1-8b
    "llama-3.1-8b": LLM("llama-3.1-8b", 8.03e9, 32, 32, 8, 128, 128256 * 4096, tied=False),
}


def step_cost(gpu: GPU, llm: LLM, chunks, flop_eff=0.6, bw_eff=0.8, overhead_s=0.002) -> dict:
    """chunks: [(start, n)] or [(start, n, logit_rows)] per scheduled request - n new tokens after
    `start` tokens already cached. FLOPs: 2 x matmul params per token, the LM head for `logit_rows`
    rows per request (default 1, its last token; a speculative verify pass needs k + 1), and
    4 x layers x heads x head_dim per (query, key) pair of attention. Bytes: the streamed weights once,
    each request's cached KV once, the new KV written. Returns FLOPs, bytes, both times and the bound.
    (A one-line version of layer 01's roofline.llm model, cheap enough to run every simulated step.)

    The same H100 is modelled differently one layer up: layer 06's
    06-gateway/scaling-admission-cost/agentic-scaling-lab-mistral/scalelab/serving.py
    (`Replica.step_seconds`) prices a decode step as bytes / (BW x 0.6) + 2 ms, with no compute
    term, where this uses max(bytes / (BW x 0.8), flops / (peak x 0.6)) + 2 ms. For a decode step
    that reads 15 GB (Llama-3.1-8B's streamed weights) that is ~9.5 ms there against 7.6 ms here,
    about 25% apart. Both efficiencies are assumptions, not measurements: calibrate either against
    a measured vLLM step (the serving lab's bench) before trusting absolute times."""
    if llm.compute_scale > 1 and not gpu.fp8:
        raise ValueError(f"{gpu.name} has no FP8 tensor cores: FP8 weights run weight-only (compute_scale=1)")
    chunks = [(c[0], c[1], c[2] if len(c) > 2 else 1) for c in chunks]
    tokens = sum(n for _, n, _ in chunks)
    pairs = sum(n * s + n * (n + 1) / 2 for s, n, _ in chunks)
    flops = (2 * llm.matmul_params * tokens + 2 * llm.embed_params * sum(r for _, _, r in chunks)
             + 4 * llm.n_layers * llm.n_heads * llm.head_dim * pairs)
    nbytes = llm.streamed_bytes + llm.kv_bytes_per_token * (sum(s for s, _, _ in chunks) + tokens)
    t_mem = nbytes / (gpu.hbm_bw * bw_eff)
    t_cmp = flops / (gpu.peak_flops * llm.compute_scale * flop_eff)
    return {"flops": flops, "bytes": nbytes, "t_memory": t_mem, "t_compute": t_cmp,
            "t": max(t_mem, t_cmp) + overhead_s, "bound": "memory" if t_mem >= t_cmp else "compute"}


def step_time(gpu: GPU, llm: LLM, chunks, **kw) -> float:
    return step_cost(gpu, llm, chunks, **kw)["t"]


def knee_tokens(gpu: GPU, llm: LLM, flop_eff=1.0, bw_eff=1.0) -> float:
    """Batch size (tokens) where compute time catches up with the weight read, treating every weight as
    a matmul weight (KV ignored): 2 P n / PEAK = P b / BW -> n = PEAK x b / (2 BW) - the ridge point in
    tokens. H100 bf16: 989e12 x 2 / (2 x 3.35e12) = 295 (the real flip for an 8B model is a little later)."""
    return gpu.peak_flops * llm.compute_scale * flop_eff * llm.bytes_per_param / (2 * gpu.hbm_bw * bw_eff)


def spec_speedup(gpu: GPU, target: LLM, draft: LLM, batch: int, ctx: int, alpha: float, k: int,
                 draft_overhead_s: float = 0.0005, **kw) -> float:
    """SIMULATED gain of draft-model speculation over plain decoding, `batch` requests at context `ctx`.
    Plain decoding: one target step per token. One round: k draft forwards - inside ONE engine step, so
    each pays only `draft_overhead_s` (a CUDA-graph replay and a draft sample; an assumption) instead of
    the step's overhead - plus one target verify pass of k + 1 tokens per request with logits at all
    k + 1 positions; it emits expected_tokens(alpha, k) per request. kw: step_cost's efficiencies and
    the engine-step overhead_s, paid once per round."""
    base = step_time(gpu, target, [(ctx, 1)] * batch, **kw)
    drafts = sum(step_time(gpu, draft, [(ctx + j, 1)] * batch, **{**kw, "overhead_s": draft_overhead_s})
                 for j in range(k))
    verify = step_time(gpu, target, [(ctx, k + 1, k + 1)] * batch, **kw)
    return expected_tokens(alpha, k) * base / (drafts + verify)


def tp_allreduces(n_layers: int, d_model: int, tokens: int, bytes_per_value: float = 2) -> tuple:
    """Tensor parallelism (Megatron split): each layer ends attention and the MLP with a row-parallel matmul
    whose partial sums are all-reduced - 2 all-reduces per layer, each of tokens x d_model activations.
    Returns (all-reduces per forward pass, bytes per all-reduce). Their time is layer 02's alpha-beta model."""
    return 2 * n_layers, tokens * d_model * bytes_per_value


def lora_params(shapes, rank: int, n_layers: int) -> int:
    """A rank-r LoRA adapter on a (d_in, d_out) matrix adds B (d_in x r) and A (r x d_out): r (d_in + d_out)."""
    return n_layers * sum(rank * (d_in + d_out) for d_in, d_out in shapes)


def kv_cache_blocks(gpu: GPU, llm: LLM, gpu_memory_utilization=0.9, block_size=16, reserve_bytes=1e9) -> int:
    """Blocks left for KV after weights and an activation/workspace reserve - roughly what vLLM's
    memory profiling computes at start-up (vllm-serving-lab's sizing.py does it from a real config)."""
    free = gpu.hbm_bytes * gpu_memory_utilization - llm.weight_bytes - reserve_bytes
    return max(0, int(free // (block_size * llm.kv_bytes_per_token)))


@dataclass
class Workload:
    n_requests: int = 200
    rate: float = 4.0                 # Poisson arrivals per second (open loop); inf = all at t = 0
    prompt_len: tuple = (200, 1000)   # uniform, inclusive
    output_len: tuple = (50, 300)
    shared_prefix: int = 0            # leading tokens every prompt shares (a system prompt)
    seed: int = 0

    def requests(self) -> list:
        rng = np.random.default_rng(self.seed)
        prefix = [int(t) for t in rng.integers(0, 32000, self.shared_prefix)]
        t, out = 0.0, []
        for _ in range(self.n_requests):
            t += rng.exponential(1 / self.rate) if np.isfinite(self.rate) else 0.0
            n_in = int(rng.integers(self.prompt_len[0], self.prompt_len[1] + 1))
            n_out = int(rng.integers(self.output_len[0], self.output_len[1] + 1))
            ids = prefix + [int(x) for x in rng.integers(0, 32000, max(1, n_in - self.shared_prefix))]
            out.append((t, ids, n_out))
        return out


@dataclass
class SimResult:
    label: str
    ttft: np.ndarray                  # seconds, per request
    tpot: np.ndarray                  # mean inter-token time per request (seconds)
    itl: np.ndarray                   # every inter-token gap of every request
    e2e: np.ndarray
    duration: float
    output_tokens: int
    steps: int
    preemptions: int
    hit_rate: float
    peak_kv_usage: float

    def pct(self, metric: str, q: float) -> float:
        return float(np.percentile(getattr(self, metric), q))

    @property
    def throughput(self) -> float:
        return self.output_tokens / self.duration

    def goodput(self, ttft_slo: float, tpot_slo: float) -> float:
        """Requests per second that met BOTH SLOs (seconds) - throughput that counts (DistServe)."""
        return float(np.sum((self.ttft <= ttft_slo) & (self.tpot <= tpot_slo))) / self.duration

    def summary(self) -> str:
        ms = lambda m, q: 1e3 * self.pct(m, q)
        return (f"SIMULATED {self.label:<22} TTFT p50 {ms('ttft', 50):7.0f} ms  p99 {ms('ttft', 99):7.0f} ms | "
                f"ITL p50 {ms('itl', 50):5.1f} ms  p99 {ms('itl', 99):6.1f} ms | {self.throughput:6.0f} tok/s | "
                f"preempt {self.preemptions:4d} | hit {self.hit_rate:4.0%} | peak KV {self.peak_kv_usage:4.0%}")


def simulate(gpu: GPU, llm: LLM, workload: Workload, *, max_num_batched_tokens=2048, max_num_seqs=256,
             enable_chunked_prefill=True, enable_prefix_caching=True, block_size=16, num_blocks=None,
             gpu_memory_utilization=0.9, admit_whole_prompt=True, watermark_blocks=0,
             flop_eff=0.6, bw_eff=0.8, overhead_s=0.002, label="") -> SimResult:
    """Run the workload through the real scheduler on a virtual clock. Without chunked prefill the
    budget is raised to the longest sequence, as vLLM requires (a whole prompt must fit in a step).
    The hit rate counts first admissions only (vllm:prefix_cache_hits / _queries); re-admissions
    after preemption are kept apart in the KV manager's CacheStats, as vLLM does."""
    arrivals = workload.requests()
    max_len = max(len(ids) + n for _, ids, n in arrivals) + 1
    budget = max_num_batched_tokens if enable_chunked_prefill else max(max_num_batched_tokens, max_len)
    kv = KVCacheManager(num_blocks or kv_cache_blocks(gpu, llm, gpu_memory_utilization, block_size),
                        block_size, enable_prefix_caching)
    sched = Scheduler(SchedulerConfig(budget, max_num_seqs, enable_chunked_prefill, max_len,
                                      watermark_blocks=watermark_blocks, admit_whole_prompt=admit_whole_prompt), kv)
    pending = deque(Request(f"q{i}", ids, SamplingParams(max_tokens=n, ignore_eos=True), arrival_time=t)
                    for i, (t, ids, n) in enumerate(arrivals))
    reqs, times = list(pending), {f"q{i}": [] for i in range(len(arrivals))}
    rng, clock, steps, peak = np.random.default_rng(workload.seed + 1), 0.0, 0, 0.0
    while pending or sched.has_unfinished():
        while pending and pending[0].arrival_time <= clock:
            sched.add_request(pending.popleft())
        out = sched.schedule()
        if not out.scheduled:
            if out.preempted:
                continue
            if sched.waiting:
                raise RuntimeError("a request can never fit in the KV cache: more blocks or shorter prompts")
            clock = pending[0].arrival_time                     # idle until the next arrival
            continue
        peak = max(peak, kv.usage)
        clock += step_time(gpu, llm, [(r.num_computed_tokens, n) for r, n in out.scheduled],
                           flop_eff=flop_eff, bw_eff=bw_eff, overhead_s=overhead_s)
        sampled = {r.request_id: int(rng.integers(0, 32000)) for r, n in out.scheduled
                   if r.num_computed_tokens + n == r.num_tokens}
        sched.update(out, sampled)
        for rid in sampled:
            times[rid].append(clock)
        steps += 1
    ts = [times[r.request_id] for r in reqs]
    arr = np.array([r.arrival_time for r in reqs])
    first, last = np.array([t[0] for t in ts]), np.array([t[-1] for t in ts])
    tpot = np.array([(t[-1] - t[0]) / (len(t) - 1) if len(t) > 1 else 0.0 for t in ts])
    itl = np.concatenate([np.diff(t) for t in ts])
    return SimResult(label, first - arr, tpot, itl, last - arr, float(last.max() - arr.min()),
                     sum(len(t) for t in ts), steps, sched.num_preemptions, kv.stats.hit_rate, peak)
