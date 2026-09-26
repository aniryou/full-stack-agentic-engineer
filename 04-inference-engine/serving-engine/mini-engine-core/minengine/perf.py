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
    bytes_per_param: float = 2.0     # bf16 = 2, fp8/int8 = 1, int4 (g128) ~ 0.52
    kv_bytes_per_value: float = 2.0  # bf16 = 2, fp8 = 1
    compute_scale: float = 1.0       # 2.0 when weights AND activations are FP8 (W8A8) on FP8 hardware

    @property
    def weight_bytes(self) -> float:
        return self.params * self.bytes_per_param

    @property
    def kv_bytes_per_token(self) -> float:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.kv_bytes_per_value


LLMS = {   # from each model's config.json (verify)
    "qwen2.5-0.5b": LLM("qwen2.5-0.5b", 0.494e9, 24, 14, 2, 64),
    "qwen3-0.6b": LLM("qwen3-0.6b", 0.596e9, 28, 16, 8, 128),
    "qwen2.5-1.5b": LLM("qwen2.5-1.5b", 1.54e9, 28, 12, 2, 128),
    "llama-3.1-8b": LLM("llama-3.1-8b", 8.03e9, 32, 32, 8, 128),
}


def step_cost(gpu: GPU, llm: LLM, chunks, flop_eff=0.6, bw_eff=0.8, overhead_s=0.002) -> dict:
    """chunks: [(start, n)] per scheduled request - n new tokens after `start` tokens already cached.
    Each request reads its cached KV once; attention does 4 x layers x heads x head_dim FLOPs per
    (query, key) pair. Returns FLOPs, bytes, both times and which bound won."""
    if llm.compute_scale > 1 and not gpu.fp8:
        raise ValueError(f"{gpu.name} has no FP8 tensor cores: FP8 weights run weight-only (compute_scale=1)")
    tokens = sum(n for _, n in chunks)
    pairs = sum(n * s + n * (n + 1) / 2 for s, n in chunks)
    flops = 2 * llm.params * tokens + 4 * llm.n_layers * llm.n_heads * llm.head_dim * pairs
    nbytes = llm.weight_bytes + llm.kv_bytes_per_token * (sum(s for s, _ in chunks) + tokens)
    t_mem = nbytes / (gpu.hbm_bw * bw_eff)
    t_cmp = flops / (gpu.peak_flops * llm.compute_scale * flop_eff)
    return {"flops": flops, "bytes": nbytes, "t_memory": t_mem, "t_compute": t_cmp,
            "t": max(t_mem, t_cmp) + overhead_s, "bound": "memory" if t_mem >= t_cmp else "compute"}


def step_time(gpu: GPU, llm: LLM, chunks, **kw) -> float:
    return step_cost(gpu, llm, chunks, **kw)["t"]


def knee_tokens(gpu: GPU, llm: LLM, flop_eff=1.0, bw_eff=1.0) -> float:
    """Batch size (tokens) where compute time catches up with the weight read (KV ignored):
    2 P n / PEAK = P b / BW  ->  n = PEAK x b / (2 BW). H100 bf16: 989e12 x 2 / (2 x 3.35e12) = 295."""
    return gpu.peak_flops * llm.compute_scale * flop_eff * llm.bytes_per_param / (2 * gpu.hbm_bw * bw_eff)


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
        """Requests per second that met BOTH SLOs - throughput that counts."""
        return float(np.sum((self.ttft <= ttft_slo) & (self.tpot <= tpot_slo))) / self.duration

    def summary(self) -> str:
        ms = lambda m, q: 1e3 * self.pct(m, q)
        return (f"SIMULATED {self.label:<22} TTFT p50 {ms('ttft', 50):7.0f} ms  p99 {ms('ttft', 99):7.0f} ms | "
                f"ITL p50 {ms('itl', 50):5.1f} ms  p99 {ms('itl', 99):6.1f} ms | {self.throughput:6.0f} tok/s | "
                f"preempt {self.preemptions:4d} | hit {self.hit_rate:4.0%} | peak KV {self.peak_kv_usage:4.0%}")


def simulate(gpu: GPU, llm: LLM, workload: Workload, *, max_num_batched_tokens=2048, max_num_seqs=256,
             enable_chunked_prefill=True, enable_prefix_caching=True, block_size=16, num_blocks=None,
             gpu_memory_utilization=0.9, flop_eff=0.6, bw_eff=0.8, overhead_s=0.002, label="") -> SimResult:
    """Run the workload through the real scheduler on a virtual clock. Without chunked prefill the
    budget is raised to the longest sequence, as vLLM requires (a whole prompt must fit in a step)."""
    arrivals = workload.requests()
    max_len = max(len(ids) + n for _, ids, n in arrivals) + 1
    budget = max_num_batched_tokens if enable_chunked_prefill else max(max_num_batched_tokens, max_len)
    kv = KVCacheManager(num_blocks or kv_cache_blocks(gpu, llm, gpu_memory_utilization, block_size),
                        block_size, enable_prefix_caching)
    sched = Scheduler(SchedulerConfig(budget, max_num_seqs, enable_chunked_prefill, max_len), kv)
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
