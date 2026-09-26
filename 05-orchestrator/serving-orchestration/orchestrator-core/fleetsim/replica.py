"""A replica is a stateful cache, not a stateless server: the engine model every router and autoscaler sees.

One Replica = one inference-engine instance (one model copy, any tensor parallelism inside it). It models
exactly what the orchestration layer can observe or exploit:

  * a waiting queue and a running batch, scheduled every step under a token budget (continuous batching with
    chunked prefill — running requests first, then waiting ones FCFS),
  * a paged KV block pool with a hash-based prefix cache: full blocks are registered by chain hash, and freed
    blocks stay cached in LRU order (a request's tail blocks are evicted first) until the pool needs them,
  * preemption by recompute when decode runs out of blocks,
  * a roofline step time, overhead + max(compute, memory), so one prefill chunk stalls every decode in the batch.

The profile numbers come from spec-sheet arithmetic in `engine_profile()`; nothing here is measured.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineProfile:
    name: str
    weight_read_s: float        # weight bytes / HBM bandwidth: every step streams the weights once
    compute_tok_s: float        # tokens/s the matmuls sustain at the assumed MFU (the prefill speed)
    kv_read_s: float            # seconds to read one token's K and V from HBM
    kv_blocks: int              # blocks in the KV pool
    kv_bytes_per_token: int
    block: int = 16             # tokens per KV block
    overhead_s: float = 0.003   # per-step scheduling, kernel launches, sampling
    max_tokens: int = 2048      # token budget per step (vLLM max_num_batched_tokens)
    max_seqs: int = 128         # vLLM max_num_seqs
    max_loras: int = 4          # adapters resident at once (vLLM max_loras)
    lora_load_s: float = 0.2    # load an adapter into a slot (illustrative)


def engine_profile(name, *, params_b, bytes_per_param, kv_bytes_per_token, hbm_gb, hbm_tb_s, dense_tflops,
                   mfu=0.5, mem_util=0.9, reserve_gb=1.0, **kw) -> EngineProfile:
    """Derive an engine from a spec sheet: weights = params x bytes; KV pool = mem_util x HBM - weights - reserve;
    decode floor = weights / bandwidth; prefill speed = MFU x peak FLOP/s / (2 x params) tokens/s."""
    weights = params_b * 1e9 * bytes_per_param
    block = kw.get("block", 16)
    kv = hbm_gb * 1e9 * mem_util - weights - reserve_gb * 1e9
    return EngineProfile(name, weights / (hbm_tb_s * 1e12), dense_tflops * 1e12 * mfu / (2 * params_b * 1e9),
                         kv_bytes_per_token / (hbm_tb_s * 1e12), int(kv // (kv_bytes_per_token * block)),
                         kv_bytes_per_token, **kw)


LLAMA_8B_KV = 2 * 32 * 8 * 128 * 2    # 131,072 B/token = 2 (K,V) x 32 layers x 8 KV heads x 128 dims x 2 B (bf16)
# Spec-sheet values (verify against 01-hardware-gpu-fabric): L4 24 GB, 0.3 TB/s, 121 dense bf16 TFLOP/s;
# H100 SXM 80 GB, 3.35 TB/s, 989 dense bf16 TFLOP/s.
L4_8B = engine_profile("8B bf16 on 1x L4 (simulated)", params_b=8, bytes_per_param=2,
                       kv_bytes_per_token=LLAMA_8B_KV, hbm_gb=24, hbm_tb_s=0.3, dense_tflops=121)
H100_8B = engine_profile("8B bf16 on 1x H100 (simulated)", params_b=8, bytes_per_param=2,
                         kv_bytes_per_token=LLAMA_8B_KV, hbm_gb=80, hbm_tb_s=3.35, dense_tflops=989, reserve_gb=2)


def step_time(p: EngineProfile, tokens: int, kv_tokens: int) -> float:
    """One engine step: overhead + max(tokens / compute speed, weight read + KV read) — a roofline."""
    return p.overhead_s + max(tokens / p.compute_tok_s, p.weight_read_s + kv_tokens * p.kv_read_s)


class BlockPool:
    """Paged KV blocks with a prefix cache, vLLM V1 style: refcounts, a hash -> block map for full blocks, and
    one LRU queue of free blocks. A freed block keeps its hash until it is reallocated — that is the cache."""

    def __init__(self, n: int):
        self.n, self.ref, self.hash_of = n, [0] * n, [None] * n
        self.cached: dict[int, int] = {}                # block hash -> block id
        self.free = OrderedDict.fromkeys(range(n))      # refcount-0 blocks; front = evicted first

    def usage(self) -> float:                           # vllm:kv_cache_usage_perc (cached-but-free counts as free)
        return 1.0 - len(self.free) / self.n

    def match(self, hashes, limit: int) -> list[int]:
        """Blocks of the longest cached prefix (at most `limit` blocks)."""
        out = []
        for i in range(limit):
            b = self.cached.get(hashes[i])
            if b is None:
                break
            out.append(b)
        return out

    def take(self, blocks):
        for b in blocks:
            if self.ref[b] == 0:
                del self.free[b]
            self.ref[b] += 1

    def allocate(self) -> int | None:
        if not self.free:
            return None
        b, _ = self.free.popitem(last=False)            # least recently freed: evict its cached content
        h = self.hash_of[b]
        if h is not None:
            if self.cached.get(h) == b:
                del self.cached[h]
            self.hash_of[b] = None
        self.ref[b] = 1
        return b

    def register(self, b: int, h: int):
        if h not in self.cached:
            self.cached[h], self.hash_of[b] = b, h

    def release(self, blocks):
        for b in reversed(blocks):                      # tail first, so a shared prefix is evicted last
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free[b] = None


class _Seq:
    __slots__ = ("req", "target", "computed", "ctx", "out", "blocks", "nreg", "last", "running", "kv_ready")

    def __init__(self, req, kv_ready=False):
        self.req, self.kv_ready, self.running, self.blocks, self.nreg = req, kv_ready, False, [], 0
        self.target = req.prompt                        # tokens to prefill (prompt + output-so-far after preemption)
        self.computed = self.ctx = req.prompt if kv_ready else 0
        self.out = 1 if kv_ready else 0                 # a KV-ready request arrives with its first token
        self.last = req.t_first


class Replica:
    def __init__(self, rid, profile: EngineProfile, *, role="both", now=0.0, ready=True, max_age=0.0):
        self.rid, self.p, self.role, self.max_age = rid, profile, role, max_age   # role: both | prefill
        self.pool = BlockPool(profile.kv_blocks)
        self.waiting, self.running, self.loras = deque(), [], OrderedDict()
        self.ready, self.draining, self.busy = ready, False, False
        self.born, self.ready_at, self.died = now, now if ready else None, None
        self.busy_time, self.itl = 0.0, []
        self.stats = dict(requests=0, prompt=0, cached=0, tokens=0, preempted=0, steps=0)
        self._batch, self._t0, self._snap, self._snap_t, self._busy_mark = [], 0.0, None, -1.0, 0.0

    # -- what a router or autoscaler can observe ---------------------------------------------------------
    def metrics(self, now: float) -> dict:
        """A scrape of the engine's /metrics, at most `max_age` s old (vLLM names in comments)."""
        if self._snap is None or self.max_age <= 0 or now - self._snap_t > self.max_age:
            self._snap = {"waiting": len(self.waiting),          # vllm:num_requests_waiting
                          "running": len(self.running),          # vllm:num_requests_running
                          "kv": self.pool.usage(),               # vllm:kv_cache_usage_perc
                          "loras": tuple(self.loras)}            # vllm:lora_requests_info
            self._snap_t = now
        return self._snap

    def cached_tokens(self, req) -> int:
        """Prompt tokens this replica could serve from its prefix cache right now (the precise view)."""
        return len(self.pool.match(req.hashes, req.pblocks)) * self.p.block

    def idle(self) -> bool:
        return not self.waiting and not self.running

    def enqueue(self, req, now, kv_ready=False):
        if -(-(req.prompt + req.output) // self.p.block) > self.p.kv_blocks:
            raise ValueError(f"request {req.rid} needs more KV than replica {self.rid} has")
        self.waiting.append(_Seq(req, kv_ready))
        self.stats["requests"] += 1

    # -- one engine step ---------------------------------------------------------------------------------
    def start_step(self, now: float) -> float | None:
        """Pick this step's batch (running first, then waiting FCFS) and return its duration; None if idle."""
        batch, budget, self._lora_s, B = [], self.p.max_tokens, 0.0, self.p.block
        for s in list(self.running):
            if budget <= 0:
                break
            if not s.running:                           # preempted a moment ago to make room
                continue
            n = 1 if s.computed >= s.target else min(s.target - s.computed, budget)
            if s.ctx + n <= len(s.blocks) * B or self._grow(s, s.ctx + n):     # a new block only every B tokens
                batch.append((s, n))
                budget -= n
        skipped = []
        while self.waiting and budget > 0 and len(self.running) < self.p.max_seqs:
            s = self.waiting.popleft()
            if s.req.lora and not self._lora_slot(s.req.lora):
                skipped.append(s)                       # no adapter slot free: let later requests go first
                continue
            if not self._admit(s, budget):
                self.waiting.appendleft(s)              # no KV room: FCFS, the head blocks the queue
                break
            self.running.append(s)
            s.running = True
            n = 1 if s.computed >= s.target else min(s.target - s.computed, budget)
            batch.append((s, n))
            budget -= n
        self.waiting.extendleft(reversed(skipped))
        if not batch:
            return None
        tokens = kv_tokens = 0
        for s, n in batch:
            tokens, kv_tokens = tokens + n, kv_tokens + s.ctx + n      # attention reads the whole context
        dur = step_time(self.p, tokens, kv_tokens) + self._lora_s
        self._batch, self._t0 = batch, now
        self.busy_time += dur
        self.stats["tokens"] += tokens
        self.stats["steps"] += 1
        return dur

    def end_step(self, now: float) -> list[tuple[str, object]]:
        """Apply the step: prefill progress, one token per decoding request. Returns (event, request) pairs:
        'first' (first token), 'done' (finished), 'handoff' (a prefill-only replica finished the prompt)."""
        events, B = [], self.p.block
        for s, n in self._batch:
            r = s.req
            if s.computed < s.target:
                s.computed += n
                s.ctx = s.computed
                self._register(s)
                if s.computed < s.target:               # more prompt chunks to go
                    continue
            else:
                s.ctx += 1
                if s.ctx % B == 0:                      # a decode block just filled: now it is cacheable
                    self._register(s)
            s.out += 1
            if r.t_first is None:
                r.t_first = now
                events.append(("first", r))
            else:
                self.itl.append(now - s.last)
            s.last = now
            if self.role == "prefill" or s.out >= r.output:
                self.running.remove(s)
                s.running = False
                self.pool.release(s.blocks)
                if self.role != "prefill":
                    r.t_done = now
                events.append(("handoff" if self.role == "prefill" else "done", r))
        self._batch = []
        return events

    # -- KV bookkeeping ----------------------------------------------------------------------------------
    def _admit(self, s, budget) -> bool:
        B, pool = self.p.block, self.pool
        if s.kv_ready:                                  # KV arrived over the network: no lookup, no prefill
            need = -(-(s.ctx + 1) // B)
            if len(pool.free) < need:
                return False
            s.blocks = [pool.allocate() for _ in range(need)]
            self._register(s)
            return True
        hit = pool.match(s.req.hashes, min((s.target - 1) // B, s.req.nblocks))   # always compute >= 1 token
        n = min(s.target - len(hit) * B, budget)
        need = -(-(len(hit) * B + n) // B) - len(hit)
        if len(pool.free) - sum(1 for b in hit if pool.ref[b] == 0) < need:
            return False
        pool.take(hit)
        s.blocks = hit + [pool.allocate() for _ in range(need)]
        s.computed = s.ctx = len(hit) * B
        s.nreg = len(hit)
        if s.req.t_first is None and not s.req.preempted:
            s.req.cached = len(hit) * B
            self.stats["prompt"] += s.req.prompt
            self.stats["cached"] += len(hit) * B
        return True

    def _grow(self, s, upto: int) -> bool:
        """Give `s` blocks for `upto` tokens, preempting the newest running request if the pool is dry."""
        need = -(-upto // self.p.block) - len(s.blocks)
        while need > 0:
            b = self.pool.allocate()
            if b is None:
                victim = self.running[-1]
                self._preempt(victim)
                if victim is s:
                    return False
                continue
            s.blocks.append(b)
            need -= 1
        return True

    def _preempt(self, v):
        """Preemption by recompute: free every block; later re-prefill prompt + tokens generated so far."""
        self.running.remove(v)
        self.pool.release(v.blocks)
        v.running, v.kv_ready, v.blocks, v.nreg = False, False, [], 0
        v.target, v.computed, v.ctx = v.req.prompt + v.out, 0, 0
        v.req.preempted += 1
        self.stats["preempted"] += 1                    # vllm:num_preemptions
        self.waiting.appendleft(v)

    def _register(self, s):
        full = min(s.ctx // self.p.block, s.req.nblocks)
        while s.nreg < full:
            self.pool.register(s.blocks[s.nreg], s.req.hashes[s.nreg])
            s.nreg += 1

    def _lora_slot(self, lora) -> bool:
        if lora in self.loras:
            self.loras.move_to_end(lora)
            return True
        if len(self.loras) >= self.p.max_loras:
            busy = {s.req.lora for s in self.running}
            victim = next((a for a in self.loras if a not in busy), None)
            if victim is None:
                return False                            # every slot serves a running request
            del self.loras[victim]
        self.loras[lora] = None
        self._lora_s += self.p.lora_load_s
        return True
