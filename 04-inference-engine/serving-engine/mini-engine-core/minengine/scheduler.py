"""scheduler.py - continuous batching: every step, decide which requests run and how many tokens each gets.

The one idea (vLLM V1's unified scheduler): there are no "prefill steps" and "decode steps". A
request has num_tokens (prompt + generated so far) and num_computed_tokens (tokens whose K/V are
in the cache). Each step hands out a token budget, max_num_batched_tokens:
  1. RUNNING requests first, in admission order, each getting min(num_tokens - num_computed_tokens,
     budget left): 1 for a decoding request, a chunk for one still prefilling (chunked prefill).
     If one needs a KV block and none is free, the newest running request is PREEMPTED: its
     blocks are freed, num_computed_tokens drops to 0, and it goes to the FRONT of the waiting
     queue to be recomputed later (preemption by recompute; its generated tokens are kept).
  2. WAITING requests in FCFS order - only if nobody was preempted this step - while budget,
     max_num_seqs and free blocks allow. A prefix-cache hit starts a request part-way computed.
A request samples a token only in the step where its computed tokens reach the end of its
sequence; a prefill chunk that stops short samples nothing. Full blocks are published to the prefix
cache at scheduling time: running requests' once the running pass is final (a request preempted
later in that pass never publishes blocks it will not compute), each admitted request's right after
its admission - so the next waiting request can hit them in the same step, as in vLLM.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .kv import KVCacheManager
from .sampler import SamplingParams


class Status(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED_STOPPED = "finished_stopped"        # EOS, a stop token or a stop string
    FINISHED_LENGTH_CAPPED = "finished_length_capped"   # max_tokens or max_model_len
    FINISHED_ABORTED = "finished_aborted"

    @property
    def finished(self) -> bool:
        return self.value.startswith("finished")


@dataclass(eq=False)                             # identity semantics: two requests are never "equal"
class Request:
    request_id: str
    token_ids: list                              # the prompt; every generated token is appended
    params: SamplingParams = field(default_factory=SamplingParams)
    arrival_time: float = 0.0
    cache_extra: object = None                   # e.g. a LoRA id: part of every block name
    priority: int = 0                            # lower = more important (only custom victim policies use it)
    eos_token_id: int | None = None
    num_prompt_tokens: int = -1
    num_computed_tokens: int = 0
    num_cached_tokens: int = 0                   # prefix-cache hit at first admission
    num_preemptions: int = 0
    status: Status = Status.WAITING
    rng: object = None                           # the engine gives each request its own generator
    fsm_state: object = None

    def __post_init__(self):
        self.token_ids = [int(t) for t in self.token_ids]
        if self.num_prompt_tokens < 0:
            self.num_prompt_tokens = len(self.token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def output_token_ids(self) -> list:
        return self.token_ids[self.num_prompt_tokens:]


@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 64             # the token budget per step
    max_num_seqs: int = 8                        # at most this many running requests
    enable_chunked_prefill: bool = True
    max_model_len: int = 512
    watermark_blocks: int = 0                    # keep this many blocks free when admitting
    admit_whole_prompt: bool = True              # admit only if the whole prompt's blocks are free

    def __post_init__(self):
        if not self.enable_chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("without chunked prefill, max_num_batched_tokens must be >= max_model_len "
                             "(a prompt must fit in one step)")


@dataclass
class SchedulerOutput:
    step: int
    scheduled: list                              # [(Request, num_new_tokens)] in batch order
    preempted: list

    @property
    def num_batched_tokens(self) -> int:
        return sum(n for _, n in self.scheduled)


class Scheduler:
    def __init__(self, config: SchedulerConfig, kv: KVCacheManager):
        if config.max_model_len > kv.num_blocks * kv.block_size:     # vLLM refuses to start in this case
            raise ValueError(f"max_model_len {config.max_model_len} exceeds the KV cache "
                             f"({kv.num_blocks * kv.block_size} tokens): one sequence could never finish")
        self.cfg, self.kv = config, kv
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.num_steps = 0
        self.num_preemptions = 0

    def add_request(self, req: Request):
        if req.num_tokens >= self.cfg.max_model_len:
            raise ValueError(f"{req.request_id}: prompt of {req.num_tokens} tokens exceeds max_model_len")
        self.waiting.append(req)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def pick_victim(self) -> Request:
        """Who loses its KV cache when memory runs out. FCFS: the most recently admitted."""
        return self.running[-1]

    def schedule(self) -> SchedulerOutput:
        cfg, kv = self.cfg, self.kv
        budget, scheduled, preempted = cfg.max_num_batched_tokens, [], []
        # 1) running requests: decodes and unfinished prefill chunks
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(req.num_tokens - req.num_computed_tokens, budget)
            while kv.allocate_slots(req.request_id, req.token_ids, req.num_computed_tokens, n) is None:
                victim = self.pick_victim()
                if self.running.index(victim) < i:
                    i -= 1
                for j, (r, m) in enumerate(scheduled):      # a victim already in this batch gives its tokens back
                    if r is victim:
                        scheduled.pop(j)
                        budget += m
                        break
                self._preempt(victim)
                preempted.append(victim)
                if victim is req:
                    break
            if req.status is not Status.RUNNING:
                break                                       # it had to preempt itself: memory is exhausted
            scheduled.append((req, n))
            budget -= n
            i += 1
        for req, n in scheduled:                            # final now: publish the blocks this step fills
            kv.cache_blocks(req.request_id, req.token_ids, req.num_computed_tokens + n, req.cache_extra)
        # 2) waiting requests, FCFS, only if memory was not just short
        while not preempted and self.waiting and budget > 0 and len(self.running) < cfg.max_num_seqs:
            req = self.waiting[0]
            hits = kv.lookup(req.token_ids, req.cache_extra)
            num_cached = len(hits) * kv.block_size
            n = req.num_tokens - num_cached
            if n > budget and not cfg.enable_chunked_prefill:
                break                                       # the whole prompt must fit in one step
            n = min(n, budget)
            if kv.allocate_slots(req.request_id, req.token_ids, num_cached, n, hits,
                                 reserve=cfg.watermark_blocks if self.running else 0,
                                 admit_whole_prompt=cfg.admit_whole_prompt,
                                 preempted=req.num_preemptions > 0) is None:
                break                                       # head-of-line waits for memory
            kv.cache_blocks(req.request_id, req.token_ids, num_cached + n, req.cache_extra)   # hittable now
            self.waiting.popleft()
            req.num_computed_tokens = num_cached
            if req.num_preemptions == 0:
                req.num_cached_tokens = num_cached
            req.status = Status.RUNNING
            self.running.append(req)
            scheduled.append((req, n))
            budget -= n
        self.num_steps += 1
        return SchedulerOutput(self.num_steps, scheduled, preempted)

    def _preempt(self, req: Request):
        self.running.remove(req)
        self.kv.free(req.request_id)
        req.status, req.num_computed_tokens = Status.PREEMPTED, 0
        req.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(req)

    def update(self, out: SchedulerOutput, sampled: dict) -> list[Request]:
        """After the forward pass: advance computed tokens, append sampled tokens, finish requests
        that hit a stop condition. Returns the requests that finished."""
        finished = []
        for req, n in out.scheduled:
            req.num_computed_tokens += n
            if req.request_id not in sampled:
                continue                                    # a prefill chunk that stopped short
            tok = sampled[req.request_id]
            req.token_ids.append(tok)
            status = self.check_stop(req, tok)
            if status:
                self.finish(req, status)
                finished.append(req)
        return finished

    def check_stop(self, req: Request, tok: int):
        p = req.params
        if (tok == req.eos_token_id and not p.ignore_eos) or tok in p.stop_token_ids:
            return Status.FINISHED_STOPPED
        if len(req.output_token_ids) >= p.max_tokens or req.num_tokens >= self.cfg.max_model_len:
            return Status.FINISHED_LENGTH_CAPPED
        return None

    def finish(self, req: Request, status: Status):
        if req in self.running:
            self.running.remove(req)
        elif req in self.waiting:
            self.waiting.remove(req)
        self.kv.free(req.request_id)
        req.status = status
