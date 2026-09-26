"""engine.py - the loop: schedule -> build the flat batch -> one forward pass -> sample -> update.

The one idea: an engine is a loop, and one iteration is one forward pass over whatever the
scheduler picked. Continuous batching, chunked prefill, prefix caching and preemption are all
decisions the scheduler makes *between* two iterations. `Engine.step` is the whole thing;
`generate` just calls it until nothing is left. In vLLM the same loop is EngineCore.step (the
scheduler and the GPU workers), while the API server, tokenizer and detokenizer live in another
process so that HTTP and string handling never stall the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kv import KVCacheManager, hash_block
from .model import EOS, Batch, PagedKVCache, TinyLM, decode, encode
from .sampler import SamplingParams, sample
from .scheduler import Request, Scheduler, SchedulerConfig, Status


@dataclass
class RequestOutput:
    request_id: str
    text: str
    token_ids: list
    finished: bool
    finish_reason: str | None
    num_cached_tokens: int               # prompt tokens served from the prefix cache
    logprobs: list                       # per generated token: (raw logprob, {token: logprob} or None)


@dataclass
class StepRecord:
    """What one iteration did - printed as one line of an engine trace."""
    step: int
    batch: list                          # [(request_id, kind, start, n, sampled token or None)]
    preempted: list
    finished: list
    waiting: list
    kv_used: int
    num_blocks: int

    @property
    def num_batched_tokens(self) -> int:
        return sum(e[3] for e in self.batch)

    def __str__(self):
        parts = []
        for rid, kind, start, n, tok in self.batch:
            out = f"->{decode([tok])!r}" if tok is not None and tok < 256 else ("->EOS" if tok is not None else "")
            parts.append(f"{rid} {kind} {n} [{start}:{start + n}]{out}")
        line = f"step {self.step:3d} | {self.num_batched_tokens:3d} tok | " + " | ".join(parts or ["idle"])
        line += f" | kv {self.kv_used}/{self.num_blocks}"
        for label, ids in (("preempted", self.preempted), ("finished", self.finished), ("waiting", self.waiting)):
            if ids:
                line += f" | {label} {','.join(ids)}"
        return line


class Engine:
    def __init__(self, model: TinyLM | None = None, *, num_blocks: int = 64, block_size: int = 16,
                 max_num_batched_tokens: int = 64, max_num_seqs: int = 8, enable_chunked_prefill: bool = True,
                 enable_prefix_caching: bool = True, max_model_len: int = 512, watermark_blocks: int = 0,
                 admit_whole_prompt: bool = True, kv_dtype: str = "float64", hash_fn=hash_block, seed: int = 0):
        self.model = model or TinyLM()
        self.kv = KVCacheManager(num_blocks, block_size, enable_prefix_caching, hash_fn)
        self.cache = PagedKVCache(self.model.cfg, num_blocks, block_size, kv_dtype)
        self.scheduler = Scheduler(SchedulerConfig(max_num_batched_tokens, max_num_seqs, enable_chunked_prefill,
                                                   max_model_len, watermark_blocks, admit_whole_prompt), self.kv)
        self.requests: dict[str, Request] = {}
        self.history: list[StepRecord] = []
        self.logprobs: dict[str, list] = {}
        self.stop_text: dict[str, str] = {}
        self.seed = seed

    # -- the API ----------------------------------------------------------------------------------
    def add_request(self, prompt, params: SamplingParams | None = None, request_id: str | None = None,
                    cache_extra=None) -> str:
        rid = request_id or f"r{len(self.requests)}"
        p = params or SamplingParams()
        rng = np.random.default_rng(p.seed if p.seed is not None else [self.seed, len(self.requests)])
        req = Request(rid, encode(prompt) if isinstance(prompt, str) else list(prompt), p, cache_extra=cache_extra,
                      eos_token_id=EOS, rng=rng, fsm_state=p.fsm.start() if p.fsm else None)
        self.scheduler.add_request(req)
        self.requests[rid], self.logprobs[rid] = req, []
        return rid

    def abort(self, request_id: str):
        req = self.requests[request_id]
        if not req.status.finished:
            self.scheduler.finish(req, Status.FINISHED_ABORTED)

    def step(self) -> list[RequestOutput]:
        """One iteration. Returns an output for every request that produced a token."""
        out = self.scheduler.schedule()
        if not out.scheduled and self.scheduler.waiting and not self.scheduler.running:
            raise RuntimeError("the next waiting request can never fit in the KV cache: raise num_blocks")
        sampled = {}
        if out.scheduled:
            logits = self.model.forward(self.build_batch(out.scheduled), self.cache)
            for (req, n), row in zip(out.scheduled, logits):
                if req.num_computed_tokens + n < req.num_tokens:
                    continue                                    # mid-prompt chunk: its logits are thrown away
                fsm = req.params.fsm
                tok, lp, top = sample(row, req.params, req.rng, req.token_ids[:req.num_prompt_tokens],
                                      req.output_token_ids, fsm.allowed(req.fsm_state) if fsm else None)
                sampled[req.request_id] = tok
                self.logprobs[req.request_id].append((lp, top))
                if fsm:
                    req.fsm_state = fsm.advance(req.fsm_state, tok)
        batch_log = [(r.request_id, self._kind(r, n), r.num_computed_tokens, n, sampled.get(r.request_id))
                     for r, n in out.scheduled]
        finished = self.scheduler.update(out, sampled)
        for rid in sampled:                                     # stop strings: the detokenizer's job
            req = self.requests[rid]
            text = decode(req.output_token_ids)
            cuts = [text.index(s) for s in req.params.stop if s in text]
            if cuts and not req.status.finished:
                self.stop_text[rid] = text[:min(cuts)]
                self.scheduler.finish(req, Status.FINISHED_STOPPED)
                finished.append(req)
        self.history.append(StepRecord(out.step, batch_log, [r.request_id for r in out.preempted],
                                       [r.request_id for r in finished],
                                       [r.request_id for r in self.scheduler.waiting],
                                       self.kv.num_blocks - self.kv.num_free_blocks, self.kv.num_blocks))
        return [self.output(rid) for rid in sampled]

    def generate(self, prompts, params=None) -> list[RequestOutput]:
        """Add every prompt, step until all are done, return the final outputs in order."""
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        plist = list(params) if isinstance(params, (list, tuple)) else [params] * len(prompts)
        rids = [self.add_request(p, sp) for p, sp in zip(prompts, plist)]
        while self.scheduler.has_unfinished():
            self.step()
        return [self.output(r) for r in rids]

    def output(self, rid: str) -> RequestOutput:
        req = self.requests[rid]
        done = req.status.finished
        return RequestOutput(rid, self.stop_text.get(rid, decode(req.output_token_ids)), list(req.output_token_ids),
                             done, req.status.value if done else None, req.num_cached_tokens, self.logprobs[rid])

    # -- internals --------------------------------------------------------------------------------
    def build_batch(self, scheduled) -> Batch:
        """Flatten the step: every request's new tokens, positions, write slots and block table."""
        B = self.kv.block_size
        toks, pos, slots, starts, tables, ctx = [], [], [], [0], [], []
        for req, n in scheduled:
            c, table = req.num_computed_tokens, self.kv.tables[req.request_id]
            toks += req.token_ids[c:c + n]
            pos += range(c, c + n)
            slots += [table[p // B] * B + p % B for p in range(c, c + n)]
            starts.append(starts[-1] + n)
            tables.append(list(table))
            ctx.append(c + n)
        return Batch(np.array(toks), np.array(pos), np.array(slots), starts, tables, ctx)

    @staticmethod
    def _kind(req: Request, n: int) -> str:
        if req.num_computed_tokens + n < req.num_tokens:
            return "chunk"                                      # part of a prompt; no token this step
        if n == 1 and req.output_token_ids:
            return "decode"
        return "recompute" if req.output_token_ids else "prefill"

    def trace(self, last: int | None = None) -> str:
        return "\n".join(str(r) for r in self.history[-last if last else 0:])
