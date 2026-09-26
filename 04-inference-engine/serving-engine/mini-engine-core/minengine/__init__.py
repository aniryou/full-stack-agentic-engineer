"""minengine - a numpy "nano-vLLM": the inside of an inference engine, small enough to read in a sitting.

    from minengine import Engine, SamplingParams
    eng = Engine(num_blocks=32, block_size=4, max_num_batched_tokens=16)
    print(eng.generate(["The engine ", "When memory "], SamplingParams(max_tokens=12))[0].text)
    print(eng.trace())                     # one line per step: who ran, how many tokens, KV blocks used

Read in this order - each module opens with the one idea it teaches:
    model.py      a tiny Llama-style decoder whose attention reads K/V through block tables
    kv.py         block pool, refcounts, hash-chained prefix cache with LRU eviction
    scheduler.py  continuous batching: token budget, chunked prefill, FCFS, preemption by recompute
    sampler.py    temperature / top-k / top-p / min-p / penalties / seeds / logprobs / FSM masks
    engine.py     the step loop that ties them together
    spec.py       speculative decoding with exact rejection sampling
    quant.py      int8 / int4 / fp8 quantization and the errors they cost
    perf.py       a roofline step-time model driving the real scheduler: SIMULATED TTFT/ITL
"""
from . import perf, quant, spec
from .engine import Engine, RequestOutput, StepRecord
from .kv import KVCacheManager, block_hashes, hash_block
from .model import EOS, SMALL, VOCAB, Batch, ModelConfig, PagedKVCache, TinyLM, decode, encode
from .sampler import ChoiceFSM, SamplingParams, sample
from .scheduler import Request, Scheduler, SchedulerConfig, SchedulerOutput, Status

__all__ = [
    "Engine", "RequestOutput", "StepRecord", "KVCacheManager", "block_hashes", "hash_block",
    "EOS", "SMALL", "VOCAB", "Batch", "ModelConfig", "PagedKVCache", "TinyLM", "decode", "encode",
    "ChoiceFSM", "SamplingParams", "sample", "Request", "Scheduler", "SchedulerConfig", "SchedulerOutput",
    "Status", "perf", "quant", "spec",
]
