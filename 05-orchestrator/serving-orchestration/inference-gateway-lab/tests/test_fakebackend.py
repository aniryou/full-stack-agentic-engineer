"""The emulated engine: vLLM-style prefix cache accounting, timing formulas, metric names."""
import asyncio
import re

import pytest

from igwlab.fakebackend import EmulatedEngine, EngineProfile, KVCache, chat_template, tokenize
from igwlab.promtext import Families


def test_prefix_cache_reuses_full_blocks_up_to_len_minus_one():
    kv = KVCache(num_blocks=100, block_size=16)
    toks = list(range(64))                                   # exactly 4 full blocks
    h = kv.hashes(toks)
    hit, rec = kv.allocate(h, blocks_needed=5, max_hit=(64 - 1) // 16)
    assert hit == 0 and kv.used == 5
    kv.free(rec)
    assert kv.used == 0 and len(kv.free_cached) == 4          # freed but still cached
    hit2, rec2 = kv.allocate(h, 5, (64 - 1) // 16)
    assert hit2 == 3                                          # the last block is recomputed (vLLM caps at len-1)
    kv.free(rec2)
    assert kv.hashes(list(range(70)))[:4] == h                # a longer prompt with the same start shares blocks


def test_eviction_takes_lru_tail_blocks_first():
    kv = KVCache(num_blocks=6, block_size=16)
    a = kv.hashes(list(range(64)))                            # 4 blocks
    _, ra = kv.allocate(a, 4, 3)
    kv.free(ra)                                               # free order puts a[3] (tail) first in LRU
    b = kv.hashes([7] * 64)
    assert kv.can_admit(b, 4, 3)
    _, rb = kv.allocate(b, 4, 3)                              # needs 4, has 2 empty -> evict a[3], a[2]
    assert a[3] not in kv.ref and a[2] not in kv.ref and a[0] in kv.ref and kv.evictions == 2
    assert kv.usage() == pytest.approx(4 / 6)


def test_timing_formulas():
    p = EngineProfile(prefill_overhead_s=0.002, prefill_s_per_token=0.00005, itl_s=0.004, itl_batch_slope=0.02)
    assert p.prefill_seconds(2000, 0) == pytest.approx(0.102)
    assert p.prefill_seconds(2000, 1600) == pytest.approx(0.022)
    assert p.itl_seconds(1) == pytest.approx(0.004) and p.itl_seconds(11) == pytest.approx(0.0048)
    args = p.sim_args("m", 8000)
    assert args[args.index("--prefill-time-per-token") + 1] == "50us" and "--enable-kvcache" in args


def test_template_and_tokenizer_preserve_prefixes():
    base = {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]}
    longer = {"messages": base["messages"] + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]}
    t1, t2 = tokenize(chat_template(base)), tokenize(chat_template(longer))
    assert t2[: len(t1) - 3] == t1[: len(t1) - 3]             # differs only from the generation prompt on
    assert tokenize("a b, c") == tokenize("a b, c") and len(tokenize("a b, c")) == 4


def test_engine_queues_beyond_max_num_seqs_and_counts_hits():
    prof = EngineProfile(max_num_seqs=1, prefill_s_per_token=0.0, prefill_overhead_s=0.0, itl_s=0.01)
    eng = EmulatedEngine(prof)
    toks = tokenize("word " * 100)

    async def run_one():
        async for _ in eng.generate(toks, 5):
            pass

    async def go():
        t1 = asyncio.create_task(run_one())
        await asyncio.sleep(0.005)
        t2 = asyncio.create_task(run_one())
        await asyncio.sleep(0.005)
        snapshot = (len(eng.running), len(eng.waiting))
        await asyncio.gather(t1, t2)
        return snapshot
    assert asyncio.run(go()) == (1, 1)
    f = Families.from_text(eng.metrics_text())
    assert f.value("vllm:prefix_cache_queries_total") == 200 and f.value("vllm:prefix_cache_hits_total") == 96
    assert f.value("vllm:num_requests_running") == 0 and f.value("vllm:generation_tokens_total") == 10
    assert f.value("vllm:request_success_total", finished_reason="length") == 2


def test_emitted_metric_names_exist_in_vllm():
    """Every vllm:* family the fake exports is defined in vLLM's loggers.py (counters add _total)."""
    names = set(re.findall(r"^(vllm:[a-z_]+?)(?:_total|_bucket|_sum|_count)?(?:\{| )",
                           EmulatedEngine(EngineProfile(max_loras=2)).metrics_text(), re.M))
    known = {"vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
             "vllm:prefix_cache_queries", "vllm:prefix_cache_hits", "vllm:num_preemptions", "vllm:prompt_tokens",
             "vllm:generation_tokens", "vllm:request_success", "vllm:time_to_first_token_seconds",
             "vllm:inter_token_latency_seconds", "vllm:request_time_per_output_token_seconds",
             "vllm:e2e_request_latency_seconds", "vllm:request_queue_time_seconds", "vllm:cache_config_info",
             "vllm:lora_requests_info"}
    assert names and names <= known, names - known          # names from vllm/v1/metrics/loggers.py (Sep 2026)
