"""The context budget: vLLM's block rules, the three memory layouts, prefill time and dollars per turn."""
import random

import pytest

from memcore import (CACHE_MIN_TOKENS, GPUS, LLMS, Budget, BudgetExceeded, PrefixCache, Salts, billed_cached,
                     cache_salt, call_cost, compute_ttft, expected_cached_tokens, hit_rate, hits_per_turn,
                     prefill_seconds, token_ids, turn_cost)


def test_token_ids_match_count_tokens_and_keep_prefixes():
    a, b = token_ids("System prompt. User: hi"), token_ids("System prompt. User: hello there")
    assert len(token_ids("abcdefgh")) == 2 and len(token_ids("abc")) == 1 and token_ids("") == []
    assert a[:4] == b[:4]                                                   # 16 shared characters -> 4 shared tokens


def test_prefix_cache_agrees_with_expected_cached_tokens():
    rng = random.Random(0)
    for _ in range(300):
        B = rng.choice([4, 8, 16])
        base = [rng.randrange(5) for _ in range(rng.randrange(1, 80))]
        prev_prompt, prev_out = base[:rng.randrange(1, len(base) + 1)], [rng.randrange(5) for _ in range(rng.randrange(0, 10))]
        new = base[:rng.randrange(0, len(base) + 1)] + [rng.randrange(5) for _ in range(rng.randrange(1, 30))]
        cache = PrefixCache(B)
        cache.serve(prev_prompt, prev_out)
        assert cache.lookup(new) == expected_cached_tokens(prev_prompt + prev_out, new, B)


def test_cache_salt_keeps_tenants_apart():
    prompt = list(range(64))
    cache = PrefixCache(16)
    cache.serve(prompt, salt="tenant-a")
    assert cache.lookup(prompt, salt="tenant-a") == 48 and cache.lookup(prompt, salt="tenant-b") == 0
    assert cache.lookup(prompt) == 0 and cache.count("tenant-a") == 3
    assert not hasattr(cache, "evict")                   # vLLM v0.30.0 has no eviction by salt; only a full reset
    assert cache.reset() and cache.lookup(prompt, salt="tenant-a") == 0 and cache.names == {}


def test_salts_are_secret_valid_for_vllm_and_rotate():
    salts = Salts(b"server-secret")
    a, b = salts.salt("acme"), salts.salt("globex")
    assert a != b and a == cache_salt(b"server-secret", "acme") != cache_salt(b"other-secret", "acme")
    assert len(a) <= 128 and not set("@/\\\0") & set(a)             # vLLM's validate_cache_salt rules
    prompt = list(range(64))
    cache = PrefixCache(16)
    cache.serve(prompt, salt=a)
    old = salts.rotate("acme")
    assert old == a and salts.salt("acme") != a and salts.retired["acme"] == [a]
    assert cache.lookup(prompt, salt=salts.salt("acme")) == 0          # unreachable at once...
    assert cache.count(a) == 3                                          # ...but resident until LRU or a reset


@pytest.mark.parametrize("layout,cached_after_turn_1", [
    ("none", lambda t: 2144 + 160 * (t - 2)),              # the history is an append-only prefix
    ("before_history", lambda t: 2000),                    # memory changes after the system prompt: history misses
    ("pinned", lambda t: 2544 + 160 * (t - 2)),            # a per-session profile is part of the stable prefix
    ("tail", lambda t: 2000 + 160 * (t - 2)),              # only the last exchange (and new memory) re-prefills
])
def test_layout_hits_hand_computed(layout, cached_after_turn_1):
    turns = hits_per_turn(layout)                           # 2,000 system, 400 memory, 40 user, 120 output, 8 turns
    assert turns[0].cached == 0
    assert [t.cached for t in turns[1:]] == [cached_after_turn_1(t) for t in range(2, 9)]
    extra = 0 if layout == "none" else 400
    assert [t.prompt for t in turns] == [2040 + extra + 160 * (t - 1) for t in range(1, 9)]


def test_layout_hit_rates():
    rates = {l: round(hit_rate(hits_per_turn(l)), 4) for l in ("none", "before_history", "pinned", "tail")}
    assert rates == {"none": 0.8831, "before_history": 0.5833, "pinned": 0.882, "tail": 0.7233}


def test_prefill_seconds_reproduces_step_cost():
    l4, h100, q, l8 = GPUS["L4"], GPUS["H100-SXM"], LLMS["qwen2.5-1.5b"], LLMS["llama-3.1-8b"]
    ms = lambda g, m, c: round(prefill_seconds(g, m, 2000, c) * 1e3, 1)
    assert (ms(l4, q, 0), ms(l4, q, 1800)) == (78.7, 15.1)
    assert (ms(l4, l8, 0), ms(l4, l8, 1800)) == (401.0, 65.6)
    assert (ms(h100, q, 0), ms(h100, q, 1800)) == (11.4, 3.2)
    assert (ms(h100, l8, 0), ms(h100, l8, 1800)) == (50.8, 7.7)
    assert round(compute_ttft(24, 2000, 1979), 4) == 0.0970


def test_call_and_turn_cost_hand_computed():
    assert round(call_cost(5000, 350, 2700), 6) == 0.007005        # (2300 x 1.50 + 2700 x 0.15 + 350 x 9.00) / 1e6
    assert round(call_cost(5000, 350, 2700, "gemini-3-flash"), 6) == 0.002335
    c = turn_cost(3000, 2000, 150, provider_minimum=False, extraction_in=600, extraction_out=60, extractions_per_turn=0.5)
    assert round(c["answer"], 6) == round((1000 * 1.5 + 2000 * 0.15 + 150 * 9) / 1e6, 6)
    assert round(c["extraction"], 6) == round(0.5 * (600 * 1.5 + 60 * 9) / 1e6, 6)


def test_the_provider_bills_cached_tokens_only_above_its_minimum():
    assert CACHE_MIN_TOKENS["gemini-3.5-flash"] == 4096
    assert billed_cached(4095, 4000) == 0 and billed_cached(4096, 4000) == 4000
    assert round(turn_cost(3000, 2000, 150)["answer"], 6) == round((3000 * 1.5 + 150 * 9) / 1e6, 6)   # below: full price
    assert round(turn_cost(5000, 2700, 350)["answer"], 6) == 0.007005                                # the scaling primer's call
    # below the minimum the layouts cost the same; above it the layout shows on the bill
    small = {l: sum(turn_cost(t.prompt, t.cached, 120)["total"] for t in hits_per_turn(l)) for l in ("before_history", "pinned")}
    big = {l: sum(turn_cost(t.prompt, t.cached, 120)["total"] for t in hits_per_turn(l, system_tokens=4000))
           for l in ("before_history", "pinned")}
    assert small["before_history"] == small["pinned"] and big["before_history"] > 1.4 * big["pinned"]


def test_budget_is_checked_before_the_work():
    b = Budget({"memory_tokens": 100, "writes": 2, "llm_calls": 3})
    b.charge("memory_tokens", 60)
    with pytest.raises(BudgetExceeded):
        b.charge("memory_tokens", 41)
    assert b.used["memory_tokens"] == 60 and b.left("memory_tokens") == 40
    b.reset()
    assert b.left("writes") == 2
