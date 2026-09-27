"""budget.py - where memory goes in the prompt decides what the prefix cache can reuse, and what a turn costs.

The one idea: an engine reuses KV only for an exact prefix, in full blocks (vLLM: 16 tokens, each block named
by a hash chained to its parent, the last token of a prompt always recomputed). Memory re-retrieved every turn
and placed BEFORE the history changes the prefix every turn, so every history block after it misses and the
conversation re-prefills on every turn. Pinning a profile once per session, or putting per-turn memory at the
tail (as ADK's PreloadMemoryTool does), keeps the history a stable prefix. The lost hits turn into prefill
time (a roofline step model, SIMULATED) and into dollars (a dated price table, verify).

Restated here, not imported (the core stays standalone; tests/test_repo_numbers.py checks the numbers match):
vllm-serving-lab notebook 04's `expected_cached_tokens`, minengine.kv's chained block names,
minengine.perf.step_cost for one prefill chunk, capacity.ttft_s, and the two price tables' call cost.
Tokens are 4-character chunks of the text, so len(token_ids(t)) == count_tokens(t) and a shared text prefix is
a shared token prefix (a real tokenizer may shift one token at the boundary).
"""
from __future__ import annotations

import hashlib
import itertools
import zlib
from dataclasses import dataclass, field

LAYOUTS = ("none", "before_history", "pinned", "tail")


def token_ids(text: str) -> list[int]:
    ids = [zlib.crc32(text[i:i + 4].encode()) for i in range(0, 4 * (len(text) // 4), 4)]
    return ids or ([zlib.crc32(text.encode())] if text else [])


def expected_cached_tokens(prev, new, block_size: int = 16) -> int:
    """vLLM's three rules: full blocks of the common prefix; `prev` left (len(prev) - 1) // B blocks;
    at least the last token of `new` is recomputed."""
    common = 0
    for x, y in zip(prev, new):
        if x != y:
            break
        common += 1
    blocks = min(common // block_size, (len(prev) - 1) // block_size, (len(new) - 1) // block_size)
    return max(blocks, 0) * block_size


def block_names(tokens, block_size: int = 16, salt: str | None = None) -> list[str]:
    """Chained names of the FULL blocks; a cache_salt enters the first block only, and the chain carries it."""
    names, parent = [], None
    for i in range(len(tokens) // block_size):
        extra = salt if i == 0 else None
        parent = hashlib.sha256(repr((parent, tuple(tokens[i * block_size:(i + 1) * block_size]), extra))
                                .encode()).hexdigest()
        names.append(parent)
    return names


class PrefixCache:
    """A block-hash prefix cache with vLLM's rules and no eviction (the pool is assumed big enough)."""

    def __init__(self, block_size: int = 16):
        self.block_size, self.names = block_size, {}      # name -> salt it was written under

    def lookup(self, prompt, salt: str | None = None) -> int:
        usable = (len(prompt) - 1) // self.block_size * self.block_size
        hits = 0
        for name in block_names(prompt[:usable], self.block_size, salt):
            if name not in self.names:
                break
            hits += self.block_size
        return hits

    def serve(self, prompt, output=(), salt: str | None = None) -> int:
        """Look the prompt up, then publish what this request computed: prompt + output minus the last token."""
        hit = self.lookup(prompt, salt)
        seq = list(prompt) + list(output)
        for name in block_names(seq[:len(seq) - 1], self.block_size, salt):
            self.names.setdefault(name, salt)
        return hit

    def count(self, salt: str) -> int:
        return sum(s == salt for s in self.names.values())

    def evict(self, salt: str) -> int:
        gone = [n for n, s in self.names.items() if s == salt]
        for n in gone:
            del self.names[n]
        return len(gone)


@dataclass
class Turn:
    prompt: int
    cached: int
    memory: int


def hits_per_turn(layout: str, *, system_tokens=2000, memory_tokens=400, user_tokens=40, output_tokens=120,
                  turns=8, block_size=16) -> list[Turn]:
    """One session through a fresh cache. Memory is retrieved anew each turn (different records each time)
    except in 'pinned', where one profile is retrieved at session start. Memory is never kept in history."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    counter = itertools.count(1)
    fresh = lambda n: [next(counter) for _ in range(n)]
    cache, system, profile, history, out = PrefixCache(block_size), fresh(system_tokens), fresh(memory_tokens), [], []
    for _ in range(turns):
        memory, user = fresh(memory_tokens), fresh(user_tokens)
        prompt = {"none": system + history + user, "before_history": system + memory + history + user,
                  "pinned": system + profile + history + user, "tail": system + history + memory + user}[layout]
        output = fresh(output_tokens)
        out.append(Turn(len(prompt), cache.serve(prompt, output), 0 if layout == "none" else memory_tokens))
        history += user + output
    return out


def hit_rate(turns: list[Turn]) -> float:
    return sum(t.cached for t in turns) / sum(t.prompt for t in turns)


# --- time: one prefill chunk on the roofline (minengine.perf.step_cost restated; SIMULATED) ----------------
@dataclass(frozen=True)
class GPU:
    name: str
    peak_flops: float        # dense 16-bit tensor FLOP/s (datasheet, verify)
    hbm_bw: float            # bytes/s


@dataclass(frozen=True)
class LLM:
    name: str
    params: float
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    embed_params: float
    tied: bool = True

    @property
    def matmul_params(self) -> float:
        return self.params - self.embed_params * (1 if self.tied else 2)

    @property
    def streamed_bytes(self) -> float:          # bf16 matmul weights + the LM head, read once per step
        return 2 * self.matmul_params + 2 * self.embed_params

    @property
    def kv_bytes_per_token(self) -> float:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * 2


GPUS = {"L4": GPU("L4", 121e12, 300e9), "H100-SXM": GPU("H100-SXM", 989e12, 3.35e12)}
LLMS = {"qwen2.5-1.5b": LLM("qwen2.5-1.5b", 1.54e9, 28, 12, 2, 128, 151936 * 1536),
        "llama-3.1-8b": LLM("llama-3.1-8b", 8.03e9, 32, 32, 8, 128, 128256 * 4096, tied=False)}


def prefill_seconds(gpu: GPU, llm: LLM, prompt_tokens: int, cached_tokens: int = 0,
                    flop_eff=0.6, bw_eff=0.8, overhead_s=0.002) -> float:
    """max(bytes / (BW x 0.8), FLOPs / (peak x 0.6)) + 2 ms for prefilling the uncached suffix after the cached
    prefix: 2 x matmul params per new token, the LM head once, attention over (new x all) pairs."""
    s, n = cached_tokens, prompt_tokens - cached_tokens
    flops = (2 * llm.matmul_params * n + 2 * llm.embed_params
             + 4 * llm.n_layers * llm.n_heads * llm.head_dim * (n * s + n * (n + 1) / 2))
    nbytes = llm.streamed_bytes + llm.kv_bytes_per_token * (s + n)
    return max(nbytes / (gpu.hbm_bw * bw_eff), flops / (gpu.peak_flops * flop_eff)) + overhead_s


def compute_ttft(active_b: float, prompt_tokens: int, tflops: float, mfu: float = 0.5) -> float:
    """The capacity primer's compute-only TTFT: 2 x params x tokens / (peak x MFU)."""
    return 2 * active_b * 1e9 * prompt_tokens / (tflops * 1e12 * mfu)


# --- money: dated price tables (USD per 1M tokens: input, cached input, output) ----------------------------
PRICES = {"gemini-3.5-flash": (1.50, 0.15, 9.00),   # scaling primer §3.4, prices of 5 Sep 2026 (verify)
          "gemini-3-flash": (0.50, 0.05, 3.00)}     # 07.2 agentlab.estimation, illustrative, Sep 2026 (verify)


def call_cost(input_tokens: int, output_tokens: int, cached_tokens: int = 0, model: str = "gemini-3.5-flash") -> float:
    inp, cached, out = PRICES[model]
    return ((input_tokens - cached_tokens) * inp + cached_tokens * cached + output_tokens * out) / 1e6


def turn_cost(prompt_tokens: int, cached_tokens: int, output_tokens: int, model: str = "gemini-3.5-flash", *,
              extraction_in: int = 0, extraction_out: int = 0, extractions_per_turn: float = 1.0) -> dict:
    """The answering call plus the memory-extraction call(s) amortised over the turn."""
    answer = call_cost(prompt_tokens, output_tokens, cached_tokens, model)
    extract = extractions_per_turn * call_cost(extraction_in, extraction_out, 0, model)
    return {"answer": answer, "extraction": extract, "total": answer + extract}


# --- budgets are code (durable primer §3.4) -------------------------------------------------------------
class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    limits: dict = field(default_factory=lambda: {"memory_tokens": 400, "writes": 5, "llm_calls": 4, "usd": 0.02})
    used: dict = field(default_factory=dict)

    def charge(self, what: str, amount: float = 1) -> None:
        """Checked before the work happens: raising here is the circuit breaker, not a log line."""
        new = self.used.get(what, 0) + amount
        if new > self.limits[what]:
            raise BudgetExceeded(f"{what}: {new:g} > {self.limits[what]:g} per turn")
        self.used[what] = new

    def left(self, what: str) -> float:
        return self.limits[what] - self.used.get(what, 0)

    def reset(self) -> None:
        self.used.clear()
