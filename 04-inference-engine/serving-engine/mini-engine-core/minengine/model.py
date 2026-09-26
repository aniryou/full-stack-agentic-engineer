"""model.py - a tiny Llama-style decoder that reads its KV cache through block tables.

The one idea: an engine never hands the model "a sequence". Each step it hands over a FLAT
batch of new tokens from many requests (a prompt chunk here, one decode token there), each
token's position, a slot mapping (where to WRITE the token's K/V) and one block table per
request (where to READ its history). The maths is ordinary attention; only the addressing is
paged. `forward_dense` is the textbook reference with no cache at all - the tests prove that
the paged path computes the same logits, whatever the chunking, sharing or block layout.

Architecture: byte-level vocabulary (256 bytes + EOS), RMSNorm, RoPE, grouped-query attention
(4 query heads share 2 KV heads), SwiGLU MLP, float64 so paged == dense to ~1e-12. The weights
are random except the embedding/unembedding, which factorise a character-bigram table of a short
English paragraph - so the model babbles English-looking text. The engine does not care what the
model says; that is the point.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

EOS = 256           # byte ids are 0..255; one extra id ends a sequence
VOCAB = 257

CORPUS = (
    "The engine runs a loop. Each step it picks the requests to run, packs their new tokens "
    "into one batch, runs the model once, and samples a token for every request that reached "
    "the end of its prompt. A request that finishes leaves the batch at once, and a waiting "
    "request takes its place on the next step. The keys and values of every token stay in a "
    "cache of small blocks, and a table for each request says where its blocks are. When two "
    "prompts start with the same text, they share the same blocks, which saves memory and time. "
    "When memory runs out, the engine stops one request, frees its blocks, and runs it again "
    "later. The rest is about the choices behind each step: how many tokens to put in a batch, "
    "when to split a long prompt, which blocks to keep, how to pick the next token, and how to "
    "guess several tokens at once."
)


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids) -> str:
    return bytes(int(t) for t in ids if t < 256).decode("utf-8", errors="replace")


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2          # GQA: each KV head serves n_heads // n_kv_heads query heads
    d_ff: int = 128
    rope_base: float = 10000.0
    mix: float = 0.5             # size of the attention/MLP updates relative to the embedding

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def kv_bytes_per_token(self, bytes_per_value: int = 2) -> int:
        """2 (K and V) x layers x kv_heads x head_dim x bytes - the number that sizes every KV cache."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * bytes_per_value


SMALL = ModelConfig(d_model=16, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=32)   # a draft-sized model


def bigram_logits(text: str = CORPUS) -> np.ndarray:
    """log P(next | current) over the byte vocabulary, smoothed towards the unigram distribution."""
    ids = encode(text) + [EOS]
    counts = np.zeros((VOCAB, VOCAB))
    np.add.at(counts, (ids[:-1], ids[1:]), 1.0)
    uni = np.bincount(ids, minlength=VOCAB) + 0.01
    uni /= uni.sum()
    probs = (counts + 0.5 * uni) / (counts.sum(1, keepdims=True) + 0.5)
    return np.log(probs)


def rms_norm(x):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)


def rope(x, pos, base):
    """Rotate each (first-half, second-half) pair of every head by an angle proportional to position."""
    half = x.shape[-1] // 2
    ang = np.asarray(pos, float)[:, None] * base ** (-np.arange(half) / half)
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


@dataclass
class Batch:
    """One engine step's input: the vLLM 'attention metadata', minus the GPU."""
    token_ids: np.ndarray        # (N,) every scheduled token of every request, concatenated
    positions: np.ndarray        # (N,) each token's absolute position in its own sequence
    slot_mapping: np.ndarray     # (N,) where its K/V is written: block_id * block_size + offset
    query_start: list            # request r owns tokens [query_start[r], query_start[r+1])
    block_tables: list           # request r's physical block ids, in logical order
    context_lens: list           # tokens request r can see after this step (computed + new)


class PagedKVCache:
    """The 'GPU memory': K and V for num_blocks * block_size token slots, per layer.

    kv_dtype="fp8" rounds every value to FP8-E4M3 on write (per-tensor scale), like
    vLLM's --kv-cache-dtype fp8: half the bytes of bf16, a small error in attention."""

    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int, kv_dtype: str = "float64",
                 kv_scale: float = 1.0):
        shape = (cfg.n_layers, num_blocks * block_size, cfg.n_kv_heads, cfg.head_dim)
        self.k, self.v = np.zeros(shape), np.zeros(shape)
        self.block_size, self.kv_dtype, self.kv_scale = block_size, kv_dtype, kv_scale

    def write(self, layer, slots, k, v):
        if self.kv_dtype == "fp8":
            from .quant import fp8_e4m3
            k, v = fp8_e4m3(k / self.kv_scale) * self.kv_scale, fp8_e4m3(v / self.kv_scale) * self.kv_scale
        self.k[layer, slots], self.v[layer, slots] = k, v

    def read(self, layer, block_table, n):
        """K and V of the first n tokens of a sequence, gathered through its block table."""
        slots = (np.asarray(block_table)[:, None] * self.block_size + np.arange(self.block_size)).ravel()[:n]
        return self.k[layer, slots], self.v[layer, slots]


def paged_attention(q, k, v, q_pos):
    """q: (n, H, D) new queries; k, v: (t, Hkv, D) the whole visible history; causal by position."""
    n, H, D = q.shape
    hkv = k.shape[1]
    qg = q.reshape(n, hkv, H // hkv, D)                      # group the query heads that share a KV head
    s = np.einsum("nkgd,tkd->nkgt", qg, k) / np.sqrt(D)
    visible = q_pos[:, None] >= np.arange(k.shape[0])[None, :]
    s = np.where(visible[:, None, None, :], s, -np.inf)
    return np.einsum("nkgt,tkd->nkgd", softmax(s), v).reshape(n, H, D)


class TinyLM:
    """A deterministic toy LLM. `forward` is what the engine calls; `forward_dense` is the reference."""

    def __init__(self, cfg: ModelConfig = ModelConfig(), seed: int = 0):
        self.cfg = c = cfg
        rng = np.random.default_rng(seed)
        big = bigram_logits()
        u, s, vt = np.linalg.svd(big - big.mean(1, keepdims=True))
        self.emb = u[:, :c.d_model] * np.sqrt(s[:c.d_model])             # (VOCAB, d)
        self.unembed = np.sqrt(s[:c.d_model])[:, None] * vt[:c.d_model]   # (d, VOCAB): emb @ unembed ~ bigram
        out = c.mix * np.sqrt((self.emb ** 2).mean())                     # target size of each residual update
        W = lambda i, o, g=1.0: rng.standard_normal((i, o)) * g / np.sqrt(i)
        H, Hkv, D = c.n_heads, c.n_kv_heads, c.head_dim
        self.layers = [dict(wq=W(c.d_model, H * D), wk=W(c.d_model, Hkv * D), wv=W(c.d_model, Hkv * D),
                            wo=W(H * D, c.d_model, out), w_gate=W(c.d_model, c.d_ff), w_up=W(c.d_model, c.d_ff),
                            w_down=W(c.d_ff, c.d_model, out)) for _ in range(c.n_layers)]

    # -- shared pieces ---------------------------------------------------------------------------
    def _qkv(self, L, x, pos):
        c, h = self.cfg, rms_norm(x)
        q = rope((h @ L["wq"]).reshape(len(x), c.n_heads, c.head_dim), pos, c.rope_base)
        k = rope((h @ L["wk"]).reshape(len(x), c.n_kv_heads, c.head_dim), pos, c.rope_base)
        return q, k, (h @ L["wv"]).reshape(len(x), c.n_kv_heads, c.head_dim)

    @staticmethod
    def _mlp(L, x):
        h = rms_norm(x)
        g = h @ L["w_gate"]
        return (g / (1 + np.exp(-g)) * (h @ L["w_up"])) @ L["w_down"]    # SwiGLU

    # -- the engine's path: flat batch in, last-token logits per request out ----------------------
    def forward(self, batch: Batch, cache: PagedKVCache) -> np.ndarray:
        x = self.emb[batch.token_ids]
        qs = batch.query_start
        for li, L in enumerate(self.layers):
            q, k, v = self._qkv(L, x, batch.positions)
            cache.write(li, batch.slot_mapping, k, v)        # 1) every new token's K/V lands in its slot...
            attn = np.empty_like(q)
            for r in range(len(batch.block_tables)):         # 2) ...then each request reads its history
                s, e = qs[r], qs[r + 1]
                K, V = cache.read(li, batch.block_tables[r], batch.context_lens[r])
                attn[s:e] = paged_attention(q[s:e], K, V, batch.positions[s:e])
            x = x + attn.reshape(len(x), -1) @ L["wo"]
            x = x + self._mlp(L, x)
        last = np.asarray(qs[1:]) - 1                         # logits only where a token may be sampled
        return x[last] @ self.unembed

    # -- the reference: whole sequence, contiguous K/V, no cache ---------------------------------
    def forward_dense(self, token_ids) -> np.ndarray:
        """(T, VOCAB) logits for every position - textbook causal attention, written independently."""
        ids = np.asarray(token_ids)
        T, G = len(ids), self.cfg.n_heads // self.cfg.n_kv_heads
        pos = np.arange(T)
        x = self.emb[ids]
        for L in self.layers:
            q, k, v = self._qkv(L, x, pos)
            k, v = np.repeat(k, G, axis=1), np.repeat(v, G, axis=1)      # expand KV heads to query heads
            s = np.einsum("thd,uhd->htu", q, k) / np.sqrt(self.cfg.head_dim)
            s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
            attn = np.einsum("htu,uhd->thd", softmax(s), v)
            x = x + attn.reshape(T, -1) @ L["wo"]
            x = x + self._mlp(L, x)
        return x @ self.unembed

    def generate_dense(self, prompt_ids, max_new_tokens: int) -> list[int]:
        """Greedy decoding by recomputing everything each step. Slow and simple: the ground truth."""
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            ids.append(int(np.argmax(self.forward_dense(ids)[-1])))
        return ids[len(prompt_ids):]

    # -- variants used by the speculative-decoding and quantization notebooks ---------------------
    def quantized(self, **kw) -> "TinyLM":
        """A copy whose linear weights went through quant.fake_quant(**kw) (embeddings kept, as usual)."""
        from .quant import fake_quant
        m = copy.deepcopy(self)
        for L in m.layers:
            for name in L:
                L[name] = fake_quant(L[name], **kw)
        return m

    def num_params(self) -> int:
        return self.emb.size + self.unembed.size + sum(w.size for L in self.layers for w in L.values())
