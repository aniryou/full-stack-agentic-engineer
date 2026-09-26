"""The approximate prefix index: which replica has *probably* cached this prompt's prefix?

The one idea: the router cannot see inside the engines' KV caches, so it remembers what it
sent where. Every prompt is cut into fixed-size blocks and each block gets a *chained* hash
(block i's hash covers block i's content AND block i-1's hash), so equal hashes at position i
mean equal prompts up to and including block i. After routing a request to endpoint E, the
router records all of the prompt's block hashes against E ("E has probably cached these").
The next request walks its own block hashes from the start and counts, per endpoint, how
many leading blocks are recorded there. That count / the request's block count is the
prefix-cache score.

    prompt  |--block 0--|--block 1--|--block 2--|--blk 3|        (last block may be partial)
    hash    h0=H(b0,H(model))  h1=H(b1,h0)  h2=H(b2,h1)  h3=H(b3,h2)
    index   h0 -> {A, B}   h1 -> {A, B}   h2 -> {A}   (h3 unknown)
    match   A: 3 blocks, B: 2 blocks, C: 0 blocks      -> scores 3/4, 2/4, 0

It is *approximate* because (1) it is written at routing time, not when the engine actually
caches; (2) the engine evicts blocks the index still lists (bounded by an LRU per endpoint,
sized like the engine's KV capacity, and optionally a TTL); (3) the pseudo-token blocks do not
line up with the engine's real 16-token KV blocks. llm-d calls this the
`approx-prefix-cache-producer`; its precise sibling subscribes to the engines' KV events.

Faithful to llm-d-router (Sep 2026): chained 64-bit hashes seeded with the model name (and
cache salt), block size clamped to >= 64 tokens, per-endpoint LRU (default 31,250 entries),
tail-first insertion so the head of a prompt is the most recently used entry, and the greedy
match that stops at the first block no endpoint holds. Lab additions (documented, off by
default): an optional TTL, and LRU capacity converted to index-block units when auto-tuned.
"""
from __future__ import annotations

import hashlib
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass

MIN_BLOCK_SIZE_TOKENS = 64          # upstream floor (llm-d-router issue #1158)
DEFAULT_BLOCK_SIZE_TOKENS = 16      # upstream default before the floor is applied (vLLM's block size)
DEFAULT_LRU_CAPACITY_PER_SERVER = 31_250
DEFAULT_MAX_PREFIX_TOKENS = 131_072
DEFAULT_MAX_PREFIX_BLOCKS = 2_048

__all__ = ["block_hashes", "PrefixMatch", "PrefixIndex", "effective_block_size", "max_blocks_for",
           "MIN_BLOCK_SIZE_TOKENS", "DEFAULT_LRU_CAPACITY_PER_SERVER"]


def _h64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def block_hashes(tokens, block_size: int, model: str = "", cache_salt: str = "", max_blocks: int | None = None) -> list[int]:
    """Chained block hashes of a token sequence (the final block may be partial).

    hash_0 = H(content(b0) || H(model + salt)),  hash_i = H(content(b_i) || hash_{i-1}).
    Different models (or salts) never share hashes, even for identical prompts.
    """
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    prev = _h64((model + cache_salt).encode())
    out: list[int] = []
    for i in range(0, len(tokens), block_size):
        if max_blocks is not None and len(out) >= max_blocks:
            break
        block = tokens[i:i + block_size]
        content = _h64(struct.pack(f"<{len(block)}I", *block))
        prev = _h64(struct.pack("<QQ", content, prev))
        out.append(prev)
    return out


def effective_block_size(configured: int = DEFAULT_BLOCK_SIZE_TOKENS, engine_block_size: int | None = None,
                         auto_tune: bool = True, minimum: int = MIN_BLOCK_SIZE_TOKENS) -> int:
    """Index block size: the engine's (if auto-tuned and known) or the configured one, floored at 64."""
    bs = configured
    if auto_tune and engine_block_size:
        bs = engine_block_size
    return max(bs, minimum)


def max_blocks_for(block_size: int, max_prefix_tokens: int = DEFAULT_MAX_PREFIX_TOKENS,
                   max_prefix_blocks: int = DEFAULT_MAX_PREFIX_BLOCKS) -> int | None:
    """Cap on hashed blocks per request: tokens cap wins when > 0; both 0 = unlimited (None)."""
    if max_prefix_tokens > 0:
        return max_prefix_tokens // block_size
    if max_prefix_blocks == 0:
        return None
    return max_prefix_blocks


@dataclass(frozen=True)
class PrefixMatch:
    """What the producer writes onto each candidate endpoint for this request."""
    match_blocks: int
    total_blocks: int
    block_size: int

    @property
    def ratio(self) -> float:
        return self.match_blocks / self.total_blocks if self.total_blocks else 0.0

    @property
    def matched_tokens(self) -> int:
        return self.match_blocks * self.block_size


class PrefixIndex:
    """hash -> endpoints, plus one LRU of hashes per endpoint (bounded memory, recency order).

    `ttl_s` (lab addition): entries older than this are treated as evicted — a crude model of
    "the engine has probably dropped it by now". Because insertion is tail-first, recency
    decreases along every recorded chain, so LRU and TTL both evict a prompt's *deepest*
    blocks first and never punch holes in the middle of a cached prefix.
    """

    def __init__(self, capacity_per_endpoint: int = DEFAULT_LRU_CAPACITY_PER_SERVER,
                 ttl_s: float | None = None, clock=time.monotonic):
        if capacity_per_endpoint <= 0:
            raise ValueError("capacity_per_endpoint must be > 0")
        self.default_capacity = capacity_per_endpoint
        self.ttl_s = ttl_s
        self.clock = clock
        self._hash_to_eps: dict[int, set[str]] = {}
        self._lru: dict[str, OrderedDict] = {}      # endpoint -> OrderedDict[hash -> last-used time]
        self._cap: dict[str, int] = {}
        self.evictions = 0

    # -- writes ---------------------------------------------------------------------------
    def add(self, hashes, endpoint: str, capacity: int | None = None) -> None:
        """Record that `endpoint` (probably) caches every block in `hashes`."""
        lru = self._lru.get(endpoint)
        if lru is None:
            lru = self._lru[endpoint] = OrderedDict()
            self._cap[endpoint] = capacity if capacity and capacity > 0 else self.default_capacity
        cap = self._cap[endpoint]
        now = self.clock()
        for h in reversed(list(hashes)):             # tail first: the head ends up most recent
            if h in lru:
                lru.move_to_end(h)
            else:
                self._hash_to_eps.setdefault(h, set()).add(endpoint)
            lru[h] = now
            while len(lru) > cap:                     # an oversized prompt keeps exactly its head
                old, _ = lru.popitem(last=False)
                self._forget(old, endpoint)
                self.evictions += 1

    def _forget(self, h: int, endpoint: str) -> None:
        eps = self._hash_to_eps.get(h)
        if eps is not None:
            eps.discard(endpoint)
            if not eps:
                del self._hash_to_eps[h]

    def remove_endpoint(self, endpoint: str) -> None:
        for h in list(self._lru.pop(endpoint, {})):
            self._forget(h, endpoint)
        self._cap.pop(endpoint, None)

    # -- reads ----------------------------------------------------------------------------
    def get(self, h: int) -> set[str]:
        """Endpoints holding block hash `h` (expired TTL entries are purged lazily)."""
        eps = self._hash_to_eps.get(h)
        if not eps:
            return set()
        if self.ttl_s is None:
            return set(eps)
        now = self.clock()
        live = set()
        for e in list(eps):
            ts = self._lru[e].get(h)
            if ts is not None and now - ts > self.ttl_s:
                del self._lru[e][h]
                self._forget(h, e)
            else:
                live.add(e)
        return live

    def match(self, hashes) -> dict[str, int]:
        """Greedy scan from block 0: every endpoint holding block i gets +1; stop at the first
        block no endpoint holds. (With hole-free chains this equals each endpoint's longest
        cached prefix, in blocks.)"""
        res: dict[str, int] = {}
        for h in hashes:
            eps = self.get(h)
            if not eps:
                break
            for e in eps:
                res[e] = res.get(e, 0) + 1
        return res

    def longest_prefix(self, hashes, endpoint: str) -> int:
        """Per-endpoint contiguous match (reference semantics used by the tests)."""
        n = 0
        for h in hashes:
            if endpoint not in self.get(h):
                break
            n += 1
        return n

    def endpoints(self) -> list[str]:
        return sorted(self._lru)

    def block_counts(self) -> dict[str, int]:
        return {e: len(l) for e, l in sorted(self._lru.items())}

    def size(self) -> int:
        return sum(len(l) for l in self._lru.values())
