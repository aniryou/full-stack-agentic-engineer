"""kv.py - the KV cache manager: a pool of fixed-size blocks, reference counts, and a prefix cache.

The one idea: the scheduler never touches tensors - it owns *block ids*. A request's block table
maps its logical blocks (tokens 0-15, 16-31, ...) to physical blocks anywhere in the pool, so
memory is handed out one block at a time as sequences grow (PagedAttention).

Prefix caching rides on the same indirection. Once a block is FULL and its K/V computed, it is
named by   hash(parent block's name, the block's tokens, extra keys such as a LoRA id)
- a chain, so a name commits to every token before it, not just the 16 inside. A new request
walks its prompt's chain and adopts every block already in the cache (refcount + 1) instead of
recomputing it. A finished request's blocks go back to the free queue *with their names*: they
are still hits until the allocator reuses them, least-recently-freed first (LRU), and a request's
blocks are freed tail-first so the shared head of a prefix is evicted last. This mirrors vLLM V1's
KVCacheManager / BlockPool (vllm/v1/core/), minus multi-group and hybrid-model machinery. One
simplification: we publish a block after the step that computed it; vLLM publishes at scheduling
time (safe there because every layer writes the step's K/V before any request attends).
"""
from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict
from dataclasses import dataclass


def hash_block(parent: bytes | None, tokens, extra=None) -> bytes:
    """A full block's name: sha256 over (parent's name, this block's tokens, extra keys)."""
    return hashlib.sha256(repr((parent, tuple(int(t) for t in tokens), extra)).encode()).digest()


def block_hashes(tokens, block_size: int, extra=None, hash_fn=hash_block) -> list[bytes]:
    """The chain of names for every FULL block of `tokens` (a partial last block has no name)."""
    out, parent = [], None
    for i in range(len(tokens) // block_size):
        parent = hash_fn(parent, tokens[i * block_size:(i + 1) * block_size], extra)
        out.append(parent)
    return out


@dataclass
class Block:
    block_id: int
    ref_cnt: int = 0                  # how many block tables point here
    block_hash: bytes | None = None   # set once full and published to the prefix cache
    tokens: tuple = ()                # what it holds (for traces and tests; vLLM keeps only the hash)


@dataclass
class CacheStats:
    queries: int = 0                  # prompt tokens looked up (vllm:prefix_cache_queries)
    hits: int = 0                     # of those, served from cache  (vllm:prefix_cache_hits)
    evictions: int = 0                # cached blocks reused for new content

    @property
    def hit_rate(self) -> float:
        return self.hits / self.queries if self.queries else 0.0


class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int = 16, enable_prefix_caching: bool = True,
                 hash_fn=hash_block):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.enable_prefix_caching, self.hash_fn = enable_prefix_caching, hash_fn
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_queue = OrderedDict.fromkeys(range(num_blocks))   # front = next to evict (LRU)
        self.cached: dict[bytes, int] = {}                           # block name -> block id
        self.tables: dict[str, list[int]] = {}                       # request id -> block table
        self.hashes: dict[str, list[bytes]] = {}                     # request id -> names of its full blocks
        self.stats = CacheStats()

    # -- read-only views --------------------------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_queue)          # includes cached-but-unreferenced blocks: they are evictable

    @property
    def usage(self) -> float:
        """Fraction of blocks some request holds - what vllm:kv_cache_usage_perc reports."""
        return 1.0 - self.num_free_blocks / self.num_blocks

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    # -- prefix cache -----------------------------------------------------------------------------
    def lookup(self, tokens, extra=None) -> list[int]:
        """Longest cached prefix, in full blocks. Never the whole prompt: the last token must be
        computed to get logits, so at most (len - 1) // block_size blocks can hit."""
        if not self.enable_prefix_caching:
            return []
        hits, parent, B = [], None, self.block_size
        for i in range((len(tokens) - 1) // B):
            parent = self.hash_fn(parent, tokens[i * B:(i + 1) * B], extra)
            if parent not in self.cached:
                break
            hits.append(self.cached[parent])
        return hits

    # -- allocation -------------------------------------------------------------------------------
    def allocate_slots(self, rid: str, tokens, num_computed: int, num_new: int, hits=(),
                       reserve: int = 0, admit_whole_prompt: bool = False):
        """Grow `rid`'s block table to cover num_computed + num_new tokens (after adopting `hits`).
        Returns the newly allocated block ids, or None if the pool cannot supply them - the
        caller then preempts someone (running request) or waits (new request).
        `reserve` blocks must stay free afterwards (the admission watermark); with
        `admit_whole_prompt` a new request is only admitted if its WHOLE prompt fits, not just
        the first chunk (vLLM's scheduler_reserve_full_isl) - otherwise chunked prefill can admit
        more than memory can finish, and the excess is preempted a few steps later."""
        table = self.tables.get(rid, [])
        target = len(tokens) if admit_whole_prompt else num_computed + num_new
        need = max(0, self.blocks_needed(target) - len(table) - len(hits))
        revived = sum(1 for b in hits if self.blocks[b].ref_cnt == 0)   # cached blocks sitting in the free queue
        if need + revived > self.num_free_blocks - reserve:
            return None
        if rid not in self.tables:                                      # first allocation: count the lookup
            self.stats.queries += len(tokens)
            self.stats.hits += len(hits) * self.block_size
            self.hashes[rid] = [self.blocks[b].block_hash for b in hits]
        self.tables[rid] = table
        for b in hits:                                                  # adopt the hits
            if self.blocks[b].ref_cnt == 0:
                del self.free_queue[b]
            self.blocks[b].ref_cnt += 1
            table.append(b)
        need = max(0, self.blocks_needed(num_computed + num_new) - len(table))
        new = [self._pop_free() for _ in range(need)]
        table += new
        return new

    def _pop_free(self) -> int:
        bid, _ = self.free_queue.popitem(last=False)                    # least recently freed
        blk = self.blocks[bid]
        if blk.block_hash is not None:                                  # lazy eviction happens here
            if self.cached.get(blk.block_hash) == bid:
                del self.cached[blk.block_hash]
                self.stats.evictions += 1
            blk.block_hash = None
        blk.tokens, blk.ref_cnt = (), 1
        return bid

    def cache_blocks(self, rid: str, tokens, num_computed: int, extra=None):
        """After a step: name every block of `rid` whose tokens are now all computed, and publish
        it to the prefix cache unless an identical block is already there."""
        names, table, B = self.hashes.setdefault(rid, []), self.tables[rid], self.block_size
        for i in range(len(names), num_computed // B):
            toks = tuple(tokens[i * B:(i + 1) * B])
            name = self.hash_fn(names[-1] if names else None, toks, extra)
            names.append(name)
            blk = self.blocks[table[i]]
            blk.tokens = toks
            if self.enable_prefix_caching and name not in self.cached:  # a duplicate stays unpublished
                self.cached[name], blk.block_hash = blk.block_id, name

    def free(self, rid: str):
        """Drop `rid`'s references. Blocks that reach zero go to the BACK of the free queue, the
        request's last block first, so a shared head survives longer than a private tail."""
        for bid in reversed(self.tables.pop(rid, [])):
            blk = self.blocks[bid]
            blk.ref_cnt -= 1
            if blk.ref_cnt == 0:
                self.free_queue[bid] = None
        self.hashes.pop(rid, None)

    # -- invariants (tests call this after every step) --------------------------------------------
    def check(self):
        refs = Counter(b for t in self.tables.values() for b in t)
        for blk in self.blocks:
            assert blk.ref_cnt == refs.get(blk.block_id, 0), f"refcount drift on block {blk.block_id}"
            assert (blk.ref_cnt == 0) == (blk.block_id in self.free_queue), f"free-queue drift on {blk.block_id}"
        for name, bid in self.cached.items():
            assert self.blocks[bid].block_hash == name, "cache map points at a renamed block"
        for rid, t in self.tables.items():
            assert len(set(t)) == len(t), f"{rid} holds a block twice"

    def describe(self, rid: str, tokens=None) -> str:
        """One request's block table as `physical_id:'contents' x refcount`, `*` if published.
        Pass the request's tokens to see partial blocks too (only full blocks record their tokens)."""
        cells, B = [], self.block_size
        for i, bid in enumerate(self.tables.get(rid, [])):
            b = self.blocks[bid]
            toks = b.tokens or (tuple(tokens[i * B:(i + 1) * B]) if tokens is not None else ())
            txt = bytes(t for t in toks if t < 256).decode("utf-8", "replace") if toks else "?"
            cells.append(f"{bid}:{txt!r}x{b.ref_cnt}{'*' if b.block_hash else ''}")
        return "[" + " ".join(cells) + "]"
