"""Paged KV: a block pool with refcounts and copy-on-write, and attention that reads through a block table.

The same algorithm as ../paged-attention/paged_attention_minimal.py (tests/test_paged.py checks the two
give the same outputs and block counts), packaged so it can be tested and reused: the pool is an object
instead of module globals, and the two things the practice notebook's exercises 4 and 5 turn on are
explicit switches you can flip to see them fail:

  paged_attention_blockwise(..., running_max=False)  resets the reference max per block instead of keeping
                                                      the running max: a later block scoring far below an
                                                      earlier one overflows alpha to inf and the output to nan
  PagedSequence.append(..., release_on_cow=False)     copies a shared block without releasing the shared
                                                      reference: the refcount no longer matches its owners and
                                                      the block never returns to the pool

Contents
  attention                        contiguous reference for one query vector
  BlockPool                        physical blocks, free list, refcounts, an invariant audit
  PagedSequence                    a block table, on-demand allocation, copy-on-write, fork, free
  gather, paged_attention_gather   rebuild contiguous K, V through the block table
  paged_attention_blockwise        one block at a time with the online softmax (what the kernel does)
  blocks_needed, allocation_waste  the fragmentation arithmetic of the primers

Standard library + numpy. Tier T0.
"""
from __future__ import annotations

import math

import numpy as np


def attention(q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """One decode step, one head: softmax(K q / sqrt(d)) V over a contiguous cache."""
    s = K @ q / math.sqrt(q.shape[-1])
    p = np.exp(s - s.max())
    return (p / p.sum()) @ V


class BlockPool:
    """One pre-allocated region of `num_blocks` blocks of `block_size` token slots (one head, `head_dim`).

    Sequences never own memory; they borrow blocks. `shuffle` hands blocks out in a scrambled order so
    block tables *look* scattered, as they are in a real engine after some churn.
    """

    def __init__(self, num_blocks: int = 64, block_size: int = 16, head_dim: int = 8,
                 shuffle: bool = True, seed: int = 0):
        self.num_blocks, self.block_size, self.head_dim = num_blocks, block_size, head_dim
        self.K = np.zeros((num_blocks, block_size, head_dim))
        self.V = np.zeros_like(self.K)
        self.free = list(range(num_blocks))
        if shuffle:
            np.random.default_rng(seed).shuffle(self.free)
        self.refcount: dict[int, int] = {}
        self.copies = 0                              # copy-on-write events, for the demos

    def allocate(self) -> int:
        if not self.free:
            raise MemoryError("KV pool exhausted: an engine would now preempt a sequence "
                              "(swap its blocks out, or free them and recompute later)")
        b = self.free.pop()
        self.refcount[b] = 1
        return b

    def share(self, b: int) -> None:
        self.refcount[b] += 1

    def release(self, b: int) -> None:
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            del self.refcount[b]
            self.free.append(b)

    def blocks_in_use(self) -> int:
        return self.num_blocks - len(self.free)

    def audit(self, sequences) -> list[str]:
        """Problems, if any: a refcount that differs from the number of block tables holding the block,
        or a block that is neither free nor held (leaked). Empty list = consistent."""
        owners: dict[int, int] = {}
        for s in sequences:
            for b in s.block_table:
                owners[b] = owners.get(b, 0) + 1
        problems = [f"block {b}: refcount {self.refcount.get(b)} but {n} owners"
                    for b, n in owners.items() if self.refcount.get(b) != n]
        problems += [f"block {b}: leaked (refcount {n}, no owner)" for b, n in self.refcount.items() if b not in owners]
        if len(self.free) + len(self.refcount) != self.num_blocks or len(set(self.free)) != len(self.free):
            problems.append("free list and refcounts do not partition the pool")
        return problems


class PagedSequence:
    """A sequence holds no tensors: a block table (logical block -> physical block) and a token count."""

    def __init__(self, pool: BlockPool):
        self.pool = pool
        self.block_table: list[int] = []
        self.num_tokens = 0
        self.log: list[tuple[np.ndarray, np.ndarray]] = []   # ground truth for tests only

    def append(self, k, v, release_on_cow: bool = True) -> None:
        """Store one token's K and V. A new block only when the last one is full; copy-on-write when the
        last block is shared (allocate, copy, release the shared reference, remap)."""
        pool, bs = self.pool, self.pool.block_size
        slot = self.num_tokens % bs
        if slot == 0:
            self.block_table.append(pool.allocate())
        else:
            phys = self.block_table[-1]
            if pool.refcount[phys] > 1:                 # shared: copy on write
                fresh = pool.allocate()
                pool.K[fresh], pool.V[fresh] = pool.K[phys], pool.V[phys]
                if release_on_cow:
                    pool.release(phys)
                self.block_table[-1] = fresh
                pool.copies += 1
        phys = self.block_table[-1]
        pool.K[phys, slot], pool.V[phys, slot] = k, v
        self.num_tokens += 1
        self.log.append((np.array(k, dtype=float), np.array(v, dtype=float)))

    def fork(self) -> "PagedSequence":
        """Parallel sampling: the child shares every block (refcount + 1); nothing is copied yet."""
        child = PagedSequence(self.pool)
        child.block_table, child.num_tokens, child.log = list(self.block_table), self.num_tokens, list(self.log)
        for b in child.block_table:
            self.pool.share(b)
        return child

    def free(self) -> None:
        for b in self.block_table:
            self.pool.release(b)
        self.block_table = []

    def contiguous(self) -> tuple[np.ndarray, np.ndarray]:
        """The K and V a contiguous cache would hold (from the test-only log)."""
        return np.stack([k for k, _ in self.log]), np.stack([v for _, v in self.log])


def gather(seq: PagedSequence) -> tuple[np.ndarray, np.ndarray]:
    """Follow the block table and trim the last, partly filled block."""
    t, pool = seq.num_tokens, seq.pool
    return (np.concatenate([pool.K[b] for b in seq.block_table])[:t],
            np.concatenate([pool.V[b] for b in seq.block_table])[:t])


def paged_attention_gather(q, seq: PagedSequence) -> np.ndarray:
    K, V = gather(seq)
    return attention(q, K, V)


def paged_attention_blockwise(q, seq: PagedSequence, running_max: bool = True, return_trace: bool = False):
    """Visit one physical block at a time and fold it into (m, l, acc); the gathered K, V never exist.

    Invariant after each block: acc / l == attention over the tokens seen so far. With
    running_max=False the reference is this block's max alone (the wrong answer exercise 4 guards against).
    """
    pool, d = seq.pool, q.shape[-1]
    m, l, acc = -math.inf, 0.0, np.zeros(pool.V.shape[-1])
    remaining, trace = seq.num_tokens, []
    for phys in seq.block_table:
        n = min(pool.block_size, remaining)
        remaining -= n
        s = pool.K[phys, :n] @ q / math.sqrt(d)
        m_new = max(m, float(s.max())) if running_max else float(s.max())
        alpha = float(np.exp(m - m_new)) if m != -math.inf else 0.0   # np.exp: inf, not OverflowError
        p = np.exp(s - m_new)
        l = alpha * l + float(p.sum())
        acc = alpha * acc + p @ pool.V[phys, :n]
        m = m_new
        trace.append({"block": phys, "tokens": n, "m": m, "alpha": alpha, "l": l})
    out = acc / l
    return (out, trace) if return_trace else out


def blocks_needed(tokens: int, block_size: int = 16) -> int:
    return -(-tokens // block_size)


def allocation_waste(lengths, block_size: int = 16, reserve: int | None = None) -> dict:
    """Slots held vs slots used for requests of the given final lengths, reserved contiguously up to
    `reserve` tokens each (pre-paging) or allocated in blocks on demand (paging)."""
    lengths = [int(x) for x in lengths]
    used = sum(lengths)
    paged = sum(blocks_needed(x, block_size) * block_size for x in lengths)
    out = {"used": used, "paged_slots": paged, "paged_waste": 1 - used / paged,
           "max_paged_waste_per_seq": block_size - 1}
    if reserve is not None:
        held = reserve * len(lengths)
        out.update(contiguous_slots=held, contiguous_waste=1 - used / held)
    return out
