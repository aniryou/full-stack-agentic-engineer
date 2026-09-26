"""PagedAttention from scratch — the core concept in ~200 lines of plain Python.

Mirrors Kwon et al., "Efficient Memory Management for LLM Serving with
PagedAttention" (SOSP 2023) — the paper behind vLLM. No GPU, no framework:
just NumPy and the memory-management idea.

  PART 1  Reference attention over a contiguous KV cache
  PART 2  A pool of fixed-size physical KV blocks + an allocator
          (the OS analogy: physical frames + frame allocator)
  PART 3  Sequences that own only a *block table*  (the page table)
  PART 4  Attention that reads K/V *through* the block table —
          first as a simple gather, then block-by-block with an
          online softmax (what the real kernel actually does)
  PART 5  Forking sequences: block sharing + copy-on-write
  PART 6  Demos: memory-waste arithmetic, correctness, sharing

Run:  python paged_attention_minimal.py     (only dependency: numpy)
"""

import numpy as np

D_HEAD = 8        # head dimension — tiny so everything prints nicely
BLOCK_SIZE = 4    # tokens per KV block   (vLLM's default is 16)
NUM_BLOCKS = 64   # physical blocks in the pool
RNG = np.random.default_rng(0)

# ============================================================================
# PART 1 — reference: ordinary attention over a CONTIGUOUS cache
# ============================================================================

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def naive_attention(q, K, V):
    """One decode step for one head. q: (d,), K,V: (t, d) contiguous."""
    scores = K @ q / np.sqrt(D_HEAD)          # (t,)
    return softmax(scores) @ V                # (d,)

# ============================================================================
# PART 2 — the physical pool. ONE big pre-allocated region, like vLLM's
# KV cache tensor. K_POOL[b, s] = the key vector stored in slot s of
# physical block b. Sequences never own memory — they borrow blocks.
# ============================================================================

K_POOL = np.zeros((NUM_BLOCKS, BLOCK_SIZE, D_HEAD))
V_POOL = np.zeros_like(K_POOL)


class BlockManager:
    """Hands out physical blocks; tracks reference counts for sharing."""

    def __init__(self, num_blocks=NUM_BLOCKS):
        if num_blocks > K_POOL.shape[0]:
            raise ValueError(f"num_blocks={num_blocks} exceeds the {K_POOL.shape[0]}-block pool tensor")
        self.num_blocks = num_blocks    # this manager's pool size (may be < NUM_BLOCKS)
        self.free = list(range(num_blocks))
        RNG.shuffle(self.free)          # so allocations *look* scattered
        self.refcount = {}

    def allocate(self):
        if not self.free:
            raise MemoryError(
                "KV pool exhausted — a real engine would now PREEMPT a "
                "victim sequence (swap its blocks to CPU, or free them "
                "and recompute later).")
        b = self.free.pop()
        self.refcount[b] = 1
        return b

    def share(self, b):
        self.refcount[b] += 1

    def release(self, b):
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            del self.refcount[b]
            self.free.append(b)

    def blocks_in_use(self):
        return self.num_blocks - len(self.free)

# ============================================================================
# PART 3 — a sequence holds NO tensors. Only a block table: a list mapping
# logical block i -> physical block id. This is the page table.
# ============================================================================

class Sequence:
    def __init__(self, mgr):
        self.mgr = mgr
        self.block_table = []     # logical block index -> physical block id
        self.num_tokens = 0
        self.kv_log = []          # ground truth for testing ONLY — a real
                                  # engine stores nothing outside the pool

    def append_kv(self, k, v):
        """Store one new token's K/V. Allocate a block only when needed."""
        slot = self.num_tokens % BLOCK_SIZE
        if slot == 0:
            # current block is full (or this is the first token) -> new block
            self.block_table.append(self.mgr.allocate())
        else:
            phys = self.block_table[-1]
            if self.mgr.refcount[phys] > 1:        # block is SHARED (Part 5)
                fresh = self.mgr.allocate()        # copy-on-write:
                K_POOL[fresh] = K_POOL[phys]       #   make a private copy,
                V_POOL[fresh] = V_POOL[phys]
                self.mgr.release(phys)             #   drop the shared ref,
                self.block_table[-1] = fresh       #   remap the page table.
        phys = self.block_table[-1]
        K_POOL[phys, slot] = k
        V_POOL[phys, slot] = v
        self.num_tokens += 1
        self.kv_log.append((np.array(k), np.array(v)))

    def free(self):
        for b in self.block_table:
            self.mgr.release(b)
        self.block_table = []

# ============================================================================
# PART 4 — attention THROUGH the block table
# ============================================================================

def paged_attention_gather(q, seq):
    """Simplest correct version: gather K/V into contiguous arrays via the
    block table, then run ordinary attention. Proves the indirection does
    not change the math at all."""
    t = seq.num_tokens
    K = np.concatenate([K_POOL[b] for b in seq.block_table])[:t]
    V = np.concatenate([V_POOL[b] for b in seq.block_table])[:t]
    return naive_attention(q, K, V)


def paged_attention_blockwise(q, seq):
    """What the real kernel does: visit ONE physical block at a time and fold
    it into a running (max, denominator, weighted-sum) accumulator — the
    'online softmax' trick, same as FlashAttention's tiling. The gathered
    K/V is never materialized.

    Invariant after each block:  acc / l == attention over tokens seen so far.
    """
    m = float("-inf")            # running max of scores (numerical stability)
    l = 0.0                      # running softmax denominator
    acc = np.zeros(D_HEAD)       # running numerator (weighted sum of values)
    remaining = seq.num_tokens
    for phys in seq.block_table:
        n = min(BLOCK_SIZE, remaining)         # last block may be partial
        remaining -= n
        Kb, Vb = K_POOL[phys, :n], V_POOL[phys, :n]
        s = Kb @ q / np.sqrt(D_HEAD)           # scores for this block, (n,)
        m_new = max(m, s.max())
        alpha = np.exp(m - m_new)              # rescale old accumulator
        p = np.exp(s - m_new)                  # this block's unnormalized probs
        l = l * alpha + p.sum()
        acc = acc * alpha + p @ Vb
        m = m_new
    return acc / l

# ============================================================================
# PART 5 — forking (parallel sampling): share every block, copy only on write
# ============================================================================

def fork(seq):
    """n samples from one prompt share ALL prompt blocks. Nothing is copied
    until a sequence writes into a shared block (see append_kv)."""
    child = Sequence(seq.mgr)
    child.block_table = list(seq.block_table)   # same physical ids
    child.num_tokens = seq.num_tokens
    child.kv_log = list(seq.kv_log)
    for b in child.block_table:
        seq.mgr.share(b)
    return child

# ============================================================================
# PART 6 — demos
# ============================================================================

def reference(q, seq):
    """Ground truth: rebuild contiguous K/V from the test-only log."""
    K = np.stack([k for k, _ in seq.kv_log])
    V = np.stack([v for _, v in seq.kv_log])
    return naive_attention(q, K, V)


def demo_memory_waste():
    print("=" * 68)
    print("DEMO 1 — why bother: pre-vLLM systems reserve max_len per request")
    print("=" * 68)
    max_len, n_req = 512, 16
    actual = RNG.integers(16, 257, size=n_req)      # real generated lengths
    used = actual.sum()
    contiguous = n_req * max_len                    # reserve worst case
    paged = sum(int(np.ceil(a / BLOCK_SIZE)) * BLOCK_SIZE for a in actual)
    print(f"  {n_req} requests, reserve {max_len} slots each, "
          f"actually generate {actual.min()}-{actual.max()} tokens")
    print(f"  contiguous allocation: {used}/{contiguous} slots used "
          f"= {100*used/contiguous:.0f}% utilization")
    print(f"  paged allocation:      {used}/{paged} slots used "
          f"= {100*used/paged:.0f}% utilization")
    print(f"  (paper measured 20-40% vs >96% — same effect)\n")


def demo_correctness(mgr):
    print("=" * 68)
    print("DEMO 2 — the block table changes WHERE bytes live, not the math")
    print("=" * 68)
    seq = Sequence(mgr)
    for _ in range(13):                             # 13 tokens -> 4 blocks,
        seq.append_kv(RNG.normal(size=D_HEAD),      # last one only 1/4 full
                      RNG.normal(size=D_HEAD))
    q = RNG.normal(size=D_HEAD)
    print(f"  13 tokens, BLOCK_SIZE={BLOCK_SIZE} -> block table "
          f"(logical -> physical): "
          f"{dict(enumerate(seq.block_table))}   <- scattered!")
    out_ref = reference(q, seq)
    out_gather = paged_attention_gather(q, seq)
    out_kernel = paged_attention_blockwise(q, seq)
    assert np.allclose(out_ref, out_gather)
    assert np.allclose(out_ref, out_kernel)
    print("  contiguous == paged-gather == paged-blockwise  "
          f"(max diff {np.abs(out_ref - out_kernel).max():.2e})\n")
    seq.free()


def demo_sharing(mgr):
    print("=" * 68)
    print("DEMO 3 — parallel sampling: share the prompt, copy on write")
    print("=" * 68)
    parent = Sequence(mgr)
    for _ in range(6):                              # 6-token prompt: one full
        parent.append_kv(RNG.normal(size=D_HEAD),   # block + one half-full
                         RNG.normal(size=D_HEAD))
    samples = [parent, fork(parent), fork(parent)]  # 3 samples, 1 prompt
    print(f"  after fork x2: refcounts on prompt blocks = "
          f"{[parent.mgr.refcount[b] for b in parent.block_table]}")
    for s in samples:                               # each sample generates its
        for _ in range(3):                          # own 3 tokens -> 9 total
            s.append_kv(RNG.normal(size=D_HEAD), RNG.normal(size=D_HEAD))
    q = RNG.normal(size=D_HEAD)
    for s in samples:                               # every divergent history
        assert np.allclose(reference(q, s), paged_attention_blockwise(q, s))
    unique = {b for s in samples for b in s.block_table}
    naive = sum(len(s.block_table) for s in samples)
    print(f"  block tables now: {[s.block_table for s in samples]}")
    print(f"  physical blocks used: {len(unique)} "
          f"(block 0 still shared) vs {naive} without sharing")
    print("  all 3 divergent sequences still match contiguous attention\n")
    for s in samples:
        s.free()
    print(f"  after freeing all sequences: {mgr.blocks_in_use()} "
          f"blocks in use — pool fully recovered\n")


def demo_exhaustion():
    print("=" * 68)
    print("DEMO 4 — what happens when the pool runs dry")
    print("=" * 68)
    tiny = BlockManager(num_blocks=2)
    tiny.allocate(); tiny.allocate()
    try:
        tiny.allocate()
    except MemoryError as e:
        print(f"  MemoryError: {e}\n")


if __name__ == "__main__":
    mgr = BlockManager()
    demo_memory_waste()
    demo_correctness(mgr)
    demo_sharing(mgr)
    demo_exhaustion()
    print("Core concept in one line: a per-sequence block table turns the KV")
    print("cache into paged virtual memory — attention just follows pointers.")
