# %% [markdown]
# # 01 · SIMT, warps and memory transactions
#
# **Tier:** T0 (CPU, simulated, numpy only). The lab's `01_kernels_in_the_simulator` runs the same
# kernels for real (Numba's CUDA simulator at T0, a real GPU at T1).
#
# ## The one-minute version
# A GPU runs your kernel as **warps of 32 threads that share one instruction stream**. Almost all
# performance rules for a memory-bound kernel are rules about what those 32 threads do *together*:
#
# * **Coalescing.** The memory system serves the 32 addresses of the warp in 32-byte *sectors*.
#   Consecutive 4-byte loads need 4 sectors, and the warp uses each byte that the memory system
#   fetches. A column walk down a row-major matrix needs 32 sectors, and the warp uses only 12.5% of
#   the bytes.
# * **Bank conflicts.** Shared memory has 32 banks, and each bank is 4 bytes wide. When lanes hit
#   *different* words in the *same* bank, the hardware serializes them. A word stride of $s$ gives a
#   $\gcd(s, 32)$-way conflict. Thus, when you pad a 32x32 tile to 33 columns, you repair the
#   transpose.
# * **Divergence.** When lanes of one warp take different branches, the warp runs each path, one
#   after the other.
#
# At the end, you can predict three values for each access pattern: its sectors per request, its
# bank-conflict degree and its SIMT efficiency. You can also explain the shared-memory
# transpose in a design review. Primer: §2 *The execution model* and §3 *Memory access patterns*
# (`../../PRIMER.md`).

# %%
import math

import numpy as np

from gpusim import simt
from gpusim.simt import bank_conflicts, coalescing, divergence, loop_divergence
from gpusim.simt import warp_addresses as W

# One warp reads a[offset + stride * lane] from a float32 array (4 bytes per element).
patterns = [("a[lane]", W(1)), ("a[lane + 1]", W(1, offset=1)), ("a[2 * lane]", W(2)),
            ("a[32 * lane]", W(32)), ("a[0] (all lanes)", W(0))]
print(f"{'pattern':18} sectors  moved  useful  efficiency")
for label, addr in patterns:
    c = coalescing(addr)
    print(f"{label:18} {c.sectors:>7} {c.moved_bytes:>6} {c.useful_bytes:>7}  {c.efficiency:>9.1%}")

# %% [markdown]
# Read the table as the memory system reads it:
#
# * `a[lane]`: 32 x 4 B = 128 contiguous bytes, aligned, thus **4 sectors**. This is the ideal.
# * `a[lane + 1]`: the same 128 bytes with an offset of 4. Thus they cross into a fifth sector, and
#   the warp uses 80% of the bytes that the memory system moves. Misalignment has a small cost.
# * `a[2 * lane]`: one float in two. Thus the warp needs 8 sectors, and it discards half of each
#   sector.
# * `a[32 * lane]`: each lane goes into its own sector, thus 32 sectors for 128 useful bytes. A read
#   of a *column* of a row-major float matrix with 32+ columns looks like this.
# * `a[0]`: one sector, a broadcast. The efficiency column shows 12.5% (4 useful bytes of 32 moved).
#   But one transaction is the best result that any access can get.
#
# Wider loads per thread (`float2`, `float4`) keep 100% efficiency with fewer instructions:

# %%
for elem in (4, 8, 16):
    c = coalescing(W(1, elem_bytes=elem), elem)
    print(f"{elem:>2}-byte loads: {c.sectors:>2} sectors for {c.useful_bytes} useful bytes ({c.efficiency:.0%})")

# %% [markdown]
# ## Case study: a matrix transpose
# `out[j][i] = in[i][j]` with one thread per element. A warp covers 32 consecutive `j` of one row
# `i`. The read is a coalesced access, and the write is a column walk:

# %%
N = 4096                                             # N x N float32 matrix
read, write = coalescing(W(1)), coalescing(W(N))
print(f"naive read : {read.sectors:>2} sectors per request")
print(f"naive write: {write.sectors:>2} sectors per request  -> writes move "
      f"{write.moved_bytes // read.moved_bytes}x the bytes they need")

# The fix: stage a 32x32 tile in shared memory. Global reads AND writes become row accesses
# (coalesced), and the "column" walk moves into shared memory, where it meets the banks:
col = lambda pitch: np.arange(32) * pitch * 4        # lane reads tile[lane][0]; row pitch in floats
print("tile[32][32] column read:", bank_conflicts(col(32)))
print("tile[32][33] column read:", bank_conflicts(col(33)))

# %% [markdown]
# With a pitch of 32 floats, each element of a column is in the same bank. Thus bank 0 holds 32
# different words, and the hardware does 32 serialized passes. When you pad each row to 33 floats,
# row $r$ moves by $r$ banks. Thus a column touches all 32 banks exactly one time. That one extra
# column per row is all that is necessary.

# %% [markdown]
# ## Exercise 1.1: count sectors
#
# Write `sectors(addresses, elem_bytes)`. It returns the number of different 32-byte sectors that
# the warp touches when each lane reads `elem_bytes` bytes that start at its address. An access can
# cross into two sectors. Thus count each byte that it touches (address `a` touches bytes
# `a .. a + elem_bytes - 1`).

# %% exercise
def sectors(addresses, elem_bytes=4):
    ### BEGIN SOLUTION
    touched = set()
    for a in np.asarray(addresses).ravel().tolist():
        touched.update(range(a // 32, (a + elem_bytes - 1) // 32 + 1))
    return len(touched)
    ### END SOLUTION

# %% check
for stride, offset, elem in [(1, 0, 4), (1, 1, 4), (2, 0, 4), (32, 0, 4), (3, 0, 4), (1, 0, 16), (1, 3, 8), (0, 0, 4)]:
    addr = W(stride, offset, elem)
    assert sectors(addr, elem) == coalescing(addr, elem).sectors, (stride, offset, elem)
assert sectors([30], 4) == 2, "a 4-byte access at byte 30 straddles two sectors"
print("✅ sectors() matches the model, including straddling accesses")

# %% [markdown]
# ## Exercise 1.2: predict before you compute
#
# Put the sectors per warp request in the blanks. Calculate them by hand. Only then, run the check.
#
# * `a[3 * lane]`, float32
# * `a[lane]`, float64
# * `a[lane // 2]`, float32 (pairs of lanes read the same float)
# * a warp that reads **column** 5 of a row-major `float32` matrix with 1024 columns

# %% exercise
predicted = {"a[3*lane] f32": None, "a[lane] f64": None, "a[lane//2] f32": None, "column of 1024-wide": None}
### BEGIN SOLUTION
predicted["a[3*lane] f32"] = 12         # span 3*31*4 + 4 = 376 bytes; stride 12 B < 32 B, so sectors 0..11
predicted["a[lane] f64"] = 8            # 256 contiguous bytes
predicted["a[lane//2] f32"] = 2         # 16 distinct floats = 64 bytes
predicted["column of 1024-wide"] = 32   # a 4096-byte stride puts every lane in its own sector
### END SOLUTION

# %% check
truth = {
    "a[3*lane] f32": coalescing(W(3)).sectors,
    "a[lane] f64": coalescing(W(1, elem_bytes=8), 8).sectors,
    "a[lane//2] f32": coalescing((np.arange(32) // 2) * 4).sectors,
    "column of 1024-wide": coalescing(W(1024, offset=5)).sectors,
}
for k, v in truth.items():
    assert predicted[k] == v, f"{k}: you said {predicted[k]}, the model says {v}"
print("✅ all four predictions right:", truth)

# %% [markdown]
# ## Exercise 1.3: the bank-conflict degree
#
# Write `conflict_degree(byte_addresses)` for one warp of **4-byte** shared-memory accesses. A word
# is `addr // 4`, and its bank is `word % 32`. The degree is the largest number of *different* words
# that go into one bank. When several lanes read the same word, that access is a broadcast. A
# broadcast counts one time.

# %% exercise
def conflict_degree(byte_addresses):
    ### BEGIN SOLUTION
    words_per_bank = {}
    for a in np.asarray(byte_addresses).ravel().tolist():
        words_per_bank.setdefault((a // 4) % 32, set()).add(a // 4)
    return max(len(w) for w in words_per_bank.values())
    ### END SOLUTION

# %% check
for s in range(0, 65):
    assert conflict_degree(W(s)) == bank_conflicts(W(s)).degree, s
    if s:
        assert conflict_degree(W(s)) == math.gcd(s, 32)
print("✅ stride s in words gives a gcd(s, 32)-way conflict; stride 0 is a broadcast")

# %% [markdown]
# ## Exercise 1.4: pick the padding
#
# A block reads a `float` tile by columns, `tile[lane][c]`, with a row pitch of `pitch` floats. Find
# the **smallest $\text{pitch} \ge 32$** that makes a column read conflict-free. Also find the
# smallest $\text{pitch} \ge 64$ for a 64-column tile. Use
# `bank_conflicts(np.arange(32) * pitch * 4)` for the search.

# %% exercise
### BEGIN SOLUTION
def smallest_conflict_free_pitch(at_least):
    pitch = at_least
    while bank_conflicts(np.arange(32) * pitch * 4).degree > 1:
        pitch += 1
    return pitch

pitch_32 = smallest_conflict_free_pitch(32)
pitch_64 = smallest_conflict_free_pitch(64)
### END SOLUTION

# %% check
assert (pitch_32, pitch_64) == (33, 65)
assert bank_conflicts(np.arange(32) * 34 * 4).degree == 2, "an even pad still conflicts 2-way"
print(f"✅ pitch {pitch_32} and {pitch_64}: any odd pitch works, because gcd(odd, 32) = 1")

# %% [markdown]
# ## Exercise 1.5: divergence in a loop
#
# A kernel gives each thread one sequence, and each thread loops over the tokens of its sequence. A
# warp continues to issue until its *longest* sequence is complete. During that time, the shorter
# lanes are idle.
#
# Write `simt_efficiency(lengths)`. It returns the useful lane-iterations (the sum
# of lengths) divided by $32 \times \sum_{\text{warps}} (\text{the warp's max length})$. Pad a
# partial last warp with zero-length lanes. Then compare unsorted lengths with lengths that you
# **sort** before you assign them to threads.

# %% exercise
def simt_efficiency(lengths):
    ### BEGIN SOLUTION
    L = np.asarray(lengths)
    L = np.concatenate([L, np.zeros((-L.size) % 32, L.dtype)]).reshape(-1, 32)
    return L.sum() / (32 * L.max(axis=1).sum())
    ### END SOLUTION

# %% check
rng = np.random.default_rng(0)
lengths = rng.geometric(1 / 200, size=4096)           # long-tailed, like real prompt lengths
unsorted, sorted_ = simt_efficiency(lengths), simt_efficiency(np.sort(lengths))
assert math.isclose(unsorted, loop_divergence(lengths).efficiency)
assert sorted_ > 2 * unsorted
print(f"✅ SIMT efficiency: {unsorted:.0%} unsorted -> {sorted_:.0%} sorted by length")

# %% [markdown]
# That is why kernels over ragged batches put the sequences into buckets by length, or sort them by
# length. It is also why attention kernels divide work by *tiles of tokens*, and not by one thread
# per sequence. A branch costs nothing extra when all 32 lanes agree. Warp-aligned branches are
# free:

# %%
tid = np.arange(256)
print("branch on tid % 2      :", f"{divergence(tid % 2, {0: 10, 1: 10}).efficiency:.0%} SIMT efficiency")
print("branch on (tid//32) % 2:", f"{divergence((tid // 32) % 2, {0: 10, 1: 10}).efficiency:.0%} SIMT efficiency")

# %% [markdown]
# ## In a design review
#
# **The two-minute version.** "The GPU schedules warps, not threads. Thus we think about 32 lanes at
# a time. The memory system serves global memory in 32-byte sectors. We lay out our loads so that
# consecutive lanes touch consecutive addresses: 4 sectors per request for 4-byte loads, with 100% of
# the bytes used.
#
# "Some algorithms walk columns by nature, as a transpose does. For these, we stage a tile through
# shared memory so that global traffic stays row-wise. We also pad the pitch of the tile to an odd
# number of words. Thus the column walk in shared memory hits 32 different banks.
#
# "Branches are warp-uniform, or they are low-cost. We sort ragged work so that the lanes of a warp
# finish together. The sectors-per-request and bank-conflict counters of Nsight Compute confirm each
# point."
#
# **Drill questions**
#
# 1. *The profiler shows 32 sectors per request on a load. What is incorrect, and what do you do?*
#    For 4-byte loads, consecutive lanes are 32 or more bytes apart. Thus each lane goes into its own
#    sector. The cause is a column walk or a large struct stride. Change the layout
#    (struct-of-arrays, or transpose the data one time). Or stage the data through a shared-memory
#    tile, so that lanes read contiguous bytes.
# 2. *Why a 33-float pitch and not 32?* With 32, each row starts in bank 0, thus a column is a
#    32-way conflict. With 33, row $r$ starts in bank $r \bmod 32$, thus a column covers all banks.
#    Any odd pitch works.
# 3. *Is `if (threadIdx.x < 16)` high-cost?* Only in the warps that cross the boundary. Warp 0 runs
#    both paths, and each other warp runs one path. Divergence is a cost per warp.
