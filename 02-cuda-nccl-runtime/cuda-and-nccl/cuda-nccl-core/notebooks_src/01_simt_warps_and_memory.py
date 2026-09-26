# %% [markdown]
# # 01 · SIMT, warps and memory transactions
#
# **Tier:** T0 (CPU, simulated; numpy only). The same kernels, run for real, are in the lab's
# `01_kernels_in_the_simulator` (Numba's CUDA simulator at T0, a real GPU at T1).
#
# ## The one-minute version
# A GPU runs your kernel as **warps of 32 threads that share one instruction stream**. Nearly
# every performance rule for a memory-bound kernel is a rule about what those 32 threads do
# *together*:
#
# * **Coalescing.** The warp's 32 addresses are served in 32-byte *sectors*. Consecutive 4-byte
#   loads need 4 sectors, and every byte fetched gets used. A column walk down a row-major
#   matrix needs 32 sectors, and only 12.5% of the bytes get used.
# * **Bank conflicts.** Shared memory has 32 banks, each 4 bytes wide. Lanes that hit
#   *different* words in the *same* bank are serialized. A word stride of s gives a
#   gcd(s, 32)-way conflict, so padding a 32x32 tile to 33 columns fixes the transpose.
# * **Divergence.** When lanes of one warp take different branches, the warp runs every path,
#   one after another.
#
# By the end you can predict, for any access pattern, its sectors per request, its bank-conflict
# degree and its SIMT efficiency. You can also explain the shared-memory transpose in a design
# review. Primer: §2 *The execution model* and §3 *Memory access patterns* (`../../PRIMER.md`).

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
# Read the table as the memory system does:
#
# * `a[lane]`: 32 x 4 B = 128 contiguous bytes, aligned, so **4 sectors**. This is the ideal.
# * `a[lane + 1]`: the same 128 bytes shifted by 4, so they straddle a fifth sector and 80% of
#   the bytes moved get used. Misalignment costs a little.
# * `a[2 * lane]`: every other float, so 8 sectors and half of every sector is thrown away.
# * `a[32 * lane]`: each lane lands in its own sector, so 32 sectors for 128 useful bytes.
#   This is what reading a *column* of a row-major float matrix with 32+ columns looks like.
# * `a[0]`: one sector, a broadcast. The efficiency column says 12.5% (4 useful bytes of 32
#   moved), but one transaction is the best any access can do.
#
# Wider per-thread loads (`float2`, `float4`) keep 100% efficiency with fewer instructions:

# %%
for elem in (4, 8, 16):
    c = coalescing(W(1, elem_bytes=elem), elem)
    print(f"{elem:>2}-byte loads: {c.sectors:>2} sectors for {c.useful_bytes} useful bytes ({c.efficiency:.0%})")

# %% [markdown]
# ## Case study: a matrix transpose
# `out[j][i] = in[i][j]` with one thread per element and a warp covering 32 consecutive `j`
# of one row `i`. The read is coalesced and the write is a column walk:

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
# With a pitch of 32 floats every element of a column sits in the same bank, so there are 32
# distinct words in bank 0 and 32 serialized passes. Padding each row to 33 floats shifts row
# `r` by `r` banks, so a column touches all 32 banks exactly once. That one extra column per row
# is the whole trick.

# %% [markdown]
# ## Exercise 1.1: count sectors
#
# Write `sectors(addresses, elem_bytes)`: the number of distinct 32-byte sectors touched when
# each lane reads `elem_bytes` bytes starting at its address. An access may straddle two
# sectors, so count every byte it touches (address `a` touches bytes `a .. a + elem_bytes - 1`).

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
# Fill in the sectors per warp request. Work them out by hand, and only then run the check.
#
# * `a[3 * lane]`, float32
# * `a[lane]`, float64
# * `a[lane // 2]`, float32 (pairs of lanes read the same float)
# * a warp reading **column** 5 of a row-major `float32` matrix with 1024 columns

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
# Write `conflict_degree(byte_addresses)` for one warp of **4-byte** shared-memory accesses. A
# word is `addr // 4` and its bank is `word % 32`. The degree is the largest number of
# *distinct* words that land in one bank. Several lanes reading the same word is a broadcast and
# counts once.

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
# A block reads a `float` tile column-wise, `tile[lane][c]`, with a row pitch of `pitch`
# floats. Find the **smallest pitch >= 32** that makes a column read conflict-free, and the
# smallest pitch >= 64 for a 64-column tile. Use `bank_conflicts(np.arange(32) * pitch * 4)`
# to search.

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
# A kernel gives each thread one sequence and loops over its tokens. A warp keeps issuing until
# its *longest* sequence is done, while shorter lanes idle. Write `simt_efficiency(lengths)`:
# the useful lane-iterations (the sum of lengths) divided by `32 x sum over warps of the warp's
# max length`. Pad a partial last warp with zero-length lanes. Then compare unsorted lengths with
# lengths **sorted** before they are assigned to threads.

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
# That is why kernels over ragged batches bucket or sort by length, and why attention kernels
# split work by *tiles of tokens* rather than one thread per sequence. A branch costs nothing
# extra when all 32 lanes agree; warp-aligned branches are free:

# %%
tid = np.arange(256)
print("branch on tid % 2      :", f"{divergence(tid % 2, {0: 10, 1: 10}).efficiency:.0%} SIMT efficiency")
print("branch on (tid//32) % 2:", f"{divergence((tid // 32) % 2, {0: 10, 1: 10}).efficiency:.0%} SIMT efficiency")

# %% [markdown]
# ## In a design review
#
# **The two-minute version.** "The GPU schedules warps, not threads, so we reason about 32 lanes
# at once. Global memory is served in 32-byte sectors. Our loads are laid out so that
# consecutive lanes touch consecutive addresses: 4 sectors per request for 4-byte loads, 100% of
# the bytes used. Where an algorithm naturally walks columns, as in a transpose, we stage a tile
# through shared memory so that global traffic stays row-wise, and we pad the tile's pitch to an
# odd number of words so the column walk in shared memory hits 32 different banks. Branches
# are either warp-uniform or cheap, and ragged work is sorted so a warp's lanes finish together.
# Nsight Compute's sectors-per-request and bank-conflict counters confirm each point."
#
# **Drill questions**
#
# 1. *The profiler shows 32 sectors per request on a load. What is wrong and what do you do?*
#    Consecutive lanes are 128 or more bytes apart, a column-style access. Change the layout
#    (struct-of-arrays, transpose the data once) or stage through a shared-memory tile so that
#    lanes read contiguous bytes.
# 2. *Why a 33-float pitch and not 32?* With 32, every row starts in bank 0, so a column is a
#    32-way conflict. With 33, row `r` starts in bank `r % 32`, so a column spans all banks.
#    Any odd pitch works.
# 3. *Is `if (threadIdx.x < 16)` expensive?* Only inside warps that straddle the boundary: warp
#    0 runs both paths and every other warp runs one. Divergence is a per-warp cost.
