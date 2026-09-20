"""Locality-Sensitive Hashing index for minifaiss.

Mirrors ``faiss.IndexLSH``. Each database vector is projected onto ``nbits``
random hyperplanes and reduced to an ``nbits``-bit binary code (sign of each
projection). Similarity search becomes a Hamming-distance scan over the packed
bit codes -- a cheap, lossy approximation of exact L2/IP search.
"""

import numpy as np

from .base import Index, _as_2d_f32, pad_results
from .metrics import METRIC_L2


# Precompute a lookup table mapping each uint8 value -> number of set bits.
# Indexing into this table is our (scalar) popcount.
# PERF: real FAISS uses a hardware SIMD popcount (e.g. the POPCNT instruction,
#       or vectorized byte-shuffle popcount) over the bit-packed codes and
#       multithreads the scan across the database. Here a 256-entry uint8 table
#       lookup stands in for that per-byte popcount.
_POPCOUNT_TABLE = np.array(
    [bin(i).count("1") for i in range(256)], dtype=np.uint8
)


class IndexLSH(Index):
    """Random-hyperplane LSH index (mirrors ``faiss.IndexLSH``).

    At construction we draw a fixed ``(nbits, d)`` Gaussian hyperplane matrix.
    Each vector is encoded to an ``nbits``-bit code by taking the sign of its
    projection onto every hyperplane; search ranks stored codes by Hamming
    distance to the query code. This is a single-table LSH: recall improves with
    more bits (and, in richer schemes, multiple tables).
    """

    def __init__(self, d, nbits, seed=1234):
        """Create an LSH index over ``d``-dim vectors using ``nbits`` bits.

        The random hyperplane matrix is drawn once here from a seeded RNG, so
        encoding is deterministic and reproducible. LSH needs no data-dependent
        training, so ``is_trained`` is True immediately.
        """
        # LSH always ranks by Hamming distance (smaller is better), regardless of
        # the original metric, so we keep the base default (METRIC_L2 semantics:
        # ascending, pad with +inf).
        super().__init__(d, metric_type=METRIC_L2)
        self.nbits = int(nbits)
        self.seed = int(seed)
        # Number of bytes each packed code occupies.
        self.code_size = (self.nbits + 7) // 8

        rng = np.random.default_rng(self.seed)
        # (nbits, d) Gaussian hyperplane normals; fixed for the life of the index.
        self.planes = rng.standard_normal((self.nbits, self.d)).astype(np.float32)

        # Packed codes, shape (ntotal, code_size), dtype uint8.
        self.codes = np.empty((0, self.code_size), dtype=np.uint8)
        self.is_trained = True

    def _encode(self, x):
        """Encode rows of ``x`` (n, d) to packed bit codes (n, code_size)."""
        x = _as_2d_f32(x)
        # Project onto hyperplanes: (n, d) @ (d, nbits) -> (n, nbits).
        # PERF: FAISS batches this projection through BLAS GEMM (SIMD + threads);
        #       we lean on numpy's BLAS for the matmul.
        proj = x @ self.planes.T
        bits = proj > 0.0  # boolean (n, nbits); True -> bit 1
        # np.packbits packs along the last axis, MSB-first, zero-padding the final
        # byte when nbits is not a multiple of 8.
        return np.packbits(bits, axis=1)

    def add(self, x):
        """Encode and store vectors ``x`` (n, d), assigning ids [ntotal, ntotal+n)."""
        self._check_trained()
        codes = self._encode(x)
        self.codes = np.concatenate([self.codes, codes], axis=0)
        self.ntotal = int(self.codes.shape[0])

    def _hamming(self, query_codes):
        """Hamming distances from each query code to every stored code.

        Returns an int array of shape (nq, ntotal).
        """
        # XOR every query byte against every stored byte via broadcasting:
        # (nq, 1, code_size) ^ (ntotal, code_size) -> (nq, ntotal, code_size).
        xor = query_codes[:, None, :] ^ self.codes[None, :, :]
        # Popcount each byte through the lookup table, then sum over bytes.
        # PERF: FAISS does this popcount with SIMD instructions over packed codes
        #       and multithreads across the database; our table-lookup + sum is
        #       the readable, single-threaded analogue.
        return _POPCOUNT_TABLE[xor].sum(axis=2)

    def search(self, x, k):
        """Return the ``k`` smallest-Hamming-distance neighbours of each query.

        ``D`` holds Hamming distances cast to float32; ``I`` holds int64 ids.
        Missing slots (fewer than ``k`` stored vectors) are padded with id -1 and
        distance +inf, matching the base L2 (smaller-is-better) convention.
        """
        self._check_trained()
        q = _as_2d_f32(x)
        nq = q.shape[0]

        D = np.empty((nq, k), dtype=np.float32)
        I = np.empty((nq, k), dtype=np.int64)

        if self.ntotal == 0:
            for i in range(nq):
                D[i], I[i] = pad_results([], [], k, self.metric_type)
            return D, I

        query_codes = self._encode(q)
        ham = self._hamming(query_codes).astype(np.float32)  # (nq, ntotal)

        kk = min(k, self.ntotal)
        # PERF: FAISS keeps a size-k heap while scanning; np.argpartition is the
        #       O(nb) partial-selection analogue used throughout minifaiss.
        part = np.argpartition(ham, kk - 1, axis=1)[:, :kk]
        rows = np.arange(nq)[:, None]
        part_d = ham[rows, part]
        order = np.argsort(part_d, axis=1)  # ascending: smaller Hamming is better
        ids = np.take_along_axis(part, order, axis=1).astype(np.int64)
        dists = np.take_along_axis(part_d, order, axis=1)

        for i in range(nq):
            D[i], I[i] = pad_results(ids[i], dists[i], k, self.metric_type)
        return D, I

    def reconstruct(self, key):
        """Not supported: LSH codes are lossy (only bit signs are stored).

        The original float vector cannot be recovered from its binary code, so
        reconstruction is intentionally unavailable (mirrors FAISS, whose
        ``IndexLSH`` does not provide meaningful reconstruction).
        """
        raise NotImplementedError(
            "IndexLSH cannot reconstruct vectors: the binary LSH code discards "
            "everything except the sign of each hyperplane projection, so the "
            "original float32 vector is unrecoverable."
        )

    def _reset(self):
        """Clear stored codes (invoked by :meth:`Index.reset`)."""
        self.codes = np.empty((0, self.code_size), dtype=np.uint8)
