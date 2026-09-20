"""Brute-force flat indexes for minifaiss.

Mirrors ``faiss.IndexFlat`` / ``faiss.IndexFlatL2`` / ``faiss.IndexFlatIP``.
A flat index stores every added vector verbatim and, at search time, computes
the exact distance from each query to every stored vector -- no approximation,
no pruning. It is the reference "ground truth" index in FAISS.
"""

import numpy as np

from .base import Index, _as_2d_f32, pad_results
from .metrics import (
    METRIC_L2,
    METRIC_INNER_PRODUCT,
    l2_sqr_distances,
    inner_products,
    topk,
)


class IndexFlat(Index):
    """Exact brute-force index over all stored vectors.

    Mirrors ``faiss.IndexFlat``. Keeps a growable float32 (ntotal, d) array of
    every added vector and scans all of them for each query, returning exact
    nearest neighbours under the chosen metric. Requires no training, so
    ``is_trained`` is True from construction.
    """

    def __init__(self, d, metric_type=METRIC_L2):
        """Create a flat index over ``d``-dim vectors using ``metric_type``."""
        super().__init__(d, metric_type)
        # Growable storage; starts empty with the right dtype/width so the first
        # vstack in add() concatenates cleanly.
        self._vectors = np.empty((0, self.d), dtype=np.float32)
        # Flat indexes need no training.
        self.is_trained = True

    def add(self, x):
        """Store vectors ``x`` (n, d), assigning sequential ids [ntotal, ntotal+n)."""
        self._check_trained()
        x = _as_2d_f32(x)
        if x.shape[1] != self.d:
            raise ValueError(
                f"expected vectors of dim {self.d}, got {x.shape[1]}"
            )
        # PERF: FAISS stores vectors in one contiguous, cache-friendly buffer and
        #       grows it in bulk; some variants keep them as float16/int8 to shrink
        #       memory bandwidth. We just vstack into a plain float32 array for
        #       clarity, ids being the implicit row order.
        self._vectors = np.vstack([self._vectors, x])
        self.ntotal = self._vectors.shape[0]

    def search(self, x, k):
        """Return ``(D, I)`` for the ``k`` exact nearest neighbours of each query.

        ``D`` is float32 (nq, k) distances, ``I`` is int64 (nq, k) ids. For L2
        the distances are squared and ranked ascending; for inner product they
        are ranked descending. Rows with fewer than ``k`` candidates are padded
        with id -1 and a worst-case distance (+inf for L2, -inf for IP).
        """
        self._check_trained()
        x = _as_2d_f32(x)
        if x.shape[1] != self.d:
            raise ValueError(
                f"expected queries of dim {self.d}, got {x.shape[1]}"
            )
        nq = x.shape[0]

        # PERF: this is a full O(nq * ntotal * d) scan with NO pruning -- every
        #       query is compared against every stored vector. FAISS runs the same
        #       exhaustive scan but funnels the distance math through BLAS GEMM
        #       (SIMD-vectorized, OpenMP multi-threaded, or a GPU kernel on
        #       faiss-gpu). Approximate indexes (IVF/PQ/HNSW) exist precisely to
        #       avoid this linear scan; IndexFlat trades speed for exactness.
        if self.metric_type == METRIC_INNER_PRODUCT:
            scores = inner_products(x, self._vectors)
            largest = True
        else:
            scores = l2_sqr_distances(x, self._vectors)
            largest = False

        # Top-k per query over the full score matrix (argpartition analogue of
        # FAISS's per-thread result heap).
        values, indices = topk(scores, k, largest)

        # Pad/truncate each query row to exactly k with the right sentinels.
        D = np.empty((nq, k), dtype=np.float32)
        I = np.empty((nq, k), dtype=np.int64)
        for i in range(nq):
            D[i], I[i] = pad_results(indices[i], values[i], k, self.metric_type)
        return D, I

    def reconstruct(self, key):
        """Return the exact stored (d,) vector for id ``key``."""
        if key < 0 or key >= self.ntotal:
            raise IndexError(
                f"id {key} out of range [0, {self.ntotal})"
            )
        # Copy so callers can't mutate our internal storage.
        return self._vectors[key].copy()

    def reconstruct_n(self, i0, ni):
        """Return exact stored vectors for ids [i0, i0+ni) as (ni, d)."""
        if i0 < 0 or i0 + ni > self.ntotal:
            raise IndexError(
                f"range [{i0}, {i0 + ni}) out of bounds [0, {self.ntotal})"
            )
        return self._vectors[i0:i0 + ni].copy()

    def _reset(self):
        """Clear stored vectors (hook called by base ``reset``)."""
        self._vectors = np.empty((0, self.d), dtype=np.float32)


class IndexFlatL2(IndexFlat):
    """Exact brute-force index using squared-L2 distance.

    Mirrors ``faiss.IndexFlatL2``. Convenience subclass of :class:`IndexFlat`
    that fixes the metric to ``METRIC_L2`` (smaller squared distance = better).
    """

    def __init__(self, d):
        """Create an L2 flat index over ``d``-dimensional vectors."""
        super().__init__(d, metric_type=METRIC_L2)


class IndexFlatIP(IndexFlat):
    """Exact brute-force index using inner-product similarity.

    Mirrors ``faiss.IndexFlatIP``. Convenience subclass of :class:`IndexFlat`
    that fixes the metric to ``METRIC_INNER_PRODUCT`` (larger score = better).
    """

    def __init__(self, d):
        """Create an inner-product flat index over ``d``-dimensional vectors."""
        super().__init__(d, metric_type=METRIC_INNER_PRODUCT)
