"""Inverted-file (IVF) index for minifaiss.

Mirrors ``faiss.IndexIVFFlat``. An IVF index partitions the vector space into
``nlist`` Voronoi cells (learned by k-means), stores each database vector in the
cell of its nearest centroid, and at search time only scans the ``nprobe`` cells
closest to the query instead of the whole dataset. "Flat" means each cell keeps
the full, uncompressed vectors, so reconstruction is exact.
"""

import numpy as np

from .base import Index, pad_results, _as_2d_f32
from .metrics import (
    METRIC_L2,
    METRIC_INNER_PRODUCT,
    l2_sqr_distances,
    inner_products,
    topk,
)
from .flat import IndexFlatL2
from .kmeans import Kmeans


class IndexIVFFlat(Index):
    """Inverted-file index with flat (uncompressed) per-cell storage.

    Mirrors ``faiss.IndexIVFFlat``. A coarse quantizer maps each vector to one of
    ``nlist`` centroids; the vector's id is appended to that centroid's inverted
    list. Searching probes only the ``nprobe`` centroids nearest the query and
    scans just those lists -- trading a little recall for a large speed-up.
    """

    def __init__(self, quantizer, d, nlist, metric_type=METRIC_L2):
        """Create an IVF index.

        Parameters
        ----------
        quantizer : Index or None
            The coarse quantizer holding the ``nlist`` centroids (matching
            FAISS's pattern of passing a quantizer object, typically an
            ``IndexFlatL2`` over the centroids). If None, an internal
            ``IndexFlatL2(d)`` is created.
        d : int
            Vector dimensionality.
        nlist : int
            Number of Voronoi cells / coarse centroids.
        metric_type : int
            ``METRIC_L2`` (squared L2, ascending) or ``METRIC_INNER_PRODUCT``
            (descending).
        """
        super().__init__(d, metric_type)
        if quantizer is None:
            quantizer = IndexFlatL2(d)
        self.quantizer = quantizer
        self.nlist = int(nlist)
        self.nprobe = 1
        # An IVF index must learn its centroids before it can add/search.
        self.is_trained = False
        # Inverted lists: one Python list of vector-ids per cell.
        self.invlists = [[] for _ in range(self.nlist)]
        # Flat storage of the full vectors, indexed by vector id (exact recon).
        self._store = []

    def train(self, x):
        """Learn ``nlist`` centroids with k-means and load them into the quantizer.

        Runs :class:`~minifaiss.kmeans.Kmeans` on ``x`` to obtain the coarse
        centroids, resets the quantizer, and adds the centroids to it so that
        ``quantizer.search`` returns cell ids in ``[0, nlist)``.
        """
        x = _as_2d_f32(x)
        km = Kmeans(self.d, self.nlist)
        km.train(x)
        # Load the learned centroids into the coarse quantizer. Reset first so
        # repeated training does not stack stale centroids.
        self.quantizer.reset()
        self.quantizer.add(km.centroids)
        self.is_trained = True

    def add(self, x):
        """Add vectors, appending each to the inverted list of its nearest cell.

        Ids are assigned sequentially as ``[ntotal, ntotal + n)`` (matching the
        base API). The full vector is stored so :meth:`reconstruct` is exact.
        """
        self._check_trained()
        x = _as_2d_f32(x)
        n = x.shape[0]

        # Coarse assignment: nearest centroid (cell) for each new vector.
        # PERF: FAISS batches this coarse quantization through BLAS; here it is
        #       a single quantizer.search call backed by numpy's matmul.
        _, cell_ids = self.quantizer.search(x, 1)   # (n, 1)
        cell_ids = cell_ids[:, 0]

        for i in range(n):
            vec_id = self.ntotal + i
            self._store.append(np.array(x[i], dtype=np.float32))
            self.invlists[int(cell_ids[i])].append(vec_id)

        self.ntotal += n

    def search(self, x, k):
        """Search the ``k`` nearest neighbours, scanning only ``nprobe`` cells.

        For each query the ``nprobe`` nearest centroids are found via the
        quantizer; the candidate vectors gathered from those cells are then
        scored exactly and reduced to the top ``k``.
        """
        self._check_trained()
        x = _as_2d_f32(x)
        nq = x.shape[0]

        largest = self.metric_type == METRIC_INNER_PRODUCT

        # Coarse search: the nprobe nearest cells for every query at once.
        probe = min(self.nprobe, self.nlist)
        _, probe_cells = self.quantizer.search(x, probe)   # (nq, probe)

        D = np.empty((nq, k), dtype=np.float32)
        I = np.empty((nq, k), dtype=np.int64)

        # PERF: only nprobe/nlist of the data is scanned -- the whole point of
        #       IVF. Real FAISS scans the probed lists in parallel with OpenMP,
        #       uses BLAS per list, and stores each list contiguously for cache
        #       efficiency. We loop over queries and cells plainly for clarity.
        for qi in range(nq):
            # Gather candidate ids from every probed (valid) cell.
            cand_ids = []
            for c in probe_cells[qi]:
                c = int(c)
                if c < 0:           # padding when nlist < nprobe
                    continue
                cand_ids.extend(self.invlists[c])

            if not cand_ids:
                D[qi], I[qi] = pad_results([], [], k, self.metric_type)
                continue

            cand_ids = np.asarray(cand_ids, dtype=np.int64)
            cand_vecs = np.stack([self._store[j] for j in cand_ids])

            if largest:
                scores = inner_products(x[qi:qi + 1], cand_vecs)   # (1, m)
            else:
                scores = l2_sqr_distances(x[qi:qi + 1], cand_vecs)  # (1, m)

            vals, cols = topk(scores, k, largest)   # (1, kk)
            sel_ids = cand_ids[cols[0]]
            D[qi], I[qi] = pad_results(sel_ids, vals[0], k, self.metric_type)

        return D, I

    def reconstruct(self, key):
        """Return the exact (d,) vector stored under id ``key``."""
        if key < 0 or key >= self.ntotal:
            raise IndexError(f"id {key} out of range [0, {self.ntotal})")
        return np.array(self._store[int(key)], dtype=np.float32)

    def _reset(self):
        """Clear inverted lists and stored vectors (called by ``Index.reset``)."""
        self.invlists = [[] for _ in range(self.nlist)]
        self._store = []
