"""Product Quantization for minifaiss.

This module is the flagship of the package. It reimplements the core ideas of
FAISS's product-quantization indexes in plain numpy:

* :class:`ProductQuantizer` -- the codebook trainer/encoder/decoder that mirrors
  ``faiss.ProductQuantizer``.
* :class:`IndexPQ` -- a flat PQ index (mirrors ``faiss.IndexPQ``) that stores
  only compact codes and searches with asymmetric distance computation (ADC).
* :class:`IndexIVFPQ` -- the inverted-file + PQ index (mirrors
  ``faiss.IndexIVFPQ``, FAISS's most-used index) that PQ-encodes *residuals*
  relative to a coarse quantizer and prunes the search to ``nprobe`` cells.

Product quantization splits each ``d``-dim vector into ``m`` contiguous
sub-vectors and quantizes each sub-vector against its own little codebook of
``ksub = 2**nbits`` centroids. A vector is then stored as just ``m`` bytes
(one code per subspace), giving a huge memory saving over the raw ``4*d`` bytes.
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
from .kmeans import Kmeans

# The IVFPQ coarse quantizer mirrors FAISS, where you pass an ``IndexFlatL2`` as
# the coarse quantizer. We import it if the sibling module exists yet; the class
# also works with any object exposing ``reset``/``add`` (or none at all, since we
# keep our own copy of the coarse centroids).
try:  # pragma: no cover - availability depends on build order
    from .flat import IndexFlatL2
except Exception:  # noqa: BLE001 - flat.py may not be present during dev
    IndexFlatL2 = None


class ProductQuantizer:
    """Product quantizer: per-subspace codebooks (mirrors ``faiss.ProductQuantizer``).

    Each ``d``-dim vector is cut into ``m`` contiguous sub-vectors of length
    ``dsub = d // m``. A separate k-means codebook of ``ksub = 2**nbits``
    centroids is learned per subspace, so a vector compresses to ``m`` integer
    codes (``m`` bytes at ``nbits=8``) instead of ``4*d`` raw float bytes.
    """

    def __init__(self, d, m, nbits=8, seed=1234):
        """Configure a product quantizer.

        Parameters
        ----------
        d : int
            Full vector dimensionality. Must be divisible by ``m``.
        m : int
            Number of subspaces (sub-quantizers). ``bytes/vector == m``.
        nbits : int
            Bits per subspace code; ``ksub = 2**nbits`` centroids per subspace.
        seed : int
            Base seed for the per-subspace k-means (reproducible; no global state).
        """
        if d % m != 0:
            raise ValueError(f"d={d} must be divisible by m={m}")
        self.d = int(d)
        self.m = int(m)
        self.nbits = int(nbits)
        self.dsub = self.d // self.m
        self.ksub = 2 ** self.nbits
        self.seed = int(seed)
        # (m, ksub, dsub) after train(): one codebook per subspace.
        self.centroids = None
        self.is_trained = False

    def _subvectors(self, x, j):
        """Return the ``j``-th contiguous sub-block of ``x`` as (n, dsub)."""
        return x[:, j * self.dsub:(j + 1) * self.dsub]

    def train(self, x):
        """Learn the ``m`` sub-codebooks from training data ``x`` (n, d).

        Runs an independent k-means per subspace and stacks the results into
        ``self.centroids`` of shape (m, ksub, dsub).
        """
        x = _as_2d_f32(x)
        if x.shape[1] != self.d:
            raise ValueError(f"expected d={self.d}, got {x.shape[1]}")

        self.centroids = np.empty((self.m, self.ksub, self.dsub), dtype=np.float32)
        for j in range(self.m):
            sub = np.ascontiguousarray(self._subvectors(x, j))
            # Distinct seed per subspace so the codebooks are independent but the
            # whole quantizer stays reproducible.
            km = Kmeans(self.dsub, self.ksub, seed=self.seed + j)
            km.train(sub)
            self.centroids[j] = km.centroids
        self.is_trained = True
        return self

    def compute_codes(self, x):
        """Encode ``x`` (n, d) to PQ codes of shape (n, m), dtype uint8.

        For each subspace we assign the sub-vector to its nearest sub-centroid.
        """
        self._check_trained()
        x = _as_2d_f32(x)
        n = x.shape[0]
        # PERF: uint8 storage assumes nbits<=8 (ksub<=256) -- exactly FAISS's
        #       default PQ layout of m bytes per vector. Larger nbits would need
        #       bit-packing, which FAISS does; we keep the common 8-bit case.
        codes = np.empty((n, self.m), dtype=np.uint8)
        for j in range(self.m):
            sub = self._subvectors(x, j)
            dists = l2_sqr_distances(sub, self.centroids[j])  # (n, ksub)
            codes[:, j] = np.argmin(dists, axis=1).astype(np.uint8)
        return codes

    def decode(self, codes):
        """Approximately reconstruct vectors from PQ ``codes`` (n, m) -> (n, d).

        Concatenates the chosen sub-centroid from each subspace. This is lossy:
        it returns the codebook approximation of the original vector.
        """
        self._check_trained()
        codes = np.asarray(codes)
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)
        n = codes.shape[0]
        out = np.empty((n, self.d), dtype=np.float32)
        for j in range(self.m):
            # Gather the sub-centroid picked for subspace j for every row.
            out[:, j * self.dsub:(j + 1) * self.dsub] = self.centroids[j][codes[:, j]]
        return out

    def distance_table(self, x):
        """Build the ADC squared-L2 lookup table for a query batch.

        Returns an array of shape (nq, m, ksub) where entry ``[i, j, c]`` is the
        squared-L2 distance from query ``i``'s ``j``-th sub-vector to sub-centroid
        ``c`` of subspace ``j``.

        This is the heart of asymmetric distance computation: once this small
        table is built, the (approximate) distance from a query to any database
        code is just the sum of ``m`` table look-ups -- no ``d``-dim arithmetic.
        """
        self._check_trained()
        x = _as_2d_f32(x)
        nq = x.shape[0]
        # PERF: ADC replaces an O(d) per-vector distance with m table look-ups.
        #       FAISS's "fast-scan" kernels pack codes so these look-ups happen in
        #       SIMD registers (shuffle instructions) across many codes at once;
        #       here we do plain fancy-indexed numpy sums.
        table = np.empty((nq, self.m, self.ksub), dtype=np.float32)
        for j in range(self.m):
            sub = self._subvectors(x, j)
            table[:, j, :] = l2_sqr_distances(sub, self.centroids[j])
        return table

    def inner_product_table(self, x):
        """Build the ADC inner-product lookup table for a query batch.

        Returns (nq, m, ksub) where entry ``[i, j, c]`` is the inner product of
        query ``i``'s ``j``-th sub-vector with sub-centroid ``c``. Summing the
        ``m`` look-ups for a code gives the approximate query-vector dot product
        (used for ``METRIC_INNER_PRODUCT`` search).
        """
        self._check_trained()
        x = _as_2d_f32(x)
        nq = x.shape[0]
        table = np.empty((nq, self.m, self.ksub), dtype=np.float32)
        for j in range(self.m):
            sub = self._subvectors(x, j)
            table[:, j, :] = inner_products(sub, self.centroids[j])
        return table

    def _check_trained(self):
        if not self.is_trained or self.centroids is None:
            raise RuntimeError("ProductQuantizer must be trained first")


def _adc_scan(table_row, codes):
    """Sum ADC look-ups for one query over a block of codes.

    Parameters
    ----------
    table_row : ndarray (m, ksub)
        The per-query lookup table (distances or inner products).
    codes : ndarray (nb, m) uint8
        Database codes to score.

    Returns
    -------
    ndarray (nb,) float32 -- the summed score for each code.
    """
    nb = codes.shape[0]
    m = table_row.shape[0]
    scores = np.zeros(nb, dtype=np.float32)
    # PERF: the loop is over m (small); FAISS fuses these m gathers into one
    #       SIMD fast-scan pass over packed codes. Our per-subspace fancy index
    #       table_row[j, codes[:, j]] is the readable numpy analogue.
    for j in range(m):
        scores += table_row[j, codes[:, j]]
    return scores


class IndexPQ(Index):
    """Flat product-quantization index (mirrors ``faiss.IndexPQ``).

    Stores each added vector as just ``m`` PQ codes (bytes), never the full
    float vector -- roughly a ``4*d / m`` memory reduction. Search uses
    asymmetric distance computation: build a per-query lookup table, then score
    every stored code by summing ``m`` table look-ups.
    """

    def __init__(self, d, m, nbits=8, metric_type=METRIC_L2, seed=1234):
        """Create an ``IndexPQ`` over ``d``-dim vectors with ``m`` subspaces."""
        super().__init__(d, metric_type)
        self.pq = ProductQuantizer(d, m, nbits, seed=seed)
        self.m = self.pq.m
        self.is_trained = False
        # Compact storage: (ntotal, m) uint8. Bytes-per-vector == m.
        self.codes = np.empty((0, self.m), dtype=np.uint8)

    def train(self, x):
        """Train the underlying product quantizer's codebooks on ``x``."""
        self.pq.train(x)
        self.is_trained = True

    def add(self, x):
        """PQ-encode ``x`` and append the codes; assign ids [ntotal, ntotal+n)."""
        self._check_trained()
        x = _as_2d_f32(x)
        new_codes = self.pq.compute_codes(x)
        self.codes = np.vstack([self.codes, new_codes]) if self.ntotal else new_codes
        self.ntotal += new_codes.shape[0]

    def search(self, x, k):
        """Search ``k`` approximate nearest neighbours via ADC.

        For ``METRIC_L2`` the lookup table holds squared-L2 distances (smaller is
        better); for ``METRIC_INNER_PRODUCT`` it holds inner products (larger is
        better). Missing slots are padded with id ``-1`` and ``+inf`` / ``-inf``.
        """
        self._check_trained()
        q = _as_2d_f32(x)
        nq = q.shape[0]
        largest = self.metric_type == METRIC_INNER_PRODUCT

        D = np.empty((nq, k), dtype=np.float32)
        I = np.empty((nq, k), dtype=np.int64)

        if self.ntotal == 0:
            for i in range(nq):
                D[i], I[i] = pad_results([], [], k, self.metric_type)
            return D, I

        if largest:
            table = self.pq.inner_product_table(q)   # (nq, m, ksub)
        else:
            table = self.pq.distance_table(q)         # (nq, m, ksub)

        # ids are the sequential row positions, so column index == vector id.
        all_ids = np.arange(self.ntotal, dtype=np.int64)
        for i in range(nq):
            scores = _adc_scan(table[i], self.codes)   # (ntotal,)
            vals, idx = topk(scores[None, :], k, largest)
            D[i], I[i] = pad_results(all_ids[idx[0]], vals[0], k, self.metric_type)
        return D, I

    def reconstruct(self, key):
        """Return the APPROXIMATE decoded vector (d,) for stored id ``key``."""
        if key < 0 or key >= self.ntotal:
            raise IndexError(f"id {key} out of range [0, {self.ntotal})")
        return self.pq.decode(self.codes[key:key + 1])[0]

    def _reset(self):
        """Clear stored codes (invoked by :meth:`Index.reset`)."""
        self.codes = np.empty((0, self.m), dtype=np.uint8)


class IndexIVFPQ(Index):
    """Inverted-file + PQ index (mirrors ``faiss.IndexIVFPQ``).

    FAISS's most-used index. A coarse k-means quantizer (``nlist`` cells) buckets
    each vector; within a cell we store the PQ code of the *residual*
    (``x - centroid``), which is smaller and more uniform so PQ quantizes it more
    accurately. Search probes only the ``nprobe`` nearest cells and scores their
    codes with ADC on the query's residual.
    """

    def __init__(self, quantizer, d, nlist, m, nbits=8, metric_type=METRIC_L2, seed=1234):
        """Create an ``IndexIVFPQ``.

        Parameters
        ----------
        quantizer : Index or None
            Coarse quantizer (mirrors passing a ``faiss.IndexFlatL2``). We keep
            our own copy of the learned centroids, and also populate this object
            via ``reset``/``add`` when it supports them, matching FAISS's API.
        d, nlist, m, nbits, metric_type : see class/PQ docs.
        seed : int
            Seed for the coarse k-means and the residual product quantizer.
        """
        super().__init__(d, metric_type)
        self.quantizer = quantizer
        self.nlist = int(nlist)
        self.m = int(m)
        self.nbits = int(nbits)
        self.seed = int(seed)
        self.nprobe = 1
        self.pq = ProductQuantizer(d, m, nbits, seed=seed)
        self.is_trained = False

        self.coarse_centroids = None            # (nlist, d) after train
        self._init_lists()

    def _init_lists(self):
        """Allocate empty inverted lists and the id->location direct map."""
        self.invlists_codes = [np.empty((0, self.m), dtype=np.uint8)
                               for _ in range(self.nlist)]
        self.invlists_ids = [np.empty((0,), dtype=np.int64)
                             for _ in range(self.nlist)]
        # id -> (cell, offset within that cell) for reconstruct().
        self.direct_map = {}

    def train(self, x):
        """Train the coarse quantizer, then the PQ on residuals.

        1. Run k-means with ``nlist`` centroids (the coarse quantizer).
        2. Assign every training vector to its nearest centroid and form the
           residuals ``x - centroid``.
        3. Train the product quantizer on those residuals.
        """
        x = _as_2d_f32(x)

        km = Kmeans(self.d, self.nlist, seed=self.seed)
        km.train(x)
        self.coarse_centroids = km.centroids.astype(np.float32, copy=True)

        # Mirror FAISS: load the coarse centroids into the passed quantizer.
        if self.quantizer is not None:
            try:
                self.quantizer.reset()
                self.quantizer.add(self.coarse_centroids)
            except Exception:  # noqa: BLE001 - tolerate a bare/absent quantizer
                pass

        # PERF: the residual trick tightens quantization -- residuals cluster near
        #       the origin with smaller variance, so a fixed PQ codebook covers
        #       them more finely than it would the raw vectors.
        assign = np.argmin(l2_sqr_distances(x, self.coarse_centroids), axis=1)
        residuals = x - self.coarse_centroids[assign]
        self.pq.train(residuals)

        self.is_trained = True

    def _assign_coarse(self, x):
        """Return the nearest coarse-cell id for each row of ``x`` (n,) int64."""
        dists = l2_sqr_distances(x, self.coarse_centroids)
        return np.argmin(dists, axis=1).astype(np.int64)

    def add(self, x):
        """Assign to a cell, PQ-encode the residual, and store in that cell."""
        self._check_trained()
        x = _as_2d_f32(x)
        n = x.shape[0]
        cells = self._assign_coarse(x)
        residuals = x - self.coarse_centroids[cells]
        codes = self.pq.compute_codes(residuals)          # (n, m) uint8

        for i in range(n):
            c = int(cells[i])
            vid = self.ntotal + i
            offset = self.invlists_ids[c].shape[0]
            self.invlists_codes[c] = np.vstack([self.invlists_codes[c],
                                                codes[i:i + 1]])
            self.invlists_ids[c] = np.append(self.invlists_ids[c], vid)
            self.direct_map[vid] = (c, offset)
        self.ntotal += n

    def search(self, x, k):
        """Search ``k`` approximate neighbours over the ``nprobe`` nearest cells.

        For each query we pick the ``nprobe`` best coarse cells, and within each
        cell build the ADC table for the query's residual w.r.t. that cell's
        centroid, score the cell's codes, then merge candidates across cells and
        keep the global top-``k``.
        """
        self._check_trained()
        q = _as_2d_f32(x)
        nq = q.shape[0]
        largest = self.metric_type == METRIC_INNER_PRODUCT

        # PERF: nprobe/nlist pruning -- we only touch cells whose centroid is
        #       closest to the query, skipping the vast majority of the database.
        if largest:
            coarse_scores = inner_products(q, self.coarse_centroids)
        else:
            coarse_scores = l2_sqr_distances(q, self.coarse_centroids)
        _, probe_cells = topk(coarse_scores, self.nprobe, largest)  # (nq, nprobe)

        D = np.empty((nq, k), dtype=np.float32)
        I = np.empty((nq, k), dtype=np.int64)

        for i in range(nq):
            cand_ids = []
            cand_scores = []
            for c in probe_cells[i]:
                c = int(c)
                codes = self.invlists_codes[c]
                if codes.shape[0] == 0:
                    continue
                centroid = self.coarse_centroids[c]
                residual = (q[i] - centroid).reshape(1, -1)
                if largest:
                    # approx q . db = q . centroid + q . decoded_residual.
                    # The stored code approximates the DATABASE residual, so the
                    # ADC table must dot the FULL query sub-vectors (NOT q-centroid)
                    # against the residual codebook: sum_j q_sub_j . residual_c_j.
                    # (Using the query residual here would drop a
                    # centroid . decoded_residual term and bias the score.)
                    table = self.pq.inner_product_table(q[i].reshape(1, -1))[0]
                    base = float(q[i] @ centroid)
                    scores = _adc_scan(table, codes) + base
                else:
                    table = self.pq.distance_table(residual)[0]
                    scores = _adc_scan(table, codes)
                cand_ids.append(self.invlists_ids[c])
                cand_scores.append(scores)

            if cand_ids:
                ids = np.concatenate(cand_ids)
                scores = np.concatenate(cand_scores)
                vals, idx = topk(scores[None, :], k, largest)
                D[i], I[i] = pad_results(ids[idx[0]], vals[0], k, self.metric_type)
            else:
                D[i], I[i] = pad_results([], [], k, self.metric_type)
        return D, I

    def reconstruct(self, key):
        """Return APPROXIMATE vector (d,) = centroid + decoded residual."""
        if key not in self.direct_map:
            raise IndexError(f"id {key} not found")
        c, offset = self.direct_map[key]
        code = self.invlists_codes[c][offset:offset + 1]
        residual = self.pq.decode(code)[0]
        return (self.coarse_centroids[c] + residual).astype(np.float32)

    def _reset(self):
        """Clear all inverted lists (invoked by :meth:`Index.reset`)."""
        self._init_lists()
