"""K-means clustering for minifaiss.

Mirrors ``faiss.Kmeans`` / FAISS's internal ``Clustering`` class. Implements
Lloyd's algorithm plainly, reusing :func:`metrics.l2_sqr_distances` for the
assignment step so the BLAS-backed distance path is shared with the indexes.
"""

import numpy as np

from .metrics import l2_sqr_distances


class Kmeans:
    """Lloyd's-algorithm k-means clustering (mirrors ``faiss.Kmeans``).

    Learns ``k`` centroids over ``d``-dimensional data by alternating a nearest-
    centroid assignment step with a centroid-update (mean) step. Used inside
    FAISS to train IVF coarse quantizers and PQ sub-quantizers.

    Initialisation uses k-means++ style seeding (probability proportional to
    squared distance from the nearest chosen centroid), which spreads the
    initial centroids out and converges faster than plain random init. Empty
    clusters are re-seeded from the farthest points each iteration.

    Attributes
    ----------
    centroids : ndarray | None
        (k, d) float32 array of learned centroids, or None before ``train``.
    """

    def __init__(self, d, k, niter=25, seed=1234, verbose=False):
        """Configure a k-means run.

        Parameters
        ----------
        d : int
            Vector dimensionality.
        k : int
            Number of clusters/centroids.
        niter : int
            Maximum number of Lloyd iterations.
        seed : int
            Seed for ``numpy.random.default_rng`` (reproducible; no global state).
        verbose : bool
            If True, print per-iteration inertia.
        """
        self.d = int(d)
        self.k = int(k)
        self.niter = int(niter)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.centroids = None
        # Reproducibility: a private RNG, never the global np.random state.
        self.rng = np.random.default_rng(seed)

    def _kpp_init(self, x):
        """k-means++ seeding: pick k spread-out initial centroids from ``x``."""
        n = x.shape[0]
        centroids = np.empty((self.k, self.d), dtype=np.float32)

        # First centroid: uniformly at random.
        first = int(self.rng.integers(n))
        centroids[0] = x[first]

        # Squared distance from each point to its nearest chosen centroid.
        closest_sq = l2_sqr_distances(x, centroids[0:1]).ravel()

        for c in range(1, self.k):
            total = float(closest_sq.sum())
            if total <= 0.0:
                # All remaining points coincide with chosen centroids; fall back
                # to picking a random distinct point.
                idx = int(self.rng.integers(n))
            else:
                probs = closest_sq / total
                idx = int(self.rng.choice(n, p=probs))
            centroids[c] = x[idx]
            # Update the running nearest-centroid squared distance.
            new_sq = l2_sqr_distances(x, centroids[c:c + 1]).ravel()
            closest_sq = np.minimum(closest_sq, new_sq)

        return centroids

    def train(self, x):
        """Run Lloyd's algorithm on ``x`` (n, d); return the final inertia.

        Inertia is the sum of squared distances from each point to its assigned
        centroid. ``self.centroids`` holds the (k, d) result afterwards.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]

        if n < self.k:
            raise ValueError(
                f"Kmeans needs at least k={self.k} points, got n={n}"
            )

        self.centroids = self._kpp_init(x)
        inertia = np.inf

        # PERF: FAISS's Clustering runs the assignment step through BLAS (one big
        #       GEMM per iteration) and multithreads the iterations with OpenMP;
        #       the assignment reuses the SIMD L2 kernels. Here each iteration is
        #       a plain numpy loop leaning on metrics.l2_sqr_distances' matmul.
        for it in range(self.niter):
            # --- Assignment step: nearest centroid for every point. ---
            dists = l2_sqr_distances(x, self.centroids)   # (n, k)
            codes = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(n), codes]
            inertia = float(min_dists.sum())

            # --- Update step: each centroid becomes the mean of its members. ---
            new_centroids = np.zeros((self.k, self.d), dtype=np.float32)
            counts = np.zeros(self.k, dtype=np.int64)
            # Accumulate sums per cluster (np.add.at handles repeated indices).
            np.add.at(new_centroids, codes, x)
            np.add.at(counts, codes, 1)

            empty = np.where(counts == 0)[0]
            nonempty = counts > 0
            new_centroids[nonempty] /= counts[nonempty][:, None].astype(np.float32)

            # Re-seed empty clusters from the points currently farthest from
            # their assigned centroid (a standard FAISS-style repair).
            if empty.size > 0:
                farthest = np.argsort(min_dists)[::-1]
                take = farthest[:empty.size]
                new_centroids[empty] = x[take]

            self.centroids = new_centroids

            if self.verbose:
                print(f"Kmeans iteration {it}: inertia={inertia:.4f}")

        return inertia

    def assign(self, x):
        """Assign each row of ``x`` to its nearest centroid.

        Returns
        -------
        codes : ndarray, shape (n,), dtype int64
            Nearest-centroid id for each input row.
        dists : ndarray, shape (n,), dtype float32
            Squared L2 distance to that nearest centroid.
        """
        if self.centroids is None:
            raise RuntimeError("Kmeans must be trained before assign()")
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        dists = l2_sqr_distances(x, self.centroids)   # (n, k)
        codes = np.argmin(dists, axis=1).astype(np.int64)
        best = dists[np.arange(x.shape[0]), codes].astype(np.float32)
        return codes, best
