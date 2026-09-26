"""Hierarchical Navigable Small World (HNSW) graph index for minifaiss.

Mirrors ``faiss.IndexHNSWFlat``. HNSW is a graph-based approximate-nearest-
neighbour index: nodes live on a stack of layers, upper layers are sparse
"express lanes" for coarse navigation and layer 0 holds every vector. Search
greedy-descends the upper layers to a good entry point, then runs a bounded
beam search on layer 0. It trades extra memory (the graph) for fast, high-recall
approximate search.

The implementation follows Malkov & Yashunin (2016) and stays deliberately
plain so the algorithm is readable on small data, even though it is slow.
"""

import math
import heapq

import numpy as np

from .base import Index, _as_2d_f32, pad_results
from .metrics import METRIC_L2, METRIC_INNER_PRODUCT, l2_sqr_distances, inner_products


class IndexHNSWFlat(Index):
    """Flat-storage HNSW graph index (mirrors ``faiss.IndexHNSWFlat``).

    Stores full float32 vectors (so ``reconstruct`` is exact) and builds a
    multi-layer navigable small-world graph over them. Requires no training
    (``is_trained`` is True from construction). Search quality/speed is tuned by
    ``efConstruction`` (build-time beam width) and ``efSearch`` (query-time beam
    width); ``M`` sets the neighbour budget per node per layer.

    Internally all "distances" are kept in a *smaller-is-closer* convention so
    the heaps behave uniformly: for L2 this is the squared distance, for inner
    product it is the negated dot product. Reported distances are converted back
    to the FAISS convention (squared L2 ascending, inner product descending).

    Attributes
    ----------
    M : int
        Neighbour budget per node on the upper layers.
    M0 : int
        Neighbour budget per node on layer 0, ``2 * M`` as in the HNSW paper
        (``M_max0``) and ``faiss.IndexHNSWFlat``. Layer 0 holds every vector and
        is where the search does its real work, so it gets the denser graph.
    efConstruction : int
        Beam width used while inserting (larger -> better graph, slower build).
    efSearch : int
        Beam width used while querying (larger -> better recall, slower search).
    """

    def __init__(self, d, M=16, metric_type=METRIC_L2, seed=1234):
        """Create an HNSW index over ``d``-dim vectors with ``M`` neighbours/node.

        Parameters
        ----------
        d : int
            Vector dimensionality.
        M : int
            Neighbours per node on the upper layers (graph degree budget);
            layer 0 gets ``2 * M``.
        metric_type : int
            ``METRIC_L2`` or ``METRIC_INNER_PRODUCT``.
        seed : int
            Seed for the private RNG that draws node levels (reproducible).
        """
        super().__init__(d, metric_type)
        self.M = int(M)
        self.M0 = 2 * self.M
        self.efConstruction = 40
        self.efSearch = 16

        # Level-generation constant: level ~ floor(-ln(U) * mL), mL = 1/ln(M).
        self._mL = 1.0 / math.log(self.M) if self.M > 1 else 1.0

        # Reproducibility: a private RNG, never the global np.random state.
        self.rng = np.random.default_rng(seed)

        self._init_storage()

    # ------------------------------------------------------------------ #
    # Storage / bookkeeping
    # ------------------------------------------------------------------ #
    def _init_storage(self):
        """(Re)initialise the vector store and graph structures."""
        # Full vectors, one 1-D float32 array per id (flat storage -> exact
        # reconstruct).
        self._data = []
        # Top layer index of each node (node lives on layers 0..level).
        self._levels = []
        # Adjacency: self._neighbors[node][layer] -> list of neighbour ids.
        self._neighbors = []
        # Global entry point (id of the node on the highest layer) and the
        # current maximum layer present in the graph.
        self._entry_point = -1
        self._max_level = -1

    def _reset(self):
        """Reset hook invoked by ``Index.reset`` -- clears the graph."""
        self._init_storage()

    # ------------------------------------------------------------------ #
    # Distance helpers (smaller = closer, uniformly, for both metrics)
    # ------------------------------------------------------------------ #
    def _closeness_many(self, q, ids):
        """Return smaller-is-closer distances from vector ``q`` to each id.

        For L2 this is the squared distance; for inner product it is the negated
        dot product, so a smaller value always means "more similar".
        """
        arr = np.stack([self._data[i] for i in ids])   # (m, d)
        qq = q.reshape(1, -1)
        if self.metric_type == METRIC_INNER_PRODUCT:
            return -inner_products(qq, arr).ravel()
        return l2_sqr_distances(qq, arr).ravel()

    def _closeness_pair(self, a, b):
        """Smaller-is-closer distance between two single vectors ``a`` and ``b``."""
        if self.metric_type == METRIC_INNER_PRODUCT:
            return float(-np.dot(a, b))
        diff = a - b
        return float(np.dot(diff, diff))

    def _to_metric(self, closeness):
        """Convert an internal closeness value to the reported metric value."""
        if self.metric_type == METRIC_INNER_PRODUCT:
            return -closeness          # closeness = -ip  ->  ip = -closeness
        return closeness               # L2 closeness already the squared dist

    # ------------------------------------------------------------------ #
    # Graph navigation
    # ------------------------------------------------------------------ #
    def _random_level(self):
        """Draw a node's top level from the exponential level distribution."""
        # U in (0, 1]; guard against log(0).
        u = float(self.rng.random())
        if u <= 0.0:
            u = np.finfo(np.float32).tiny
        return int(math.floor(-math.log(u) * self._mL))

    def _greedy_descend(self, q, entry, entry_c, layer):
        """Greedily walk ``layer`` toward ``q`` starting from ``entry``.

        Returns the closest node found and its closeness value. Used for the
        coarse top-down descent through the sparse upper layers (ef = 1).
        """
        cur, cur_c = entry, entry_c
        improved = True
        while improved:
            improved = False
            neigh = self._neighbors[cur][layer]
            if not neigh:
                break
            ds = self._closeness_many(q, neigh)
            j = int(np.argmin(ds))
            if ds[j] < cur_c:
                cur, cur_c = neigh[j], float(ds[j])
                improved = True
        return cur, cur_c

    def _search_layer(self, q, entry_points, ef, layer):
        """Beam search ``layer`` and return up to ``ef`` closest nodes.

        Parameters
        ----------
        q : ndarray (d,)
            Query vector.
        entry_points : list of (closeness, id)
            Seeds for the search (closeness = smaller-is-closer).
        ef : int
            Beam width (result-set size cap).
        layer : int
            Layer to explore.

        Returns
        -------
        list of (closeness, id), unsorted.
        """
        visited = set()
        # candidates: min-heap by closeness (closest popped first).
        candidates = []
        # results: max-heap emulated by storing (-closeness, id) so the current
        # worst (largest closeness) sits at the top for O(1) eviction.
        results = []
        for c, node in entry_points:
            heapq.heappush(candidates, (c, node))
            heapq.heappush(results, (-c, node))
            visited.add(node)

        while candidates:
            c, node = heapq.heappop(candidates)
            worst = -results[0][0]
            if c > worst and len(results) >= ef:
                break

            # Batch the distance to all unvisited neighbours at once (this is the
            # BLAS-backed matmul path in metrics).
            fresh = [e for e in self._neighbors[node][layer] if e not in visited]
            if not fresh:
                continue
            for e in fresh:
                visited.add(e)
            ds = self._closeness_many(q, fresh)

            for de, e in zip(ds, fresh):
                de = float(de)
                worst = -results[0][0]
                if de < worst or len(results) < ef:
                    heapq.heappush(candidates, (de, e))
                    heapq.heappush(results, (-de, e))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [(-nc, nid) for nc, nid in results]

    def _select_neighbors_heuristic(self, candidates, M):
        """Prune ``candidates`` to ``M`` ids using FAISS's diversity heuristic.

        ``candidates`` is a list of ``(closeness_to_base, id)``. A candidate is
        kept only if it is closer to the base node than to any already-selected
        neighbour; this favours a diverse, well-spread neighbourhood over a tight
        cluster and keeps the graph navigable.
        """
        ordered = sorted(candidates, key=lambda t: t[0])
        selected = []
        for dist_to_base, cid in ordered:
            if len(selected) >= M:
                break
            cvec = self._data[cid]
            keep = True
            for sid in selected:
                if self._closeness_pair(cvec, self._data[sid]) < dist_to_base:
                    keep = False
                    break
            if keep:
                selected.append(cid)
        return selected

    def _max_degree(self, layer):
        """Neighbour budget on ``layer``: ``M0 = 2M`` on layer 0, ``M`` above
        (faiss's ``nb_neighbors(layer)``)."""
        return self.M0 if layer == 0 else self.M

    def _prune(self, node, layer):
        """Shrink ``node``'s neighbour list on ``layer`` back to its budget via heuristic."""
        neigh = self._neighbors[node][layer]
        budget = self._max_degree(layer)
        if len(neigh) <= budget:
            return
        base = self._data[node]
        cand = [(self._closeness_pair(base, self._data[c]), c) for c in neigh]
        self._neighbors[node][layer] = self._select_neighbors_heuristic(cand, budget)

    # ------------------------------------------------------------------ #
    # Insertion
    # ------------------------------------------------------------------ #
    def add(self, x):
        """Add vectors ``x`` (n, d), inserting each into the graph one by one.

        Ids are assigned sequentially as ``[ntotal, ntotal + n)``.
        """
        self._check_trained()
        x = _as_2d_f32(x)
        if x.shape[1] != self.d:
            raise ValueError(f"expected dim {self.d}, got {x.shape[1]}")
        for row in x:
            self._insert(np.ascontiguousarray(row, dtype=np.float32))

    def _insert(self, q):
        """Insert a single vector ``q`` (d,) into the graph."""
        node = len(self._data)
        level = self._random_level()

        self._data.append(q)
        self._levels.append(level)
        self._neighbors.append([[] for _ in range(level + 1)])

        # First-ever node: it becomes the entry point and we are done.
        if self._entry_point == -1:
            self._entry_point = node
            self._max_level = level
            self.ntotal += 1
            return

        ep = self._entry_point
        ep_c = float(self._closeness_many(q, [ep])[0])

        # Phase 1: greedy descent through layers above this node's top level.
        for lc in range(self._max_level, level, -1):
            ep, ep_c = self._greedy_descend(q, ep, ep_c, lc)

        # Phase 2: from the node's top layer down to 0, beam-search + connect.
        for lc in range(min(level, self._max_level), -1, -1):
            found = self._search_layer(q, [(ep_c, ep)], self.efConstruction, lc)
            selected = self._select_neighbors_heuristic(found, self._max_degree(lc))

            for nb in selected:
                self._neighbors[node][lc].append(nb)
                self._neighbors[nb][lc].append(node)
                # Keep the neighbour's degree within budget.
                self._prune(nb, lc)

            # Descend using the closest node found on this layer.
            if found:
                best = min(found, key=lambda t: t[0])
                ep, ep_c = best[1], best[0]

        # A taller new node takes over as the global entry point.
        if level > self._max_level:
            self._max_level = level
            self._entry_point = node

        self.ntotal += 1

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(self, x, k):
        """Search the ``k`` approximate nearest neighbours of each query.

        Greedy-descends the upper layers to a good entry point, then runs a
        layer-0 beam search with ``ef = max(efSearch, k)`` and returns the top-k.
        """
        self._check_trained()
        x = _as_2d_f32(x)
        if x.shape[1] != self.d:
            raise ValueError(f"expected dim {self.d}, got {x.shape[1]}")
        nq = x.shape[0]

        D = np.empty((nq, k), dtype=np.float32)
        I = np.empty((nq, k), dtype=np.int64)

        # PERF: FAISS's HNSW uses SIMD distance kernels, cache-friendly contiguous
        #       neighbour storage, software prefetching of neighbour vectors, and
        #       (in some builds) multithreaded construction. Graph ANN trades extra
        #       memory (the neighbour lists) for fast, high-recall search. Here we
        #       keep plain python loops + numpy for clarity and process one query
        #       at a time rather than batching across queries.
        for i in range(nq):
            q = np.ascontiguousarray(x[i], dtype=np.float32)

            if self._entry_point == -1:
                # Empty index: nothing to return, pad everything.
                D[i], I[i] = pad_results([], [], k, self.metric_type)
                continue

            ep = self._entry_point
            ep_c = float(self._closeness_many(q, [ep])[0])
            for lc in range(self._max_level, 0, -1):
                ep, ep_c = self._greedy_descend(q, ep, ep_c, lc)

            ef = max(self.efSearch, k)
            found = self._search_layer(q, [(ep_c, ep)], ef, 0)

            # Sort by closeness (smaller = better) and keep the best k.
            found.sort(key=lambda t: t[0])
            best = found[:k]
            ids = [nid for _, nid in best]
            dists = [self._to_metric(c) for c, _ in best]
            D[i], I[i] = pad_results(ids, dists, k, self.metric_type)

        return D, I

    # ------------------------------------------------------------------ #
    # Reconstruction (exact -- flat storage)
    # ------------------------------------------------------------------ #
    def reconstruct(self, key):
        """Return a copy of the (d,) vector stored under id ``key`` (exact)."""
        if key < 0 or key >= len(self._data):
            raise IndexError(f"id {key} out of range [0, {len(self._data)})")
        return np.array(self._data[key], dtype=np.float32, copy=True)
