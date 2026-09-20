"""Vector store -- the semantic index behind local search.

GraphRAG embeds entities, text units and community reports, and looks them up
by vector similarity to a query. That lookup is exactly the job of the
``minifaiss`` package next door (or, in production GraphRAG, LanceDB / Azure AI
Search). We keep a trivial brute-force store as the default so the package is
self-contained, and define a small protocol so you can drop in a real index.

PERF: ``BruteForceVectorStore.query`` scans every vector (O(n*d) per query).
That is fine for a demo with a few hundred entities. Swap it for an approximate
index -- e.g. ``minifaiss.IndexHNSWFlat`` or ``IndexIVFFlat`` -- to scale; see
``MiniFaissVectorStore`` below for a ready adapter, and the vector-DB note in
the README.
"""

import numpy as np


def cosine_normalize(vectors):
    """L2-normalize rows so that inner product == cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


class VectorStore:
    """Minimal add/query interface. Ids are arbitrary strings."""

    def add(self, ids, vectors):
        raise NotImplementedError

    def query(self, vector, k=10):
        """Return ``[(id, score), ...]`` best-first (score = cosine similarity)."""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class BruteForceVectorStore(VectorStore):
    """Exact cosine-similarity search by scanning all stored vectors."""

    def __init__(self):
        self._ids = []
        self._matrix = None  # (n, d) float32, L2-normalized

    def add(self, ids, vectors):
        vectors = cosine_normalize(vectors)
        self._ids.extend(ids)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def query(self, vector, k=10):
        if not self._ids:
            return []
        q = cosine_normalize(vector)[0]
        # PERF: one dense mat-vec here; a real ANN index avoids touching every row.
        scores = self._matrix @ q
        k = min(k, len(self._ids))
        # argpartition for top-k, then sort just those k descending.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top]

    def __len__(self):
        return len(self._ids)


class MiniFaissVectorStore(VectorStore):
    """Adapter backing the store with a ``minifaiss`` index (optional).

    Demonstrates the real integration point: identical interface, but nearest
    neighbours come from an ANN index instead of a full scan. Pass any inner-
    product minifaiss index (``IndexFlatIP``, ``IndexHNSWFlat``, ...). Vectors
    are L2-normalized so inner product ranks by cosine similarity.
    """

    def __init__(self, index):
        self._index = index  # a minifaiss Index using METRIC_INNER_PRODUCT
        self._ids = []

    def add(self, ids, vectors):
        vectors = cosine_normalize(vectors)
        self._index.add(vectors)
        self._ids.extend(ids)

    def query(self, vector, k=10):
        if not self._ids:
            return []
        q = cosine_normalize(vector)
        k = min(k, len(self._ids))
        D, I = self._index.search(q, k)
        return [(self._ids[i], float(s)) for s, i in zip(D[0], I[0]) if i >= 0]

    def __len__(self):
        return len(self._ids)
