"""Base Index API for minifaiss.

Mirrors ``faiss.Index`` (the abstract base of every FAISS index). Concrete
indexes subclass :class:`Index` and implement ``add`` and ``search``; the base
handles input coercion, id bookkeeping, training state, and result padding.
"""

import numpy as np

from .metrics import METRIC_L2, METRIC_INNER_PRODUCT


def _as_2d_f32(x):
    """Coerce ``x`` to a float32 2-D array of shape (n, d).

    Mirrors FAISS's numpy interface: inputs are always float32 (n, d); a single
    1-D vector is treated as a batch of one.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def pad_results(ids, dists, k, metric_type):
    """Pad or truncate one query's already-sorted results to length ``k``.

    FAISS always returns dense ``(nq, k)`` result blocks. When a query has fewer
    than ``k`` candidates, the empty slots are filled with a sentinel id of -1
    and a "worst possible" distance: ``+inf`` for L2 (smaller-is-better) or
    ``-inf`` for inner product (larger-is-better).

    Parameters
    ----------
    ids : sequence of int
        Vector ids for this query, ALREADY sorted best-first.
    dists : sequence of float
        Distances/scores aligned with ``ids``, best-first.
    k : int
        Desired output length.
    metric_type : int
        ``METRIC_L2`` or ``METRIC_INNER_PRODUCT`` -- selects the pad value.

    Returns
    -------
    D_row : ndarray, shape (k,), dtype float32
    I_row : ndarray, shape (k,), dtype int64
    """
    ids = np.asarray(ids, dtype=np.int64).ravel()
    dists = np.asarray(dists, dtype=np.float32).ravel()

    # Truncate if we somehow have more than k (keep the best k, already sorted).
    ids = ids[:k]
    dists = dists[:k]

    n = ids.shape[0]
    pad = k - n

    if metric_type == METRIC_INNER_PRODUCT:
        pad_dist = np.float32(-np.inf)   # larger is better -> worst is -inf
    else:
        pad_dist = np.float32(np.inf)    # smaller is better -> worst is +inf

    D_row = np.empty(k, dtype=np.float32)
    I_row = np.empty(k, dtype=np.int64)
    D_row[:n] = dists
    I_row[:n] = ids
    if pad > 0:
        D_row[n:] = pad_dist
        I_row[n:] = -1
    return D_row, I_row


class Index:
    """Abstract base class for all minifaiss indexes.

    Mirrors ``faiss.Index``. Stores dimensionality ``d``, a ``metric_type``,
    the running vector count ``ntotal``, and an ``is_trained`` flag. Concrete
    subclasses implement ``add`` and ``search``; this base provides input
    coercion, the default (no-op) ``train``, ``reset``, and reconstruction
    stubs.

    Attributes
    ----------
    d : int
        Vector dimensionality.
    metric_type : int
        ``metrics.METRIC_L2`` or ``metrics.METRIC_INNER_PRODUCT``.
    ntotal : int
        Number of stored vectors (0 initially).
    is_trained : bool
        True when the index is ready to add/search.
    """

    def __init__(self, d, metric_type=METRIC_L2):
        """Create an index over ``d``-dimensional vectors with ``metric_type``."""
        self.d = int(d)
        self.metric_type = int(metric_type)
        self.ntotal = 0
        # Flat indexes need no training; subclasses that do (e.g. IVF/PQ) reset
        # this to False in their own __init__.
        self.is_trained = True

    def train(self, x):
        """Train the index on a sample ``x`` of shape (n, d).

        Default implementation is a no-op that simply marks the index trained,
        matching FAISS indexes (like ``IndexFlat``) that require no training.
        Subclasses that learn structure (quantizers, PQ codebooks) override this.
        """
        _as_2d_f32(x)  # coerce/validate shape even though we discard the result
        self.is_trained = True

    def add(self, x):
        """Add vectors ``x`` of shape (n, d), assigning ids [ntotal, ntotal+n).

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Index subclasses must implement add()")

    def search(self, x, k):
        """Search the ``k`` nearest neighbours of each query in ``x``.

        Returns ``(D, I)`` with ``D`` float32 (nq, k) distances and ``I`` int64
        (nq, k) ids. Must be implemented by subclasses.
        """
        raise NotImplementedError("Index subclasses must implement search()")

    def reset(self):
        """Remove all stored vectors, resetting ``ntotal`` to 0.

        Calls the optional subclass hook ``_reset()`` (if defined) so subclasses
        can clear their own storage without re-implementing this method.
        """
        self.ntotal = 0
        reset_hook = getattr(self, "_reset", None)
        if callable(reset_hook):
            reset_hook()

    def reconstruct(self, key):
        """Return the (d,) vector stored under id ``key`` (exact or approximate).

        Must be implemented by subclasses that keep reconstructable storage.
        """
        raise NotImplementedError("Index subclasses must implement reconstruct()")

    def reconstruct_n(self, i0, ni):
        """Return vectors for the contiguous id range [i0, i0+ni) as (ni, d).

        Convenience wrapper around :meth:`reconstruct`; subclasses may override
        with a more efficient batched version.
        """
        rows = [self.reconstruct(i0 + i) for i in range(ni)]
        return np.asarray(rows, dtype=np.float32).reshape(ni, self.d)

    def _check_trained(self):
        """Raise ``RuntimeError`` if the index is not yet trained."""
        if not self.is_trained:
            raise RuntimeError("Index must be trained before add/search")
