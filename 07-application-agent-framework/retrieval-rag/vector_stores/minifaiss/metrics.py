"""Distance / similarity kernels and top-k selection for minifaiss.

Mirrors the low-level distance routines FAISS uses internally (see
``faiss/utils/distances.cpp``). We keep the math explicit and lean on numpy's
BLAS backing for the one matrix multiply that dominates runtime.
"""

import numpy as np

# Metric type codes. These match FAISS's numeric values so downstream code can
# use them interchangeably with the FAISS convention.
METRIC_INNER_PRODUCT = 0
METRIC_L2 = 1


def _as_2d_f32(x):
    """Coerce ``x`` to a contiguous float32 2-D array of shape (n, d)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def l2_sqr_distances(queries, database):
    """Return the SQUARED L2 distance between every query and every db vector.

    Parameters
    ----------
    queries : array-like, shape (nq, d)
    database : array-like, shape (nb, d)

    Returns
    -------
    ndarray, shape (nq, nb), dtype float32
        ``dist[i, j] = ||queries[i] - database[j]||^2`` (squared, like FAISS).

    Computed via the identity
        ``||q - b||^2 = ||q||^2 + ||b||^2 - 2 * q . b^T``
    so the heavy lifting is a single matrix multiply.
    """
    q = _as_2d_f32(queries)
    b = _as_2d_f32(database)

    # ||q||^2 as a column vector and ||b||^2 as a row vector.
    q_sq = np.einsum("ij,ij->i", q, q)[:, None]   # (nq, 1)
    b_sq = np.einsum("ij,ij->i", b, b)[None, :]   # (1, nb)

    # PERF: this single matmul (q @ b.T) is where FAISS spends most of its time.
    #       FAISS dispatches it to a BLAS GEMM kernel, which is SIMD-vectorized
    #       and multi-threaded (OpenMP), and on faiss-gpu runs as a CUDA kernel.
    #       Here we simply rely on numpy's BLAS backend for the same operation.
    cross = q @ b.T                                # (nq, nb)

    dist = q_sq + b_sq - 2.0 * cross

    # Floating-point round-off can push exact-zero distances slightly negative;
    # clip those tiny negatives back to 0 so callers never see negative "squares".
    np.maximum(dist, 0.0, out=dist)
    return dist.astype(np.float32, copy=False)


def inner_products(queries, database):
    """Return the inner product between every query and every db vector.

    Parameters
    ----------
    queries : array-like, shape (nq, d)
    database : array-like, shape (nb, d)

    Returns
    -------
    ndarray, shape (nq, nb), dtype float32
        ``ip[i, j] = queries[i] . database[j]``.
    """
    q = _as_2d_f32(queries)
    b = _as_2d_f32(database)

    # PERF: same story as l2_sqr_distances -- FAISS routes this q @ b.T through
    #       BLAS GEMM (SIMD + OpenMP threads, or a GPU kernel on faiss-gpu).
    ip = q @ b.T
    return ip.astype(np.float32, copy=False)


def topk(scores, k, largest):
    """Select the top-``k`` entries of each row of ``scores``.

    Parameters
    ----------
    scores : ndarray, shape (nq, nb)
        Per-query scores (e.g. distances or inner products).
    k : int
        Number of results wanted per row. Capped at the number of columns.
    largest : bool
        If True, keep the LARGEST scores (inner-product ranking).
        If False, keep the SMALLEST scores (L2 ranking).

    Returns
    -------
    values : ndarray, shape (nq, kk), dtype float32
    indices : ndarray, shape (nq, kk), dtype int64
        Column indices of the selected entries. Each row is sorted best-first
        (descending for ``largest=True``, ascending for ``largest=False``).
        ``kk = min(k, nb)``.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    nq, nb = scores.shape
    kk = min(k, nb)

    # PERF: FAISS maintains a per-thread max-heap (or min-heap) of size k and
    #       streams candidates through it in O(nb log k). np.argpartition is our
    #       analogue: it does an O(nb) partial selection instead of a full sort,
    #       so we only pay a full sort on the kk selected entries.
    if largest:
        # Partition so the kk largest land in the last kk slots, then sort desc.
        part = np.argpartition(scores, nb - kk, axis=1)[:, nb - kk:]
    else:
        # Partition so the kk smallest land in the first kk slots, then sort asc.
        part = np.argpartition(scores, kk - 1, axis=1)[:, :kk]

    rows = np.arange(nq)[:, None]
    part_scores = scores[rows, part]

    # Order the kk selected entries within each row.
    order = np.argsort(part_scores, axis=1)
    if largest:
        order = order[:, ::-1]

    indices = np.take_along_axis(part, order, axis=1).astype(np.int64)
    values = np.take_along_axis(part_scores, order, axis=1).astype(np.float32)
    return values, indices
