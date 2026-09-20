"""String-driven index construction for minifaiss.

Mirrors ``faiss.index_factory``: build an index from a compact description
string (e.g. ``"IVF64,PQ8"``) instead of wiring the classes together by hand.
This is the ergonomic front door most FAISS users reach for first.
"""

import re

from .metrics import METRIC_L2, METRIC_INNER_PRODUCT
from .flat import IndexFlatL2, IndexFlatIP
from .ivf import IndexIVFFlat
from .pq import IndexPQ, IndexIVFPQ
from .lsh import IndexLSH
from .hnsw import IndexHNSWFlat


def _parse_int(token, prefix):
    """Extract the trailing integer from a ``<prefix><int>`` token.

    Returns the parsed int, or raises ``ValueError`` if the suffix after
    ``prefix`` is not a plain non-negative integer.
    """
    suffix = token[len(prefix):]
    if not re.fullmatch(r"\d+", suffix):
        raise ValueError(
            f"expected an integer after '{prefix}' in token {token!r}"
        )
    return int(suffix)


def index_factory(d, description, metric_type=METRIC_L2):
    """Build an index from a FAISS-style ``description`` string.

    Mirrors ``faiss.index_factory``. The description is split on commas (each
    part whitespace-tolerant) into a coarse/refinement pipeline. Supported
    descriptions:

    ==================  ==========================================================
    Description         Result
    ==================  ==========================================================
    ``"Flat"``          :class:`~minifaiss.flat.IndexFlatL2` (or ``IndexFlatIP``
                        when ``metric_type == METRIC_INNER_PRODUCT``)
    ``"IVF<nlist>,Flat"``   :class:`~minifaiss.ivf.IndexIVFFlat` over an
                        ``IndexFlatL2`` coarse quantizer
    ``"IVF<nlist>,PQ<m>"``  :class:`~minifaiss.pq.IndexIVFPQ` (nbits=8)
    ``"PQ<m>"``         :class:`~minifaiss.pq.IndexPQ` (nbits=8)
    ``"LSH<nbits>"``    :class:`~minifaiss.lsh.IndexLSH`
    ``"HNSW<M>"``       :class:`~minifaiss.hnsw.IndexHNSWFlat`
    ==================  ==========================================================

    Parameters
    ----------
    d : int
        Vector dimensionality.
    description : str
        The factory string, e.g. ``"IVF64,PQ8"`` or ``"HNSW16"``.
    metric_type : int
        ``METRIC_L2`` or ``METRIC_INNER_PRODUCT``.

    Returns
    -------
    Index
        A freshly constructed (untrained) index.

    Raises
    ------
    ValueError
        On an empty or unrecognised description / token combination.
    """
    # PERF: real FAISS's factory composes far more than we do -- it can prepend
    #       vector transforms (OPQ rotation, PCA/PCAR dimensionality reduction,
    #       L2 normalisation) and select SIMD fast-scan code variants (e.g. the
    #       "PQ16x4fs" AVX layout) or GPU index clones. minifaiss handles only
    #       the plain coarse+encoding cases below; transforms and SIMD variants
    #       are intentionally omitted for clarity.
    d = int(d)
    parts = [p.strip() for p in description.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"empty index description: {description!r}")

    head = parts[0]

    # --- Single-token descriptions -------------------------------------- #
    if len(parts) == 1:
        if head == "Flat":
            if metric_type == METRIC_INNER_PRODUCT:
                return IndexFlatIP(d)
            return IndexFlatL2(d)
        if head.startswith("PQ"):
            m = _parse_int(head, "PQ")
            return IndexPQ(d, m, nbits=8, metric_type=metric_type)
        if head.startswith("LSH"):
            nbits = _parse_int(head, "LSH")
            return IndexLSH(d, nbits)
        if head.startswith("HNSW"):
            M = _parse_int(head, "HNSW")
            return IndexHNSWFlat(d, M, metric_type=metric_type)
        raise ValueError(f"unknown index token: {head!r}")

    # --- Two-token IVF descriptions ------------------------------------- #
    if len(parts) == 2 and head.startswith("IVF"):
        nlist = _parse_int(head, "IVF")
        tail = parts[1]
        # The IVF coarse quantizer mirrors FAISS: a flat L2 index over centroids.
        coarse = IndexFlatL2(d)
        if tail == "Flat":
            return IndexIVFFlat(coarse, d, nlist, metric_type=metric_type)
        if tail.startswith("PQ"):
            m = _parse_int(tail, "PQ")
            return IndexIVFPQ(coarse, d, nlist, m, nbits=8,
                              metric_type=metric_type)
        raise ValueError(f"unknown IVF sub-index token: {tail!r}")

    raise ValueError(f"unsupported index description: {description!r}")
