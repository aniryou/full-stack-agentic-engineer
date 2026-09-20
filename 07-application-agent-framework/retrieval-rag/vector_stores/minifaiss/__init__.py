"""minifaiss -- an educational reimplementation of the core ideas of FAISS.

minifaiss rebuilds the central concepts of FAISS (Facebook AI Similarity
Search) in pure Python + numpy, for LEARNING rather than performance. Every
index is written plainly so a reader can follow the algorithm, and each place
where real FAISS reaches for a performance technique (BLAS/GEMM matrix
multiplies, SIMD vectorization, OpenMP threads, GPU kernels, PQ fast-scan
lookup tables, SIMD popcount, contiguous/float16/int8 memory layout,
memory-mapped on-disk indexes, IVF pruning) is flagged with a ``# PERF:``
comment. Grep the source for ``# PERF:`` to find every one, and read the
README's "Where performance lives" section for the guided tour.

Public surface (mirrors the ``faiss`` module):

* metrics: ``METRIC_L2``, ``METRIC_INNER_PRODUCT``
* base + clustering: :class:`Index`, :class:`Kmeans`
* flat / exact: :class:`IndexFlat`, :class:`IndexFlatL2`, :class:`IndexFlatIP`
* inverted file: :class:`IndexIVFFlat`
* product quantization: :class:`ProductQuantizer`, :class:`IndexPQ`,
  :class:`IndexIVFPQ`
* other ANN: :class:`IndexLSH`, :class:`IndexHNSWFlat`
* construction helper: :func:`index_factory`
"""

__version__ = "0.1.0"

from .metrics import METRIC_L2, METRIC_INNER_PRODUCT
from .base import Index
from .kmeans import Kmeans
from .flat import IndexFlat, IndexFlatL2, IndexFlatIP
from .ivf import IndexIVFFlat
from .pq import ProductQuantizer, IndexPQ, IndexIVFPQ
from .lsh import IndexLSH
from .hnsw import IndexHNSWFlat
from .index_factory import index_factory

__all__ = [
    "__version__",
    "METRIC_L2",
    "METRIC_INNER_PRODUCT",
    "Index",
    "Kmeans",
    "IndexFlat",
    "IndexFlatL2",
    "IndexFlatIP",
    "IndexIVFFlat",
    "ProductQuantizer",
    "IndexPQ",
    "IndexIVFPQ",
    "IndexLSH",
    "IndexHNSWFlat",
    "index_factory",
]
