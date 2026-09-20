# minifaiss

An **educational** reimplementation of the core ideas of
[FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity
Search) in pure Python + numpy. It exists to be *read*, not to be fast: every
algorithm is written plainly, and every place where real FAISS reaches for a
performance technique is flagged with a `# PERF:` comment in the source.

> **The library** (`minifaiss/`) is standard library + numpy only — no `faiss`,
> `scipy`, `sklearn`, or `torch`. The optional real-data demo additionally uses
> `nltk` (only to fetch the public-domain Project Gutenberg corpus) and a tiny
> homemade TF-IDF vectorizer; neither is part of the library.

---

## What is FAISS, and what does minifaiss cover?

FAISS is a C++/CUDA library for similarity search over dense vectors: given a
query vector, find its nearest neighbours among millions or billions of stored
vectors, either exactly or approximately. It gets its speed from BLAS, SIMD,
multithreading, GPUs, and compact vector encodings.

minifaiss keeps the *concepts* and drops the speed. Each class mirrors a real
FAISS class:

| minifaiss class    | faiss class          | Core idea |
|--------------------|----------------------|-----------|
| `IndexFlat`        | `faiss.IndexFlat`    | Store every vector; exact brute-force scan. |
| `IndexFlatL2`      | `faiss.IndexFlatL2`  | Flat index, squared-L2 metric. |
| `IndexFlatIP`      | `faiss.IndexFlatIP`  | Flat index, inner-product metric. |
| `IndexIVFFlat`     | `faiss.IndexIVFFlat` | Partition space into cells (k-means); scan only nearby cells. |
| `ProductQuantizer` | `faiss.ProductQuantizer` | Split vectors into sub-vectors; quantize each to a small codebook. |
| `IndexPQ`          | `faiss.IndexPQ`      | Store PQ codes only; search by asymmetric distance computation (ADC). |
| `IndexIVFPQ`       | `faiss.IndexIVFPQ`   | IVF cells + PQ codes on residuals; FAISS's workhorse index. |
| `IndexLSH`         | `faiss.IndexLSH`     | Random-hyperplane binary codes; Hamming-distance search. |
| `IndexHNSWFlat`    | `faiss.IndexHNSWFlat`| Multi-layer navigable-small-world graph; greedy + beam search. |
| `Kmeans`           | `faiss.Kmeans`       | Lloyd's algorithm; trains IVF/PQ quantizers. |
| `index_factory`    | `faiss.index_factory`| Build an index from a description string like `"IVF64,PQ8"`. |

**Distance conventions (identical to FAISS):**
- `METRIC_L2`: distances are **squared** L2; smaller is better (ascending).
- `METRIC_INNER_PRODUCT`: larger is better (descending).

---

## Quickstart

```python
import numpy as np
from minifaiss import IndexFlatL2

d = 32
rng = np.random.default_rng(0)
database = rng.normal(size=(1000, d)).astype(np.float32)
queries = rng.normal(size=(5, d)).astype(np.float32)

index = IndexFlatL2(d)      # exact, squared-L2
index.add(database)         # ids assigned 0..999
D, I = index.search(queries, k=10)
# D: (5, 10) float32 squared distances, ascending
# I: (5, 10) int64 neighbour ids (-1 pads empty slots)
```

Build the same thing from a factory string:

```python
from minifaiss import index_factory
index = index_factory(d, "Flat")          # -> IndexFlatL2
ivfpq = index_factory(d, "IVF64,PQ8")      # -> IndexIVFPQ
ivfpq.train(database); ivfpq.add(database); ivfpq.nprobe = 8
```

Indexes that learn structure (`IVF*`, `PQ*`) must be `train`-ed before `add`;
`Flat`, `LSH`, and `HNSW` need no training.

---

## The indexes, one at a time

Each index is a different point on the **space / speed / recall** trade-off.

### Flat (`flat.py`)
Stores every vector uncompressed and compares the query against all of them.
Exact and simple, but O(n·d) per query and 4·d bytes/vector. It is the ground
truth other indexes are measured against. **Space:** high. **Speed:** slow.
**Recall:** perfect.

### IVF — inverted file (`ivf.py`)
k-means carves the space into `nlist` Voronoi cells. Each vector lives in its
nearest cell; a query only scans the `nprobe` closest cells. Larger `nprobe`
→ higher recall, slower search. Still stores full vectors. **Space:** high.
**Speed:** fast (scans `nprobe/nlist` of the data). **Recall:** tunable, near
perfect at `nprobe = nlist`.

### PQ — product quantization (`pq.py`)
Splits each vector into `m` sub-vectors and quantizes each against its own
256-entry codebook, so a vector becomes just `m` bytes. Search uses *asymmetric
distance computation* (ADC): build a small per-query lookup table, then score
each stored code by summing `m` table look-ups. **Space:** tiny (`m` bytes).
**Speed:** fast. **Recall:** lossy (depends on `m`).

### IVFPQ (`pq.py`)
FAISS's most-used index: IVF cells for pruning + PQ codes for compactness.
Crucially, PQ encodes the *residual* `x - centroid`, which has smaller variance
and quantizes more accurately. **Space:** tiny. **Speed:** fast. **Recall:**
good for the byte budget — the sweet spot for billion-scale search.

### LSH — locality-sensitive hashing (`lsh.py`)
Projects each vector onto `nbits` random hyperplanes and keeps only the sign
bits. Search is a Hamming-distance scan over packed bit codes. Single-table LSH
(as here) is the weakest recall of the bunch; real systems use many bits and
multiple tables. **Space:** tiny (`nbits/8` bytes). **Speed:** fast. **Recall:**
low unless heavily tuned.

### HNSW — hierarchical navigable small world (`hnsw.py`)
Builds a layered proximity graph: sparse upper layers act as express lanes, and
layer 0 holds everyone. Search greedy-descends the upper layers, then beam-
searches layer 0. Stores full vectors plus the graph. **Space:** high (vectors +
edges). **Speed:** very fast. **Recall:** high, tuned by `efSearch`.

---

## Where performance lives

The whole point of minifaiss is to show *where the speed would come from* in
real FAISS. Every such spot is marked with a comment beginning `# PERF:`.

**Find them all:**

```bash
grep -rn "# PERF:" minifaiss/
```

| Real-FAISS optimization | What it does | minifaiss file · `# PERF:` location |
|-------------------------|--------------|-------------------------------------|
| **BLAS / GEMM matmul**  | Batched distance math as one big matrix multiply — the dominant cost. | `metrics.py` (`l2_sqr_distances`, `inner_products`); also flagged in `flat.py`, `ivf.py`, `lsh.py`, `kmeans.py`. We rely on numpy's BLAS. |
| **SIMD vectorization**  | Distance/popcount kernels use vector CPU instructions. | Called out alongside the GEMM notes in `metrics.py`, and in `lsh.py` (`_hamming`). |
| **OpenMP threads**      | Parallel scan across queries / inverted lists / k-means iterations. | `metrics.py`, `ivf.py` (`search`), `kmeans.py` (`train`). |
| **GPU kernels** (faiss-gpu) | CUDA GEMM and index kernels. | Noted with the GEMM `# PERF:` comments in `metrics.py`, `flat.py`. |
| **PQ fast-scan / SIMD LUT** | Pack codes so ADC look-ups run in SIMD shuffle registers over many codes at once. | `pq.py` (`distance_table`, `_adc_scan`). |
| **SIMD popcount**       | Hardware `POPCNT` / byte-shuffle popcount for Hamming distance. | `lsh.py` (`_POPCOUNT_TABLE`, `_hamming`). |
| **Contiguous memory layout** | Cache-friendly contiguous buffers per list. | `flat.py` (`add`), `ivf.py` (`search`), `hnsw.py` (`search`). |
| **float16 / int8 storage** | Store vectors at reduced precision to cut bandwidth. | `flat.py` (`add`). |
| **PQ code layout (m bytes/vec)** | uint8 codes, one byte per subspace. | `pq.py` (`compute_codes`). |
| **IVF pruning (nprobe/nlist)** | Skip cells whose centroid is far from the query. | `ivf.py` (`search`), `pq.py` (`IndexIVFPQ.search`). |
| **Result heap (size-k)**| Per-thread max-heap instead of a full sort. | `metrics.py` (`topk`), `lsh.py` (`search`). |
| **mmap / on-disk indexes** | Memory-map huge indexes instead of loading them. | Conceptually replaces our in-RAM Python lists (see `ivf.py` / `pq.py` inverted lists); not implemented. |
| **Factory transforms / SIMD variants** | OPQ, PCA, `PQxxfs` fast-scan variants, GPU clones. | `index_factory.py` — omitted here. |

---

## Real-data demo: semantic search over Project Gutenberg

`demo_gutenberg.py` runs vector search over real public-domain books. It:

1. loads passages from the [NLTK Gutenberg corpus](https://www.nltk.org/nltk_data/)
   (`gutenberg_corpus.py`),
2. turns each passage into a vector with a small homemade **TF-IDF** vectorizer
   (`tfidf.py`), L2-normalized so **inner product = cosine similarity**,
3. does natural-language **semantic search** (exact, `IndexFlatIP`), then
4. measures each approximate index's **recall@10, memory, and build/search
   time** against that exact baseline.

```bash
# one-time: fetch the corpus into the venv
./.venv/bin/python -m nltk.downloader gutenberg

# run it (deterministic, ~2s; add --no-hnsw to skip the slowest build)
./.venv/bin/python demo_gutenberg.py
```

Example (1500 passages, d=256, cosine):

```
Query: "murder of the king and the bloody crown"
  #1  cos=0.572  [Shakespeare — Hamlet]
  #2  cos=0.548  [Shakespeare — Hamlet]
  #3  cos=0.547  [Shakespeare — Macbeth]

index                                  bytes/vec  recall@10  build s search s
-----------------------------------------------------------------------------
IndexFlatIP (exact)                         1024      1.000     0.00    0.002
IndexIVFFlat(nlist=48,nprobe=8)             1024      0.765     0.06    0.014
IndexPQ(m=8)                                   8      0.611     0.20    0.006
IndexIVFPQ(nlist=48,m=8,nprobe=8)              8      0.570     0.25    0.036
IndexLSH(nbits=256)                           32      0.411     0.00    0.011
IndexHNSWFlat(M=16)                         1024      0.979     0.59    0.057
```

The trade-off reads straight off the table: `IndexFlatIP` and `HNSW` keep the
most recall but store 1024 bytes/vector; `IndexIVFPQ` holds ~0.57 recall at just
**8 bytes/vector** (128× smaller); single-table `IndexLSH` is cheapest but
weakest.

> **Why TF-IDF, not real embeddings?** To keep the repo dependency-light and
> offline. TF-IDF is *lexical*, so a query for "white rabbit" can match Moby
> Dick's "white … hurried" passages — the words overlap even though the meaning
> doesn't. A neural embedding model (sentence-transformers, OpenAI/Cohere, a
> local model) would capture meaning; **you would feed its vectors into exactly
> these same minifaiss indexes.** FAISS itself never embeds text — it only
> indexes vectors you already have.

## Running the synthetic demo and the tests

```bash
# Synthetic demo: clustered Gaussian blobs, recall@10 table vs exact search.
./.venv/bin/python demo.py

# Tests:
./.venv/bin/python -m pytest tests -q
```

The synthetic `demo.py` is deterministic (seeded). Example output:

```
index                               bytes/vec   recall@10
---------------------------------  ----------  ----------
IndexFlatL2                               128       1.000
IndexIVFFlat(nlist=64,nprobe=8)           128       0.999
IndexPQ(m=8)                                8       0.548
IndexIVFPQ(nlist=64,m=8,nprobe=8)           8       0.755
IndexLSH(nbits=64)                          8       0.108
IndexHNSWFlat(M=16)                       128       0.955
```
