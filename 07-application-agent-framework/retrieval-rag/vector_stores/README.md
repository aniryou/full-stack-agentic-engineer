# vector_stores — vector search and GraphRAG, rebuilt from scratch in numpy

After this lab, you can do these three things:

- Explain the trade-off between memory, speed and recall in each index family of FAISS (flat, IVF, PQ, IVFPQ,
  LSH, HNSW).
- Measure that trade-off yourself.
- Trace a GraphRAG pipeline from the chunks to an answer from the community summaries.

## Start here

1. Install the lab and run the tests ("Run it"). There are 50 tests. They take about 25 s on a laptop CPU.
2. Run `python demo.py`. It prints a recall@10 table for each index on seeded synthetic data (~20 s).
3. Read [`minifaiss/hnsw.py`](minifaiss/hnsw.py) or [`minifaiss/pq.py`](minifaiss/pq.py) together with the
   applicable section of the [vector databases primer](../vector-databases-primer.md). Then read
   [`minigraphrag/`](minigraphrag/README.md).

## What you get

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`minifaiss/`](minifaiss/) + `demo.py` | Build each FAISS index family, adjust its parameters, and read the recall/memory table. | 2–3 h | T0 |
| `demo_gutenberg.py` | Run the same indexes on real text (TF-IDF over Project Gutenberg). | 30 min | T0 (downloads the corpus one time) |
| [`minigraphrag/`](minigraphrag/README.md) + `demo_graphrag.py` | Trace GraphRAG: extraction, graph, communities, local search and global search. | 2 h | T0 (offline mock model) |
| `tests/` | Compare each index with exact search. | — | T0 |

T0 is a laptop or a Colab CPU, at no cost. It needs no GPU and no API key.

## Run it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt       # numpy and pytest; nltk only for the Gutenberg demo
python -m pytest tests -q             # 49 passed, 1 skipped (the corpus loader test runs once the corpus is fetched)
python demo.py                        # synthetic recall@10 table
python gutenberg_corpus.py            # one-time corpus fetch; non-interactive (nltk.download(quiet=True))
python demo_gutenberg.py              # semantic-ish search + recall/memory/time per index on real passages
```

Do not use `python -m nltk.downloader gutenberg`. When a download fails, this command shows the prompt "Retry?
[n/y/e]". In a non-interactive shell, the command then crashes. Recent NLTK releases also do not permit a download
through an HTTP(S) proxy. If you trust your proxy, set `NLTK_ALLOW_PROXIED_URLOPEN=1` for the download (examined
with NLTK 3.10.3 on 2026-09-26, verify).

---

## minifaiss

minifaiss is an **educational** reimplementation of the core ideas of
[FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) in pure Python and numpy. Its
purpose is that you *read* it, not that it is fast. The code shows each algorithm in a plain form. At each location
where the real FAISS uses a performance technique, a `# PERF:` comment in the source marks it.

> **The library** (`minifaiss/`) uses only the standard library and numpy. It does not use `faiss`, `scipy`,
> `sklearn` or `torch`. The optional demo on real data also uses `nltk`, only to download the public-domain Project
> Gutenberg corpus. It also uses a small TF-IDF vectorizer that we wrote by hand. These two are not part of the
> library.

---

## What is FAISS, and what does minifaiss cover?

FAISS is a C++/CUDA library for similarity search over dense vectors. For a query vector, it finds the nearest
neighbours among millions or billions of stored vectors. The search is exact or approximate. Its speed comes from
BLAS, SIMD, multiple threads, GPUs and compact codes for the vectors.

minifaiss keeps the *concepts*, but not the speed. Each class matches a real FAISS class:

| minifaiss class    | faiss class          | Core idea |
|--------------------|----------------------|-----------|
| `IndexFlat`        | `faiss.IndexFlat`    | Stores each vector and does an exact brute-force scan. |
| `IndexFlatL2`      | `faiss.IndexFlatL2`  | Flat index, squared-L2 metric. |
| `IndexFlatIP`      | `faiss.IndexFlatIP`  | Flat index, inner-product metric. |
| `IndexIVFFlat`     | `faiss.IndexIVFFlat` | Divides the space into cells (k-means) and scans only the near cells. |
| `ProductQuantizer` | `faiss.ProductQuantizer` | Divides each vector into sub-vectors and quantizes each sub-vector to a small codebook. |
| `IndexPQ`          | `faiss.IndexPQ`      | Stores only the PQ codes and searches with asymmetric distance computation (ADC). |
| `IndexIVFPQ`       | `faiss.IndexIVFPQ`   | IVF cells and PQ codes on the residuals. This is the main general-purpose index of FAISS. |
| `IndexLSH`         | `faiss.IndexLSH`     | Binary codes from random hyperplanes, Hamming-distance search. |
| `IndexHNSWFlat`    | `faiss.IndexHNSWFlat`| A multi-layer navigable-small-world graph, with a greedy search and then a beam search. |
| `Kmeans`           | `faiss.Kmeans`       | Uses Lloyd's algorithm to train the IVF/PQ quantizers. |
| `index_factory`    | `faiss.index_factory`| Builds an index from a description string, for example `"IVF64,PQ8"`. |

**Distance conventions (identical to FAISS):**
- `METRIC_L2`: each distance is a **squared** L2 distance. A smaller distance is better (order: smallest first).
- `METRIC_INNER_PRODUCT`: a larger value is better (order: largest first).

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

Build the same index from a factory string:

```python
from minifaiss import index_factory
index = index_factory(d, "Flat")          # -> IndexFlatL2
ivfpq = index_factory(d, "IVF64,PQ8")      # -> IndexIVFPQ
ivfpq.train(database); ivfpq.add(database); ivfpq.nprobe = 8
```

You must call `train` on an index that learns a structure (`IVF*`, `PQ*`) before you call `add`. `Flat`, `LSH` and
`HNSW` need no training.

---

## The indexes, one at a time

Each index is a different point on the **space / speed / recall** trade-off.

### Flat (`flat.py`)
This index stores each vector without compression and compares the query with all of them. It is exact and simple.
But it costs O(n·d) per query and 4·d bytes/vector. It is the ground truth. We measure the other indexes against it.

**Space:** high. **Speed:** slow. **Recall:** perfect.

### IVF — inverted file (`ivf.py`)
k-means divides the space into `nlist` Voronoi cells. Each vector goes into its nearest cell. A query scans only the
`nprobe` nearest cells. A larger `nprobe` gives a higher recall and a slower search. The index still stores the full
vectors.

**Space:** high. **Speed:** fast (it scans `nprobe/nlist` of the data). **Recall:** adjustable, near perfect at
`nprobe = nlist`.

### PQ — product quantization (`pq.py`)
This index divides each vector into `m` sub-vectors. It quantizes each sub-vector against its own 256-entry
codebook. Thus a vector becomes only `m` bytes. The search uses *asymmetric distance computation* (ADC). First, it
builds a small lookup table for each query. Then it gives each stored code a score: the sum of `m` table look-ups.

**Space:** small (`m` bytes). **Speed:** fast. **Recall:** lossy (it depends on `m`).

### IVFPQ (`pq.py`)
This is the most-used index of FAISS. It uses IVF cells to prune the search and PQ codes to make the storage
compact. The important point is that PQ encodes the *residual* `x - centroid`. The residual has a smaller variance,
thus PQ quantizes it more accurately.

**Space:** small. **Speed:** fast. **Recall:** good for the byte budget. This is the best balance for billion-scale
search.

### LSH — locality-sensitive hashing (`lsh.py`)
This index projects each vector onto `nbits` random hyperplanes and keeps only the sign bits. The search is a
Hamming-distance scan over packed bit codes. Single-table LSH (as in this lab) has the lowest recall of all these
indexes. Systems in production use many bits and many tables.

**Space:** small (`nbits/8` bytes). **Speed:** fast. **Recall:** low if you do not adjust its parameters a lot.

### HNSW — hierarchical navigable small world (`hnsw.py`)
This index builds a layered proximity graph. The sparse upper layers are express lanes, and layer 0 holds all the
nodes. Each node keeps up to `M` neighbours in each upper layer and `M0 = 2M` on layer 0, as in the HNSW paper and
`faiss.IndexHNSWFlat`. The search goes down the upper layers with a greedy search. Then it does
a beam search on layer 0. The index stores the full vectors and the graph.

**Space:** high (vectors and edges). **Speed:** very fast. **Recall:** high, adjusted with `efSearch`.

---

## Where performance lives

The main purpose of minifaiss is to show *where the speed comes from* in the real FAISS. A comment that starts with
`# PERF:` marks each of these locations.

**Find them all:**

```bash
grep -rn "# PERF:" minifaiss/
```

| Real-FAISS optimization | What it does | minifaiss file · `# PERF:` location |
|-------------------------|--------------|-------------------------------------|
| **BLAS / GEMM matmul**  | It does the batched distance math as one large matrix multiply. This is the largest cost. | `metrics.py` (`l2_sqr_distances`, `inner_products`). `flat.py`, `ivf.py`, `lsh.py` and `kmeans.py` also have this mark. We use the BLAS of numpy. |
| **SIMD vectorization**  | Distance/popcount kernels use vector CPU instructions. | Next to the GEMM notes in `metrics.py`, and in `lsh.py` (`_hamming`). |
| **OpenMP threads**      | A parallel scan across queries, inverted lists or k-means iterations. | `metrics.py`, `ivf.py` (`search`), `kmeans.py` (`train`). |
| **GPU kernels** (faiss-gpu) | CUDA GEMM and index kernels. | In the GEMM `# PERF:` comments in `metrics.py`, `flat.py`. |
| **PQ fast-scan / SIMD LUT** | It packs the codes, so the ADC look-ups run in SIMD shuffle registers over many codes at the same time. | `pq.py` (`distance_table`, `_adc_scan`). |
| **SIMD popcount**       | Hardware `POPCNT` / byte-shuffle popcount for Hamming distance. | `lsh.py` (`_POPCOUNT_TABLE`, `_hamming`). |
| **Contiguous memory layout** | Cache-friendly contiguous buffers for each list. | `flat.py` (`add`), `ivf.py` (`search`), `hnsw.py` (`search`). |
| **float16 / int8 storage** | It stores the vectors at a lower precision to decrease the bandwidth. | `flat.py` (`add`). |
| **PQ code layout (m bytes/vec)** | uint8 codes, one byte for each subspace. | `pq.py` (`compute_codes`). |
| **IVF pruning (nprobe/nlist)** | It does not scan the cells whose centroid is far from the query. | `ivf.py` (`search`), `pq.py` (`IndexIVFPQ.search`). |
| **Result heap (size-k)**| A max-heap for each thread, not a full sort. | `metrics.py` (`topk`), `lsh.py` (`search`). |
| **mmap / on-disk indexes** | It maps large indexes into memory and does not load them. | In concept, it replaces our in-RAM Python lists (see the inverted lists in `ivf.py` / `pq.py`). minifaiss does not implement it. |
| **Factory transforms / SIMD variants** | OPQ, PCA, `PQxxfs` fast-scan variants, GPU clones. | `index_factory.py`. minifaiss does not include them. |

---

## Real-data demo: semantic search over Project Gutenberg

`demo_gutenberg.py` does a vector search over real public-domain books. It does these steps:

1. It loads passages from the [NLTK Gutenberg corpus](https://www.nltk.org/nltk_data/)
   (`gutenberg_corpus.py`).
2. It changes each passage into a vector with a small **TF-IDF** vectorizer that we wrote by hand (`tfidf.py`). It
   L2-normalizes the vectors, so **inner product = cosine similarity**.
3. It does a natural-language **semantic search** (exact, `IndexFlatIP`).
4. Then it measures the **recall@10, memory, and build/search time** of each approximate index against that exact
   baseline.

```bash
# one-time: fetch the corpus (non-interactive; see "Run it" if you are behind a proxy)
python gutenberg_corpus.py

# run it (deterministic; add --no-hnsw to skip the slowest build)
python demo_gutenberg.py
```

Example output (1500 passages, d=256, cosine). The recall is deterministic. The times come from a shared 4-vCPU
machine on 2026-09-26, and on your machine they will be different:

```
Query: "murder of the king and the bloody crown"
  #1  cos=0.572  [Shakespeare — Hamlet]
  #2  cos=0.548  [Shakespeare — Hamlet]
  #3  cos=0.547  [Shakespeare — Macbeth]

index                                  bytes/vec  recall@10  build s search s
-----------------------------------------------------------------------------
IndexFlatIP (exact)                         1024      1.000     0.00    0.022
IndexIVFFlat(nlist=48,nprobe=8)             1024      0.765     0.82    0.183
IndexPQ(m=8)                                   8      0.611     4.77    0.368
IndexIVFPQ(nlist=48,m=8,nprobe=8)              8      0.570     6.65    0.088
IndexLSH(nbits=256)                           32      0.407     0.01    0.173
IndexHNSWFlat(M=16)                         1024      0.985     1.69    0.212
```

The table shows the trade-off directly:

- `IndexFlatIP` and `HNSW` keep the most recall, but they store 1024 bytes/vector.
- `IndexIVFPQ` keeps ~0.57 recall at only **8 bytes/vector** (128× smaller).
- The single-table `IndexLSH` has the lowest cost, but also the lowest recall.

> **Why TF-IDF, not real embeddings?** The reason is to keep the repo dependency-light and offline. TF-IDF is
> *lexical*. Thus a query for "white rabbit" can match the "white … hurried" passages of Moby Dick. The texts have
> words in common, but not the meaning.
>
> If you use a neural embedding model (sentence-transformers, OpenAI/Cohere, a local model), it captures the
> meaning. **Then you put its vectors into exactly these same minifaiss indexes.** FAISS itself never embeds text. It
> only indexes the vectors that you already have.

## Running the synthetic demo and the tests

```bash
# Synthetic demo: clustered Gaussian blobs, recall@10 table vs exact search.
python demo.py

# Tests:
python -m pytest tests -q
```

The synthetic `demo.py` is deterministic (seeded). Example output:

```
index                               bytes/vec   recall@10
---------------------------------  ----------  ----------
IndexFlatL2                               128       1.000
IndexIVFFlat(nlist=64,nprobe=8)           128       0.999
IndexPQ(m=8)                                8       0.535
IndexIVFPQ(nlist=64,m=8,nprobe=8)           8       0.755
IndexLSH(nbits=64)                          8       0.107
IndexHNSWFlat(M=16)                       128       0.969
```
