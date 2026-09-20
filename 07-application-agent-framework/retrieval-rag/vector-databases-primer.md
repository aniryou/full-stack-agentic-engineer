# Vector Databases: A Comprehensive Primer

*August 2026*

This primer goes from first principles to production. It is written for engineers and architects who already know what a machine-learning model is and want a complete mental model of vector search: why it exists, how the indexes work, what turns an index into a database, how to build retrieval that actually performs, how the vendor landscape is shaped, and where it is heading.

If you already know what an embedding is, skip to Part 2. Parts 3 and 4 are the heart of the document: what separates a database from an index library, and how to build retrieval that works. Part 5 maps the landscape and gives a decision framework.

**Contents**

- Part 1 — Foundations: why vector databases exist, embeddings, metrics, the nearest-neighbor problem
- Part 2 — Indexing: index families, tuning knobs, memory sizing
- Part 3 — From index to database: mutability, freshness, filtering, hybrid search, scale-out
- Part 4 — Building retrieval systems: the pipeline, embedding strategy, evaluation
- Part 5 — The landscape: taxonomy, decision framework, use-case patterns
- Part 6 — Operations, pitfalls, and trends
- Appendix A — Glossary; Appendix B — Further reading

---

## Part 1 — Foundations

### 1. Why vector databases exist

Traditional databases answer questions about exact values and ranges: `WHERE customer_id = 42`, `WHERE price < 100`, `WHERE name LIKE 'Jo%'`. They are built on B-trees, hash indexes, and inverted indexes — structures that exploit ordering or exact token matches.

A whole class of modern questions has no exact-match formulation: "find documents that mean roughly the same thing as this query," "find images that look like this one," "find products this user would like," "have we seen a support ticket like this before?" Neural networks answer these by mapping objects into a high-dimensional vector space where geometric proximity approximates semantic similarity. The database question becomes: given a query vector, find the *k* stored vectors closest to it — nearest-neighbor search.

Nearest-neighbor search in hundreds or thousands of dimensions defeats every classic index. There is no total ordering to exploit (B-trees fail), no exact key to hash (hash indexes fail), and no discrete tokens to invert (inverted indexes fail). You are left with brute force — comparing the query against every vector — which is O(N·d) per query: fine for 100k vectors, not for 100 million at 1,000 queries per second.

Vector databases exist to solve two problems at once:

1. **Approximate nearest-neighbor (ANN) search** — specialized index structures that find near-optimal neighbors in sub-linear time by trading a small amount of recall for orders-of-magnitude speed.
2. **Everything a database does** — persistence, updates and deletes, metadata filtering, consistency guarantees, replication, sharding, access control, backups, and day-two operability.

The first problem has thirty years of literature behind it. The second is where most real-world complexity, and most product differentiation, actually lives.

### 2. Embeddings and vector spaces

An **embedding** is a fixed-length array of floating-point numbers produced by a model that maps an input (text, image, audio, code, a user's behaviour history, a molecule) to a point in R^d. Good embedding models are trained — usually with contrastive objectives — so that inputs with similar meaning land near each other.

Key properties:

- **Dimensionality (d).** Common values are 384, 768, 1024, 1536, 3072 and 4096. Higher d generally carries more information but costs linearly more memory and compute. Models trained with Matryoshka Representation Learning (§13) let you truncate to the first 256 or 512 dimensions with graceful quality loss.
- **Dense vs. sparse.** Dense embeddings have a nonzero value in every dimension. Sparse vectors (BM25 term weights, SPLADE, BGE-M3's sparse head) have tens of thousands of dimensions with mostly zeros and are served by inverted lists, not ANN graphs. Modern systems increasingly store both.
- **Single vs. multi-vector.** Most systems store one vector per item. Late-interaction models (ColBERT for text, ColPali for document images) produce one vector per token or image patch — dozens to hundreds per item — and score with MaxSim. Higher quality on many tasks, much higher storage.
- **Model coupling.** Vectors from different models, or different versions of the same model, are not comparable. An embedding is only meaningful relative to the model that produced it. This has large operational consequences (§13).

### 3. Similarity and distance metrics

| Metric | Definition (for vectors a, b) | Use when |
|---|---|---|
| Cosine similarity | a·b / (‖a‖ ‖b‖) | Direction matters, magnitude doesn't. Default for most text embeddings. |
| Dot (inner) product | a·b | Model trained with dot product; magnitude carries signal (e.g. item popularity in recsys). "Maximum inner-product search" (MIPS). |
| Euclidean (L2) | ‖a − b‖ | Model trained with L2; some image and scientific embeddings. |
| Manhattan (L1) | ‖a − b‖₁ | Rare; some sparse or robustness settings. |
| Hamming | number of differing bits | Binary-quantized vectors. |
| Jaccard | size(A∩B) ÷ size(A∪B) | Set-valued data (tags, shingles). |

Two things to internalize:

- **If vectors are L2-normalized (unit length), cosine, dot product and Euclidean produce identical rankings**, because ‖a − b‖² = 2 − 2(a·b) on the unit sphere. Most vector databases normalize on ingest when you choose cosine, which turns cosine into a cheap dot product.
- **Use the metric the embedding model was trained with.** Mixing them silently degrades quality without producing an error.

### 4. The nearest-neighbor problem

**Exact k-NN** (brute force, "flat" search) computes every distance and keeps the top k. It is embarrassingly parallel and, with SIMD (AVX-512, NEON) or GPUs, faster than people expect: a few million 768-d vectors scan in tens of milliseconds on one node. Exact search is the right answer more often than vendors admit — it is simple, always 100% recall, supports arbitrary filters trivially, and needs no index build.

**Approximate NN (ANN)** trades recall for speed. **Recall@k** here means: of the true k nearest neighbors, what fraction did the index return? Production systems typically target 0.90–0.99. Why we accept less than 1.0:

- **The curse of dimensionality.** In high dimensions, distances between random points concentrate (nearest and farthest neighbors become nearly equidistant), and space-partitioning structures such as KD-trees degrade toward linear scan above roughly 20 dimensions.
- **Real embedding data is not random.** It lives on a much lower-dimensional manifold with cluster structure, which is exactly what ANN indexes exploit. (This is also why benchmarking on synthetic random vectors tells you nothing.)

The fundamental trade-off is a triangle: **recall ↔ query latency and throughput ↔ memory and build time.** Every index family and every tuning knob moves you around this triangle.

---

## Part 2 — Indexing: how ANN search works

### 5. Index families

**Flat / brute force.** No index; scan everything. Best for fewer than ~1M vectors, for highly selective filtered queries (brute-force the filtered subset), and as ground truth for measuring recall. GPU flat search (Faiss GPU, NVIDIA cuVS) extends this to tens of millions.

**Tree-based (KD-tree, ball tree, random-projection forests).** Partition space recursively. Annoy (Spotify) builds many random-hyperplane trees and unions their leaves; it is simple, memory-mappable, and immutable after build. Largely superseded by graphs for high-dimensional data.

**Locality-sensitive hashing (LSH).** Hash vectors so that similar ones collide with high probability (random hyperplanes for cosine, p-stable distributions for L2). Elegant sublinear guarantees, but reaching high recall in practice needs many tables and a lot of memory. Rarely the production choice today outside streaming and near-duplicate detection (SimHash, MinHash).

**Inverted file (IVF) / clustering.** Run k-means to get `nlist` centroids and assign each vector to its nearest centroid's list. At query time, find the `nprobe` closest centroids and scan only their lists. This is Faiss's workhorse. Build is cheap (one k-means pass), memory is just vectors plus centroids, and it composes with compression (IVF-PQ). Weaknesses: recall depends on cluster quality, boundary effects (a true neighbor may sit in an un-probed cluster), and lists need re-training as the data distribution drifts. Rules of thumb: `nlist ≈ 4√N to 16√N`; train on at least ~30–50 × `nlist` samples.

**Graph-based (NSW, HNSW, NSG, Vamana/DiskANN, CAGRA).** Build a proximity graph where each vector links to a small number of near — and a few far — neighbors, then search by greedy traversal: start somewhere, move to whichever neighbor is closest to the query, repeat, keeping a candidate beam. **HNSW** (Hierarchical Navigable Small World; Malkov & Yashunin, 2016) adds a hierarchy of sparser layers for fast coarse navigation, like a skip list. HNSW is the default in most vector databases because it offers the best recall-per-millisecond at moderate scale, supports incremental inserts, and is well understood. Costs: memory (graph links on top of raw vectors), slow single-threaded build (roughly O(N log N) with a large constant), and awkward deletes.

**Disk-resident graphs (DiskANN/Vamana, SPANN).** DiskANN (Microsoft, 2019) builds a single-layer graph with deliberately added long-range edges, keeps a product-quantized copy of every vector in RAM for navigation, and stores the graph plus full-precision vectors on NVMe SSD. Result: billion-scale search on a single machine with tens of gigabytes of RAM, ~95% recall, and single-digit-millisecond latency. Streaming/fresh variants handle inserts and deletes in place. The design is used or adapted by Azure (Cosmos DB, Bing), pgvectorscale (StreamingDiskANN), Milvus, JVector (Cassandra/Astra) and others. SPANN takes an IVF-like approach with centroids in memory and posting lists on disk.

**Quantization — compression, orthogonal to the choice of index.** Quantization reduces bytes per vector so more fits in RAM and cache, and distance computations get faster:

- **Scalar quantization (SQ):** float32 → int8 (4×) or float16/bfloat16 (2×). Near-lossless for most embeddings — usually the free win.
- **Product quantization (PQ):** split the vector into `m` sub-vectors, k-means each subspace into 256 centroids, store one byte per sub-vector. A 1536-d float vector (6,144 bytes) becomes, say, 96 bytes — 64× smaller. Distances come from precomputed lookup tables (asymmetric distance computation). Lossy, so pair it with **rescoring**: retrieve 2–5× more candidates using compressed vectors, then re-rank them with exact vectors from disk. OPQ (a learned rotation applied before PQ) reduces error.
- **Binary quantization (BQ):** one bit per dimension (the sign). 32× compression; distances become Hamming (a popcount). Works surprisingly well for high-dimensional (≥768) embeddings when paired with rescoring; poorly below ~384 dimensions. **RaBitQ** (2024) put this on a theoretical footing with provable error bounds and extended it to multi-bit codes; its ideas now underpin the "binary plus rescore" modes in Elasticsearch (BBQ), Milvus, and others.
- **Matryoshka truncation:** not quantization, but the same move — search with the first 256 dims, rescore with all 1,536.

**GPU-native indexes.** NVIDIA cuVS provides CAGRA, a graph index built and searched on GPU, typically 10–50× faster to build than CPU HNSW, with graphs exportable to CPU HNSW format. Milvus, OpenSearch, Faiss, and Qdrant (build-time) integrate GPU indexing. GPUs shine for bulk index builds and very-high-QPS batch search; for latency-sensitive single queries with filters, CPU graphs still dominate.

**Hybrid structures.** Real engines combine these: IVF with an HNSW index over the centroids (Faiss `IVF_HNSW`), HNSW over SQ/PQ/BQ codes, DiskANN with PQ in memory, and tiered layouts (hot in RAM, warm on SSD, cold in object storage).

**Cheat sheet**

| Family | Recall / latency | Memory | Build | Updates | Sweet spot |
|---|---|---|---|---|---|
| Flat | 100% / linear in N | vectors only | none | trivial | < 1M vectors, ground truth, heavy filters |
| IVF (-PQ) | good / fast | low with PQ | fast | needs periodic re-train | 10M–1B, memory-constrained, batch |
| HNSW | excellent / fastest | high (vectors + graph) | slow | inserts fine, deletes awkward | 100k–100M, low latency |
| DiskANN | very good / fast | low RAM, SSD-heavy | slow | streaming variants | 100M–1B+ on one box, cost-sensitive |
| GPU (CAGRA) | excellent / very fast | GPU memory | very fast | rebuild | bulk builds, high-QPS batch |
| LSH | fair / fast | high | fast | trivial | streaming dedup, theory |

### 6. Tuning knobs

**HNSW**

- `M` — maximum links per node per layer (typical 8–64; 16 is a common default). Higher M → better recall, more memory, slower build. High-dimensional or hard datasets want 32–48.
- `efConstruction` — beam width during build (typical 100–500). Higher → better graph quality, slower build. A one-time cost, so err high.
- `efSearch` (a.k.a. `ef`, `hnsw_ef`) — beam width at query time; must be ≥ k, typically 50–500. **This is the runtime recall dial.** Tune it per query class; many systems accept it per request.

**IVF**

- `nlist` (number of partitions) and `nprobe` (partitions visited per query). Recall rises with `nprobe / nlist`; latency rises roughly linearly with `nprobe`.

**Quantization**

- The quantizer itself, and the oversampling/rescore factor (how many compressed candidates get re-ranked with exact vectors).

**Universal**

- `k`, and the filter. A highly selective filter changes the optimal strategy entirely (§9).

The correct tuning procedure is empirical: take a representative sample of real queries, compute the exact top-k as ground truth (flat search), then sweep the knobs and plot recall against p99 latency. Never tune on latency alone.

### 7. Memory sizing (worked example)

For HNSW, memory per vector ≈ `4·d` bytes (float32) + `~8·M` bytes for layer-0 graph links (2M links × 4-byte ids) + a small overhead for the upper layers.

Ten million vectors, d = 1536, M = 16:

- Raw float32: 10M × 6,144 B ≈ **61 GB**
- Graph links: 10M × 128 B ≈ 1.3 GB
- → roughly **63 GB of RAM** for the index alone, before payloads, replicas and headroom.

The same collection with compression:

- int8 scalar quantization: 10M × (1,536 + 128) ≈ **17 GB**
- binary quantization with rescoring: 10M × (192 + 128) ≈ **3.2 GB** in RAM, full-precision vectors on SSD or object storage for rescoring
- Matryoshka-truncated to 512 dims + int8: 10M × (512 + 128) ≈ **6.4 GB**

Two lessons: dimensionality and precision dominate the bill, and compression is what turns "needs a 128 GB box" into "runs on a laptop." Then add payload and metadata storage (often larger than the vectors), write-ahead logs, and one or two replicas.

---

## Part 3 — From index to database

An ANN library (Faiss, hnswlib, USearch) gives you an in-memory index and a search call. A vector *database* has to handle everything that happens after the demo.

### 8. What the database layer adds

**Mutability.** Inserting into HNSW is natural — that is how it gets built. **Deleting is not:** removing a node breaks graph paths, so engines mark deletions with tombstones, filter them out at query time, and periodically rebuild or compact. A collection carrying 30% tombstones has worse recall and latency than its size suggests; monitor the ratio and vacuum. Updates are delete-plus-insert. IVF lists tolerate churn but drift away from their centroids.

**Segments and compaction (the LSM pattern).** Most modern engines (Milvus, Qdrant, Weaviate, LanceDB, and the Lucene-based systems) write new vectors into a small mutable "growing" segment — often searched by brute force — then seal it into an immutable indexed segment and merge segments in the background. Queries fan out across segments and merge top-k. This gives fast writes and immutable index files that are easy to replicate and back up, at the cost of query fan-out and background CPU.

**Freshness and consistency.** "Is my just-written vector searchable?" Answers range from immediately (Postgres/pgvector, single-node Qdrant) to eventually (Pinecone serverless, Milvus at its default level, any sharded system with asynchronous replication). Some engines expose tunable levels (Milvus: Strong, Bounded, Session, Eventual). RAG usually tolerates seconds of lag; fraud detection or within-session agent memory may not.

**Durability and transactions.** Write-ahead log plus snapshots is standard. Full ACID transactions spanning vectors and relational data essentially exist only inside general-purpose databases (Postgres/pgvector, Oracle, SQL Server, Cosmos DB, MongoDB). Purpose-built vector databases typically offer per-operation atomicity and eventual cross-shard consistency — fine for most workloads, but worth knowing before you rely on it.

**Payload and metadata storage.** Vectors are pointers to things; the things (text chunks, document ids, tenant ids, timestamps, ACLs, JSON) need storing, indexing for filters, and returning with results. Payload indexing — keyword, numeric range, geo, full-text — is a major differentiator between products.

**Operability.** Backup and restore, rolling upgrades, observability (recall drift, segment counts, tombstone ratio, cache hit rates), RBAC, encryption, quotas, multi-region.

### 9. Metadata filtering — the hardest "easy" problem

Almost every real query is "nearest neighbors *where* tenant = X *and* date > Y *and* category in (…)". There are three strategies:

- **Post-filtering.** Run ANN for the top k′ (k′ > k), then drop results that fail the filter. Simple, but with a selective filter (say, 1% of rows match) you either return nothing or need k′ in the hundreds of thousands. Naïve implementations fail silently with short result lists.
- **Pre-filtering.** Evaluate the filter first to get the allowed id set, then search only within it. Exact and simple when the filter is selective — brute-force over the matching subset. But restricting an HNSW traversal to a sparse allowed set breaks graph connectivity (the greedy walk hits dead ends), and recall collapses.
- **In-traversal (single-stage) filtering.** Apply the predicate during graph traversal while still expanding through non-matching nodes to preserve connectivity. This is what serious engines do now: Qdrant's "filterable HNSW" adds extra links per payload value; Weaviate implements ACORN (predicate-agnostic expansion); Milvus, Vespa and Pinecone evaluate filters inline; Lucene-based engines intersect a bitset with the HNSW walk and fall back to exact search when the filter is very selective.

Good engines choose a strategy per query from the **estimated filter selectivity**: brute-force the subset if it is small, filtered-graph search if it is large, partition-scoped search if the filter aligns with a partition key. The design implication: **if a filter is always present and highly selective (tenant, user, document set), make it a partition or namespace, not a filter.**

### 10. Hybrid search, multi-vector, and reranking

Dense vectors are strong on semantics and weak on exact tokens: product codes, names, rare terms, numbers. Lexical search (BM25) is the reverse. Production retrieval combines them.

**Hybrid search** = dense ANN + sparse/lexical retrieval + fusion.

- Sparse side: classic BM25 over an inverted index, or *learned sparse* models (SPLADE, Elastic's ELSER, BGE-M3's sparse head) that expand terms with learned weights — capturing synonyms while staying exact-match friendly.
- Fusion: **Reciprocal Rank Fusion** — `score(d) = Σᵢ 1 / (k + rankᵢ(d))` with k ≈ 60 — is the robust default because it ignores incomparable score scales. Weighted score fusion (normalize each list, then α·dense + (1−α)·sparse) gives more control but needs calibration.
- Native support: Elasticsearch/OpenSearch, Vespa, Weaviate, Qdrant, Milvus, Pinecone (sparse-dense), Azure AI Search, MongoDB Atlas; pgvector via a join with `tsvector`.

**Reranking.** ANN returns the top 50–200 by approximate similarity; a **cross-encoder** (Cohere Rerank, bge-reranker, Jina, Voyage, or an LLM) then scores each (query, document) pair jointly and reorders. Rerankers cost more per pair but are far more accurate — most of the quality in a RAG pipeline comes from this stage. Several databases now bundle reranking as a query step.

**Multi-vector / late interaction.** ColBERT-style models keep one vector per token and score with MaxSim (for each query token, the best-matching document token; sum over query tokens). Quality approaches a cross-encoder at near-ANN speed, but storage is 50–300× that of a single vector. Supported through native multi-vector fields (Vespa, Weaviate, Qdrant, Milvus, LanceDB), usually with compression (token pooling, 2-bit quantization). ColPali/ColQwen apply the idea to page images, which sidesteps OCR for document RAG.

### 11. Scale-out: multi-tenancy, sharding, replication, storage tiers

**Multi-tenancy patterns**, in rough order of isolation:

1. Shared collection + `tenant_id` filter — simplest; depends on filtered-search quality; noisy-neighbour risk.
2. Partitions or namespaces within a collection (Pinecone namespaces, Milvus partition keys, Turbopuffer namespaces) — cheap isolation; a query is scoped to one partition.
3. Collection or shard per tenant (Weaviate native multi-tenancy, Qdrant per-tenant shards) — strong isolation, inactive tenants can be offloaded to cold storage; watch out for thousands of tiny indexes.
4. Cluster per tenant — regulated or very large enterprise accounts only.

**Sharding.** Vectors are distributed by id hash (or partition key) across shards; a query is scattered to every shard, each returns its top-k, and the coordinator merges (scatter-gather). Latency equals the slowest shard, and more shards mean more fan-out. Partition-aligned routing avoids the fan-out.

**Replication.** For availability and read throughput. Immutable segments make replication easy (copy files); the write path replicates through consensus (Raft) or a log.

**Storage architectures — the big shift of 2024–2026.**

- *Memory-resident:* everything in RAM (classic HNSW deployments). Fastest and most expensive.
- *SSD-resident:* DiskANN-style; RAM holds compressed vectors and graph metadata.
- *Object-storage-native / serverless:* vectors and indexes live in S3/GCS/Azure Blob as immutable files; stateless query nodes fetch and cache what they need; writes go through a log. Cold-query latency is higher (hundreds of milliseconds until the cache is warm), but cost per million vectors drops by an order of magnitude or more, and idle collections cost close to nothing. Pinecone serverless, Turbopuffer, LanceDB, Chroma Cloud, Milvus's tiered designs, and Amazon S3 Vectors follow this pattern. It suits the long tail of many small tenants and bursty agent workloads; it is the wrong fit for sub-10 ms p99 on hot data unless the cache is kept warm.

---

## Part 4 — Building retrieval systems

### 12. The retrieval pipeline end to end

**Ingest**

1. Parse and clean source content. Keep the raw source as the system of record — not the vector database.
2. Chunk (§13) and attach metadata: source, tenant, ACL, timestamps, section titles.
3. Embed in batches — dense, plus sparse or multi-vector if used. Record model name and version with every vector.
4. Upsert with idempotent ids (hash of source + chunk position) so re-runs don't create duplicates.

**Query**

1. Optional query rewriting: LLM expansion or decomposition, HyDE, multi-query.
2. Embed the query with the same model — including the query-side instruction or prefix if the model uses one.
3. Retrieve candidates: dense ANN (plus sparse) with filters; k′ ≈ 50–200.
4. Fuse (RRF), deduplicate by source, apply business rules (recency boosts, ACL checks).
5. Rerank to the final k ≈ 5–20.
6. Assemble context or return results; log the query, candidates and outcomes for evaluation.

A typical p95 latency budget: embedding 20–50 ms → ANN 5–30 ms → reranking 50–200 ms. The reranker, not the vector database, usually dominates latency; the vector database usually dominates cost.

### 13. Embedding strategy

**Model choice.** Start from the MTEB/BEIR leaderboards but validate on your own data — leaderboard gaps are often within noise for a specific domain. Consider language coverage, maximum input length, dimension, license, cost and latency of self-hosting versus an API, and whether a matching reranker exists. Fine-tuning an embedding model on even a few thousand in-domain (query, positive passage) pairs often beats switching to a bigger model.

**Asymmetric vs. symmetric.** Retrieval is asymmetric (short query, long passage), and many models expect different prefixes or instructions for the two sides (`query:` / `passage:`, or an instruction such as "Represent this sentence for searching relevant passages"). Omitting them costs several points of recall.

**Chunking.** The most under-invested decision. Options: fixed token windows with overlap (baseline); sentence- or paragraph-aware; structure-aware (headings, tables, code blocks); semantic (split where embedding similarity drops); and "contextual" chunking, which prepends a generated summary of the document or section to each chunk so it embeds with its context. A typical sweet spot is 200–500 tokens for question answering, larger for summarization-style tasks. Store parent/child relations so you can match on a small chunk and return its larger parent.

**Dimensions.** More is not automatically better. Matryoshka-trained models (OpenAI text-embedding-3, Gemini embeddings, Nomic, Jina, and many open models) allow truncation to 256–512 dimensions at a cost of a few points — a 3–6× reduction in storage and compute. Combine with int8 or binary quantization for another 4–32×.

**Versioning and migration.** Changing the embedding model means re-embedding everything; you cannot mix. Store `embedding_model` and `embedding_version` on every record, keep raw text so re-embedding is a batch job rather than a data-recovery exercise, and migrate by dual-writing to a new collection, validating recall and quality, then cutting over. Budget for this from day one — models improve every few months.

**Multimodal.** CLIP- and SigLIP-family models embed images and text into one space; audio and video encoders likewise. The same database machinery applies; payloads point to blobs in object storage.

### 14. Evaluation and benchmarks

Measure three different things, and don't confuse them:

1. **Index quality — recall@k against exact search.** Purely about the ANN approximation. Compute ground truth by flat search over a sample of real queries; target ≥ 0.95 for RAG, higher for dedup and recsys candidate generation.
2. **Retrieval quality — does the right material come back?** Requires a labelled set of (query, relevant documents). Metrics: nDCG@k, MRR, precision/recall@k, hit rate. Build the set from real logs plus LLM-assisted labelling with human spot checks; 200–500 queries is enough to start. This is the yardstick for chunking, embedding, hybrid and reranking decisions.
3. **End-to-end task quality.** For RAG: answer correctness, faithfulness/groundedness, citation accuracy. For recsys: CTR and conversion in A/B tests.

Plus the operational metrics: p50/p95/p99 query latency, QPS per node, ingest throughput, index build time, RAM/SSD per million vectors, cost per million queries.

Public benchmarks: **ANN-Benchmarks** (algorithm-level recall/QPS curves on standard datasets), **VectorDBBench** (database-level, with filters and streaming inserts), **BigANN** (billion-scale challenges including filtered and streaming tracks), and **BEIR/MTEB** (embedding and retrieval quality — not databases). Treat vendor-published benchmarks as marketing until reproduced on your data.

---

## Part 5 — The landscape

### 15. Taxonomy of options (2026)

The market has settled into four shapes. The names below are representative rather than exhaustive, and the space moves quickly — verify current status and pricing before committing.

**A. ANN libraries — embed in your process; you own persistence and operations.**

- **Faiss** (Meta) — the reference toolkit: flat, IVF, PQ, HNSW, GPU. Research-grade breadth.
- **hnswlib** — the canonical HNSW implementation, used inside many databases.
- **USearch** — compact, fast HNSW with quantization and many language bindings; powers the vector indexes in ClickHouse and DuckDB.
- **ScaNN** (Google) — anisotropic quantization, very strong on CPU; powers AlloyDB's vector index.
- **DiskANN/Vamana** (Microsoft), **cuVS/CAGRA** (NVIDIA), **Voyager** (Spotify's successor to Annoy), **Annoy** (legacy).

**B. Purpose-built vector databases.**

- **Milvus / Zilliz Cloud** — distributed and cloud-native, with the broadest index menu (HNSW, IVF, DiskANN, GPU, RaBitQ-style quantizers) and tiered storage; built for billion scale.
- **Qdrant** — Rust; filterable HNSW, rich payload indexing, scalar/binary/product quantization, multi-vector, GPU-assisted builds; excellent single-binary ergonomics.
- **Weaviate** — Go; HNSW with PQ/BQ/SQ compression, native multi-tenancy, hybrid BM25 + dense, multi-vector, modular embedding and reranker integrations.
- **Pinecone** — fully managed and serverless (object-storage-native), namespaces, sparse-dense hybrid, integrated inference and reranking; zero-ops positioning.
- **Chroma** — developer-first; an embedded mode for prototyping plus a distributed, Rust-based cloud offering.
- **LanceDB** — built on the Lance columnar format; embedded or serverless on object storage; strong for multimodal and versioned datasets.
- **Turbopuffer** — object-storage-native with a namespace-per-tenant model and hybrid search; popular with very-many-tenant SaaS and coding assistants.
- **Vespa** — older and broader than "vector DB": a full search and ranking engine with HNSW, tensors, multi-vector, and multi-phase ranking. The choice when ranking logic is the product.

**C. Vector search inside general-purpose databases and search engines.** For most teams, this is the default answer.

- **PostgreSQL + pgvector** — HNSW and IVFFlat, half-precision and binary vector types, sparse vectors, iterative index scans for filtered queries; **pgvectorscale** adds StreamingDiskANN and label-based filtering. Available on AlloyDB (which also offers a ScaNN index), Aurora, Neon, Supabase, Timescale, and every other managed Postgres.
- **Elasticsearch / OpenSearch** — Lucene HNSW (OpenSearch also offers Faiss and NMSLIB engines), BM25 + learned sparse + dense hybrid, BBQ and int8 quantization, GPU-accelerated index builds in OpenSearch. The natural choice if you already run them.
- Also: **MongoDB Atlas Vector Search**, **Redis (Query Engine)**, **Cassandra / DataStax Astra (JVector)**, **ClickHouse**, **DuckDB (vss)**, **SQLite (sqlite-vec)**, **Oracle AI Vector Search**, **SQL Server / Azure SQL (native vector type)**, **MySQL HeatWave**, **Snowflake Cortex Search**, **BigQuery vector search**, **Databricks Mosaic AI Vector Search**, **Azure Cosmos DB (DiskANN)**.

**D. Cloud-native managed search and vector services.**

- **Azure AI Search** (HNSW, hybrid, semantic ranker, agentic retrieval), **Google Vertex AI Vector Search** (ScaNN, very large scale), **Amazon OpenSearch Serverless**, **Amazon S3 Vectors** (vectors as an S3-native primitive, aimed at cheap cold and warm storage), and **Amazon Bedrock Knowledge Bases** (managed RAG over the above).

The through-line of 2025–2026: vectors moved from being a database *category* to being a *data type*. Standalone vendors now differentiate on scale, tenancy, and retrieval sophistication rather than on having vectors at all.

### 16. Choosing: a decision framework

Ask, in this order:

1. **Do you need ANN at all?** Under ~1M vectors with modest QPS, flat search in whatever database you already have — or NumPy — is correct, exact, and filter-friendly.
2. **Where does the source-of-truth data live?** If it is in Postgres, Mongo, or Elastic, start with that system's vector support. One system means one consistency model, one backup, one ACL model, and joins for free. Move out only when you hit a *measured* wall: index build time, memory, filtered recall, QPS.
3. **What are the scale and shape?**
   - 1–50M vectors, single tenant, latency-sensitive → HNSW in pgvector, Qdrant, Weaviate, or Elasticsearch.
   - 100M–1B+, high QPS, or GPU builds → Milvus/Zilliz, Vespa, Vertex.
   - Thousands of tenants, bursty, cost-first → object-storage-native: Turbopuffer, Pinecone serverless, LanceDB, S3 Vectors.
   - Laptop, edge, or prototype → Chroma, LanceDB, sqlite-vec, DuckDB, Faiss.
4. **How complex is retrieval?** Heavy hybrid, complex ranking, multi-vector → Vespa, Elastic/OpenSearch, Weaviate, Qdrant, Milvus.
5. **What is the filtering profile?** An always-present selective filter → partition-based design. Arbitrary rich filters → engines with in-traversal filtering and payload indexes.
6. **What is your operating posture?** Managed vs. self-hosted; data residency, VPC, compliance; vendor risk (several vendors are venture-funded startups — assess longevity); exit cost (exporting vectors and payloads is easy; reproducing filter and hybrid semantics elsewhere is not).

A useful heuristic for 2026: **treat vector search as a feature of your data platform first and as a separate database second.** The dedicated systems win when scale, tenancy, or retrieval complexity *is* the product.

### 17. Use-case patterns

- **RAG and enterprise search** — chunk-level dense + sparse hybrid, ACL filtering via metadata, reranking, parent-document retrieval. Freshness within seconds is fine.
- **Agent memory** — episodic (conversation turns), semantic (facts), and procedural (successful tool sequences) memories stored as vectors with rich metadata: time, user, task, importance. Queries are frequent, small-k, and heavily filtered by user or session; retention, decay, and deduplication matter; per-user namespaces fit well; freshness within a session must be immediate. Agents also issue far more queries per task than a human user, which stresses per-query latency and cost.
- **Semantic caching** — cache LLM responses keyed by query embedding and return the cached answer when a near-duplicate query arrives (threshold-tuned). Cuts cost and latency; beware over-eager matches on subtly different questions.
- **Recommendations / two-tower retrieval** — a user vector queries the item index for candidates (MIPS with dot product), followed by a heavier ranker. Very high QPS, moderate recall requirements, frequent re-embedding.
- **Deduplication and near-duplicate detection** — high recall required; often binary or MinHash codes plus rescoring; batch-oriented.
- **Anomaly and fraud detection** — distance to the nearest known-good or known-bad examples as a feature; low latency, streaming inserts.
- **Multimodal search** — image, video, and audio retrieval with CLIP-family embeddings; large blobs live in object storage.
- **Code search and repository intelligence** — code-specific embeddings, symbol-aware chunking, hybrid with exact identifier matching; per-repository namespaces.

---

## Part 6 — Operations, pitfalls, and trends

### 18. Operating a vector database in production

- **Capacity planning.** Size RAM and SSD from §7 with the quantization strategy decided up front; keep 2× headroom for re-indexing and migrations.
- **Index lifecycle.** Build offline where possible (GPU, or high-`efConstruction` CPU builds) and swap atomically; schedule compaction and vacuum; alert on tombstone ratio and segment count.
- **Recall monitoring.** Sample queries daily, compute exact ground truth, and track recall@k over time. Recall decays silently with deletes, distribution drift, and stale IVF centroids.
- **Embedding drift and model upgrades.** Pin model versions, keep raw text, and rehearse the re-embedding migration before you need it.
- **Security.** Tenant isolation (partitions rather than filters where isolation must be hard); row- or document-level ACLs as metadata enforced at query time; encryption at rest; private networking; audit logs. Treat embeddings themselves as sensitive: inversion attacks recover a large fraction of the source text from a vector.
- **Cost levers**, in rough order of impact: dimensions, quantization, storage tier, replica count, reranker calls.
- **Backups and disaster recovery.** Snapshot immutable segments together with the payload store; test restores, including index rebuild time.

### 19. Common pitfalls

1. Wrong or mismatched distance metric; unnormalized vectors under cosine.
2. Treating the vector database as the system of record and losing the raw text and metadata.
3. Never measuring recall; tuning `ef` or `nprobe` on latency alone.
4. Post-filtering with selective filters, producing empty or truncated result sets.
5. Ignoring tombstone accumulation and index degradation after heavy deletes.
6. Chunks too large (diluted embeddings) or too small (no context), with no parent retrieval.
7. Forgetting query and passage instruction prefixes.
8. Mixing embeddings from different models or versions in one collection.
9. Skipping the reranker and blaming the database for irrelevant results.
10. Provisioning RAM for float32 when int8 or binary with rescoring would be indistinguishable in quality.
11. Assuming vector similarity equals relevance — recency, authority, and ACLs still need explicit logic.
12. Modelling tenants as a filter when they should be partitions (or, for thousands of tiny tenants, the reverse).
13. Benchmarking on synthetic random vectors, which lack the manifold structure of real data, so results don't transfer.
14. Ignoring cold-start latency in serverless and object-storage designs.
15. Trusting vendor-run benchmarks: each major vendor publishes a suite that flatters its own architecture.

### 20. Where things are heading (2025–2026)

- **Vector search is a feature, not a category.** Every major relational, document, and analytical database ships vector types and ANN indexes. Standalone vendors differentiate on scale, tenancy, and retrieval sophistication.
- **Quantization by default.** Binary or int8 with rescoring (RaBitQ-family methods) is becoming the out-of-the-box path; float32 in RAM is now the exception.
- **Storage–compute separation.** Object-storage-native architectures with local caches are winning on cost for the long tail; tiered hot/warm/cold is becoming table stakes.
- **GPU-accelerated index builds** remove the biggest operational pain of HNSW.
- **Better filtered search.** ACORN-style graphs, partition-aware planners, and cost-based selection between brute force and graph traversal are narrowing the historical gap between vector and relational query planning.
- **Multi-vector and late interaction** are moving from research to product, especially for visual document retrieval.
- **Agentic retrieval.** Agents issue many small, iterative, filtered queries rather than one large one — pressure on per-query latency, freshness, per-user namespaces, and "memory" APIs layered on vector stores.
- **Retrieval survives long context.** Million-token windows change *how much* you retrieve, not *whether*: cost, latency, freshness, and access control keep retrieval in the loop.
- **Learned sparse + dense hybrid** is the default retrieval stack, with LLM-based rerankers where the latency budget allows.
- **Edge and on-device vector search** (SQLite/DuckDB extensions, embedded engines) for data that cannot leave the device or the premises.

---

## Appendix A — Glossary

- **ANN** — approximate nearest-neighbor search.
- **Recall@k** — fraction of the true top-k neighbors an approximate method returns.
- **HNSW** — Hierarchical Navigable Small World graph index.
- **IVF** — inverted file index: cluster, then probe a few clusters.
- **PQ / SQ / BQ** — product / scalar / binary quantization.
- **RaBitQ** — quantization method with provable error bounds; basis of several "binary + rescore" implementations.
- **DiskANN / Vamana** — SSD-resident graph index and its graph construction algorithm.
- **efSearch / nprobe** — the runtime recall dials for HNSW / IVF.
- **Rescoring (refinement)** — re-ranking compressed-vector candidates with exact vectors.
- **RRF** — reciprocal rank fusion, for combining ranked lists.
- **BM25 / SPLADE** — classic lexical scoring / learned sparse retrieval.
- **Cross-encoder** — a model that scores a (query, document) pair jointly; used for reranking.
- **Late interaction / ColBERT / MaxSim** — multi-vector retrieval and its scoring function.
- **MRL (Matryoshka)** — embeddings whose leading dimensions form a usable shorter embedding.
- **Tombstone** — deletion marker in an immutable index segment.
- **Scatter-gather** — fan a query to all shards and merge the top-k.
- **MIPS** — maximum inner-product search.
- **ACORN** — a filtered-graph-search technique that keeps HNSW traversal connected under arbitrary predicates.

## Appendix B — Further reading

- Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs* (2016) — arXiv:1603.09320
- Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node* (NeurIPS 2019)
- Jégou, Douze & Schmid, *Product Quantization for Nearest Neighbor Search* (IEEE TPAMI 2011)
- Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization* — ScaNN (ICML 2020)
- Gao & Long, *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound* (SIGMOD 2024)
- Patel et al., *ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data* (SIGMOD 2024)
- Kusupati et al., *Matryoshka Representation Learning* (NeurIPS 2022)
- Khattab & Zaharia, *ColBERT* (SIGIR 2020); Faysse et al., *ColPali* (2024)
- Douze et al., *The Faiss library* (2024) — arXiv:2401.08281
- Pan, Wang & Li, *Survey of Vector Database Management Systems* (VLDB Journal 2024)
- ANN-Benchmarks (ann-benchmarks.com), VectorDBBench, and the BigANN benchmark — for reproducible comparisons
