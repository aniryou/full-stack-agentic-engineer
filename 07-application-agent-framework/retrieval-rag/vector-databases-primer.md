# Vector Databases: A Comprehensive Primer

*August 2026*

This primer starts from first principles and goes to production. It is for engineers and architects who already know what a machine-learning model is. They want a complete mental model of vector search:

- why vector search exists,
- how the indexes work,
- what turns an index into a database,
- how to build retrieval that gives good results in practice,
- what shape the vendor landscape has,
- and where vector search goes next.

If you already know what an embedding is, go to Part 2. Parts 3 and 4 are the most important parts of the document. Part 3 tells what makes a database different from an index library. Part 4 tells how to build retrieval that works. Part 5 shows the landscape and gives a decision framework.

**Contents**

- Part 1 (Foundations): why vector databases exist, embeddings, metrics, the nearest-neighbor problem
- Part 2 (Indexing): index families, tuning knobs, memory sizing
- Part 3 (From index to database): mutability, freshness, filtering, hybrid search, scale-out
- Part 4 (Building retrieval systems): the pipeline, embedding strategy, evaluation
- Part 5 (The landscape): taxonomy, decision framework, use-case patterns
- Part 6: Operations, pitfalls, and trends
- Appendix A (Glossary) and Appendix B (Further reading)

---

## Part 1 — Foundations

### 1. Why vector databases exist

Traditional databases answer questions about exact values and ranges: `WHERE customer_id = 42`, `WHERE price < 100`, `WHERE name LIKE 'Jo%'`. They use B-trees, hash indexes and inverted indexes. These structures use the order of the keys or exact token matches.

A full class of modern questions has no exact-match formulation. Some examples are:

- "find documents that mean roughly the same thing as this query"
- "find images that look like this one"
- "find products this user would like"
- "have we seen a support ticket like this before?"

Neural networks answer these questions. They map objects into a high-dimensional vector space, where geometric proximity approximates semantic similarity. Then the database question becomes this: for a query vector, find the $k$ stored vectors that are nearest to it. This is nearest-neighbor search.

Nearest-neighbor search in hundreds or thousands of dimensions defeats every classic index. There is no total order to use, so B-trees fail. There is no exact key to hash, so hash indexes fail. There are no discrete tokens to invert, so inverted indexes fail. Only brute force stays: you compare the query with every vector. That costs $O(N \cdot d)$ per query, which is fine for 100k vectors, but not for 100 million at 1,000 queries per second.

Vector databases exist to solve two problems at the same time:

1. **Approximate nearest-neighbor (ANN) search.** Specialized index structures find near-optimal neighbors in sub-linear time. They give up a small quantity of recall to get orders of magnitude more speed.
2. **Everything a database does.** This is persistence, updates and deletes, metadata filtering, consistency guarantees, replication, sharding, access control, backups and day-two operability.

The first problem has thirty years of literature. Most of the complexity in production, and most of the differences between products, are in the second problem.

### 2. Embeddings and vector spaces

An **embedding** is a fixed-length array of floating-point numbers. A model produces it. The model maps an input to a point in $\mathbb{R}^d$. The input can be text, an image, audio, code, the behaviour history of a user, or a molecule. The training of a good embedding model puts inputs with a similar meaning near each other. Usually this training uses contrastive objectives.

Key properties:

- **Dimensionality (d).** Common values are 384, 768, 1024, 1536, 3072 and 4096. A higher d usually carries more information. But it also costs linearly more memory and compute. Models trained with Matryoshka Representation Learning (§13) let you truncate the vector to the first 256 or 512 dimensions, with a gradual loss of quality.
- **Dense vs. sparse.** Dense embeddings have a nonzero value in every dimension. Sparse vectors (BM25 term weights, SPLADE, the sparse head of BGE-M3) have tens of thousands of dimensions, and most of the values are zero. Inverted lists serve them, not ANN graphs. Modern systems store both kinds more and more often.
- **Single vs. multi-vector.** Most systems store one vector per item. Late-interaction models (ColBERT for text, ColPali for document images) produce one vector per token or per image patch. That is dozens to hundreds of vectors per item. These models score with MaxSim. They give higher quality on many tasks, but they use much more storage.
- **Model coupling.** You cannot compare vectors from different models, or from different versions of the same model. An embedding has a meaning only relative to the model that produced it. This has large operational consequences (§13).

### 3. Similarity and distance metrics

| Metric | Definition (for vectors $a$, $b$) | Use when |
|---|---|---|
| Cosine similarity | $a \cdot b \,/\, (\lVert a \rVert \, \lVert b \rVert)$ | The direction is important. The magnitude is not. This is the default for most text embeddings. |
| Dot (inner) product | $a \cdot b$ | The training of the model used the dot product. The magnitude carries a signal (for example, item popularity in recsys). This is "maximum inner-product search" (MIPS). |
| Euclidean (L2) | $\lVert a - b \rVert$ | The training of the model used L2. Some image and scientific embeddings use it. |
| Manhattan (L1) | $\lVert a - b \rVert_1$ | Rare. Some sparse or robustness settings use it. |
| Hamming | number of bits that differ | Binary-quantized vectors. |
| Jaccard | $\operatorname{size}(A \cap B) \div \operatorname{size}(A \cup B)$ | Set-valued data (tags, shingles). |

Two things to know well:

- **If the vectors are L2-normalized (unit length), cosine, dot product and Euclidean give the same ranking.** The reason is that $\lVert a - b \rVert^2 = 2 - 2(a \cdot b)$ on the unit sphere. When you select cosine, most vector databases normalize the vectors at ingest. This turns cosine into a low-cost dot product.
- **Use the metric that the training of the embedding model used.** If you mix metrics, the quality decreases silently, and you get no error.

### 4. The nearest-neighbor problem

**Exact k-NN** (brute force, "flat" search) calculates every distance and keeps the top k. It is embarrassingly parallel. With SIMD (AVX-512, NEON) or GPUs, it is faster than people expect: one node scans a few million 768-d vectors in tens of milliseconds. Exact search is the correct answer more often than vendors admit. It is simple and always gives 100% recall. It supports arbitrary filters with no extra work, and it needs no index build.

**Approximate NN (ANN)** gives up recall to get speed. Here, **Recall@k** means this: of the true k nearest neighbors, what fraction did the index return? Production systems usually aim for 0.90–0.99. We accept less than 1.0 for these reasons:

- **The curse of dimensionality.** In high dimensions, the distances between random points concentrate: the nearest and the farthest neighbors become almost equidistant. Above approximately 20 dimensions, space-partitioning structures such as KD-trees become almost as slow as a linear scan.
- **Real embedding data is not random.** It is on a manifold of much lower dimension, and it has cluster structure. ANN indexes use exactly this structure. (This is also why a benchmark on synthetic random vectors tells you nothing.)

The fundamental trade-off is a triangle. Its three corners are **recall**, **query latency and throughput**, and **memory and build time**. Every index family and every tuning knob moves you around this triangle.

---

## Part 2 — Indexing: how ANN search works

### 5. Index families

**Flat / brute force.** There is no index, and the engine scans everything. It is best for fewer than ~1M vectors and for highly selective filtered queries (scan the filtered subset with brute force). It is also the ground truth when you measure recall. GPU flat search (Faiss GPU, NVIDIA cuVS) extends this to tens of millions of vectors.

**Tree-based (KD-tree, ball tree, random-projection forests).** These indexes partition the space recursively. Annoy (Spotify) builds many random-hyperplane trees and takes the union of their leaves. It is simple, memory-mappable, and immutable after the build. For high-dimensional data, graphs have mostly replaced tree indexes.

**Locality-sensitive hashing (LSH).** LSH hashes vectors so that similar vectors collide with high probability. It uses random hyperplanes for cosine and p-stable distributions for L2. It has elegant sublinear guarantees. But to get high recall in practice, it needs many tables and much memory. Today it is rarely the production choice, except for streaming and near-duplicate detection (SimHash, MinHash).

**Inverted file (IVF) / clustering.** Run k-means to get `nlist` centroids, and assign each vector to the list of its nearest centroid. At query time, find the `nprobe` nearest centroids and scan only their lists. This is the main index of Faiss. The build is low-cost (one k-means pass), and the memory holds only the vectors plus the centroids. IVF also combines with compression (IVF-PQ).

IVF has three weaknesses. Recall depends on the quality of the clusters. Boundary effects occur: a true neighbor can be in a cluster that the query does not probe. Also, the lists need new training when the data distribution drifts. General rules: `nlist` $\approx 4\sqrt{N}$ to $16\sqrt{N}$, and train on at least ~30–50 × `nlist` samples.

**Graph-based (NSW, HNSW, NSG, Vamana/DiskANN, CAGRA).** Build a proximity graph in which each vector links to a small number of near neighbors and to a few far neighbors. Then search with a greedy traversal: start at a point, and move to the neighbor that is nearest to the query. Do this step again and again, and keep a candidate beam.

**HNSW** (Hierarchical Navigable Small World, by Malkov and Yashunin, 2016) adds a hierarchy of sparser layers for fast coarse navigation, like a skip list. Most vector databases use HNSW as the default. It gives the best recall per millisecond at moderate scale, it supports incremental inserts, and engineers know it well. Its costs are memory (graph links on top of the raw vectors), a slow single-threaded build (approximately $O(N \log N)$ with a large constant), and awkward deletes.

**Disk-resident graphs (DiskANN/Vamana, SPANN).** DiskANN (Microsoft, 2019) builds a single-layer graph, and adds long-range edges on purpose. It keeps a product-quantized copy of every vector in RAM for navigation. It stores the graph and the full-precision vectors on an NVMe SSD. The result is billion-scale search on a single machine with tens of gigabytes of RAM, ~95% recall and single-digit-millisecond latency. Streaming/fresh variants handle inserts and deletes in place.

Azure (Cosmos DB, Bing), pgvectorscale (StreamingDiskANN), Milvus, JVector (Cassandra/Astra) and others use or adapt the design. SPANN uses an approach similar to IVF: the centroids are in memory and the posting lists are on disk.

**Quantization — compression, orthogonal to the choice of index.** Quantization decreases the bytes per vector. Thus more vectors fit in RAM and in cache, and the distance calculations become faster:

- **Scalar quantization (SQ):** float32 to int8 (4×), or to float16/bfloat16 (2×). For most embeddings it is near-lossless. Usually it is the gain that costs nothing.
- **Product quantization (PQ):** divide the vector into `m` sub-vectors. Run k-means in each subspace to get 256 centroids, and store one byte per sub-vector. For example, a 1536-d float vector (6,144 bytes) becomes 96 bytes, which is 64× smaller. The distances come from precomputed lookup tables (asymmetric distance computation). PQ is lossy, so use it with **rescoring**: retrieve 2–5× more candidates with the compressed vectors, then rerank them with the exact vectors from disk. OPQ (a learned rotation applied before PQ) decreases the error.
- **Binary quantization (BQ):** one bit per dimension (the sign). This gives 32× compression, and the distances become Hamming distances (a popcount). With rescoring, it works surprisingly well for high-dimensional (≥768) embeddings. Below ~384 dimensions, it works poorly. **RaBitQ** (2024) gave this method a theoretical basis with provable error bounds, and extended it to multi-bit codes. Its ideas are now the basis of the "binary plus rescore" modes in Elasticsearch (BBQ), Milvus, and others.
- **Matryoshka truncation:** this is not quantization, but it is the same move. Search with the first 256 dims, then rescore with all 1,536.

**GPU-native indexes.** NVIDIA cuVS supplies CAGRA, a graph index whose build and search run on the GPU. Its build is usually 10–50× faster than a CPU HNSW build, and you can export its graphs to the CPU HNSW format. Milvus, OpenSearch, Faiss, and Qdrant (build-time) integrate GPU indexing. GPUs are best for bulk index builds and for batch search at very high QPS. For latency-sensitive single queries with filters, CPU graphs still dominate.

**Hybrid structures.** Real engines combine these methods. Examples are IVF with an HNSW index over the centroids (Faiss `IVF_HNSW`), HNSW over SQ/PQ/BQ codes, and DiskANN with PQ in memory. Tiered layouts are another example: hot data in RAM, warm data on SSD, cold data in object storage.

**Cheat sheet**

| Family | Recall / latency | Memory | Build | Updates | Sweet spot |
|---|---|---|---|---|---|
| Flat | 100% / linear in N | vectors only | none | simple | < 1M vectors, ground truth, heavy filters |
| IVF (-PQ) | good / fast | low with PQ | fast | needs periodic new training | 10M–1B, memory-constrained, batch |
| HNSW | excellent / fastest | high (vectors + graph) | slow | inserts fine, deletes awkward | 100k–100M, low latency |
| DiskANN | very good / fast | low RAM, SSD-heavy | slow | streaming variants | 100M–1B+ on one box, cost-sensitive |
| GPU (CAGRA) | excellent / very fast | GPU memory | very fast | rebuild | bulk builds, high-QPS batch |
| LSH | fair / fast | high | fast | simple | streaming dedup, theory |

### 6. Tuning knobs

**HNSW**

- `M`: the maximum number of links per node per layer (usually 8–64, and 16 is a common default). A higher M gives better recall, more memory and a slower build. High-dimensional or hard datasets need 32–48.
- `efConstruction`: the beam width during the build (usually 100–500). A higher value gives better graph quality and a slower build. It is a one-time cost, so prefer a value that is too high to one that is too low.
- `efSearch` (also named `ef` or `hnsw_ef`): the beam width at query time. It must be $\ge k$, and it is usually 50–500. **This is the runtime recall dial.** Adjust it for each query class. Many systems accept it per request.

**IVF**

- `nlist` (the number of partitions) and `nprobe` (the partitions that a query visits). Recall increases with `nprobe / nlist`. Latency increases approximately linearly with `nprobe`.

**Quantization**

- The quantizer itself, and the oversampling/rescore factor (how many compressed candidates the engine reranks with exact vectors).

**Universal**

- `k`, and the filter. A highly selective filter changes the optimal strategy completely (§9).

The correct way to adjust the knobs is empirical. Take a representative sample of real queries. Calculate the exact top-k as ground truth (flat search). Then sweep the knobs and plot recall against p99 latency. Never adjust the knobs on latency alone.

### 7. Memory sizing (worked example)

For HNSW, the memory per vector is approximately the sum of these parts:

- $4 \cdot d$ bytes (float32),
- ${\sim}8 \cdot M$ bytes for the layer-0 graph links ($2\,M$ links × 4-byte ids),
- a small overhead for the upper layers.

Ten million vectors, $d = 1536$, $M = 16$:

- Raw float32: 10M × 6,144 B ≈ **61 GB**
- Graph links: 10M × 128 B ≈ 1.3 GB
- Total: roughly **63 GB of RAM** for the index alone, before payloads, replicas and headroom.

The same collection with compression:

- int8 scalar quantization: 10M × (1,536 + 128) ≈ **17 GB**
- binary quantization with rescoring: 10M × (192 + 128) ≈ **3.2 GB** in RAM, and the full-precision vectors on SSD or in object storage for the rescoring
- Matryoshka-truncated to 512 dims + int8: 10M × (512 + 128) ≈ **6.4 GB**

There are two lessons. First, dimensionality and precision dominate the bill. Second, compression is what turns "needs a 128 GB box" into "runs on a laptop". Then add the payload and metadata storage (often larger than the vectors), the write-ahead logs, and one or two replicas.

---

## Part 3 — From index to database

An ANN library (Faiss, hnswlib, USearch) gives you an in-memory index and a search call. A vector *database* must handle everything that occurs after the demo.

### 8. What the database layer adds

**Mutability.** An insert into HNSW is natural, because the build makes the graph with inserts. **A delete is not:** when you remove a node, graph paths break. Thus engines mark deletions with tombstones, filter them out at query time, and periodically rebuild or compact the index. A collection with 30% tombstones has worse recall and latency than its size suggests. Monitor the ratio and vacuum the collection.

An update is a delete plus an insert. IVF lists tolerate churn, but they drift away from their centroids.

**Segments and compaction (the LSM pattern).** Most modern engines (Milvus, Qdrant, Weaviate, LanceDB, and the Lucene-based systems) write new vectors into a small mutable "growing" segment. They often search this segment by brute force. Then they seal it into an immutable indexed segment, and they merge segments in the background.

Queries fan out across segments and merge the top-k. This gives fast writes, and immutable index files that are easy to replicate and back up. The cost is the query fan-out and the background CPU.

**Freshness and consistency.** The question is this: can a search find my just-written vector? The answers go from immediately (Postgres/pgvector, single-node Qdrant) to eventually (Pinecone serverless, Milvus at its default level, any sharded system with asynchronous replication). Some engines expose tunable levels (Milvus: Strong, Bounded, Session, Eventual). RAG usually tolerates seconds of lag. It is possible that fraud detection or within-session agent memory does not.

**Durability and transactions.** A write-ahead log plus snapshots is standard. Full ACID transactions over vectors and relational data together exist almost only inside general-purpose databases (Postgres/pgvector, Oracle, SQL Server, Cosmos DB, MongoDB). Purpose-built vector databases usually give per-operation atomicity and eventual cross-shard consistency. This is fine for most workloads, but know it before you rely on it.

**Payload and metadata storage.** Vectors are pointers to things. The database must store the things (text chunks, document ids, tenant ids, timestamps, ACLs, JSON), index them for filters, and return them with the results. Payload indexing (keyword, numeric range, geo, full-text) is a major difference between products.

**Operability.** This means backup and restore, rolling upgrades, observability (recall drift, segment counts, tombstone ratio, cache hit rates), RBAC, encryption, quotas and multi-region.

### 9. Metadata filtering — the hardest "easy" problem

Almost every real query is "nearest neighbors *where* tenant = X *and* date > Y *and* category in (…)". There are three strategies:

- **Post-filtering.** Run ANN for the top $k'$ ($k' > k$), then drop the results that fail the filter. This is simple. But with a selective filter (for example, 1% of rows match), you either return nothing or need a $k'$ in the hundreds of thousands. Simple implementations fail silently and return short result lists.
- **Pre-filtering.** Evaluate the filter first to get the set of permitted ids, then search only in that set. When the filter is selective, this is exact and simple: brute force over the subset that matches. But when you restrict an HNSW traversal to a sparse permitted set, graph connectivity breaks (the greedy walk hits dead ends), and recall collapses.
- **In-traversal (single-stage) filtering.** Apply the predicate during the graph traversal, but continue to expand through nodes that do not match, to keep the connectivity. Serious engines do this now. Qdrant's "filterable HNSW" adds more links per payload value. Weaviate implements ACORN (predicate-agnostic expansion). Milvus, Vespa and Pinecone evaluate filters inline. Lucene-based engines intersect a bitset with the HNSW walk, and fall back to exact search when the filter is highly selective.

Good engines select a strategy for each query from the **estimated filter selectivity**. If the subset is small, they scan it with brute force. If it is large, they use filtered-graph search. If the filter aligns with a partition key, they use partition-scoped search. This has a design implication. **If a filter is always present and highly selective (tenant, user, document set), make it a partition or namespace, not a filter.**

### 10. Hybrid search, multi-vector, and reranking

Dense vectors are strong on semantics and weak on exact tokens: product codes, names, rare terms, numbers. Lexical search (BM25) is the reverse. Production retrieval combines them.

**Hybrid search** is dense ANN, plus sparse/lexical retrieval, plus fusion.

- Sparse side: classic BM25 over an inverted index, or *learned sparse* models (SPLADE, Elastic's ELSER, BGE-M3's sparse head). These models expand terms with learned weights. Thus they capture synonyms and stay friendly to exact matches.
- Fusion: **Reciprocal Rank Fusion** ($\operatorname{score}(d) = \sum_{i} 1 / (k + \operatorname{rank}_{i}(d))$ with $k \approx 60$) is the robust default, because it ignores score scales that are not comparable. Weighted score fusion (normalize each list, then $\alpha \cdot \text{dense} + (1 - \alpha) \cdot \text{sparse}$) gives more control, but it needs calibration.
- Native support: Elasticsearch/OpenSearch, Vespa, Weaviate, Qdrant, Milvus, Pinecone (sparse-dense), Azure AI Search, MongoDB Atlas. pgvector supports it through a join with `tsvector`.

**Reranking.** ANN returns the top 50–200 by approximate similarity. Then a **cross-encoder** (Cohere Rerank, bge-reranker, Jina, Voyage, or an LLM) scores each (query, document) pair jointly and puts the list in a new order. Rerankers cost more per pair, but they are far more accurate. Most of the quality in a RAG pipeline comes from this stage. Several databases now include reranking as a query step.

**Multi-vector / late interaction.** ColBERT-style models keep one vector per token and score with MaxSim. MaxSim takes, for each query token, the document token with the best match, and adds these values over the query tokens. The quality approaches that of a cross-encoder at near-ANN speed, but the storage is 50–300× that of a single vector. Native multi-vector fields (Vespa, Weaviate, Qdrant, Milvus, LanceDB) support it, usually with compression (token pooling, 2-bit quantization). ColPali/ColQwen apply the idea to page images, and thus document RAG does not need OCR.

### 11. Scale-out: multi-tenancy, sharding, replication, storage tiers

**Multi-tenancy patterns**, in rough order of isolation:

1. Shared collection + `tenant_id` filter. This is the simplest pattern. It depends on the quality of filtered search, and it has a noisy-neighbour risk.
2. Partitions or namespaces within a collection (Pinecone namespaces, Milvus partition keys, Turbopuffer namespaces). This gives low-cost isolation. The scope of a query is one partition.
3. Collection or shard per tenant (Weaviate native multi-tenancy, Qdrant per-tenant shards). This gives strong isolation, and you can offload inactive tenants to cold storage. Be careful with thousands of small indexes.
4. Cluster per tenant. Use it only for regulated or very large enterprise accounts.

**Sharding.** The system distributes vectors across shards by id hash (or by partition key). A query goes to every shard, each shard returns its top-k, and the coordinator merges the results (scatter-gather). The latency is equal to that of the slowest shard, and more shards give more fan-out. Partition-aligned routing prevents the fan-out.

**Replication.** It is for availability and read throughput. Immutable segments make replication easy (copy the files). The write path replicates through consensus (Raft) or through a log.

**Storage architectures — the big shift of 2024–2026.**

- *Memory-resident:* everything is in RAM (classic HNSW deployments). This is the fastest architecture and the one with the highest cost.
- *SSD-resident:* DiskANN-style. RAM holds the compressed vectors and the graph metadata.
- *Object-storage-native / serverless:* the vectors and indexes are in S3/GCS/Azure Blob as immutable files. Stateless query nodes fetch and cache what they need, and writes go through a log. A cold query has a higher latency (hundreds of milliseconds until the cache is warm). But the cost per million vectors decreases by an order of magnitude or more, and idle collections cost almost nothing. Pinecone serverless, Turbopuffer, LanceDB, Chroma Cloud, Milvus's tiered designs, and Amazon S3 Vectors use this pattern.

  This pattern suits the long tail of many small tenants and bursty agent workloads. It is the incorrect fit for sub-10 ms p99 on hot data, unless you keep the cache warm.

---

## Part 4 — Building retrieval systems

### 12. The retrieval pipeline end to end

**Ingest**

1. Parse and clean the source content. Keep the raw source as the system of record, not the vector database.
2. Chunk the content (§13), and attach metadata: source, tenant, ACL, timestamps, section titles.
3. Embed in batches: dense, plus sparse or multi-vector if you use them. Record the model name and version with every vector.
4. Upsert with idempotent ids (a hash of the source + the chunk position). Then a re-run does not make duplicates.

**Query**

1. Optionally, rewrite the query: LLM expansion or decomposition, HyDE, multi-query.
2. Embed the query with the same model. If the model uses a query-side instruction or prefix, include it.
3. Retrieve candidates: dense ANN (plus sparse) with filters, with $k'$ ≈ 50–200.
4. Fuse the lists (RRF), and remove duplicates by source. Apply business rules (recency boosts, ACL checks).
5. Rerank to the final $k$ ≈ 5–20.
6. Assemble the context or return the results. Log the query, the candidates and the outcomes for evaluation.

A typical p95 latency budget is: embedding 20–50 ms, then ANN 5–30 ms, then reranking 50–200 ms. Usually the reranker dominates the latency, not the vector database. Usually the vector database dominates the cost.

### 13. Embedding strategy

**Model choice.** Start from the MTEB/BEIR leaderboards, but measure the models on your own data. For a specific domain, the gaps on a leaderboard are often within the noise. Consider language coverage, maximum input length, dimension, license, and the cost and latency of a self-hosted model against an API. Also consider if a reranker that matches the model exists. Fine-tuning an embedding model on even a few thousand in-domain (query, positive passage) pairs often gives better results than a change to a larger model.

**Asymmetric vs. symmetric.** Retrieval is asymmetric (short query, long passage). Many models expect different prefixes or instructions for the two sides (`query:` / `passage:`, or an instruction such as "Represent this sentence for searching relevant passages"). If you omit them, you lose several points of recall.

**Chunking.** It is the decision with the most under-investment. The options are:

- fixed token windows with overlap (the baseline),
- sentence- or paragraph-aware chunks,
- structure-aware chunks (headings, tables, code blocks),
- semantic chunks (split where the embedding similarity decreases),
- and "contextual" chunking, which puts a generated summary of the document or section before each chunk, so that the chunk embeds with its context.

A typical sweet spot is 200–500 tokens for question-and-answer tasks, and larger for summarization-style tasks. Store parent/child relations, so that you can match on a small chunk and return its larger parent.

**Dimensions.** More dimensions are not automatically better. Matryoshka-trained models (OpenAI text-embedding-3, Gemini embeddings, Nomic, Jina, and many open models) permit truncation to 256–512 dimensions, at a cost of a few points. This is a 3–6× reduction in storage and compute. Combine it with int8 or binary quantization for another 4–32×.

**Versioning and migration.** A change of the embedding model means that you must embed everything again, because you cannot mix vectors from two models. Store `embedding_model` and `embedding_version` on every record. Keep the raw text, so that a new embedding pass is a batch job and not a data-recovery exercise.

To migrate, write to a new collection and to the old one at the same time. Then examine recall and quality, and after that, move the traffic to the new collection. Budget for this from day one, because models improve every few months.

**Multimodal.** CLIP- and SigLIP-family models embed images and text into one space. Audio and video encoders do the same. The same database machinery applies, and the payloads point to blobs in object storage.

### 14. Evaluation and benchmarks

Measure three different things, and do not confuse them:

1. **Index quality — recall@k against exact search.** This is only about the ANN approximation. Calculate the ground truth by flat search over a sample of real queries. Aim for ≥ 0.95 for RAG, and higher for dedup and for recsys candidate generation.
2. **Retrieval quality — does the right material come back?** This needs a labelled set of (query, relevant documents). Metrics: nDCG@k, MRR, precision/recall@k, hit rate. Build the set from real logs plus labels made with LLM help, with spot checks by a person. To start, 200–500 queries are sufficient. This set is the yardstick for decisions about chunking, embedding, hybrid search and reranking.
3. **End-to-end task quality.** For RAG: answer correctness, faithfulness/groundedness, citation accuracy. For recsys: CTR and conversion in A/B tests.

Also, there are the operational metrics: p50/p95/p99 query latency, QPS per node, ingest throughput, index build time, RAM/SSD per million vectors, cost per million queries.

The public benchmarks are:

- **ANN-Benchmarks**: algorithm-level recall/QPS curves on standard datasets.
- **VectorDBBench**: database-level, with filters and streaming inserts.
- **BigANN**: billion-scale challenges, with filtered and streaming tracks.
- **BEIR/MTEB**: embedding and retrieval quality, not databases.

Treat the benchmarks that vendors publish as sales material until you reproduce them on your data.

---

## Part 5 — The landscape

### 15. Taxonomy of options (2026)

The market has settled into four shapes. The names in this section are representative, not exhaustive. The space changes fast, so make sure of the current status and pricing before you commit.

**A. ANN libraries — embed in your process; you own persistence and operations.**

- **Faiss** (Meta): the reference toolkit, with flat, IVF, PQ, HNSW and GPU indexes. It has research-grade breadth.
- **hnswlib**: the canonical HNSW implementation. Many databases use it inside.
- **USearch**: a compact, fast HNSW with quantization and many language bindings. It powers the vector indexes in ClickHouse and DuckDB.
- **ScaNN** (Google): anisotropic quantization, strong on CPU to a high degree. It powers the vector index of AlloyDB.
- **DiskANN/Vamana** (Microsoft), **cuVS/CAGRA** (NVIDIA), **Voyager** (Spotify's successor to Annoy), **Annoy** (legacy).

**B. Purpose-built vector databases.**

- **Milvus / Zilliz Cloud**: distributed and cloud-native, with the broadest index menu (HNSW, IVF, DiskANN, GPU, RaBitQ-style quantizers) and tiered storage. Its design is for billion scale.
- **Qdrant**: written in Rust, with filterable HNSW, rich payload indexing, scalar/binary/product quantization, multi-vector, and GPU-assisted builds. Its single-binary ergonomics are excellent.
- **Weaviate**: written in Go. It has HNSW with PQ/BQ/SQ compression, native multi-tenancy, hybrid BM25 + dense, multi-vector, and modular embedding and reranker integrations.
- **Pinecone**: fully managed and serverless (object-storage-native). It has namespaces, sparse-dense hybrid, and integrated inference and reranking. Its market position is zero-ops.
- **Chroma**: developer-first. It has an embedded mode for prototypes, plus a distributed, Rust-based cloud product.
- **LanceDB**: built on the Lance columnar format. It runs embedded, or serverless on object storage. It is strong for multimodal and versioned datasets.
- **Turbopuffer**: object-storage-native, with a namespace-per-tenant model and hybrid search. It is popular with SaaS products that have very many tenants, and with assistants that write code.
- **Vespa**: older and broader than a "vector DB". It is a full search and ranking engine with HNSW, tensors, multi-vector, and multi-phase ranking. It is the choice when the ranking logic is the product.

**C. Vector search inside general-purpose databases and search engines.** For most teams, this is the default answer.

- **PostgreSQL + pgvector**: HNSW and IVFFlat, half-precision and binary vector types, sparse vectors, and iterative index scans for filtered queries. **pgvectorscale** adds StreamingDiskANN and label-based filtering. It is available on AlloyDB (which also has a ScaNN index), Aurora, Neon, Supabase, Timescale, and every other managed Postgres.
- **Elasticsearch / OpenSearch**: Lucene HNSW (OpenSearch also has Faiss and NMSLIB engines), and BM25 + learned sparse + dense hybrid. They also have BBQ and int8 quantization, and OpenSearch has GPU-accelerated index builds. This is the natural choice if you already operate them.
- Also:
  - **MongoDB Atlas Vector Search**
  - **Redis (Query Engine)**
  - **Cassandra / DataStax Astra (JVector)**
  - **ClickHouse**
  - **DuckDB (vss)**
  - **SQLite (sqlite-vec)**
  - **Oracle AI Vector Search**
  - **SQL Server / Azure SQL (native vector type)**
  - **MySQL HeatWave**
  - **Snowflake Cortex Search**
  - **BigQuery vector search**
  - **Databricks Mosaic AI Vector Search**
  - **Azure Cosmos DB (DiskANN)**

**D. Cloud-native managed search and vector services.**

- The services are **Azure AI Search** (HNSW, hybrid, semantic ranker, agentic retrieval) and **Google Vertex AI Vector Search** (ScaNN, very large scale). Then there are **Amazon OpenSearch Serverless** and **Amazon S3 Vectors** (vectors as an S3-native primitive, for low-cost cold and warm storage). Also, **Amazon Bedrock Knowledge Bases** gives managed RAG over the services before it in this list.

The through-line of 2025–2026 is this: vectors moved from a database *category* to a *data type*. Standalone vendors now differentiate on scale, tenancy, and retrieval sophistication, not on the presence of vectors.

### 16. Choosing: a decision framework

Ask these questions, in this order:

1. **Do you need ANN at all?** Under ~1M vectors with modest QPS, flat search is correct, exact, and filter-friendly. Use any database that you already have, or NumPy.
2. **Where does the source-of-truth data live?** If it is in Postgres, Mongo, or Elastic, start with the vector support of that system. One system means one consistency model, one backup, one ACL model, and joins at no cost. Move out only when you hit a *measured* wall: index build time, memory, filtered recall, QPS.
3. **What are the scale and shape?**
   - 1–50M vectors, single tenant, latency-sensitive: HNSW in pgvector, Qdrant, Weaviate, or Elasticsearch.
   - 100M–1B+, high QPS, or GPU builds: Milvus/Zilliz, Vespa, Vertex.
   - Thousands of tenants, bursty, cost-first: object-storage-native (Turbopuffer, Pinecone serverless, LanceDB, S3 Vectors).
   - Laptop, edge, or prototype: Chroma, LanceDB, sqlite-vec, DuckDB, Faiss.
4. **How complex is retrieval?** For heavy hybrid, complex ranking, or multi-vector, use Vespa, Elastic/OpenSearch, Weaviate, Qdrant, or Milvus.
5. **What is the filtering profile?** For an always-present selective filter, use a partition-based design. For arbitrary rich filters, use engines with in-traversal filtering and payload indexes.
6. **What is your operating posture?** Consider managed against self-hosted, and data residency, VPC and compliance. Consider vendor risk: several vendors are venture-funded startups, so assess their longevity. Consider the exit cost. Export of vectors and payloads is easy. But to reproduce filter and hybrid semantics in another system is not easy.

A useful heuristic for 2026: **treat vector search as a feature of your data platform first and as a separate database second.** The dedicated systems win when scale, tenancy, or retrieval complexity *is* the product.

### 17. Use-case patterns

- **RAG and enterprise search**: chunk-level dense + sparse hybrid, ACL filtering through metadata, reranking, parent-document retrieval. Freshness within seconds is fine.
- **Agent memory**: the system stores episodic (conversation turns), semantic (facts), and procedural (successful tool sequences) memories as vectors with rich metadata: time, user, task, importance. Queries are frequent and small-k, and they have heavy filters by user or session. Retention, decay, and deduplication are important. Per-user namespaces fit well, and freshness within a session must be immediate. Agents also send far more queries per task than a person, which puts stress on per-query latency and cost.
- **Semantic caching**: cache LLM responses with the query embedding as the key. When a near-duplicate query arrives, return the cached answer (threshold-tuned). This decreases cost and latency. Be careful with over-eager matches on questions that are subtly different.
- **Recommendations / two-tower retrieval**: a user vector queries the item index for candidates (MIPS with dot product), and a heavier ranker comes after it. This pattern has very high QPS, moderate recall requirements, and frequent new embeddings.
- **Deduplication and near-duplicate detection**: high recall is necessary. It often uses binary or MinHash codes plus rescoring. It is batch-oriented.
- **Anomaly and fraud detection**: the distance to the nearest known-good or known-bad examples is a feature. It needs low latency and streaming inserts.
- **Multimodal search**: image, video, and audio retrieval with CLIP-family embeddings. Large blobs are in object storage.
- **Code search and repository intelligence**: code-specific embeddings, symbol-aware chunking, and hybrid search with exact identifier match. It uses per-repository namespaces.

---

## Part 6 — Operations, pitfalls, and trends

### 18. Operating a vector database in production

- **Capacity planning.** Size RAM and SSD from §7, and decide the quantization strategy at the start. Keep 2× headroom for index rebuilds and migrations.
- **Index lifecycle.** Build the index offline where possible (GPU, or CPU builds with a high `efConstruction`), and swap it atomically. Schedule compaction and vacuum. Alert on the tombstone ratio and the segment count.
- **Recall monitoring.** Sample queries daily, calculate the exact ground truth, and track recall@k over time. Recall decreases silently with deletes, distribution drift, and stale IVF centroids.
- **Embedding drift and model upgrades.** Pin the model versions and keep the raw text. Do a test run of the migration to new embeddings before you need it.
- **Security.** Use tenant isolation (partitions, not filters, where the isolation must be hard). Put row- or document-level ACLs in the metadata, and enforce them at query time. Use encryption at rest, private networks, and audit logs. Treat the embeddings themselves as sensitive: inversion attacks recover a large fraction of the source text from a vector.
- **Cost levers**, in rough order of impact: dimensions, quantization, storage tier, replica count, reranker calls.
- **Backups and disaster recovery.** Take snapshots of the immutable segments together with the payload store. Test the restores, and include the index rebuild time.

### 19. Common pitfalls

1. An incorrect or mismatched distance metric, and unnormalized vectors under cosine.
2. You use the vector database as the system of record, and you lose the raw text and metadata.
3. You never measure recall, and you adjust `ef` or `nprobe` on latency alone.
4. You use post-filtering with selective filters, and you get empty or truncated result sets.
5. You ignore the accumulation of tombstones and the index degradation after heavy deletes.
6. Chunks are too large (diluted embeddings) or too small (no context), and there is no parent retrieval.
7. You forget the query and passage instruction prefixes.
8. You mix embeddings from different models or versions in one collection.
9. You skip the reranker and blame the database for irrelevant results.
10. You size RAM for float32 when int8 or binary with rescoring gives quality that you cannot distinguish from it.
11. You assume that vector similarity equals relevance. Recency, authority, and ACLs still need explicit logic.
12. You model tenants as a filter when partitions are correct for them (or the reverse, for thousands of small tenants).
13. You run benchmarks on synthetic random vectors. These lack the manifold structure of real data, so the results do not transfer.
14. You ignore the cold-start latency in serverless and object-storage designs.
15. You trust benchmarks that vendors run. Each major vendor publishes a suite that flatters its own architecture.

### 20. Where things are heading (2025–2026)

- **Vector search is a feature, not a category.** Every major relational, document, and analytical database ships vector types and ANN indexes. Standalone vendors differentiate on scale, tenancy, and retrieval sophistication.
- **Quantization by default.** Binary or int8 with rescoring (RaBitQ-family methods) becomes the out-of-the-box path more and more. Float32 in RAM is now the exception.
- **Storage–compute separation.** Object-storage-native architectures with local caches now win on cost for the long tail. Tiered hot/warm/cold storage gradually becomes table stakes.
- **GPU-accelerated index builds** remove the largest operational pain of HNSW.
- **Better filtered search.** ACORN-style graphs, partition-aware planners, and cost-based selection between brute force and graph traversal now make a historical gap smaller. This is the gap between vector and relational query planning.
- **Multi-vector and late interaction** move from research to product, especially for visual document retrieval.
- **Agentic retrieval.** Agents send many small, iterative, filtered queries, not one large query. This puts pressure on per-query latency, freshness, per-user namespaces, and "memory" APIs on top of vector stores.
- **Retrieval survives long context.** Million-token windows change *how much* you retrieve, not *if* you retrieve. Cost, latency, freshness, and access control keep retrieval in the loop.
- **Learned sparse + dense hybrid** is the default retrieval stack, with LLM-based rerankers where the latency budget permits.
- **Edge and on-device vector search** (SQLite/DuckDB extensions, embedded engines) is for data that cannot leave the device or the premises.

---

## Appendix A — Glossary

- **ANN**: approximate nearest-neighbor search.
- **Recall@k**: the fraction of the true top-k neighbors that an approximate method returns.
- **HNSW**: Hierarchical Navigable Small World graph index.
- **IVF**: inverted file index. It makes clusters, then probes a few clusters.
- **PQ / SQ / BQ**: product / scalar / binary quantization.
- **RaBitQ**: a quantization method with provable error bounds. It is the basis of several "binary + rescore" implementations.
- **DiskANN / Vamana**: an SSD-resident graph index and its graph construction algorithm.
- **efSearch / nprobe**: the runtime recall dials for HNSW / IVF.
- **Rescoring (refinement)**: reranking of the compressed-vector candidates with exact vectors.
- **RRF**: reciprocal rank fusion, which combines ranked lists.
- **BM25 / SPLADE**: a classic lexical score function / learned sparse retrieval.
- **Cross-encoder**: a model that scores a (query, document) pair jointly. Reranking uses it.
- **Late interaction / ColBERT / MaxSim**: multi-vector retrieval and its score function.
- **MRL (Matryoshka)**: embeddings whose first dimensions form a usable shorter embedding.
- **Tombstone**: a deletion marker in an immutable index segment.
- **Scatter-gather**: send a query to all shards and merge the top-k.
- **MIPS**: maximum inner-product search.
- **ACORN**: a filtered-graph-search technique that keeps the HNSW traversal connected under arbitrary predicates.

## Appendix B — Further reading

- Malkov and Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs* (2016), arXiv:1603.09320
- Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node* (NeurIPS 2019)
- Jégou, Douze and Schmid, *Product Quantization for Nearest Neighbor Search* (IEEE TPAMI 2011)
- Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*, ScaNN (ICML 2020)
- Gao and Long, *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound* (SIGMOD 2024)
- Patel et al., *ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data* (SIGMOD 2024)
- Kusupati et al., *Matryoshka Representation Learning* (NeurIPS 2022)
- Khattab and Zaharia, *ColBERT* (SIGIR 2020), and Faysse et al., *ColPali* (2024)
- Douze et al., *The Faiss library* (2024), arXiv:2401.08281
- Pan, Wang and Li, *Survey of Vector Database Management Systems* (VLDB Journal 2024)
- ANN-Benchmarks (ann-benchmarks.com), VectorDBBench, and the BigANN benchmark, for reproducible comparisons
