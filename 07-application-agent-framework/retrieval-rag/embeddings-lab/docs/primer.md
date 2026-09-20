# Embeddings: A Comprehensive Primer

*From distributional semantics to the frontier of representation learning, retrieval systems, and interpretability. Written August 2026.*

---

## How to read this

- **Part I — Foundations.** What an embedding is, the classical lineage, and what contextual representations inside transformers actually are. Skim if you know it; section 3 has details most people miss.
- **Part II — Modern text embedding models.** The training recipe, representation formats beyond a single dense vector, and how to evaluate.
- **Part III — Geometry.** Similarity, high-dimensional phenomena, linear structure and superposition, non-Euclidean spaces. This is where embeddings meet interpretability.
- **Part IV — Beyond text.** Vision, multimodal, code, science, recommenders, graphs.
- **Part V — Systems.** ANN search, vector stores, and retrieval pipelines for RAG and agents, including failure modes and security.
- **Part VI — Frontier.** Open problems and the 2025–2026 research direction, a model-landscape snapshot, and a decision checklist.
- **Appendices.** Formulas, worked numbers, reading list, glossary.

Notation: `d` is embedding dimension, `N` corpus size, `q`/`d` query and document, `cos(a,b) = a·b / (‖a‖‖b‖)`, `τ` temperature, `|V|` vocabulary size.

---

# Part I — Foundations

## 1. What an embedding is (and isn't)

An **embedding** is a learned map `f: X → ℝ^d` from a space of objects `X` (tokens, sentences, images, users, graph nodes, molecules, audio clips) into a vector space, constructed so that geometric relations in `ℝ^d` — inner products, distances, directions — encode the relations among objects that matter for some task.

Three ideas are hiding inside that sentence:

1. **Continuity.** Discrete objects get placed in a continuous space where "similar" is computable and, crucially, differentiable. Gradients flow through the embedding, so it can be learned end-to-end with whatever consumes it.
2. **Compression.** `d` is far smaller than the raw dimensionality. A 50k-word one-hot vocabulary is 50,000-dimensional and says nothing about similarity (every word is orthogonal to every other); a 300-d word vector says a great deal.
3. **Relational encoding.** The coordinates themselves are arbitrary — an embedding space is meaningful only up to rotation and relative to itself. Vectors from two different models (or two versions of one model) are not comparable. This single fact drives a surprising amount of system design (section 14).

### Three things people mean by "embedding"

| Term | What it is | Trained for |
|---|---|---|
| **Embedding layer / table** | A lookup matrix `E` of shape (vocabulary size × `d`) mapping ids (tokens, users, products) to vectors | Whatever the enclosing model is trained for; every transformer and recommender has one |
| **Embedding model / encoder** | A whole network mapping a variable-length object to one vector | Similarity, search, clustering — usually via a contrastive objective |
| **Representation** | Any intermediate activation (a hidden state) | Nothing in particular — a by-product of the model's own objective |

The most common practical confusion is treating the third as if it were the second: pulling hidden states out of an LLM and expecting them to behave like a retrieval embedding. They don't, for reasons covered in section 3.

### Why it works: the distributional hypothesis

Harris (1954) and Firth (1957): *you shall know a word by the company it keeps.* The general form is that an object's identity is captured by the pattern of contexts it appears in — words in sentences, products in baskets, users in sessions, nodes in graphs, residues in proteins, patches in images. Anything with co-occurrence structure can be embedded, and most of this document is variations on one recipe: **define the contexts, then learn a low-rank map that predicts co-occurrence.**

### What an embedding is not

It is a lossy summary optimized for a notion of similarity fixed at training time. A general-purpose text embedder trained on web pairs encodes topical and semantic relatedness. It does not encode exact equality, negation, arithmetic, temporal ordering, or the difference between `invoice 4471` and `invoice 4417`. Section 15 catalogs these failure modes. And "similarity" is a design choice: what counts as similar for legal search, for duplicate detection, and for a recommender are three different geometries.

## 2. The classical lineage (why word2vec was matrix factorization all along)

**Sparse representations.** One-hot vectors, bag-of-words, TF-IDF weighting. Cosine on TF-IDF works well for lexical overlap but has no notion that *car* and *automobile* are related.

**LSA / LSI (Deerwester et al., 1990).** Build the term–document matrix `X`, take the truncated SVD `X ≈ U_k Σ_k V_kᵀ`. Rows of `U_k Σ_k` are word embeddings; rows of `V_k Σ_k` are document embeddings. By Eckart–Young the rank-`k` truncation is the best least-squares approximation. Synonyms co-occur with the same documents and collapse into nearby directions. This is the prototype of every embedding method: **an implicit co-occurrence matrix plus a low-rank factorization.**

**PMI and PPMI.** Raw counts are dominated by frequency. Pointwise mutual information `PMI(w,c) = log P(w,c) / (P(w)P(c))` measures association beyond chance; PPMI clips negatives to zero. SVD of a PPMI matrix (Bullinaria & Levy, 2007) gives strong word vectors.

**word2vec (Mikolov et al., 2013).** CBOW predicts a word from its context; skip-gram predicts context words from the center word. A full softmax over `|V|` is too expensive, so *skip-gram with negative sampling* (SGNS) turns it into binary classification: for each observed pair `(w, c)`, sample `k` random negative contexts from a smoothed unigram distribution (`P(w)^{3/4}`) and maximize

```
log σ(w·c) + Σ_{i=1..k} E_{c_i ~ P_n} [ log σ(−w·c_i) ]
```

**The key theoretical result (Levy & Goldberg, 2014).** At the optimum, SGNS satisfies `w·c = PMI(w,c) − log k`. word2vec implicitly factorizes a *shifted PMI matrix*. Neural word embeddings are LSA with a better matrix and a better loss (one that weights observed pairs and ignores the zeros). Follow-up work (Levy, Goldberg & Dagan, 2015) showed that hyperparameters — window size, subsampling, negative count, context-distribution smoothing — explain more of word2vec's advantage over count methods than the architecture does.

**GloVe (Pennington et al., 2014)** makes the factorization explicit: fit `w_i·c_j + b_i + b_j ≈ log X_ij` with a weighting `f(X_ij)` that caps high counts. Same family.

**fastText (Bojanowski et al., 2017)** represents a word as a bag of character n-grams and sums their vectors — handles morphology and out-of-vocabulary words, and is the conceptual ancestor of subword tokenization's role in modern models.

**Analogies and linear structure.** `king − man + woman ≈ queen`. Why linear? If vectors are (approximately) factorizations of log co-occurrence, then relations that *multiply* co-occurrence ratios *add* in log space (Arora et al., 2016 give a generative model; Ethayarajh, Duvenaud & Hirst, 2019 tie it to co-occurrence shift). Caveats: the standard 3CosAdd evaluation excludes the input words from the candidate set, which inflates results (Linzen, 2016), and many analogy types fail. But the linear structure is real, and it reappears in LLMs (section 9).

**Limits of static embeddings.** One vector per word type: *bank* is an average of river and finance. No composition beyond averaging, no word order.

> **Static embeddings are not dead.** (a) The embedding tables inside every transformer and recommender are static embeddings. (b) Distilled static models (e.g., Model2Vec, 2024) run 100–500× faster than transformers and retain most of the quality on many tasks — the right tool for high-throughput filtering, edge inference, or first-stage candidate generation.

## 3. Contextual representations: from ELMo to LLM hidden states

**ELMo (2018)** produced a token representation as a learned weighted sum of biLSTM layer states, different in every context. **BERT (2018)** did the same with a bidirectional transformer trained by masked language modeling; each token's final hidden state is a contextual embedding. Polysemy is solved by construction.

**What lives where.** Probing studies (Tenney et al., 2019; Jawahar et al., 2019) found lower layers capture surface and lexical information, middle layers syntax, upper layers semantics — with the final layer specializing toward the pretraining objective. For similarity tasks the last layer of a *raw* model is often not the best; averaging the last few layers or using a middle-upper layer frequently helps. Modern embedding models make this moot by fine-tuning the pooled output directly.

**Anisotropy — the narrow cone.** Ethayarajh (2019) showed contextual embeddings from BERT and GPT-2 occupy a narrow cone: two random words can have cosine > 0.6 in upper layers. Causes include a few "rogue" dimensions with huge variance (Timkey & van Schijndel, 2021), frequency effects that push frequent and rare tokens into different regions (Gao et al., 2019, "representation degeneration"), and the geometry induced by the softmax objective. Consequences: raw cosine similarities are uninformative and uncalibrated, and retrieval with raw BERT `[CLS]` vectors is often *worse* than averaged GloVe vectors (Reimers & Gurevych, 2019). Remedies: mean-centering plus removal of top principal components (*All-but-the-Top*, Mu & Viswanath, 2018), whitening (Su et al., 2021), flow-based mappings (BERT-flow), and — most effectively — contrastive fine-tuning, which pushes representations toward uniformity on the sphere (section 4).

**Pooling.** `[CLS]` works only if the model was trained to use it (NSP or a contrastive head). Mean pooling over tokens is the robust default for bidirectional encoders. For **decoder-only (causal) models**, only the last token has attended to the whole input, so either use last-token pooling (typically an appended EOS after an instruction template) or convert the model to bidirectional attention and mean-pool (LLM2Vec, NV-Embed — section 4). Mean-pooling a causal model is subtly wrong: early tokens' states know nothing about later tokens.

**Inside an LLM.** The embedding matrix `E ∈ ℝ^{|V|×d_model}` initializes the residual stream (`E[token]` plus positional information); attention and MLP blocks add to it layer by layer; the unembedding `W_U` maps the final residual to logits. Small models often tie `W_U = Eᵀ` (Press & Wolf, 2017). The *logit lens* (nostalgebraist, 2020) applies `W_U` to intermediate residuals to read off what the model "would predict" at each layer — evidence that the residual stream keeps a roughly consistent basis across depth. Undertrained rows of `E` produce *glitch tokens* (Rumbelow & Watkins, 2023, "SolidGoldMagikarp"): vocabulary entries rare in training data whose embeddings sit near initialization and trigger bizarre behavior.

**Tokenization determines what gets embedded.** Numbers, code identifiers, product codes, and rare names are split into fragments the model must recombine — part of why embedding models are weak on exact identifiers (section 15).

**Positional embeddings.** Learned absolute (BERT, GPT-2); sinusoidal (original transformer); **RoPE** (rotary: rotate query/key pairs by an angle proportional to position so `q·k` depends only on relative offset — the default in modern LLMs and long-context embedders); **ALiBi** (a linear bias on attention scores, used by MosaicBERT and jina-embeddings-v2 for 8k context). RoPE extension methods (position interpolation, NTK-aware scaling, YaRN) matter when a long-context embedder is trained at one length and served at another — quality at the advertised maximum length is rarely equal to quality at 512 tokens.

---
# Part II — Modern text embedding models

## 4. From representation to embedding model: the training recipe

### Bi-encoders, cross-encoders, and what sits between

- **Bi-encoder.** `f(q)` and `g(d)` are computed independently; `score = f(q)·g(d)`. Documents are pre-embedded and indexed; a query costs one forward pass plus an ANN lookup. Loses token-level interaction between query and document.
- **Cross-encoder.** `h([q; d])` attends over both jointly. Far more accurate at relevance, but O(N) forward passes per query and nothing to index. Used as a *reranker* over the top-k from a first-stage retriever.
- **Late interaction** (ColBERT, section 5) pre-computes per-token document vectors and defers a cheap interaction to query time.

Retrieve-then-rerank is the standard production pipeline (section 15).

**Sentence-BERT (Reimers & Gurevych, 2019)** established the pattern: a pretrained encoder, mean pooling, and a pairwise/contrastive objective on NLI (entailment pairs as positives, contradictions as hard negatives) and STS.

### The canonical modern recipe

E5, GTE, BGE, Nomic, Arctic, Jina, Qwen3-Embedding, and the hosted models all follow variants of this:

1. **Backbone.** A pretrained bidirectional encoder (BERT/XLM-R family, often with RoPE or ALiBi extended to 8k tokens), or a decoder LLM (Mistral-7B, Qwen, Gemma).
2. **Weakly supervised contrastive pre-training** on 10⁸–10⁹ naturally occurring pairs: (title, body), (question, answer), (citation context, cited abstract), (post, reply), (docstring, code), (query, clicked result). Pairs are mined from the web with heuristic filtering — E5's *consistency filtering* keeps a pair only if a preliminary model ranks the positive in the top-k among random candidates. Very large in-batch negatives (effective batches of 16k–64k via GradCache and cross-device negatives).
3. **Supervised fine-tuning** on curated data with hard negatives: MS MARCO, Natural Questions, HotpotQA, NLI, FEVER, Quora duplicates, plus multilingual sets (MIRACL, Mr. TyDi). Typically around a million examples.
4. **Instruction or prefix conditioning.** `query: ` / `passage: ` (E5), or natural-language task instructions (Instructor, E5-Mistral, Qwen3-Embedding, Gemini's task types). One model then serves asymmetric retrieval, symmetric STS, clustering, and classification with different geometry.
5. **Optional: LLM-synthesized data and distillation.** E5-Mistral generated hundreds of thousands of (instruction, query, positive, hard negative) tuples across ~90 languages and dozens of task types; Gecko used an LLM to generate queries and *relabel* positives and negatives; cross-encoder or LLM-reranker distillation is now routine.
6. **Optional: Matryoshka training** (section 5), sparse and multi-vector heads (BGE-M3), and *model merging* of checkpoints trained on different data mixtures (Qwen3-Embedding).

### The contrastive objective

InfoNCE, also called NT-Xent or *multiple negatives ranking loss* in sentence-transformers:

```
L = −(1/B) Σ_i log  exp(s(q_i, d_i⁺)/τ)
                     ─────────────────────────────────────────────────────────────
                     exp(s(q_i, d_i⁺)/τ) + Σ_{j≠i} exp(s(q_i, d_j⁺)/τ) + Σ_h exp(s(q_i, d_h⁻)/τ)
```

- `s` is cosine (or dot product). **Temperature `τ`** is small for cosine (0.01–0.05): it sharpens the softmax so the loss concentrates on the hardest negatives. Too small → instability and hubness; learnable `τ` is common (CLIP).
- **In-batch negatives** make every other example's positive a free negative, so signal scales with batch size — hence the obsession with huge batches. **GradCache** (Gao et al., 2021) decouples batch size from GPU memory by caching representations and back-propagating in chunks.
- **Hard negatives** are passages that look relevant but aren't. Sources: BM25 top-k (lexically similar), earlier-checkpoint dense retrieval (ANCE refreshes the negative index asynchronously during training), and cross-encoder-scored candidates. **False negatives** — "negatives" that are actually unlabeled positives — are the dominant noise source in retrieval training (RocketQA's *denoised* negatives discard candidates a cross-encoder scores as likely positives; a margin rule such as "skip if `s(neg) > s(pos) − margin`" is standard).
- **Symmetric vs asymmetric.** For STS and duplicate detection apply the loss in both directions; for retrieval, one direction plus a query-side prefix.
- **Distillation variants.** Margin-MSE (Hofstätter et al., 2020) matches the teacher's score *gap* between positive and negative — smoother than hard labels; TAS-B composes batches from topically similar queries so in-batch negatives are hard; KL distillation from listwise reranker scores.

**Alignment and uniformity (Wang & Isola, 2020).** Contrastive loss asymptotically optimizes two things: *alignment* (positives map close together) and *uniformity* (features spread evenly on the hypersphere, maximizing information). Anisotropy (section 3) is a uniformity failure, which is precisely what contrastive fine-tuning repairs. Both quantities are cheap diagnostics for a model you fine-tuned yourself.

### Self-supervised and unsupervised recipes (text but no pairs)

- **SimCSE** (Gao, Yao & Chen, 2021): pass the same sentence twice with different dropout masks and treat the two views as a positive pair. Remarkably effective; supervised SimCSE adds NLI.
- **Contriever** (Izacard et al., 2021): two random spans of the same document ("independent cropping") as a positive pair — unsupervised retrieval pretraining at scale.
- **TSDAE**: denoising autoencoder (delete words, reconstruct) for domain adaptation.
- **GPL** (Generative Pseudo-Labeling): generate pseudo-queries over your own corpus with a seq2seq model, label them with a cross-encoder, fine-tune — the workhorse recipe for adapting to an unlabeled domain.

### Decoder LLMs as embedders (the 2024–2026 shift)

- **Why:** larger backbones with far more pretraining knowledge, better instruction following, native multilinguality, long context. E5-Mistral-7B (2023/24) showed a decoder fine-tuned on synthetic plus public data could top MTEB. **NV-Embed** (2024) removed the causal mask during contrastive training and added a *latent attention* pooling layer. **LLM2Vec** (2024) gave a general recipe: enable bidirectional attention → masked-next-token-prediction adaptation → unsupervised SimCSE → optional supervised stage. **GritLM** (2024) unified generation and embedding in one model with a mode switch. **Qwen3-Embedding** (2025; 0.6B/4B/8B) and **Gemini Embedding** (2025) trained on synthetic data at scale and led MMTEB. By mid-2026 the top of the multilingual boards is dominated by LLM-backbone embedders (Tencent's KaLM-Embedding on Gemma-3-12B, Microsoft's 27B Harrier-OSS-v1, Qwen3-Embedding-8B — section 17).
- **Cost:** a 7B embedder is roughly 20–50× the compute of a 110M BERT-class model. For 10⁸ chunks that is a real bill. Common practice: use the large model to *generate* training data and *teach* (as a reranker or distillation target), then distill into a 100–600M model for the indexing path. The gap between a 0.6B and an 8B model on in-domain retrieval is usually smaller than the leaderboard gap suggests.
- **Pooling:** last-token (EOS) after an instruction template, or bidirectional conversion with mean pooling. Instructions go on the query side, not on documents.

### Fine-tuning your own

1. Build an eval set first (section 6).
2. Mine pairs from your data: query logs and clicks, ticket → resolution, question → answer in documentation, synthetic queries generated by an LLM over your chunks (with a cross-encoder or LLM judge to filter).
3. Start from a strong open model. Train with multiple-negatives ranking loss plus hard negatives (BM25 and dense-mined, cross-encoder filtered), learning rate 1e-5 to 2e-5, the largest batch memory allows, 1–3 epochs.
4. Expect +5–15 points nDCG@10 in specialized domains (legal, medical, code, internal jargon), sometimes more; expect *regression* on generic tasks — keep a general eval as a guardrail.
5. Consider LoRA adapters to keep one base with several task-specific heads (jina-embeddings-v3's design).

## 5. Representation formats beyond one dense vector

**Learned sparse — SPLADE (Formal et al., 2021).** Use the MLM head to project every token onto the vocabulary, apply `log(1 + ReLU(·))`, max-pool over positions → a sparse `|V|`-dimensional vector with *learned term weights and term expansion* (a passage about "cardiac" activates "heart"). A FLOPS regularizer keeps it sparse. Runs on ordinary inverted indexes; combines BM25's exact-match strength with learned semantics; strong out of domain. SPLADE-v3, uniCOIL, and the sparse head of BGE-M3 are the usual choices.

**Multi-vector / late interaction — ColBERT (Khattab & Zaharia, 2020).** Keep one vector per token (projected to ~128-d). Score is *MaxSim*:

```
score(q, d) = Σ_{i ∈ q}  max_{j ∈ d}  q_i · d_j
```

Retains token-level matching — rare terms, entities, out-of-domain robustness — while remaining pre-computable. Cost is storage: tokens × 128 dims per document; ColBERTv2's residual compression brings it to ~20–36 bytes per token and PLAID makes search fast with centroid pruning. **ColPali (2024)** applied late interaction to *document page images* using a vision-language model (each patch becomes a vector) and beat OCR-then-embed pipelines on visually rich documents — tables, figures, slides, scanned forms (the ViDoRe benchmark). **MUVERA (Google, 2024)** maps multi-vector sets to fixed-dimensional encodings so standard MIPS indexes can serve them. jina-embeddings-v4 and several 2025–26 models emit single- and multi-vector outputs from one backbone.

**Why multi-vector matters in theory.** Weller et al. (2025), *On the Theoretical Limitations of Embedding-Based Retrieval*, show that for single-vector embeddings of dimension `d`, the number of distinct top-k document subsets any query can retrieve is bounded (via the sign-rank of the query–document relevance matrix). Their **LIMIT** dataset — trivial queries such as "who likes quokkas?" over documents listing what people like — breaks state-of-the-art embedders (recall@100 strikingly low, often under 20%) while BM25 and multi-vector models do far better: the combinatorics of *which subset to return* exceed what a `d`-dimensional dot product can express. Implication: instruction-following and combinatorial retrieval ("docs mentioning A and B but not C") will not be solved by scaling single-vector embedders. Use sparse, multi-vector, or reasoning/agentic retrieval.

**Hybrid dense + lexical.** BM25 remains a strong, sometimes winning baseline for out-of-domain retrieval (BEIR, Thakur et al., 2021), identifiers, and rare terms. Combine with **Reciprocal Rank Fusion** (`score = Σ_i 1/(k + rank_i)`, `k ≈ 60`) or a learned combination of normalized scores, then rerank. Production retrieval should be hybrid by default.

**Matryoshka Representation Learning (Kusupati et al., 2022).** Train the loss simultaneously on nested prefixes of the vector (first 64, 128, 256, … `d` dims) so truncated vectors are themselves good embeddings, with information front-loaded. Supported by OpenAI text-embedding-3, Nomic, Gemini Embedding, Jina, Cohere embed-v4, and most 2025–26 models. Enables *adaptive retrieval*: shortlist with 256-d vectors on a small fast index, rescore with full vectors. Truncating 1024 → 256 typically costs 1–3 nDCG points.

**Model-level quantization.** int8 scalar quantization (4× smaller, ~0–1% loss); **binary** (32× smaller, Hamming distance; typically retains ~90–96% of quality if you rescore the top candidates with float vectors). Combined with MRL: 64–100× storage reduction. Distinct from index-level product quantization (section 13).

**Long context and the dilution problem.** 8k–32k-token embedders exist (BGE-M3, nomic-embed, jina-v3, Voyage), but one vector for a 30-page document is an average that loses specifics; fine-grained question recall drops relative to chunking. Fixes that restore document context to chunks:

- **Late chunking** (Jina, 2024): run the whole document through the transformer so every token attends to full context, *then* mean-pool per chunk. Chunks keep document-level referents (what "it" refers to, which product the section is about).
- **Contextual retrieval** (Anthropic, 2024): prepend a short LLM-written context to each chunk ("This chunk is from the Q2 report of ACME, section on churn…") before embedding and before BM25 indexing; reported ~49% fewer failed retrievals at top-20, ~67% fewer with reranking.
- **Contextual Document Embeddings** (Morris & Rush, 2024): condition the embedding on corpus-level statistics so what is *distinctive within this corpus* is emphasized — an embedding-model analogue of IDF.

## 6. Evaluation

**Benchmarks.** MTEB (Muennighoff et al., 2022) covers eight task families — retrieval, reranking, STS, classification, clustering, pair classification, summarization, bitext mining. MMTEB (Enevoldsen et al., 2025) expanded to ~500 tasks across 250+ languages, and the leaderboard split into task- and language-specific boards; MTEB v2 scores are not comparable with v1. BEIR is the zero-shot retrieval suite (18 heterogeneous datasets). Retrieval metrics: **nDCG@10** (graded, position-discounted), **Recall@k** (the one that matters for RAG: did the answer make it into the context?), MRR@10, MAP. STS: Spearman correlation between cosine and human judgments.

**How to read a leaderboard.**

1. The overall average hides task-level trade-offs. A model that tops clustering may be mediocre for retrieval; read the column that matches your workload.
2. Benchmark-adjacent training and outright contamination are common; the top ten reshuffles monthly and 1–2 point differences are noise for your purposes.
3. Domain shift is large. On FinMTEB (finance), top MTEB models dropped ~8 points and the ranking changed; a 2026 production comparison found the winner on the team's own data (a BGE-large model) ranked eleventh on MTEB.
4. **Reasoning-intensive retrieval** (BRIGHT, 2024 — queries whose relevant documents don't look like the query, e.g., a coding question answered by a post about a structurally similar but superficially different problem) leaves every embedder far below ceiling. **Instruction-following retrieval** (FollowIR) shows most models nearly ignore instructions.

**Build your own eval.** 100–500 (query, relevant chunk) judgments from real query logs or SME-written questions; LLM-assisted relevance labeling with a human audit; measure Recall@k at the `k` you will actually pass to the LLM, with your chunking. Re-run on every change to chunker, model, or index parameters. This is the highest-leverage investment in a retrieval system, and the only defense against the leaderboard pathologies above.

**Component vs end-to-end.** The final RAG metric is answer quality, but retrieval recall is the cheap, measurable bottleneck that correlates with it. Track both; when answer quality moves, check recall first.

---
# Part III — The geometry of embedding spaces

## 7. Similarity measures, normalization, calibration

**The identity that ties the metrics together.** For any vectors, `‖a − b‖² = ‖a‖² + ‖b‖² − 2 a·b`; for unit vectors this is `2 − 2 cos(a, b)`. So on normalized vectors, nearest by Euclidean = nearest by cosine = nearest by dot product. Use the metric the model was trained with — most modern text embedders use cosine and emit unit vectors.

**When magnitude matters.** Unnormalized dot products let the norm carry information. In recommenders, item norm correlates with popularity (a useful prior). Some dense retrievers (DPR) were trained with dot product, and document norm ends up encoding "how many queries this could answer." Normalizing throws that away — which may be what you want or not. Maximum inner product search reduces to nearest-neighbor search by appending one coordinate (`√(M² − ‖x‖²)`, Bachrach et al., 2014; Shrivastava & Li, 2014), which is how graph indexes support dot product.

**Cosine is not a universal similarity.** Steck, Ekanadham & Kallus (2024) showed that for embeddings from regularized matrix factorization, cosine similarity can be arbitrary: the objective is invariant to per-dimension rescalings that cosine is not. The lesson generalizes — similarity is meaningful only under the geometry the training loss induced. Contrastive models trained with cosine are fine; embeddings pulled from a model trained for something else are not.

**Calibration.** A cosine of 0.8 means nothing across models. Score distributions are model- and anisotropy-specific (raw BERT gives 0.9 for unrelated sentences; a contrastive model gives 0.2). Every threshold in your system — semantic-cache hits, dedup cut-offs, "no relevant document" abstention — must be tuned per model on held-out data and re-tuned when the model changes. If you need probabilities, fit a small calibrator (Platt scaling, isotonic regression) on judged pairs, or use a cross-encoder.

## 8. High-dimensional phenomena

**Concentration of distances.** For high-dimensional data with roughly independent coordinates, the ratio of nearest to farthest neighbor distance tends to 1 (Beyer et al., 1999) — nearest neighbors become meaningless. Learned embeddings escape this because their coordinates are far from independent: the **intrinsic dimensionality** of text embeddings (estimated with TwoNN or MLE estimators) is typically in the tens even when `d` is 768–4096. This is also why aggressive truncation (MRL, PCA) works.

**Hubness (Radovanović et al., 2010).** In high-d spaces some points become nearest neighbors of a disproportionate number of others (hubs), while others are never retrieved (anti-hubs). Hubs are usually points near the data mean. Severe in cross-modal and cross-lingual retrieval. Remedies: centering; **CSLS** (cross-domain similarity local scaling, Conneau et al., 2018 — penalize candidates that are close to everything); mutual nearest neighbors; inverted softmax. The production symptom: the same few chunks surface for every query.

**Isotropy metrics.** Average cosine between random pairs (≈0 when isotropic); the spectrum of the covariance matrix (effective rank, participation ratio); IsoScore. Improving isotropy helps *raw* models; contrastive-trained models are already near the optimum for their task — perfect isotropy is not the goal, uniformity subject to alignment is.

**Johnson–Lindenstrauss.** Any `n` points in `ℝ^d` can be projected by a random Gaussian matrix into `k = O(log n / ε²)` dimensions while preserving every pairwise distance within a factor of `(1 ± ε)`. Consequences: random projection to a few hundred dimensions is nearly free for nearest-neighbor purposes; PCA does better by using structure; MRL does better still by training for it. JL is also why LSH works, and why superposition (section 9) is possible.

**Dimension is not quality.** A 4096-d vector from a 7B model is not proportionally "richer." `d` is mostly a capacity knob traded against storage and latency, with fast-diminishing returns above a few hundred dimensions — *except* that, by the Weller et al. bound, `d` sets a hard ceiling on combinatorial retrieval expressivity. That is the one principled argument for larger `d`.

## 9. Linear structure, features, superposition — where embeddings meet interpretability

**The linear representation hypothesis** (Park, Choe & Veitch, 2023; roots in word analogies and probing): high-level concepts are represented as *directions* in activation space, a concept's presence is a projection onto its direction, and interventions are additions along it (activation steering: add a "refusal" or "honesty" direction). Evidence: linear probes work, steering works, analogies work.

**Categorical and hierarchical concepts** (Park et al., 2024): under an appropriate "causal inner product" (a whitening of the unembedding space), categorical concepts such as {mammal, bird, fish} form simplices, and hierarchical relations (mammal ⊂ animal) become *orthogonal* directions. The geometry of an ontology is literally encoded as orthogonality in the model — a useful mental model for anyone who thinks about domain models and embeddings together.

**Superposition** (Elhage et al., 2022, *Toy Models of Superposition*): a network with `d` dimensions can represent `m ≫ d` sparse features by assigning them nearly orthogonal (not exactly orthogonal) directions and tolerating small interference — JL guarantees exponentially many such directions exist. This explains polysemantic neurons and predicts that the "true" features are recoverable by sparse dictionary learning. **Sparse autoencoders** (Bricken et al., 2023; Templeton et al., 2024, *Scaling Monosemanticity*; Gao et al., 2024) decompose residual-stream activations into tens of thousands of interpretable features — an over-complete, sparse *re-embedding of the dense embedding*. Beyond interpretability, SAE features have been used for retrieval and controllable embeddings, and they let you audit what an embedding model is keying on: topic, style, formatting, or the thing you actually care about.

**Universal geometry.** The **Platonic Representation Hypothesis** (Huh et al., 2024): representations of different models, even across modalities, become increasingly similar (measured by mutual kNN alignment) as they scale, converging toward a shared statistical model of the world. **vec2vec** (Jha et al., 2025) pushes to the strong form: an unsupervised translator (adversarial + cycle-consistency, no paired data) maps embeddings from model A's space into model B's with cosine up to ~0.9 — well enough to run attribute inference and inversion on the translated vectors. Two implications: (a) migrating between embedding models without full re-embedding may become feasible (today it is not production-reliable); (b) the obscurity of your embedding model is not a security control.

**Embedding inversion.** Vec2Text (Morris et al., 2023) recovers input text from embeddings by iterative correction: ~92% exact reconstruction of 32-token inputs from OpenAI ada-002 vectors, and most of the content of longer inputs; follow-ups cover multilingual and other models. **Treat vectors as the data.** Same access control, encryption at rest, residency, retention, and deletion obligations as the source text. Adding noise degrades inversion but also degrades retrieval; the practical control is access control on the store, not obfuscation of the vectors.

## 10. Non-Euclidean and structured embedding spaces

**Hyperbolic embeddings.** Trees have exponentially many nodes at depth `r`; Euclidean balls grow polynomially in radius, hyperbolic balls exponentially — so hyperbolic space embeds hierarchies with low distortion in few dimensions. **Poincaré embeddings** (Nickel & Kiela, 2017) placed WordNet's noun hierarchy in 5–10 dimensions with lower distortion than Euclidean in 200; the Lorentz model (2018) trains more stably; hyperbolic GNNs and hyperbolic vision-language models (MERU, 2023) followed. Use when the data *is* a taxonomy, ontology, or org chart and you need is-a geometry: norm encodes depth/generality, angle encodes branch.

**Order and box embeddings.** Represent concepts as regions so that containment models entailment and hypernymy: order embeddings (Vendrov et al., 2016), box embeddings (Vilnis et al., 2018). Probabilistic box lattices give calibrated `P(A | B)` from volume overlap — useful where "dog ⊂ mammal" must be transitive, which a symmetric cosine cannot express. **Gaussian embeddings** (Vilnis & McCallum, 2015; probabilistic CLIP variants) represent an object as a distribution; variance models ambiguity.

**Knowledge-graph embeddings.** Triples `(h, r, t)`. **TransE**: `h + r ≈ t` (elegant; fails on one-to-many and symmetric relations). **DistMult / ComplEx**: bilinear scoring; ComplEx handles asymmetry through complex conjugation. **RotatE**: `r` is a rotation in complex space, modeling symmetry, antisymmetry, inversion, and composition. Used for link prediction and as a similarity signal in entity resolution. Limits: transductive (new entities need retraining), blind to textual attributes. Modern practice combines text embeddings of entity descriptions with relational GNNs (R-GCN, CompGCN) or LLM-based completion. In an enterprise ontology, KGEs are a *signal* for completion and matching, never the source of truth.

**Graph node embeddings.** **DeepWalk / node2vec** (2014/2016): random walks produce "sentences" of nodes, then skip-gram; node2vec's return/in-out parameters `p, q` interpolate between BFS-like (structural role) and DFS-like (community) neighborhoods. Transductive. **GNNs** (GCN, GraphSAGE, GAT, GIN) compute *inductive* embeddings by message passing over node features — a `k`-layer GNN's node embedding summarizes a `k`-hop neighborhood. **Over-smoothing**: with depth, node embeddings converge and become indistinguishable — the graph analogue of anisotropy — mitigated by residual connections, normalization, or decoupling propagation from transformation. Graph transformers need positional/structural encodings (Laplacian eigenvectors, random-walk encodings): literal positional embeddings for graphs. In learned physics simulators (MeshGraphNets and successors), each mesh node's latent is an embedding of local physical state and geometry; the same over-smoothing and Weisfeiler–Lehman expressivity limits apply, which is why multi-scale and hierarchical message passing is used to capture long-range interactions.

---

# Part IV — Beyond text

## 11. Vision and multimodal

**CLIP** (Radford et al., 2021): image and text encoders trained with a symmetric InfoNCE on 400M (image, caption) pairs; zero-shot classification by embedding "a photo of a {class}" prompts. ALIGN scaled to noisier data; OpenCLIP/LAION reproduced it openly. **SigLIP** (Zhai et al., 2023) replaced the batch softmax with a pairwise sigmoid loss — no global normalization across the batch, better behavior at both small and very large batch sizes; **SigLIP 2** (2025) added captioning-based pretraining, self-distillation, and multilingual data.

**The modality gap** (Liang et al., 2022): image and text embeddings from CLIP-style models occupy two separate cones offset by a nearly constant vector, caused by initialization (the two encoders start in different regions) and *preserved* by the contrastive loss at low temperature. Consequences: image–image and text–text similarities live on different scales from cross-modal ones; thresholds don't transfer between modes; you can shift one modality toward the other to trade off tasks. Natively multimodal models trained on interleaved inputs shrink the gap but don't eliminate it — evaluate cross-modal thresholds separately.

**Self-supervised vision** (DINOv2, 2023; DINOv3, 2025; MAE): features trained without text via self-distillation or masked reconstruction. Stronger than CLIP for dense and geometric tasks (segmentation, depth, visual-similarity retrieval), since CLIP is biased toward whatever captions describe.

**Document images.** ColPali (2024, section 5): per-patch vectors from a vision-language model plus late interaction, directly on page images. Beats OCR → chunk → embed on tables, figures, slides, and scans. Cost is ~1000 vectors per page; use a multi-vector index or pooled/compressed variants.

**Natively multimodal, single-space embedders (2025–2026).** Cohere embed-v4 (text + images, 128k context, int8/binary output), Voyage multimodal-3, jina-embeddings-v4 (text + images, single- and multi-vector), and **Gemini Embedding 2** (Google, March 2026): one model maps text, images, video, audio, and PDFs — including *interleaved* combinations — into one 3072-d Matryoshka space (truncatable to 1536/768), with an 8k-token text window, task instructions, and native audio understanding rather than transcription. This collapses the old CLIP-for-images + BERT-for-text + ASR-for-audio pipelines into a single index. The fine print still matters: per-request limits on video length and PDF pages, per-call cost, and incompatibility with any existing text-only index.

**Audio and speech.** wav2vec 2.0 and HuBERT (self-supervised speech representations); CLAP (audio–text contrastive); Whisper encoder states as general audio features; speaker embeddings (x-vectors, ECAPA-TDNN) for verification. **Video:** CLIP per frame plus temporal pooling, VideoCLIP, InternVideo, and now unified models.

## 12. Code, science, and structured data

**Code.** Trained on (docstring, function), (query, code), and (code, equivalent code) pairs: CodeBERT and UniXcoder → CodeT5+ → voyage-code-3, Qwen3-Embedding, Nomic Embed Code. Chunk by function/class (structure-preserving) rather than by token count. For coding agents, pure embedding search over a repository underperforms a good symbol index plus grep for navigation, but wins on intent queries ("find where retries are implemented"); the practical answer is hybrid.

**Science.** Protein language models (ESM-2, ESM-3) yield residue- and sequence-level embeddings that predict structure and function; molecular embeddings from SMILES transformers (ChemBERTa) or GNNs over molecular graphs compete with classical fingerprints (ECFP remains a strong baseline); time-series foundation models (Chronos, MOMENT) produce window embeddings for similarity search and anomaly detection; geospatial embeddings (AlphaEarth Foundations, 2025) embed every 10-m pixel of the planet's surface.

**Tabular and categorical.** *Entity embeddings* (Guo & Berkhahn, 2016) replace one-hot categoricals with learned vectors inside a neural net; the space often exposes structure (postal codes cluster geographically). Sizing rules of thumb: `d ≈ min(600, round(1.6 · n^0.56))` (fastai) or `∝ n^0.25`. For gradient-boosted trees, target encoding usually wins; embeddings matter when categoricals interact and cardinality is huge.

**High-cardinality ids and the hashing trick.** With 10⁸–10⁹ ids (users, ads, URLs), embedding tables become the largest part of the model — terabytes in industrial recommenders. Hashing trick (Weinberger et al., 2009): hash the id into a smaller table and accept collision noise. **Quotient–remainder / compositional embeddings** (Shi et al., 2020): represent an id by combining rows from two small tables (`id mod m`, `id div m`). Mixed-dimension embeddings give popular ids more capacity; tensor-train compression (TT-Rec) shrinks the tables. Infrastructure (DLRM, TorchRec) shards tables across GPUs row- or table-wise, with all-to-all lookups the dominant communication cost.

**Recommender systems are embedding systems.** Matrix factorization (Koren et al., 2009) *is* user and item embeddings with dot-product scoring. **Two-tower retrieval** (Covington et al., 2016; Yi et al., 2019) trains user and item towers with sampled softmax and a log-Q correction for the sampling bias of popular items; item2vec/prod2vec apply word2vec to sessions and baskets; sequential models (GRU4Rec, SASRec, BERT4Rec) produce a user embedding from a behavior sequence. Serving = ANN over item embeddings, then a heavier ranker. Recurring problems: popularity bias, cold start (fall back to content embeddings of item text/images), and drift of the embedding tables. The two-tower model is also the industrial ancestor of bi-encoder text retrieval.

---
# Part V — Systems

## 13. Approximate nearest neighbor (ANN) search

**Exact search first.** Brute force is O(N·d) per query. With SIMD or a GPU it is fine to roughly 10⁵–10⁶ vectors (FAISS flat on GPU handles millions at low-millisecond latency). Always benchmark exact: it is the recall ceiling and is often good enough.

**Graph-based — HNSW** (Malkov & Yashunin, 2016/2018). A multi-layer proximity graph: sparse upper layers act as express lanes, the dense bottom layer holds every point; search is greedy best-first, descending layers. Parameters: `M` (edges per node; memory ∝ N·M), `efConstruction` (build quality), `efSearch` (query-time beam width — *the* recall/latency knob). Recall@10 of 0.95–0.99 at ~1 ms over millions of vectors in RAM. Downsides: RAM-resident (≈ `N × (4d + 8M)` bytes plus overhead — 10M × 1024-d is ~40 GB of raw vectors alone), slow builds, deletes via tombstones that degrade the graph until a rebuild. **DiskANN / Vamana** (Subramanya et al., 2019): a single-layer graph with a compressed in-memory copy and full vectors on SSD — billion-scale on one machine; FreshDiskANN handles updates.

**Clustering and quantization — IVF and PQ.** *Inverted file* (IVF): k-means into `n_list` centroids; at query time scan only the `n_probe` nearest lists. *Product quantization* (Jégou et al., 2011): split `d` into `m` sub-vectors, k-means each into 256 centroids → one byte per sub-vector (a 4 KB fp32 vector becomes 64–128 bytes); distances via lookup tables (asymmetric distance computation). IVF-PQ with re-ranking by full vectors is the classical FAISS billion-scale configuration; OPQ rotates the space first for lower distortion. **ScaNN** (Guo et al., 2020) uses *anisotropic* quantization — penalize error in the direction parallel to the vector, which is what perturbs inner products — and is the reference design for MIPS at scale. **Scalar quantization** (int8, binary) is the simpler cousin, now standard in most vector stores, usually with rescoring of the top-k′ by full-precision vectors.

**LSH.** Random hyperplanes → binary codes. Theoretical guarantees, trivially shardable and streamable, but inferior recall/efficiency to graphs and IVF-PQ on static corpora. Still used for near-duplicate detection (SimHash) and streaming.

**Trade-off summary.**

| Method | Recall/latency | Memory | Updates | Notes |
|---|---|---|---|---|
| Exact (flat) | ceiling | N·d floats | trivial | fine to ~10⁶, GPU to ~10⁷ |
| HNSW | best | highest | inserts OK, deletes degrade | rebuild periodically |
| IVF-PQ | tunable | lowest | re-train on drift | needs representative training sample |
| DiskANN | very good | SSD-resident | FreshDiskANN | billion-scale single node |
| LSH | weakest | low | streaming | dedup, sketches |

Compare on recall-vs-QPS curves (ann-benchmarks.com style). A 2–5% recall loss at 10× throughput is typical and usually acceptable — but measure the downstream effect, not just recall.

**Filtered search — the hard part in practice.** Real queries carry predicates: tenant, date range, document type, access-control lists. *Post-filtering* (search, then drop non-matching results) collapses recall when the filter is selective — the top-100 may contain zero matches. *Pre-filtering* (materialize the matching set, then exact-search it) is correct but expensive unless the set is small. *Filter-aware traversal* (Qdrant, Weaviate, Vespa; ACORN, Patel et al., 2024) keeps exploring through non-matching nodes to preserve graph connectivity. *Partitioning* (one index per tenant) is the pragmatic answer for multi-tenant products. Test with your most selective filters — that is where systems fail silently.

**Do the storage arithmetic before choosing anything.**

```
N = 50M chunks × 1024 dims × 4 bytes  = 205 GB  (fp32)
int8                                   =  51 GB
binary                                 = 6.4 GB
MRL → 256 dims + int8                  = 12.8 GB
HNSW graph overhead (M = 16, ~8M B/vec) ≈ 6.4 GB
Embedding cost: 50M chunks × ~300 tokens = 15B tokens
  hosted API at ~$0.02–0.13 / M tokens  → ~$300–$2,000 per full re-embed
  self-hosted 100M-param model: ~1–5k chunks/s per GPU → hours
  self-hosted 7B model: days, or a cluster
```

Re-embedding cost is the main reason model upgrades are rare and painful. Plan for it from day one.

## 14. Where the vectors live — and the embedding model as schema

**Options.** Libraries (FAISS, hnswlib, ScaNN, DiskANN, usearch) for in-process search. General databases with vector indexes: pgvector/pgvectorscale, Elasticsearch/OpenSearch (natural for hybrid lexical + dense), Vespa (strongest for multi-vector and ranking expressions), MongoDB Atlas, Redis, DuckDB/LanceDB for local and analytical work. Dedicated vector databases: Milvus, Qdrant, Weaviate, Pinecone, Turbopuffer, Chroma.

**Guidance.** Below ~10–50M vectors at moderate QPS, the vector index inside the database you already run (Postgres, Elastic) minimizes operational surface and keeps vectors transactionally consistent with the rows they describe. Dedicated systems earn their keep at scale, under heavy filtering, for multi-tenancy, or for multi-vector. The embedding model, chunker, and reranker matter far more for quality than the choice of store.

**The embedding model is a schema, not an implementation detail.** Vectors from model A cannot be compared with vectors from model B — nor from A at a different version, dimension, normalization, or prefix template. Consequences:

1. Store, per collection and ideally per vector, the model id and version, dimensions, normalization, prefix/instruction template, and chunker version.
2. Any model change is a full re-embed plus a **blue/green index swap**: dual-write the new vectors, validate on your eval set, cut over, keep the old index through the rollback window.
3. Hosted providers deprecate models. Design for it (abstraction layer, eval set, budget line for re-embeds).
4. *Backward-compatible training* (Shen et al., 2020; standard in face recognition) trains a new model whose embeddings are compatible with the old gallery, avoiding re-indexing — an option for in-house models, rarely offered by vendors. vec2vec-style translation (section 9) is a research path, not yet a production one.

**Operations.** Batch embedding throughput is dominated by padding — sort inputs by length. Cache embeddings keyed by `hash(model_version + template + text)`. Deduplicate before embedding. ONNX/TensorRT and int8 weights for CPU serving of small models. Monitor the query-score distribution and Recall@k on a fixed canary set to catch drift (model change, chunker change, index degradation after deletes). Track filtered-query latency separately from unfiltered.

## 15. Retrieval pipelines for RAG and agents

### Chunking — the most underrated variable

- **Fixed-size with overlap.** 256–512 tokens, 10–20% overlap. Simple, robust baseline.
- **Recursive / structure-aware.** Split on headings → paragraphs → sentences; keep tables and code blocks intact; carry the heading path into the chunk text.
- **Semantic chunking.** Split where consecutive-sentence embedding similarity drops. Modest gains, high cost.
- **Proposition-level.** An LLM rewrites text into atomic facts. Best precision, most expensive.
- **Parent–child (small-to-big).** Embed small chunks for precision; return the enclosing section to the LLM for context.
- **Late chunking and contextual retrieval** (section 5) to restore document context to chunks.

Chunk size interacts with the embedder (quality degrades well before the advertised max length), the LLM's context budget, and the question type (factoid vs synthesis). Evaluate two or three chunkers on your eval set; it often moves recall more than the choice of embedding model.

### Query side

- **Get the instruction/prefix template exactly right.** A missing `query: ` prefix on E5 or the wrong task type silently costs points.
- **Rewriting and decomposition** by an LLM for multi-part questions; **multi-query** (several paraphrases, union of results); **step-back** prompting for over-specific questions.
- **HyDE** (Gao et al., 2022): generate a hypothetical answer and embed *that*, closing the query–document asymmetry gap. Useful zero-shot; adds latency; can hallucinate the wrong topic.
- **Metadata extraction into filters** — dates, product names, document types parsed from the query. The single most effective fix for "the answer was in a 2025 document but retrieval returned the 2019 version."

### Fusion and reranking

Retrieve top-50–200 from dense + BM25 (+ learned sparse) → RRF → cross-encoder reranker (bge-reranker-v2, Qwen3-Reranker, Cohere Rerank, jina-reranker, or an LLM listwise reranker for the top-20) → top 5–10 to the LLM → optional MMR for diversity. Rerankers are the cheapest large quality gain in most pipelines; late interaction is the middle ground where rerank latency is too high.

### Known failure modes of dense retrieval — and what to do

| Failure | Why | Mitigation |
|---|---|---|
| Negation, antonyms ("laptops *without* touchscreen") | embeddings mostly ignore "not" (NevIR, 2023) | reranker or LLM filter |
| Exact identifiers, part numbers, error strings, names | tokenized into fragments; semantics ≠ identity | BM25 / learned sparse + metadata filters; hybrid by default |
| Numbers, dates, ranges, comparisons | not geometrically encoded | extract to structured filters; route to SQL |
| Very long documents | dilution | chunk; late chunking; contextual retrieval |
| Multi-hop, reasoning-intensive questions | no single chunk resembles the question | iterative/agentic retrieval, HyDE, reasoning rerankers |
| Combinatorial / instruction-following retrieval | single-vector capacity bound (section 5) | multi-vector, sparse, agentic |
| Cross-lingual queries | weak alignment, hubness | multilingual models trained on bitext; check hubness per language |
| Jargon, distribution shift | training data mismatch | fine-tune (section 4) or add synthetic domain pairs |
| "Everything scores 0.7" | anisotropic model or wrong pooling | check alignment/uniformity; use a contrastive-trained model |

### Embeddings elsewhere in agent systems

- **Tool and skill retrieval.** With hundreds of tools, embed descriptions plus example invocations and retrieve the top-k into the prompt. Measure tool-selection accuracy, not cosine.
- **Memory.** Episodic memory stores embedded (often LLM-summarized) events; retrieval combines similarity with recency and importance (Park et al., 2023, *Generative Agents*); periodic reflection/consolidation reduces hubness of frequently retrieved memories. Structured facts belong in a database, not a vector store.
- **Semantic caching.** Match new queries to previously answered ones by similarity. Dangerous false positives ("Q3 2024 revenue" vs "Q3 2025 revenue") — require high thresholds, exact match on extracted entities and dates, or an LLM verifier; measure cache precision explicitly.
- **Routing and intent classification.** kNN over labeled example embeddings is a strong, cheap, instantly updatable classifier (add an example, behavior changes, no retraining); abstain on distance to the nearest labeled example.
- **Few-shot example selection.** Retrieve the most similar labeled examples into the prompt (kNN in-context learning) — reliably better than random examples.
- **Deduplication and clustering.** Near-duplicate removal before indexing (cosine above ~0.95 on the same model, or MinHash for lexical duplicates); HDBSCAN/k-means for topic discovery; UMAP for *visualization only* — it distorts distances.
- **Entity resolution and schema matching.** Embeddings of records or columns as a blocking key and one similarity feature among several (string similarity, rules). Good recall for "same entity, different spelling"; never the sole decision signal in a governed identity model.
- **Anomaly detection.** Distance to nearest cluster centroid or kNN density over embeddings of logs, tickets, or transactions.
- **Evaluation and analysis.** Cluster model outputs to find failure modes; embedding-based similarity metrics (BERTScore family) for generation evaluation, remembering they reward topical rather than factual agreement.

### Security and governance

- **Inversion** (section 9): vectors are the data. Same controls as source text.
- **Access control at retrieval time.** Filter by ACL before or within search (pre-filter or filter-aware index). Never rely on the LLM to withhold a retrieved chunk. Per-tenant indexes for hard isolation.
- **Poisoning** (PoisonedRAG, Zou et al., 2024): an attacker who can write to the corpus — public web, shared drives, ticket systems — crafts passages optimized to be retrieved for target queries and steer the answer. Defenses: provenance-weighted ranking, outlier and duplicate detection on new content, isolating untrusted sources, and treating retrieved text as untrusted input (prompt-injection hardening).
- **Extraction via similarity APIs.** Membership inference and corpus extraction through repeated queries; rate-limit and log.

---

# Part VI — Frontier and practice

## 16. Where the research is moving (2025–2026)

1. **LLM-backbone embedders and the blurring of generation and representation.** GritLM-style unified models; embeddings from reasoning-tuned models; *test-time compute for retrieval* — retrievers trained on synthetic reasoning-intensive queries (ReasonIR) and listwise LLM rerankers that reason — closing the BRIGHT gap.
2. **Capacity limits and the multi-vector renaissance.** The Weller et al. bound reframes "which representation?" as a capacity question. Expect more late-interaction (ColBERT/ColPali), sparse + dense hybrids, and fixed-dimensional encodings (MUVERA) in production, and more benchmarks built to expose combinatorial failure (LIMIT, FollowIR, BRIGHT).
3. **Native multimodality in one space.** Gemini Embedding 2 (text/image/video/audio/PDF), Cohere embed-v4, jina-v4, Voyage multimodal. Open questions: the modality gap under interleaved training, cross-modal calibration, and whether unified spaces sacrifice within-modality fidelity.
4. **Universal geometry, translation, and privacy.** Platonic convergence; vec2vec translation; steadily better inversion; regulators treating embeddings as personal data. Watch for embedding-space migration tooling and privacy-preserving retrieval (encrypted/secure ANN, differential privacy on embeddings).
5. **Context- and corpus-aware embeddings.** Late chunking, contextual retrieval, Contextual Document Embeddings — representations that depend on what else is in the index; document-level versus passage-level.
6. **Scaling laws for retrieval.** Fang et al. (2024) find contrastive entropy follows power laws in model size and annotation volume, with annotation *quality* trading off against model size. Practical reading: money spent on hard-negative quality and synthetic data often beats scaling the encoder.
7. **Efficiency.** MRL + binary + int8 for 64–100× smaller indexes; static-embedding distillation for 100–500× faster encoding; 100–600M distilled models within a few points of 7B+ teachers; on-device retrieval.
8. **Interpretable and controllable embeddings.** SAEs over embedding models; feature-level retrieval and steering ("match on this feature, ignore style"); concept-geometry results (Park et al.) as a bridge between embedding spaces and explicit ontologies.
9. **Long-context LLMs versus retrieval.** Million-token contexts reduce the need for fine chunking but not for retrieval: cost, latency, freshness, permissions, and the lost-in-the-middle effect keep it central. The role shifts toward coarse selection of larger units plus **agentic search** — the model issues multiple searches, reads, and refines — where embeddings are one tool among lexical search, structured queries, and graph traversal.
10. **Dynamic and continual settings.** Temporal embeddings, drift detection, backward-compatible training so indexes survive model updates, streaming ANN with cheap deletes.

## 17. Model landscape snapshot (mid-2026) — verify before choosing

- **Hosted:** OpenAI text-embedding-3 small/large (MRL; cheap; text only). Google Gemini Embedding 2 (multimodal, 3072-d MRL, 8k, task instructions; text-only gemini-embedding-001 still available). Cohere embed-v4 (text + image, 128k context, int8/binary). Voyage (voyage-3.5 / 3-large with 32k context, voyage-code-3, voyage-multimodal-3). Jina (v4 multimodal and multi-vector; v5-text-small under Apache-2.0).
- **Open weights:** Qwen3-Embedding 0.6B/4B/8B (strong multilingual and code, Apache-2.0). Tencent KaLM-Embedding-Gemma3-12B (led the MMTEB multilingual board as of July 2026). Microsoft Harrier-OSS-v1 (27B, MIT; top MTEB v2 English scores). BGE-M3 (dense + sparse + multi-vector, 8k, 100+ languages — still the multilingual workhorse). bge-large-en-v1.5, E5 and GTE families (small, fast, well understood). Nomic Embed v2-MoE, Snowflake Arctic-Embed 2.0, Stella/Jasper. ColBERTv2 and ColPali for late interaction; SPLADE-v3 for learned sparse.
- **Rerankers:** bge-reranker-v2 (m3, gemma), Qwen3-Reranker, jina-reranker, Cohere Rerank 3.5, mxbai-rerank; LLM listwise rerankers for the final top-20.
- **Rules of thumb.** For English enterprise RAG, a 100–600M open model + hybrid retrieval + a reranker usually lands within a few points of the best 8–27B model *on your data* at a fraction of the cost. Choose multilingual models by the languages you actually serve. Choose multimodal only when you need cross-modal search — the price is higher and the space is incompatible with your text index. Open versus hosted is now mostly a data-residency, deprecation-risk, and operations decision rather than a quality one. Always run the top three candidates on your own eval set.

## 18. Decision guide and pitfalls

**Choosing a model.** Task type (asymmetric retrieval vs symmetric similarity vs clustering) · languages · domain · the input length you need and quality *at that length* · latency and throughput budget · hosted vs self-hosted (residency, deprecation) · license · dimensions and MRL/quantization support (do the storage math) · instruction/prefix format · multimodal needs · reranker pairing · fine-tuning feasibility.

**Building the pipeline.** (1) Eval set → (2) chunker → (3) hybrid retrieval (dense + BM25/sparse) → (4) metadata filters → (5) reranker → (6) `k` tuned to the LLM → (7) monitoring canaries → (8) versioned index with blue/green swap.

**Pitfalls (each of these has cost someone a quarter).**

- Wrong or missing prefix/instruction template.
- Mixing vectors from two models or two versions in one index.
- Comparing cosine values, or reusing thresholds, across models.
- Post-filtering with selective filters.
- Single-vector retrieval for combinatorial or instruction-following queries.
- Embedding whole documents instead of chunks.
- Trusting MTEB rank order over your own eval.
- Semantic-cache false positives on entities and dates.
- No budget or plan for re-embedding when the model changes.
- Vectors stored with weaker access controls than the source text.
- UMAP distances treated as real distances.
- Mean-pooling a causal (decoder) model.
- Cosine on embeddings that weren't trained with cosine.
- IVF/PQ codebooks trained on a non-representative sample, never retrained after drift.
- Deletes silently degrading HNSW recall (no rebuild schedule).

---

# Appendices

## A. Key formulas

```
Cosine              cos(a,b) = a·b / (‖a‖ ‖b‖)
Distance identity   ‖a − b‖² = ‖a‖² + ‖b‖² − 2 a·b   (= 2 − 2 cos(a,b) for unit vectors)
PMI                 PMI(w,c) = log [ P(w,c) / (P(w) P(c)) ]
SGNS optimum        w·c = PMI(w,c) − log k                      (Levy & Goldberg, 2014)
GloVe               min Σ_ij f(X_ij) (w_i·c_j + b_i + b_j − log X_ij)²
InfoNCE             L = −log [ e^{s(q,d⁺)/τ} / Σ_{d ∈ {d⁺} ∪ negatives} e^{s(q,d)/τ} ]
Alignment           E_{(x,y)~pos} ‖f(x) − f(y)‖²
Uniformity          log E_{x,y~data} e^{−2‖f(x) − f(y)‖²}
MaxSim (ColBERT)    score = Σ_{i∈q} max_{j∈d} q_i·d_j
RRF                 score(d) = Σ_{rankers i} 1 / (k + rank_i(d)),  k ≈ 60
JL lemma            k = O(log n / ε²) dims preserve all pairwise distances within (1 ± ε)
MIPS → NN           x' = [x, √(M² − ‖x‖²)],  q' = [q, 0]   (M ≥ max ‖x‖)
```

## B. Working defaults

| Knob | Typical range | Notes |
|---|---|---|
| Contrastive temperature `τ` (cosine) | 0.02–0.05 | learnable in CLIP-style training |
| Fine-tuning LR | 1e-5 – 2e-5 | 1–3 epochs; warm-up 5–10% |
| Hard negatives per query | 1–7 | cross-encoder-filtered; margin ≈ 0.05–0.1 |
| Chunk size | 256–512 tokens | 10–20% overlap; keep structure intact |
| First-stage `k` | 50–200 | before RRF and reranking |
| Final `k` to LLM | 5–10 | tune on answer quality |
| HNSW `M` / `efConstruction` / `efSearch` | 16–32 / 200 / 64–256 | `efSearch` is the runtime recall knob |
| IVF `n_list` / `n_probe` | 4√N – 16√N / 1–5% of lists | retrain codebooks on drift |
| MRL truncation | 1024 → 256–512 | ~1–3 nDCG points; rescore with full dims |
| Dedup threshold | cosine ≳ 0.95 | model-specific; validate |

## C. Reading list by topic

**Foundations.** Deerwester et al. 1990 (LSA) · Mikolov et al. 2013 (word2vec) · Pennington et al. 2014 (GloVe) · Levy & Goldberg 2014 (SGNS as implicit matrix factorization) · Levy, Goldberg & Dagan 2015 · Bojanowski et al. 2017 (fastText) · Arora et al. 2016 (RAND-WALK) · Linzen 2016 (analogy evaluation).

**Contextual representations.** Peters et al. 2018 (ELMo) · Devlin et al. 2018 (BERT) · Tenney et al. 2019 · Ethayarajh 2019 (anisotropy) · Gao et al. 2019 (representation degeneration) · Mu & Viswanath 2018 (All-but-the-Top) · Timkey & van Schijndel 2021 (rogue dimensions) · Press & Wolf 2017 (tied embeddings) · nostalgebraist 2020 (logit lens) · Rumbelow & Watkins 2023 (glitch tokens) · Su et al. 2021 (RoPE) · Press et al. 2021 (ALiBi).

**Embedding models and training.** Reimers & Gurevych 2019 (SBERT) · Karpukhin et al. 2020 (DPR) · Xiong et al. 2020 (ANCE) · Qu et al. 2020 (RocketQA) · Hofstätter et al. 2020/2021 (Margin-MSE, TAS-B) · Gao et al. 2021 (GradCache; SimCSE) · Izacard et al. 2021 (Contriever) · Wang & Isola 2020 (alignment/uniformity) · Wang et al. 2022 (E5) · Su et al. 2022 (Instructor) · Xiao et al. 2023 (BGE / C-Pack) · Li et al. 2023 (GTE) · Wang et al. 2023 (E5-Mistral, synthetic data) · Lee et al. 2024 (Gecko; NV-Embed) · BehnamGhader et al. 2024 (LLM2Vec) · Muennighoff et al. 2024 (GritLM) · Chen et al. 2024 (BGE-M3) · Zhang et al. 2025 (Qwen3-Embedding) · Lee et al. 2025 (Gemini Embedding) · Google DeepMind 2026 (Gemini Embedding 2).

**Representation formats.** Formal et al. 2021 (SPLADE) · Khattab & Zaharia 2020, Santhanam et al. 2021/2022 (ColBERT, v2, PLAID) · Faysse et al. 2024 (ColPali) · Dhulipala et al. 2024 (MUVERA) · Kusupati et al. 2022 (Matryoshka) · Günther et al. 2024 (late chunking) · Anthropic 2024 (contextual retrieval) · Morris & Rush 2024 (Contextual Document Embeddings) · Weller et al. 2025 (theoretical limitations; LIMIT) · Thakur et al. 2021 (BEIR) · Cormack et al. 2009 (RRF).

**Evaluation.** Muennighoff et al. 2022 (MTEB) · Enevoldsen et al. 2025 (MMTEB) · Su et al. 2024 (BRIGHT) · Weller et al. 2023/2024 (NevIR, FollowIR) · Tang & Yang 2024 (FinMTEB).

**Geometry and interpretability.** Beyer et al. 1999 · Radovanović et al. 2010 (hubness) · Conneau et al. 2018 (CSLS) · Steck et al. 2024 (cosine caveat) · Park, Choe & Veitch 2023 (linear representation hypothesis) · Park et al. 2024 (categorical/hierarchical geometry) · Elhage et al. 2022 (superposition) · Bricken et al. 2023, Templeton et al. 2024, Gao et al. 2024 (sparse autoencoders) · Huh et al. 2024 (Platonic Representation Hypothesis) · Jha et al. 2025 (vec2vec) · Morris et al. 2023 (Vec2Text inversion) · Nickel & Kiela 2017/2018 (Poincaré, Lorentz) · Vilnis et al. 2018 (box embeddings) · Bordes et al. 2013, Trouillon et al. 2016, Sun et al. 2019 (TransE, ComplEx, RotatE) · Grover & Leskovec 2016 (node2vec) · Hamilton et al. 2017 (GraphSAGE).

**Multimodal and other domains.** Radford et al. 2021 (CLIP) · Zhai et al. 2023 (SigLIP) · Tschannen et al. 2025 (SigLIP 2) · Liang et al. 2022 (modality gap) · Oquab et al. 2023 (DINOv2) · Guo & Berkhahn 2016 (entity embeddings) · Weinberger et al. 2009 (hashing trick) · Shi et al. 2020 (compositional embeddings) · Naumov et al. 2019 (DLRM) · Covington et al. 2016, Yi et al. 2019 (two-tower) · Lin et al. 2023 (ESM-2).

**Systems.** Malkov & Yashunin 2018 (HNSW) · Jégou et al. 2011 (PQ) · Guo et al. 2020 (ScaNN) · Subramanya et al. 2019 (DiskANN) · Patel et al. 2024 (ACORN) · Shen et al. 2020 (backward-compatible training) · Gao et al. 2022 (HyDE) · Zou et al. 2024 (PoisonedRAG) · Park et al. 2023 (Generative Agents memory) · Fang et al. 2024 (scaling laws for dense retrieval).

## D. Glossary

- **Anisotropy** — embeddings concentrated in a narrow cone; random pairs have high cosine.
- **ANN** — approximate nearest neighbor search; trades recall for speed and memory.
- **Bi-encoder / cross-encoder** — independent encoding with a dot-product score vs joint encoding with full attention.
- **Contrastive learning** — pull positives together, push negatives apart (InfoNCE and relatives).
- **Hard negative** — a non-relevant candidate that looks relevant; the main lever in retrieval training.
- **Hubness** — some points are nearest neighbors of disproportionately many others.
- **Hybrid retrieval** — fusing lexical (BM25/sparse) and dense results, typically with RRF.
- **Late interaction** — per-token vectors with query-time MaxSim scoring (ColBERT).
- **Learned sparse** — vocabulary-sized sparse vectors with learned weights and expansion (SPLADE).
- **Matryoshka (MRL)** — training so that prefixes of the vector are usable embeddings.
- **Modality gap** — offset between modality clusters in a joint image–text space.
- **Pooling** — collapsing token vectors into one (mean, CLS, last token, latent attention).
- **Product quantization (PQ)** — compressing vectors into per-sub-vector centroid ids.
- **Reranker** — second-stage model (usually a cross-encoder) re-scoring first-stage candidates.
- **Sign-rank bound** — the limit on distinct top-k subsets a `d`-dimensional dot product can express.
- **Superposition** — more features than dimensions, stored as nearly orthogonal directions.
- **Temperature (`τ`)** — softmax sharpness in contrastive loss; smaller = focus on hardest negatives.
- **Uniformity / alignment** — the two quantities contrastive loss optimizes on the hypersphere.
