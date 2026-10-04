# Embeddings: A Comprehensive Primer

*This primer goes from distributional semantics to the frontier of representation learning, retrieval systems and interpretability. This text is from August 2026.*

---

## How to read this

- **Part I: Foundations.** This part tells what an embedding is, the classical lineage, and what the contextual representations inside transformers are in fact. If you know this material, read it fast. Section 3 has details that most people miss.
- **Part II: Modern text embedding models.** This part gives the training recipe, the representation formats other than a single dense vector, and the method of evaluation.
- **Part III: Geometry.** This part covers similarity, high-dimensional phenomena, linear structure and superposition, and non-Euclidean spaces. This part is where embeddings meet interpretability.
- **Part IV: Beyond text.** This part covers vision, multimodal embeddings, code, science, recommenders and graphs.
- **Part V: Systems.** This part covers ANN search, vector stores, and retrieval pipelines for RAG and agents. It includes their failure modes and security.
- **Part VI: Frontier.** This part covers the open problems and the research direction of 2025–2026, a snapshot of the model landscape, and a checklist for decisions.
- **Appendices.** The appendices hold formulas, worked numbers, a list of papers to read and a glossary.

Notation: $d$ is the embedding dimension, and $N$ is the corpus size. $q$/$d$ is the query and the document. $\cos(a,b) = a \cdot b \,/\, (\lVert a \rVert \, \lVert b \rVert)$ is the cosine. $\tau$ is the temperature, and $\lvert V \rvert$ is the vocabulary size.

---

# Part I — Foundations

## 1. What an embedding is (and isn't)

An **embedding** is a learned map $f\colon X \to \mathbb{R}^d$ from a space of objects $X$ into a vector space. The objects can be tokens, sentences, images, users, graph nodes, molecules or audio clips. Its construction has one goal. The geometric relations in $\mathbb{R}^d$ (inner products, distances, directions) must encode the relations among the objects that are important for some task.

That definition contains three ideas:

1. **Continuity.** The embedding puts discrete objects in a continuous space. In that space, "similar" is computable and, most importantly, differentiable. Gradients flow through the embedding. Thus training can learn it end-to-end with the component that uses it.
2. **Compression.** $d$ is much smaller than the raw dimensionality. A 50k-word one-hot vocabulary is 50,000-dimensional. It says nothing about similarity, because every word is orthogonal to every other word. A 300-d word vector says much.
3. **Relational encoding.** The coordinates themselves are arbitrary. An embedding space is meaningful only up to rotation, and only relative to itself. You cannot compare vectors from two different models, or from two versions of one model. This one fact drives more of the system design than you expect (section 14).

### Three things people mean by "embedding"

| Term | What it is | Trained for |
|---|---|---|
| **Embedding layer / table** | A lookup matrix $E$ of shape (vocabulary size × $d$). It maps ids (tokens, users, products) to vectors. | The objective of the model that contains it. Every transformer and every recommender has one. |
| **Embedding model / encoder** | A full network that maps a variable-length object to one vector. | Similarity, search and clustering, usually with a contrastive objective. |
| **Representation** | Any intermediate activation (a hidden state). | Nothing specific. It is a by-product of the objective of the model. |

The most frequent confusion in practice is to use the third as if it is the second. That is, you take the hidden states out of an LLM, and you expect them to behave like a retrieval embedding. They do not, and section 3 gives the reasons.

### Why it works: the distributional hypothesis

Harris (1954) and Firth (1957) gave the idea: *you shall know a word by the company it keeps.* In the general form, the pattern of contexts in which an object occurs holds the identity of that object. Examples are words in sentences, products in baskets, users in sessions, nodes in graphs, residues in proteins and patches in images. You can embed anything that has a co-occurrence structure. Most of this document is a variation on one recipe: **define the contexts, then learn a low-rank map that predicts co-occurrence.**

### What an embedding is not

An embedding is a lossy summary. Training optimizes it for one notion of similarity, and that notion does not change after training. A general-purpose text embedder trained on web pairs encodes topical and semantic relatedness. It does not encode exact equality, negation, arithmetic, temporal order, or the difference between `invoice 4471` and `invoice 4417`. Section 15 lists these failure modes.

Also, "similarity" is a design decision. Similarity for legal search, similarity for duplicate detection and similarity for a recommender are three different geometries.

## 2. The classical lineage (why word2vec was matrix factorization all along)

**Sparse representations.** These are one-hot vectors, bag-of-words and TF-IDF weights. Cosine on TF-IDF works well for lexical overlap. But it has no notion of a relation between *car* and *automobile*.

**LSA / LSI (Deerwester et al., 1990).** Build the term–document matrix $X$. Then take the truncated SVD $X \approx U_k \Sigma_k V_k^{\top}$. The rows of $U_k \Sigma_k$ are word embeddings, and the rows of $V_k \Sigma_k$ are document embeddings. By the Eckart–Young theorem, the rank-$k$ truncation is the best least-squares approximation.

Synonyms co-occur with the same documents. Thus their vectors collapse into directions that are near each other. LSA is the prototype of every embedding method: **an implicit co-occurrence matrix plus a low-rank factorization.**

**PMI and PPMI.** Frequency dominates the raw counts. Pointwise mutual information, $\operatorname{PMI}(w,c) = \log P(w,c) / (P(w)P(c))$, measures the association above chance. PPMI clips the negative values to zero. The SVD of a PPMI matrix (Bullinaria and Levy, 2007) gives strong word vectors.

**word2vec (Mikolov et al., 2013).** CBOW predicts a word from its context. Skip-gram predicts the context words from the center word. A full softmax over $\lvert V \rvert$ costs too much. Thus *skip-gram with negative sampling* (SGNS) changes the task into a binary classification. For each observed pair ${(w, c)}$, sample $k$ random negative contexts from a smoothed unigram distribution ($P(w)^{3/4}$), and maximize

$$
\log \sigma(w \cdot c) + \sum_{i=1}^{k} \mathbb{E}_{c_i \sim P_n} \left[ \log \sigma(-w \cdot c_i) \right]
$$

**The key theoretical result (Levy and Goldberg, 2014).** At the optimum, SGNS satisfies $w \cdot c = \operatorname{PMI}(w,c) - \log k$. That is, word2vec implicitly factorizes a *shifted PMI matrix*. Neural word embeddings are LSA with a better matrix and a better loss, which gives weights to the observed pairs and ignores the zeros. Later work (Levy, Goldberg and Dagan, 2015) showed that the hyperparameters explain more of the advantage of word2vec over count methods than the architecture does. These hyperparameters are window size, subsampling, negative count and context-distribution smoothing.

**GloVe (Pennington et al., 2014)** makes the factorization explicit. It fits $w_i \cdot c_j + b_i + b_j \approx \log X_{ij}$, with a weight function $f(X_{ij})$ that puts a limit on high counts. It is in the same family.

**fastText (Bojanowski et al., 2017)** represents a word as a bag of character n-grams, and it adds their vectors together. It handles morphology and out-of-vocabulary words. It is also the conceptual ancestor of the role that subword tokenization has in modern models.

**Analogies and linear structure.** $\text{king} - \text{man} + \text{woman} \approx \text{queen}$. Why is this linear? If the vectors are (approximately) factorizations of log co-occurrence, then relations that *multiply* co-occurrence ratios *add* in log space. Arora et al. (2016) give a generative model, and Ethayarajh, Duvenaud and Hirst (2019) tie it to co-occurrence shift.

There are caveats. The standard 3CosAdd evaluation removes the input words from the candidate set, and this inflates the results (Linzen, 2016). Also, many analogy types fail. But the linear structure is real, and it occurs again in LLMs (section 9).

**Limits of static embeddings.** A static embedding has one vector per word type. Thus *bank* is an average of river and finance. There is no composition other than an average, and there is no word order.

> **Static embeddings are not dead.** First, the embedding tables inside every transformer and recommender are static embeddings. Second, distilled static models (for example, Model2Vec, 2024) run 100–500× faster than transformers. They keep most of the quality on many tasks. Thus they are the correct tool for high-throughput filters, edge inference, or first-stage candidate generation.

## 3. Contextual representations: from ELMo to LLM hidden states

**ELMo (2018)** made a token representation as a learned weighted sum of biLSTM layer states. This representation is different in every context. **BERT (2018)** did the same with a bidirectional transformer, trained with masked language modeling. The final hidden state of each token is a contextual embedding. By construction, this design solves polysemy.

**What lives where.** Studies with probes (Tenney et al., 2019, and Jawahar et al., 2019) found that lower layers hold surface and lexical information. Middle layers hold syntax and upper layers hold semantics, while the final layer specializes toward the pretraining objective. For similarity tasks, the last layer of a *raw* model is frequently not the best. An average of the last few layers, or a middle-upper layer, frequently helps. Modern embedding models make this question unimportant, because they fine-tune the pooled output directly.

**Anisotropy: the narrow cone.** Ethayarajh (2019) showed that contextual embeddings from BERT and GPT-2 occupy a narrow cone. In upper layers, two random words can have a cosine > 0.6. The causes include:

- a few "rogue" dimensions with a large variance (Timkey and van Schijndel, 2021),
- frequency effects that push frequent and rare tokens into different regions (Gao et al., 2019, "representation degeneration"),
- the geometry that the softmax objective causes.

The consequences: raw cosine similarities give no information, and they have no calibration. Also, retrieval with raw BERT `[CLS]` vectors is frequently *worse* than retrieval with averaged GloVe vectors (Reimers and Gurevych, 2019). One remedy is subtraction of the mean plus removal of the top principal components (*All-but-the-Top*, Mu and Viswanath, 2018). Other remedies are whitening (Su et al., 2021) and flow-based mappings (BERT-flow). The most effective remedy is contrastive fine-tuning, which pushes representations toward uniformity on the sphere (section 4).

**Pooling.** `[CLS]` works only if training taught the model to use it (NSP or a contrastive head). Mean pooling over tokens is the robust default for bidirectional encoders.

For **decoder-only (causal) models**, only the last token attends to the whole input. Thus, select one of two methods. Use last-token pooling, usually on an EOS token that you append after an instruction template. Alternatively, convert the model to bidirectional attention and use mean pooling (LLM2Vec, NV-Embed, section 4). Mean pooling of a causal model is incorrect in a subtle way: the states of early tokens know nothing about later tokens.

**Inside an LLM.** The embedding matrix $E \in \mathbb{R}^{\lvert V \rvert \times d_{\text{model}}}$ initializes the residual stream with $E[\text{token}]$ plus positional information. The attention and MLP blocks add to it layer by layer, and the unembedding $W_U$ maps the final residual to logits. Small models frequently tie $W_U = E^{\top}$ (Press and Wolf, 2017).

The *logit lens* (nostalgebraist, 2020) applies $W_U$ to intermediate residuals to read the prediction of the model at each layer. It reads each layer as if that layer is the last one. This is evidence that the residual stream keeps an approximately consistent basis across depth.

Rows of $E$ with insufficient training make *glitch tokens* (Rumbelow and Watkins, 2023, "SolidGoldMagikarp"). These are vocabulary entries that are rare in the training data. Their embeddings stay near initialization, and these tokens cause strange behavior.

**Tokenization decides what the model embeds.** The tokenizer splits numbers, code identifiers, product codes and rare names into fragments. The model must then combine these fragments again. This is part of the reason why embedding models are weak on exact identifiers (section 15).

**Positional embeddings.** The options are:

- Learned absolute embeddings (BERT, GPT-2).
- Sinusoidal embeddings (the original transformer).
- **RoPE** (rotary). It rotates query/key pairs by an angle proportional to position, so $q \cdot k$ depends only on the relative offset. It is the default in modern LLMs and long-context embedders.
- **ALiBi** (a linear bias on attention scores). MosaicBERT and jina-embeddings-v2 use it for 8k context.

RoPE extension methods (position interpolation, NTK-aware scaling, YaRN) are important when you train a long-context embedder at one length and serve it at another length. The quality at the advertised maximum length is rarely equal to the quality at 512 tokens.

---
# Part II — Modern text embedding models

## 4. From representation to embedding model: the training recipe

### Bi-encoders, cross-encoders, and what sits between

- **Bi-encoder.** The model calculates ${f(q)}$ and ${g(d)}$ independently, and $\text{score} = f(q) \cdot g(d)$. You embed and index the documents in advance. A query costs one forward pass plus an ANN lookup. The bi-encoder loses the token-level interaction between the query and the document.
- **Cross-encoder.** ${h([q; d])}$ attends over both together. It is much more accurate at relevance. But it needs ${O(N)}$ forward passes per query, and it has nothing to index. You use it as a *reranker* over the top-k from a first-stage retriever.
- **Late interaction** (ColBERT, section 5) calculates per-token document vectors in advance. It delays a low-cost interaction until query time.

Retrieve-then-rerank is the standard pipeline in production (section 15).

**Sentence-BERT (Reimers and Gurevych, 2019)** set the pattern: a pretrained encoder, mean pooling, and a pairwise/contrastive objective on NLI and STS. In NLI, the entailment pairs are the positives, and the contradictions are the hard negatives.

### The canonical modern recipe

E5, GTE, BGE, Nomic, Arctic, Jina, Qwen3-Embedding and the hosted models all use variants of this recipe:

1. **Backbone.** The backbone is a pretrained bidirectional encoder or a decoder LLM. The encoder is from the BERT/XLM-R family, frequently with RoPE or ALiBi extended to 8k tokens. Examples of decoder LLMs are Mistral-7B, Qwen and Gemma.
2. **Weakly supervised contrastive pretraining.** The model trains on 10⁸–10⁹ pairs that occur naturally. The pairs are (title, body), (question, answer), (citation context, cited abstract), (post, reply), (docstring, code) and (query, clicked result). The pairs come from the web, with heuristic filters. E5 uses *consistency filtering*. It keeps a pair only if a preliminary model ranks the positive in the top-k among random candidates. The number of in-batch negatives is large: effective batches of 16k–64k, with GradCache and cross-device negatives.
3. **Supervised fine-tuning.** The model trains on curated data with hard negatives. The data sets are MS MARCO, Natural Questions, HotpotQA, NLI, FEVER and Quora duplicates, plus multilingual sets (MIRACL, Mr. TyDi). The usual size is approximately one million examples.
4. **Instruction or prefix conditioning.** E5 uses the prefixes `query: ` / `passage: `. Other models use natural-language task instructions (Instructor, E5-Mistral, Qwen3-Embedding, the task types of Gemini). Then one model serves asymmetric retrieval, symmetric STS, clustering and classification, each with a different geometry.
5. **Optional: LLM-synthesized data and distillation.** E5-Mistral generated hundreds of thousands of (instruction, query, positive, hard negative) tuples. These tuples cover ~90 languages and dozens of task types. Gecko used an LLM to generate queries and to *relabel* positives and negatives. Distillation from a cross-encoder or from an LLM reranker is now usual practice.
6. **Optional: Matryoshka training** (section 5). Other optional parts are sparse and multi-vector heads (BGE-M3), and *model merging* of checkpoints trained on different data mixtures (Qwen3-Embedding).

### The contrastive objective

The objective is InfoNCE. Its other names are NT-Xent and, in sentence-transformers, *multiple negatives ranking loss*:

$$
L = -\frac{1}{B} \sum_i \log \frac{\exp(s(q_i, d_i^{+})/\tau)}{\exp(s(q_i, d_i^{+})/\tau) + \sum_{j \ne i} \exp(s(q_i, d_j^{+})/\tau) + \sum_h \exp(s(q_i, d_h^{-})/\tau)}
$$

- $s$ is the cosine (or the dot product). **Temperature $\tau$** is small for cosine (0.01–0.05). It sharpens the softmax, so that the loss concentrates on the hardest negatives. A $\tau$ that is too small causes instability and hubness. A learnable $\tau$ is common (CLIP).
- **In-batch negatives** make the positive of every other example a free negative. Thus the signal scales with batch size, and this explains the strong focus on large batches. **GradCache** (Gao et al., 2021) makes batch size independent of GPU memory. It caches representations and does back-propagation in chunks.
- **Hard negatives** are passages that look relevant but are not relevant. Their sources are the BM25 top-k (lexically similar), dense retrieval from an earlier checkpoint, and candidates that a cross-encoder scored. ANCE uses the earlier-checkpoint source, and it refreshes the negative index asynchronously during training. **False negatives** are "negatives" that are in fact unlabeled positives, and they are the dominant source of noise in retrieval training. The *denoised* negatives of RocketQA discard candidates that a cross-encoder scores as likely positives. A margin rule such as "skip if $s(\text{neg}) > s(\text{pos}) - \text{margin}$" is standard.
- **Symmetric against asymmetric.** For STS and duplicate detection, apply the loss in both directions. For retrieval, apply it in one direction and add a query-side prefix.
- **Distillation variants.** Margin-MSE (Hofstätter et al., 2020) matches the score *gap* of the teacher between the positive and the negative. This is smoother than hard labels. TAS-B makes batches from queries that are similar in topic, so that the in-batch negatives are hard. A third variant is KL distillation from the scores of a listwise reranker.

**Alignment and uniformity (Wang and Isola, 2020).** Asymptotically, contrastive loss optimizes two things. The first is *alignment*: positives map near each other. The second is *uniformity*: features spread evenly on the hypersphere, and this maximizes the information. Anisotropy (section 3) is a failure of uniformity, and contrastive fine-tuning repairs exactly this failure. Both quantities are low-cost diagnostics for a model that you fine-tuned yourself.

### Self-supervised and unsupervised recipes (text but no pairs)

- **SimCSE** (Gao, Yao and Chen, 2021): pass the same sentence two times with different dropout masks. Then treat the two views as a positive pair. This method is highly effective. Supervised SimCSE adds NLI.
- **Contriever** (Izacard et al., 2021) uses two random spans of the same document as a positive pair ("independent cropping"). This method gives unsupervised retrieval pretraining at scale.
- **TSDAE** is a denoising autoencoder for domain adaptation. It deletes words, and then it reconstructs the input.
- **GPL** (Generative Pseudo-Labeling) has three steps. Generate pseudo-queries over your own corpus with a seq2seq model. Label them with a cross-encoder. Then fine-tune. This is the usual recipe to adapt a model to an unlabeled domain.

### Decoder LLMs as embedders (the 2024–2026 shift)

- **Why:** decoder LLMs have larger backbones with much more pretraining knowledge. They also obey instructions better, they are natively multilingual, and they have a long context. E5-Mistral-7B (2023/24) showed that a decoder, fine-tuned on synthetic plus public data, was able to top MTEB. **NV-Embed** (2024) removed the causal mask during contrastive training, and it added a *latent attention* pooling layer. **LLM2Vec** (2024) gave a general recipe in four stages. First, set the attention to bidirectional. Then do a masked-next-token-prediction adaptation, then unsupervised SimCSE, then an optional supervised stage. **GritLM** (2024) unified generation and embedding in one model with a mode switch. **Qwen3-Embedding** (2025, 0.6B/4B/8B) and **Gemini Embedding** (2025) trained on synthetic data at scale, and they led MMTEB. By mid-2026, LLM-backbone embedders dominate the top of the multilingual boards. Examples are Tencent's KaLM-Embedding on Gemma-3-12B, Microsoft's 27B Harrier-OSS-v1 and Qwen3-Embedding-8B (section 17).
- **Cost:** a 7B embedder needs approximately 20–50× the compute of a 110M BERT-class model. For 10⁸ chunks, that is a large bill. The common practice is to use the large model to *generate* training data and to *teach*, as a reranker or a distillation target. Then distill the large model into a 100–600M model for the indexing path. On in-domain retrieval, the gap between a 0.6B and an 8B model is usually smaller than the leaderboard gap suggests.
- **Pooling:** use last-token (EOS) pooling after an instruction template, or a bidirectional conversion with mean pooling. Put the instructions on the query side, not on the documents.

### Fine-tuning your own

1. Build an eval set first (section 6).
2. Mine pairs from your data. Sources are query logs and clicks, (ticket, resolution) pairs and (question, answer) pairs from documentation. Another source is synthetic queries that an LLM generates over your chunks. Use a cross-encoder or an LLM judge to filter them.
3. Start from a strong open model. Train with multiple-negatives ranking loss plus hard negatives (BM25 and dense-mined, cross-encoder filtered). Use a learning rate of 1e-5 to 2e-5, the largest batch that memory permits, and 1–3 epochs.
4. Expect +5–15 points nDCG@10 in specialized domains (legal, medical, code, internal jargon), and sometimes more. Expect a *regression* on generic tasks. Keep a general eval as a guardrail.
5. Think about LoRA adapters: they keep one base with several task-specific heads. This is the design of jina-embeddings-v3.

## 5. Representation formats beyond one dense vector

**Learned sparse: SPLADE (Formal et al., 2021).** SPLADE uses the MLM head to project every token onto the vocabulary. Then it applies $\log(1 + \operatorname{ReLU}(\cdot))$ and does a max-pool over the positions. The result is a sparse $\lvert V \rvert$-dimensional vector with *learned term weights and term expansion*. For example, a passage about "cardiac" activates "heart". A FLOPS regularizer keeps the vector sparse.

SPLADE runs on ordinary inverted indexes. It combines the exact-match strength of BM25 with learned semantics, and it is strong out of domain. SPLADE-v3, uniCOIL and the sparse head of BGE-M3 are the usual choices.

**Multi-vector / late interaction: ColBERT (Khattab and Zaharia, 2020).** ColBERT keeps one vector per token, projected to ~128-d. The score is *MaxSim*:

$$
\operatorname{score}(q, d) = \sum_{i \in q} \max_{j \in d} q_i \cdot d_j
$$

ColBERT keeps the match at the token level: rare terms, entities and out-of-domain robustness. Also, you can still calculate the document vectors in advance. The cost is storage: tokens × 128 dims per document. The residual compression of ColBERTv2 decreases this to ~20–36 bytes per token. PLAID makes the search fast with centroid pruning.

**ColPali (2024)** applied late interaction to *document page images*, with a vision-language model. Each patch becomes a vector. ColPali beat OCR-then-embed pipelines on visually rich documents: tables, figures, slides and scanned forms (the ViDoRe benchmark). **MUVERA (Google, 2024)** maps multi-vector sets to fixed-dimensional encodings, so that standard MIPS indexes can serve them. The jina-embeddings-v4 model and several 2025–26 models give single- and multi-vector outputs from one backbone.

**Why multi-vector matters in theory.** Weller et al. (2025), *On the Theoretical Limitations of Embedding-Based Retrieval*, show a bound for single-vector embeddings of dimension $d$. For these embeddings, the number of distinct top-k document subsets that any query can retrieve has a limit. The bound comes from the sign-rank of the query–document relevance matrix.

Their **LIMIT** dataset has simple queries such as "who likes quokkas?" over documents that list what people like. It breaks state-of-the-art embedders: recall@100 is surprisingly low, frequently under 20%. But BM25 and multi-vector models do much better. The reason is that the combinatorics of *which subset to return* exceed what a $d$-dimensional dot product can express.

The implication: larger single-vector embedders will not solve instruction-following retrieval and combinatorial retrieval ("docs mentioning A and B but not C"). Use sparse, multi-vector, or reasoning/agentic retrieval.

**Hybrid dense + lexical.** BM25 stays a strong baseline for out-of-domain retrieval (BEIR, Thakur et al., 2021), identifiers and rare terms. Sometimes it is the best method. Combine it with **Reciprocal Rank Fusion** ($\text{score} = \sum_i 1/(k + \operatorname{rank}_i)$, $k \approx 60$) or with a learned combination of normalized scores. Then rerank. Make retrieval in production hybrid by default.

**Matryoshka Representation Learning (Kusupati et al., 2022).** Train the loss at the same time on nested prefixes of the vector (the first 64, 128, 256, … up to $d$ dims). Then the truncated vectors are good embeddings themselves, and the first dimensions hold the most information. OpenAI text-embedding-3, Nomic, Gemini Embedding, Jina, Cohere embed-v4 and most 2025–26 models support it. It permits *adaptive retrieval*: make a shortlist with 256-d vectors on a small fast index, then rescore with full vectors. A truncation from 1024 to 256 usually costs 1–3 nDCG points.

**Model-level quantization.** int8 scalar quantization makes vectors 4× smaller, with a ~0–1% loss. **Binary** quantization makes them 32× smaller and uses Hamming distance. It usually keeps ~90–96% of the quality if you rescore the top candidates with float vectors. Together with MRL, the storage reduction is 64–100×. Model-level quantization is different from index-level product quantization (section 13).

**Long context and the dilution problem.** 8k–32k-token embedders exist (BGE-M3, nomic-embed, jina-v3, Voyage). But one vector for a 30-page document is an average, and it loses the specific details. The recall on fine-grained questions decreases relative to chunking. These solutions put the document context back into the chunks:

- **Late chunking** (Jina, 2024): run the whole document through the transformer, so that every token attends to the full context. *Then* do mean pooling for each chunk. The chunks keep document-level referents: what "it" refers to, and which product the section is about.
- **Contextual retrieval** (Anthropic, 2024): add a short LLM-written context to the start of each chunk. An example context: "This chunk is from the Q2 report of ACME, section on churn…". Do this before embedding and before BM25 indexing. The reported result is ~49% fewer failed retrievals at top-20, and ~67% fewer with reranking.
- **Contextual Document Embeddings** (Morris and Rush, 2024): condition the embedding on corpus-level statistics. Then the embedding gives more weight to what is *distinctive within this corpus*. This is an analogue of IDF for an embedding model.

## 6. Evaluation

**Benchmarks.** MTEB (Muennighoff et al., 2022) covers eight task families: retrieval, reranking, STS, classification, clustering, pair classification, summarization and bitext mining. MMTEB (Enevoldsen et al., 2025) expanded to ~500 tasks in 250+ languages. Also, the leaderboard split into task- and language-specific boards. You cannot compare MTEB v2 scores with v1 scores. BEIR is the zero-shot retrieval suite (18 heterogeneous datasets).

The retrieval metrics are **nDCG@10** (graded, position-discounted), **Recall@k**, MRR@10 and MAP. Recall@k is the important metric for RAG, because it answers this question: did the answer make it into the context? For STS, the metric is the Spearman correlation between cosine and human judgments.

**How to read a leaderboard.**

1. The overall average hides task-level trade-offs. A model that tops clustering can be mediocre for retrieval. Read the column that matches your workload.
2. Benchmark-adjacent training and actual contamination with benchmark data are common. The top ten changes order every month. For your purposes, differences of 1–2 points are noise.
3. Domain shift is large. On FinMTEB (finance), the top MTEB models decreased by ~8 points, and the ranking changed. A 2026 comparison in production found the winner on the team's own data: a BGE-large model. That model was eleventh on MTEB.
4. **Reasoning-intensive retrieval** (BRIGHT, 2024) leaves every embedder far below the ceiling. In BRIGHT, the relevant documents of a query do not look like the query. For example, a post answers a question about code. The post is about a problem that has a similar structure but looks different. **Instruction-following retrieval** (FollowIR) shows that most models almost ignore instructions.

**Build your own eval.** Collect 100–500 (query, relevant chunk) judgments from real query logs or from questions that SMEs wrote. Use LLM-assisted relevance labels with an audit by a person. Measure Recall@k at the $k$ that you will in fact pass to the LLM, with your chunking. Run the eval again after every change to the chunker, the model or the index parameters.

Your own eval set is the investment with the highest leverage in a retrieval system. It is also the only defense against the leaderboard pathologies in "How to read a leaderboard".

**Component against end-to-end.** The final RAG metric is answer quality. But retrieval recall is the low-cost, measurable bottleneck that correlates with answer quality. Track both. When answer quality changes, examine recall first.

---
# Part III — The geometry of embedding spaces

## 7. Similarity measures, normalization, calibration

**The identity that ties the metrics together.** For any vectors, $\lVert a - b \rVert^2 = \lVert a \rVert^2 + \lVert b \rVert^2 - 2\,a \cdot b$. For unit vectors, this is $2 - 2\cos(a, b)$. Thus, on normalized vectors, nearest by Euclidean = nearest by cosine = nearest by dot product. Use the metric that the model trained with. Most modern text embedders use cosine and give unit vectors.

**When magnitude matters.** Unnormalized dot products let the norm carry information. In recommenders, the item norm correlates with popularity, and this is a useful prior. Some dense retrievers (DPR) trained with the dot product. In these retrievers, the document norm then encodes "how many queries this could answer". Normalization removes that information, and you want that in some cases but not in others.

Maximum inner product search becomes a nearest-neighbor search if you add one coordinate ($\sqrt{M^2 - \lVert x \rVert^2}$, Bachrach et al., 2014, and Shrivastava and Li, 2014). This is how graph indexes support the dot product.

**Cosine is not a universal similarity.** Steck, Ekanadham and Kallus (2024) showed that cosine similarity can be arbitrary for embeddings from regularized matrix factorization. The reason is that the objective is invariant to per-dimension rescalings, but cosine is not. The lesson is general: similarity is meaningful only under the geometry that the training loss induced. Contrastive models that trained with cosine are fine. Embeddings taken from a model that trained for a different purpose are not.

**Calibration.** A cosine of 0.8 means nothing across models. Score distributions are specific to the model and to its anisotropy: raw BERT gives 0.9 for unrelated sentences, and a contrastive model gives 0.2. You must adjust every threshold in your system per model on held-out data, and adjust it again when the model changes. Examples are semantic-cache hits, dedup cut-offs and "no relevant document" abstention. If you need probabilities, fit a small calibrator (Platt scaling, isotonic regression) on judged pairs, or use a cross-encoder.

## 8. High-dimensional phenomena

**Concentration of distances.** For high-dimensional data with approximately independent coordinates, the ratio of the nearest to the farthest neighbor distance approaches 1 (Beyer et al., 1999). Thus nearest neighbors become meaningless. Learned embeddings do not have this problem, because their coordinates are far from independent. The **intrinsic dimensionality** of text embeddings, as TwoNN or MLE estimators estimate it, is usually in the tens, even when $d$ is 768–4096. This is also why aggressive truncation (MRL, PCA) works.

**Hubness (Radovanović et al., 2010).** In high-d spaces, some points become the nearest neighbors of a disproportionate number of other points (hubs). The search never retrieves some other points (anti-hubs). Hubs are usually points near the data mean. Hubness is severe in cross-modal and cross-lingual retrieval.

The remedies are subtraction of the mean, **CSLS**, mutual nearest neighbors and inverted softmax. CSLS (cross-domain similarity local scaling, Conneau et al., 2018) penalizes candidates that are near everything. In production, the symptom is that the same few chunks come up for every query.

**Isotropy metrics.** The metrics are the average cosine between random pairs (≈0 when isotropic), the spectrum of the covariance matrix (effective rank, participation ratio) and IsoScore. Better isotropy helps *raw* models. Contrastive-trained models are already near the optimum for their task. Perfect isotropy is not the goal. The goal is uniformity subject to alignment.

**Johnson–Lindenstrauss.** A random Gaussian matrix can project any $n$ points in $\mathbb{R}^d$ into $k = O(\log n / \varepsilon^2)$ dimensions, and keep every pairwise distance within a factor of $(1 \pm \varepsilon)$. The consequences: a random projection to a few hundred dimensions costs almost nothing for nearest-neighbor purposes. PCA does better, because it uses the structure. MRL does even better than PCA, because it trains for that structure. JL is also the reason why LSH works, and why superposition (section 9) is possible.

**Dimension is not quality.** A 4096-d vector from a 7B model is not proportionally "richer." $d$ is mostly a capacity knob that you trade against storage and latency. Above a few hundred dimensions, the returns decrease fast. There is one *exception*: by the Weller et al. bound, $d$ sets a hard ceiling on the expressivity of combinatorial retrieval. That is the one principled argument for a larger $d$.

## 9. Linear structure, features, superposition — where embeddings meet interpretability

**The linear representation hypothesis** (Park, Choe and Veitch, 2023) has its roots in word analogies and probes. It states three things. A model represents high-level concepts as *directions* in activation space. The presence of a concept is a projection onto its direction. An intervention is an addition along that direction (activation steering: add a "refusal" or "honesty" direction). The evidence: linear probes work, steering works, and analogies work.

**Categorical and hierarchical concepts** (Park et al., 2024): the result holds under an applicable "causal inner product" (a whitening of the unembedding space). Under it, categorical concepts such as {mammal, bird, fish} form simplices. Hierarchical relations (mammal ⊂ animal) become *orthogonal* directions. The model encodes the geometry of an ontology as orthogonality, in the literal sense. This is a useful mental model for anyone who thinks about domain models and embeddings together.

**Superposition** (Elhage et al., 2022, *Toy Models of Superposition*): a network with $d$ dimensions can represent $m \gg d$ sparse features. It gives them directions that are almost orthogonal (not exactly orthogonal), and it accepts a small interference. JL guarantees that exponentially many such directions exist. This explains polysemantic neurons. It also predicts that sparse dictionary learning can recover the "true" features.

**Sparse autoencoders** come from three papers: Bricken et al., 2023, Templeton et al., 2024 (*Scaling Monosemanticity*), and Gao et al., 2024. They decompose residual-stream activations into tens of thousands of interpretable features. The result is an over-complete, sparse *re-embedding of the dense embedding*. Beyond interpretability, researchers used SAE features for retrieval and for controllable embeddings. SAE features also let you audit what an embedding model keys on: topic, style, format, or the thing that is important to you.

**Universal geometry.** The **Platonic Representation Hypothesis** (Huh et al., 2024): as models scale, the representations of different models become more and more similar, even across modalities. Mutual kNN alignment measures this similarity. The representations converge toward a shared statistical model of the world.

The **vec2vec** method (Jha et al., 2025) goes to the strong form, with an unsupervised translator (adversarial + cycle-consistency, no paired data). The translator maps embeddings from the space of model A into the space of model B, with cosine up to ~0.9. This is sufficiently accurate to run attribute inference and inversion on the translated vectors.

There are two implications. First, it is possible that a migration between embedding models without a full re-embed becomes feasible, but today it is not reliable in production. Second, the obscurity of your embedding model is not a security control.

**Embedding inversion.** Vec2Text (Morris et al., 2023) recovers input text from embeddings with iterative correction. It reconstructs ~92% of 32-token inputs exactly from OpenAI ada-002 vectors, and most of the content of longer inputs. Later work covers multilingual and other models.

**Treat vectors as the data.** Apply the same access control, encryption at rest, residency, retention and deletion obligations as for the source text. Added noise makes inversion worse, but it also makes retrieval worse. The practical control is access control on the store, not obfuscation of the vectors.

## 10. Non-Euclidean and structured embedding spaces

**Hyperbolic embeddings.** Trees have exponentially many nodes at depth $r$. Euclidean balls grow polynomially in radius, but hyperbolic balls grow exponentially. Thus hyperbolic space embeds hierarchies with low distortion in few dimensions.

**Poincaré embeddings** (Nickel and Kiela, 2017) put the noun hierarchy of WordNet in 5–10 dimensions, with lower distortion than Euclidean embeddings in 200. The Lorentz model (2018) trains more stably. Hyperbolic GNNs and hyperbolic vision-language models (MERU, 2023) followed. Use hyperbolic embeddings when the data *is* a taxonomy, ontology or org chart, and you need is-a geometry. In that geometry, the norm encodes depth/generality, and the angle encodes the branch.

**Order and box embeddings.** These methods represent concepts as regions, so that containment models entailment and hypernymy. The two forms are order embeddings (Vendrov et al., 2016) and box embeddings (Vilnis et al., 2018). Probabilistic box lattices give a calibrated $P(A \mid B)$ from the volume overlap. This is useful where "dog ⊂ mammal" must be transitive, and a symmetric cosine cannot express that. **Gaussian embeddings** (Vilnis and McCallum, 2015, and probabilistic CLIP variants) represent an object as a distribution, and the variance models ambiguity.

**Knowledge-graph embeddings.** The data is triples ${(h, r, t)}$. **TransE** uses $h + r \approx t$. It is elegant, but it fails on one-to-many and symmetric relations. **DistMult / ComplEx** use a bilinear score, and ComplEx handles asymmetry through complex conjugation. In **RotatE**, $r$ is a rotation in complex space, which models symmetry, antisymmetry, inversion and composition.

People use these embeddings for link prediction and as a similarity signal in entity resolution. Their limits: they are transductive (new entities need new training), and they are blind to textual attributes. Modern practice combines text embeddings of entity descriptions with relational GNNs (R-GCN, CompGCN) or with LLM-based completion. In an enterprise ontology, KGEs are a *signal* for completion and matching, never the source of truth.

**Graph node embeddings.** **DeepWalk / node2vec** (2014/2016): random walks make "sentences" of nodes, and then skip-gram trains on them. The return/in-out parameters ${p, q}$ of node2vec interpolate between BFS-like neighborhoods (structural role) and DFS-like neighborhoods (community). These methods are transductive. **GNNs** (GCN, GraphSAGE, GAT, GIN) calculate *inductive* embeddings with message passing over node features. The node embedding of a $k$-layer GNN summarizes a $k$-hop neighborhood.

**Over-smoothing**: with depth, node embeddings converge and become indistinguishable. This is the graph analogue of anisotropy. Residual connections, normalization, or the separation of propagation from transformation decrease it. Graph transformers need positional/structural encodings (Laplacian eigenvectors, random-walk encodings). These are literal positional embeddings for graphs.

In learned physics simulators (MeshGraphNets and successors), the latent of each mesh node is an embedding of the local physical state and geometry. The same limits apply: over-smoothing and Weisfeiler–Lehman expressivity. This is why these simulators use multi-scale and hierarchical message passing to represent long-range interactions.

---

# Part IV — Beyond text

## 11. Vision and multimodal

**CLIP** (Radford et al., 2021) has image and text encoders, trained with a symmetric InfoNCE on 400M (image, caption) pairs. It does zero-shot classification: it embeds prompts such as "a photo of a {class}". ALIGN scaled to noisier data. OpenCLIP/LAION reproduced CLIP openly.

**SigLIP** (Zhai et al., 2023) replaced the batch softmax with a pairwise sigmoid loss. Thus there is no global normalization across the batch, and the behavior is better at both small and very large batch sizes. **SigLIP 2** (2025) added captioning-based pretraining, self-distillation and multilingual data.

**The modality gap** (Liang et al., 2022): image and text embeddings from CLIP-style models occupy two separate cones. An almost constant vector offsets the two cones from each other. Initialization causes the gap, because the two encoders start in different regions. The contrastive loss at low temperature then *keeps* the gap.

The consequences: image–image and text–text similarities are on different scales from cross-modal similarities. Thresholds do not transfer between modes. You can move one modality toward the other to trade off tasks. Natively multimodal models that train on interleaved inputs make the gap smaller, but they do not remove it. Examine cross-modal thresholds separately.

**Self-supervised vision** (DINOv2 from 2023, DINOv3 from 2025, MAE): these features train without text, with self-distillation or masked reconstruction. They are stronger than CLIP for dense and geometric tasks (segmentation, depth, visual-similarity retrieval). The reason is that CLIP has a bias toward whatever the captions describe.

**Document images.** ColPali (2024, section 5) uses per-patch vectors from a vision-language model plus late interaction, directly on page images. It beats a pipeline that does OCR, then chunks the text, then embeds the chunks, on tables, figures, slides and scans. The cost is ~1000 vectors per page. Use a multi-vector index or pooled/compressed variants.

**Natively multimodal, single-space embedders (2025–2026).** These embedders are:

- Cohere embed-v4 (text + images, 128k context, int8/binary output),
- Voyage multimodal-3,
- jina-embeddings-v4 (text + images, single- and multi-vector),
- **Gemini Embedding 2** (Google, March 2026). One model maps text, images, video, audio and PDFs into one 3072-d Matryoshka space (truncatable to 1536/768). This includes *interleaved* combinations. It has an 8k-token text window and task instructions. It understands audio natively, and does not transcribe it first.

The single space of Gemini Embedding 2 collapses the old CLIP-for-images + BERT-for-text + ASR-for-audio pipelines into a single index. The fine print is still important. It includes per-request limits on video length and PDF pages, and the per-call cost. It also includes the incompatibility with any text-only index that you already have.

**Audio and speech.** The options are wav2vec 2.0 and HuBERT (self-supervised speech representations), CLAP (audio–text contrastive) and Whisper encoder states as general audio features. Speaker embeddings (x-vectors, ECAPA-TDNN) are for verification. **Video:** the options are CLIP per frame plus temporal pooling, VideoCLIP, InternVideo, and now unified models.

## 12. Code, science, and structured data

**Code.** Code embedders train on (docstring, function), (query, code) and (code, equivalent code) pairs. The lineage is CodeBERT and UniXcoder, then CodeT5+, then voyage-code-3, Qwen3-Embedding and Nomic Embed Code. Chunk by function/class, which keeps the structure, not by token count. For agents that write code, pure embedding search over a repository is worse than a good symbol index plus grep for navigation. But it wins on intent queries ("find where retries are implemented"), and the practical answer is hybrid.

**Science.** Protein language models (ESM-2, ESM-3) give residue- and sequence-level embeddings that predict structure and function. Molecular embeddings from SMILES transformers (ChemBERTa) or from GNNs over molecular graphs compete with classical fingerprints. ECFP stays a strong baseline. Time-series foundation models (Chronos, MOMENT) make window embeddings for similarity search and anomaly detection. Geospatial embeddings (AlphaEarth Foundations, 2025) embed every 10-m pixel of the surface of the planet.

**Tabular and categorical.** *Entity embeddings* (Guo and Berkhahn, 2016) replace one-hot categoricals with learned vectors inside a neural net. The space frequently shows structure: for example, postal codes cluster geographically. The general rules for the size are $d \approx \min(600, \operatorname{round}(1.6 \cdot n^{0.56}))$ (fastai) or $\propto n^{0.25}$. For gradient-boosted trees, target encoding usually wins. Embeddings are important when categoricals interact and the cardinality is large.

**High-cardinality ids and the hashing trick.** With 10⁸–10⁹ ids (users, ads, URLs), the embedding tables become the largest part of the model. In industrial recommenders, they are terabytes. The hashing trick (Weinberger et al., 2009): hash the id into a smaller table, and accept the collision noise. **Quotient–remainder / compositional embeddings** (Shi et al., 2020): represent an id as a combination of rows from two small tables ($\text{id} \bmod m$, $\text{id} \operatorname{div} m$).

Mixed-dimension embeddings give popular ids more capacity. Tensor-train compression (TT-Rec) makes the tables smaller. The infrastructure (DLRM, TorchRec) shards tables across GPUs, row-wise or table-wise. Then all-to-all lookups are the dominant communication cost.

**Recommender systems are embedding systems.** Matrix factorization (Koren et al., 2009) *is* user and item embeddings with a dot-product score. **Two-tower retrieval** (Covington et al., 2016, and Yi et al., 2019) trains user and item towers with sampled softmax. It adds a log-Q correction for the sampling bias of popular items. The item2vec/prod2vec methods apply word2vec to sessions and baskets. Sequential models (GRU4Rec, SASRec, BERT4Rec) make a user embedding from a behavior sequence.

To serve recommendations, the system does an ANN search over item embeddings, then uses a heavier ranker. The problems that occur again and again are popularity bias, cold start and drift of the embedding tables. For cold start, use content embeddings of the item text/images instead. The two-tower model is also the industrial ancestor of bi-encoder text retrieval.

---
# Part V — Systems

## 13. Approximate nearest neighbor (ANN) search

**Exact search first.** Brute force is $O(N \cdot d)$ per query. With SIMD or a GPU, it is fine up to approximately 10⁵–10⁶ vectors. FAISS flat on a GPU handles millions of vectors at low-millisecond latency. Always run a benchmark of exact search. It is the recall ceiling, and frequently it is sufficiently good.

**Graph-based: HNSW** (Malkov and Yashunin, 2016/2018). HNSW is a multi-layer proximity graph. The sparse upper layers act as express lanes, and the dense bottom layer holds every point. The search is greedy best-first, and it goes down the layers. The parameters are `M` (edges per node, memory $\propto N \cdot M$), `efConstruction` (build quality) and `efSearch` (the query-time beam width, *the* recall/latency knob).

HNSW gives a Recall@10 of 0.95–0.99 at ~1 ms over millions of vectors in RAM. The disadvantages: HNSW is RAM-resident, with ≈ $N \times (4d + 8M)$ bytes plus overhead. For example, 10M × 1024-d is ~40 GB of raw vectors alone. Builds are slow. Deletes use tombstones, and the tombstones make the graph worse until a rebuild.

**DiskANN / Vamana** (Subramanya et al., 2019) is a single-layer graph with a compressed in-memory copy and full vectors on SSD. It reaches billion-scale on one machine. FreshDiskANN handles updates.

**Clustering and quantization: IVF and PQ.** *Inverted file* (IVF): k-means puts the vectors into `n_list` centroids. At query time, scan only the `n_probe` nearest lists. *Product quantization* (Jégou et al., 2011): divide $d$ into $m$ sub-vectors, and run k-means on each into 256 centroids. The result is one byte per sub-vector, so a 4 KB fp32 vector becomes 64–128 bytes. The distances come from lookup tables (asymmetric distance computation).

IVF-PQ with reranking by full vectors is the classical FAISS configuration at billion scale. OPQ rotates the space first, for lower distortion.

**ScaNN** (Guo et al., 2020) uses *anisotropic* quantization. It penalizes the error in the direction parallel to the vector, because that component of the error changes the inner products. ScaNN is the reference design for MIPS at scale. **Scalar quantization** (int8, binary) is the simpler cousin. It is now standard in most vector stores, usually with a rescore of the top-k′ by full-precision vectors.

**LSH.** Random hyperplanes make binary codes. LSH has theoretical guarantees, and it is easy to shard and to stream. But on static corpora, its recall/efficiency is worse than graphs and IVF-PQ. People still use it for near-duplicate detection (SimHash) and for streaming.

**Trade-off summary.**

| Method | Recall/latency | Memory | Updates | Notes |
|---|---|---|---|---|
| Exact (flat) | ceiling | $N \cdot d$ floats | simple | fine to ~10⁶, on a GPU to ~10⁷ |
| HNSW | best | highest | inserts OK, deletes degrade it | rebuild it periodically |
| IVF-PQ | adjustable | lowest | train again on drift | needs a representative training sample |
| DiskANN | very good | SSD-resident | FreshDiskANN | billion scale on a single node |
| LSH | weakest | low | streaming | dedup, sketches |

Compare the methods on recall-vs-QPS curves (in the style of ann-benchmarks.com). A 2–5% recall loss at 10× throughput is typical, and it is usually acceptable. But measure the downstream effect, not only the recall.

**Filtered search: the hard part in practice.** Real queries carry predicates: tenant, date range, document type and access-control lists. With *post-filtering*, you search and then drop the results that do not match. This makes the recall collapse when the filter is selective, and the top-100 can contain zero matches. With *pre-filtering*, you materialize the set that matches and then do an exact search on it. This is correct, but it is high-cost unless the set is small.

*Filter-aware traversal* (Qdrant, Weaviate, Vespa, and ACORN, Patel et al., 2024) continues the search through nodes that do not match, to keep the graph connected. *Partitioning* (one index per tenant) is the pragmatic answer for multi-tenant products. Do tests with your most selective filters. Systems fail silently with these filters.

**Do the storage arithmetic before you select anything.**

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

The cost of a re-embed is the main reason why model upgrades are rare and painful. Plan for it from day one.

## 14. Where the vectors live — and the embedding model as schema

**Options.** The options are in three groups:

- Libraries for in-process search: FAISS, hnswlib, ScaNN, DiskANN, usearch.
- General databases with vector indexes: pgvector/pgvectorscale, Elasticsearch/OpenSearch, Vespa, MongoDB Atlas, Redis, DuckDB/LanceDB. Elasticsearch/OpenSearch is natural for hybrid lexical + dense search. Vespa is the strongest for multi-vector and ranking expressions. DuckDB/LanceDB is for local and analytical work.
- Dedicated vector databases: Milvus, Qdrant, Weaviate, Pinecone, Turbopuffer, Chroma.

**Guidance.** Below ~10–50M vectors at moderate QPS, the vector index inside the database that you already run (Postgres, Elastic) minimizes the operational surface. It also keeps the vectors transactionally consistent with the rows that they describe. Dedicated systems are worth their cost at scale, with heavy filters, for multi-tenancy, or for multi-vector. For quality, the embedding model, the chunker and the reranker are much more important than the choice of store.

**The embedding model is a schema, not an implementation detail.** You cannot compare vectors from model A with vectors from model B. You also cannot compare them with vectors from A at a different version, dimension, normalization or prefix template. The consequences:

1. Store the model id and version, dimensions, normalization, prefix/instruction template and chunker version. Store them per collection, and ideally per vector.
2. Any model change is a full re-embed plus a **blue/green index swap**. Dual-write the new vectors. Validate them on your eval set. Then switch to the new index, and keep the old index through the rollback window.
3. Hosted providers deprecate models. Design for this: an abstraction layer, an eval set and a budget line for re-embeds.
4. *Backward-compatible training* (Shen et al., 2020) is standard in face recognition. It trains a new model whose embeddings are compatible with the old gallery. This prevents a re-index. It is an option for in-house models, and vendors rarely offer it. The vec2vec-style translation (section 9) is a research path, not yet a production path.

**Operations.** Padding dominates the throughput of batch embedding, so sort the inputs by length. Cache embeddings with the key `hash(model_version + template + text)`. Remove duplicates before embedding. Use ONNX/TensorRT and int8 weights for CPU serving of small models.

Monitor the query-score distribution and Recall@k on a canary set that does not change. This finds drift: a model change, a chunker change, or index degradation after deletes. Track the latency of filtered queries separately from the latency of unfiltered queries.

## 15. Retrieval pipelines for RAG and agents

### Chunking — the most underrated variable

- **Fixed-size with overlap.** The chunks are 256–512 tokens, with 10–20% overlap. This is a simple, robust baseline.
- **Recursive / structure-aware.** Split on headings first, then on paragraphs, then on sentences. Keep tables and code blocks intact. Copy the heading path into the chunk text.
- **Semantic chunking.** Split where the embedding similarity of consecutive sentences decreases. The gains are modest, and the cost is high.
- **Proposition-level.** An LLM rewrites the text into atomic facts. This gives the best precision, at the highest cost.
- **Parent–child (small-to-big).** Embed small chunks for precision. Return the section that contains the chunk to the LLM, for context.
- **Late chunking and contextual retrieval** (section 5) put the document context back into the chunks.

Chunk size interacts with the embedder, the context budget of the LLM, and the question type (factoid or synthesis). The quality of the embedder decreases long before the advertised max length. Run your eval set on two or three chunkers. The chunker frequently changes recall more than the choice of embedding model does.

### Query side

- **Get the instruction/prefix template exactly correct.** If an E5 query does not have its `query: ` prefix, or if the task type is incorrect, you silently lose points.
- **Rewrite and decompose** multi-part questions with an LLM. Use **multi-query**: several paraphrases and the union of their results. Use **step-back** prompts for over-specific questions.
- **HyDE** (Gao et al., 2022): generate a hypothetical answer and embed *that*. This closes the query–document asymmetry gap. It is useful zero-shot, but it adds latency, and it can hallucinate the incorrect topic.
- **Metadata extraction into filters**: parse dates, product names and document types from the query. This is the most effective solution of all for "the answer was in a 2025 document but retrieval returned the 2019 version."

### Fusion and reranking

The pipeline has these steps:

1. Retrieve the top-50–200 from dense + BM25 (+ learned sparse).
2. Combine the results with RRF.
3. Rerank with a cross-encoder reranker (bge-reranker-v2, Qwen3-Reranker, Cohere Rerank, jina-reranker), or with an LLM listwise reranker for the top-20.
4. Give the top 5–10 to the LLM.
5. Optionally, apply MMR for diversity.

In most pipelines, rerankers give a large gain in quality at the lowest cost. Late interaction is the middle ground where the rerank latency is too high.

### Known failure modes of dense retrieval — and what to do

| Failure | Why | Mitigation |
|---|---|---|
| Negation, antonyms ("laptops *without* touchscreen") | Embeddings mostly ignore "not" (NevIR, 2023). | A reranker or an LLM filter. |
| Exact identifiers, part numbers, error strings, names | The tokenizer splits them into fragments. Semantics ≠ identity. | BM25 / learned sparse + metadata filters. Use hybrid by default. |
| Numbers, dates, ranges, comparisons | The geometry does not encode them. | Extract them to structured filters. Route the query to SQL. |
| Very long documents | Dilution. | Chunk the document. Use late chunking or contextual retrieval. |
| Multi-hop, reasoning-intensive questions | No single chunk resembles the question. | Iterative/agentic retrieval, HyDE, reasoning rerankers. |
| Combinatorial / instruction-following retrieval | The capacity bound of a single vector (section 5). | Multi-vector, sparse, agentic. |
| Cross-lingual queries | Weak alignment, hubness. | Multilingual models trained on bitext. Examine hubness per language. |
| Jargon, distribution shift | A mismatch in the training data. | Fine-tune (section 4), or add synthetic domain pairs. |
| "Everything scores 0.7" | An anisotropic model or incorrect pooling. | Examine alignment/uniformity. Use a contrastive-trained model. |

### Embeddings elsewhere in agent systems

- **Tool and skill retrieval.** With hundreds of tools, embed the descriptions plus example invocations. Then retrieve the top-k into the prompt. Measure the accuracy of tool selection, not cosine.
- **Memory.** Episodic memory stores embedded (frequently LLM-summarized) events. Retrieval combines similarity with recency and importance (Park et al., 2023, *Generative Agents*). Periodic reflection/consolidation decreases the hubness of memories that retrieval returns frequently. Structured facts belong in a database, not in a vector store.
- **Semantic caching.** Match new queries by similarity to queries that you answered before. False positives are dangerous ("Q3 2024 revenue" against "Q3 2025 revenue"). Use high thresholds, an exact match on extracted entities and dates, or an LLM verifier. Measure the cache precision explicitly.
- **Routing and intent classification.** kNN over embeddings of labeled examples is a strong, low-cost classifier that you can update immediately. Add an example, and the behavior changes, with no new training. Use the distance to the nearest labeled example to decide when to abstain.
- **Few-shot example selection.** Retrieve the most similar labeled examples into the prompt (kNN in-context learning). This is reliably better than random examples.
- **Deduplication and clustering.** Remove near-duplicates before indexing: cosine above ~0.95 on the same model, or MinHash for lexical duplicates. Use HDBSCAN/k-means for topic discovery. Use UMAP for *visualization only*, because it distorts distances.
- **Entity resolution and schema matching.** Embeddings of records or columns serve as a blocking key, and as one similarity feature among several (string similarity, rules). They give good recall for "same entity, different spelling". They are never the only decision signal in a governed identity model.
- **Anomaly detection.** Use the distance to the nearest cluster centroid, or the kNN density, over embeddings of logs, tickets or transactions.
- **Evaluation and analysis.** Cluster model outputs to find failure modes. Embedding-based similarity metrics (BERTScore family) are for generation evaluation. Remember that they reward topical agreement, not factual agreement.

### Security and governance

- **Inversion** (section 9): vectors are the data. Apply the same controls as for the source text.
- **Access control at retrieval time.** Filter by ACL before or during the search (pre-filter or filter-aware index). Never trust the LLM to hold back a retrieved chunk. Use per-tenant indexes for hard isolation.
- **Poisoning** (PoisonedRAG, Zou et al., 2024): an attacker who can write to the corpus crafts passages optimized for retrieval on target queries. The corpus can be the public web, shared drives or ticket systems. These passages steer the answer. The defenses are provenance-weighted ranking, outlier and duplicate detection on new content, and isolation of untrusted sources. Another defense is to treat retrieved text as untrusted input (prompt-injection hardening).
- **Extraction via similarity APIs.** Repeated queries permit membership inference and corpus extraction. Rate-limit and log the queries.

---

# Part VI — Frontier and practice

## 16. Where the research is moving (2025–2026)

1. **LLM-backbone embedders and the blur between generation and representation.** The examples are GritLM-style unified models and embeddings from reasoning-tuned models. Another example is *test-time compute for retrieval*. It includes retrievers trained on synthetic reasoning-intensive queries (ReasonIR) and listwise LLM rerankers that reason. Test-time compute for retrieval makes the BRIGHT gap smaller.
2. **Capacity limits and the multi-vector renaissance.** The Weller et al. bound changes "which representation?" into a capacity question. Expect more of these in production: late interaction (ColBERT/ColPali), sparse + dense hybrids and fixed-dimensional encodings (MUVERA). Also expect more benchmarks built to expose combinatorial failure (LIMIT, FollowIR, BRIGHT).
3. **Native multimodality in one space.** Examples are Gemini Embedding 2 (text/image/video/audio/PDF), Cohere embed-v4, jina-v4 and Voyage multimodal. The open questions are the modality gap under interleaved training and cross-modal calibration. Another open question is if unified spaces give up within-modality fidelity.
4. **Universal geometry, translation, and privacy.** The trends are Platonic convergence, vec2vec translation, steadily better inversion, and regulators who treat embeddings as personal data. Watch for tools for migration between embedding spaces. Also watch for retrieval that protects privacy (encrypted/secure ANN, differential privacy on embeddings).
5. **Context- and corpus-aware embeddings.** Late chunking, contextual retrieval and Contextual Document Embeddings are representations that depend on what else is in the index. Another topic is document-level versus passage-level representations.
6. **Scaling laws for retrieval.** Fang et al. (2024) find that contrastive entropy obeys power laws in model size and annotation volume. Annotation *quality* trades off against model size. The practical conclusion: money for hard-negative quality and synthetic data frequently gives more than a larger encoder.
7. **Efficiency.** MRL + binary + int8 give 64–100× smaller indexes. Static-embedding distillation makes the encoder 100–500× faster. Distilled models of 100–600M are within a few points of 7B+ teachers. Retrieval on the device is another trend.
8. **Interpretable and controllable embeddings.** The directions are SAEs over embedding models and feature-level retrieval and steering ("match on this feature, ignore style"). Another direction is concept-geometry results (Park et al.), as a bridge between embedding spaces and explicit ontologies.
9. **Long-context LLMs versus retrieval.** Million-token contexts decrease the need for fine chunking, but not the need for retrieval. Cost, latency, freshness, permissions and the lost-in-the-middle effect keep retrieval central. The role moves toward coarse selection of larger units plus **agentic search**. In agentic search, the model does multiple searches, reads, and refines. There, embeddings are one tool among lexical search, structured queries and graph traversal.
10. **Dynamic and continual settings.** The topics are temporal embeddings, drift detection, and backward-compatible training so that indexes survive model updates. Another topic is streaming ANN with low-cost deletes.

## 17. Model landscape snapshot (mid-2026) — verify before choosing

- **Hosted:** OpenAI text-embedding-3 small/large supports MRL, and it is low-cost and text only. Google Gemini Embedding 2 is multimodal, with 3072-d MRL, an 8k-token window and task instructions. The text-only gemini-embedding-001 is still available. Cohere embed-v4 has text + image, 128k context and int8/binary output. Voyage has voyage-3.5 / 3-large with 32k context, voyage-code-3 and voyage-multimodal-3. Jina has v4 (multimodal and multi-vector) and v5-text-small under Apache-2.0.
- **Open weights:** Qwen3-Embedding 0.6B/4B/8B is strong on multilingual and code, under Apache-2.0. Tencent KaLM-Embedding-Gemma3-12B led the MMTEB multilingual board as of July 2026. Microsoft Harrier-OSS-v1 (27B, MIT) has the top MTEB v2 English scores. BGE-M3 (dense + sparse + multi-vector, 8k, 100+ languages) is still the usual multilingual choice. The bge-large-en-v1.5, E5 and GTE families are small, fast and well understood. Other models are Nomic Embed v2-MoE, Snowflake Arctic-Embed 2.0 and Stella/Jasper. ColBERTv2 and ColPali are for late interaction, and SPLADE-v3 is for learned sparse.
- **Rerankers:** the options are bge-reranker-v2 (m3, gemma), Qwen3-Reranker, jina-reranker, Cohere Rerank 3.5 and mxbai-rerank. LLM listwise rerankers are for the final top-20.
- **General rules.** For English enterprise RAG, a 100–600M open model + hybrid retrieval + a reranker usually gets within a few points of the best 8–27B model. This holds *on your data*, at a fraction of the cost. Select multilingual models by the languages that you serve in practice. Select a multimodal model only when you need cross-modal search. The price is higher, and the space is incompatible with your text index. Open versus hosted is now mostly a decision about data residency, deprecation risk and operations, not about quality. Always run the top three candidates on your own eval set.

## 18. Decision guide and pitfalls

**Criteria to select a model.**

- Task type (asymmetric retrieval, symmetric similarity or clustering).
- Languages.
- Domain.
- The input length that you need, and the quality *at that length*.
- Latency and throughput budget.
- Hosted or self-hosted (residency, deprecation).
- License.
- Dimensions and MRL/quantization support (do the storage math).
- Instruction/prefix format.
- Multimodal needs.
- The reranker to pair with the model.
- Fine-tuning feasibility.

**Steps to build the pipeline.**

1. Eval set.
2. Chunker.
3. Hybrid retrieval (dense + BM25/sparse).
4. Metadata filters.
5. Reranker.
6. $k$ adjusted to the LLM.
7. Monitoring canaries.
8. Versioned index with blue/green swap.

**Pitfalls (each of these cost someone a quarter).**

- An incorrect or absent prefix/instruction template.
- A mix of vectors from two models or two versions in one index.
- Cosine values compared, or thresholds used again, across models.
- Post-filtering with selective filters.
- Single-vector retrieval for combinatorial or instruction-following queries.
- Embeddings of whole documents, not of chunks.
- Trust in the MTEB rank order over your own eval.
- Semantic-cache false positives on entities and dates.
- No budget or plan for a re-embed when the model changes.
- Vectors stored with weaker access controls than the source text.
- UMAP distances treated as real distances.
- Mean pooling of a causal (decoder) model.
- Cosine on embeddings from a model that did not train with cosine.
- IVF/PQ codebooks trained on a non-representative sample, and never trained again after drift.
- Deletes that silently decrease HNSW recall (no rebuild schedule).

---

# Appendices

## A. Key formulas

$$
\begin{aligned}
&\text{Cosine} && \cos(a,b) = a \cdot b \,/\, (\lVert a \rVert \, \lVert b \rVert) \\
&\text{Distance identity} && \lVert a - b \rVert^2 = \lVert a \rVert^2 + \lVert b \rVert^2 - 2\,a \cdot b \quad (= 2 - 2\cos(a,b) \text{ for unit vectors}) \\
&\text{PMI} && \operatorname{PMI}(w,c) = \log \bigl[ P(w,c) \,/\, (P(w)\,P(c)) \bigr] \\
&\text{SGNS optimum} && w \cdot c = \operatorname{PMI}(w,c) - \log k \quad \text{(Levy }\&\text{ Goldberg, 2014)} \\
&\text{GloVe} && \min \sum_{ij} f(X_{ij})\,(w_i \cdot c_j + b_i + b_j - \log X_{ij})^2 \\
&\text{InfoNCE} && L = -\log \frac{e^{s(q,d^{+})/\tau}}{\sum_{d \in \lbrace d^{+} \rbrace \cup \text{negatives}} e^{s(q,d)/\tau}} \\
&\text{Alignment} && \mathbb{E}_{(x,y) \sim \text{pos}} \lVert f(x) - f(y) \rVert^2 \\
&\text{Uniformity} && \log \mathbb{E}_{x,y \sim \text{data}}\, e^{-2\lVert f(x) - f(y) \rVert^2} \\
&\text{MaxSim (ColBERT)} && \text{score} = \sum_{i \in q} \max_{j \in d} q_i \cdot d_j \\
&\text{RRF} && \operatorname{score}(d) = \sum_{\text{rankers } i} \frac{1}{k + \operatorname{rank}_i(d)}, \quad k \approx 60 \\
&\text{JL lemma} && k = O(\log n / \varepsilon^2) \text{ dims preserve all pairwise distances within } (1 \pm \varepsilon) \\
&\text{MIPS} \to \text{NN} && x' = \bigl[x,\ \sqrt{M^2 - \lVert x \rVert^2}\,\bigr], \quad q' = [q,\ 0] \quad (M \ge \max \lVert x \rVert)
\end{aligned}
$$

## B. Working defaults

| Knob | Typical range | Notes |
|---|---|---|
| Contrastive temperature $\tau$ (cosine) | 0.02–0.05 | Learnable in CLIP-style training. |
| Fine-tuning LR | 1e-5 to 2e-5 | 1–3 epochs, warm-up 5–10%. |
| Hard negatives per query | 1–7 | Cross-encoder-filtered, margin ≈ 0.05–0.1. |
| Chunk size | 256–512 tokens | 10–20% overlap. Keep the structure intact. |
| First-stage $k$ | 50–200 | Before RRF and reranking. |
| Final $k$ to LLM | 5–10 | Adjust it on answer quality. |
| HNSW `M` / `efConstruction` / `efSearch` | 16–32 / 200 / 64–256 | `efSearch` is the runtime recall knob. |
| IVF `n_list` / `n_probe` | $4\sqrt{N}$ to $16\sqrt{N}$ / 1–5% of lists | Train the codebooks again on drift. |
| MRL truncation | 1024 to 256–512 | ~1–3 nDCG points. Rescore with full dims. |
| Dedup threshold | cosine ≳ 0.95 | Model-specific. Validate it. |

## C. Reading list by topic

**Foundations.**

- Deerwester et al. 1990 (LSA)
- Mikolov et al. 2013 (word2vec)
- Pennington et al. 2014 (GloVe)
- Levy and Goldberg 2014 (SGNS as implicit matrix factorization)
- Levy, Goldberg and Dagan 2015
- Bojanowski et al. 2017 (fastText)
- Arora et al. 2016 (RAND-WALK)
- Linzen 2016 (analogy evaluation)

**Contextual representations.**

- Peters et al. 2018 (ELMo)
- Devlin et al. 2018 (BERT)
- Tenney et al. 2019
- Ethayarajh 2019 (anisotropy)
- Gao et al. 2019 (representation degeneration)
- Mu and Viswanath 2018 (All-but-the-Top)
- Timkey and van Schijndel 2021 (rogue dimensions)
- Press and Wolf 2017 (tied embeddings)
- nostalgebraist 2020 (logit lens)
- Rumbelow and Watkins 2023 (glitch tokens)
- Su et al. 2021 (RoPE)
- Press et al. 2021 (ALiBi)

**Embedding models and training.**

- Reimers and Gurevych 2019 (SBERT)
- Karpukhin et al. 2020 (DPR)
- Xiong et al. 2020 (ANCE)
- Qu et al. 2020 (RocketQA)
- Hofstätter et al. 2020/2021 (Margin-MSE, TAS-B)
- Gao et al. 2021 (GradCache, SimCSE)
- Izacard et al. 2021 (Contriever)
- Wang and Isola 2020 (alignment/uniformity)
- Wang et al. 2022 (E5)
- Su et al. 2022 (Instructor)
- Xiao et al. 2023 (BGE / C-Pack)
- Li et al. 2023 (GTE)
- Wang et al. 2023 (E5-Mistral, synthetic data)
- Lee et al. 2024 (Gecko, NV-Embed)
- BehnamGhader et al. 2024 (LLM2Vec)
- Muennighoff et al. 2024 (GritLM)
- Chen et al. 2024 (BGE-M3)
- Zhang et al. 2025 (Qwen3-Embedding)
- Lee et al. 2025 (Gemini Embedding)
- Google DeepMind 2026 (Gemini Embedding 2)

**Representation formats.**

- Formal et al. 2021 (SPLADE)
- Khattab and Zaharia 2020, Santhanam et al. 2021/2022 (ColBERT, v2, PLAID)
- Faysse et al. 2024 (ColPali)
- Dhulipala et al. 2024 (MUVERA)
- Kusupati et al. 2022 (Matryoshka)
- Günther et al. 2024 (late chunking)
- Anthropic 2024 (contextual retrieval)
- Morris and Rush 2024 (Contextual Document Embeddings)
- Weller et al. 2025 (theoretical limitations, LIMIT)
- Thakur et al. 2021 (BEIR)
- Cormack et al. 2009 (RRF)

**Evaluation.**

- Muennighoff et al. 2022 (MTEB)
- Enevoldsen et al. 2025 (MMTEB)
- Su et al. 2024 (BRIGHT)
- Weller et al. 2023/2024 (NevIR, FollowIR)
- Tang and Yang 2024 (FinMTEB)

**Geometry and interpretability.**

- Beyer et al. 1999
- Radovanović et al. 2010 (hubness)
- Conneau et al. 2018 (CSLS)
- Steck et al. 2024 (cosine caveat)
- Park, Choe and Veitch 2023 (linear representation hypothesis)
- Park et al. 2024 (categorical/hierarchical geometry)
- Elhage et al. 2022 (superposition)
- Bricken et al. 2023, Templeton et al. 2024, Gao et al. 2024 (sparse autoencoders)
- Huh et al. 2024 (Platonic Representation Hypothesis)
- Jha et al. 2025 (vec2vec)
- Morris et al. 2023 (Vec2Text inversion)
- Nickel and Kiela 2017/2018 (Poincaré, Lorentz)
- Vilnis et al. 2018 (box embeddings)
- Bordes et al. 2013, Trouillon et al. 2016, Sun et al. 2019 (TransE, ComplEx, RotatE)
- Grover and Leskovec 2016 (node2vec)
- Hamilton et al. 2017 (GraphSAGE)

**Multimodal and other domains.**

- Radford et al. 2021 (CLIP)
- Zhai et al. 2023 (SigLIP)
- Tschannen et al. 2025 (SigLIP 2)
- Liang et al. 2022 (modality gap)
- Oquab et al. 2023 (DINOv2)
- Guo and Berkhahn 2016 (entity embeddings)
- Weinberger et al. 2009 (hashing trick)
- Shi et al. 2020 (compositional embeddings)
- Naumov et al. 2019 (DLRM)
- Covington et al. 2016, Yi et al. 2019 (two-tower)
- Lin et al. 2023 (ESM-2)

**Systems.**

- Malkov and Yashunin 2018 (HNSW)
- Jégou et al. 2011 (PQ)
- Guo et al. 2020 (ScaNN)
- Subramanya et al. 2019 (DiskANN)
- Patel et al. 2024 (ACORN)
- Shen et al. 2020 (backward-compatible training)
- Gao et al. 2022 (HyDE)
- Zou et al. 2024 (PoisonedRAG)
- Park et al. 2023 (Generative Agents memory)
- Fang et al. 2024 (scaling laws for dense retrieval)

## D. Glossary

- **Anisotropy**: embeddings concentrated in a narrow cone. Random pairs have a high cosine.
- **ANN**: approximate nearest neighbor search. It trades recall for speed and memory.
- **Bi-encoder / cross-encoder**: a bi-encoder encodes the two inputs independently and uses a dot-product score. A cross-encoder encodes them together, with full attention.
- **Contrastive learning**: pull positives together, and push negatives apart (InfoNCE and its relatives).
- **Hard negative**: a non-relevant candidate that looks relevant. It is the main lever in retrieval training.
- **Hubness**: some points are the nearest neighbors of disproportionately many other points.
- **Hybrid retrieval**: the fusion of lexical (BM25/sparse) and dense results, usually with RRF.
- **Late interaction**: per-token vectors with a MaxSim score at query time (ColBERT).
- **Learned sparse**: vocabulary-sized sparse vectors with learned weights and expansion (SPLADE).
- **Matryoshka (MRL)**: training so that the prefixes of the vector are usable embeddings.
- **Modality gap**: the offset between modality clusters in a joint image–text space.
- **Pooling**: an operation that collapses token vectors into one vector (mean, CLS, last token, latent attention).
- **Product quantization (PQ)**: an operation that compresses vectors into per-sub-vector centroid ids.
- **Reranker**: a second-stage model (usually a cross-encoder) that scores the first-stage candidates again.
- **Sign-rank bound**: the limit on the distinct top-k subsets that a $d$-dimensional dot product can express.
- **Superposition**: more features than dimensions, stored as almost orthogonal directions.
- **Temperature ($\tau$)**: the softmax sharpness in contrastive loss. A smaller $\tau$ puts the focus on the hardest negatives.
- **Uniformity / alignment**: the two quantities that contrastive loss optimizes on the hypersphere.
