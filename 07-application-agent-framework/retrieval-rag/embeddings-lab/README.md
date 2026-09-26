# Embeddings Lab

Hands-on companion to the embeddings primer (`docs/primer.md`). Every core idea
is implemented from scratch in plain NumPy — no torch, no sklearn — small
enough to read in one sitting, real enough that the phenomena actually show up.

**Time and tier:** ~8 h (rough); module 07.4, with the other retrieval labs and primers (~28 h in all) in [`CURRICULUM.md`](../../../CURRICULUM.md). T0 = a laptop or Colab CPU, free: numpy and matplotlib, no torch, no GPU, no key.

## Setup

```
pip install numpy matplotlib jupyter
jupyter lab notebooks/
```

Notebook 01 writes `artifacts/word_vectors.npz`, used by notebook 03 and
exercises 01/03. A pre-built copy ships in the repo, so everything also works
out of the box.

## How to work

1. Read the worked notebook — outputs are baked in, so it reviews well even
   without running.
2. Do the matching `exercises/exNN.ipynb`: fill each `YOUR CODE HERE` block.
   Every task has a self-check (hand-computable values, invariance properties,
   or finite-difference gradient checks) — run the cell to verify yourself.
3. Compare with `solutions/exNN_solutions.ipynb` (executed).

## Map

| Notebook | Core idea | Primer |
|---|---|---|
| `01_counts_to_vectors` | Embeddings are low-rank factorizations of co-occurrence: PPMI + SVD, SGNS from scratch, the Levy–Goldberg `w·c = PMI − log k` check, analogies | §1–2 |
| `02_contrastive_bi_encoder` | InfoNCE with hand-derived gradients on crop pairs; alignment & uniformity; temperature ablation | §4 |
| `03_geometry` | Anisotropy + all-but-the-top; hubness + CSLS; the JL lemma; SVD as the original Matryoshka | §3, §7–8 |
| `04_vector_search` | IVF and PQ from scratch; recall-vs-work curves; compress-then-rerank; the filtered-search trap | §13 |
| `05_retrieval_pipeline` | BM25 + LSA dense + RRF, evaluated with nDCG on judged queries with planted paraphrase / exact-id / negation failure modes | §5–6, §15 |
| `06_superposition` | Minimal replication of *Toy Models of Superposition*; gradients verified numerically; the sparsity phase change and the pentagon | §9 |

## Data

- `data/tiny_corpus.txt` — synthetic corpus with engineered structure: topic
  clusters, a gender × royalty grid so `king − man + woman = queen` holds by
  construction, and subject-pure paragraphs so contrastive pairs are learnable.
  Regenerate: `python data/make_corpus.py`.
- `data/docs.jsonl` + `data/queries.jsonl` — 36 docs, 10 judged queries with
  planted retrieval phenomena. Regenerate: `python data/make_retrieval_set.py`.

## Rebuilding everything

Notebooks are generated from `src/*.py` (jupytext percent format — the single
source of truth). `python build.py` re-executes all worked and solution
notebooks (verifying every assert) and re-strips the exercise notebooks.
Build-only deps: `pip install jupytext nbclient nbformat nbconvert ipykernel`.

## Deliberately missing

Real transformer embedders and multimodal models. To upgrade notebook 05, swap
`dense_scores` for a sentence-transformers model — the pipeline and eval code
don't change. That's the point.
