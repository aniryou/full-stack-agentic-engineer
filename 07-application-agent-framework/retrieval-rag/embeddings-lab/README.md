# Embeddings Lab

This lab is a hands-on companion to the embeddings primer (`docs/primer.md`). It implements each core idea by hand
in plain NumPy, with no torch and no sklearn. Each implementation is small, so you can read it in one session. But
it is also sufficiently real that the effects actually occur.

**Time and tier:** ~8 h (rough). The lab is part of module 07.4, together with the other retrieval labs and primers (~28 h in all), in [`CURRICULUM.md`](../../../CURRICULUM.md). T0 is a laptop or a Colab CPU, at no cost. It needs numpy and matplotlib, and no torch, no GPU and no key.

## Setup

```
pip install numpy matplotlib jupyter
jupyter lab notebooks/
```

Notebook 01 writes `artifacts/word_vectors.npz`. Notebook 03 and the exercises ex01/ex03 use this file. The repo
contains a pre-built copy, so all of the notebooks also operate immediately, with no more steps.

## How to work

1. Read the worked notebook. It contains its outputs, so you can review it, even if you do not run it.
2. Do the related `notebooks/exNN.ipynb`. Fill in each `YOUR CODE HERE` block.
   Each task has a self-check: values that you can calculate by hand, invariance properties,
   or finite-difference gradient checks. Run the cell to examine your work.
3. Compare your work with `solutions/exNN.ipynb`. It contains the outputs of a run, and it has the same file name as the exercise.

## Map

| Notebook | Core idea | Primer |
|---|---|---|
| `01_counts_to_vectors` | Embeddings are low-rank factorizations of co-occurrence. The notebook has PPMI and SVD, SGNS written by hand, the Levy–Goldberg `w·c = PMI − log k` check, and analogies. | §1–2 |
| `02_contrastive_bi_encoder` | InfoNCE with gradients derived by hand on crop pairs, alignment and uniformity, a temperature ablation | §4 |
| `03_geometry` | Anisotropy and all-but-the-top, hubness and CSLS, the JL lemma, SVD as the original Matryoshka | §3, §7–8 |
| `04_vector_search` | IVF and PQ written by hand, curves of recall against work, compress-then-rerank, the filtered-search trap | §13 |
| `05_retrieval_pipeline` | BM25, LSA dense and RRF. The notebook measures them with nDCG on judged queries that have planted failure modes: paraphrase, exact id and negation. | §5–6, §15 |
| `06_superposition` | A minimal replication of *Toy Models of Superposition*, a numerical check of the gradients, the sparsity phase change and the pentagon | §9 |

## Data

- `data/tiny_corpus.txt`: a synthetic corpus with a structure made by design. It has topic
  clusters and a gender × royalty grid, so `king − man + woman = queen` is true by
  construction. It also has paragraphs about one subject each, so a model can learn the contrastive pairs.
  To make the file again, run `python data/make_corpus.py`.
- `data/docs.jsonl` and `data/queries.jsonl`: 36 docs and 10 judged queries with
  planted retrieval effects. To make the files again, run `python data/make_retrieval_set.py`.

## Rebuilding everything

The notebooks come from `src/*.py`. These files use the jupytext percent format, and they are the single source of
truth. `python build.py` runs the worked notebooks and the solutions again, and makes sure that each assert passes.
It also removes the solutions from the exercise notebooks (`notebooks/exNN.ipynb`) again.
Only the build needs these dependencies: `pip install jupytext nbclient nbformat nbconvert ipykernel`.

## Deliberately missing

The lab does not contain real transformer embedders or multimodal models. To improve notebook 05, replace
`dense_scores` with a sentence-transformers model. The pipeline code and the eval code do not change. That is the
purpose of the design.
