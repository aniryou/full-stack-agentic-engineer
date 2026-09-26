# RAG from scratch

A hands-on companion to the RAG primer. You learn retrieval by **building it** —
every core mechanism (cosine search, BM25, RRF, reranking, chunking, recall@k /
MRR, an iterative multi-hop loop) is a short, plain-Python function you write
yourself, then check against a self-grading `assert`.

The heavy lifting nobody learns anything from — the embedding model — is a black
box. Everything that teaches is ~40 lines you can read in one sitting.

**Time and tier:** ~8 h (rough); module 07.4, with the other retrieval labs and primers (~28 h in all) in [`CURRICULUM.md`](../../../CURRICULUM.md). T0 = a laptop or Colab CPU, free: numpy only, with a hashing embedder (table below). T0 + torch or Colab adds the real embedding and reranking models.

## Setup

```bash
pip install -r requirements.txt        # T0: numpy + pytest, no torch (seconds, a few MB)
python -m pytest -q                    # 12 tests: reference primitives + the embedder fallback
pip install -r requirements-full.txt   # optional: the real models (sentence-transformers, so torch: a multi-GB install)
```

| Where you run it | Embedder the notebooks get | What you learn |
|---|---|---|
| **T0** — laptop or Colab CPU, `requirements.txt` only | `hashing embedder (T0 fallback; not semantic)`: bag-of-words feature hashing | every mechanism; all self-checks pass. Dense search is lexical, so the "semantic beats lexical" results do not show |
| **T0 + torch** (`requirements-full.txt`) or **Colab** (its runtime ships sentence-transformers and torch as of 2026-09-26 (verify); if yours does not, `pip install -r requirements-full.txt`) | `all-MiniLM-L6-v2` (~90 MB as of 2026-09-26 (verify), downloaded once, then offline on CPU) and the `ms-marco-MiniLM-L-6-v2` cross-encoder | the same, plus the real dense-vs-lexical and reranking differences |

`ragkit.embed.get_embedder()` and `get_cross_encoder()` pick the real model when
`sentence-transformers` is installed and otherwise print which fallback they
use; `RAGKIT_EMBEDDER=hashing` forces the fallback. **Generation is optional:**
with no API key, a deterministic *extractive* fallback answers straight from the
retrieved text, so every notebook runs end to end offline. To use a real model
instead, set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` — it's picked up
automatically, no code change.

Open the notebooks with Jupyter (`pip install jupyterlab && jupyter lab`) or in
VS Code / Cursor.

## Learning path

Work through `notebooks/` in order. Each has a short concept write-up, worked
code you run, and `# YOUR CODE HERE` exercises with inline checks. Completed
versions are in `solutions/`.

| # | Notebook | What you build |
|---|----------|----------------|
| 00 | `setup_and_corpus` | Orientation: the corpus, the eval set, "embedding = vector" |
| 01 | `minimal_rag` | The whole loop: embed → cosine search → grounded prompt → generate |
| 02 | `chunking` | Fixed vs structure-aware vs small-to-big; why splitting decides retrieval |
| 03 | `hybrid_search` | BM25 from scratch + dense, fused with Reciprocal Rank Fusion |
| 04 | `reranking` | Cheap first-stage recall, then a cross-encoder for precision |
| 05 | `evaluation` | Recall@k and MRR; score dense vs BM25 vs hybrid, by question type |
| 06 | `iterative_rag` | *(advanced)* multi-hop questions and the loop behind agentic RAG |

The self-checks test the **shape** of your implementation (sorted correctly,
right length, correct maths) rather than which document a model happens to rank
first — so they pass with the hashing fallback or the real embedder, and a
green notebook means your code is right. The *observations* (which retriever
wins on which question type in 03 and 05) need the real model.

## What's in the box

```
ragkit/                 tiny support library (NOT the lesson — plumbing only)
  corpus.py             load the docs + eval labels; the shared tokenizer
  embed.py              sentence-transformers wrapper (dense + cross-encoder), hashing fallback at T0
  llm.py                generation: real API if a key is set, else extractive
  reference.py          reference implementations (the answer key; later
                        notebooks import primitives earlier ones built)
  data/corpus/          9 short Markdown docs for a fictional company, "Meridian"
  data/eval/qrels.json  18 labelled questions (lexical / semantic / multi-hop)
notebooks/              the course — practice versions with blanks
solutions/              the same notebooks, filled in
tests/                  correctness checks for the reference code + notebooks
tools_make_data.py      regenerates the corpus + eval set
tools_build_notebooks.py regenerates notebooks/ and solutions/ from one source
```

The corpus is deliberately small and adversarial: it has exact IDs and codes
(`ERR_4290`, `SEC-011`, `DR-07`) that favour lexical search, paraphrasable facts
that favour dense search, and cross-document questions that need more than one
retrieval — so each notebook's lesson actually shows up when you measure it.

## Verify the reference code

```bash
python -m pytest -q                               # the from-scratch primitives + the embedder fallback
PYTHONPATH=. python tests/test_notebooks_run.py   # every solution runs + asserts pass
```

## Where to go after 06

Swap the notebook 06 planner for a real model, let it choose *which* retriever to
call and *when to stop*, fine-tune the retriever/reranker on your own query logs,
and evaluate whole trajectories rather than single answers. Parts III–IV of the
primer map that territory.
