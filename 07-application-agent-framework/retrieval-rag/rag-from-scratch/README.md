# RAG from scratch

This lab is a hands-on companion to the RAG primer. You learn retrieval when you **build it**. Each core
mechanism is a short function in plain Python. You write the function yourself, then an automatic `assert`
examines it. The core mechanisms are cosine search, BM25, RRF, reranking, chunking, recall@k / MRR and an
iterative multi-hop loop.

The embedding model does the heavy work, and nobody learns anything from it. It is a black box. Everything that
teaches is ~40 lines, and you can read it in one session.

**Time and tier:** ~8 h (rough). The lab is part of module 07.4, together with the other retrieval labs and primers (~28 h in all), in [`CURRICULUM.md`](../../../CURRICULUM.md). T0 is a laptop or a Colab CPU, at no cost. It needs only numpy, and it uses a hashing embedder (see the table in Setup). T0 + torch, or Colab, adds the real embedding and reranker models.

## Setup

```bash
pip install -r requirements.txt        # T0: numpy + pytest, no torch (seconds, a few MB)
python -m pytest -q                    # 12 tests: reference primitives + the embedder fallback
pip install -r requirements-full.txt   # optional: the real models (sentence-transformers, so torch: a multi-GB install)
```

| Where you run it | Embedder the notebooks get | What you learn |
|---|---|---|
| **T0**: a laptop or a Colab CPU, `requirements.txt` only | `hashing embedder (T0 fallback; not semantic)`: bag-of-words feature hashing | Every mechanism. All self-checks pass. Dense search is lexical, so the "semantic beats lexical" results do not appear. |
| **T0 + torch** (`requirements-full.txt`) or **Colab**. The Colab runtime contains sentence-transformers and torch as of 2026-09-26 (verify). If your runtime does not contain them, run `pip install -r requirements-full.txt`. | `all-MiniLM-L6-v2` and the `ms-marco-MiniLM-L-6-v2` cross-encoder. The embedder is ~90 MB as of 2026-09-26 (verify). The first use downloads it one time, then it runs offline on the CPU. | The same as T0. You also see the real differences between dense and lexical search, and the real reranking differences. |

If the environment has `sentence-transformers`, `ragkit.embed.get_embedder()` and `get_cross_encoder()` select the real
model. If not, they print the fallback that they use. `RAGKIT_EMBEDDER=hashing` forces the fallback.

**Generation is optional:** with no API key, a deterministic *extractive* fallback gives the answer directly from
the retrieved text. Thus each notebook runs from start to end offline. To use a real model instead, set
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`. The code finds the key automatically, and you do not change the code.

Open the notebooks with Jupyter (`pip install jupyterlab && jupyter lab`), or open them in VS Code / Cursor.

## Learning path

Do the notebooks in `notebooks/` in their sequence. Each notebook has a short text about the concept, worked code
that you run, and `# YOUR CODE HERE` exercises with inline checks. The completed versions are in `solutions/`.

| # | Notebook | What you build |
|---|----------|----------------|
| 00 | `setup_and_corpus` | An introduction: the corpus, the eval set, "embedding = vector" |
| 01 | `minimal_rag` | The full loop, in sequence: embed, cosine search, grounded prompt, generate |
| 02 | `chunking` | Fixed, structure-aware and small-to-big chunks compared. Why the split decides retrieval. |
| 03 | `hybrid_search` | BM25 written by hand, and dense search, fused with Reciprocal Rank Fusion |
| 04 | `reranking` | Low-cost first-stage recall, then a cross-encoder for precision |
| 05 | `evaluation` | Hit@k, Recall@k (all gold docs) and MRR. Scores of dense, BM25 and hybrid search, compared by question type. |
| 06 | `iterative_rag` | *(advanced)* multi-hop questions and the loop that agentic RAG uses |

The self-checks examine the **shape** of your implementation: the correct sort order, the correct length and the
correct maths. They do not examine which document a model puts first by chance. Thus they pass with the hashing
fallback or with the real embedder. A green notebook means that your code is correct. The *observations* need the
real model: which retriever is better on which question type, in 03 and 05.

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

The corpus is small and adversarial by design. It has exact IDs and codes (`ERR_4290`, `SEC-011`, `DR-07`) that
give an advantage to lexical search. It has facts that you can paraphrase, and these give an advantage to dense
search. It also has cross-document questions that need more than one retrieval. Thus the lesson of each notebook
actually appears when you measure it.

## Verify the reference code

```bash
python -m pytest -q                               # the from-scratch primitives + the embedder fallback
PYTHONPATH=. python tests/test_notebooks_run.py   # every solution runs + asserts pass
```

## Where to go after 06

Replace the planner of notebook 06 with a real model. Let the model select *which* retriever to call and *when to
stop*. Fine-tune the retriever/reranker on your own query logs. Do an eval of full trajectories, not of single
answers. Parts III–IV of the primer describe these subjects.
