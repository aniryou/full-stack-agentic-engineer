"""minifaiss demo on the Project Gutenberg corpus.

Turns real public-domain books into vectors and runs vector search over them:

  1. SEMANTIC SEARCH  -- type a natural-language query, get the most similar
     passages back (exact cosine search with IndexFlatIP).
  2. ANN QUALITY      -- measure how well each approximate index (IVF, PQ,
     IVFPQ, LSH, HNSW) reproduces the exact top-10, plus its memory footprint
     and build/search time. This is the classic FAISS accuracy/speed/size
     trade-off, on real data.
  3. index_factory    -- build the same index from a FAISS-style string.

Text is vectorized with a small self-contained TF-IDF vectorizer (see
``tfidf.py``); in a production system you would swap that for a neural embedding
model and feed its vectors into exactly the same minifaiss indexes.

Run:  ./.venv/bin/python demo_gutenberg.py
      ./.venv/bin/python demo_gutenberg.py --max-chunks 800 --no-hnsw
"""

import argparse
import time

import numpy as np

from tfidf import TfidfVectorizer
from gutenberg_corpus import load_chunks

from minifaiss import (
    METRIC_INNER_PRODUCT,
    IndexFlatIP,
    IndexIVFFlat,
    IndexPQ,
    IndexIVFPQ,
    IndexLSH,
    IndexHNSWFlat,
    index_factory,
)

# Natural-language probes chosen to land in the default book set.
DEMO_QUERIES = [
    "the great white whale hunted across the open sea",
    "a young woman receives a proposal of marriage",
    "murder of the king and the bloody crown",
    "the white rabbit hurried down into the garden",
    "the fall of the angels from heaven into hell",
]


def snippet(text, n_words=24):
    """First ``n_words`` words of ``text`` on a single line, for display."""
    words = text.split()
    s = " ".join(words[:n_words])
    return s + (" ..." if len(words) > n_words else "")


def recall_at_k(approx_I, truth_I):
    """Mean fraction of each query's true top-k that the approx index found."""
    k = truth_I.shape[1]
    hits = 0
    for a_row, t_row in zip(approx_I, truth_I):
        hits += len(set(a_row.tolist()) & set(t_row.tolist()))
    return hits / (truth_I.shape[0] * k)


def build_and_query(index, xb, xq, k, needs_train=True, train_vectors=None):
    """Train (optionally), add ``xb``, search ``xq``; return (I, build_s, search_s)."""
    t0 = time.perf_counter()
    if needs_train:
        index.train(train_vectors if train_vectors is not None else xb)
    index.add(xb)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, I = index.search(xq, k)
    search_s = time.perf_counter() - t0
    return I, build_s, search_s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chunks", type=int, default=1500,
                        help="cap on number of passages indexed")
    parser.add_argument("--words-per-chunk", type=int, default=90)
    parser.add_argument("--max-features", type=int, default=256,
                        help="TF-IDF vocabulary size == vector dimension d")
    parser.add_argument("--queries", type=int, default=150,
                        help="number of passages used as ANN benchmark queries")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--no-hnsw", action="store_true",
                        help="skip HNSW (its pure-Python build is the slowest)")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- Load + vectorize -------------------------------------------------- #
    print("Loading Project Gutenberg passages ...")
    texts, books_of = load_chunks(
        words_per_chunk=args.words_per_chunk,
        max_chunks=args.max_chunks,
        seed=args.seed,
    )
    vec = TfidfVectorizer(max_features=args.max_features)
    xb = vec.fit_transform(texts)                      # (N, d) float32, L2-normed
    d = vec.dim
    n = xb.shape[0]
    n_books = len(set(books_of))
    print(f"  {n} passages from {n_books} books, vector dim d={d} "
          f"(TF-IDF, cosine via inner product)\n")

    # ======================================================================= #
    # PART 1 — Semantic search (exact cosine with IndexFlatIP)
    # ======================================================================= #
    print("=" * 74)
    print("PART 1  Semantic search over the corpus (exact, IndexFlatIP = cosine)")
    print("=" * 74)
    flat = IndexFlatIP(d)
    flat.add(xb)
    qvecs = vec.transform(DEMO_QUERIES)
    Dq, Iq = flat.search(qvecs, 3)
    for qi, query in enumerate(DEMO_QUERIES):
        print(f'\nQuery: "{query}"')
        for rank in range(3):
            doc_id = int(Iq[qi, rank])
            score = float(Dq[qi, rank])
            print(f"  #{rank + 1}  cos={score:.3f}  [{books_of[doc_id]}]")
            print(f"        {snippet(texts[doc_id])}")

    # ======================================================================= #
    # PART 2 — Approximate-index quality vs the exact top-k
    # ======================================================================= #
    print("\n" + "=" * 74)
    print(f"PART 2  Approximate search quality (recall@{args.k}) vs exact IndexFlatIP")
    print("=" * 74)

    # Ground truth: exact top-k for a random sample of passages used as queries.
    q_ids = rng.choice(n, size=min(args.queries, n), replace=False)
    xq = xb[q_ids]
    _, truth_I = flat.search(xq, args.k)

    nlist, nprobe, m, nbits_lsh = 48, 8, 8, 256
    # d must be divisible by m for PQ; fall back to m=4 if not.
    if d % m != 0:
        m = 4 if d % 4 == 0 else 2

    rows = []
    # (label, index, needs_train, bytes/vector)
    plan = [
        (f"IndexIVFFlat(nlist={nlist},nprobe={nprobe})",
         _make_ivfflat(d, nlist, nprobe), True, 4 * d),
        (f"IndexPQ(m={m})",
         _make_pq(d, m), True, m),
        (f"IndexIVFPQ(nlist={nlist},m={m},nprobe={nprobe})",
         _make_ivfpq(d, nlist, m, nprobe), True, m),
        (f"IndexLSH(nbits={nbits_lsh})",
         IndexLSH(d, nbits_lsh), False, nbits_lsh // 8),
    ]
    if not args.no_hnsw:
        plan.append((f"IndexHNSWFlat(M=16)",
                     _make_hnsw(d), False, 4 * d))

    # Exact baseline row (recall is 1.0 by definition).
    rows.append(("IndexFlatIP (exact)", 4 * d, 1.0,
                 _time_flat(d, xb, xq, args.k)))

    for label, index, needs_train, bpv in plan:
        I, build_s, search_s = build_and_query(
            index, xb, xq, args.k, needs_train=needs_train, train_vectors=xb)
        rows.append((label, bpv, recall_at_k(I, truth_I), (build_s, search_s)))

    # Print table.
    print(f"\n{'index':<38}{'bytes/vec':>10}{'recall@'+str(args.k):>11}"
          f"{'build s':>9}{'search s':>9}")
    print("-" * 77)
    for label, bpv, rec, (b, s) in [
        (r[0], r[1], r[2], r[3]) for r in rows
    ]:
        print(f"{label:<38}{bpv:>10}{rec:>11.3f}{b:>9.2f}{s:>9.3f}")

    # ======================================================================= #
    # PART 3 — index_factory
    # ======================================================================= #
    print("\n" + "=" * 74)
    print("PART 3  Build an index from a FAISS-style string with index_factory")
    print("=" * 74)
    spec = f"IVF{nlist},PQ{m}"
    fac = index_factory(d, spec, metric_type=METRIC_INNER_PRODUCT)
    fac.nprobe = nprobe
    fac.train(xb)
    fac.add(xb)
    _, I = fac.search(xq, args.k)
    print(f'\n  index_factory(d={d}, "{spec}", METRIC_INNER_PRODUCT)'
          f"  ->  {type(fac).__name__}")
    print(f"  recall@{args.k} = {recall_at_k(I, truth_I):.3f}")

    print("\nSee README.md -> \"Where performance lives\" for how each real-FAISS")
    print('optimization maps to a "# PERF:" comment in the source '
          '(grep -rn "# PERF:" minifaiss/).')


# --- small index builders (keep the plan table readable) -------------------- #
def _make_ivfflat(d, nlist, nprobe):
    idx = IndexIVFFlat(None, d, nlist, metric_type=METRIC_INNER_PRODUCT)
    idx.nprobe = nprobe
    return idx


def _make_pq(d, m):
    return IndexPQ(d, m, metric_type=METRIC_INNER_PRODUCT)


def _make_ivfpq(d, nlist, m, nprobe):
    idx = IndexIVFPQ(None, d, nlist, m, metric_type=METRIC_INNER_PRODUCT)
    idx.nprobe = nprobe
    return idx


def _make_hnsw(d):
    idx = IndexHNSWFlat(d, M=16, metric_type=METRIC_INNER_PRODUCT)
    idx.efConstruction = 32
    idx.efSearch = 48
    return idx


def _time_flat(d, xb, xq, k):
    t0 = time.perf_counter()
    idx = IndexFlatIP(d)
    idx.add(xb)
    b = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx.search(xq, k)
    s = time.perf_counter() - t0
    return (b, s)


if __name__ == "__main__":
    main()
