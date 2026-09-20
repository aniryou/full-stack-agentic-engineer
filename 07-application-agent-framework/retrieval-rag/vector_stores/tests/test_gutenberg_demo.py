"""Tests for the Gutenberg-demo support code and the cosine (inner-product) path.

Covers:
* the homemade TF-IDF vectorizer (``tfidf.py``),
* the corpus loader (``gutenberg_corpus.py``) -- skipped if the NLTK corpus is
  not installed,
* ``IndexIVFPQ`` under ``METRIC_INNER_PRODUCT`` on L2-normalized (cosine)
  vectors, which exercises the residual-ADC inner-product path the demo relies
  on.
"""

import numpy as np
import pytest

from tfidf import TfidfVectorizer, tokenize
from minifaiss import (
    METRIC_L2,
    METRIC_INNER_PRODUCT,
    IndexFlatIP,
    IndexIVFPQ,
)


# --------------------------------------------------------------------------- #
# TF-IDF vectorizer
# --------------------------------------------------------------------------- #
DOCS = [
    "the whale swam across the deep blue ocean sea",
    "a whale is a great creature of the ocean and sea",
    "she accepted his proposal and they were married in spring",
    "the wedding and the marriage were a joyful spring celebration",
    "angels fell from heaven down into the fiery pit of hell",
]


def test_tokenize_drops_stopwords_and_short_tokens():
    toks = tokenize("The Whale, a GREAT creature of the sea!")
    assert "the" not in toks and "of" not in toks and "a" not in toks
    assert "whale" in toks and "great" in toks and "creature" in toks


def test_tfidf_shapes_and_normalization():
    vec = TfidfVectorizer(max_features=32, min_df=1)
    X = vec.fit_transform(DOCS)
    assert X.shape == (len(DOCS), vec.dim)
    assert X.dtype == np.float32
    # Every non-empty row is L2-normalized (unit length).
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_tfidf_cosine_matches_topic_structure():
    vec = TfidfVectorizer(max_features=64, min_df=1)
    X = vec.fit_transform(DOCS)
    # Inner product == cosine because rows are normalized.
    sim = X @ X.T
    # The two whale/ocean docs (0,1) are more similar to each other than doc 0
    # is to the marriage doc (2) or the heaven/hell doc (4).
    assert sim[0, 1] > sim[0, 2]
    assert sim[0, 1] > sim[0, 4]
    # And the two marriage docs (2,3) cluster together too.
    assert sim[2, 3] > sim[2, 0]


def test_tfidf_identical_doc_has_unit_self_similarity():
    vec = TfidfVectorizer(max_features=32, min_df=1)
    X = vec.fit_transform(DOCS)
    self_sim = np.einsum("ij,ij->i", X, X)  # diagonal of X @ X.T
    assert np.allclose(self_sim, 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# Inner-product IVFPQ (the residual-ADC cosine path)
# --------------------------------------------------------------------------- #
def _normalized_clusters(n=1200, d=16, n_blobs=10, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 5, size=(n_blobs, d)).astype(np.float32)
    labels = rng.integers(0, n_blobs, size=n)
    x = (centres[labels] + rng.normal(0, 1, (n, d))).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _recall_at_k(approx_I, truth_I):
    k = truth_I.shape[1]
    hits = sum(len(set(a.tolist()) & set(t.tolist()))
               for a, t in zip(approx_I, truth_I))
    return hits / (truth_I.shape[0] * k)


def test_ivfpq_inner_product_adc_equals_exact_over_reconstructions():
    """The IP residual-ADC score must equal q . reconstruct(id) exactly.

    IVFPQ approximates each database vector as ``centroid + decoded_residual``;
    ``reconstruct`` returns that approximation. A correct inner-product ADC pass
    computes exactly ``q . reconstruct(id)``, so with ``nprobe == nlist``
    (every cell scanned) its top-k must match an EXACT inner-product search over
    the reconstructed vectors. This is the precise regression guard for the
    residual-ADC math: the earlier bug (building the table from the query
    *residual* instead of the full query) made the score differ from
    ``q . reconstruct`` and would fail here.
    """
    x = _normalized_clusters()
    d = x.shape[1]
    nlist, m = 16, 4
    xq = x[:100]

    ivfpq = IndexIVFPQ(None, d, nlist, m, metric_type=METRIC_INNER_PRODUCT)
    ivfpq.train(x)
    ivfpq.add(x)
    ivfpq.nprobe = nlist                       # exhaustive over cells
    D, I = ivfpq.search(xq, 10)

    # Exact IP search over the (approximate) reconstructed vectors.
    R = np.stack([ivfpq.reconstruct(i) for i in range(ivfpq.ntotal)])
    flatR = IndexFlatIP(d)
    flatR.add(R.astype(np.float32))
    _, truthR = flatR.search(xq, 10)

    # ADC ranking must reproduce the exact-over-reconstructions ranking.
    assert _recall_at_k(I, truthR) >= 0.98

    # And the returned score for a hit equals q . reconstruct(id) to fp32.
    q0 = xq[0]
    for rank, vid in enumerate(I[0]):
        vid = int(vid)
        if vid < 0:
            continue
        assert abs(float(D[0, rank]) - float(q0 @ ivfpq.reconstruct(vid))) < 1e-3

    # Sanity: it still retrieves genuinely useful neighbours of the TRUE vectors.
    flat_true = IndexFlatIP(d)
    flat_true.add(x)
    _, truth_true = flat_true.search(xq, 10)
    assert _recall_at_k(I, truth_true) > 0.3


def test_ivfpq_inner_product_scores_descending_and_finite():
    x = _normalized_clusters(n=600, d=16, seed=1)
    d = x.shape[1]
    idx = IndexIVFPQ(None, d, 8, 4, metric_type=METRIC_INNER_PRODUCT)
    idx.train(x)
    idx.add(x)
    idx.nprobe = 8
    D, I = idx.search(x[:20], 5)
    assert D.shape == (20, 5) and I.shape == (20, 5)
    for row in D:
        real = row[np.isfinite(row)]
        # Inner-product results are ranked large -> small.
        assert np.all(np.diff(real) <= 1e-4)


# --------------------------------------------------------------------------- #
# Corpus loader (optional -- needs the downloaded corpus)
# --------------------------------------------------------------------------- #
def test_gutenberg_loader_if_available():
    pytest.importorskip("nltk")
    try:
        from gutenberg_corpus import load_chunks
        texts, books = load_chunks(max_chunks=120, seed=0)
    except LookupError:
        pytest.skip("NLTK 'gutenberg' corpus not downloaded")
    assert 0 < len(texts) <= 120
    assert len(texts) == len(books)
    assert all(isinstance(t, str) and t for t in texts)
    assert len(set(books)) >= 2  # passages come from several books
