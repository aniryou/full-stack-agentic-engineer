"""Verify the reference primitives with a deterministic stub embedder.

Uses seeded random vectors (retrieval-algorithm correctness does not depend on
embedding *quality*), plus a tiny hand-checkable BM25/RRF/metrics case.
Run: `python -m pytest tests/ -q`  or  `python tests/test_reference.py`.
"""
import numpy as np

from ragkit import reference as R
from ragkit.corpus import load_documents, tokenize


class StubEmbedder:
    """Deterministic bag-of-words hashing embedder — good enough to test wiring
    and cosine ranking without downloading a model."""
    dim = 64

    def encode(self, texts):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out = np.zeros((len(items), self.dim), dtype="float32")
        for r, t in enumerate(items):
            for tok in tokenize(t):
                out[r, hash(tok) % self.dim] += 1.0
            n = np.linalg.norm(out[r])
            if n:
                out[r] /= n
        return out[0] if single else out


def test_chunkers():
    docs = load_documents()
    doc_id = "expense-policy"
    fixed = R.fixed_size_chunks(doc_id, docs[doc_id], size=40, overlap=10)
    assert len(fixed) >= 2 and all(c.doc_id == doc_id for c in fixed)

    sect = R.structure_aware_chunks(doc_id, docs[doc_id])
    headings = {c.section for c in sect}
    assert {"Submitting a Claim", "Receipts", "Meal Limits"} <= headings
    # header line is prepended and self-describing
    meal = next(c for c in sect if c.section == "Meal Limits")
    assert meal.text.startswith("Expense Policy > Meal Limits")
    assert "SGD 90" in meal.text

    small, parent_of = R.sentence_chunks(doc_id, docs[doc_id])
    assert len(small) > len(sect)
    # every small chunk maps back to a section chunk
    assert all(parent_of[c.chunk_id].section for c in small)


def test_dense_search_ranks_self_highest():
    emb = StubEmbedder()
    docs = load_documents()
    chunks = R.chunk_corpus(docs, R.structure_aware_chunks)
    mat = R.build_dense_index(emb, chunks)
    assert mat.shape == (len(chunks), emb.dim)
    # querying with a chunk's own text should retrieve that chunk at rank 1
    target = next(c for c in chunks if c.section == "Meal Limits")
    hits = R.dense_search(emb, mat, chunks, target.text, k=3)
    assert hits[0][0].chunk_id == target.chunk_id
    assert hits[0][1] > 0.99  # cosine with itself ~1


def test_bm25_exact_term():
    docs = load_documents()
    chunks = R.chunk_corpus(docs, R.structure_aware_chunks)
    bm = R.BM25([tokenize(c.text) for c in chunks])
    top_i, top_score = bm.search("ERR_4290 certificate expired", k=1)[0]
    assert chunks[top_i].doc_id == "vpn-setup"
    assert top_score > 0
    # a term in no document contributes zero
    assert bm.score(tokenize("zzzznonexistent"), 0) == 0.0


def test_bm25_hand_checkable():
    # doc0 has the query term twice, doc1 once, doc2 never.
    corpus = [["cat", "cat", "mat"], ["cat", "hat"], ["dog", "log"]]
    bm = R.BM25(corpus, k1=1.5, b=0.75)
    s0 = bm.score(["cat"], 0)
    s1 = bm.score(["cat"], 1)
    s2 = bm.score(["cat"], 2)
    assert s0 > s1 > 0            # more term frequency -> higher score
    assert s2 == 0.0             # absent term -> zero


def test_rrf_orders_by_consensus():
    # 'b' is high in both lists; 'a' tops one; RRF should favour 'b' overall.
    r1 = ["a", "b", "c"]
    r2 = ["b", "c", "a"]
    fused = dict(R.reciprocal_rank_fusion([r1, r2], k=60))
    assert fused["b"] > fused["a"] > fused["c"] or fused["b"] > fused["c"] > fused["a"]
    assert max(fused, key=fused.get) == "b"


def _unit(deg):
    r = np.radians(deg)
    v = np.array([np.cos(r), np.sin(r)], dtype="float32")
    return v / np.linalg.norm(v)


def test_mmr_promotes_diversity():
    # Query near two near-identical docs, plus one less-relevant but *novel* doc.
    q = _unit(10)
    vecs = np.stack([_unit(0), _unit(2), _unit(40)])   # dup1, dup2, diverse
    ids = ["dup1", "dup2", "diverse"]
    rel = vecs @ q

    # Plain top-2 by relevance would take the two near-duplicates and drop novelty.
    plain_top2 = [ids[i] for i in np.argsort(-rel)[:2]]
    assert "diverse" not in plain_top2

    # MMR (balanced) keeps the best hit but swaps the redundant twin for novelty.
    picked = R.mmr(q, vecs, ids, lambda_=0.5, k=2)
    assert len(picked) == 2
    assert "diverse" in picked
    assert not {"dup1", "dup2"} <= set(picked)   # not both near-duplicates


def test_metrics():
    ranked = ["a", "b", "c", "d"]
    assert R.recall_at_k(ranked, ["c"], k=3) == 1.0
    assert R.recall_at_k(ranked, ["c"], k=2) == 0.0
    assert R.recall_at_k(ranked, ["a", "d"], k=2, require_all=True) == 0.0
    assert R.recall_at_k(ranked, ["a", "b"], k=2, require_all=True) == 1.0
    assert R.mrr(ranked, ["b"]) == 0.5
    assert R.mrr(ranked, ["x"]) == 0.0
    assert R.to_doc_ranking(["d1#s1", "d1#s2", "d2#s1"]) == ["d1", "d2"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"\nAll {len(fns)} reference tests passed.")
