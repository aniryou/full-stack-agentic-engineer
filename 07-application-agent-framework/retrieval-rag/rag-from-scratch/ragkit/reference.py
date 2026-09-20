"""Reference implementations of everything you build in the notebooks.

Two uses:
  1. The `solutions/` notebooks match this code — it is the answer key.
  2. Later notebooks import primitives that earlier notebooks built (e.g. the
     structure-aware chunker from 02) instead of re-deriving them.

Every function here is short on purpose. Read it *after* attempting the
exercise, not before.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from .corpus import Chunk, tokenize

# --------------------------------------------------------------------------- #
# Chunking (notebook 02)
# --------------------------------------------------------------------------- #

def fixed_size_chunks(doc_id: str, text: str, size: int = 120,
                      overlap: int = 20) -> list[Chunk]:
    """Split into fixed word windows with overlap. Ignores structure."""
    words = text.split()
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        window = words[i:i + size]
        if not window:
            break
        chunks.append(Chunk(chunk_id=f"{doc_id}#fixed-{len(chunks)}",
                            doc_id=doc_id, text=" ".join(window)))
        if i + size >= len(words):
            break
    return chunks


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def structure_aware_chunks(doc_id: str, text: str) -> list[Chunk]:
    """One chunk per `##` section, with the doc title + heading prepended so
    the chunk is self-describing ('Expense Policy > Meal Limits\\n...')."""
    lines = text.splitlines()
    title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("# ") else doc_id
    chunks, heading, buf = [], None, []

    def flush():
        if heading and buf:
            body = "\n".join(buf).strip()
            if body:
                header = f"{title} > {heading}"
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}#{_slug(heading)}", doc_id=doc_id,
                    text=f"{header}\n{body}", section=heading,
                    meta={"title": title}))

    for line in lines:
        if line.startswith("## "):
            flush()
            heading, buf = line[3:].strip(), []
        elif not line.startswith("# "):
            buf.append(line)
    flush()
    return chunks


def sentence_chunks(doc_id: str, text: str):
    """small-to-big: return (small_chunks, parent_of) where each small chunk is
    one sentence and parent_of[small_id] is the big (section) Chunk to return."""
    parents = structure_aware_chunks(doc_id, text)
    small, parent_of = [], {}
    for parent in parents:
        # skip the "title > heading" header line when splitting into sentences
        body = parent.text.split("\n", 1)[1] if "\n" in parent.text else parent.text
        for j, sent in enumerate(re.split(r"(?<=[.!?])\s+", body)):
            sent = sent.strip()
            if not sent:
                continue
            sid = f"{parent.chunk_id}::s{j}"
            small.append(Chunk(chunk_id=sid, doc_id=doc_id, text=sent,
                              section=parent.section))
            parent_of[sid] = parent
    return small, parent_of


def chunk_corpus(documents: dict[str, str], chunker=structure_aware_chunks) -> list[Chunk]:
    out = []
    for doc_id, text in documents.items():
        out.extend(chunker(doc_id, text))
    return out


# --------------------------------------------------------------------------- #
# Dense retrieval (notebook 01)
# --------------------------------------------------------------------------- #

def build_dense_index(embedder, chunks: list[Chunk]) -> np.ndarray:
    """Return an (N, dim) matrix of L2-normalised chunk vectors."""
    return embedder.encode([c.text for c in chunks])


def dense_search(embedder, matrix: np.ndarray, chunks: list[Chunk],
                 query: str, k: int = 5):
    """Cosine search: because rows are normalised, dot product == cosine."""
    qv = embedder.encode(query)
    scores = matrix @ qv                       # (N,)
    order = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in order]


# --------------------------------------------------------------------------- #
# Sparse retrieval: BM25 (notebook 03)
# --------------------------------------------------------------------------- #

class BM25:
    """Textbook Okapi BM25 over pre-tokenised documents."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus_tokens = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / max(1, self.N)
        self.tf = [Counter(d) for d in corpus_tokens]
        df = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        # BM25 idf with +1 to keep it non-negative
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, query_tokens: list[str], i: int) -> float:
        dl = len(self.corpus_tokens[i])
        s = 0.0
        for t in query_tokens:
            f = self.tf[i].get(t, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
        return s

    def search(self, query: str, k: int = 5):
        q = tokenize(query)
        scores = [(i, self.score(q, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# --------------------------------------------------------------------------- #
# Fusion & diversity (notebook 03 / 04)
# --------------------------------------------------------------------------- #

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Combine several ranked id-lists into one. Rank-based, so it doesn't care
    that BM25 and cosine scores live on different scales."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, cand_ids: list[str],
        lambda_: float = 0.7, k: int = 5) -> list[str]:
    """Maximal Marginal Relevance: trade relevance against redundancy."""
    selected, selected_idx = [], []
    remaining = list(range(len(cand_ids)))
    rel = cand_vecs @ query_vec
    while remaining and len(selected) < k:
        best_j, best_score = None, -1e9
        for j in remaining:
            if selected_idx:
                redundancy = max(float(cand_vecs[j] @ cand_vecs[s]) for s in selected_idx)
            else:
                redundancy = 0.0
            score = lambda_ * float(rel[j]) - (1 - lambda_) * redundancy
            if score > best_score:
                best_score, best_j = score, j
        selected.append(cand_ids[best_j])
        selected_idx.append(best_j)
        remaining.remove(best_j)
    return selected


# --------------------------------------------------------------------------- #
# Evaluation metrics (notebook 05)
# --------------------------------------------------------------------------- #

def to_doc_ranking(chunk_ranking: list[str]) -> list[str]:
    """Collapse a ranked list of chunk_ids to unique doc_ids, keeping order.
    (gold labels are at document level.)"""
    seen, out = set(), []
    for cid in chunk_ranking:
        doc = cid.split("#", 1)[0]
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


def recall_at_k(ranked_docs: list[str], gold_docs: list[str], k: int,
                require_all: bool = False) -> float:
    top = ranked_docs[:k]
    hits = [g in top for g in gold_docs]
    return float(all(hits)) if require_all else float(any(hits))


def mrr(ranked_docs: list[str], gold_docs: list[str]) -> float:
    for rank, doc in enumerate(ranked_docs, start=1):
        if doc in gold_docs:
            return 1.0 / rank
    return 0.0
