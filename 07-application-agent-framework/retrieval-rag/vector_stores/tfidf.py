"""A tiny, self-contained TF-IDF text vectorizer (numpy only).

This is NOT part of the ``minifaiss`` library -- FAISS itself never turns text
into vectors; it only indexes vectors you already have. In a real system you
would embed text with a neural model (e.g. sentence-transformers, OpenAI /
Cohere embeddings, or a local model) and feed the resulting float vectors into
FAISS.

Here we use classic TF-IDF purely so the Gutenberg demo has *some* meaningful
dense vectors without pulling in a heavyweight embedding model or downloading
network weights. The pipeline is:

    tokenize -> build a top-N vocabulary -> TF-IDF weight -> L2-normalize

L2-normalizing means the inner product between two vectors equals their cosine
similarity, so we can search with ``METRIC_INNER_PRODUCT`` and read the scores
as "how similar" in [-1, 1].
"""

import re
import math

import numpy as np

# A compact English stop-word list. Small on purpose: this is a demo helper,
# not a linguistics project.
_STOPWORDS = {
    "the", "and", "of", "to", "a", "in", "that", "it", "is", "was", "he",
    "for", "on", "with", "as", "his", "her", "she", "him", "had", "have",
    "has", "be", "by", "not", "but", "at", "this", "which", "or", "from",
    "they", "you", "all", "we", "an", "are", "were", "their", "them", "so",
    "if", "no", "would", "there", "what", "one", "will", "who", "when",
    "there", "been", "my", "your", "our", "its", "into", "than", "then",
    "such", "did", "do", "does", "up", "out", "more", "some", "these",
    "those", "i", "me", "am", "shall", "should", "could", "may", "might",
    "must", "can", "very", "upon", "about", "any", "how", "now", "too",
}

_TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(text):
    """Lower-case ``text`` and split into alphabetic tokens >= 2 chars.

    Stop-words are dropped. Deliberately simple (no stemming/lemmatization).
    """
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if len(t) >= 2 and t not in _STOPWORDS
    ]


class TfidfVectorizer:
    """Fit a top-``max_features`` TF-IDF vocabulary, then vectorize documents.

    Parameters
    ----------
    max_features : int
        Keep only the ``max_features`` most frequent (by document frequency)
        vocabulary terms. This is the output dimensionality ``d``.
    min_df : int
        Ignore terms that appear in fewer than ``min_df`` documents.
    max_df_ratio : float
        Ignore terms that appear in more than this fraction of documents
        (removes corpus-wide boilerplate that carries little signal).
    """

    def __init__(self, max_features=256, min_df=2, max_df_ratio=0.5):
        self.max_features = int(max_features)
        self.min_df = int(min_df)
        self.max_df_ratio = float(max_df_ratio)
        self.vocabulary_ = {}      # term -> column index
        self.idf_ = None           # (d,) float32 inverse-document-frequency
        self.terms_ = []           # column index -> term (for inspection)

    def fit(self, docs):
        """Learn the vocabulary and IDF weights from an iterable of strings."""
        n_docs = len(docs)
        tokenized = [tokenize(doc) for doc in docs]

        # Document frequency: in how many docs each term appears.
        df = {}
        for toks in tokenized:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1

        # Filter by df bounds, then keep the most frequent terms.
        max_df = self.max_df_ratio * n_docs
        candidates = [
            (term, c) for term, c in df.items()
            if c >= self.min_df and c <= max_df
        ]
        # Sort by descending document frequency (ties broken alphabetically for
        # determinism), then take the top max_features.
        candidates.sort(key=lambda kv: (-kv[1], kv[0]))
        candidates = candidates[:self.max_features]

        self.terms_ = [term for term, _ in candidates]
        self.vocabulary_ = {term: j for j, term in enumerate(self.terms_)}

        # Smoothed IDF (adds 1 to numerator/denominator, like scikit-learn).
        d = len(self.terms_)
        idf = np.empty(d, dtype=np.float32)
        for term, j in self.vocabulary_.items():
            idf[j] = math.log((1 + n_docs) / (1 + df[term])) + 1.0
        self.idf_ = idf
        return self

    def transform(self, docs):
        """Vectorize ``docs`` into an L2-normalized (n, d) float32 matrix."""
        d = len(self.terms_)
        out = np.zeros((len(docs), d), dtype=np.float32)
        for i, doc in enumerate(docs):
            counts = {}
            for term in tokenize(doc):
                j = self.vocabulary_.get(term)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                # Sub-linear term frequency (log) dampens very repetitive terms.
                out[i, j] = (1.0 + math.log(c)) * self.idf_[j]

        # L2-normalize each row so inner product == cosine similarity. Rows that
        # contain no vocabulary terms stay all-zero (norm forced to 1 to avoid
        # divide-by-zero); their similarity to anything is 0.
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        out /= norms
        return out

    def fit_transform(self, docs):
        """Convenience: :meth:`fit` then :meth:`transform` on the same docs."""
        return self.fit(docs).transform(docs)

    @property
    def dim(self):
        """Output dimensionality (== number of vocabulary terms kept)."""
        return len(self.terms_)
