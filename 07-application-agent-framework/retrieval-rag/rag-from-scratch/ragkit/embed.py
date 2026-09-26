"""Thin wrapper around the embedding model, with a labelled T0 fallback.

You do NOT reimplement a transformer to learn RAG, so the embedder is a black
box on purpose. Everything you *do* build (search, fusion, reranking) operates
on the vectors this returns.

Default model: `all-MiniLM-L6-v2` (~90 MB as of 2026-09-26 (verify), CPU-fine). First run downloads it.
Reranker:      `cross-encoder/ms-marco-MiniLM-L-6-v2` (used in notebook 04).
Both need `sentence-transformers` (and so torch): `pip install -r requirements-full.txt`.

T0 fallback: when `sentence-transformers` is not installed, or fails to import
(a broken torch install can raise OSError), `get_embedder()` returns a
`HashingEmbedder` (bag-of-words feature hashing) and `get_cross_encoder()` a
`TokenOverlapReranker`, and each says so once. Every
notebook then runs and every self-check passes (they grade the *shape* of your
code, not model quality), but the vectors are lexical, not semantic: "holidays"
and "vacation" share nothing. Set `RAGKIT_EMBEDDER=hashing` to force the
fallback even when the model is installed.
"""

from __future__ import annotations

import os
import zlib

import numpy as np

from .corpus import tokenize

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FALLBACK_LABEL = "hashing embedder (T0 fallback; not semantic)"
RERANKER_FALLBACK_LABEL = "token-overlap reranker (T0 fallback; not a cross-encoder)"

_INSTALL_HINT = (
    "sentence-transformers is required for real embeddings.\n"
    "    pip install -r requirements-full.txt      # or: pip install sentence-transformers\n"
    "(It pulls in torch; the first call downloads a ~90 MB (verify) model, then it is "
    "cached and runs offline on CPU.)"
)


def have_sentence_transformers() -> bool:
    """True if the real model can be used (installed and not forced off)."""
    if os.environ.get("RAGKIT_EMBEDDER", "").lower() == "hashing":
        return False
    try:
        import sentence_transformers  # noqa: F401
    except Exception:  # not installed (ImportError) or a broken torch (e.g. OSError)
        return False
    return True


class Embedder:
    """Encode text to L2-normalised vectors, so a dot product == cosine."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(_INSTALL_HINT) from e
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        """texts: str or list[str] -> (n, dim) float32, rows L2-normalised."""
        single = isinstance(texts, str)
        arr = self._model.encode(
            [texts] if single else list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,   # <-- makes dot product = cosine
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        return arr[0] if single else arr


class HashingEmbedder:
    """T0 fallback: bag-of-words feature hashing into `dim` buckets, L2-normalised.

    Same interface as `Embedder` (`model_name`, `dim`, `encode`). Deterministic
    across processes (crc32, not Python's salted `hash`). Lexical only: two texts
    are similar exactly when they share tokens, so it behaves like a weak BM25,
    not like a semantic model.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.model_name = FALLBACK_LABEL

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        """texts: str or list[str] -> (n, dim) float32, rows L2-normalised."""
        if isinstance(texts, str):
            return self._vec(texts)
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return np.stack([self._vec(t) for t in texts])


class TokenOverlapReranker:
    """T0 fallback for the cross-encoder: `.predict([(query, passage), ...])`
    scores each pair by the fraction of distinct query tokens found in the passage."""

    model_name = RERANKER_FALLBACK_LABEL

    def predict(self, pairs, **_kwargs) -> np.ndarray:
        out = []
        for q, p in pairs:
            qs, ps = set(tokenize(q)), set(tokenize(p))
            out.append(len(qs & ps) / len(qs) if qs else 0.0)
        return np.asarray(out, dtype="float32")


_CACHE: dict = {}
_ANNOUNCED: set = set()


def _announce(label: str) -> None:
    if label not in _ANNOUNCED:
        _ANNOUNCED.add(label)
        print(f"ragkit: using the {label}. For the real model: pip install -r requirements-full.txt")


def get_embedder(model_name: str = _DEFAULT_MODEL):
    """The embedder factory every notebook uses; one instance per process.

    Returns the sentence-transformers `Embedder` when it is installed, else the
    labelled `HashingEmbedder` fallback.
    """
    if not have_sentence_transformers():
        _announce(FALLBACK_LABEL)
        return _CACHE.setdefault("__hashing__", HashingEmbedder())
    if model_name not in _CACHE:
        _CACHE[model_name] = Embedder(model_name)
    return _CACHE[model_name]


def get_cross_encoder(model_name: str = _DEFAULT_RERANKER):
    """Return a reranker for notebook 04: `.predict([(query, passage), ...])` -> scores.

    A sentence-transformers `CrossEncoder` when installed, else the labelled
    `TokenOverlapReranker` fallback.
    """
    if not have_sentence_transformers():
        _announce(RERANKER_FALLBACK_LABEL)
        return TokenOverlapReranker()
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)
