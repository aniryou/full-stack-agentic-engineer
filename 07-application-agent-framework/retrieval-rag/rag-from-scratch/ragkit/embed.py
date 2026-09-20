"""Thin wrapper around the embedding model.

You do NOT reimplement a transformer to learn RAG, so the embedder is a black
box on purpose. Everything you *do* build (search, fusion, reranking) operates
on the vectors this returns.

Default model: `all-MiniLM-L6-v2` (~90 MB, CPU-fine). First run downloads it.
Reranker:      `cross-encoder/ms-marco-MiniLM-L-6-v2` (used in notebook 04).
"""

from __future__ import annotations

import numpy as np

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_INSTALL_HINT = (
    "sentence-transformers is required for real embeddings.\n"
    "    pip install sentence-transformers\n"
    "(The first call downloads a ~90 MB model; afterwards it is cached and "
    "runs offline on CPU.)"
)


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


def get_embedder(model_name: str = _DEFAULT_MODEL) -> Embedder:
    """One embedder per process is plenty; cache it."""
    global _CACHE
    try:
        _CACHE
    except NameError:
        _CACHE = {}
    if model_name not in _CACHE:
        _CACHE[model_name] = Embedder(model_name)
    return _CACHE[model_name]


def get_cross_encoder(model_name: str = _DEFAULT_RERANKER):
    """Return a sentence-transformers CrossEncoder for notebook 04.

    Call `.predict([(query, passage), ...])` to get relevance scores.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:  # pragma: no cover
        raise ImportError(_INSTALL_HINT) from e
    return CrossEncoder(model_name)
