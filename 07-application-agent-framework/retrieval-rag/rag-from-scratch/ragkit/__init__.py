"""ragkit — a deliberately tiny toolkit for learning RAG by building it.

Only the *plumbing* lives here (loading the corpus, wrapping the embedding
model, a stub LLM). The parts that actually teach retrieval — cosine search,
BM25, RRF, reranking, chunking, recall@k, MRR — you implement in the
notebooks. Reference implementations are in `ragkit.reference` so later
notebooks can import what earlier ones built, but try the exercises first.
"""

from .corpus import Chunk, load_corpus, load_qrels, tokenize
from .embed import Embedder, HashingEmbedder, get_cross_encoder, get_embedder

__all__ = [
    "Chunk",
    "load_corpus",
    "load_qrels",
    "tokenize",
    "Embedder",
    "HashingEmbedder",
    "get_embedder",
    "get_cross_encoder",
]
