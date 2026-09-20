"""minigraphrag -- an educational reimplementation of Microsoft's GraphRAG.

GraphRAG answers questions over a corpus by first building a *knowledge graph*
from it with an LLM, clustering that graph into a hierarchy of communities,
summarizing each community, and then answering with either a focused subgraph
(local search) or a map-reduce over community summaries (global search).

minigraphrag rebuilds those ideas in pure Python + numpy, for LEARNING rather
than performance or accuracy. It runs fully offline by default via a
deterministic ``MockLLM``; point it at a real model (``AnthropicLLM``) for real
quality. Two families of comments flag where the real system differs:

* ``# LLM:``  -- a call that a real language / embedding model performs. Grep
                 for these to see exactly where intelligence enters the pipeline.
* ``# PERF:`` -- a spot where real GraphRAG reaches for scale (concurrency,
                 caching, a vector DB, C-backed graph clustering, ...).

    grep -rn "# LLM:"  minigraphrag/
    grep -rn "# PERF:" minigraphrag/

Typical use::

    from minigraphrag import GraphRAGIndex, Document, LocalSearch, GlobalSearch

    docs = [Document(id="d1", text="...", title="...")]
    index = GraphRAGIndex().build(docs)          # offline MockLLM by default

    LocalSearch(index).search("Who is Alice?").answer
    GlobalSearch(index).search("What are the main themes?").answer
"""

__version__ = "0.1.0"

from .models import (
    Document, TextUnit, Entity, Relationship, Community, CommunityReport, normalize_name,
)
from .chunking import chunk_documents, chunk_text
from .graph import KnowledgeGraph
from .llm import (
    LanguageModel, MockLLM, AnthropicLLM, EmbeddingModel, HashingEmbedding,
    heuristic_extract,
)
from .extraction import extract_graph, parse_records
from .community import detect_communities, louvain_dendrogram, modularity
from .summarize import generate_reports
from .vector_store import VectorStore, BruteForceVectorStore, MiniFaissVectorStore
from .index import GraphRAGIndex, GraphRAGConfig
from .local_search import LocalSearch, LocalSearchResult
from .global_search import GlobalSearch, GlobalSearchResult

__all__ = [
    "__version__",
    # data model
    "Document", "TextUnit", "Entity", "Relationship", "Community", "CommunityReport",
    "normalize_name",
    # pipeline stages
    "chunk_documents", "chunk_text",
    "KnowledgeGraph",
    "extract_graph", "parse_records", "heuristic_extract",
    "detect_communities", "louvain_dendrogram", "modularity",
    "generate_reports",
    # model layer
    "LanguageModel", "MockLLM", "AnthropicLLM", "EmbeddingModel", "HashingEmbedding",
    # vector stores
    "VectorStore", "BruteForceVectorStore", "MiniFaissVectorStore",
    # top-level
    "GraphRAGIndex", "GraphRAGConfig",
    "LocalSearch", "LocalSearchResult",
    "GlobalSearch", "GlobalSearchResult",
]
