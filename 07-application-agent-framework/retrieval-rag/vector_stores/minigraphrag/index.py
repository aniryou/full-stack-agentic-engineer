"""GraphRAGIndex -- the end-to-end indexing pipeline and the object search runs on.

Calling :meth:`GraphRAGIndex.build` walks the whole GraphRAG indexing pipeline:

    documents
      -> chunk           (chunking.py)
      -> extract graph   (extraction.py + llm.py)      # LLM
      -> detect communities (community.py)
      -> summarize communities (summarize.py + llm.py) # LLM
      -> embed everything (llm.py + vector_store.py)   # embedding model

and stores every artifact in memory. The two search classes
(``local_search.py`` / ``global_search.py``) read from this object.

PERF: the whole pipeline runs single-process and sequential. Production GraphRAG
runs it as a DAG of parallel workflow steps with checkpointing to parquet, so a
failed run resumes instead of re-calling the LLM for everything.
"""

import json
from dataclasses import dataclass

from .models import Document, Entity, Relationship, TextUnit, Community, CommunityReport, _to_jsonable
from .chunking import chunk_documents
from .extraction import extract_graph
from .graph import KnowledgeGraph
from .community import detect_communities, modularity
from .summarize import generate_reports
from .llm import MockLLM, HashingEmbedding
from .vector_store import BruteForceVectorStore


@dataclass
class GraphRAGConfig:
    """Knobs for the indexing pipeline."""

    chunk_size: int = 120        # words per text unit
    chunk_overlap: int = 20      # words shared between adjacent chunks
    embedding_dim: int = 512     # dimensionality of the default hashing embedder
    summarize_threshold: int = 3  # merge descriptions with the LLM past this many
    seed: int = 0                # community-detection determinism
    verbose: bool = False


def _entity_embed_text(entity):
    return f"{entity.name} ({entity.type}). {entity.description}"


class GraphRAGIndex:
    """Holds the LLM/embedder plus every artifact produced by :meth:`build`."""

    def __init__(self, llm=None, embedder=None, vector_store_factory=None, config=None):
        self.config = config or GraphRAGConfig()
        self.llm = llm or MockLLM()
        self.embedder = embedder or HashingEmbedding(self.config.embedding_dim)
        self.vector_store_factory = vector_store_factory or BruteForceVectorStore

        # Artifacts (populated by build()).
        self.documents = {}       # id -> Document
        self.text_units = {}      # id -> TextUnit
        self.graph = KnowledgeGraph()
        self.communities = []     # list[Community]
        self.reports = {}         # community id -> CommunityReport

        # Vector stores.
        self.entity_store = None
        self.text_unit_store = None
        self.report_store = None

    # ------------------------------------------------------------------ #
    def _log(self, msg):
        if self.config.verbose:
            print(msg)

    def build(self, documents):
        """Run the full indexing pipeline over ``documents`` (list[Document])."""
        cfg = self.config
        self.documents = {d.id: d for d in documents}

        self._log("[1/5] chunking documents...")
        units = chunk_documents(documents, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
        self.text_units = {u.id: u for u in units}
        self._log(f"      {len(units)} text units")

        self._log("[2/5] extracting entities & relationships (LLM)...")
        self.graph = extract_graph(units, self.llm, summarize_threshold=cfg.summarize_threshold)
        self._log(f"      {self.graph}")

        self._log("[3/5] detecting communities...")
        self.communities = detect_communities(self.graph, seed=cfg.seed)
        n_levels = self.max_level + 1 if self.communities else 0
        self._log(f"      {len(self.communities)} communities across {n_levels} level(s)")

        self._log("[4/5] summarizing communities (LLM)...")
        self.reports = generate_reports(self.communities, self.graph, self.llm)

        self._log("[5/5] embedding entities, text units, reports...")
        self.build_embeddings()
        self._log("      done.")
        return self

    def build_embeddings(self):
        """(Re)build the three vector stores from current artifacts."""
        # Entities.
        self.entity_store = self.vector_store_factory()
        entities = list(self.graph.entities.values())
        if entities:
            vecs = self.embedder.embed([_entity_embed_text(e) for e in entities])
            self.entity_store.add([e.id for e in entities], vecs)

        # Text units.
        self.text_unit_store = self.vector_store_factory()
        units = list(self.text_units.values())
        if units:
            vecs = self.embedder.embed([u.text for u in units])
            self.text_unit_store.add([u.id for u in units], vecs)

        # Community reports.
        self.report_store = self.vector_store_factory()
        reports = list(self.reports.values())
        if reports:
            vecs = self.embedder.embed([r.full_content for r in reports])
            self.report_store.add([str(r.community_id) for r in reports], vecs)

    # ---- accessors ---------------------------------------------------- #
    @property
    def max_level(self):
        return max((c.level for c in self.communities), default=-1)

    def communities_at_level(self, level):
        return [c for c in self.communities if c.level == level]

    def reports_at_level(self, level):
        return [self.reports[c.id] for c in self.communities_at_level(level) if c.id in self.reports]

    def modularity_at_level(self, level):
        partition = {
            eid: c.id for c in self.communities_at_level(level) for eid in c.entity_ids
        }
        return modularity(self.graph, partition) if partition else 0.0

    def stats(self):
        return {
            "documents": len(self.documents),
            "text_units": len(self.text_units),
            "entities": len(self.graph.entities),
            "relationships": len(self.graph.relationships),
            "communities": len(self.communities),
            "levels": self.max_level + 1 if self.communities else 0,
            "reports": len(self.reports),
        }

    # ---- serialization ------------------------------------------------ #
    def to_dict(self):
        """Serialize artifacts (not embeddings -- those are rebuilt on load)."""
        return {
            "config": _to_jsonable(self.config),
            "documents": [_to_jsonable(d) for d in self.documents.values()],
            "text_units": [_to_jsonable(u) for u in self.text_units.values()],
            "entities": [_to_jsonable(e) for e in self.graph.entities.values()],
            "relationships": [_to_jsonable(r) for r in self.graph.relationships.values()],
            "communities": [_to_jsonable(c) for c in self.communities],
            "reports": [_to_jsonable(r) for r in self.reports.values()],
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data, llm=None, embedder=None, vector_store_factory=None, build_embeddings=True):
        """Rebuild an index (and its vector stores) from :meth:`to_dict` output."""
        cfg = GraphRAGConfig(**data.get("config", {}))
        idx = cls(llm=llm, embedder=embedder, vector_store_factory=vector_store_factory, config=cfg)
        idx.documents = {d["id"]: Document(**d) for d in data["documents"]}
        idx.text_units = {u["id"]: TextUnit(**u) for u in data["text_units"]}

        idx.graph = KnowledgeGraph()
        for e in data["entities"]:
            ent = Entity(**e)
            # JSON turns dict int keys into strings; restore int community levels
            # so that `community_by_level.get(level)` matches during search.
            ent.community_by_level = {int(k): v for k, v in ent.community_by_level.items()}
            idx.graph.entities[ent.id] = ent
            idx.graph._adj.setdefault(ent.id, {})
        for r in data["relationships"]:
            rel = Relationship(**r)
            key = (rel.source, rel.target)
            idx.graph.relationships[key] = rel
            idx.graph._adj[rel.source][rel.target] = rel.weight
            idx.graph._adj[rel.target][rel.source] = rel.weight

        idx.communities = [Community(**c) for c in data["communities"]]
        idx.reports = {r["community_id"]: CommunityReport(**r) for r in data["reports"]}

        if build_embeddings:
            idx.build_embeddings()
        return idx

    @classmethod
    def load(cls, path, **kwargs):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f), **kwargs)

    def __repr__(self):
        s = self.stats()
        return (f"GraphRAGIndex(entities={s['entities']}, relationships={s['relationships']}, "
                f"communities={s['communities']}, levels={s['levels']})")
