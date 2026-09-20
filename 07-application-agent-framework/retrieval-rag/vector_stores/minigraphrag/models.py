"""Data model for minigraphrag -- the artifacts a GraphRAG index is made of.

These plain dataclasses are the "tables" that Microsoft's GraphRAG writes to
parquet during indexing (``documents``, ``text_units``, ``entities``,
``relationships``, ``communities``, ``community_reports``). Here they live in
memory as ordinary Python objects so the whole pipeline is easy to inspect:

    Document  --chunk-->  TextUnit  --LLM extract-->  Entity + Relationship
                                          |
                                          +--graph + clustering--> Community
                                                                       |
                                                          --LLM summarize--> CommunityReport

Nothing here is numpy -- vectors live in the vector stores (see
``vector_store.py``), not on the entities -- so this module has no heavy deps.
"""

from dataclasses import dataclass, field, asdict


def normalize_name(name):
    """Canonical key for an entity name: trimmed, collapsed spaces, upper-case.

    Entity resolution in real GraphRAG is much richer (an LLM decides that
    "Dr. Watson" and "John Watson" are the same node). Here we simply upper-case
    the surface form, so identical spellings across chunks merge into one node.
    """
    return " ".join(str(name).split()).upper()


@dataclass
class Document:
    """A source document fed into the pipeline."""

    id: str
    text: str
    title: str = ""


@dataclass
class TextUnit:
    """A chunk of a document -- GraphRAG's atomic unit of retrieval ("text unit").

    Extraction runs per text unit, and local search cites text units as the raw
    "sources" behind an answer. ``entity_ids`` / ``relationship_ids`` are filled
    in during extraction so we can walk back from a graph element to the text it
    came from.
    """

    id: str
    text: str
    document_id: str
    index: int  # position of this chunk within its document (0-based)
    entity_ids: list = field(default_factory=list)
    relationship_ids: list = field(default_factory=list)


@dataclass
class Entity:
    """A node in the knowledge graph (a person, org, place, concept, ...).

    ``id`` is the normalized name (the merge key); ``name`` is the display form.
    ``description`` is the merged, possibly LLM-summarized, gloss. ``degree`` and
    ``rank`` are graph statistics used to order context during search.
    """

    id: str
    name: str
    type: str = "UNKNOWN"
    description: str = ""
    text_unit_ids: list = field(default_factory=list)
    # community id per hierarchy level, e.g. {0: 3, 1: 7}. Filled by clustering.
    community_by_level: dict = field(default_factory=dict)
    degree: int = 0
    rank: float = 0.0


@dataclass
class Relationship:
    """An undirected, weighted edge between two entities.

    ``weight`` is the combined strength of the tie (the LLM rates each mention
    1-10 in real GraphRAG; here we sum co-occurrence counts). It is the edge
    weight that community detection optimizes over.
    """

    id: str
    source: str  # entity id
    target: str  # entity id
    description: str = ""
    weight: float = 1.0
    text_unit_ids: list = field(default_factory=list)


@dataclass
class Community:
    """A cluster of entities found by community detection, at one hierarchy level.

    Communities are hierarchical: a fine-grained community at level ``L`` rolls
    up into a coarser ``parent_id`` at level ``L-1``. Level 0 is the coarsest
    (fewest, largest communities); higher levels are finer. (Microsoft GraphRAG
    numbers levels the same way, via hierarchical Leiden.)
    """

    id: int  # globally unique across all levels
    level: int
    entity_ids: list = field(default_factory=list)
    relationship_ids: list = field(default_factory=list)
    parent_id: int = -1  # community id one level coarser, or -1 at level 0
    child_ids: list = field(default_factory=list)
    size: int = 0


@dataclass
class CommunityReport:
    """An LLM-authored summary of one community -- GraphRAG's headline artifact.

    Global search reads *only* these reports (never the raw graph), so they are
    what lets GraphRAG answer whole-corpus questions. ``full_content`` is the
    flattened text used for embedding and map-reduce.
    """

    community_id: int
    level: int
    title: str = ""
    summary: str = ""
    findings: list = field(default_factory=list)  # list of {summary, explanation}
    rank: float = 0.0  # importance 0-10; orders reports for global search
    full_content: str = ""

    def as_context(self):
        """Render the report as a compact block of text for a prompt/context."""
        lines = [f"# {self.title}", "", self.summary, ""]
        for f in self.findings:
            lines.append(f"## {f.get('summary', '')}")
            lines.append(f.get("explanation", ""))
            lines.append("")
        return "\n".join(lines).strip()


def _to_jsonable(obj):
    """Recursively turn a dataclass (or list/dict of them) into plain JSON types."""
    from dataclasses import is_dataclass

    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj
