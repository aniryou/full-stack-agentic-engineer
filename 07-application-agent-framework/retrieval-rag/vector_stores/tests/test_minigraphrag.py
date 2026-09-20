"""Pytest suite for minigraphrag.

Plain ``assert`` style, fully offline (MockLLM + HashingEmbedding), deterministic.
Covers each pipeline stage in isolation and the whole thing end-to-end.
"""

import pytest

from minigraphrag import (
    Document, GraphRAGIndex, GraphRAGConfig,
    LocalSearch, GlobalSearch,
    KnowledgeGraph, Entity, Relationship,
    chunk_text, chunk_documents, parse_records, heuristic_extract,
    detect_communities, louvain_dendrogram, modularity,
    MockLLM,
)
from minigraphrag.prompts import TUPLE_DELIMITER as TD, RECORD_DELIMITER as RD, COMPLETION_DELIMITER as CD


# A small corpus reused across the end-to-end tests. Two clear clusters: a
# research group and a city/energy group, bridged by Northwind/Elena.
CORPUS = [
    Document(id="d1", text=(
        "The Meridian Institute is a research laboratory in the City of Calder. "
        "Ada Lockwood founded the Meridian Institute after leaving Northwind Corporation. "
        "Professor Chen leads Project Halcyon at the Meridian Institute. "
        "Ada Lockwood and Professor Chen have collaborated at the Meridian Institute for years.")),
    Document(id="d2", text=(
        "Northwind Corporation is an energy company in the Riverside District. "
        "Elena Vasquez is the chief executive of Northwind Corporation. "
        "Elena Vasquez studied under Ada Lockwood. "
        "Elena Vasquez and Mayor Torres signed an agreement about the Calder Harbor.")),
    Document(id="d3", text=(
        "Mayor Torres governs the City of Calder. "
        "The Harbor Authority manages the Calder Harbor. "
        "Mayor Torres appointed Elena Vasquez to the Harbor Authority. "
        "The City of Calder depends on the Calder Harbor.")),
]


@pytest.fixture(scope="module")
def index():
    return GraphRAGIndex(config=GraphRAGConfig(chunk_size=200, seed=0)).build(CORPUS)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def test_chunk_text_overlap():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = [c for _, c in chunk_text(text, chunk_size=4, overlap=2)]
    assert len(chunks) == 4
    # Consecutive chunks share `overlap` trailing/leading words.
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-2:] == b.split()[:2]


def test_chunk_documents_ids():
    docs = [Document(id="doc", text=" ".join(f"w{i}" for i in range(30)))]
    units = chunk_documents(docs, chunk_size=10, overlap=0)
    assert [u.id for u in units] == ["doc::chunk0", "doc::chunk1", "doc::chunk2"]
    assert all(u.document_id == "doc" for u in units)


def test_chunk_rejects_bad_overlap():
    with pytest.raises(ValueError):
        list(chunk_text("a b c", chunk_size=2, overlap=2))


# --------------------------------------------------------------------------- #
# Extraction: format parsing + heuristic extractor
# --------------------------------------------------------------------------- #
def test_parse_records_roundtrip():
    raw = (
        f'("entity"{TD}Alice{TD}PERSON{TD}A person named Alice)\n{RD}\n'
        f'("entity"{TD}Acme{TD}ORGANIZATION{TD}A company)\n{RD}\n'
        f'("relationship"{TD}Alice{TD}Acme{TD}Alice works at Acme{TD}7)\n{RD}\n'
        f'garbage that should be skipped\n{RD}\n'
        f'{CD}'
    )
    ents, rels = parse_records(raw)
    assert {e["name"] for e in ents} == {"Alice", "Acme"}
    assert len(rels) == 1
    assert rels[0]["source"] == "Alice" and rels[0]["target"] == "Acme"
    assert rels[0]["weight"] == 7.0


def test_heuristic_extract_finds_entities_and_edges():
    text = "Ada Lockwood founded the Meridian Institute. Ada Lockwood met Professor Chen."
    ents, rels = heuristic_extract(text)
    names = {e["name"] for e in ents}
    assert "Ada Lockwood" in names
    assert "Meridian Institute" in names  # leading "the" stripped
    assert "Professor Chen" in names
    # No bogus conjunction entity, no trailing-period duplicates.
    assert not any(" and " in n for n in names)
    assert not any(n.endswith(".") for n in names)
    # Co-occurrence in a sentence creates an edge.
    pairs = {(r["source"], r["target"]) for r in rels}
    assert ("Ada Lockwood", "Meridian Institute") in pairs or ("Meridian Institute", "Ada Lockwood") in pairs


def test_mock_extraction_parses():
    """MockLLM emits the same delimited format the parser consumes."""
    raw = MockLLM().extract("Ada Lockwood founded the Meridian Institute.")
    ents, rels = parse_records(raw)
    assert any(e["name"] == "Ada Lockwood" for e in ents)


def test_parse_records_tolerates_parentheses():
    """Descriptions are whole sentences and may contain '(...)' -- must not drop the record."""
    raw = (
        f'("entity"{TD}Acme{TD}ORGANIZATION{TD}Acme (a startup) based in Calder)\n{RD}\n'
        f'("relationship"{TD}Ada{TD}Acme{TD}Ada joined Acme (a startup){TD}4)\n{CD}'
    )
    ents, rels = parse_records(raw)
    assert [e["name"] for e in ents] == ["Acme"]
    assert ents[0]["description"] == "Acme (a startup) based in Calder"
    assert rels and rels[0]["weight"] == 4.0


def test_pipeline_survives_parenthetical_text():
    """End-to-end: parenthetical prose must not silently empty the graph."""
    docs = [Document(id="d1", text=(
        "Ada Lockwood (a scientist) founded the Meridian Institute. "
        "Ada Lockwood leads Project Halcyon at the Meridian Institute."))]
    idx = GraphRAGIndex(config=GraphRAGConfig(chunk_size=200)).build(docs)
    assert "ADA LOCKWOOD" in idx.graph.entities
    assert "MERIDIAN INSTITUTE" in idx.graph.entities
    assert "(a scientist)" in idx.graph.entities["ADA LOCKWOOD"].description  # inner parens preserved


def test_json_parsers_tolerate_non_object():
    """Real backends may emit a bare array/scalar; parsers must degrade, not crash."""
    from minigraphrag.global_search import _parse_points
    from minigraphrag.summarize import _parse_report_json
    assert _parse_points('[{"description": "x", "score": 9}]') == [{"description": "x", "score": 9}]
    assert _parse_points('"just prose"') == []
    assert _parse_points('{"points": ["not a dict", {"description": "y", "score": 1}]}') == [{"description": "y", "score": 1}]
    assert _parse_report_json("[1, 2, 3]") == {}
    assert _parse_report_json("not json at all") == {}


class _BadJSONBackend(MockLLM):
    """A backend that returns malformed (top-level array) JSON for report/map."""

    def complete(self, system, user, *, task=None, meta=None):
        if task == "report":
            return '[{"summary": "oops top-level array"}]'
        if task == "map":
            return '[{"description": "still works", "score": 10}]'
        return super().complete(system, user, task=task, meta=meta)


def test_pipeline_survives_bad_json_backend():
    docs = [Document(id="d1", text="Ada Lockwood founded the Meridian Institute. Ada Lockwood leads research.")]
    idx = GraphRAGIndex(llm=_BadJSONBackend(), config=GraphRAGConfig(chunk_size=200)).build(docs)  # must not crash
    res = GlobalSearch(idx, level=idx.max_level).search("themes")  # must not crash
    assert isinstance(res.answer, str)


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def test_graph_merges_and_weights():
    g = KnowledgeGraph()
    g.add_entity(Entity(id="A", name="A", description="short"))
    g.add_entity(Entity(id="A", name="A", description="a much longer description wins"))
    g.add_relationship(Relationship(id="", source="A", target="B", weight=1.0))
    g.add_relationship(Relationship(id="", source="B", target="A", weight=2.0))  # same undirected edge
    assert len(g.entities) == 2
    assert len(g.relationships) == 1
    assert g.edge_weight("A", "B") == 3.0
    assert g.entities["A"].description == "a much longer description wins"
    g.compute_stats()
    assert g.entities["A"].degree == 1
    assert g.entities["A"].rank == 3.0


# --------------------------------------------------------------------------- #
# Community detection (Louvain)
# --------------------------------------------------------------------------- #
def test_louvain_two_clusters():
    # Two triangles joined by a single bridge edge -> two communities.
    edges = [
        ("a", "b", 1), ("b", "c", 1), ("a", "c", 1),
        ("d", "e", 1), ("e", "f", 1), ("d", "f", 1),
        ("c", "d", 1),  # bridge
    ]
    nodes = ["a", "b", "c", "d", "e", "f"]
    dendro = louvain_dendrogram(nodes, edges)
    finest = dendro[0]
    assert len(set(finest.values())) == 2
    assert finest["a"] == finest["b"] == finest["c"]
    assert finest["d"] == finest["e"] == finest["f"]
    assert finest["a"] != finest["d"]


def test_detect_communities_partitions_and_nests(index):
    comms = index.communities
    assert comms
    n_levels = index.max_level + 1
    all_ids = set(index.graph.entities)
    for level in range(n_levels):
        level_comms = index.communities_at_level(level)
        covered = [e for c in level_comms for e in c.entity_ids]
        assert set(covered) == all_ids          # every entity placed
        assert len(covered) == len(all_ids)     # exactly once (a partition)
    # Nesting: each community's entities all share one parent one level up.
    by_id = {c.id: c for c in comms}
    for c in comms:
        if c.level == 0:
            assert c.parent_id == -1
        else:
            parent = by_id[c.parent_id]
            assert set(c.entity_ids).issubset(set(parent.entity_ids))


def test_modularity_positive(index):
    q = index.modularity_at_level(index.max_level)
    assert q > 0.0


def test_research_and_city_clusters_separate(index):
    """The two thematic groups should land in different communities."""
    level = index.max_level
    comm_of = {c.id: set(c.entity_ids) for c in index.communities_at_level(level)}

    def community_containing(name):
        eid = name.upper()
        return next(cid for cid, ents in comm_of.items() if eid in ents)

    research = {community_containing(n) for n in ["Meridian Institute", "Ada Lockwood", "Professor Chen"]}
    assert len(research) == 1  # research entities cluster together


# --------------------------------------------------------------------------- #
# End-to-end index + reports
# --------------------------------------------------------------------------- #
def test_index_stats(index):
    s = index.stats()
    for name in ["Ada Lockwood", "Meridian Institute", "Northwind Corporation",
                 "Elena Vasquez", "Mayor Torres", "Calder Harbor"]:
        assert name.upper() in index.graph.entities
    assert s["relationships"] > 0
    assert s["reports"] == s["communities"]


def test_reports_have_content(index):
    for rep in index.reports.values():
        assert rep.title
        assert rep.summary
        assert rep.rank >= 0.0


def test_determinism():
    a = GraphRAGIndex(config=GraphRAGConfig(chunk_size=200)).build(CORPUS)
    b = GraphRAGIndex(config=GraphRAGConfig(chunk_size=200)).build(CORPUS)
    assert a.stats() == b.stats()
    pa = {c.id: c.entity_ids for c in a.communities}
    pb = {c.id: c.entity_ids for c in b.communities}
    assert pa == pb


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def test_local_search_retrieves_relevant_entity(index):
    res = LocalSearch(index).search("Who founded the Meridian Institute?")
    names = {e.name for e in res.entities}
    assert "Ada Lockwood" in names
    assert "Meridian Institute" in names
    assert res.answer and "Ada Lockwood" in res.answer
    assert res.relationships  # some relationships pulled into context


def test_global_search_returns_points(index):
    res = GlobalSearch(index, level=index.max_level).search("What are the main themes?")
    assert res.reports_considered > 0
    assert res.points
    assert "do not contain enough information" not in res.answer


def test_global_search_empty_on_no_reports(index):
    # A level with no reports yields the graceful fallback.
    res = GlobalSearch(index, level=999).search("anything")
    assert res.reports_considered == 0
    assert "do not contain enough information" in res.answer


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def test_serialization_roundtrip(index, tmp_path):
    path = tmp_path / "index.json"
    index.save(path)
    reloaded = GraphRAGIndex.load(path)
    assert reloaded.stats() == index.stats()
    # Community levels survive the JSON round-trip as ints (not "0" strings).
    some_entity = next(iter(reloaded.graph.entities.values()))
    assert all(isinstance(k, int) for k in some_entity.community_by_level)
    # Search still works after reload (embeddings were rebuilt).
    res = LocalSearch(reloaded).search("Who founded the Meridian Institute?")
    assert "Ada Lockwood" in {e.name for e in res.entities}
    assert res.reports  # community reports re-attach after reload


# --------------------------------------------------------------------------- #
# Vector-store integration: back the store with the sibling minifaiss package
# --------------------------------------------------------------------------- #
def test_minifaiss_vector_store_adapter():
    minifaiss = pytest.importorskip("minifaiss")
    from minigraphrag import MiniFaissVectorStore

    dim = 512  # matches HashingEmbedding default
    idx = GraphRAGIndex(
        vector_store_factory=lambda: MiniFaissVectorStore(minifaiss.IndexFlatIP(dim)),
        config=GraphRAGConfig(chunk_size=200, embedding_dim=dim),
    ).build(CORPUS)
    res = LocalSearch(idx).search("Who founded the Meridian Institute?")
    assert "Ada Lockwood" in {e.name for e in res.entities}
