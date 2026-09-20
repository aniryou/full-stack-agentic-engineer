"""End-to-end minigraphrag demo -- fully offline, deterministic.

Builds a knowledge graph from a tiny hand-written corpus, clusters it into
communities, summarizes them, and then answers questions with both local and
global search -- using the default ``MockLLM`` (no API key, no network).

    ./.venv/bin/python demo_graphrag.py

Everything printed here is produced by the pipeline, not hard-coded. To see how
much a real model helps, swap ``MockLLM`` for ``AnthropicLLM`` (see the very
bottom of this file).
"""

from minigraphrag import (
    Document, GraphRAGIndex, GraphRAGConfig, LocalSearch, GlobalSearch,
)

# --------------------------------------------------------------------------- #
# A small fictional corpus. Proper nouns are consistent so the (crude) offline
# extractor can find them, and sentences deliberately co-mention related
# entities so that relationships (and thus communities) emerge.
# --------------------------------------------------------------------------- #
CORPUS = [
    Document(
        id="d1", title="The Meridian Institute",
        text=(
            "The Meridian Institute is a research laboratory in the City of Calder. "
            "Ada Lockwood founded the Meridian Institute after leaving Northwind Corporation. "
            "Professor Chen leads Project Halcyon at the Meridian Institute. "
            "Project Halcyon studies tidal energy along the Calder Harbor. "
            "Ada Lockwood and Professor Chen have collaborated at the Meridian Institute for years. "
            "The Meridian Institute receives funding from Northwind Corporation."
        ),
    ),
    Document(
        id="d2", title="Northwind Corporation",
        text=(
            "Northwind Corporation is an energy company based in the Riverside District. "
            "Elena Vasquez is the chief executive of Northwind Corporation. "
            "Elena Vasquez studied under Ada Lockwood before joining Northwind Corporation. "
            "Northwind Corporation operates the Calder Harbor power station. "
            "Elena Vasquez and Mayor Torres signed an agreement about the Calder Harbor."
        ),
    ),
    Document(
        id="d3", title="The City of Calder",
        text=(
            "Mayor Torres governs the City of Calder. "
            "The Riverside District is the oldest neighborhood in the City of Calder. "
            "Mayor Torres proposed a plan to redevelop the Riverside District. "
            "The Harbor Authority manages the Calder Harbor. "
            "Mayor Torres appointed Elena Vasquez to the Harbor Authority. "
            "The City of Calder depends on tidal energy from the Calder Harbor."
        ),
    ),
    Document(
        id="d4", title="Project Halcyon",
        text=(
            "Project Halcyon is an ambitious research program at the Meridian Institute. "
            "Professor Chen and Ada Lockwood presented Project Halcyon at the Calder Energy Summit. "
            "The Calder Energy Summit gathered leaders from Northwind Corporation and the Harbor Authority. "
            "Project Halcyon aims to power the City of Calder with tidal energy. "
            "The Meridian Institute published the results of Project Halcyon."
        ),
    ),
]


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    rule("INDEXING")
    # Small chunk size so each short document stays a single text unit.
    index = GraphRAGIndex(config=GraphRAGConfig(chunk_size=200, chunk_overlap=20, verbose=True))
    index.build(CORPUS)
    print("\n", index, sep="")
    print("stats:", index.stats())

    # ---- the knowledge graph ---- #
    rule("KNOWLEDGE GRAPH -- entities (by rank = weighted degree)")
    entities = sorted(index.graph.entities.values(), key=lambda e: -e.rank)
    for e in entities:
        print(f"  {e.name:<24} {e.type:<13} degree={e.degree} rank={e.rank:g}")

    rule("KNOWLEDGE GRAPH -- relationships (by weight)")
    rels = sorted(index.graph.relationships.values(), key=lambda r: -r.weight)
    for r in rels[:12]:
        s = index.graph.entities[r.source].name
        t = index.graph.entities[r.target].name
        print(f"  {s} — {t}  (w={r.weight:g})")
    if len(rels) > 12:
        print(f"  ... and {len(rels) - 12} more")

    # ---- communities ---- #
    rule("COMMUNITIES (hierarchical Louvain; level 0 = coarsest)")
    for level in range(index.max_level + 1):
        q = index.modularity_at_level(level)
        comms = index.communities_at_level(level)
        print(f"\n  Level {level}: {len(comms)} communities, modularity Q={q:.3f}")
        for c in comms:
            names = [index.graph.entities[e].name for e in c.entity_ids]
            print(f"    community {c.id}: {', '.join(names)}")

    # ---- community reports ---- #
    rule("COMMUNITY REPORTS (finest level)")
    for rep in sorted(index.reports_at_level(index.max_level), key=lambda r: -r.rank):
        print(f"\n  [{rep.rank:g}] {rep.title}")
        print(f"      {rep.summary}")
        for f in rep.findings:
            print(f"        - {f['summary']}")

    # ---- LOCAL search ---- #
    rule("LOCAL SEARCH  (entity-centric questions)")
    ls = LocalSearch(index)
    for q in [
        "Who is Ada Lockwood and what did she found?",
        "How is Elena Vasquez connected to the Calder Harbor?",
    ]:
        res = ls.search(q)
        print(f"\nQ: {q}")
        print(f"   retrieved entities: {[e.name for e in res.entities[:6]]}")
        print("   answer:")
        for line in res.answer.splitlines():
            print(f"     {line}")

    # ---- GLOBAL search ---- #
    rule("GLOBAL SEARCH  (whole-corpus questions, map-reduce over reports)")
    gs = GlobalSearch(index, level=index.max_level)
    for q in [
        "What are the main themes across these documents?",
        "What role does tidal energy play in this corpus?",
    ]:
        res = gs.search(q)
        print(f"\nQ: {q}")
        print(f"   ({res.reports_considered} reports mapped, {len(res.points)} points kept)")
        print("   answer:")
        for line in res.answer.splitlines():
            print(f"     {line}")

    # ---- integration notes ---- #
    rule("SWAPPING IN REAL COMPONENTS")
    print("""\
  # Real LLM (needs `pip install anthropic` + ANTHROPIC_API_KEY):
      from minigraphrag import AnthropicLLM
      index = GraphRAGIndex(llm=AnthropicLLM(model="claude-opus-5")).build(CORPUS)

  # Real ANN vector store (reuse the sibling minifaiss package):
      from minifaiss import IndexFlatIP
      from minigraphrag import MiniFaissVectorStore
      index = GraphRAGIndex(
          vector_store_factory=lambda: MiniFaissVectorStore(IndexFlatIP(512)),
      ).build(CORPUS)

  grep -rn "# LLM:"  minigraphrag/   # every place a model is called
  grep -rn "# PERF:" minigraphrag/   # every place real GraphRAG scales""")


if __name__ == "__main__":
    main()
