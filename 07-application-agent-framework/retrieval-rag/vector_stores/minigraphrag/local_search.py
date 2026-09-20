"""Local search -- answer entity-centric questions from a focused subgraph.

Good for "who is X?", "how are X and Y related?", "what happened at Z?". The
recipe (a trimmed version of GraphRAG's local search):

    1. embed the query and find the most similar ENTITIES (vector search)   # PERF: vector DB
    2. expand to their 1-hop neighbours and the relationships among them
    3. pull the source text units that mention them, and the community
       reports of the communities they belong to
    4. assemble all of that into one context and let the LLM answer          # LLM

The intuition: instead of retrieving loose text passages (classic RAG), we
retrieve a connected slice of the knowledge graph, so the model sees how the
relevant entities relate, not just that they were mentioned nearby.
"""

from dataclasses import dataclass, field


@dataclass
class LocalSearchResult:
    answer: str = ""
    entities: list = field(default_factory=list)       # Entity objects, best-first
    relationships: list = field(default_factory=list)  # Relationship objects
    reports: list = field(default_factory=list)         # CommunityReport objects
    text_units: list = field(default_factory=list)      # TextUnit objects
    context: str = ""                                    # the assembled prompt context


class LocalSearch:
    """Entity-centric retrieval + answer over a :class:`GraphRAGIndex`."""

    def __init__(self, index, top_k_entities=8, max_entities=15, max_relationships=20,
                 max_text_units=5, max_reports=3, level=None):
        self.index = index
        self.top_k_entities = top_k_entities
        self.max_entities = max_entities
        self.max_relationships = max_relationships
        self.max_text_units = max_text_units
        self.max_reports = max_reports
        # Reports come from this hierarchy level (default: finest = most specific).
        self.level = level if level is not None else index.max_level

    def search(self, query):
        idx = self.index
        graph = idx.graph
        if idx.entity_store is None or len(idx.entity_store) == 0:
            return LocalSearchResult(answer="The index contains no entities.")

        # 1. Semantic search for seed entities.
        qv = idx.embedder.embed_one(query)
        hits = idx.entity_store.query(qv, k=self.top_k_entities)  # PERF: ANN index here
        seed_ids = [eid for eid, _ in hits]
        seed_rank = {eid: i for i, eid in enumerate(seed_ids)}

        # 2. Expand to 1-hop neighbours.
        context_ids = list(seed_ids)
        for eid in seed_ids:
            for nb in graph.neighbors(eid):
                if nb not in context_ids:
                    context_ids.append(nb)

        # Order: seeds first (by retrieval rank), then neighbours by graph rank.
        def sort_key(eid):
            return (seed_rank.get(eid, len(seed_ids)), -graph.entities[eid].rank)

        context_ids = sorted(set(context_ids), key=sort_key)[:self.max_entities]
        context_set = set(context_ids)
        entities = [graph.entities[eid] for eid in context_ids]

        # 3a. Relationships fully inside the context set, strongest first.
        rels = []
        seen = set()
        for eid in context_ids:
            for r in graph.relationships_of(eid):
                if r.source in context_set and r.target in context_set and r.id not in seen:
                    seen.add(r.id)
                    rels.append(r)
        rels.sort(key=lambda r: -r.weight)
        rels = rels[:self.max_relationships]

        # 3b. Source text units, ranked by how many context entities they mention.
        unit_scores = {}
        for eid in context_ids:
            for uid in graph.entities[eid].text_unit_ids:
                unit_scores[uid] = unit_scores.get(uid, 0) + 1
        ranked_units = sorted(unit_scores, key=lambda u: -unit_scores[u])[:self.max_text_units]
        text_units = [idx.text_units[u] for u in ranked_units if u in idx.text_units]

        # 3c. Community reports for the seeds' communities at the chosen level.
        report_ids, reports = [], []
        for eid in seed_ids:
            cid = graph.entities[eid].community_by_level.get(self.level)
            if cid is not None and cid not in report_ids and cid in idx.reports:
                report_ids.append(cid)
                reports.append(idx.reports[cid])
        reports.sort(key=lambda r: -r.rank)
        reports = reports[:self.max_reports]

        # 4. Assemble context + answer.
        context_text, structured = self._render(entities, rels, reports, text_units)
        answer = idx.llm.local_answer(query, context_text, structured)  # LLM
        return LocalSearchResult(
            answer=answer, entities=entities, relationships=rels,
            reports=reports, text_units=text_units, context=context_text,
        )

    def _render(self, entities, rels, reports, text_units):
        """Build the context string + the structured hints for the mock LLM."""
        name = self.index.graph.entities
        lines = ["## Entities"]
        ent_struct = []
        for e in entities:
            lines.append(f"- {e.name} ({e.type}) [rank {e.rank:g}]: {e.description}")
            ent_struct.append({"name": e.name, "type": e.type, "description": e.description, "rank": e.rank})

        lines.append("\n## Relationships")
        rel_struct = []
        for r in rels:
            s, t = name[r.source].name, name[r.target].name
            lines.append(f"- {s} — {t} (weight {r.weight:g}): {r.description}")
            rel_struct.append({"source": s, "target": t, "description": r.description})

        lines.append("\n## Community reports")
        rep_struct = []
        for rep in reports:
            lines.append(f"- {rep.title}: {rep.summary}")
            rep_struct.append({"title": rep.title, "summary": rep.summary})

        lines.append("\n## Sources (text units)")
        src_struct = []
        for u in text_units:
            snippet = " ".join(u.text.split())[:300]
            lines.append(f"- ({u.id}) {snippet}")
            src_struct.append(u.text)

        structured = {
            "entities": ent_struct, "relationships": rel_struct,
            "reports": rep_struct, "sources": src_struct,
        }
        return "\n".join(lines), structured
