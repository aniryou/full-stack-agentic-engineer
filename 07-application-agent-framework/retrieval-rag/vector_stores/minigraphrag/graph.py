"""KnowledgeGraph -- a tiny undirected, weighted graph of entities.

We roll our own instead of importing ``networkx`` so the data structures stay
visible and the package has no graph dependency. The API is deliberately small:
just what community detection and search need.

Relationships are treated as **undirected** for graph analysis (a tie between
A and B is the same tie as B and A), and parallel edges between the same pair
are merged into one weighted edge -- this is what GraphRAG feeds to clustering.

PERF: at real scale you would use ``networkx`` / ``igraph`` / ``graph-tool``
(C-backed adjacency, CSR sparse matrices, parallel algorithms) rather than
Python dicts of sets.
"""

from .models import Entity, Relationship


def edge_key(a, b):
    """Canonical, order-independent key for the undirected edge {a, b}."""
    return (a, b) if a <= b else (b, a)


class KnowledgeGraph:
    """Undirected weighted multigraph collapsed to simple weighted edges.

    Attributes
    ----------
    entities : dict[str, Entity]
        Node id -> entity.
    relationships : dict[tuple, Relationship]
        Canonical edge key -> merged relationship.
    _adj : dict[str, dict[str, float]]
        Adjacency: node -> {neighbor: edge weight}.
    """

    def __init__(self):
        self.entities = {}
        self.relationships = {}
        self._adj = {}

    # ---- construction ---------------------------------------------------- #
    def add_entity(self, entity):
        """Insert an entity, or merge into the existing node with the same id."""
        existing = self.entities.get(entity.id)
        if existing is None:
            self.entities[entity.id] = entity
            self._adj.setdefault(entity.id, {})
            return existing  # None -> newly created
        # Merge: keep the longer description, union the source text units.
        if len(entity.description) > len(existing.description):
            existing.description = entity.description
        if existing.type in ("", "UNKNOWN") and entity.type not in ("", "UNKNOWN"):
            existing.type = entity.type
        existing.text_unit_ids = sorted(set(existing.text_unit_ids) | set(entity.text_unit_ids))
        return existing

    def add_relationship(self, rel):
        """Insert an edge, or merge weights/descriptions into an existing edge."""
        # Both endpoints must be nodes; create bare ones if missing.
        for eid in (rel.source, rel.target):
            if eid not in self.entities:
                self.entities[eid] = Entity(id=eid, name=eid)
                self._adj.setdefault(eid, {})
        if rel.source == rel.target:
            return  # ignore self-loops as relationships
        key = edge_key(rel.source, rel.target)
        existing = self.relationships.get(key)
        if existing is None:
            rel.id = f"{key[0]}__{key[1]}"
            rel.source, rel.target = key
            self.relationships[key] = rel
        else:
            existing.weight += rel.weight
            if rel.description and rel.description not in existing.description:
                sep = " " if existing.description else ""
                existing.description = f"{existing.description}{sep}{rel.description}"
            existing.text_unit_ids = sorted(set(existing.text_unit_ids) | set(rel.text_unit_ids))
        w = self.relationships[key].weight
        self._adj[key[0]][key[1]] = w
        self._adj[key[1]][key[0]] = w

    # ---- queries --------------------------------------------------------- #
    def neighbors(self, node_id):
        """Neighbor ids of ``node_id`` (empty if unknown/isolated)."""
        return list(self._adj.get(node_id, {}).keys())

    def edge_weight(self, a, b):
        return self._adj.get(a, {}).get(b, 0.0)

    def degree(self, node_id):
        """Weighted degree: sum of incident edge weights."""
        return sum(self._adj.get(node_id, {}).values())

    def relationships_of(self, node_id):
        """All relationships incident to ``node_id``."""
        out = []
        for nb in self.neighbors(node_id):
            out.append(self.relationships[edge_key(node_id, nb)])
        return out

    def compute_stats(self):
        """Populate each entity's ``degree`` (edge count) and ``rank`` (weighted)."""
        for eid, ent in self.entities.items():
            ent.degree = len(self._adj.get(eid, {}))
            ent.rank = self.degree(eid)

    def edge_list(self):
        """Return ``[(u, v, weight), ...]`` once per undirected edge."""
        return [(u, v, r.weight) for (u, v), r in self.relationships.items()]

    def __repr__(self):
        return f"KnowledgeGraph(entities={len(self.entities)}, relationships={len(self.relationships)})"
