"""Community detection -- carve the knowledge graph into nested clusters.

This is the step that lets GraphRAG answer *global* ("what are the themes?")
questions: entities that talk about each other a lot get grouped, each group is
summarized, and the summaries become the corpus's table of contents.

Real GraphRAG uses **hierarchical Leiden** (from ``graspologic``). We implement
plain **Louvain** from scratch, which is the same idea one generation earlier:
greedily move nodes between communities to increase *modularity*, then collapse
each community into a super-node and repeat. Each round of collapsing yields a
coarser level, so we get a hierarchy for free.

    Modularity Q = Σ_c [ L_c / m  -  (deg_c / 2m)^2 ]

where, per community c, ``L_c`` is its internal edge weight, ``deg_c`` the total
degree of its members, and ``m`` the graph's total edge weight. High Q means
many edges fall inside communities and few between them.

PERF: Leiden adds a refinement phase (Louvain can leave badly-connected
communities) and is implemented in C. graspologic also parallelizes. Our version
iterates nodes in sorted order (deterministic, no randomization) and is pure
Python -- clear, not fast.
"""

from .models import Community


# --------------------------------------------------------------------------- #
# Core Louvain
# --------------------------------------------------------------------------- #
def _one_level(adj, self_loops, m):
    """One Louvain pass: local moving until no node move improves modularity.

    ``adj`` is ``{node: {neighbor: weight}}`` (symmetric, no self entries);
    ``self_loops[node]`` is the weight of edges internal to a super-node.
    Returns ``(community_of_node, moved_any)``.
    """
    nodes = sorted(adj.keys())
    # Weighted degree; a self-loop contributes twice, per the usual convention.
    deg = {n: sum(adj[n].values()) + 2.0 * self_loops.get(n, 0.0) for n in nodes}
    comm = {n: n for n in nodes}          # start: every node in its own community
    sigma_tot = {n: deg[n] for n in nodes}  # total degree per community
    two_m = 2.0 * m

    moved_any = False
    improved = True
    while improved:
        improved = False
        for n in nodes:
            k_i = deg[n]
            current = comm[n]

            # Sum of edge weights from n into each neighboring community.
            dnc = {}
            for nb, w in adj[n].items():
                c = comm[nb]
                dnc[c] = dnc.get(c, 0.0) + w

            # Remove n from its community (only sigma_tot changes).
            sigma_tot[current] -= k_i

            # Baseline: rejoining its own (now n-free) community.
            best_com = current
            best_gain = dnc.get(current, 0.0) - sigma_tot[current] * k_i / two_m
            for c, d in dnc.items():
                gain = d - sigma_tot[c] * k_i / two_m
                if gain > best_gain + 1e-12:  # strict improvement -> guarantees termination
                    best_gain = gain
                    best_com = c

            sigma_tot[best_com] += k_i
            comm[n] = best_com
            if best_com != current:
                improved = True
                moved_any = True
    return comm, moved_any


def _aggregate(adj, self_loops, comm):
    """Collapse each community into a super-node; return the coarser graph.

    Returns ``(new_adj, new_self_loops, relabel)`` where ``relabel`` maps old
    community ids to contiguous ``0..K-1`` super-node ids.
    """
    cids = sorted(set(comm.values()))
    relabel = {c: i for i, c in enumerate(cids)}
    new_adj = {i: {} for i in range(len(cids))}
    new_self = {i: 0.0 for i in range(len(cids))}

    for node in adj:
        c = relabel[comm[node]]
        new_self[c] += self_loops.get(node, 0.0)
        for nb, w in adj[node].items():
            cn = relabel[comm[nb]]
            if cn == c:
                new_self[c] += w / 2.0   # each intra edge is seen from both ends
            else:
                new_adj[c][cn] = new_adj[c].get(cn, 0.0) + w
    return new_adj, new_self, relabel


def louvain_dendrogram(nodes, edges, seed=0):
    """Run hierarchical Louvain; return a list of partitions, finest first.

    Each partition is ``{original_node: community_label}``. ``dendrogram[0]`` is
    the finest clustering; each later entry is strictly coarser (a coarsening of
    the previous), forming a nested hierarchy.
    """
    nodes = sorted(set(nodes))
    adj = {n: {} for n in nodes}
    for u, v, w in edges:
        if u == v:
            continue
        adj[u][v] = adj[u].get(v, 0.0) + w
        adj[v][u] = adj[v].get(u, 0.0) + w
    self_loops = {n: 0.0 for n in nodes}

    dendrogram = []
    node_to_current = {n: n for n in nodes}  # original node -> current super-node label
    cur_adj, cur_self = adj, self_loops

    while True:
        m_cur = sum(sum(d.values()) for d in cur_adj.values()) / 2.0 + sum(cur_self.values())
        if m_cur == 0:
            break
        comm, moved = _one_level(cur_adj, cur_self, m_cur)
        if not moved or len(set(comm.values())) == len(cur_adj):
            break  # no further coarsening possible
        new_adj, new_self, relabel = _aggregate(cur_adj, cur_self, comm)
        node_to_current = {o: relabel[comm[node_to_current[o]]] for o in nodes}
        dendrogram.append(dict(node_to_current))
        cur_adj, cur_self = new_adj, new_self

    if not dendrogram:  # no edges, or nothing merged: trivial singleton partition
        dendrogram = [{n: i for i, n in enumerate(nodes)}]
    return dendrogram


def modularity(graph, partition):
    """Modularity Q of ``partition`` (dict entity id -> label) on ``graph``."""
    edges = graph.edge_list()
    m = sum(w for _, _, w in edges)
    if m == 0:
        return 0.0
    two_m = 2.0 * m
    deg = {n: 0.0 for n in graph.entities}
    for u, v, w in edges:
        deg[u] += w
        deg[v] += w
    tot, intra = {}, {}
    for n, c in partition.items():
        tot[c] = tot.get(c, 0.0) + deg.get(n, 0.0)
    for u, v, w in edges:
        if partition[u] == partition[v]:
            intra[partition[u]] = intra.get(partition[u], 0.0) + w
    return sum(intra.get(c, 0.0) / m - (tot[c] / two_m) ** 2 for c in tot)


# --------------------------------------------------------------------------- #
# Build Community objects (with hierarchy) from the dendrogram
# --------------------------------------------------------------------------- #
def detect_communities(graph, seed=0):
    """Cluster ``graph`` and return a flat list of :class:`Community` across levels.

    Level 0 is the coarsest partition; higher levels are finer. Also writes each
    entity's ``community_by_level`` mapping.
    """
    edges = graph.edge_list()
    dendrogram = louvain_dendrogram(list(graph.entities.keys()), edges, seed=seed)
    levels = list(reversed(dendrogram))  # levels[0] = coarsest

    communities = []
    entity_gid_by_level = []  # per level: {entity id -> global community id}
    gid = 0

    for lvl, partition in enumerate(levels):
        groups = {}
        for ent, label in partition.items():
            groups.setdefault(label, []).append(ent)

        ent_to_gid = {}
        for label in sorted(groups):
            ents = sorted(groups[label])
            ent_set = set(ents)
            rel_ids = sorted(
                r.id for (u, v), r in graph.relationships.items()
                if u in ent_set and v in ent_set
            )
            communities.append(Community(
                id=gid, level=lvl, entity_ids=ents,
                relationship_ids=rel_ids, size=len(ents),
            ))
            for e in ents:
                ent_to_gid[e] = gid
                graph.entities[e].community_by_level[lvl] = gid
            gid += 1
        entity_gid_by_level.append(ent_to_gid)

    # Wire parent (one level coarser) and child links via nesting.
    by_id = {c.id: c for c in communities}
    for c in communities:
        if c.level == 0:
            continue
        parent_gid = entity_gid_by_level[c.level - 1][c.entity_ids[0]]
        c.parent_id = parent_gid
        by_id[parent_gid].child_ids.append(c.id)

    return communities
