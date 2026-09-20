"""minifaiss demo: build several indexes and compare recall vs exact search.

Run with the project venv:

    /Users/Anil_Choudhary/code/learning/vector_stores/.venv/bin/python \
        /Users/Anil_Choudhary/code/learning/vector_stores/demo.py

The script makes synthetic clustered data, uses an exact IndexFlatL2 as ground
truth for k=10, then measures how well each approximate index recovers those
same neighbours (recall@10). Everything is seeded, so the numbers are
deterministic across runs.
"""

import numpy as np

from minifaiss import (
    METRIC_L2,
    IndexFlatL2,
    IndexIVFFlat,
    IndexPQ,
    IndexIVFPQ,
    IndexLSH,
    IndexHNSWFlat,
    index_factory,
)


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def make_data(n=2000, d=32, n_blobs=10, n_queries=100, seed=1234):
    """Return (database (n, d), queries (n_queries, d)) as float32 blobs.

    Draws ``n_blobs`` random cluster centres, then samples database and query
    vectors around them so nearest-neighbour structure actually exists.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, 10.0, size=(n_blobs, d)).astype(np.float32)

    db_labels = rng.integers(0, n_blobs, size=n)
    database = (centres[db_labels] + rng.normal(0.0, 1.0, size=(n, d))).astype(np.float32)

    q_labels = rng.integers(0, n_blobs, size=n_queries)
    queries = (centres[q_labels] + rng.normal(0.0, 1.0, size=(n_queries, d))).astype(np.float32)
    return database, queries


# --------------------------------------------------------------------------- #
# Recall metric
# --------------------------------------------------------------------------- #
def recall_at_k(approx_ids, truth_ids):
    """Mean fraction of each query's true top-k neighbours found by ``approx``.

    Both arrays are (nq, k) id matrices; -1 padding entries simply never match.
    """
    nq, k = truth_ids.shape
    hits = 0
    for i in range(nq):
        truth = set(int(t) for t in truth_ids[i] if t >= 0)
        approx = set(int(a) for a in approx_ids[i] if a >= 0)
        hits += len(truth & approx)
    return hits / (nq * k)


def bytes_per_vector(name, d, m=None, nbits=None):
    """Report the per-vector storage footprint used by each index kind."""
    if name in ("Flat", "IVFFlat", "HNSW"):
        return 4 * d                     # float32 full vectors
    if name in ("PQ", "IVFPQ"):
        return m                         # one uint8 code per subspace
    if name == "LSH":
        return nbits // 8                # packed bits
    return 4 * d


def main():
    d = 32
    k = 10
    database, queries = make_data(n=2000, d=d, n_blobs=10, n_queries=100, seed=1234)

    print(f"minifaiss demo: {database.shape[0]} db vectors, "
          f"{queries.shape[0]} queries, d={d}, k={k}\n")

    # --- Exact ground truth ------------------------------------------------ #
    flat = IndexFlatL2(d)
    flat.add(database)
    truth_D, truth_I = flat.search(queries, k)

    rows = []  # (label, bytes/vector, recall@10)

    # Flat is exact -> recall is 1.0 by definition; include it as the baseline.
    rows.append(("IndexFlatL2", bytes_per_vector("Flat", d), 1.0))

    # --- IVF flat ---------------------------------------------------------- #
    ivf = IndexIVFFlat(IndexFlatL2(d), d, nlist=64, metric_type=METRIC_L2)
    ivf.train(database)
    ivf.add(database)
    ivf.nprobe = 8
    _, ivf_I = ivf.search(queries, k)
    rows.append(("IndexIVFFlat(nlist=64,nprobe=8)",
                 bytes_per_vector("IVFFlat", d),
                 recall_at_k(ivf_I, truth_I)))

    # --- Flat PQ ----------------------------------------------------------- #
    pq = IndexPQ(d, m=8)
    pq.train(database)
    pq.add(database)
    _, pq_I = pq.search(queries, k)
    rows.append(("IndexPQ(m=8)",
                 bytes_per_vector("PQ", d, m=8),
                 recall_at_k(pq_I, truth_I)))

    # --- IVF + PQ ---------------------------------------------------------- #
    ivfpq = IndexIVFPQ(IndexFlatL2(d), d, nlist=64, m=8)
    ivfpq.train(database)
    ivfpq.add(database)
    ivfpq.nprobe = 8
    _, ivfpq_I = ivfpq.search(queries, k)
    rows.append(("IndexIVFPQ(nlist=64,m=8,nprobe=8)",
                 bytes_per_vector("IVFPQ", d, m=8),
                 recall_at_k(ivfpq_I, truth_I)))

    # --- LSH --------------------------------------------------------------- #
    lsh = IndexLSH(d, nbits=64)
    lsh.add(database)
    _, lsh_I = lsh.search(queries, k)
    rows.append(("IndexLSH(nbits=64)",
                 bytes_per_vector("LSH", d, nbits=64),
                 recall_at_k(lsh_I, truth_I)))

    # --- HNSW -------------------------------------------------------------- #
    hnsw = IndexHNSWFlat(d, M=16)
    hnsw.add(database)
    _, hnsw_I = hnsw.search(queries, k)
    rows.append(("IndexHNSWFlat(M=16)",
                 bytes_per_vector("HNSW", d),
                 recall_at_k(hnsw_I, truth_I)))

    # --- Tidy table -------------------------------------------------------- #
    name_w = max(len(r[0]) for r in rows)
    print(f"{'index'.ljust(name_w)}  {'bytes/vec':>10}  {'recall@10':>10}")
    print(f"{'-' * name_w}  {'-' * 10}  {'-' * 10}")
    for label, bpv, rec in rows:
        print(f"{label.ljust(name_w)}  {bpv:>10d}  {rec:>10.3f}")

    # --- index_factory demonstration -------------------------------------- #
    print("\nindex_factory demonstration:")
    fac = index_factory(d, "IVF64,PQ8")
    fac.train(database)
    fac.add(database)
    fac.nprobe = 8
    _, fac_I = fac.search(queries, k)
    print(f"  index_factory(d, 'IVF64,PQ8') -> {type(fac).__name__}, "
          f"recall@10 = {recall_at_k(fac_I, truth_I):.3f}")

    print("\nSee the README section \"Where performance lives\" for how each real"
          "\nFAISS optimization maps to a '# PERF:' comment in the source "
          "(grep for it).")


if __name__ == "__main__":
    main()
