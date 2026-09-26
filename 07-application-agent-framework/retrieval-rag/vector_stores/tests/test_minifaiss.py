"""Pytest suite for minifaiss.

Plain ``assert`` style. Seeds are fixed and thresholds are lenient so the
approximate-index tests do not flake. Distances follow FAISS conventions:
squared L2 (ascending) and inner product (descending).
"""

import numpy as np

from minifaiss import (
    METRIC_L2,
    METRIC_INNER_PRODUCT,
    Kmeans,
    IndexFlat,
    IndexFlatL2,
    IndexFlatIP,
    IndexIVFFlat,
    ProductQuantizer,
    IndexPQ,
    IndexIVFPQ,
    IndexLSH,
    IndexHNSWFlat,
    index_factory,
)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def make_clustered(n=1000, d=16, n_blobs=8, n_queries=50, seed=0):
    """Return (database, queries) float32 arrays of clustered blob data."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, 10.0, size=(n_blobs, d)).astype(np.float32)
    db_labels = rng.integers(0, n_blobs, size=n)
    database = (centres[db_labels] + rng.normal(0, 1, (n, d))).astype(np.float32)
    q_labels = rng.integers(0, n_blobs, size=n_queries)
    queries = (centres[q_labels] + rng.normal(0, 1, (n_queries, d))).astype(np.float32)
    return database, queries


def brute_force_l2(queries, database, k):
    """Independent numpy brute-force top-k squared-L2 (ids ascending by dist)."""
    diff = queries[:, None, :] - database[None, :, :]
    d2 = np.sum(diff * diff, axis=2)              # (nq, nb) squared L2
    order = np.argsort(d2, axis=1, kind="stable")[:, :k]
    dists = np.take_along_axis(d2, order, axis=1)
    return dists, order


def recall_at_k(approx_ids, truth_ids):
    """Mean overlap fraction between approx and truth (nq, k) id matrices."""
    nq, k = truth_ids.shape
    hits = 0
    for i in range(nq):
        truth = set(int(t) for t in truth_ids[i] if t >= 0)
        approx = set(int(a) for a in approx_ids[i] if a >= 0)
        hits += len(truth & approx)
    return hits / (nq * k)


# --------------------------------------------------------------------------- #
# Flat exactness
# --------------------------------------------------------------------------- #
def test_flat_l2_matches_bruteforce_ids_and_order():
    rng = np.random.default_rng(42)
    d, n, nq, k = 8, 300, 20, 5
    db = rng.normal(size=(n, d)).astype(np.float32)
    q = rng.normal(size=(nq, d)).astype(np.float32)

    idx = IndexFlatL2(d)
    idx.add(db)
    D, I = idx.search(q, k)

    bf_D, bf_I = brute_force_l2(q, db, k)

    # Ids and their order must match the independent brute force exactly.
    assert np.array_equal(I, bf_I)
    assert np.allclose(D, bf_D, rtol=1e-4, atol=1e-3)


def test_flat_ip_ranks_by_inner_product_descending():
    rng = np.random.default_rng(7)
    d, n, nq, k = 8, 200, 10, 5
    db = rng.normal(size=(n, d)).astype(np.float32)
    q = rng.normal(size=(nq, d)).astype(np.float32)

    idx = IndexFlatIP(d)
    idx.add(db)
    D, I = idx.search(q, k)

    # Scores must be non-increasing along each row (descending ranking).
    assert np.all(D[:, :-1] >= D[:, 1:] - 1e-4)

    # Cross-check the top-1 against a direct dot-product argmax.
    ip = q @ db.T
    assert np.array_equal(I[:, 0], np.argmax(ip, axis=1))


def test_flat_base_class_and_metric_default():
    idx = IndexFlat(4)
    assert idx.metric_type == METRIC_L2
    assert idx.is_trained is True


# --------------------------------------------------------------------------- #
# Output shape / padding contract
# --------------------------------------------------------------------------- #
def test_search_shapes_and_padding_when_k_gt_ntotal():
    d, k = 5, 10
    idx = IndexFlatL2(d)
    rng = np.random.default_rng(1)
    db = rng.normal(size=(3, d)).astype(np.float32)   # only 3 vectors
    idx.add(db)
    q = rng.normal(size=(4, d)).astype(np.float32)
    D, I = idx.search(q, k)

    assert D.shape == (4, k)
    assert I.shape == (4, k)
    assert D.dtype == np.float32
    assert I.dtype == np.int64

    # First 3 ids valid, remaining padded with -1 and +inf (L2).
    for row in I:
        assert set(int(v) for v in row if v >= 0) <= {0, 1, 2}
        assert np.count_nonzero(row == -1) == k - 3
    for row in D:
        assert np.isinf(row[3:]).all()


def test_ids_are_valid_or_padding():
    d, k = 6, 3
    idx = IndexFlatL2(d)
    rng = np.random.default_rng(2)
    db = rng.normal(size=(50, d)).astype(np.float32)
    idx.add(db)
    D, I = idx.search(rng.normal(size=(5, d)).astype(np.float32), k)
    assert ((I >= 0) & (I < 50) | (I == -1)).all()


# --------------------------------------------------------------------------- #
# Kmeans
# --------------------------------------------------------------------------- #
def test_kmeans_centroid_count_and_inertia_monotone():
    db, _ = make_clustered(n=500, d=8, n_blobs=6, seed=3)

    km_few = Kmeans(8, 6, niter=1, seed=1234)
    inertia_few = km_few.train(db)
    assert km_few.centroids.shape == (6, 8)

    km_many = Kmeans(8, 6, niter=25, seed=1234)
    inertia_many = km_many.train(db)
    assert km_many.centroids.shape == (6, 8)

    # More iterations should not increase inertia (lenient: allow tiny slack).
    assert inertia_many <= inertia_few + 1e-3


def test_kmeans_assign_shapes():
    db, _ = make_clustered(n=200, d=8, n_blobs=5, seed=4)
    km = Kmeans(8, 5, seed=1234)
    km.train(db)
    codes, dists = km.assign(db)
    assert codes.shape == (200,)
    assert dists.shape == (200,)
    assert codes.dtype == np.int64
    assert (codes >= 0).all() and (codes < 5).all()


# --------------------------------------------------------------------------- #
# Product quantizer
# --------------------------------------------------------------------------- #
def test_pq_decode_bounded_error():
    db, _ = make_clustered(n=2000, d=16, n_blobs=10, seed=5)
    pq = ProductQuantizer(16, m=8, nbits=8, seed=1234)
    pq.train(db)
    codes = pq.compute_codes(db)
    recon = pq.decode(codes)

    assert codes.shape == (2000, 8)
    assert codes.dtype == np.uint8
    # Lenient: mean abs reconstruction error should be small on clustered data.
    mae = float(np.mean(np.abs(recon - db)))
    assert mae < 1.0


# --------------------------------------------------------------------------- #
# IVF flat
# --------------------------------------------------------------------------- #
def test_ivfflat_full_probe_matches_flat_top1():
    db, q = make_clustered(n=800, d=16, n_blobs=8, n_queries=40, seed=6)

    flat = IndexFlatL2(16)
    flat.add(db)
    _, flat_I = flat.search(q, 1)

    ivf = IndexIVFFlat(IndexFlatL2(16), 16, nlist=32)
    ivf.train(db)
    ivf.add(db)
    ivf.nprobe = ivf.nlist            # exhaustive -> exact
    _, ivf_I = ivf.search(q, 1)

    assert np.array_equal(ivf_I[:, 0], flat_I[:, 0])


def test_ivfflat_recall_on_easy_data():
    db, q = make_clustered(n=1500, d=16, n_blobs=10, n_queries=60, seed=7)
    k = 10

    flat = IndexFlatL2(16)
    flat.add(db)
    _, truth = flat.search(q, k)

    ivf = IndexIVFFlat(IndexFlatL2(16), 16, nlist=32)
    ivf.train(db)
    ivf.add(db)
    ivf.nprobe = 8
    _, ivf_I = ivf.search(q, k)

    assert recall_at_k(ivf_I, truth) >= 0.7


def test_ivfflat_reconstruct_exact():
    db, _ = make_clustered(n=300, d=8, n_blobs=5, seed=8)
    ivf = IndexIVFFlat(None, 8, nlist=16)
    ivf.train(db)
    ivf.add(db)
    # Exact flat storage -> reconstruct returns the original vector.
    assert np.allclose(ivf.reconstruct(10), db[10], atol=1e-5)


# --------------------------------------------------------------------------- #
# HNSW
# --------------------------------------------------------------------------- #
def test_hnsw_recall_on_easy_data():
    db, q = make_clustered(n=1500, d=16, n_blobs=10, n_queries=60, seed=9)
    k = 10

    flat = IndexFlatL2(16)
    flat.add(db)
    _, truth = flat.search(q, k)

    hnsw = IndexHNSWFlat(16, M=16)
    hnsw.efSearch = 32
    hnsw.add(db)
    _, hnsw_I = hnsw.search(q, k)

    assert recall_at_k(hnsw_I, truth) >= 0.7


def test_hnsw_layer0_budget_is_2M():
    """Layer 0 keeps up to M0 = 2M neighbours (paper's M_max0, faiss's nb_neighbors(0));
    upper layers keep up to M."""
    db, _ = make_clustered(n=600, d=8, n_blobs=6, seed=11)
    hnsw = IndexHNSWFlat(8, M=4)
    hnsw.add(db)
    assert hnsw.M0 == 8
    layer0 = [len(nb[0]) for nb in hnsw._neighbors]
    upper = [len(nb[l]) for nb in hnsw._neighbors for l in range(1, len(nb))]
    assert max(layer0) <= hnsw.M0 and max(layer0) > hnsw.M      # the budget is used
    assert not upper or max(upper) <= hnsw.M


def test_hnsw_recall_on_random_data_default_efsearch():
    """Regression: with layer 0 capped at M (not 2M) this scored recall@10 = 0.834;
    with M0 = 2M it scores 0.903, matching faiss.IndexHNSWFlat(16, 16) with
    efConstruction=40, efSearch=16 on the same seeded data (0.905, measured
    2026-09-26). ~10 s: 5,000 inserts in pure Python."""
    d, k = 16, 10
    rng = np.random.default_rng(0)
    db = rng.standard_normal((5000, d)).astype(np.float32)
    q = rng.standard_normal((100, d)).astype(np.float32)
    flat = IndexFlatL2(d)
    flat.add(db)
    _, truth = flat.search(q, k)

    hnsw = IndexHNSWFlat(d, M=16)
    assert hnsw.efSearch == 16 and hnsw.efConstruction == 40   # the defaults, as in faiss
    hnsw.add(db)
    _, hnsw_I = hnsw.search(q, k)
    assert recall_at_k(hnsw_I, truth) >= 0.87


def test_hnsw_reconstruct_exact():
    db, _ = make_clustered(n=200, d=8, n_blobs=5, seed=10)
    hnsw = IndexHNSWFlat(8, M=8)
    hnsw.add(db)
    assert np.allclose(hnsw.reconstruct(7), db[7], atol=1e-5)


# --------------------------------------------------------------------------- #
# PQ / IVFPQ approximate validity
# --------------------------------------------------------------------------- #
def test_indexpq_valid_ids_and_positive_recall():
    db, q = make_clustered(n=1500, d=16, n_blobs=10, n_queries=50, seed=11)

    flat = IndexFlatL2(16)
    flat.add(db)
    _, truth1 = flat.search(q, 1)

    pq = IndexPQ(16, m=8)
    pq.train(db)
    pq.add(db)
    D, I = pq.search(q, 5)

    assert I.shape == (50, 5)
    assert ((I >= 0) & (I < 1500) | (I == -1)).all()
    # recall@1: at least some queries recover the true nearest neighbour.
    hit = np.mean([truth1[i, 0] in set(I[i]) for i in range(50)])
    assert hit > 0.0


def test_indexivfpq_valid_ids_and_positive_recall():
    db, q = make_clustered(n=1500, d=16, n_blobs=10, n_queries=50, seed=12)

    flat = IndexFlatL2(16)
    flat.add(db)
    _, truth1 = flat.search(q, 1)

    ivfpq = IndexIVFPQ(IndexFlatL2(16), 16, nlist=32, m=8)
    ivfpq.train(db)
    ivfpq.add(db)
    ivfpq.nprobe = 8
    D, I = ivfpq.search(q, 5)

    assert I.shape == (50, 5)
    assert ((I >= 0) & (I < 1500) | (I == -1)).all()
    hit = np.mean([truth1[i, 0] in set(I[i]) for i in range(50)])
    assert hit > 0.0


# --------------------------------------------------------------------------- #
# LSH
# --------------------------------------------------------------------------- #
def test_lsh_self_query_zero_distance():
    db, _ = make_clustered(n=200, d=16, n_blobs=5, seed=13)
    lsh = IndexLSH(16, nbits=64)
    lsh.add(db)
    D, I = lsh.search(db[:5], 1)
    # A vector's own code has Hamming distance 0 to itself; it should rank top.
    assert np.all(D[:, 0] == 0.0)


# --------------------------------------------------------------------------- #
# index_factory
# --------------------------------------------------------------------------- #
def test_index_factory_returns_expected_classes():
    d = 32
    assert isinstance(index_factory(d, "Flat"), IndexFlatL2)
    assert isinstance(index_factory(d, "Flat", METRIC_INNER_PRODUCT), IndexFlatIP)
    assert isinstance(index_factory(d, "IVF64,Flat"), IndexIVFFlat)
    assert isinstance(index_factory(d, "IVF64,PQ8"), IndexIVFPQ)
    assert isinstance(index_factory(d, "PQ8"), IndexPQ)
    assert isinstance(index_factory(d, "LSH32"), IndexLSH)
    assert isinstance(index_factory(d, "HNSW16"), IndexHNSWFlat)

    # Parsed parameters land where expected.
    assert index_factory(d, "IVF64,Flat").nlist == 64
    assert index_factory(d, "PQ8").m == 8
    assert index_factory(d, "LSH32").nbits == 32
    assert index_factory(d, "HNSW16").M == 16

    # Whitespace tolerance.
    assert isinstance(index_factory(d, " IVF64 , PQ8 "), IndexIVFPQ)


def test_index_factory_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        index_factory(32, "Bogus")
    with pytest.raises(ValueError):
        index_factory(32, "IVF64,Nope")
    with pytest.raises(ValueError):
        index_factory(32, "")
