"""Collectives: exact results, step counts and alpha-beta costs, busbw as nccl-tests defines it."""
import numpy as np
import pytest

from gpusim import collectives as C

AR = ["ring", "tree", "one_shot", "two_shot", "switch"]


def _bufs(p, n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(-1000, 1000, n) for _ in range(p)]      # integers: every summation order is exact


@pytest.mark.parametrize("p", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("algo", AR)
def test_all_reduce_equals_numpy_on_every_rank(p, algo):
    bufs = _bufs(p, 6 * p + 5)                                  # size not divisible by p
    out = C.all_reduce(bufs, algo, chunks=3).out
    for o in out:
        assert np.array_equal(o, np.sum(bufs, axis=0))


@pytest.mark.parametrize("p", [2, 3, 5, 8])
def test_other_collectives_equal_numpy(p):
    bufs = _bufs(p, 4 * p, seed=p)
    for algo in ("ring", "direct"):
        for got, want in zip(C.reduce_scatter(bufs, algo).out, C.reference("reduce_scatter", bufs)):
            assert np.array_equal(got, want)
        for got, want in zip(C.all_gather(bufs, algo).out, C.reference("all_gather", bufs)):
            assert np.array_equal(got, want)
    for algo, k in (("ring", 1), ("ring", 4), ("tree", 1)):
        assert all(np.array_equal(o, bufs[p - 1]) for o in C.broadcast(bufs, p - 1, algo, k).out)
    for algo in ("pairwise", "direct"):
        for got, want in zip(C.all_to_all(bufs, algo).out, C.reference("all_to_all", bufs)):
            assert np.array_equal(got, want)


def test_ring_all_reduce_is_reduce_scatter_then_all_gather():
    p, n = 4, 12
    tr = C.all_reduce(_bufs(p, n), "ring").trace
    assert tr.n_steps == 2 * (p - 1)
    assert tr.phases == ["reduce-scatter"] * 3 + ["all-gather"] * 3
    assert all(s.hi - s.lo == n // p for step in tr.steps for s in step)            # every message is S/p
    assert set(tr.sent_bytes().values()) == {2 * (p - 1) * (n // p) * 8}            # 2(p-1)/p * S per rank


def test_ring_time_hand_computed():
    # p=4, S=96 B (12 int64), alpha=1 us, B=1 GB/s: 6 steps x (1 us + 24 B / 1e9) = 6.144 us
    tr = C.all_reduce(_bufs(4, 12), "ring").trace
    assert tr.time(1e-6, 1e9) == pytest.approx(6.144e-6)


@pytest.mark.parametrize("p", [2, 3, 4, 8, 16])
@pytest.mark.parametrize("k", [1, 4])
def test_traced_steps_and_time_equal_the_closed_forms(p, k):
    n, one = p * k * 4, np.ones(p * k * 4, np.float32)
    runs = [("all_reduce", a, C.all_reduce([one] * p, a, k)) for a in AR]
    runs += [("reduce_scatter", a, C.reduce_scatter([one] * p, a)) for a in ("ring", "direct")]
    runs += [("all_gather", a, C.all_gather([one[: n // p]] * p, a)) for a in ("ring", "direct")]
    runs += [("broadcast", a, C.broadcast([one] * p, 0, a, k)) for a in ("ring", "tree")]
    runs += [("all_to_all", a, C.all_to_all([one] * p, a)) for a in ("pairwise", "direct")]
    for op, algo, res in runs:
        steps, _ = C.cost_terms(op, algo, p, k)
        assert res.trace.n_steps == steps, (op, algo)
        assert res.trace.time(3e-6, 7e9) == pytest.approx(C.model_time(op, algo, res.trace.size, p, 3e-6, 7e9, k))


def test_in_switch_reduction_halves_the_bytes_each_gpu_sends():
    p, n = 8, 64
    ring = C.all_reduce(_bufs(p, n), "ring").trace.sent_bytes()
    switch = C.all_reduce(_bufs(p, n), "switch", chunks=4).trace.sent_bytes()
    assert set(ring.values()) == {2 * (p - 1) * n * 8 // p} and set(switch.values()) == {n * 8}


def test_busbw_factors_match_nccl_tests():
    s, t, p = 8e9, 1.0, 8
    assert C.busbw("all_reduce", s, t, p) == pytest.approx(8e9 * 2 * 7 / 8)
    for op in ("reduce_scatter", "all_gather", "all_to_all"):
        assert C.busbw(op, s, t, p) == pytest.approx(8e9 * 7 / 8)
    for op in ("broadcast", "reduce", "send_recv"):
        assert C.busbw(op, s, t, p) == 8e9


@pytest.mark.parametrize("op,algo", [("all_reduce", "ring"), ("reduce_scatter", "ring"),
                                     ("all_gather", "ring"), ("all_to_all", "pairwise")])
def test_busbw_of_a_bandwidth_optimal_algorithm_is_the_link_bandwidth(op, algo):
    bw, size = 400e9, 2 ** 30
    for p in (2, 4, 8, 64):                                # bandwidth term only: busbw == B for any p
        assert C.busbw(op, size, C.model_time(op, algo, size, p, 0.0, bw), p) == pytest.approx(bw)
    # with a latency term busbw falls short of B, more so as p grows (the ring's alpha term grows with p)
    b = [C.busbw(op, size, C.model_time(op, algo, size, p, 5e-6, bw), p) for p in (2, 8, 64)]
    assert bw > b[0] > b[1] > b[2]


def test_latency_bandwidth_crossover():
    assert C.crossover_bytes("all_reduce", "ring", 8, 1e-6, 400e9) == pytest.approx(3.2e6)


def test_tp_decode_all_reduce_is_latency_bound_and_prefill_is_not():
    decode = C.tp_comm(layers=80, tokens=32, hidden=8192, p=8, alpha=1e-6, bw=400e9)
    prefill = C.tp_comm(layers=80, tokens=8192, hidden=8192, p=8, alpha=1e-6, bw=400e9)
    assert decode["message_bytes"] == 512 * 1024 and decode["calls"] == 160
    assert decode["latency_share"] > 0.8 and prefill["latency_share"] < 0.05
    assert decode["per_call_s"] == pytest.approx(14e-6 + 1.75 * 524288 / 400e9)


def test_small_messages_prefer_few_steps_large_messages_prefer_few_bytes():
    assert C.best_algorithm("all_reduce", 8, 8, 1e-6, 400e9)[0][1] == "one_shot"
    assert C.best_algorithm("all_reduce", 2 ** 30, 8, 1e-6, 400e9)[0][1] == "switch"


def test_first_mismatched_call_is_where_the_job_hangs():
    ok = [("all_reduce", 4096)] * 3
    calls = {0: ok, 1: ok, 2: [("all_reduce", 4096), ("all_gather", 4096), ("all_reduce", 4096)], 3: ok}
    m = C.first_mismatch(calls)
    assert m["call"] == 1 and m["odd_ranks"] == [2] and m["expected"] == ("all_reduce", 4096)
    assert C.first_mismatch({0: ok, 1: ok[:2]})["odd_calls"] == {1: None}          # rank 1 never arrived
    assert C.first_mismatch({0: ok, 1: ok}) is None
