"""Collective semantics, the shared sweep loop, and a real ring over OS pipes."""
import numpy as np
import pytest

from gpurt import nccltests
from gpurt.dist import pipes, semantics
from gpurt.dist.sweep import format_table, rank_input, sweep

rng = np.random.default_rng(0)


def test_all_reduce_is_reduce_scatter_then_all_gather():
    xs = [rng.integers(0, 9, 12).astype(np.float32) for _ in range(4)]
    rs = semantics.reduce_scatter(xs)
    assert all(np.array_equal(a, b) for a, b in zip(semantics.all_gather(rs), semantics.all_reduce(xs)))


def test_alltoall_is_a_distributed_transpose():
    xs = [np.arange(r * 6, r * 6 + 6, dtype=np.float32) for r in range(3)]
    once = semantics.alltoall(xs)
    assert np.array_equal(once[1], np.array([2, 3, 8, 9, 14, 15], np.float32))  # chunk 1 of every rank
    assert all(np.array_equal(a, b) for a, b in zip(semantics.alltoall(once), xs))


def test_rooted_and_point_to_point_semantics():
    xs = [np.full(3, r, np.float32) for r in range(3)]
    assert all(np.array_equal(b, xs[2]) for b in semantics.broadcast(xs, root=2))
    red = semantics.reduce(xs, root=1)
    assert np.array_equal(red[1], np.full(3, 3.0)) and np.array_equal(red[0], xs[0])
    assert np.array_equal(semantics.sendrecv(xs)[0], xs[2])


class LoopbackComm:
    """A one-rank Comm: exercises the sweep loop without processes."""
    backend, rank, world_size = "loopback", 0, 1

    def empty(self, count):
        return np.empty(count, np.float32)

    def fill(self, buf, values):
        buf[...] = values

    def to_numpy(self, buf):
        return buf.copy()

    def run(self, op, send, recv, root=0):
        if recv is not send:
            recv[...] = send

    def barrier(self):
        pass

    def synchronize(self):
        pass

    def allreduce_scalar(self, x, how="avg"):
        return x


def test_sweep_loop_and_table_round_trip_through_the_parser():
    rows = sweep(LoopbackComm(), "all_gather", sizes=[8, 16, 64, 4096], iters=3, warmup=1)
    # 8 B is 2 floats: below nccl-tests' 16-byte per-rank chunk, so the point is skipped as nccl-tests' size-0 row
    assert [r.size for r in rows] == [16, 64, 4096] and all(r.wrong == 0 for r in rows)
    parsed = nccltests.parse(format_table(rows))
    assert parsed.op == "all_gather" and parsed.nranks == 1 and parsed.backend == "loopback"
    assert [r.size for r in parsed.rows] == [16, 64, 4096] and nccltests.recheck(parsed) == []


def test_rank_inputs_make_float32_sums_exact():
    total = sum(rank_input(r, 1000).astype(np.float64) for r in range(64))
    assert np.array_equal(total.astype(np.float32).astype(np.float64), total)


@pytest.mark.parametrize("op,n", [("all_reduce", 3), ("reduce_scatter", 3), ("all_gather", 2),
                                  ("broadcast", 3), ("sendrecv", 2)])
def test_ring_over_pipes_is_correct(op, n):
    rows = pipes.run(op, n, sizes=[8, 1000, 65536], iters=2, warmup=1, timeout=120)
    assert rows and all(r.wrong == 0 and r.backend == "pipes" and r.time_us > 0 for r in rows)


def test_pipes_backend_refuses_alltoall_cleanly():
    with pytest.raises(RuntimeError, match="all-pairs"):
        pipes.run("alltoall", 2, sizes=[64], iters=1, warmup=0, timeout=120)


def test_ring_schedule_sends_2_n_minus_1_chunks_per_rank_and_ends_complete():
    n = 4
    sched = semantics.ring_schedule(n)
    for r in range(n):
        assert sum(1 for _, _, rank, _, _ in sched if rank == r) == 2 * (n - 1)  # the busbw factor, counted
    for phase, s, r, send, recv in sched:  # what r receives is what r-1 sends in the same step
        assert recv == semantics.ring_chunks((r - 1) % n, s, n, phase)[0]


def test_ring_schedule_matches_the_primer_trace():
    # PRIMER.md §5.2, 4 ranks: step 1 "0->1:c3+ 1->2:c0+ 2->3:c1+ 3->0:c2+", step 4 (AG 1) "0->1:c0 ...",
    # step 6 (AG 3) "0->1:c2 1->2:c3 2->3:c0 3->0:c1"
    sends = lambda phase, s: [semantics.ring_chunks(r, s, 4, phase)[0] for r in range(4)]  # noqa: E731
    assert sends("rs", 1) == [3, 0, 1, 2] and sends("rs", 3) == [1, 2, 3, 0]
    assert sends("ag", 1) == [0, 1, 2, 3] and sends("ag", 3) == [2, 3, 0, 1]
    with pytest.raises(ValueError):
        semantics.ring_chunks(0, 0, 4, "rs")  # steps are numbered 1 … n-1


@pytest.mark.parametrize("n,length", [(2, 8), (3, 7), (4, 16), (8, 20)])
def test_executing_the_ring_schedule_gives_all_reduce_and_the_busbw_bytes(n, length):
    xs = [rng.integers(0, 9, length).astype(np.float32) for _ in range(n)]
    out, sent = semantics.ring_all_reduce(xs)
    want = semantics.all_reduce(xs)
    assert all(np.array_equal(a, b) for a, b in zip(out, want))
    if length % n == 0:  # equal chunks: exactly 2(n-1)/n of the buffer leaves every rank
        assert sent == [2 * (n - 1) * xs[0].nbytes // n] * n


def test_a_wrong_ring_schedule_is_caught():
    xs = [np.arange(8, dtype=np.float32) + r for r in range(4)]
    with pytest.raises(ValueError, match="expects chunk"):  # inconsistent: receiver and sender disagree
        semantics.ring_all_reduce(xs, lambda r, s, n, p: (r, (r - 2) % n))
    stale = semantics.ring_all_reduce(xs, lambda r, s, n, p: (r % n, (r - 1) % n))  # consistent but wrong
    assert not all(np.array_equal(a, b) for a, b in zip(stale[0], semantics.all_reduce(xs)))
