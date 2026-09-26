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
    rows = sweep(LoopbackComm(), "all_gather", sizes=[8, 64, 4096], iters=3, warmup=1)
    assert [r.size for r in rows] == [8, 64, 4096] and all(r.wrong == 0 for r in rows)
    parsed = nccltests.parse(format_table(rows))
    assert parsed.op == "all_gather" and parsed.nranks == 1 and parsed.backend == "loopback"
    assert [r.size for r in parsed.rows] == [8, 64, 4096] and nccltests.recheck(parsed) == []


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
