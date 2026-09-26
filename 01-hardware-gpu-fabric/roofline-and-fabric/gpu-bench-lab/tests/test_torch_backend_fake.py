"""The torch backend's accounting and call conventions, with a stand-in ``torch`` module.

No GPU (and no PyTorch) is needed: ``FakeTorch`` implements just the API surface the backend
uses — shapes, element sizes, streams, events, the float32 matmul-precision switch — and checks
the calls it receives against the upstream signatures (``torch._scaled_mm(self, mat2, scale_a,
scale_b, bias=None, scale_result=None, out_dtype=None, use_fast_accum=False)`` with a row-major
``self`` and a column-major ``mat2``). So every op the GPU tiers build is pinned here to the
FLOPs and bytes accounting.py says it costs.
"""
import math
import sys
import types
from contextlib import contextmanager

import numpy as np
import pytest

from gpubench import get_backend, inventory
from gpubench.accounting import gemm_cost, stream_cost, transfer_cost
from gpubench.backends import BackendUnavailable


class Dtype:
    def __init__(self, name, itemsize):
        self.name, self.itemsize = name, itemsize

    def __repr__(self):
        return f"torch.{self.name}"


class Tensor:
    def __init__(self, shape, dtype, device="cpu", col_major=False, pinned=False, log=None):
        self.shape = tuple(int(d) for d in shape)
        self.dtype, self.device, self.col_major, self.pinned = dtype, str(device), col_major, pinned
        self.log = log if log is not None else []
        self._np = None

    def element_size(self):
        return self.dtype.itemsize

    def numel(self):
        return math.prod(self.shape)

    def t(self):
        return Tensor(self.shape[::-1], self.dtype, self.device, not self.col_major, self.pinned, self.log)

    def to(self, dtype):
        return Tensor(self.shape, dtype, self.device, self.col_major, self.pinned, self.log)

    def fill_(self, value):
        return self

    def mul_(self, value):
        self.log.append(("mul_", self.numel()))
        return self

    def copy_(self, other, non_blocking=False):
        assert self.numel() == other.numel(), "copy_ between tensors of different sizes"
        self.log.append(("copy_", other.device, self.device, other.numel() * other.element_size(), non_blocking))
        return self

    def __getitem__(self, s):
        assert len(self.shape) == 1 and isinstance(s, slice)
        return Tensor((len(range(self.shape[0])[s]),), self.dtype, self.device, pinned=self.pinned, log=self.log)

    def numpy(self):                 # pinned staging buffers are filled through numpy views
        if self._np is None:
            self._np = np.zeros(self.numel(), dtype=np.uint8)
        return self._np


class Event:
    def __init__(self, torch, enable_timing=False):
        self.torch, self.t = torch, None

    def record(self, stream=None):
        self.t = self.torch.clock

    def synchronize(self):
        pass

    def elapsed_time(self, end):
        return (end.t - self.t) * 1e3           # milliseconds, as CUDA reports them


class FakeTorch(types.ModuleType):
    def __init__(self, capability=(8, 9), ngpu=2, peer=True, available=True):
        super().__init__("torch")
        self.__version__ = "9.9.9-fake"
        self.version = types.SimpleNamespace(cuda="12.6", hip=None)
        for name, size in (("float64", 8), ("float32", 4), ("float16", 2), ("bfloat16", 2), ("uint8", 1),
                           ("float8_e4m3fn", 1)):
            setattr(self, name, Dtype(name, size))
        self.precision, self.clock, self.log, self.scaled_mm_calls = "highest", 0.0, [], []
        self.capability, self.ngpu, self.peer, self.available = capability, ngpu, peer, available
        self.cuda = self._cuda()

    # -- tensors ------------------------------------------------------------------------------------------
    def _shape(self, shape):
        return shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape

    def device(self, kind, index=0):
        return f"{kind}:{index}"

    def Generator(self, device=None):
        return types.SimpleNamespace(manual_seed=lambda s: None)

    def randn(self, *shape, device=None, dtype=None, generator=None):
        return Tensor(self._shape(shape), dtype or self.float32, device, log=self.log)

    def empty(self, *shape, dtype=None, device="cpu", pin_memory=False):
        return Tensor(self._shape(shape), dtype or self.float32, device, pinned=pin_memory, log=self.log)

    def full(self, shape, value, device=None, dtype=None):
        return Tensor(shape, dtype or self.float32, device, log=self.log)

    def ones(self, *shape, device=None, dtype=None):
        return Tensor(self._shape(shape), dtype or self.float32, device, log=self.log)

    def tensor(self, data, dtype=None):
        return Tensor((len(data),), dtype or self.float32, log=self.log)

    def matmul(self, a, b, out=None):
        assert a.shape[1] == b.shape[0] and out.shape == (a.shape[0], b.shape[1])
        self.log.append(("matmul", a.dtype.name, self.precision))

    def mul(self, a, q, out=None):
        self.log.append(("mul", a.numel()))

    def add(self, a, b, alpha=1, out=None):
        self.log.append(("add", a.numel(), alpha))

    def _scaled_mm(self, a, b, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None,
                   use_fast_accum=False):
        assert not a.col_major and b.col_major, "_scaled_mm wants a row-major mat1 and a column-major mat2"
        assert a.shape[1] == b.shape[0] and all(d % 16 == 0 for d in (*a.shape, b.shape[1]))
        assert a.dtype.name == b.dtype.name == "float8_e4m3fn"
        assert scale_a.dtype.name == scale_b.dtype.name == "float32" and out_dtype is not None
        self.scaled_mm_calls.append((a.shape, b.shape, out_dtype.name))
        return Tensor((a.shape[0], b.shape[1]), out_dtype)

    def get_float32_matmul_precision(self):
        return self.precision

    def set_float32_matmul_precision(self, p):
        assert p in ("highest", "high", "medium")
        self.precision = p

    # -- torch.cuda -------------------------------------------------------------------------------------
    def _cuda(self):
        torch = self

        @contextmanager
        def ctx(*_a, **_k):
            yield

        return types.SimpleNamespace(
            is_available=lambda: torch.available, device_count=lambda: torch.ngpu,
            get_device_capability=lambda i=None: torch.capability,
            get_device_properties=lambda i=None: types.SimpleNamespace(
                name="Fake GPU", major=torch.capability[0], minor=torch.capability[1],
                multi_processor_count=58, total_memory=24 << 30, L2_cache_size=48 << 20),
            synchronize=lambda i=None: None,
            Event=lambda enable_timing=False: Event(torch, enable_timing),
            Stream=lambda device=None, priority=0: types.SimpleNamespace(device=device, synchronize=lambda: None),
            stream=ctx, device=ctx,
            can_device_access_peer=lambda a, b: torch.peer)


@pytest.fixture
def fake(monkeypatch):
    t = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", t)
    return t


def backend(fake):
    be = get_backend("torch")
    assert be.name == "torch" and be.is_gpu
    return be


# -- GEMM -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype, width", [("float32", 4), ("tf32", 4), ("float16", 2), ("bfloat16", 2)])
def test_gemm_is_charged_2mnk_and_its_compulsory_bytes(fake, dtype, width):
    op = backend(fake).make_gemm(256, 128, 64, dtype)
    assert op.cost == gemm_cost(256, 128, 64, width)
    assert op.cost.flops == 2 * 256 * 128 * 64 and op.cost.bytes == (256 * 64 + 64 * 128 + 256 * 128) * width


def test_tf32_runs_under_high_precision_and_restores_the_callers_setting(fake):
    be = backend(fake)
    fake.set_float32_matmul_precision("medium")               # the caller's own choice
    for dtype, inside in (("tf32", "high"), ("float32", "highest")):
        op = be.make_gemm(64, 64, 64, dtype)
        with op.context():
            op.fn()
            assert fake.get_float32_matmul_precision() == inside
        assert fake.get_float32_matmul_precision() == "medium"
    assert [e[2] for e in fake.log if e[0] == "matmul"] == ["high", "highest"]


def test_fp8_gemm_reads_1_byte_writes_bf16_and_calls_scaled_mm_correctly(fake):
    be = backend(fake)
    op = be.make_gemm(64, 32, 128, "fp8")
    assert op.cost.bytes_read == (64 * 128 + 128 * 32) * 1 and op.cost.bytes_written == 64 * 32 * 2
    assert op.cost == gemm_cost(64, 32, 128, 1, out_bytes=2)
    op.fn()
    assert fake.scaled_mm_calls == [((64, 128), (128, 32), "bfloat16")]


@pytest.mark.parametrize("cap, expected", [((7, 5), ["float32", "float16", "bfloat16"]),
                                           ((8, 0), ["float32", "tf32", "float16", "bfloat16"]),
                                           ((8, 9), ["float32", "tf32", "float16", "bfloat16", "float8_e4m3fn"])])
def test_dtype_gating_by_compute_capability(fake, cap, expected):
    fake.capability = cap
    be = backend(fake)
    assert be.gemm_dtypes() == expected
    if cap < (8, 9):
        with pytest.raises(BackendUnavailable, match="8, 9"):
            be.make_gemm(64, 64, 64, "fp8")


# -- memory ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kernel, arrays, flops", [("copy", 2, 0), ("scale", 2, 1), ("add", 3, 1), ("triad", 3, 2)])
def test_gpu_stream_is_one_kernel_charged_streams_count(fake, kernel, arrays, flops):
    op = backend(fake).make_stream(kernel, 1000, "float32")
    assert op.cost == stream_cost(kernel, 1000, 4)
    assert op.cost.bytes == arrays * 1000 * 4 and op.cost.flops == flops * 1000
    assert op.extras == {"stream_bytes": arrays * 1000 * 4, "passes": 1}
    op.fn()
    if kernel == "triad":
        assert fake.log[-1] == ("add", 1000, 3.0)              # a = b + q·c as one fused kernel


def test_cache_ladder_probe_reads_and_writes_the_working_set(fake):
    op = backend(fake).make_scale_inplace(1 << 20, "float32")
    assert op.cost.bytes == 2 * (1 << 20) and op.extras["working_set_bytes"] == 1 << 20


# -- transfers ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pinned", [True, False])
def test_host_device_copies_count_delivered_bytes_once(fake, pinned):
    be = backend(fake)
    op = be.make_transfer(1 << 20, "h2d", pinned)
    assert op.cost == transfer_cost(1 << 20) and op.cost.bytes == 1 << 20 and op.timer == "wall"
    assert op.max_inner is None and op.extras["sync_each"] is False
    op.fn()
    assert fake.log[-1] == ("copy_", "cpu", "cuda:0", 1 << 20, pinned)   # only a pinned copy is asynchronous
    lat = be.make_transfer(4096, "d2h", pinned, sync_each=True)
    assert lat.max_inner == 1 and "synchronised" in lat.note


def test_p2p_unidirectional_and_bidirectional_bytes(fake):
    be = backend(fake)
    uni, bi = be.make_p2p(0, 1, 1 << 20), be.make_p2p(0, 1, 1 << 20, bidirectional=True)
    assert uni.cost.bytes == 1 << 20 and bi.cost.bytes == 2 << 20 and bi.cost.basis == "transfer"
    assert uni.extras["peer_access"] is True
    bi.fn()
    copies = [e for e in fake.log if e[0] == "copy_"][-2:]
    assert [(c[1], c[2]) for c in copies] == [("cuda:0", "cuda:1"), ("cuda:1", "cuda:0")]
    fake.peer = False
    assert "staged" in be.make_p2p(0, 1, 4096).note


def test_disk_to_device_pipeline_reads_every_byte(fake, tmp_path):
    be = backend(fake)
    data = np.random.default_rng(0).integers(0, 256, 10_000, dtype=np.uint8)
    path = tmp_path / "w.bin"
    path.write_bytes(data.tobytes())
    op = be.make_file_to_device(str(path), chunk_bytes=4096, buffers=2)
    assert op.cost == transfer_cost(10_000)
    op.fn()
    moved = [e[3] for e in fake.log if e[0] == "copy_" and e[2] == "cuda:0" and e[1] == "cpu"]
    assert moved[-3:] == [4096, 4096, 10_000 - 8192]               # three chunks, double-buffered


# -- timing and availability ------------------------------------------------------------------------------
def test_cuda_event_timing_converts_milliseconds_and_divides_by_inner(fake):
    be = backend(fake)

    def fn():
        fake.clock += 2e-3                                        # every call "takes" 2 ms on the GPU

    t = be.time(fn, repeats=3, min_time=0.01)
    assert t.method == "cuda-events" and t.inner >= 5
    assert t.samples == pytest.approx((2e-3,) * 3)


def test_unusable_driver_is_reported_as_such(monkeypatch):
    fake = FakeTorch(available=False)
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(inventory, "query_gpus", lambda: ([{"name": "NVIDIA L4", "driver_version": "550.54"}], ""))
    with pytest.raises(BackendUnavailable, match="driver 550.54.*R580.*cu126"):
        get_backend("torch")
    monkeypatch.setattr(inventory, "query_gpus", lambda: (None, "nvidia-smi not found"))
    with pytest.raises(BackendUnavailable, match="no CUDA GPU is visible"):
        get_backend("torch")
    assert get_backend("auto", verbose=False).name == "numpy"
