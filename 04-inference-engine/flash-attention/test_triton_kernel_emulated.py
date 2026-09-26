"""Runs the Triton kernel printed in flash-attention-deep-dive.md, section 11.1, on a CPU.

Neither torch nor triton is needed: the kernel's source is taken from the page, and its body is
executed program instance by program instance against a small numpy stand-in for the `tl`
operations it uses (loads and stores with masks, tl.dot with fp32 accumulation, exp2, ...).
This checks the kernel's indexing, masking and algebra; it does not check Triton's compiler.
For that, run section 11.4 (TRITON_INTERPRET=1) or section 11.3 on a GPU.

Offline, CPU only:  python3 -m pytest -q test_triton_kernel_emulated.py
"""
import builtins
import math
import pathlib
import re
import types

import numpy as np
import pytest

PAGE = pathlib.Path(__file__).with_name("flash-attention-deep-dive.md")


# --- a numpy stand-in for the parts of torch / triton / triton.language the kernel uses ------
class _TA(np.ndarray):
    """ndarray with Triton's .to(dtype)."""
    def to(self, dtype):
        return np.asarray(self).astype(dtype).view(_TA)


def _t(x):
    return np.asarray(x).view(_TA)


class _Ptr:
    """A pointer (or a block of pointers): a flat array plus element offsets."""
    def __init__(self, arr, off):
        self.arr, self.off = arr, off

    def __add__(self, o):
        return _Ptr(self.arr, self.off + np.asarray(o))

    __radd__ = __add__

    @property
    def dtype(self):
        return types.SimpleNamespace(element_ty=self.arr.dtype)


def _make_tl(pid):
    def load(ptr, mask=None, other=0.0):
        off = np.asarray(ptr.off)
        mask = np.ones(off.shape, bool) if mask is None else np.asarray(mask)
        off, mask = np.broadcast_arrays(off, mask)
        assert (off[mask] >= 0).all() and (off[mask] < ptr.arr.size).all(), "unmasked out-of-bounds load"
        return _t(np.where(mask, ptr.arr[np.where(mask, off, 0)], other).astype(ptr.arr.dtype))

    def store(ptr, val, mask=None):
        off = np.asarray(ptr.off)
        mask = np.ones(off.shape, bool) if mask is None else np.asarray(mask)
        off, mask, val = np.broadcast_arrays(off, mask, np.asarray(val))
        assert (off[mask] >= 0).all() and (off[mask] < ptr.arr.size).all(), "unmasked out-of-bounds store"
        ptr.arr[off[mask]] = val[mask].astype(ptr.arr.dtype)

    def exp2(x):
        with np.errstate(invalid="ignore"):
            return _t(np.exp2(np.asarray(x, dtype=np.float32)))

    return types.SimpleNamespace(
        constexpr=object, float32=np.float32, float16=np.float16,
        program_id=lambda axis: np.int64(pid[axis]),
        arange=lambda a, b: _t(np.arange(a, b)),
        full=lambda shape, value, dtype: _t(np.full(shape, value, dtype=dtype)),
        zeros=lambda shape, dtype: _t(np.zeros(shape, dtype=dtype)),
        dot=lambda a, b: _t(np.asarray(a, np.float32) @ np.asarray(b, np.float32)),
        where=lambda c, a, b: _t(np.where(c, a, b).astype(np.float32)),
        maximum=lambda a, b: _t(np.maximum(a, b)),
        minimum=np.minimum,
        max=lambda x, axis: _t(np.max(x, axis=axis)),
        sum=lambda x, axis: _t(np.sum(x, axis=axis, dtype=np.float32)),
        math=types.SimpleNamespace(exp2=exp2, log2=lambda x: _t(np.log2(np.asarray(x, np.float32)))),
        load=load, store=store)


class _Tensor:
    def __init__(self, a):
        self.a = a
    shape = property(lambda self: self.a.shape)
    dtype = property(lambda self: self.a.dtype)
    device = "cpu"

    def stride(self, dim=None):
        s = tuple(x // self.a.itemsize for x in self.a.strides)
        return s if dim is None else s[dim]


def _page_source():
    text = PAGE.read_text(encoding="utf-8")
    sec = text[text.index("### 11.1 The kernel"):text.index("### 11.2")]
    return re.search(r"```python\n(.*?)```", sec, re.S).group(1)


def _load_page_kernel(src=None):
    """Exec section 11.1's code block with fake torch / triton modules. Returns flash_attention."""
    src = _page_source() if src is None else src
    pid = [0, 0]
    tl = _make_tl(pid)

    class _Kernel:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            def launch(*args, num_warps=None, num_stages=None, **kw):
                args = [_Ptr(a.a.reshape(-1), 0) if isinstance(a, _Tensor) else a for a in args]
                for p1 in range(grid[1]):
                    for p0 in range(grid[0]):
                        pid[:] = [p0, p1]
                        self.fn(*args, **kw)
            return launch

    triton = types.SimpleNamespace(jit=_Kernel, cdiv=lambda a, b: (a + b - 1) // b, language=tl)
    torch = types.SimpleNamespace(
        float32=np.float32,
        empty_like=lambda t: _Tensor(np.full_like(t.a, np.nan)),
        empty=lambda shape, device=None, dtype=np.float32: _Tensor(np.full(shape, np.nan, dtype=dtype)))
    fakes = {"torch": torch, "triton": triton, "triton.language": tl}

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in fakes:
            return triton if name == "triton.language" and not fromlist else fakes[name]
        return builtins.__import__(name, globals, locals, fromlist, level)

    ns = {"__builtins__": {**vars(builtins), "__import__": fake_import}}
    exec(compile(src, "flash-attention-deep-dive.md#11.1", "exec"), ns)
    return ns["flash_attention"]


def _reference(q, k, v, causal):
    q, k, v = (x.astype(np.float64) for x in (q, k, v))
    s = q @ np.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])
    if causal:
        n = s.shape[-1]
        s = np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)
    mx = s.max(-1, keepdims=True)
    p = np.exp(s - mx)
    return (p / p.sum(-1, keepdims=True)) @ v, mx[..., 0] + np.log(p.sum(-1))


FLASH = _load_page_kernel()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape,blocks", [
    ((1, 1, 1, 64), (32, 16)),        # N = 1
    ((1, 2, 64, 64), (32, 16)),       # a tile multiple
    ((2, 3, 65, 64), (64, 32)),       # a tile multiple plus one; batch and head strides
    ((1, 2, 200, 128), (128, 64)),    # the default tiles, ragged tail
    ((1, 1, 300, 64), (16, 64)),      # BLOCK_M < BLOCK_N: diagonal tiles span several Q blocks
])
def test_page_kernel_matches_reference(shape, blocks, causal):
    rng = np.random.default_rng(sum(shape) + causal)
    q, k, v = (rng.normal(size=shape).astype(np.float16) for _ in range(3))
    o, lse = FLASH(_Tensor(q), _Tensor(k), _Tensor(v), causal=causal, block_m=blocks[0], block_n=blocks[1])
    o_ref, lse_ref = _reference(q, k, v, causal)
    assert not np.isnan(o.a).any() and not np.isnan(lse.a).any()       # every output row written
    assert np.abs(o.a.astype(np.float64) - o_ref).max() < 5e-3          # fp16 P and O rounding
    b, h, n, _ = shape
    np.testing.assert_allclose(lse.a.reshape(b, h, n) * math.log(2), lse_ref, atol=1e-3)   # base 2


@pytest.mark.parametrize("old,new", [
    ("offs_m[:, None] >= offs_k[None, :]", "offs_m[:, None] > offs_k[None, :]"),   # hides the diagonal
    ("(pid_m + 1) * BLOCK_M, N_CTX", "pid_m * BLOCK_M, N_CTX"),                     # stops one tile early
    ("acc * alpha[:, None]", "acc"),                                                # forgets the rescale
])
def test_the_check_catches_a_broken_kernel(old, new):
    src = _page_source()
    assert src.count(old) == 1                                  # the line the page prints
    broken = _load_page_kernel(src.replace(old, new))
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(1, 2, 130, 64)).astype(np.float16) for _ in range(3))
    with np.errstate(all="ignore"):
        o, _ = broken(_Tensor(q), _Tensor(k), _Tensor(v), causal=True, block_m=32, block_n=16)
    o_ref, _ = _reference(q, k, v, causal=True)
    assert not (np.abs(o.a.astype(np.float64) - o_ref).max() < 5e-3)
