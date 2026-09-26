"""safetensors from scratch, honest load measurements, and the cold-start model."""
import json
import os
import struct

import numpy as np
import pytest

from gpubench import loading


def test_roundtrip_header_alignment_and_offsets(tmp_path):
    path = tmp_path / "t.safetensors"
    a = np.arange(6, dtype="<f4").reshape(2, 3)
    b = np.arange(5, dtype="<i8")
    n = loading.write_safetensors(path, {"a": ("F32", (2, 3), a), "b": ("I64", (5,), b), "s": ("F16", (), b"\x00\x3c")},
                                  metadata={"format": "pt"})
    assert n == os.path.getsize(path)
    header, start = loading.read_header(path)
    assert start % 8 == 0                                            # data section is 8-byte aligned
    assert header["a"]["data_offsets"] == [0, 24] and header["b"]["data_offsets"] == [24, 64]
    assert header["s"]["data_offsets"] == [64, 66] and header["__metadata__"] == {"format": "pt"}
    assert loading.validate(header, n, start) == []
    t = loading.load_tensors(path)
    assert np.array_equal(t["a"], a) and np.array_equal(t["b"], b) and t["s"].item() == 1.0
    with open(path, "rb") as f:                                      # the raw layout, byte for byte
        (hlen,) = struct.unpack("<Q", f.read(8))
        assert json.loads(f.read(hlen)) == header


def test_validate_catches_holes_and_wrong_sizes():
    bad = {"x": {"dtype": "F32", "shape": [4], "data_offsets": [0, 12]},
           "y": {"dtype": "BF16", "shape": [2], "data_offsets": [16, 20]}}
    problems = loading.validate(bad, 8 + 8 + 20, 16)
    assert any("shape needs 16" in p for p in problems) and any("hole or overlap" in p for p in problems)


def test_optional_crosscheck_with_the_safetensors_library(tmp_path):
    st = pytest.importorskip("safetensors.numpy")
    path = tmp_path / "x.safetensors"
    w = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    loading.write_safetensors(path, {"w": ("F32", w.shape, w)})
    assert np.array_equal(st.load_file(str(path))["w"], w)


def test_synthetic_checkpoint_and_every_read_method_returns_the_file(tmp_path):
    path = tmp_path / "ckpt.safetensors"
    info = loading.synthetic_checkpoint(path, 1 << 20, hidden=128, vocab=1024)
    assert info["bytes"] == os.path.getsize(path) >= 1 << 20
    header, start = loading.read_header(path)
    assert loading.validate(header, info["bytes"], start) == []
    ref = path.read_bytes()
    for method, threads in (("read", 1), ("pread", 3), ("mmap", 1)):
        dst = np.zeros(info["bytes"], dtype=np.uint8)
        loading.METHODS[method](path, dst, *((threads, 1 << 16) if method == "pread" else ()))
        assert dst.tobytes() == ref, method


def test_load_measurements_count_file_bytes(tmp_path):
    path = tmp_path / "ckpt.safetensors"
    loading.synthetic_checkpoint(path, 1 << 20, hidden=128, vocab=1024)
    size = os.path.getsize(path)
    ms = loading.bench_load(path, methods=("read", "pread"), caches=("warm",), threads=(2,), repeats=2)
    assert [m.op for m in ms] == ["load.read", "load.pread"]
    assert all(m.cost.bytes == size and m.cost.basis == "transfer" for m in ms)
    op = loading.make_load_op(path, "mmap", cold=True)
    assert op.max_inner == 1 and op.setup is not None
    op.fn()
    assert op.verify()


def test_cold_start_model():
    t, slowest = loading.load_time(16e9, [("gcs", 1.6e9), ("pcie", 25e9)])
    assert (t, slowest) == (pytest.approx(10.0), "gcs")                        # pipelined: the slowest tier
    t2, _ = loading.load_time(16e9, [("gcs", 1.6e9), ("pcie", 25e9)], pipelined=False)
    assert t2 == pytest.approx(10.0 + 0.64)                                    # store-and-forward: the sum
    assert loading.streams_needed(10e9, 0.05, 16e6) == pytest.approx(31.25)   # Little's law
    cs = loading.cold_start({"provision": 60, "image": 30, "weights": 90, "init": 20})
    assert cs["total_s"] == 200 and cs["largest"] == "weights" and cs["shares"]["weights"] == pytest.approx(0.45)
