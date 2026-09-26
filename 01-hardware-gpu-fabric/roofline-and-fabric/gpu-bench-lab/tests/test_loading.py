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
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 8)).astype(np.float32)
    h = rng.standard_normal((3,)).astype(np.float16)
    i = np.arange(5, dtype=np.int64)
    # ours -> the library: several dtypes and metadata (the header is space-padded to 8 bytes)
    ours = tmp_path / "ours.safetensors"
    loading.write_safetensors(ours, {"w": ("F32", w.shape, w), "h": ("F16", h.shape, h), "i": ("I64", i.shape, i)},
                              metadata={"format": "pt", "note": "x"})
    back = st.load_file(str(ours))
    assert np.array_equal(back["w"], w) and np.array_equal(back["h"], h) and np.array_equal(back["i"], i)
    # the library -> ours: its own layout (it orders tensors by alignment, not by name)
    theirs = tmp_path / "theirs.safetensors"
    st.save_file({"w": w, "h": h, "i": i}, str(theirs), metadata={"format": "np"})
    header, start = loading.read_header(theirs)
    assert header["__metadata__"] == {"format": "np"} and start % 8 == 0
    assert loading.validate(header, theirs.stat().st_size, start) == []
    t = loading.load_tensors(theirs)
    assert np.array_equal(t["w"], w) and np.array_equal(t["h"], h) and np.array_equal(t["i"], i)


def test_optional_bf16_crosscheck_with_the_safetensors_library(tmp_path):
    lib = pytest.importorskip("safetensors")
    if not hasattr(lib, "deserialize"):
        pytest.skip("this safetensors has no deserialize()")
    bf16 = np.array([0x3F80, 0x4000, 0xC040], dtype="<u2")        # 1.0, 2.0, -3.0 as raw bfloat16
    f32 = np.arange(3, dtype="<f4")
    path = tmp_path / "bf16.safetensors"
    loading.write_safetensors(path, {"b": ("BF16", (3,), bf16), "f": ("F32", (3,), f32)}, metadata={"k": "v"})
    parsed = dict(lib.deserialize(path.read_bytes()))                # the library parses our header and padding
    assert parsed["b"]["dtype"] == "BF16" and parsed["b"]["shape"] == [3]
    assert bytes(parsed["b"]["data"]) == bf16.tobytes() and bytes(parsed["f"]["data"]) == f32.tobytes()
    assert np.array_equal(loading.load_tensors(path)["b"], bf16)     # and ours returns the raw bits


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


MOUNTINFO = """\
28 1 254:0 / / rw,relatime - ext4 /dev/vda rw
40 28 0:30 / /tmp rw,nosuid,nodev - tmpfs tmpfs rw,size=8G
41 28 254:1 / /mnt/my\\040disk rw,relatime - xfs /dev/vdb rw
"""


def test_filesystem_type_takes_the_deepest_mount_point():
    assert loading.filesystem_type("/tmp/x/ckpt.safetensors", MOUNTINFO) == "tmpfs"
    assert loading.filesystem_type("/tmpfoo/ckpt", MOUNTINFO) == "ext4"          # a prefix is not a parent
    assert loading.filesystem_type("/mnt/my disk/ckpt", MOUNTINFO) == "xfs"      # \\040 is a space
    assert loading.filesystem_type("/home/u/ckpt", MOUNTINFO) == "ext4"


def test_a_file_on_tmpfs_is_never_called_cold(tmp_path, monkeypatch):
    path = tmp_path / "ckpt.safetensors"
    loading.synthetic_checkpoint(path, 1 << 18, hidden=64, vocab=256)
    monkeypatch.setattr(loading, "filesystem_type", lambda p, mountinfo=None: "tmpfs")
    ok, why = loading.cold_read_possible(path)
    assert not ok and "tmpfs" in why and loading.drop_page_cache(path) is False
    skipped = []
    ms = loading.bench_load(path, methods=("read",), threads=(2,), repeats=1, skipped=skipped)
    assert [m.params["cache"] for m in ms] == ["warm"]                         # no row claims to be cold
    assert skipped and "tmpfs" in skipped[0]["reason"]
    op = loading.make_load_op(path, "read", cold=True)
    assert op.note.startswith("NOT cold") and op.extras["cold_valid"] is False


def test_default_workdir_avoids_ram_backed_temp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loading.tempfile, "gettempdir", lambda: "/ramdisk-tmp")
    monkeypatch.setattr(loading, "filesystem_type", lambda p, mountinfo=None: "tmpfs" if p == "/ramdisk-tmp" else "ext4")
    d = loading.default_workdir()
    assert d.startswith(str(tmp_path))                                          # the cwd, not the tmpfs /tmp
