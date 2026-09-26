"""The bytes on disk: FP8, bf16 and INT4 packing, bit for bit."""
import numpy as np
import pytest

from quantlab import numerics as N, stio


def test_e4m3_and_e5m2_facts():
    e4 = N.representable("e4m3")
    assert len(e4) == 127 and e4[-1] == 448.0 and e4[1] == 2.0 ** -9       # 127 non-negative finite values
    assert N.E4M3.min_normal == 2.0 ** -6 and N.E5M2.max_finite == 57344.0 and N.E5M2.min_subnormal == 2.0 ** -16
    x = np.array([448.0, 500.0, 0.1, 2.0 ** -9, 2.0 ** -10, 1.0625, 1.1875, -3.3, 0.0])
    np.testing.assert_array_equal(N.minifloat_round(x, "e4m3"), [448, 448, 0.1015625, 2.0 ** -9, 0, 1.0, 1.25, -3.25, 0])
    assert N.fp8_encode([1.0, -2.0, 448.0], "e4m3").tolist() == [56, 192, 126]   # 0x38, 0xC0, 0x7E
    assert np.isnan(N.fp8_decode([0x7F], "e4m3")[0]) and np.isinf(N.fp8_decode([0x7C], "e5m2")[0])


def test_minifloat_codes_match_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.standard_normal(20000) * s for s in (1e-3, 1, 30, 300)]).astype(np.float32)
    for fmt, td in (("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)):
        mine = N.fp8_encode(x, fmt)
        ref = torch.from_numpy(x).to(td).view(torch.uint8).numpy()
        assert (mine == ref).all(), fmt
        codes = np.arange(256, dtype=np.uint8)
        np.testing.assert_array_equal(N.fp8_decode(codes, fmt), torch.from_numpy(codes).view(td).float().numpy())
    assert (N.bf16_encode(x) == torch.from_numpy(x).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)).all()


def test_int4_packing_is_compressed_tensors_order():
    word = N.pack_int32(np.array([[-8, -7, 0, 1, 2, 3, 4, 7]]), 4)
    assert word.dtype == np.int32 and hex(int(word.view(np.uint32)[0, 0])) == "0xfcba9810"
    q = np.random.default_rng(1).integers(-8, 8, (6, 896))
    p = N.pack_int32(q, 4)
    assert p.shape == (6, 112)                                               # [4096, 896] -> [4096, 112]
    assert (N.unpack_int32(p, 4, 896) == q).all()
    with pytest.raises(ValueError):
        N.pack_int32(np.array([[8]]), 4)


def test_safetensors_roundtrip_and_header(tmp_path):
    t = {"a.weight": stio.bf16(np.array([[1.0, -2.5], [3.0, 0.1]])), "a.weight_packed": np.array([[1, 2]], np.int32),
         "a.fp8": stio.f8_e4m3(np.array([1.0, 448.0])), "a.shape": np.array([2, 16], np.int64)}
    size = stio.save(t, tmp_path / "m.safetensors", metadata={"format": "pt"})
    raw = (tmp_path / "m.safetensors").read_bytes()
    n = int.from_bytes(raw[:8], "little")
    assert n % 8 == 0 and size == len(raw) == 8 + n + 8 + 8 + 2 + 16
    back = stio.load(tmp_path / "m.safetensors")
    assert back["a.weight"].dtype == "BF16" and back["a.fp8"].numpy().tolist() == [1.0, 448.0]
    np.testing.assert_array_equal(back["a.weight"].numpy(), N.bf16_round([[1.0, -2.5], [3.0, 0.1]]))
    assert stio.read_header(tmp_path / "m.safetensors")["__metadata__"] == {"format": "pt"}


def test_files_load_with_the_safetensors_package(tmp_path):
    st = pytest.importorskip("safetensors.numpy")
    stio.save({"x": np.arange(6, dtype=np.float32).reshape(2, 3), "p": np.array([7], np.int32)}, tmp_path / "a.safetensors")
    got = st.load_file(str(tmp_path / "a.safetensors"))
    assert got["x"].tolist() == [[0, 1, 2], [3, 4, 5]] and got["p"].tolist() == [7]
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file
    stio.save({"w": stio.bf16([1.5, -0.25]), "f": stio.f8_e4m3([0.5, 3.0])}, tmp_path / "b.safetensors")
    got = load_file(str(tmp_path / "b.safetensors"))
    assert got["w"].dtype == torch.bfloat16 and got["w"].float().tolist() == [1.5, -0.25]
    assert got["f"].dtype == torch.float8_e4m3fn and got["f"].float().tolist() == [0.5, 3.0]
