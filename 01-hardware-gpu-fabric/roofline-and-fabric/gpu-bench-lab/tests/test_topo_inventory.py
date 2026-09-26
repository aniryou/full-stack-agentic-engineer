"""Parsing `nvidia-smi` output (bundled samples in the documented format, illustrative) and
the placement decisions that follow from it."""
import pytest

from gpubench import inventory, p2p, topo


@pytest.fixture(scope="module")
def hgx():
    return topo.parse(topo.load_fixture("hgx-h100-8gpu"))


def test_parse_hgx_matrix(hgx):
    assert hgx.gpus == [f"GPU{i}" for i in range(8)] and hgx.nics == [f"NIC{i}" for i in range(8)]
    assert hgx.link("GPU0", "GPU7") == "NV18" and hgx.nvlinks("GPU2", "GPU5") == 18
    assert hgx.link("GPU0", "GPU0") == "X"
    assert hgx.nic_names["NIC3"] == "mlx5_3"
    assert hgx.numa_affinity["GPU5"] == "1"


def test_link_rank_orders_paths():
    order = ["NV18", "NV12", "NV4", "PIX", "PXB", "PHB", "NODE", "SYS"]
    assert sorted(order, key=topo.link_rank, reverse=True) == order


def test_best_group_prefers_the_pcie_switch_pair():
    t = topo.parse(topo.load_fixture("pcie-4gpu-2socket"))
    assert topo.best_group(t, 2) == ("GPU0", "GPU1")          # PIX beats SYS
    assert t.link("GPU1", "GPU2") == "SYS"


def test_nearest_nic_and_cpu_affinity(hgx):
    assert topo.nearest_nic(hgx, "GPU6") == ("NIC6", "PXB")
    cpus = topo.cpu_list(hgx, "GPU5")
    assert cpus[:2] == [48, 49] and len(cpus) == 96
    assert topo.parse_cpu_list("0-3,8") == [0, 1, 2, 3, 8] and topo.parse_cpu_list("N/A") == []


def test_parser_accepts_space_separated_copy_paste():
    text = topo.load_fixture("a2-highgpu-2g").replace("\t", "    ")
    t = topo.parse(text)
    assert t.gpus == ["GPU0", "GPU1"] and t.link("GPU0", "GPU1") == "NV12" and t.cpu_affinity["GPU1"] == "0-23"


def test_predicted_p2p_is_labelled_a_model():
    hgx = topo.parse(topo.load_fixture("hgx-h100-8gpu"))
    est = p2p.predict(hgx, arch="hopper")[("GPU0", "GPU1")]
    assert est.gbs == 450 and "NVLink4" in est.why
    pcie = p2p.predict(topo.parse(topo.load_fixture("pcie-4gpu-2socket")), pcie_gen=4)
    assert pcie[("GPU0", "GPU1")].gbs == pytest.approx(31.508, abs=1e-3)
    assert pcie[("GPU0", "GPU2")].gbs == pytest.approx(31.508 / 2, abs=1e-3)   # staged through host


def test_p2p_model_says_what_it_assumes_about_peer_access():
    kaggle = topo.parse(topo.load_fixture("kaggle-2xt4"))
    unknown = p2p.predict(kaggle, arch="turing", pcie_gen=3)[("GPU0", "GPU1")]
    assert unknown.path == "PHB" and unknown.mode == "upper-bound"
    assert unknown.gbs == pytest.approx(15.754, abs=1e-3)                        # only if peer access works
    off = p2p.predict(kaggle, arch="turing", pcie_gen=3, peer_access={("GPU0", "GPU1"): False})[("GPU0", "GPU1")]
    assert off.mode == "staged" and off.gbs == pytest.approx(15.754 / 2, abs=1e-3)
    assert p2p.path_bandwidth("PHB", pcie_gen=3, peer_access=True).mode == "direct"
    assert p2p.path_bandwidth("PIX", pcie_gen=4, peer_access=False).mode == "staged"
    assert p2p.path_bandwidth("SYS", pcie_gen=4).mode == "staged"
    assert p2p.path_bandwidth("NV12", arch="ampere").gbs == 300
    with pytest.raises(ValueError):
        p2p.path_bandwidth("XYZ")


def test_ring_allreduce_formula():
    assert p2p.ring_allreduce_time(1e9, 8, 0.0, 100e9) == pytest.approx(0.0175)     # 2·7/8 · 1 GB / 100 GB/s
    assert p2p.ring_allreduce_time(0, 8, 10e-6, 1e9) == pytest.approx(140e-6)      # 2·7 latency steps
    assert p2p.ring_allreduce_time(123, 1, 1e-3, 1e9) == 0.0
    per_token = p2p.tp_allreduce_per_token(8192, 80, 1, 2, 8, 10e-6, 450e9)
    assert per_token == pytest.approx(80 * 2 * (14 * 10e-6 + 1.75 * 16384 / 450e9))


def test_inventory_csv_units_and_missing_values():
    rows = inventory.parse_csv(inventory.load_fixture("hgx-h100-8gpu"))
    assert len(rows) == 8
    r = rows[0]
    assert r["name"] == "NVIDIA H100 80GB HBM3" and r["compute_cap"] == "9.0"
    assert r["memory.total"] == 81559 and r["power.limit"] == 700.0 and r["pcie.link.width.max"] == 16
    assert r["_units"]["memory.total"] == "MiB"
    na = inventory.parse_csv("index, power.draw [W], name\n0, [N/A], Tesla T4\n")
    assert na[0]["power.draw"] is None and na[0]["name"] == "Tesla T4"


def test_health_finds_the_sick_gpus():
    rows = inventory.parse_csv(inventory.load_fixture("hgx-h100-8gpu"))
    warns = {(f.gpu, f.message.split(":")[0].split(" ")[0]) for f in inventory.health(rows) if f.level == "WARN"}
    assert {g for g, _ in warns} == {3, 5, 6}                  # power-capped, x8 link, memory in use
    t4 = inventory.health(inventory.parse_csv(inventory.load_fixture("colab-t4")))
    assert [f.level for f in t4] == ["INFO", "INFO"]           # idle link gen + persistence mode: not faults


def test_cpu_info_describes_this_machine():
    info = inventory.cpu_info()
    assert info["logical_cpus"] >= 1 and info["simd_bits"] in (128, 256, 512)
    assert isinstance(info["caches"], dict)
