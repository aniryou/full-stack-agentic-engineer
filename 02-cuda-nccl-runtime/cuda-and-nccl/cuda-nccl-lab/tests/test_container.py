"""How a container sees its GPU: mounts, device nodes, driver versions and the compatibility rules."""
import pytest

from gpurt import container as c


def test_toolkit_injection_from_the_docker_sample(fixture_text):
    sections = c.parse_probe_log(fixture_text("probe_docker_t4_sample.log"))
    mounts = c.parse_mountinfo(sections["mountinfo"])
    files = c.injected_driver_files(mounts)
    assert "driver API (libcuda)" in {f.kind for f in files}
    assert c.injection_mechanism(files).startswith("NVIDIA Container Toolkit")
    assert c.driver_version_from_paths([f.mount_point for f in files]) == "570.172.08"


def test_gke_device_plugin_from_the_gke_sample(fixture_text):
    rep = c.from_probe_log(fixture_text("probe_gke_l4_sample.log"))
    assert rep.mechanism.startswith("GKE device plugin")
    assert len([n for n in rep.nodes if n.kind == "gpu"]) == 1 and rep.devices[0]["cc"] == [8, 9]
    assert rep.driver_version == "570.172.08" and rep.driver_cuda == "12.8"
    assert rep.verdicts[0].level == "ok"


def test_driver_branch_to_cuda_version():
    assert [c.cuda_of_driver(v) for v in ("470.82.01", "535.104.05", "550.54.15", "580.65.06")] == \
        ["11.4", "12.2", "12.4", "13.0"]


def test_proc_version_strings_proprietary_and_open():
    assert c.parse_proc_driver_version("NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.54.15  Tue") == "550.54.15"
    assert c.parse_proc_driver_version(
        "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  570.172.08  Release Build") == "570.172.08"


def test_binary_and_ptx_compatibility_rules():
    assert c.sass_runs_on("sm_80", (8, 9)) and c.sass_runs_on("sm_80", (8, 6))
    assert not c.sass_runs_on("sm_86", (8, 0)) and not c.sass_runs_on("sm_80", (9, 0))
    assert c.sass_runs_on("sm_90a", (9, 0)) and not c.sass_runs_on("sm_90a", (10, 0))
    assert c.ptx_jits_on("compute_80", (9, 0)) and not c.ptx_jits_on("compute_90", (8, 9))
    assert c.parse_arch("sm_120") == ("sass", 12, 0, "")


@pytest.mark.parametrize("driver,runtime,level,needle", [
    ("12.8", "12.4", "ok", "newer drivers run older runtimes"),
    ("12.2", "12.4", "warn", "minor-version compatibility"),
    ("11.4", "12.1", "fail", "insufficient"),
])
def test_driver_runtime_verdicts(driver, runtime, level, needle):
    v = c.compat_verdicts(driver, runtime)[0]
    assert v.level == level and needle in v.message


def test_no_kernel_image_and_ptx_jit_verdicts():
    fail = c.compat_verdicts("12.8", "12.8", (7, 5), ["sm_80", "sm_86", "sm_90"])[-1]
    assert fail.level == "fail" and "no kernel image" in fail.message
    jit = c.compat_verdicts("12.8", "12.8", (12, 0), ["sm_80", "sm_90", "compute_90"])[-1]
    assert jit.level == "warn" and "JIT" in jit.message
    blocked = c.compat_verdicts("12.4", "12.8", (12, 0), ["sm_90", "compute_90"])[-1]
    assert blocked.level == "fail"  # the only route is JIT, and PTX from 12.8 is too new for a 12.4 driver
    assert c.compat_verdicts(None, "12.8")[0].level == "fail"


def test_device_nodes_and_ls_parsing(tmp_path):
    for name in ("nvidia0", "nvidiactl", "nvidia-uvm", "not-a-gpu"):
        (tmp_path / name).write_text("")
    kinds = {n.kind for n in c.device_nodes(str(tmp_path))}
    assert kinds == {"gpu", "control", "unified memory"}
    nodes = c.parse_ls_dev("crw-rw-rw- 1 root root 195,   0 Sep 20 09:59 /dev/nvidia0")
    assert (nodes[0].major, nodes[0].minor, nodes[0].kind) == (195, 0, "gpu")


def test_live_probe_runs_anywhere():
    rep = c.probe(driver=False)  # no GPU here: a report that says so, not an exception
    assert "Device nodes" in c.explain(rep)


def test_libcuda_baked_into_an_image_is_flagged(tmp_path):
    injected_dir, baked = tmp_path / "injected", tmp_path / "image"
    injected_dir.mkdir(), baked.mkdir()
    (injected_dir / "libcuda.so.570.172.08").write_text("")
    (baked / "libcuda.so.535.104.05").write_text("")
    mounts = [c.InjectedFile(str(injected_dir / "libcuda.so.570.172.08"), "driver API (libcuda)", "/")]
    assert c.stray_libcuda(mounts, lib_dirs=(str(injected_dir), str(baked))) == [str(baked / "libcuda.so.535.104.05")]
