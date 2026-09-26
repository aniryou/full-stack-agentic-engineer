"""The two gates (driver <-> runtime, kernel image <-> GPU) and the errors they produce."""
from gpusim import compat


def test_driver_reports_its_max_cuda_version():
    cases = {"535.104.05": "12.2", "550.54.15": "12.4", "570.86.10": "12.8", "580.65.06": "13.0",
             "470.82.01": "11.4", "450.80.02": "11.0", "595.10": "13.1", "595.45.04": "13.2",
             "610.43.02": "13.3", "384.81": "9.0", "375.26": None}
    for driver, cuda in cases.items():
        assert compat.driver_cuda(driver) == cuda


def test_sass_runs_within_a_major_on_equal_or_newer_minors():
    assert compat.sass_runs_on("sm_80", "8.6") and compat.sass_runs_on("sm_80", "8.9")
    assert not compat.sass_runs_on("sm_86", "8.0")          # never backwards
    assert not compat.sass_runs_on("sm_80", "9.0")          # never across majors
    assert compat.sass_runs_on("sm_90a", "9.0") and not compat.sass_runs_on("sm_90a", "10.0")
    assert compat.sass_runs_on("sm_120", "12.0") and not compat.sass_runs_on("sm_100", "12.0")


def test_ptx_jit_compiles_forward_across_majors():
    assert compat.ptx_jits_to("compute_80", "9.0") and compat.ptx_jits_to("compute_80", "12.0")
    assert not compat.ptx_jits_to("compute_90", "8.9")
    assert not compat.ptx_jits_to("compute_90a", "10.0")


def test_family_targets_stay_in_their_family():
    assert compat.ptx_jits_to("compute_100f", "10.3") and not compat.ptx_jits_to("compute_100f", "12.0")
    assert compat.sass_runs_on("sm_100f", "10.3") and not compat.sass_runs_on("sm_100f", "12.0")
    assert compat.check("12.9", "575.51.03", gpu="RTX 5090", targets="compute_100f").code == 209
    assert compat.check("12.9", "575.51.03", gpu="B300", targets="compute_100f").ok


def test_parse_targets_both_spellings():
    assert compat.parse_targets("8.0 8.6 9.0+PTX") == (["sm_80", "sm_86", "sm_90"], ["compute_90"])
    assert compat.parse_targets("sm_90a,compute_90a") == (["sm_90a"], ["compute_90a"])


def test_minor_version_compatibility_runs_sass():
    v = compat.check("12.4", "535.104.05", gpu="H100", targets="8.0 9.0")
    assert v.ok and any("minor-version" in r for r in v.reasons)


def test_minor_version_compatibility_cannot_jit_newer_ptx():
    assert compat.check("12.4", "535.104.05", gpu="H100", targets="8.0+PTX").code == 222
    assert compat.check("12.2", "535.104.05", gpu="H100", targets="8.0+PTX").ok      # same version: JIT fine


def test_newer_major_needs_a_newer_driver_or_forward_compat():
    assert compat.check("13.0", "535.104.05", gpu="H100", targets="9.0").code == 35
    assert compat.check("13.0", "535.104.05", gpu="H100", targets="9.0", compat_cuda="13.0").ok
    assert compat.check("13.0", "535.104.05", gpu="RTX 4090", targets="8.9", compat_cuda="13.0").code == 804
    assert compat.check("13.0", "535.104.05", gpu="H100", targets="9.0", compat_cuda="13.0",
                        compat_branch_ok=False).code == 803


def test_forward_compat_only_over_the_listed_kernel_driver_branches():
    assert 535 in compat.COMPAT_BRANCHES["13.0"] and 560 not in compat.COMPAT_BRANCHES["13.0"]
    assert compat.check("13.0", "550.127.05", gpu="H100", targets="9.0", compat_cuda="13.0").ok
    assert compat.check("13.0", "560.35.03", gpu="H100", targets="9.0", compat_cuda="13.0").code == 803
    untabulated = compat.check("13.2", "535.183.01", gpu="H100", targets="9.0", compat_cuda="13.2")
    assert untabulated.ok and any("verify" in r for r in untabulated.reasons)
    native = compat.check("13.0", "580.65.06", gpu="H100", targets="9.0", compat_cuda="13.0")
    assert native.ok and any("not needed" in r for r in native.reasons)


def test_no_kernel_image_for_a_new_architecture():
    wheel = "8.0 8.6 9.0"
    assert compat.check("12.8", "570.86.10", gpu="RTX 5090", targets=wheel).code == 209
    assert compat.check("12.8", "570.86.10", gpu="T4", targets=wheel).code == 209           # no sm_75
    jit = compat.check("12.8", "570.86.10", gpu="RTX 5090", targets=wheel + "+PTX")
    assert jit.ok and any("JIT" in r for r in jit.reasons)


def test_driver_older_than_the_gpu_sees_no_device():
    assert compat.check("12.8", "550.54.15", gpu="B200", targets="10.0").code == 100


def test_cuda_11_era_drivers_are_judged_not_mistaken_for_no_device():
    assert compat.check("11.0", "450.80.02", gpu="A100", targets="8.0").ok
    assert compat.check("11.2", "460.32.03", gpu="T4", targets="7.5").ok
    v = compat.check("11.8", "450.80.02", gpu="A100", targets="8.0")          # CUDA 11.x minor-version floor
    assert v.ok and any("minor-version" in r for r in v.reasons)
    assert compat.check("11.8", "450.51.05", gpu="A100", targets="8.0").code == 35   # below the 11.x floor


def test_outside_the_table_the_engine_says_so():
    v = compat.check("9.0", "375.26", gpu="V100", targets="7.0")
    assert v.ok is None and v.code is None and str(v).startswith("CANNOT JUDGE")
    inferred = compat.check("12.8", "615.10", gpu="H100", targets="9.0")
    assert inferred.ok and "inferred" in inferred.reasons[0] and "13.4" in compat.INFERRED


def test_container_failure_modes_and_origins():
    assert compat.explain_container("12.4", "535.104.05", "L4", "8.9", gpus_injected=False).code == 100
    assert compat.explain_container("12.4", "535.104.05", "L4", "8.9", libcuda_in_image=True).code == 803
    assert compat.explain_container("12.4", "550.54.15", "L4", "8.9").ok
    assert compat.origin("/usr/lib/x86_64-linux-gnu/libcuda.so.580.65.06").startswith("host")
    assert compat.origin("/dev/nvidia0").startswith("host") and compat.origin("libnvidia-ml.so.1").startswith("host")
    assert compat.origin("libcudart.so.12") == "image" and compat.origin("libnccl.so.2") == "image"
    assert compat.origin("/usr/local/cuda/compat/libcuda.so.1").startswith("image")
