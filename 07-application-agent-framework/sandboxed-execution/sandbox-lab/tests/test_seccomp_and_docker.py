"""The derived seccomp profile removes exactly what it says; the hardened docker run carries every flag,
the shell script and the Python builder agree, and the naive spec is naive."""
import json
import re
from pathlib import Path

from sandboxlab import seccomp
from sandboxlab.docker import FLAGS, DockerSandbox, explain

ROOT = Path(__file__).resolve().parents[1]


def test_derived_profile_is_docker_default_minus_the_documented_removals():
    base, prof = seccomp.load_base(), seccomp.sandbox_profile()
    assert prof["defaultAction"] == base["defaultAction"] == "SCMP_ACT_ERRNO" and prof["archMap"] == base["archMap"]
    gone = seccomp.allowed_unconditionally(base) - seccomp.allowed_unconditionally(prof)
    assert gone == set(seccomp.REMOVED) == {"memfd_create", "memfd_secret", "execveat", "socketcall"}
    assert seccomp.allowed_unconditionally(prof) <= seccomp.allowed_unconditionally(base)   # nothing added
    assert len(seccomp.allowed_unconditionally(base)) == 361
    assert not any(set(r["names"]) == seccomp.REMOVED_RULE for r in prof["syscalls"])


def test_socket_families():
    base, prof = seccomp.load_base(), seccomp.sandbox_profile()
    assert seccomp.socket_allowed(base, "AF_NETLINK") and seccomp.socket_allowed(base, "AF_PACKET")
    assert seccomp.socket_families(prof) == {1, 2, 10}
    assert not seccomp.socket_allowed(prof, "AF_NETLINK") and seccomp.socket_allowed(prof, "AF_INET6")


def test_generated_profile_file_is_current():
    on_disk = json.loads((ROOT / "deploy/docker/seccomp-sandbox.json").read_text())
    assert on_disk == seccomp.sandbox_profile()


def test_hardened_argv_has_every_control():
    a = DockerSandbox.hardened().argv()
    s = " ".join(a)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                 "--security-opt seccomp=", "--pids-limit 64", "--memory 320m --memory-swap 320m", "--cpus 1",
                 "--user 65534:65534", "--tmpfs /work:rw,nosuid,nodev,noexec,size=16m", "--init", "--rm"):
        assert flag in s, flag
    assert "--runtime" not in a and "--nproc" not in a                     # runc: NPROC would count host-wide
    g = DockerSandbox.hardened("runsc").argv()
    assert g[g.index("--runtime") + 1] == "runsc" and "--nproc" in g        # inside gVisor it counts the sandbox
    assert a[a.index("--budgets") + 1] == json.dumps(json.loads(a[a.index("--budgets") + 1]), separators=(",", ":"))


def test_naive_spec_is_the_contrast():
    a = " ".join(DockerSandbox.naive().argv("print(1)"))
    for flag in ("--read-only", "--cap-drop", "--pids-limit", "--memory", "--user", "seccomp=", "--network none"):
        assert flag not in a
    assert a.endswith("python:3.12-slim python3 -c print(1)")
    assert DockerSandbox.naive().name == "docker:default" and DockerSandbox.hardened("runsc").name == "docker:runsc"


def test_shell_script_and_builder_agree():
    script = (ROOT / "deploy/docker/run-hardened.sh").read_text()
    py = DockerSandbox.hardened().argv()
    for i, tok in enumerate(py):
        if tok.startswith("--") and tok not in ("--label", "--volume", "--budgets", "--workspace", "--stdin", "--env"):
            val = py[i + 1] if i + 1 < len(py) and not py[i + 1].startswith("--") else ""
            if tok == "--security-opt" and val.startswith("seccomp="):
                val = "seccomp="
            assert tok in script and val.split("=")[0] in script, (tok, val)


def test_flag_table_names_what_each_flag_stops():
    assert len(FLAGS) >= 12 and all(stops for _, _, stops in FLAGS)
    assert "--runtime runsc" in explain() and re.search(r"--pids-limit 64 .*fork_bomb", explain())
