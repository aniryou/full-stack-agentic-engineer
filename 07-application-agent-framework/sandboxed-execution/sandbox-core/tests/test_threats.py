"""The attack probes: what a process sandbox contains, and the one thing it cannot (egress)."""
import sys

import pytest

from sandboxcore import PROBES, ProcessSandbox, UnsafeExecutor, run_probe, running_as_root

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimits only")


def test_probes_are_harmless_by_construction():
    # every probe program is short and self-contained; none names a real host or a real secret.
    for p in PROBES:
        assert "attacker" not in p.code.lower()
        assert p.contained_as in {"ok", "cpu_time", "wall_timeout", "file_too_large", "pids"}


def test_process_sandbox_contains_everything_except_egress():
    sb = ProcessSandbox()
    verdicts = {p.name: run_probe(p, sb) for p in PROBES}
    # the one it cannot stop:
    assert verdicts["egress_connect"].contained is False
    assert verdicts["egress_connect"].leaked is True     # it reached the harness's loopback trap
    # everything else is contained (fork bomb only if we can drop UID for RLIMIT_NPROC):
    for name, v in verdicts.items():
        if name == "egress_connect":
            continue
        if name == "fork_bomb" and not running_as_root():
            # cannot drop to nobody without privilege on this host; skip the NPROC assertion
            continue
        assert v.contained, f"{name}: {v.detail}"
        assert not v.leaked, f"{name} leaked: {v.detail}"


def test_secrets_leak_without_a_sandbox():
    # the contrast that motivates the sandbox: run the leak probes with NO isolation.
    ue = UnsafeExecutor()
    for name in ("read_env_secret", "read_ssh_key", "egress_connect"):
        p = next(p for p in PROBES if p.name == name)
        v = run_probe(p, ue)
        assert v.leaked is True, f"{name} should leak unsandboxed"


def test_resource_probes_marked_unsafe_to_run_unsandboxed():
    unsafe = {p.name for p in PROBES if not p.safe_unsandboxed}
    assert unsafe == {"fork_bomb", "disk_fill", "cpu_spin"}
