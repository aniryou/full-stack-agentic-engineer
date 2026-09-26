"""The attack probes: what a process sandbox contains, what it needs a UID for, and what it cannot (egress)."""
import ipaddress
import re
import sys

import pytest

from sandboxcore import PROBES, PROBES_BY_NAME, ProcessSandbox, SandboxConfig, UnsafeExecutor, run_probe, running_as_root

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimits only")
UID_DEPENDENT = {"read_ssh_key", "fork_bomb", "escape_session"}   # contained only with a per-execution UID


def test_probes_are_harmless_by_construction():
    for p in PROBES:
        # no hard-coded destination: every target comes from a harness-written workspace file
        for literal in re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", p.code):
            assert ipaddress.ip_address(literal).is_loopback, f"{p.name} names {literal}"
        assert "http://" not in p.code and "https://" not in p.code, p.name
        assert "expanduser('~')" not in p.code.replace(" ", "")      # never the bare real home
        assert p.contained_as in {"ok", "cpu_time", "wall_timeout", "file_too_large", "pids"}
    # every loop that could run away unsandboxed is bounded in the program itself...
    assert "while n < 500" in PROBES_BY_NAME["fork_bomb"].code
    assert "range(4096)" in PROBES_BY_NAME["disk_fill"].code
    assert "range(50_000)" in PROBES_BY_NAME["output_flood"].code
    assert "time.sleep(1.0)" in PROBES_BY_NAME["escape_session"].code
    # ...and the ones that are not (spin, sleep) are cut by the executor's wall clock; the resource grabs
    # are flagged so a notebook never runs them outside a sandbox
    unsafe = {p.name for p in PROBES if not p.safe_unsandboxed}
    assert unsafe == {"fork_bomb", "disk_fill", "cpu_spin"}


def test_process_sandbox_with_its_own_uid_contains_everything_except_egress():
    sb = ProcessSandbox()
    verdicts = {p.name: run_probe(p, sb) for p in PROBES}
    assert verdicts["egress_connect"].contained is False
    assert verdicts["egress_connect"].leaked is True     # it reached the harness's loopback trap
    dropped = sb.isolation_report()["uid_dropped"]
    for name, v in verdicts.items():
        if name == "egress_connect" or (name in UID_DEPENDENT and not dropped):
            continue                                     # not root: see the next test for the honest verdict
        assert v.contained, f"{name}: {v.detail}"
        assert not v.leaked, f"{name} leaked: {v.detail}"


def test_without_a_uid_drop_the_key_and_the_escape_get_through():
    # What a non-root laptop gets (and root with drop_to_uid=None): HOME points at the workspace, but the
    # key opens by absolute path, and a setsid() child outlives the call.
    sb = ProcessSandbox(SandboxConfig(drop_to_uid=None, drop_to_gid=None))
    assert sb.isolation_report()["filesystem_isolated"] is False
    for name in ("read_ssh_key", "escape_session"):
        v = run_probe(PROBES_BY_NAME[name], sb)
        assert v.leaked and not v.contained, f"{name}: {v.detail}"
    for name in ("read_env_secret", "cpu_spin", "sleep_forever", "output_flood"):
        assert run_probe(PROBES_BY_NAME[name], sb).contained, name     # these never needed a UID


def test_secrets_leak_without_a_sandbox():
    # the contrast that motivates the sandbox: run the leak probes with NO isolation.
    ue = UnsafeExecutor()
    for name in ("read_env_secret", "read_ssh_key", "egress_connect", "escape_session"):
        v = run_probe(PROBES_BY_NAME[name], ue)
        assert v.leaked is True, f"{name} should leak unsandboxed"


@pytest.mark.skipif(not running_as_root(), reason="needs root to switch to a per-execution UID")
def test_fork_bomb_is_stopped_by_nproc_only_off_root():
    assert run_probe(PROBES_BY_NAME["fork_bomb"], ProcessSandbox()).exit_reason == "pids"
