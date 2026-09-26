"""Probes are harmless and judged correctly: the naive executor leaks, the sandbox contains, and the
bundled Docker verdicts are labelled as illustrative sample output in the same format."""
import sys

import pytest

from sandboxlab import env
from sandboxlab.probes import (BY_NAME, CONTAINED, LEAKED, NA, PROBES, load_sample_runs, run_probe, run_suite,
                               sample_label, standin_host, unsandboxed_for, verdict_table)
from sandboxlab.process import Budgets, ProcessSandbox

LINUX = sys.platform.startswith("linux")
B = Budgets(cpu_s=1, wall_s=2)


def test_every_probe_is_documented():
    assert len(PROBES) == len(BY_NAME) == 14
    for p in PROBES:
        assert p.attack and p.stopped_by and p.risk.startswith("ASI")
        assert "P[" in p.body or p.name in ("infinite_loop", "sleep_forever")   # parameters, not hard-coded values
        assert "169.254.169.254" not in p.body                                 # the real metadata address only via P


def test_core_probe_name_map_covers_every_lab_probe():
    from sandboxlab.probes import CORE_PROBE_NAMES
    assert set(CORE_PROBE_NAMES) == set(BY_NAME)
    mapped = [v for v in CORE_PROBE_NAMES.values() if v]
    assert len(mapped) == len(set(mapped)) == 9          # all nine core probes have a lab counterpart
    try:                                                 # the lab never imports the core; check it only if present
        from sandboxcore import PROBES_BY_NAME as CORE
    except ImportError:
        pytest.skip("sandboxcore not installed")
    assert set(mapped) == set(CORE)


def test_standin_host_holds_only_canaries():
    with standin_host() as h:
        assert h.canary.startswith("LABCANARY-") and h.canary in h.key_path.read_text()
        assert oct(h.root.stat().st_mode)[-3:] == "700"
        assert h.agent_env()["LAB_CANARY_API_KEY"] == h.canary
        root = h.root
    assert not root.exists()                                                  # cleaned up


@pytest.mark.skipif(not LINUX, reason="Linux")
def test_naive_executor_leaks_what_the_sandbox_contains():
    names = ["env_secret", "egress", "huge_output"]
    with standin_host() as h:
        naive = {r.probe: r.verdict for r in run_suite(unsandboxed_for(h), h, names, budgets=B)}
        sand = {r.probe: r.verdict for r in run_suite(ProcessSandbox(B), h, names)}
    assert naive == {"env_secret": LEAKED, "egress": LEAKED, "huge_output": LEAKED}
    assert sand["env_secret"] == CONTAINED and sand["huge_output"] == CONTAINED
    assert sand["egress"] == (CONTAINED if env.netns_mode() else LEAKED)       # rlimits never stop the network


@pytest.mark.skipif(not (LINUX and env.is_root()), reason="the dedicated-UID verdicts need root")
def test_dedicated_uid_contains_file_and_process_probes():
    with standin_host() as h:
        res = {r.probe: r for r in run_suite(ProcessSandbox(B), h, ["ssh_key", "proc_environ", "write_outside",
                                                                     "fork_bomb", "escape_session"])}
    assert all(r.verdict == CONTAINED for r in res.values()), {k: (v.verdict, v.evidence) for k, v in res.items()}


@pytest.mark.skipif(not LINUX, reason="Linux")
def test_per_file_limits_miss_many_small_files():
    with standin_host() as h:
        one = run_probe(ProcessSandbox(B, netns=False), "disk_fill", h)
        many = run_probe(ProcessSandbox(B, netns=False), "disk_fill_many", h)
    assert one.verdict == CONTAINED and "File too large" in one.evidence
    assert many.verdict == LEAKED                                             # RLIMIT_FSIZE is per file


def test_sample_verdicts_are_labelled_and_complete():
    runs = load_sample_runs()
    assert sample_label() == "sample output in the documented format (illustrative)"
    assert set(runs) == {"docker:default", "docker:runc", "docker:runsc"}
    for iso, rows in runs.items():
        assert [r.probe for r in rows] == [p.name for p in PROBES]
        assert all(r.label == sample_label() and r.verdict in (LEAKED, CONTAINED, NA) for r in rows)
    assert sum(r.verdict == LEAKED for r in runs["docker:runc"]) == 0
    assert {r.probe for r in runs["docker:default"] if r.verdict == LEAKED} >= {"egress", "fork_bomb", "memory_hog"}
    table = verdict_table(runs)
    assert table.splitlines()[-1].split()[0] == "LEAKED"


def test_metadata_probe_targets_a_standin_unless_opted_in(monkeypatch):
    # The harness must not touch the real metadata server (node and pod credentials on a cloud VM) unasked.
    import socket
    real_connect = socket.create_connection

    def guard(addr, *a, **k):
        assert addr[0] != "169.254.169.254", "the real metadata server was contacted"
        return real_connect(addr, *a, **k)

    monkeypatch.delenv("SANDBOXLAB_PROBE_REAL_METADATA", raising=False)
    monkeypatch.setattr(socket, "create_connection", guard)
    with standin_host() as h:
        params = h.params("metadata", B)
        assert params["metadata_standin"] is True and params["metadata_hosts"] == ["127.0.0.1"]
        assert params["metadata_port"] == h.metadata_listener.port
        v = run_probe(unsandboxed_for(h), "metadata", h, B)
    assert v.verdict == LEAKED and "stand-in" in v.evidence            # unsandboxed: the stand-in is reachable


@pytest.mark.skipif(not LINUX, reason="Linux")
def test_ssh_key_probe_never_reads_a_real_home(tmp_path):
    # An executor that forgets to point HOME at the stand-in (here: HOME is a planted "real home")
    # must not make the probe read that home's key; only the stand-in key path is ever opened.
    from sandboxlab.probes import probe_code
    from sandboxlab.process import Unsandboxed
    real = tmp_path / "real-home"
    (real / ".ssh").mkdir(parents=True)
    (real / ".ssh" / "id_ed25519").write_text("REAL-PRIVATE-KEY")
    with standin_host() as h:
        code = probe_code(BY_NAME["ssh_key"], h.params("ssh_key", B))
        res = Unsandboxed(env={"PATH": "/usr/bin:/bin", "HOME": str(real)}, cwd=str(tmp_path),
                          harness_timeout_s=3).run(code, B)
        naive = run_probe(unsandboxed_for(h), "ssh_key", h, B)
    assert "REAL-PRIVATE-KEY" not in res.stdout and h.canary in res.stdout   # the absolute stand-in path still leaks
    assert naive.verdict == LEAKED                                            # HOME = the stand-in home: still leaks
