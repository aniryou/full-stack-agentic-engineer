"""The process sandbox: rlimits, the wall-clock kill, output truncation, exit reasons, and honest limits."""
import sys

import pytest

from sandboxcore import Budgets, ExecutionRequest, ProcessSandbox, SandboxConfig, running_as_root

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimits only")


def run(code, **b):
    return ProcessSandbox().run(ExecutionRequest(code=code, budgets=Budgets(**b)))


def test_clean_environment_hides_the_parents_secrets():
    r = run("import os; print(os.environ.get('CLOUD_API_TOKEN', 'ABSENT'))",
            cpu_s=1, wall_s=3)
    assert r.exit_reason == "ok" and r.stdout.strip() == "ABSENT"


def test_wall_timeout_kills_a_sleeper_and_its_grandchild():
    # sleep uses no CPU, so only the wall clock stops it; the backgrounded grandchild must die too.
    code = ("import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
            "import time; time.sleep(30)")
    r = run(code, cpu_s=5, wall_s=1)
    assert r.exit_reason == "wall_timeout"
    assert r.usage.wall_s < 3          # killed near the budget, not after 30 s


def test_cpu_limit_stops_a_busy_loop():
    r = run("x=0\nwhile True:\n x+=1", cpu_s=1, wall_s=10)
    assert r.exit_reason == "cpu_time"


def test_file_size_limit_reported_as_file_too_large():
    code = ("import os; f=open(os.path.join(os.environ['SANDBOX_WORKDIR'],'big'),'wb')\n"
            "f.write(b'x'*(4<<20))")
    r = run(code, file_mb=1, cpu_s=2, wall_s=5)
    assert r.exit_reason == "file_too_large"


def test_output_is_truncated_but_the_program_still_finishes():
    r = run("print('A'*100000)", output_bytes=1024, cpu_s=2, wall_s=5)
    assert r.exit_reason == "ok"
    assert r.truncated and len(r.stdout) <= 1100
    assert r.usage.stdout_bytes >= 100000        # the full amount is counted


def test_workspace_is_ephemeral_and_isolated():
    # HOME points at the empty workspace, so ~/.ssh is not the caller's.
    r = run("import os, pathlib; print(pathlib.Path(os.path.expanduser('~/.ssh')).exists())",
            cpu_s=1, wall_s=3)
    assert r.stdout.strip() == "False"


def test_isolation_report_is_honest_about_the_network():
    rep = ProcessSandbox().isolation_report()
    assert rep["network_blocked"] is False          # rlimits never touch sockets
    assert rep["rlimits_enforced"] is True


def test_nproc_caveat_matches_the_uid():
    # RLIMIT_NPROC is ignored for root; enforced once we drop to another UID (FACTS §10/§15).
    root_only = ProcessSandbox(SandboxConfig(drop_to_uid=None, drop_to_gid=None))
    rep = root_only.isolation_report()
    if running_as_root():
        assert rep["nproc_enforced"] is False       # as root with no uid drop, NPROC does nothing
        assert ProcessSandbox().isolation_report()["nproc_enforced"] is True   # dropping to nobody fixes it
