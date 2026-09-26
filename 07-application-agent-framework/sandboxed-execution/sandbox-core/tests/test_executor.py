"""The process sandbox: rlimits, the wall-clock kill, streamed output, exit reasons, and honest limits."""
import os
import resource
import stat
import sys
import tempfile
import time

import pytest

from sandboxcore import (Budgets, ExecutionRequest, ProcessSandbox, SandboxConfig, rlimits_for,
                         running_as_root)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX rlimits only")
needs_root = pytest.mark.skipif(not running_as_root(), reason="a per-execution UID needs root to switch to")
AS_ME = SandboxConfig(drop_to_uid=None, drop_to_gid=None)     # what a non-root laptop gets


def run(code, config=None, **b):
    return ProcessSandbox(config).run(ExecutionRequest(code=code, budgets=Budgets(**b)))


def test_clean_environment_hides_the_parents_secrets():
    os.environ["CLOUD_API_TOKEN"] = "sk-planted-in-the-parent"
    try:
        r = run("import os; print(os.environ.get('CLOUD_API_TOKEN', 'ABSENT'))", cpu_s=1, wall_s=3)
    finally:
        del os.environ["CLOUD_API_TOKEN"]
    assert r.exit_reason == "ok" and r.stdout.strip() == "ABSENT"


def test_wall_timeout_kills_a_sleeper_and_its_grandchild():
    # sleep uses no CPU, so only the wall clock stops it; the backgrounded grandchild must die too.
    code = ("import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
            "import time; time.sleep(30)")
    r = run(code, cpu_s=5, wall_s=1)
    assert r.exit_reason == "wall_timeout" and r.reason_source == "parent"
    assert r.usage.wall_s < 3          # killed near the budget, not after 30 s


def test_cpu_limit_stops_a_busy_loop():
    r = run("x=0\nwhile True:\n x+=1", cpu_s=1, wall_s=10)
    assert r.exit_reason == "cpu_time"
    assert r.usage.cpu_s >= 0.9        # measured by the parent (wait4), not reported by the code


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


def test_an_output_flood_is_streamed_killed_and_never_buffered():
    # >100 MB/s of output: the parent must keep only output_bytes, stop at output_kill_bytes, and not
    # grow its own memory by the flood (the old communicate() held it all).
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.monotonic()
    r = run("import sys\nb = b'x' * (1 << 20)\nwhile True:\n    sys.stdout.buffer.write(b)",
            output_bytes=1024, output_kill_bytes=8 << 20, wall_s=3, cpu_s=3)
    elapsed = time.monotonic() - t0
    grew_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / (1024 if sys.platform != "darwin" else 2**20)
    assert r.exit_reason == "output_limit" and r.reason_source == "parent"
    assert len(r.stdout) <= 1024 and r.usage.stdout_bytes > 8 << 20
    assert elapsed < 3 + 1                       # within wall_s + 1 s whatever happens
    assert grew_mb < 50


def test_a_process_that_leaves_the_group_cannot_hold_the_call_open():
    # setsid() escapes the process-group kill; the call must still return near its deadline.
    code = ("import os, time\n"
            "if os.fork() == 0:\n"
            "    os.setsid(); time.sleep(8); os._exit(0)\n"
            "time.sleep(30)")
    t0 = time.monotonic()
    r = run(code, wall_s=1, cpu_s=2)
    assert r.exit_reason == "wall_timeout"
    assert time.monotonic() - t0 < 1 + 1


@needs_root
def test_per_execution_uid_sweeps_escapees_and_leftover_files():
    marker = tempfile.mkdtemp(prefix="sweep-test-")
    os.chmod(marker, 0o1777)
    code = ("import os, time\n"
            f"open('{marker}/left-behind', 'w').write('x')\n"
            "if os.fork() == 0:\n"
            "    os.setsid()\n"
            "    fd = os.open(os.devnull, os.O_RDWR)\n"
            "    for i in (0, 1, 2): os.dup2(fd, i)\n"
            f"    time.sleep(1); open('{marker}/escaped', 'w').write('x'); os._exit(0)\n"
            "print('bye')")
    try:
        r = run(code, wall_s=3)
        time.sleep(1.5)
        assert r.exit_reason == "ok"
        assert r.swept["processes"] >= 1
        assert not os.path.exists(f"{marker}/escaped")          # the escapee was killed before it wrote
        assert not os.path.exists(f"{marker}/left-behind")      # the file it left outside was removed
    finally:
        import shutil
        shutil.rmtree(marker, ignore_errors=True)


@needs_root
@pytest.mark.skipif(not hasattr(os, "waitid") or not os.path.isdir("/proc/self"), reason="needs waitid and /proc")
def test_a_uid_still_holding_a_zombie_is_not_reused(monkeypatch):
    # zombies count toward RLIMIT_NPROC, so a UID with one would start the next run short of processes
    import itertools
    import subprocess
    from sandboxcore import executor
    monkeypatch.setattr(executor, "_uid_counter", itertools.count(0))
    uid = executor._next_uid()
    p = subprocess.Popen(["true"], user=uid, group=uid, extra_groups=[])
    os.waitid(os.P_PID, p.pid, os.WEXITED | os.WNOWAIT)          # exited, not reaped: a zombie of uid
    try:
        assert executor._uid_pids(uid) == [] and executor._uid_pids(uid, zombies=True) == [p.pid]
        monkeypatch.setattr(executor, "_uid_counter", itertools.count(0))
        assert executor._next_uid() != uid
    finally:
        p.wait()


@needs_root
def test_a_different_uid_is_what_protects_your_files():
    victim = tempfile.mkdtemp(prefix="victim-")                # 0700, owned by us
    key = os.path.join(victim, "id_ed25519")
    with open(key, "w") as f:
        f.write("STANDIN-KEY")
    code = f"print(open({key!r}).read())"
    try:
        sandboxed = run(code, wall_s=3)
        as_me = run(code, AS_ME, wall_s=3)
    finally:
        import shutil
        shutil.rmtree(victim, ignore_errors=True)
    assert "STANDIN-KEY" not in sandboxed.stdout and "PermissionError" in sandboxed.stderr
    assert "STANDIN-KEY" in as_me.stdout          # HOME redirection alone does not stop an absolute path


def test_home_redirection_is_not_a_boundary():
    # With our own UID, HOME points at the workspace but a stand-in "home" still opens by absolute path.
    victim = tempfile.mkdtemp(prefix="victim-home-")
    key = os.path.join(victim, "id_ed25519")
    with open(key, "w") as f:
        f.write("STANDIN-KEY")
    try:
        r = run(f"import os; print(os.path.expanduser('~') != {victim!r}, open({key!r}).read())", AS_ME, wall_s=3)
    finally:
        import shutil
        shutil.rmtree(victim, ignore_errors=True)
    assert r.stdout.split() == ["True", "STANDIN-KEY"]
    assert ProcessSandbox(AS_ME).isolation_report()["filesystem_isolated"] is False


def test_workspace_is_private_and_ephemeral():
    r = run("import os; st = os.stat(os.environ['SANDBOX_WORKDIR']); print(oct(st.st_mode & 0o777), "
            "os.environ['SANDBOX_WORKDIR'])", wall_s=3)
    mode, path = r.stdout.split()
    assert mode == "0o700"                        # never 0777: other runs and users cannot read or plant
    assert not os.path.exists(path)               # deleted when the call returned


def test_exit_reasons_read_from_stderr_are_labelled_as_the_codes_claim():
    r = run("import sys; sys.stderr.write('MemoryError\\n'); sys.exit(1)", wall_s=3)
    assert r.exit_reason == "memory" and r.reason_source == "code"      # forged, and labelled so


def test_peak_memory_is_measured_per_execution():
    r = run("x = bytearray(100 * 2**20); print(len(x))", memory_mb=512, wall_s=5)
    assert r.exit_reason == "ok" and r.usage.max_rss_mb >= 100


@pytest.mark.skipif(not hasattr(os, "waitid") or not hasattr(os, "WNOWAIT"), reason="needs waitid(WNOWAIT)")
def test_reaping_the_leader_keeps_its_rusage():
    # Popen.poll() reaps with waitpid and drops the rusage; the executor's non-blocking wait4 must keep it.
    import subprocess
    from sandboxcore import executor
    p = subprocess.Popen([sys.executable, "-c", "x = bytearray(40 * 2**20)"])
    os.waitid(os.P_PID, p.pid, os.WEXITED | os.WNOWAIT)     # wait for the exit without reaping it
    exited, ru = executor._try_reap(p)
    assert exited and p.returncode == 0
    assert ru is not None and executor._rss_mb(ru) >= 40


def test_usage_survives_a_child_that_writes_until_it_exits():
    # the child exits while the parent is busy reading: its CPU and peak memory must still be recorded
    code = "import sys\nx = bytearray(50 * 2**20)\nfor i in range(3000):\n    sys.stdout.write('y' * 200 + '\\n')\n"
    for _ in range(12):
        r = run(code, wall_s=5)
        assert r.exit_reason == "ok" and r.usage.cpu_s > 0 and r.usage.max_rss_mb >= 50, r.usage


def test_rlimits_cpu_soft_below_hard():
    lims = {name: (soft, hard) for name, soft, hard in rlimits_for(Budgets(cpu_s=2), nproc=16)}
    assert lims["RLIMIT_CPU"] == (2, 3)
    assert lims["RLIMIT_NPROC"] == (16, 16)


def test_isolation_report_is_honest_about_the_network():
    rep = ProcessSandbox().isolation_report()
    assert rep["network_blocked"] is False          # rlimits never touch sockets
    assert rep["rlimits_enforced"] is True


def test_nproc_caveat_matches_the_uid():
    # RLIMIT_NPROC is ignored for root; enforced once we run as another UID (FACTS §10/§15).
    rep = ProcessSandbox(AS_ME).isolation_report()
    if running_as_root():
        assert rep["nproc_enforced"] is False       # as root with no uid drop, NPROC does nothing
        assert rep["limits_raisable_by_code"] is True
        per_exec = ProcessSandbox().isolation_report()
        assert per_exec["nproc_enforced"] is True and per_exec["nproc_shared"] is False
        assert per_exec["escapes_swept"] is (os.path.isdir("/proc/self"))
    else:
        assert rep["uid_dropped"] is False and rep["nproc_shared"] is True


def test_workspace_permissions_constant():
    # mkdtemp's 0700 is what the sandbox relies on; guard against a platform that differs
    d = tempfile.mkdtemp()
    try:
        assert stat.S_IMODE(os.stat(d).st_mode) == 0o700
    finally:
        os.rmdir(d)
