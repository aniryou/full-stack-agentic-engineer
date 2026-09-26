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
        assert rep["pids_scope"] == "none" and per_exec["pids_scope"] == "uid"
    else:
        assert rep["uid_dropped"] is False and rep["nproc_shared"] is True
        assert rep["pids_scope"] == ("tree" if os.path.isdir("/proc/self") else "none")


def test_workspace_permissions_constant():
    # mkdtemp's 0700 is what the sandbox relies on; guard against a platform that differs
    d = tempfile.mkdtemp()
    try:
        assert stat.S_IMODE(os.stat(d).st_mode) == 0o700
    finally:
        os.rmdir(d)


def test_an_interpreter_behind_a_private_directory_is_not_world_executable():
    # A venv's bin/python is a symlink to a system interpreter, but a sandbox UID starts it by the link's
    # path: a venv under a 0700 directory (/root, a private scratch dir) cannot be used by it.
    from sandboxcore import executor
    base = tempfile.mkdtemp()
    try:
        os.chmod(base, 0o755)
        target = os.path.join(base, "python3")
        with open(target, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(target, 0o755)
        if not executor._world_executable(target):
            pytest.skip("the temp directory itself is not world-traversable here")
        private = os.path.join(base, "private")
        os.mkdir(private, 0o700)
        link = os.path.join(private, "python")
        os.symlink(target, link)
        assert executor._world_executable(link) is False
        os.chmod(private, 0o755)
        assert executor._world_executable(link) is True
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)


@needs_root
def test_a_venv_the_sandbox_uid_cannot_reach_falls_back_to_a_system_python():
    base = tempfile.mkdtemp()                                   # 0700: like /root or a private scratch dir
    link = os.path.join(base, "python")
    os.symlink(os.path.realpath(sys.executable), link)
    try:
        r = run("print('hello')", SandboxConfig(python=link), wall_s=5)
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)
    if r.notes and "no world-executable python3" in r.notes[0]:
        assert r.exit_reason == "ok" and r.isolation["uid_dropped"] is False   # ran as us, and said so
    else:
        assert r.exit_reason == "ok" and r.stdout.strip() == "hello", (r.exit_reason, r.stderr)


# ---- the process budget without a UID switch: the run's own tree, not everything the UID holds ----------
needs_proc = pytest.mark.skipif(not os.path.isdir("/proc/self") or not hasattr(resource, "RLIMIT_NPROC"),
                                reason="counts tasks from /proc (Linux)")
THREAD_HOG = ("import sys, threading, time\n"          # another process of the same UID holding many threads,
              "for _ in range(int(sys.argv[1])):\n"      # like a CI agent or a browser
              "    threading.Thread(target=time.sleep, args=(120,), daemon=True).start()\n"
              "print('up', flush=True)\n"
              "time.sleep(120)\n")
AS_ME_SCENARIOS = r'''
import json, subprocess, sys
from sandboxcore import Budgets, ExecutionRequest, ProcessSandbox, SandboxConfig
hog = subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]], stdout=subprocess.PIPE, text=True)
try:
    assert hog.stdout.readline().strip() == "up"
    sb = ProcessSandbox(SandboxConfig(drop_to_uid=None, drop_to_gid=None))
    out = [sb.isolation_report().get("pids_scope")]
    for code, budgets in json.loads(sys.argv[3]):
        r = sb.run(ExecutionRequest(code=code, budgets=Budgets(**budgets)))
        out.append([r.exit_reason, r.reason_source, r.usage.wall_s, r.stderr[-300:]])
finally:
    hog.kill()
print(json.dumps(out))
'''


def run_as_an_unprivileged_user(scenarios, hog_threads):
    """Run ``scenarios`` through ``ProcessSandbox`` without a UID switch, as a non-root user, while another
    process of that same user holds ``hog_threads`` threads. As root the helper runs as a free UID from the
    per-execution range (what CI's unprivileged runner user is); otherwise as us."""
    import json
    import shutil
    import subprocess
    from pathlib import Path

    from sandboxcore import executor
    base = tempfile.mkdtemp()
    kwargs, uid = {}, None
    try:
        os.chmod(base, 0o755)
        shutil.copytree(Path(executor.__file__).parent, os.path.join(base, "sandboxcore"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        for root_dir, dirs, files in os.walk(base):
            for n in dirs + files:
                os.chmod(os.path.join(root_dir, n), 0o755)
        python = sys.executable
        if running_as_root():
            python = ProcessSandbox()._python(True)
            if python is None or not executor._world_executable(base):
                pytest.skip("no interpreter or temp directory an unprivileged UID can reach")
            uid = executor._next_uid()
            kwargs = {"user": uid, "group": uid, "extra_groups": []}
        p = subprocess.run([python, "-I", "-c", f"import sys; sys.path.insert(0, {base!r})\n" + AS_ME_SCENARIOS,
                            THREAD_HOG, str(hog_threads), json.dumps(scenarios)],
                           cwd=base, capture_output=True, text=True, timeout=60, **kwargs)
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout)
    finally:
        if uid is not None:
            executor._kill_uid(uid)
        shutil.rmtree(base, ignore_errors=True)


SLEEPER_AND_GRANDCHILD = ("import subprocess, sys; "
                          "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                          "import time; time.sleep(30)")
SETSID_ESCAPE = ("import os, time\n"
                 "if os.fork() == 0:\n"
                 "    os.setsid(); time.sleep(8); os._exit(0)\n"
                 "time.sleep(30)")


@needs_proc
def test_the_pid_budget_ignores_other_threads_of_the_same_user():
    # The CI failure: the unprivileged runner user holds dozens of threads (the runner agent), the old
    # headroom counted processes, not tasks, and the run's first fork failed with EAGAIN ('pids') long
    # before its wall clock ran out. 64 threads is four times the default budget.
    scope, *verdicts = run_as_an_unprivileged_user(
        [[SLEEPER_AND_GRANDCHILD, {"cpu_s": 5, "wall_s": 1}], [SETSID_ESCAPE, {"cpu_s": 2, "wall_s": 1}]],
        hog_threads=64)
    for reason, source, wall, stderr in verdicts:
        assert (reason, source) == ("wall_timeout", "parent"), stderr
        assert wall < 3
    assert scope == "tree"


@needs_proc
def test_the_pid_budget_is_the_process_trees_own():
    # Ten sleeping children against a budget of 6: the parent counts the run's tree (not the user's other
    # 64 threads, which would be over any small budget) and ends the run as pids before its wall clock.
    ten_children = ("import subprocess, sys, time\n"
                    "kids = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']) "
                    "for _ in range(10)]\n"
                    "time.sleep(30)")
    one_task = "import time; time.sleep(30)"
    scope, over, under = run_as_an_unprivileged_user(
        [[ten_children, {"pids": 6, "wall_s": 4, "cpu_s": 4}], [one_task, {"pids": 2, "wall_s": 1}]],
        hog_threads=64)
    assert over[:2] == ["pids", "parent"] and over[2] < 4, over
    assert under[:2] == ["wall_timeout", "parent"], under
    assert scope == "tree"


@needs_proc
def test_the_tree_count_covers_threads_the_group_and_a_setsid_child():
    import subprocess
    from sandboxcore import executor
    code = ("import os, threading, time\n"
            "for _ in range(5): threading.Thread(target=time.sleep, args=(30,), daemon=True).start()\n"
            "if os.fork() == 0:\n"
            "    os.setsid(); print('up', flush=True); time.sleep(30); os._exit(0)\n"
            "time.sleep(30)")
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, start_new_session=True)
    escapee = []
    try:
        assert p.stdout.readline().strip() == "up"
        escapee = [q for q in executor._uid_pids(os.getuid()) or [] if (executor._proc_stat(str(q)) or [0])[0] == p.pid]
        assert len(escapee) == 1 and executor._proc_stat(str(escapee[0]))[2] == escapee[0]    # its own session
        # the leader, its 5 threads and the setsid child — and nothing else of this UID, however busy
        assert executor._tree_tasks(p.pid) == 1 + 5 + 1
    finally:
        os.killpg(p.pid, 9)
        for q in escapee:
            os.kill(q, 9)                                     # it left the group: the group kill missed it
        p.wait()


def test_parent_verdicts_have_a_fixed_order():
    from sandboxcore.executor import PARENT_VERDICT_ORDER, _parent_verdict
    assert PARENT_VERDICT_ORDER == ("wall_timeout", "pids", "output_limit")
    assert _parent_verdict(True, True, True) == "wall_timeout"     # past the deadline the run is over
    assert _parent_verdict(False, True, True) == "pids"
    assert _parent_verdict(False, False, True) == "output_limit"
    assert _parent_verdict(False, False, False) is None
