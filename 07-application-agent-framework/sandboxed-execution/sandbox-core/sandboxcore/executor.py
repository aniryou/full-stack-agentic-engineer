"""Run code in a child process with rlimits and a wall-clock kill — and say plainly what it cannot stop.

The one idea: in-process restrictions are not a boundary (the code shares your interpreter, so it can
undo them), so the first real boundary is a **separate process** you start clean and bound from outside.
This module has two runners on purpose:

* ``UnsafeExecutor`` — runs code in a child that inherits this process's environment and home. It exists so
  notebook 01 can *watch a secret leak* and *watch the network get reached*; never use it for real work.
* ``ProcessSandbox`` — the honest T0 sandbox: a clean environment, an ephemeral workspace (mode 0700),
  POSIX resource limits (CPU, address space, processes, file size, open files), a wall-clock deadline,
  output read **as it streams** (the first ``output_bytes`` kept, the rest counted, the run killed once the
  total passes ``output_kill_bytes``) and a kill of the whole **process group** at the end. When it runs as
  root it also gives every execution its **own unprivileged UID**, which is what makes three more controls
  real: ``RLIMIT_NPROC`` (per real UID, ignored for root), file access (a different UID cannot open your
  0700 home — pointing ``HOME`` elsewhere hides nothing), and a sweep after the run that kills any process
  of that UID (code can ``setsid()`` out of the process group) and removes files it left in ``/tmp``.

What it does not stop, stated in ``isolation_report()``: the **network** (rlimits never touch sockets), the
**host kernel** (every syscall reaches it), and — without root — your files and a ``setsid`` escape.

The limits are applied without ``preexec_fn``: ``Popen`` switches the UID and starts a new session in C
(``user=``, ``group=``, ``extra_groups=[]``, ``start_new_session=True``, all safe in a threaded parent),
and a short ``_LAUNCHER`` run by the child's own interpreter lowers the rlimits before it executes the code.
Lowering a limit needs no privilege and cannot be undone by an unprivileged process. The mechanisms and
their gotchas are in FACTS §10 and §15; latencies are measured on the host that runs them.
"""
from __future__ import annotations

import itertools
import json
import os
import platform
import resource
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .contract import Budgets, ExecutionRequest, ExecutionResult, Usage, digest

IS_POSIX = os.name == "posix"
IS_LINUX = sys.platform.startswith("linux")
PER_EXECUTION = "per-execution"      # SandboxConfig.drop_to_uid: a fresh UID for every run (when root)
UID_BASE, UID_SPAN = 62000, 997       # the per-execution UID range (the lab uses 61000-61996)
SWEEP_DIRS = ("/tmp", "/var/tmp", "/dev/shm")   # world-writable places a sandbox UID can persist files

# Runs in the child's own interpreter: lower the rlimits, set no_new_privs, then execute the code. The UID
# switch and the new session have already happened (in C, inside Popen), so nothing here needs privilege.
_LAUNCHER = """\
import json, resource, sys
for _name, _soft, _hard in json.loads(sys.argv[1]):
    _r = getattr(resource, _name, None)
    if _r is None:
        continue
    try:
        _cs, _ch = resource.getrlimit(_r)
        if _ch != resource.RLIM_INFINITY:
            _hard = min(_hard, _ch)
        resource.setrlimit(_r, (min(_soft, _hard), _hard))
    except (ValueError, OSError):
        pass
if sys.platform.startswith("linux"):
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0)   # PR_SET_NO_NEW_PRIVS
    except Exception:
        pass
_code = sys.argv[2]
sys.argv = ["-c"]
exec(compile(_code, "<sandbox>", "exec"), {"__name__": "__main__", "__builtins__": __builtins__})
"""

_uid_counter = itertools.count()


def running_as_root() -> bool:
    return IS_POSIX and hasattr(os, "geteuid") and os.geteuid() == 0


@dataclass
class SandboxConfig:
    """How the process sandbox is set up, separate from the per-call budgets."""
    # PER_EXECUTION: a fresh UID per run (needs root) — RLIMIT_NPROC, file isolation and the sweep all
    # depend on it. An int: that fixed UID (shared by concurrent runs, so no sweep). None: keep our UID.
    drop_to_uid: int | str | None = PER_EXECUTION
    drop_to_gid: int | str | None = PER_EXECUTION
    clean_env: bool = True            # start from an empty environment, not the parent's
    isolate_home: bool = True         # HOME -> the workspace: tidy, but NOT a boundary (a path still opens)
    python: str = sys.executable


def rlimits_for(budgets: Budgets, nproc: int | None) -> list[tuple[str, int, int]]:
    """The ``(resource name, soft, hard)`` limits the launcher applies, in order.

    CPU gets soft one below hard: the soft limit raises a catchable ``SIGXCPU`` first, and the hard limit
    guarantees a ``SIGKILL`` if the code ignores it (equal limits would be an immediate ``SIGKILL``).
    """
    mib = 1024 * 1024
    lims = [("RLIMIT_FSIZE", budgets.file_mb * mib, budgets.file_mb * mib),
            ("RLIMIT_AS", budgets.memory_mb * mib, budgets.memory_mb * mib),
            ("RLIMIT_NOFILE", budgets.open_files, budgets.open_files),
            ("RLIMIT_CORE", 0, 0),
            ("RLIMIT_CPU", max(int(budgets.cpu_s), 1), max(int(budgets.cpu_s), 1) + 1)]
    if nproc is not None:
        lims.append(("RLIMIT_NPROC", nproc, nproc))
    return lims


def _world_executable(path: str) -> bool:
    """Can an arbitrary UID run ``path`` (file world r-x, every parent directory world-traversable)?

    Both the path as given and the file it resolves to: a virtualenv's ``bin/python`` is a symlink to a
    system interpreter, but the sandbox UID starts it by the link's own path, so a venv under a 0700
    directory (``/root``, a private scratch dir) is unusable even though its target is world-executable.
    """
    given, real = Path(os.path.abspath(path)), Path(os.path.realpath(path))
    try:
        if (real.stat().st_mode & 0o005) != 0o005:
            return False
        return all(parent.stat().st_mode & 0o001 for parent in (*given.parents, *real.parents))
    except OSError:
        return False


def _uid_pids(uid: int, *, zombies: bool = False) -> list[int] | None:
    """Processes whose real UID is ``uid`` (live ones; zombies too if asked), from ``/proc``; None where
    there is no ``/proc``."""
    if not os.path.isdir("/proc/self"):
        return None
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/status") as f:
                fields = dict(line.split(":", 1) for line in f if ":" in line)
            if int(fields["Uid"].split()[0]) == uid and (zombies or fields["State"].split()[0] != "Z"):
                out.append(int(pid))
        except (OSError, KeyError, ValueError, IndexError):
            continue
    return out


def _next_uid() -> int:
    """A per-execution UID in [UID_BASE, UID_BASE + UID_SPAN) with no process at all right now.

    Zombies count: a killed escapee re-parented to a PID 1 that does not reap (some containers) stays a
    zombie of that UID, and zombies still count toward ``RLIMIT_NPROC`` — reusing its UID would start the
    next run with part of its process budget gone.
    """
    start = (os.getpid() * 7 + next(_uid_counter)) % UID_SPAN
    for i in range(UID_SPAN):
        uid = UID_BASE + (start + i) % UID_SPAN
        if not _uid_pids(uid, zombies=True):
            return uid
    return UID_BASE + start


def _kill_uid(uid: int) -> int:
    """SIGKILL every live process of ``uid`` (repeat: one may fork while we sweep); how many were killed."""
    killed: set[int] = set()
    for _ in range(5):
        pids = _uid_pids(uid) or []
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.add(pid)
            except OSError:
                pass
        time.sleep(0.02)
    return len(killed)


def _remove_uid_files(uid: int, skip: str) -> list[str]:
    """Remove what ``uid`` left in the world-writable directories (and one level into world-writable
    subdirectories). Best effort: a directory it could write deeper inside is not searched."""
    removed: list[str] = []

    def scan(d: str, depth: int) -> None:
        try:
            entries = list(os.scandir(d))
        except OSError:
            return
        for e in entries:
            if os.path.realpath(e.path) == os.path.realpath(skip):
                continue
            try:
                st = e.stat(follow_symlinks=False)
            except OSError:
                continue
            if st.st_uid == uid:
                try:
                    if stat.S_ISDIR(st.st_mode):
                        shutil.rmtree(e.path)
                    else:
                        os.unlink(e.path)
                    removed.append(e.path)
                except OSError:
                    pass
            elif depth == 0 and stat.S_ISDIR(st.st_mode) and st.st_mode & 0o002:
                scan(e.path, 1)

    for d in SWEEP_DIRS:
        if os.path.isdir(d):
            scan(d, 0)
    return removed


class _Truncator:
    """Fed chunk by chunk: keep the first ``keep`` bytes, count the rest (memory stays bounded)."""

    def __init__(self, keep: int):
        self.keep = keep
        self.kept = bytearray()
        self.total = 0

    def feed(self, data: bytes) -> None:
        self.total += len(data)
        if len(self.kept) < self.keep:
            self.kept += data[: self.keep - len(self.kept)]

    @property
    def truncated(self) -> bool:
        return self.total > self.keep

    def text(self) -> str:
        return self.kept.decode("utf-8", "replace")


class UnsafeExecutor:
    """Runs code with NO isolation, inheriting this process's env and home. For contrast only."""

    def run(self, req: ExecutionRequest) -> ExecutionResult:
        with tempfile.TemporaryDirectory(prefix="unsafe-") as work:
            _write_inputs(work, req.files)
            env = dict(os.environ)
            env.update(req.env)
            env["SANDBOX_WORKDIR"] = work
            return _spawn([sys.executable, "-c", req.code], env, work, req.budgets,
                          popen_kwargs={"start_new_session": True} if IS_POSIX else {},
                          isolation={"kind": "none",
                                     "note": "inherits the parent env, home, UID and network; not a boundary"})


class ProcessSandbox:
    """A child process with a clean env, an ephemeral workspace, rlimits, a wall-clock kill and — as root —
    its own UID per execution."""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()

    # ---- what this machine lets us enforce ------------------------------------------------------------
    def _uid_mode(self) -> str:
        if not running_as_root() or self.config.drop_to_uid is None:
            return "none"
        return "per-execution" if self.config.drop_to_uid == PER_EXECUTION else "fixed"

    def _python(self, dropping: bool) -> str | None:
        """An interpreter the sandbox UID can execute (this one, or a system python3)."""
        if not dropping or _world_executable(self.config.python):
            return self.config.python
        for cand in ("/usr/bin/python3", "/usr/local/bin/python3"):
            if os.path.exists(cand) and _world_executable(cand):
                return cand
        return None

    def isolation_report(self) -> dict:
        """What this sandbox can and cannot enforce on this machine, decided at run time."""
        root = running_as_root()
        mode = self._uid_mode()
        dropped = mode != "none" and self._python(True) is not None
        per_exec = dropped and mode == "per-execution"
        has_proc = os.path.isdir("/proc/self")
        return {
            "kind": "process",
            "os": platform.system(),
            "clean_env": self.config.clean_env,
            "isolate_home": self.config.isolate_home,     # HOME redirection only: a convenience
            "uid_dropped": dropped,
            "uid_mode": mode if dropped else "none",
            # Only a different UID stops the code opening your files by absolute path; HOME redirection does not.
            "filesystem_isolated": dropped,
            "rlimits_enforced": IS_POSIX,
            "as_limit_enforced": IS_LINUX,                # macOS accepts RLIMIT_AS but does not enforce it
            # RLIMIT_NPROC counts every process of the real UID and is ignored for uid 0.
            "nproc_enforced": bool(hasattr(resource, "RLIMIT_NPROC") and (dropped or (not root and has_proc))),
            "nproc_shared": bool(not dropped or mode == "fixed"),   # budget shared with other processes of that UID
            # setsid() leaves the process group; only a per-execution UID lets the sweep find the escapee.
            "escapes_swept": per_exec and has_proc,
            "limits_raisable_by_code": bool(root and not dropped),  # root can raise its own hard limits
            "network_blocked": False,                     # rlimits never touch sockets — see egress_connect
            "note": ("network egress is NOT stopped here; use an egress proxy with a network namespace, "
                     "container --network none or a default-deny NetworkPolicy. "
                     + ("Each execution runs as its own UID: files, process count and leftovers are bounded."
                        if per_exec else
                        "No UID drop here (not root, or disabled): the code runs as YOUR UID, so it can read "
                        "your files by absolute path, RLIMIT_NPROC is shared with your processes (or ignored "
                        "as root), a process that calls setsid() can outlive the call, and files it writes "
                        "outside the workspace (e.g. in /tmp) stay behind.")),
        }

    def run(self, req: ExecutionRequest) -> ExecutionResult:
        mode = self._uid_mode()
        python = self._python(mode != "none")
        notes: list[str] = []
        if mode != "none" and python is None:
            notes.append("no world-executable python3 for the sandbox UID: running as our own UID")
            mode, python = "none", self.config.python
        uid = gid = None
        if mode == "per-execution":
            uid = gid = _next_uid()
        elif mode == "fixed":
            uid = int(self.config.drop_to_uid)
            gid = int(self.config.drop_to_gid) if isinstance(self.config.drop_to_gid, int) else uid
        report = self.isolation_report()
        if uid is not None:
            report["uid"] = uid
        work = tempfile.mkdtemp(prefix="sandbox-")          # 0700, owned by us
        try:
            _write_inputs(work, req.files)
            if uid is not None:
                for root_dir, dirs, files in os.walk(work):  # hand the workspace (and inputs) to the sandbox UID
                    for n in [root_dir, *(os.path.join(root_dir, x) for x in dirs + files)]:
                        os.chown(n, uid, gid)
            env = dict(req.env) if not self.config.clean_env else {}
            # A clean, minimal environment. -I already ignores PYTHON* and user site; we still scrub.
            env.setdefault("PATH", "/usr/bin:/bin")
            env["SANDBOX_WORKDIR"] = work
            env["OPENBLAS_NUM_THREADS"] = "1"       # keep BLAS from mmap'ing huge arenas under RLIMIT_AS
            env["HOME"] = work if self.config.isolate_home else os.environ.get("HOME", work)
            env["TMPDIR"] = work
            nproc = None
            if hasattr(resource, "RLIMIT_NPROC"):
                if uid is not None:
                    nproc = req.budgets.pids                 # a fresh UID: the budget is exactly this run's
                elif not running_as_root():
                    mine = _uid_pids(os.getuid())
                    if mine is not None:                     # our UID's processes + the budget (shared!)
                        nproc = len(mine) + req.budgets.pids
            popen_kwargs: dict = {"start_new_session": True, "umask": 0o077}
            if uid is not None:
                popen_kwargs.update(user=uid, group=gid, extra_groups=[])
            cmd = [python, "-I", "-c", _LAUNCHER, json.dumps(rlimits_for(req.budgets, nproc)), req.code]
            return _spawn(cmd, env, work, req.budgets, popen_kwargs=popen_kwargs, isolation=report,
                          sweep_uid=uid if mode == "per-execution" else None, notes=notes)
        finally:
            shutil.rmtree(work, ignore_errors=True)


def _write_inputs(work: str, files: dict[str, str]) -> None:
    for name, content in files.items():
        p = Path(work) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _classify(rc: int, err_text: str, cpu_s: float, budgets: Budgets) -> tuple[str, str]:
    """(exit_reason, reason_source) for a child that was not stopped by the parent.

    ``signal``: the kernel delivered it (the code could also have signalled itself); ``parent``: the
    parent's own measurement agrees; ``code``: read from the untrusted program's exit status and stderr,
    so a program can forge it (print 'MemoryError', exit 1) — a hint for the model, not evidence.
    """
    if rc == 0:
        return "ok", "code"
    if rc < 0:
        sig = -rc
        if sig == signal.SIGXCPU:
            return "cpu_time", "signal"
        if sig == signal.SIGKILL and cpu_s >= max(int(budgets.cpu_s), 1) - 0.05:
            return "cpu_time", "parent"       # the hard CPU limit (the code ignored SIGXCPU)
        if sig == getattr(signal, "SIGXFSZ", -1):
            return "file_too_large", "signal"
        return "killed", "signal"
    low = err_text.lower()
    if "file too large" in low or "errno 27" in low:      # a Python child sees EFBIG, not SIGXFSZ
        return "file_too_large", "code"
    if "memoryerror" in low or ("openblas" in low and "memory" in low):
        return "memory", "code"
    if "too many open files" in low or "errno 24" in low:
        return "open_files", "code"
    if "blockingioerror" in low or "resource temporarily unavailable" in low or "errno 11" in low:
        return "pids", "code"
    return "error", "code"


def _spawn(cmd, env, work, budgets: Budgets, *, popen_kwargs, isolation=None, sweep_uid=None,
           notes=None) -> ExecutionResult:
    """Start the child, read its output as it streams, enforce wall time and output, collect usage."""
    notes = list(notes or [])
    out, err = _Truncator(budgets.output_bytes), _Truncator(budgets.output_bytes)
    start = time.monotonic()
    try:
        proc = subprocess.Popen(cmd, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, **popen_kwargs)
    except OSError as e:
        return ExecutionResult("sandbox_error", None, "", f"could not start the sandbox: {e}", False, Usage(),
                               isolation=isolation or {}, reason_source="parent", notes=notes)
    deadline = start + budgets.wall_s
    preset = None
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ, out)
    sel.register(proc.stderr, selectors.EVENT_READ, err)
    exited_at = None
    leader_ru = None
    while sel.get_map():
        now = time.monotonic()
        if now >= deadline:
            preset = "wall_timeout"
            break
        if exited_at is None:
            exited, leader_ru = _try_reap(proc)
            if exited:
                exited_at = now
        if exited_at is not None and now - exited_at > 0.2:
            notes.append("a background process held the output pipes after the main process exited")
            break
        for key, _ in sel.select(timeout=min(deadline - now, 0.1)):
            chunk = os.read(key.fileobj.fileno(), 65536)
            if not chunk:
                sel.unregister(key.fileobj)
                continue
            key.data.feed(chunk)                       # keeps output_bytes, counts the rest
        if out.total + err.total > budgets.output_kill_bytes:
            preset = "output_limit"
            break
    sel.close()
    _kill_group(proc)            # the leader on timeout/limit, and anything still in its group either way
    ru = leader_ru if proc.returncode is not None else _reap(proc)
    wall = time.monotonic() - start
    for f in (proc.stdout, proc.stderr):
        f.close()
    swept = {}
    if sweep_uid is not None:
        n = _kill_uid(sweep_uid)
        files = _remove_uid_files(sweep_uid, skip=work)
        if n:
            notes.append(f"killed {n} process(es) of the sandbox UID that outlived the run (setsid escape)")
        if files:
            notes.append(f"removed {len(files)} file(s) the sandbox left outside its workspace")
        swept = {"processes": n, "files": files}
    rc = proc.returncode
    cpu = (ru.ru_utime + ru.ru_stime) if ru is not None else 0.0
    if preset:
        reason, source = preset, "parent"
    else:
        reason, source = _classify(rc, err.text(), cpu, budgets)
    usage = Usage(cpu_s=round(cpu, 3), wall_s=round(wall, 3), max_rss_mb=_rss_mb(ru),
                  disk_bytes=_dir_size(work), stdout_bytes=out.total, stderr_bytes=err.total)
    over = _over_budget(usage, budgets)
    return ExecutionResult(reason, rc, out.text(), err.text(), out.truncated or err.truncated, usage,
                           artifacts=_artifacts(work), over_budget=over, isolation=isolation or {},
                           reason_source=source, swept=swept, notes=notes)


def _try_reap(proc: subprocess.Popen):
    """(exited, rusage): a non-blocking ``wait4`` that keeps the leader's rusage.

    ``Popen.poll()`` would reap the leader with ``waitpid`` and throw its rusage away, and a later
    ``wait4`` would then find nothing: the run's CPU and peak memory would silently read 0.
    """
    if proc.returncode is not None:
        return True, None
    if not hasattr(os, "wait4"):
        return proc.poll() is not None, None
    try:
        pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
    except ChildProcessError:
        return True, None
    if pid:
        proc.returncode = os.waitstatus_to_exitcode(status)
        return True, ru
    return False, None


def _reap(proc: subprocess.Popen):
    """Wait for the leader with ``wait4`` so CPU and peak memory are *this* child's, not every child's.

    It covers the leader and the descendants it reaped itself; a grandchild killed by the group kill is
    reparented to init and never counted here — its CPU is invisible to a process sandbox.
    """
    if not hasattr(os, "wait4"):
        proc.wait()
        return None
    for _ in range(200):                                # SIGKILL was sent: this returns within milliseconds
        try:
            pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
        except ChildProcessError:
            proc.wait()
            return None
        if pid:
            proc.returncode = os.waitstatus_to_exitcode(status)
            return ru
        time.sleep(0.01)
    _, status, ru = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    return ru


def _rss_mb(ru) -> float:
    if ru is None:
        return 0.0
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024     # bytes on macOS, KiB on Linux
    return round(ru.ru_maxrss / scale, 1)


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the child's whole process group, so a backgrounded grandchild dies with it (FACTS §10).

    The child is a session leader (``start_new_session``), so its process-group id equals its pid, and the
    kill reaches every descendant that stayed in the group — not one that called ``setsid()`` itself.
    If the leader has already been reaped, POSIX still does not reuse a process-group ID while the group
    has members, so the kill cannot hit a stranger's group.
    """
    if IS_POSIX:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


def _dir_size(work: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(work):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _artifacts(work: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for root, _dirs, files in os.walk(work):
        for f in files:
            p = Path(root) / f
            rel = str(p.relative_to(work))
            try:
                out[rel] = digest(p.read_bytes().hex())
            except OSError:
                pass
    return out


def _over_budget(usage: Usage, budgets: Budgets) -> list[str]:
    over = []
    if usage.disk_bytes > budgets.disk_mb * 1024 * 1024:
        over.append("disk_mb")
    return over
