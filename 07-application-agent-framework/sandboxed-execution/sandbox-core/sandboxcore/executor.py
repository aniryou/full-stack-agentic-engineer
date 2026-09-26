"""Run code in a child process with rlimits and a wall-clock kill — and say plainly what it cannot stop.

The one idea: in-process restrictions are not a boundary (the code shares your interpreter, so it can
undo them), so the first real boundary is a **separate process** you start clean and bound from outside.
This module has two runners on purpose:

* ``UnsafeExecutor`` — runs code in a child that inherits this process's environment and home. It exists so
  notebook 01 can *watch a secret leak* and *watch the network get reached*; never use it for real work.
* ``ProcessSandbox`` — the honest T0 sandbox: a clean environment, an ephemeral workspace, POSIX resource
  limits (CPU, address space, processes, file size, open files) applied in the child before it runs the
  code, a wall-clock timeout that kills the **whole process group** (so a backgrounded grandchild dies too),
  and streamed output truncation. It states what it does *not* stop — the network (rlimits don't touch
  sockets), and, when run as root, the process limit (RLIMIT_NPROC is per real UID and ignored for uid 0).

Everything is measured on the host it runs on and labelled as such. The mechanisms and their gotchas are in
FACTS §10 and §15; the numbers in the primer come from probing this machine.
"""
from __future__ import annotations

import contextlib
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .contract import Budgets, ExecutionRequest, ExecutionResult, Usage, digest

IS_POSIX = os.name == "posix"
IS_LINUX = sys.platform.startswith("linux")

# A signal number the child died from -> the contract exit reason it means.
_SIGNAL_REASON = {
    signal.SIGKILL: "killed",     # our wall-clock kill, or RLIMIT_CPU hard limit, or the OOM killer
    signal.SIGXCPU: "cpu_time",   # RLIMIT_CPU soft limit reached (soft < hard)
    signal.SIGXFSZ: "file_too_large",   # RLIMIT_FSIZE, for a non-Python child (a Python child sees EFBIG)
    signal.SIGSEGV: "error",
    signal.SIGABRT: "error",
}


def running_as_root() -> bool:
    return IS_POSIX and hasattr(os, "geteuid") and os.geteuid() == 0


@dataclass
class SandboxConfig:
    """How the process sandbox is set up, separate from the per-call budgets."""
    drop_to_uid: int | None = 65534   # 'nobody': needed for RLIMIT_NPROC to bite; None = keep current uid
    drop_to_gid: int | None = 65534
    clean_env: bool = True            # start from an empty environment, not the parent's
    isolate_home: bool = True         # point HOME at the workspace, so ~/.ssh and ~/.config are empty
    python: str = sys.executable


def _limit_reason(rc: int) -> str:
    """Map a subprocess return code to a contract exit reason."""
    if rc == 0:
        return "ok"
    if rc < 0:                       # killed by signal -rc
        return _SIGNAL_REASON.get(-rc, "killed")
    return "error"                   # non-zero exit: the program raised or exited non-zero


def _preexec(cfg: SandboxConfig, budgets: Budgets):
    """Return a function that runs in the child after fork, before exec.

    It sets resource limits and drops privileges. ``preexec_fn`` runs between fork and exec and is unsafe
    with threads (it may deadlock); ProcessSandbox therefore does its own work single-threaded, and the
    child re-exec's a fresh ``python -I`` so nothing of ours is inherited. See FACTS §10/§15 pitfall 4.
    """
    uid, gid = cfg.drop_to_uid, cfg.drop_to_gid

    def apply() -> None:
        # File size first: a Python child turns SIGXFSZ into OSError(EFBIG); a shell child would be killed.
        _set(resource.RLIMIT_FSIZE, budgets.file_mb * 1024 * 1024)
        _set(resource.RLIMIT_AS, budgets.memory_mb * 1024 * 1024)
        _set(resource.RLIMIT_NOFILE, budgets.open_files)
        _set(resource.RLIMIT_CORE, 0)
        # CPU: soft one below hard, so the soft limit raises SIGXCPU first (a catchable warning) and the
        # hard limit guarantees a SIGKILL if the code ignores it. If they were equal it would be SIGKILL.
        _set(resource.RLIMIT_CPU, int(budgets.cpu_s), max(int(budgets.cpu_s) + 1, 1))
        if hasattr(resource, "RLIMIT_NPROC"):
            _set(resource.RLIMIT_NPROC, budgets.pids)
        os.setsid()                  # own process group, so the whole group can be killed on timeout
        if gid is not None and running_as_root():
            os.setgroups([])
            os.setgid(gid)
        if uid is not None and running_as_root():
            os.setuid(uid)           # after this the child cannot regain privilege (no CAP_SETUID)

    return apply


def _set(res: int, soft: int, hard: int | None = None) -> None:
    hard = soft if hard is None else hard
    try:
        cur_soft, cur_hard = resource.getrlimit(res)
        if cur_hard != resource.RLIM_INFINITY:
            hard = min(hard, cur_hard)
            soft = min(soft, hard)
        resource.setrlimit(res, (soft, hard))
    except (ValueError, OSError):
        pass                         # a limit the OS will not accept (e.g. macOS RLIMIT_AS): skip, don't crash


class _Truncator:
    """Read a stream to EOF, keep the first ``keep`` bytes, count the rest."""

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
                          isolation={"kind": "none",
                          "note": "inherits the parent env and home; not a boundary"})


class ProcessSandbox:
    """A child process with a clean env, an ephemeral workspace, rlimits and a wall-clock kill."""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()

    def isolation_report(self) -> dict:
        """What this sandbox can and cannot enforce on this machine, decided at run time."""
        root = running_as_root()
        return {
            "kind": "process",
            "os": platform.system(),
            "clean_env": self.config.clean_env,
            "isolate_home": self.config.isolate_home,
            "uid_dropped": bool(self.config.drop_to_uid is not None and root),
            "rlimits_enforced": IS_POSIX,
            "as_limit_enforced": IS_LINUX,          # macOS accepts RLIMIT_AS but does not enforce it
            "nproc_enforced": bool(hasattr(resource, "RLIMIT_NPROC")
                                   and not (root and self.config.drop_to_uid is None)),
            "network_blocked": False,               # rlimits never touch sockets — see egress_connect probe
            "note": ("network egress is NOT stopped here; use an egress proxy or a network namespace / "
                     "container --network none. RLIMIT_NPROC is ignored for uid 0."),
        }

    def run(self, req: ExecutionRequest) -> ExecutionResult:
        work = tempfile.mkdtemp(prefix="sandbox-")
        try:
            _write_inputs(work, req.files)
            # If we drop to another UID, that UID must be able to enter and write the workspace, which
            # mkdtemp created 0700 owned by us. Open it up (the workspace is ephemeral and per-call).
            if self.config.drop_to_uid is not None and running_as_root():
                os.chmod(work, 0o777)
            env = dict(req.env) if not self.config.clean_env else {}
            # A clean, minimal environment. -I already ignores PYTHON* and user site; we still scrub.
            env.setdefault("PATH", "/usr/bin:/bin")
            env["SANDBOX_WORKDIR"] = work
            env["OPENBLAS_NUM_THREADS"] = "1"       # keep BLAS from mmap'ing huge arenas under RLIMIT_AS
            env["HOME"] = work if self.config.isolate_home else os.environ.get("HOME", work)
            env["TMPDIR"] = work
            # ``python -I`` = isolated mode: ignore PYTHON* env, user site-packages and the cwd on sys.path.
            cmd = [self.config.python, "-I", "-c", req.code]
            return _spawn(cmd, env, work, req.budgets,
                          preexec=_preexec(self.config, req.budgets),
                          isolation=self.isolation_report())
        finally:
            shutil.rmtree(work, ignore_errors=True)


def _write_inputs(work: str, files: dict[str, str]) -> None:
    for name, content in files.items():
        p = Path(work) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def _spawn(cmd, env, work, budgets: Budgets, *, preexec=None, isolation=None) -> ExecutionResult:
    """Start the child, stream and truncate its output, enforce the wall clock, collect usage."""
    out, err = _Truncator(budgets.output_bytes), _Truncator(budgets.output_bytes)
    start = time.monotonic()
    kwargs = dict(cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if preexec is not None:
        kwargs["preexec_fn"] = preexec
    elif IS_POSIX:
        kwargs["start_new_session"] = True          # even the unsafe runner gets its own group to kill
    usage_before = _child_cpu()
    proc = subprocess.Popen(cmd, **kwargs)
    reason = "ok"
    try:
        kept_out, kept_err = proc.communicate(timeout=budgets.wall_s)
        out.feed(kept_out or b"")
        err.feed(kept_err or b"")
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        kept_out, kept_err = proc.communicate()
        out.feed(kept_out or b"")
        err.feed(kept_err or b"")
        reason = "wall_timeout"
    else:
        # The main child is done, but it may have left backgrounded children in its group holding
        # resources (a fork bomb's sleepers). Reap the whole group so nothing lingers past the call.
        _kill_group(proc)
    wall = time.monotonic() - start
    rc = proc.returncode
    if reason != "wall_timeout":
        reason = _limit_reason(rc)
        # A Python child that hit RLIMIT_FSIZE exits non-zero with 'File too large' in stderr, not a signal.
        low = err.text().lower()
        if reason == "error":
            if "file too large" in low or "errno 27" in low:
                reason = "file_too_large"
            elif "memoryerror" in low or ("openblas" in low and "memory" in low):
                reason = "memory"
            elif "too many open files" in low or "errno 24" in low:
                reason = "open_files"
            elif "blockingioerror" in low or "resource temporarily unavailable" in low or "errno 11" in low:
                reason = "pids"
    cpu = max(_child_cpu() - usage_before, 0.0)
    usage = Usage(cpu_s=round(cpu, 3), wall_s=round(wall, 3),
                  disk_bytes=_dir_size(work), stdout_bytes=out.total, stderr_bytes=err.total)
    over = _over_budget(usage, work, budgets)
    return ExecutionResult(reason, rc, out.text(), err.text(), out.truncated or err.truncated,
                           usage, artifacts=_artifacts(work), over_budget=over, isolation=isolation or {})


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the child's whole process group, so a backgrounded grandchild dies with it (FACTS §10).

    The child called ``setsid()``, so its process-group id equals its pid; killing that group reaches
    every descendant even after the group leader itself has exited.
    """
    if IS_POSIX:
        try:
            os.killpg(proc.pid, signal.SIGKILL)   # setsid() made pgid == pid
            return
        except (ProcessLookupError, PermissionError):
            pass
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()


def _child_cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


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


def _over_budget(usage: Usage, work: str, budgets: Budgets) -> list[str]:
    over = []
    if usage.disk_bytes > budgets.disk_mb * 1024 * 1024:
        over.append("disk_mb")
    return over
