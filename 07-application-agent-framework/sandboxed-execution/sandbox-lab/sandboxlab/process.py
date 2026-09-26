"""process.py — the bottom two rungs of the isolation ladder, runnable on any laptop.

One idea: a process sandbox *budgets* untrusted code (CPU, memory, files, processes, output,
time) and, with a dedicated UID, hides the agent's files and processes from it — but it does not
isolate the kernel, and without a network namespace it does not touch the network at all.
``Unsandboxed`` is the executor people ship by accident; ``ProcessSandbox`` is the best you can do
without containers (PRIMER §2 "The isolation ladder"). The probes in ``probes.py`` measure the
difference on this machine.

Both return an ``ExecResult`` — the lab's execution contract (PRIMER §3): truncated output,
usage, and one exit reason — and ``ExecResult.to_tool_result()`` turns it into the result shape
the 07.1 agent loop expects (``{"ok": ..., "data" | "error", "message", "hint"}``).
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from . import env as _env

WRAPPER = Path(__file__).with_name("wrapper.py")
MARKER = "@@SANDBOX_RESULT@@"

# The exit reasons of the contract, and what an agent should do about each (PRIMER §3).
EXIT_REASONS: dict[str, str] = {
    "ok": "exited 0",
    "error": "exited non-zero: a bug in the code; show the model stderr",
    "timeout": "wall-clock budget exhausted (sleeping, blocked or just slow code); the process group was killed",
    "cpu_limit": "CPU-seconds budget exhausted (SIGXCPU, then SIGKILL)",
    "memory_limit": "address-space or memory budget exhausted (MemoryError, or the OOM killer)",
    "file_too_large": "a file exceeded the per-file budget (EFBIG, or SIGXFSZ for non-Python children)",
    "pids_limit": "could not create another process (fork returned EAGAIN)",
    "output_limit": "printed more than the output budget; killed",
    "disk_limit": "the workspace volume filled up or the pod was evicted for exceeding its sizeLimit",
    "killed": "killed by a signal the sandbox did not send for a budget",
    "sandbox_error": "the sandbox itself failed (image pull, admission, a missing runtime handler)",
    "harness_timeout": "no budget at all: the harness killed it after its own timeout (Unsandboxed only)",
}
RETRYABLE = {"sandbox_error"}      # infrastructure failures; everything else is a property of the code


@dataclass(frozen=True)
class Budgets:
    """What one execution may use. ``disk_mib`` is enforced by a sized tmpfs/emptyDir (containers);
    a process sandbox can only cap each file (``file_mib``)."""
    cpu_s: float = 2.0
    wall_s: float = 5.0
    memory_mib: int = 256
    pids: int = 32
    file_mib: int = 8
    disk_mib: int = 16
    nofile: int = 64
    output_bytes: int = 64 * 1024
    output_kill_bytes: int = 1024 * 1024

    def wrapper_json(self, *, memory_via_rlimit: bool = True) -> str:
        d = asdict(self)
        d.pop("disk_mib")
        if not memory_via_rlimit:
            d["memory_mib"] = 0          # containers: the memory cgroup enforces it, not RLIMIT_AS
        return json.dumps(d, separators=(",", ":"))

    def with_(self, **kw) -> "Budgets":
        return replace(self, **kw)


@dataclass
class ExecResult:
    exit_reason: str
    returncode: int | None
    stdout: str
    stderr: str
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    wall_s: float = 0.0            # the code's run time, as the wrapper measured it
    total_s: float = 0.0           # the caller's round trip: startup + run + teardown
    cpu_s: float | None = None
    max_rss_kib: int | None = None
    isolation: str = "none"
    stragglers_killed: int = 0
    notes: list[str] = field(default_factory=list)
    simulated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_reason == "ok"

    @property
    def startup_s(self) -> float:
        return max(0.0, self.total_s - self.wall_s)

    @classmethod
    def from_wrapper(cls, d: dict, *, isolation: str, total_s: float, notes: list[str] | None = None) -> "ExecResult":
        keys = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in keys}
        kw["notes"] = list(d.get("notes", [])) + list(notes or [])
        return cls(isolation=isolation, total_s=round(total_s, 4), **kw)

    def usage(self) -> dict:
        return {"wall_s": self.wall_s, "cpu_s": self.cpu_s, "max_rss_kib": self.max_rss_kib,
                "stdout_bytes": self.stdout_bytes, "stderr_bytes": self.stderr_bytes}

    def to_tool_result(self) -> dict:
        """The 07.1 tool contract: ``{"ok": True, "data": ...}`` or ``{"ok": False, "error": kind, ...}``.
        The exit reason is the error kind; truncated output rides along in ``data`` either way."""
        data = {"stdout": self.stdout, "stderr": self.stderr[-4000:], "exit_reason": self.exit_reason,
                "truncated": self.stdout_truncated or self.stderr_truncated, "usage": self.usage()}
        if self.ok:
            return {"ok": True, "data": data}
        hint = {"error": "Read stderr, fix the code, and call run_code again.",
                "timeout": "Make it faster or smaller; the budget will not grow.",
                "cpu_limit": "Make it faster or smaller; the budget will not grow.",
                "memory_limit": "Use less memory (stream, chunk, or sample the data).",
                "output_limit": "Print a summary, not the data.",
                "sandbox_error": "Infrastructure failure: retrying the same call may succeed."}.get(
                    self.exit_reason, "Do not retry the same code.")
        return {"ok": False, "error": self.exit_reason, "message": EXIT_REASONS.get(self.exit_reason, ""),
                "hint": hint, "data": data}

    def line(self) -> str:
        cpu = "-" if self.cpu_s is None else f"{self.cpu_s:.2f}"
        return (f"{self.isolation:<14} {self.exit_reason:<15} rc={str(self.returncode):<5} wall {self.wall_s:6.3f}s "
                f"cpu {cpu:>5}s out {self.stdout_bytes}B{' (truncated)' if self.stdout_truncated else ''}")


def parse_wrapper_output(text: str) -> dict | None:
    """Find the wrapper's result line in stdout / ``docker run`` output / ``kubectl logs``."""
    for line in reversed(text.splitlines()):
        if line.startswith(MARKER):
            try:
                return json.loads(line[len(MARKER):].strip())
            except ValueError:
                return None
    return None


class Unsandboxed:
    """The executor to never ship: ``subprocess.run([python, "-c", code], timeout=...)`` in the
    agent's own environment, working directory, UID and network. Its only limit is a harness
    timeout (so a notebook cannot hang) — and that timeout kills the direct child only."""

    name = "none"

    def __init__(self, env: dict | None = None, cwd: str | None = None, harness_timeout_s: float = 4.0):
        self.env = env
        self.cwd = cwd
        self.harness_timeout_s = harness_timeout_s

    def run(self, code: str, budgets: Budgets | None = None, env: dict | None = None) -> ExecResult:
        t0 = time.monotonic()
        try:
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, cwd=self.cwd,
                               env=env if env is not None else (self.env if self.env is not None else os.environ.copy()),
                               timeout=self.harness_timeout_s)
            out, err, rc, reason = p.stdout, p.stderr, p.returncode, None
        except subprocess.TimeoutExpired as e:
            out, err, rc, reason = e.stdout or b"", e.stderr or b"", None, "harness_timeout"
        wall = time.monotonic() - t0
        err_text = err.decode("utf-8", "replace")
        if reason is None:
            reason = "ok" if rc == 0 else ("memory_limit" if "MemoryError" in err_text else "error")
        return ExecResult(exit_reason=reason, returncode=rc, stdout=out.decode("utf-8", "replace"), stderr=err_text,
                          stdout_bytes=len(out), stderr_bytes=len(err), wall_s=round(wall, 4), total_s=round(wall, 4),
                          isolation=self.name, notes=["no budgets: the agent's env, files, UID and network"])


def _world_ok(path: str) -> bool:
    """Can an arbitrary UID execute ``path`` (every parent traversable, file world r-x)?"""
    p = Path(os.path.realpath(path))
    try:
        if (p.stat().st_mode & 0o005) != 0o005:
            return False
        return all(parent.stat().st_mode & 0o001 for parent in p.parents)
    except OSError:
        return False


def child_python(drop_uid: bool) -> str | None:
    """An interpreter the sandbox UID can run: this one, or a system python3 when this one lives
    somewhere private (a venv under /root)."""
    if not drop_uid or _world_ok(sys.executable):
        return sys.executable
    for cand in ("/usr/bin/python3", "/usr/local/bin/python3"):
        if os.path.exists(cand) and _world_ok(cand):
            return cand
    return None


def default_sandbox_uid() -> int:
    """A per-process UID in 61000-61996. A dedicated UID is what makes RLIMIT_NPROC and the
    kill-by-UID sweep mean something; per-process keeps two concurrent test runs apart."""
    return 61000 + os.getpid() % 997


class ProcessSandbox:
    """Budgets + a clean environment + a throwaway workspace, and — when this process is root — a
    dedicated unprivileged UID; optionally an empty network namespace (``unshare``).

    What it enforces: CPU seconds, wall time with a process-group kill, address space, per-file
    size, open files, process count (dedicated UID only), output bytes, no inherited secrets in
    the environment, no setuid escalation (no_new_privs). What it cannot: kernel isolation (every
    syscall reaches the host kernel), the network unless ``netns`` is on, total disk (only per
    file), the filesystem when it runs as your own UID, and on macOS most of the rlimits.
    """

    def __init__(self, budgets: Budgets | None = None, *, drop_uid: bool | None = None, uid: int | None = None,
                 netns: bool | None = None, extra_env: dict | None = None, keep_workspace: bool = False,
                 workspace_root: str | None = None):
        self.budgets = budgets or Budgets()
        want_drop = _env.is_root() if drop_uid is None else drop_uid
        self.python = child_python(want_drop)
        self.drop_uid = bool(want_drop and _env.is_root() and self.python)
        self.uid = (uid or default_sandbox_uid()) if self.drop_uid else None
        self.netns = bool(_env.netns_mode()) if netns is None else (netns and bool(_env.netns_mode()))
        self.extra_env = dict(extra_env or {})
        self.keep_workspace = keep_workspace
        self.workspace_root = workspace_root or ("/tmp" if os.path.isdir("/tmp") else None)
        self.notes: list[str] = []
        if want_drop and not self.drop_uid:
            self.notes.append("cannot switch to a dedicated UID (not root, or no world-executable python3): "
                              "the code runs as your UID and can read what you can read")
        # a user namespace maps only our own UID, so setuid to the sandbox UID is impossible inside it
        if self.netns and _env.netns_mode() == "userns" and self.drop_uid:
            self.drop_uid, self.uid = False, None

    @property
    def name(self) -> str:
        return "process+netns" if self.netns else "process"

    def command(self, budgets: Budgets, workspace: str) -> list[str]:
        cmd = _env.netns_prefix() if self.netns else []
        cmd += [sys.executable, "-I", str(WRAPPER), "--stdin", "--budgets", budgets.wrapper_json(),
                "--workspace", workspace]
        if self.drop_uid:
            cmd += ["--uid", str(self.uid), "--gid", str(self.uid), "--kill-uid", "--nproc"]
        else:
            cmd += ["--nproc"]
        if self.python and self.python != sys.executable:
            cmd += ["--python", self.python]
        for k in self.extra_env:
            cmd += ["--pass-env", k]
        return cmd

    def run(self, code: str, budgets: Budgets | None = None, env: dict | None = None) -> ExecResult:
        b = budgets or self.budgets
        ws = tempfile.mkdtemp(prefix="sbx-ws-", dir=self.workspace_root)
        os.chmod(ws, stat.S_IRWXU | stat.S_IXGRP | stat.S_IXOTH)
        penv = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", **self.extra_env, **(env or {})}
        t0 = time.monotonic()
        try:
            p = subprocess.run(self.command(b, ws), input=code.encode(), capture_output=True, env=penv,
                               timeout=b.wall_s + 10, start_new_session=True)
            d = parse_wrapper_output(p.stdout.decode("utf-8", "replace"))
            if d is None:
                d = {"exit_reason": "sandbox_error", "returncode": p.returncode, "stdout": "",
                     "stderr": p.stderr.decode("utf-8", "replace")[-2000:], "notes": ["no result line from the wrapper"]}
        except subprocess.TimeoutExpired:
            d = {"exit_reason": "sandbox_error", "returncode": None, "stdout": "", "stderr": "",
                 "notes": ["the wrapper did not return within wall_s + 10 s"]}
        finally:
            if not self.keep_workspace:
                shutil.rmtree(ws, ignore_errors=True)
        return ExecResult.from_wrapper(d, isolation=self.name, total_s=time.monotonic() - t0, notes=self.notes)

    def describe(self) -> str:
        who = f"dedicated uid {self.uid}" if self.drop_uid else f"your uid {os.getuid() if hasattr(os, 'getuid') else '?'}"
        net = f"empty network namespace ({_env.netns_mode()})" if self.netns else "the host network (NOT isolated)"
        return (f"ProcessSandbox: {who}; {net}; budgets {asdict(self.budgets)}")
