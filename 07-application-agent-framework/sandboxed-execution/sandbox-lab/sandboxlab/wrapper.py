"""wrapper.py — the execution contract, enforced from *inside* whatever isolation you chose.

One idea: the same ~250-line, standard-library file runs untrusted code at every level of the
isolation ladder (PRIMER §2) — as a plain process on a laptop, as the entrypoint of a hardened
``docker run``, inside a Kubernetes pod under gVisor — and always returns the same result shape
(PRIMER §3 "The execution contract"). Isolation decides *what the code can reach*; the wrapper
decides *how much it may use and what comes back*: CPU seconds, wall time, address space, file
size, open files, processes, output bytes, and one exit reason.

    python3 -I wrapper.py --stdin --budgets '{"cpu_s": 2, "wall_s": 5}' < code.py
    python3 -I wrapper.py --code-env SANDBOX_CODE --workspace /work        # inside a pod

It prints exactly one line to stdout: ``@@SANDBOX_RESULT@@ {json}``. Everything the untrusted
code printed is inside that JSON, truncated to the budget — so the result survives ``docker run``,
``kubectl logs`` and ``kubectl exec`` without any other channel.

Design notes (the pitfalls this file exists to avoid):
* It is **single-threaded**, so ``preexec_fn`` is safe here (it is not in a threaded parent).
* The child gets a **new session** and the wall timeout kills the **process group** — killing
  only the direct child leaves grandchildren running (``subprocess.run(timeout=)`` does that).
* ``RLIMIT_CPU`` counts CPU seconds; sleeping code never reaches it, so there is a wall clock too.
  Soft limit < hard limit: SIGXCPU first, SIGKILL one second later.
* ``RLIMIT_NPROC`` counts **every process of the real UID, system-wide**, and is ignored for root.
  It only means something with a dedicated UID (``--uid``); inside containers use the pids cgroup.
* ``RLIMIT_AS`` is virtual memory: too low and the interpreter (or numpy's OpenBLAS) cannot start.
* A process group is advisory: code can call ``setsid()`` and leave it. With a dedicated UID
  (``--kill-uid``) the wrapper sweeps every process of that UID after the run — the process-level
  stand-in for what a container's cgroup or PID namespace does for free.
"""
from __future__ import annotations

import argparse
import json
import os
import selectors
import signal
import subprocess
import sys
import time

MARKER = "@@SANDBOX_RESULT@@"
DEFAULTS = {
    "cpu_s": 2.0,             # RLIMIT_CPU (soft); hard = soft + 1
    "wall_s": 5.0,            # wall clock, then SIGKILL to the process group
    "memory_mib": 256,        # RLIMIT_AS (0 = do not set: containers use the memory cgroup)
    "pids": 32,               # RLIMIT_NPROC headroom (only with --nproc)
    "file_mib": 8,            # RLIMIT_FSIZE, per file
    "nofile": 64,             # RLIMIT_NOFILE
    "output_bytes": 65536,    # kept per stream; the rest is counted and dropped
    "output_kill_bytes": 1048576,  # total output that ends the run (exit reason output_limit)
}
CLEAN_PATH = "/usr/local/bin:/usr/bin:/bin"


def _uid_process_count(uid: int) -> int:
    """Processes whose real UID is ``uid`` (Linux ``/proc``); -1 where there is no ``/proc``."""
    if not os.path.isdir("/proc/self"):
        return -1
    n = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("Uid:"):
                        n += int(line.split()[1]) == uid
                        break
        except OSError:
            continue
    return n


def _kill_uid(uid: int) -> int:
    """SIGKILL every live process whose real UID is ``uid``; returns how many distinct ones."""
    killed: set[int] = set()
    for _ in range(3):   # a process can fork while we sweep; repeat until nothing live is left
        found = 0
        for pid in os.listdir("/proc"):
            if not pid.isdigit() or int(pid) == os.getpid():
                continue
            try:
                with open(f"/proc/{pid}/status") as f:
                    fields = dict(l.split(":", 1) for l in f if ":" in l)
                real, state = int(fields["Uid"].split()[0]), fields["State"].split()[0]
            except (OSError, KeyError, ValueError, IndexError):
                continue
            if real == uid and state != "Z":          # zombies are already dead
                found += 1
                killed.add(int(pid))
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except OSError:
                    pass
        if not found:
            break
        time.sleep(0.05)
    return len(killed)


def _no_new_privs() -> bool:
    """prctl(PR_SET_NO_NEW_PRIVS): setuid binaries and file capabilities stop working (Linux)."""
    try:
        import ctypes
        return ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0) == 0
    except Exception:
        return False


def _limits(b: dict, uid: int | None, gid: int | None, nproc: int | None):
    """Runs in the child between fork and exec. Order matters: limits first, identity last."""
    import resource

    def lim(res, soft, hard=None):
        try:
            resource.setrlimit(res, (int(soft), int(hard if hard is not None else soft)))
        except (ValueError, OSError):
            pass   # e.g. RLIMIT_AS on macOS; the result's notes say which limits were applied

    def apply():
        lim(resource.RLIMIT_CPU, max(1, int(b["cpu_s"] + 0.999)), max(1, int(b["cpu_s"] + 0.999)) + 1)
        if b.get("memory_mib"):
            lim(resource.RLIMIT_AS, b["memory_mib"] * 2**20)
        lim(resource.RLIMIT_FSIZE, b["file_mib"] * 2**20)
        lim(resource.RLIMIT_NOFILE, b["nofile"])
        lim(resource.RLIMIT_CORE, 0)
        if nproc is not None:
            lim(resource.RLIMIT_NPROC, nproc)
        _no_new_privs()
        if gid is not None:
            os.setgroups([])
            os.setgid(gid)
        if uid is not None:
            os.setuid(uid)
    return apply


def classify(rc: int | None, preset: str | None, stderr_tail: str, cpu_s: float | None, cpu_budget: float) -> str:
    """One exit reason per run, in the primer's vocabulary (PRIMER §3; ``sandboxcore.contract.EXIT_REASONS``).
    The order encodes which evidence wins. Reasons read from ``stderr_tail`` are the program's own say-so
    (it can print ``MemoryError`` and exit 1); signals and the presets are observed by the wrapper."""
    if preset:
        return preset
    if rc is None:
        return "sandbox_error"
    if rc == 0:
        return "ok"
    if rc < 0:
        sig = -rc
        if sig == signal.SIGXCPU or (sig == signal.SIGKILL and cpu_s is not None and cpu_s >= cpu_budget - 0.05):
            return "cpu_time"
        if sig == getattr(signal, "SIGXFSZ", 25):
            return "file_too_large"
        return "killed"
    if "MemoryError" in stderr_tail or "Cannot allocate memory" in stderr_tail:
        return "memory"
    if "[Errno 27]" in stderr_tail or "File too large" in stderr_tail:
        return "file_too_large"
    if "[Errno 11]" in stderr_tail and ("fork" in stderr_tail or "BlockingIOError" in stderr_tail):
        return "pids"
    return "error"


def run(code: str, budgets: dict, workspace: str, *, uid: int | None = None, gid: int | None = None,
        use_nproc: bool = False, kill_uid: bool = False, extra_env: dict | None = None,
        python: str | None = None) -> dict:
    b = {**DEFAULTS, **(budgets or {})}
    os.makedirs(workspace, exist_ok=True)
    main = os.path.join(workspace, "main.py")
    with open(main, "w") as f:
        f.write(code)
    notes = []
    if uid is not None:
        try:
            os.chown(workspace, uid, gid if gid is not None else uid)
            os.chown(main, uid, gid if gid is not None else uid)
        except OSError as e:
            notes.append(f"could not chown the workspace to uid {uid}: {e}")
    nproc = None
    if use_nproc:
        who = uid if uid is not None else os.getuid()
        if who == 0:
            notes.append("RLIMIT_NPROC not applied: it is not enforced for root")
        else:
            base = _uid_process_count(who)
            if base < 0:
                notes.append("RLIMIT_NPROC not applied: no /proc to count this UID's processes")
            else:
                nproc = base + int(b["pids"])
                notes.append(f"RLIMIT_NPROC = {nproc} ({base} processes of uid {who} already running + {b['pids']})")
    env = {"PATH": CLEAN_PATH, "HOME": workspace, "TMPDIR": workspace, "LANG": "C.UTF-8",
           "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    env.update(extra_env or {})
    before = _rusage_children()
    oom_before = _oom_kills()
    t0 = time.monotonic()
    try:
        p = subprocess.Popen([python or sys.executable, "-I", main], cwd=workspace, env=env,
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True, preexec_fn=_limits(b, uid, gid, nproc))
    except OSError as e:
        return {"exit_reason": "sandbox_error", "returncode": None, "stdout": "", "stderr": str(e),
                "stdout_bytes": 0, "stderr_bytes": 0, "stdout_truncated": False, "stderr_truncated": False,
                "wall_s": 0.0, "cpu_s": None, "max_rss_kib": None, "stragglers_killed": 0,
                "budgets": b, "notes": notes + [f"could not start: {e}"]}
    keep = int(b["output_bytes"])
    out = {"stdout": bytearray(), "stderr": bytearray()}
    total = {"stdout": 0, "stderr": 0}
    sel = selectors.DefaultSelector()
    sel.register(p.stdout, selectors.EVENT_READ, "stdout")
    sel.register(p.stderr, selectors.EVENT_READ, "stderr")
    deadline = t0 + float(b["wall_s"])
    preset = None
    exited_at = None
    while sel.get_map():
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            preset = "wall_timeout"
            break
        if exited_at is None and p.poll() is not None:
            exited_at = now
        if exited_at is not None and now - exited_at > 0.3:
            # the main process is gone but something it started still holds stdout/stderr open
            notes.append("a background process kept the output pipes open after the main process exited")
            break
        for key, _ in sel.select(timeout=min(remaining, 0.25)):
            chunk = os.read(key.fileobj.fileno(), 65536)
            if not chunk:
                sel.unregister(key.fileobj)
                continue
            name = key.data
            total[name] += len(chunk)
            room = keep - len(out[name])
            if room > 0:
                out[name] += chunk[:room]
        if total["stdout"] + total["stderr"] > int(b["output_kill_bytes"]):
            preset = "output_limit"
            break
    if preset:
        _killpg(p.pid)
    try:
        rc = p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _killpg(p.pid)
        rc = p.wait()
    wall = time.monotonic() - t0
    _killpg(p.pid)                      # anything left in the child's process group
    stragglers = _kill_uid(uid) if (kill_uid and uid is not None) else 0
    if stragglers:
        notes.append(f"killed {stragglers} process(es) of uid {uid} that outlived the run (setsid or double fork)")
    after = _rusage_children()
    cpu = None if before is None or after is None else round(after[0] - before[0], 3)
    rss = None if after is None else after[1]
    stderr_text = out["stderr"].decode("utf-8", "replace")
    oom_after = _oom_kills()
    if oom_before is not None and oom_after is not None and oom_after > oom_before and not preset:
        preset = "memory"               # the cgroup's OOM killer, not a budget the wrapper enforced
        notes.append(f"memory cgroup oom_kill {oom_before} -> {oom_after}")
    reason = classify(rc, preset, stderr_text[-2000:], cpu, float(b["cpu_s"]))
    return {"exit_reason": reason, "returncode": rc,
            "stdout": out["stdout"].decode("utf-8", "replace"), "stderr": stderr_text,
            "stdout_bytes": total["stdout"], "stderr_bytes": total["stderr"],
            "stdout_truncated": total["stdout"] > keep, "stderr_truncated": total["stderr"] > keep,
            "wall_s": round(wall, 4), "cpu_s": cpu, "max_rss_kib": rss, "stragglers_killed": stragglers,
            "budgets": b, "notes": notes}


def _killpg(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


def _oom_kills() -> int | None:
    """The memory cgroup's ``oom_kill`` counter (cgroup v2, visible inside containers and pods)."""
    try:
        with open("/sys/fs/cgroup/memory.events") as f:
            return next(int(l.split()[1]) for l in f if l.startswith("oom_kill "))
    except (OSError, StopIteration, ValueError):
        return None


def _rusage_children():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        return (r.ru_utime + r.ru_stime, r.ru_maxrss)
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--stdin", action="store_true", help="read the code from stdin")
    src.add_argument("--code-file")
    src.add_argument("--code-env", help="read the code from this environment variable")
    ap.add_argument("--budgets", default="{}", help="JSON object overriding DEFAULTS")
    ap.add_argument("--workspace", default=os.environ.get("SANDBOX_WORKSPACE", "/work"))
    ap.add_argument("--uid", type=int)
    ap.add_argument("--gid", type=int)
    ap.add_argument("--nproc", action="store_true", help="apply RLIMIT_NPROC (process mode only)")
    ap.add_argument("--kill-uid", action="store_true", help="after the run, kill every process of --uid")
    ap.add_argument("--pass-env", action="append", default=[], help="copy this variable into the child")
    ap.add_argument("--python", help="interpreter for the child (default: this one)")
    a = ap.parse_args(argv)
    if a.stdin:
        code = sys.stdin.read()
    elif a.code_file:
        with open(a.code_file) as f:
            code = f.read()
    else:
        code = os.environ.get(a.code_env, "")
    extra = {k: os.environ[k] for k in a.pass_env if k in os.environ}
    res = run(code, json.loads(a.budgets), a.workspace, uid=a.uid, gid=a.gid, use_nproc=a.nproc,
              kill_uid=a.kill_uid, extra_env=extra, python=a.python)
    sys.stdout.write(f"{MARKER} {json.dumps(res, separators=(',', ':'))}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
