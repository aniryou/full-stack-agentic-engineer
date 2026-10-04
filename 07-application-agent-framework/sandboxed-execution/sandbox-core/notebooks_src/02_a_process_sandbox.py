# %% [markdown]
# # 02 · A process sandbox: rlimits, timeouts, streamed output — and what it cannot stop
#
# **Tier:** T0: a laptop, a Colab CPU or CI, free, about a minute. Real container and microVM isolation is in the
# lab (`../sandbox-lab`, notebook 01 hardened containers). The concepts are all here.
#
# ## The one-minute version
# In-process restrictions are not a boundary, because code that shares your interpreter can undo them. Thus the first
# real boundary is a **separate process** that you start clean and bound from outside.
#
# A process sandbox does six things:
#
# 1. A **clean environment**, so no credentials come with the code.
# 2. An **ephemeral workspace** (mode 0700, deleted after the call).
# 3. **POSIX resource limits** in the child: CPU seconds, address space, processes, file size, open files.
# 4. A **wall-clock deadline**, and a kill of the full process group.
# 5. Output **read as it streams**. The sandbox keeps the first `output_bytes` and stops the run past
#    `output_kill_bytes`, so a flood never lands in your memory.
# 6. When it runs as root, **its own UID per execution**. This control makes the process limit real and keeps
#    your files unreadable. It also lets the sandbox find and kill a process that left the group.
#
# The sandbox states what it does *not* stop: the network, the host kernel and, without that UID, your files and a
# `setsid()` escape. Primer: `../PRIMER.md` §2 (the isolation ladder), §3 (the execution contract). This notebook
# measures the numbers here **on this machine**, and it labels them so.

# %%
from sandboxcore import Budgets, ExecutionRequest, ProcessSandbox, SandboxConfig, running_as_root

sb = ProcessSandbox()
print("running as root:", running_as_root())
for k, v in sb.isolation_report().items():
    print(f"  {k:24} {v}")

# %% [markdown]
# ## Worked example 1 — a clean environment, and why `HOME` is not a boundary
# The child cannot see the environment of the parent. We put a token in *our* environment, and it comes back
# absent. `HOME` points at the workspace, so `~/.ssh` is empty. But that is a convenience, not isolation: the real
# home is one `pwd` lookup away. Only a **different UID** turns "you can find it" into "you cannot open it". Compare
# the per-execution UID with `drop_to_uid=None` (what a non-root laptop gets).

# %%
import os
import shutil
import tempfile

# A stand-in for the agent's home: a 0700 directory of ours holding a fake key (never the real ~).
victim_home = tempfile.mkdtemp(prefix="victim-home-")
with open(os.path.join(victim_home, "id_ed25519"), "w") as f:
    f.write("STAND-IN KEY, not a real one")
os.environ["DEMO_TOKEN"] = "sk-live-do-not-leak"
probe = ("import os\n"
         "print('token:', os.environ.get('DEMO_TOKEN', 'ABSENT'))\n"
         "print('~ is', os.path.expanduser('~'), '| running as uid', os.getuid())\n"
         "try:\n"
         f"    print('the key by absolute path:', open({victim_home + '/id_ed25519'!r}).read())\n"
         "except OSError as e:\n"
         "    print('the key by absolute path:', type(e).__name__)\n")
for label, box in (("per-execution UID", sb),
                   ("drop_to_uid=None", ProcessSandbox(SandboxConfig(drop_to_uid=None, drop_to_gid=None)))):
    r = box.run(ExecutionRequest(code=probe, budgets=Budgets(cpu_s=1, wall_s=3)))
    print(f"{label}:\n  " + (r.stdout.strip() or r.stderr.strip()[-300:]).replace("\n", "\n  "))
shutil.rmtree(victim_home)
del os.environ["DEMO_TOKEN"]

# %% [markdown]
# The token is `ABSENT` in the two cases. With the per-execution UID, the key is a path that the code knows but
# cannot open (a 0700 directory of a different UID). Without that UID, on any non-root laptop, the code runs as you:
# `~` moved, but your files did not.
#
# ## Worked example 2 — the resource limits, one at a time
# Each budget maps to a POSIX `setrlimit` that the sandbox applies in the child. Two budgets are the exception,
# because the parent enforces them while it reads: the wall clock and the output kill. The exit reason tells the
# caller *why* the run stopped. `reason_source` tells who decided it. The source is the measurement of the parent,
# a kernel signal, or the exit status and stderr of the code. The code can forge that last source.

# %%
cases = [
    ("busy loop", "x=0\nwhile True:\n x+=1", Budgets(cpu_s=1, wall_s=10)),
    ("sleep forever", "import time; time.sleep(60)", Budgets(cpu_s=5, wall_s=1)),
    ("huge file", "import os;open(os.path.join(os.environ['SANDBOX_WORKDIR'],'b'),'wb').write(b'x'*(9<<20))",
     Budgets(file_mb=1, cpu_s=2, wall_s=5)),
    ("print 200 KB", "print('A'*200000)", Budgets(output_bytes=2048, cpu_s=2, wall_s=5)),
    ("print forever", "import sys\nb=b'x'*(1<<20)\nwhile True: sys.stdout.buffer.write(b)",
     Budgets(output_bytes=2048, cpu_s=2, wall_s=5)),
    ("forge a reason", "import sys; sys.stderr.write('MemoryError\\n'); sys.exit(1)", Budgets(wall_s=3)),
]
for label, code, b in cases:
    r = sb.run(ExecutionRequest(code=code, budgets=b))
    extra = f" kept {len(r.stdout)} of {r.usage.stdout_bytes} bytes" if r.truncated else ""
    print(f"{label:15} -> {r.exit_reason:14} ({r.reason_source:6}) cpu={r.usage.cpu_s:.2f}s "
          f"wall={r.usage.wall_s:.2f}s rss={r.usage.max_rss_mb}MB{extra}")

# %% [markdown]
# "print forever" writes gigabytes a second. The parent keeps 2 KB, counts the rest, and stops the run when the
# total passes `output_kill_bytes` (1 MiB by default). Its own memory never sees the flood. "forge a reason" ends as
# `memory` with source `code`: the program *said* MemoryError. When you look for abuse, count only the `parent` and
# `signal` reasons (the audit of notebook 03, primer §8).
#
# ## Worked example 3 — the grandchild, and the process that leaves the group
# `subprocess.run(timeout=...)` kills only the direct child. A grandchild in the background survives, and init
# becomes its new parent. The sandbox starts the child in a new **session** and kills the full **process group**,
# and that catches the grandchild.
#
# But the group is advisory: code can call `setsid()` itself and leave it. Only
# its **UID** finds that escapee. With a per-execution UID, the sandbox kills every process of that UID after the
# run and removes all the files that it left in `/tmp`. Without that UID, the sandbox cannot tell the escapee from
# your own processes.

# %%
grandchild = ("import subprocess, sys, time; "
              "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
              "time.sleep(30)")
r = sb.run(ExecutionRequest(code=grandchild, budgets=Budgets(cpu_s=5, wall_s=1)))
print("grandchild in the group:", r.exit_reason, "| wall:", round(r.usage.wall_s, 2), "s (not 30 s)")

escape = ("import os, time\n"
          "if os.fork() == 0:\n"
          "    os.setsid()\n"
          "    fd = os.open(os.devnull, os.O_RDWR)\n"
          "    for i in (0, 1, 2): os.dup2(fd, i)\n"
          "    time.sleep(3); os._exit(0)\n"
          "print('parent returns; the escapee keeps running')")
r = sb.run(ExecutionRequest(code=escape, budgets=Budgets(wall_s=3)))
print("setsid escapee:", r.exit_reason, "| swept:", r.swept or "nothing (no per-execution UID here)")
for note in r.notes:
    print("  note:", note)

# %% [markdown]
# ## Worked example 4 — how the limits are applied (and why not `preexec_fn`)
# The `preexec_fn` of `subprocess` runs Python code in the child between `fork` and `exec`. In a parent with other
# threads (a proxy server, a listener, a thread pool), that can deadlock the child, and the docs say so. Thus the
# sandbox asks `Popen` to do the privileged part in C: `user=`, `group=`, `extra_groups=[]`,
# `start_new_session=True`. Then the interpreter of the child itself lowers the limits (`executor._LAUNCHER`)
# before it executes the code. A process needs no privilege to lower a limit, and an unprivileged process cannot
# raise its hard limit again.
#
# The root caveat: `RLIMIT_NPROC` counts **every task of the real UID**, and the kernel **ignores it for uid 0**.
# Thus, as root without a UID switch, the sandbox does not stop a fork bomb. Root can even raise its own limits.

# %%
from sandboxcore import rlimits_for
for name, soft, hard in rlimits_for(Budgets(), nproc=16):
    print(f"  {name:14} soft={soft:<10} hard={hard}")
rep = ProcessSandbox(SandboxConfig(drop_to_uid=None, drop_to_gid=None)).isolation_report()
print("without a UID switch here: nproc_enforced =", rep["nproc_enforced"],
      "| limits raisable by the code =", rep["limits_raisable_by_code"])

# %% [markdown]
# ## Exercise 2.1 — set the resource limits in the child
# Implement `child_limits(budgets)`. It returns the list of `(resource name, soft, hard)` tuples for file size,
# address space, open files and CPU (names like `"RLIMIT_CPU"`). File size, address space and open files get soft ==
# hard, in bytes where the budget is in MiB. **CPU gets soft one below hard**. Thus the soft limit raises a
# `SIGXCPU` that the code can catch, before the `SIGKILL` of the hard limit.

# %% exercise
def child_limits(budgets):
    ### BEGIN SOLUTION
    mib = 1024 * 1024
    return [
        ("RLIMIT_FSIZE", budgets.file_mb * mib, budgets.file_mb * mib),
        ("RLIMIT_AS", budgets.memory_mb * mib, budgets.memory_mb * mib),
        ("RLIMIT_NOFILE", budgets.open_files, budgets.open_files),
        ("RLIMIT_CPU", int(budgets.cpu_s), int(budgets.cpu_s) + 1),
    ]
    ### END SOLUTION

# %% check
b = Budgets(cpu_s=3, file_mb=4, memory_mb=320, open_files=32)
mine = {n: (s, h) for n, s, h in child_limits(b)}
lib = {n: (s, h) for n, s, h in rlimits_for(b, nproc=None)}
assert mine["RLIMIT_CPU"][0] < mine["RLIMIT_CPU"][1], "CPU soft must be below hard so SIGXCPU fires first"
assert all(mine[n] == lib[n] for n in mine), {n: (mine[n], lib[n]) for n in mine if mine[n] != lib[n]}
print("✅ your limits match the sandbox's; CPU is a warning at", mine["RLIMIT_CPU"][0], "s, then a kill")

# %% [markdown]
# ## Exercise 2.2 — map a real child's return code to an exit reason
# When a signal stops a child, `subprocess` returns `-signum`. When the child exits, it returns the status. Write
# `reason_for_returncode(rc)` in the vocabulary of the contract (`contract.EXIT_REASONS`). Answer these questions:
#
# - What does the *soft* signal of the CPU limit mean?
# - What does the file-size signal mean, and what is any other signal?
# - What is a clean exit, and what is any non-zero status (before you read stderr)?
#
# The check kills real children with real signals. Then it compares your answer with the classifier of the
# sandbox itself.

# %% exercise
import signal


def reason_for_returncode(rc):
    ### BEGIN SOLUTION
    if rc == 0:
        return "ok"
    if rc < 0:
        return {signal.SIGXCPU: "cpu_time", signal.SIGXFSZ: "file_too_large"}.get(-rc, "killed")
    return "error"
    ### END SOLUTION

# %% check
import subprocess
import sys
from sandboxcore import executor
for sig in (signal.SIGXCPU, signal.SIGXFSZ, signal.SIGKILL, signal.SIGSEGV):
    # a shell child that signals itself (a Python child would ignore SIGXFSZ and see EFBIG instead)
    rc = subprocess.run(["sh", "-c", f"kill -{signal.Signals(sig).name[3:]} $$"]).returncode
    want = executor._classify(rc, "", 0.0, Budgets(cpu_s=5))[0]
    assert reason_for_returncode(rc) == want, f"rc={rc} ({signal.Signals(sig).name}): the sandbox says {want!r}"
for code, want in (("pass", "ok"), ("raise SystemExit(3)", "error")):
    assert reason_for_returncode(subprocess.run([sys.executable, "-c", code]).returncode) == want
print("✅ exit reasons let the model recover (‘cpu_time: use a cheaper algorithm’) instead of retrying blind")

# %% [markdown]
# ## Exercise 2.3 — predict, then measure, the sleeper
# A program that only does `time.sleep(30)` uses no CPU. Predict the exit reason under `Budgets(cpu_s=5,
# wall_s=1)`, with a generous CPU budget and a tight wall budget. Then run it and confirm.

# %% exercise
predicted = None
### BEGIN SOLUTION
predicted = "wall_timeout"
### END SOLUTION

# %% check
r = sb.run(ExecutionRequest(code="import time; time.sleep(30)", budgets=Budgets(cpu_s=5, wall_s=1)))
print(f"predicted {predicted}, measured {r.exit_reason} in {r.usage.wall_s:.2f}s (measured on this machine)")
assert predicted == r.exit_reason and r.usage.wall_s < 3
print("✅ CPU limits never fire on a sleeper; the wall clock does")

# %% [markdown]
# ## Exercise 2.4 — does the escapee survive?
# For this machine, predict if a `setsid()` escapee (the code of worked example 3) is **still alive after the call
# returns**. Predict it for two configurations: the default sandbox, and `drop_to_uid=None`. Derive the answer from
# the `isolation_report()` of each sandbox, not from a guess. The check runs both and looks for the process.

# %% exercise
def survives(sandbox):
    ### BEGIN SOLUTION
    return not sandbox.isolation_report()["escapes_swept"]
    ### END SOLUTION

# %% check
import time as _t
from sandboxcore import PROBES_BY_NAME, run_probe
for label, box in (("default", ProcessSandbox()),
                   ("drop_to_uid=None", ProcessSandbox(SandboxConfig(drop_to_uid=None, drop_to_gid=None)))):
    v = run_probe(PROBES_BY_NAME["escape_session"], box)
    print(f"{label:17} predicted survives={survives(box)!s:5} measured: {v.detail}")
    assert survives(box) == v.leaked
print("✅ the process group is advisory; only the UID (or a cgroup / PID namespace) finds an escapee")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "The process sandbox is the T0 boundary. It is a separate child that starts from a
# clean environment in a throwaway 0700 workspace, so no credentials come with it. Before the code runs, the child
# lowers its own POSIX limits: CPU seconds, address space, open files, file size and the process count.
#
# "I hold a
# wall-clock deadline and read its output as it streams. I keep a budget's worth and kill the run if it floods. Thus
# neither a sleeper nor a print loop can hurt the caller.
#
# "On a host where I can, each execution also gets its own unprivileged UID. That UID makes the process limit bite.
# It keeps my files unreadable, and a HOME that points to a different directory does not. It also lets me kill a
# process that left the group with setsid().
#
# "I say clearly what this does not do. It does not block the network, and it shares the kernel. Without the UID, my
# files and escapees have no protection. For a stronger boundary I move up the ladder: a container, then gVisor, then a
# microVM. But the contract and the limits have the same shape."
#
# **Drill questions**
# 1. *Why a separate process, not a restricted interpreter?* In-process restrictions share the state of the
#    interpreter, and the code can reach around them. A separate process with limits that the OS enforces is the
#    first real boundary.
# 2. *`RLIMIT_CPU` soft == hard against soft < hard: what changes?* When they are equal, the code gets an
#    immediate `SIGKILL`. When soft is below hard, the soft limit raises a `SIGXCPU` that the code can catch. The
#    hard limit guarantees a `SIGKILL`.
# 3. *You set a 2-second CPU limit and the code hangs for a minute. Why?* The code sleeps or waits for I/O, and it
#    uses no CPU. Only the wall-clock timeout stops it. Always run both.
# 4. *We kill the process group on timeout. Can anything survive?* Yes. A descendant that called `setsid()` is in
#    a different group. Kill by a per-execution UID after the run. Or use a cgroup or PID namespace (a container)
#    that stops with the sandbox.
