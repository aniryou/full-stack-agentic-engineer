# %% [markdown]
# # 02 · A process sandbox: rlimits, timeouts, truncation — and what it cannot stop
#
# **Tier:** T0 — laptop / Colab CPU / CI, free, about a minute. Real container and microVM isolation is the
# lab (`../sandbox-lab`, notebook 01 hardened containers); the concepts are all here.
#
# ## The one-minute version
# In-process restrictions are not a boundary — code that shares your interpreter can undo them — so the
# first real boundary is a **separate process** you start clean and bound from outside. A process sandbox
# does five things: (1) a **clean environment**, so no inherited credentials; (2) an **ephemeral workspace**
# with home pointed at it, so no private files; (3) **POSIX resource limits** set in the child before it
# runs — CPU seconds, address space, processes, file size, open files; (4) a **wall-clock kill** of the
# whole process group, so a sleeper or a backgrounded grandchild dies; (5) **output truncation**, so a flood
# cannot fill your logs. It also states plainly what it does *not* stop — the network — and the sharp edges:
# `RLIMIT_NPROC` is per real UID and **ignored for root**, `RLIMIT_CPU` measures CPU not wall time, and a
# grandchild survives a naive `subprocess` timeout. Primer: `../PRIMER.md` §2 (the isolation ladder), §3
# (the execution contract). Numbers here are **measured on this machine** and labelled so.

# %%
from sandboxcore import Budgets, ExecutionRequest, ProcessSandbox, SandboxConfig, running_as_root

sb = ProcessSandbox()
print("running as root:", running_as_root())
print("this sandbox enforces:", sb.isolation_report())

# %% [markdown]
# ## Worked example 1 — a clean environment and an isolated home
# The child cannot see the parent's environment or home. We plant a token in *our* environment and ask the
# child to read it; it comes back absent.

# %%
import os
os.environ["DEMO_TOKEN"] = "sk-live-do-not-leak"
r = sb.run(ExecutionRequest(
    code="import os, pathlib; "
         "print('token:', os.environ.get('DEMO_TOKEN', 'ABSENT')); "
         "print('~/.ssh exists:', pathlib.Path(os.path.expanduser('~/.ssh')).exists()); "
         "print('cwd:', os.getcwd())",
    budgets=Budgets(cpu_s=1, wall_s=3)))
print(r.stdout)
del os.environ["DEMO_TOKEN"]

# %% [markdown]
# The token is `ABSENT`, `~/.ssh` does not exist (home is the throwaway workspace), and the cwd is a fresh
# temp dir that is deleted when the call returns. Nothing the code writes survives, and nothing it reads was
# yours.
#
# ## Worked example 2 — the resource limits, one at a time
# Each budget maps to a POSIX `setrlimit` applied in the child (and, on Linux, an address-space limit). The
# exit reason tells the caller *why* it stopped, so the model can recover.

# %%
cases = [
    ("busy loop", "x=0\nwhile True:\n x+=1", Budgets(cpu_s=1, wall_s=10)),
    ("sleep forever", "import time; time.sleep(60)", Budgets(cpu_s=5, wall_s=1)),
    ("huge file", "import os;open(os.path.join(os.environ['SANDBOX_WORKDIR'],'b'),'wb').write(b'x'*(9<<20))",
     Budgets(file_mb=1, cpu_s=2, wall_s=5)),
    ("print flood", "print('A'*200000)", Budgets(output_bytes=2048, cpu_s=2, wall_s=5)),
]
for label, code, b in cases:
    r = sb.run(ExecutionRequest(code=code, budgets=b))
    extra = " (truncated)" if r.truncated else ""
    print(f"{label:14} -> exit_reason={r.exit_reason:14} cpu={r.usage.cpu_s:.2f}s wall={r.usage.wall_s:.2f}s{extra}")

# %% [markdown]
# ## Worked example 3 — the grandchild that a naive timeout leaves alive
# `subprocess.run(timeout=...)` kills only the direct child. A program that backgrounds a process leaves the
# grandchild re-parented to init. The sandbox starts a new **session** (`setsid`) and kills the whole
# **process group** on timeout, so nothing survives.

# %%
grandchild = ("import subprocess, sys, time; "
              "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
              "time.sleep(30)")
r = sb.run(ExecutionRequest(code=grandchild, budgets=Budgets(cpu_s=5, wall_s=1)))
print("exit_reason:", r.exit_reason, "| wall:", round(r.usage.wall_s, 2), "s (killed near the budget, not 30 s)")

# %% [markdown]
# ## Worked example 4 — the root caveat, stated out loud
# `RLIMIT_NPROC` limits processes **per real UID, system-wide**, and is **ignored for uid 0**. So a fork
# bomb is only stopped if the sandbox can drop to an unprivileged UID first — which needs privilege to do.
# Colab and many CI containers run as root; the sandbox must report this rather than pretend.

# %%
report = ProcessSandbox().isolation_report()
print("nproc_enforced here:", report["nproc_enforced"], "(needs a UID drop, which needs root to perform)")
print("network_blocked here:", report["network_blocked"], "(rlimits never touch sockets — see notebook 04)")

# %% [markdown]
# ## Exercise 2.1 — set the resource limits in the child
# Implement `child_limits(budgets)` returning the list of `(resource, soft, hard)` tuples the sandbox should
# apply. Rules: file size, address space and open-files get soft == hard; **CPU gets soft one below hard**
# so the soft limit raises a catchable `SIGXCPU` before the hard limit's `SIGKILL`. Use the `resource`
# module's constants.

# %%
import resource

# %% exercise
def child_limits(budgets):
    ### BEGIN SOLUTION
    lim = [
        (resource.RLIMIT_FSIZE, budgets.file_mb * 1024 * 1024, budgets.file_mb * 1024 * 1024),
        (resource.RLIMIT_AS, budgets.memory_mb * 1024 * 1024, budgets.memory_mb * 1024 * 1024),
        (resource.RLIMIT_NOFILE, budgets.open_files, budgets.open_files),
        (resource.RLIMIT_CPU, int(budgets.cpu_s), int(budgets.cpu_s) + 1),
    ]
    return lim
    ### END SOLUTION

# %% check
lims = dict((r, (s, h)) for r, s, h in child_limits(Budgets(cpu_s=2, file_mb=8, memory_mb=256, open_files=64)))
cpu_soft, cpu_hard = lims[resource.RLIMIT_CPU]
assert cpu_soft < cpu_hard, "CPU soft must be below hard so SIGXCPU fires before SIGKILL"
assert lims[resource.RLIMIT_FSIZE] == (8 * 1024 * 1024, 8 * 1024 * 1024)
assert lims[resource.RLIMIT_AS][0] == 256 * 1024 * 1024
print("✅ limits set; CPU soft", cpu_soft, "< hard", cpu_hard, "so the CPU limit is a warning, then a kill")

# %% [markdown]
# ## Exercise 2.2 — map a signal to an exit reason
# When the child dies from a signal, `subprocess` returns a negative code (`-signum`). Write
# `reason_for_returncode(rc)`: `0` → `"ok"`, a negative code for `SIGXCPU` → `"cpu_time"`, for `SIGKILL` →
# `"killed"`, for `SIGXFSZ` → `"file_too_large"`, any other negative → `"killed"`, any positive → `"error"`.

# %%
import signal

# %% exercise
def reason_for_returncode(rc):
    ### BEGIN SOLUTION
    if rc == 0:
        return "ok"
    if rc < 0:
        return {signal.SIGXCPU: "cpu_time", signal.SIGKILL: "killed",
                signal.SIGXFSZ: "file_too_large"}.get(-rc, "killed")
    return "error"
    ### END SOLUTION

# %% check
assert reason_for_returncode(0) == "ok"
assert reason_for_returncode(-int(signal.SIGXCPU)) == "cpu_time"
assert reason_for_returncode(-int(signal.SIGKILL)) == "killed"
assert reason_for_returncode(1) == "error"
print("✅ exit reasons let the model recover (‘cpu_time: use a cheaper algorithm’) instead of retrying blind")

# %% [markdown]
# ## Exercise 2.3 — predict, then measure, the sleeper
# A program that only `time.sleep(30)`s uses no CPU. Predict the exit reason under `Budgets(cpu_s=5,
# wall_s=1)` — CPU budget generous, wall budget tight — then run it and confirm.

# %% exercise
### BEGIN SOLUTION
predicted = "wall_timeout"
### END SOLUTION

# %% check
r = sb.run(ExecutionRequest(code="import time; time.sleep(30)", budgets=Budgets(cpu_s=5, wall_s=1)))
assert predicted == "wall_timeout" == r.exit_reason
assert r.usage.wall_s < 3
print(f"✅ predicted {predicted}, measured {r.exit_reason} in {r.usage.wall_s:.2f}s (measured on this machine)")

# %% [markdown]
# ## Exercise 2.4 — why not `preexec_fn` with threads?
# The sandbox runs `python -I` (isolated mode) rather than a Python `preexec_fn` that does everything. Given
# the note "`preexec_fn` is not safe in the presence of threads — it may deadlock before exec", set
# `safe_when` to the condition under which a `preexec_fn` is safe, from the options.

# %% exercise
options = {"a": "always", "b": "only in a single-threaded parent", "c": "only as root"}
### BEGIN SOLUTION
safe_when = "b"
### END SOLUTION

# %% check
assert safe_when == "b"
print("✅ our proxy and agent may be threaded, so the executor sets limits via the child, not a shared preexec_fn")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "The process sandbox is the T0 boundary: a separate child started from a clean
# environment, with home and the working directory pointed at a throwaway workspace, so no credentials and
# no private files come along. Before the code runs I set POSIX limits in the child — CPU seconds, address
# space, open files, file size, and, when I can drop to an unprivileged UID, the process count — and I run a
# wall-clock timer that kills the whole process group, because CPU limits don't stop a sleeper and a naive
# timeout leaves grandchildren alive. Output is truncated as it streams. I say out loud what this does not
# do: it does not block the network, and `RLIMIT_NPROC` does nothing as root, so fork-bomb protection needs
# a UID drop. For a stronger boundary I move up the ladder — a container, then gVisor, then a microVM — but
# the contract and the limits are the same shape."
#
# **Drill questions**
# 1. *Why a separate process, not a restricted interpreter?* — In-process restrictions share the
#    interpreter's state; the code can reach around them. A separate process with OS-enforced limits is the
#    first real boundary.
# 2. *`RLIMIT_CPU` soft == hard vs soft < hard — what changes?* — Equal gives an immediate `SIGKILL`; soft
#    below hard raises a catchable `SIGXCPU` at the soft limit and guarantees `SIGKILL` at the hard one.
# 3. *You set a 2-second CPU limit and the code hangs for a minute. Why?* — It is sleeping or blocked on I/O,
#    using no CPU. Only the wall-clock timeout stops it; always run both.
