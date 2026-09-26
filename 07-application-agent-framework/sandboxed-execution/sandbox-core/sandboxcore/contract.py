"""The execution contract: what a caller asks a sandbox for, and what it gets back.

The one idea: a sandbox is an API with a contract, not a place. The request carries the code, its
inputs and a **budget for every resource the code could exhaust** (CPU seconds, wall seconds, memory,
processes, file size, disk, output bytes, open files). The result carries truncated output, an **exit
reason** the caller can act on (and where that reason came from — the parent's own measurement, a kernel
signal, or the untrusted program's say-so), what was actually used, and hashes that make the call
auditable. The idempotency key follows the scaling primer's recipe (turn, step, call index and a hash of the
arguments), so a redelivered step gets the stored result back instead of running twice — within the limits
``ResultStore`` states.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


def digest(obj: Any) -> str:
    """sha256 of canonical JSON, first 16 hex chars: the identity lab's `args_digest` recipe."""
    canonical = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def idempotency_key(turn_id: str, step: int, call_index: int, args: dict) -> str:
    """Scaling primer §5.4: key every side effect by turn, step, call index and a hash of the arguments."""
    return f"{turn_id}:{step}:{call_index}:{digest(args)}"


@dataclass(frozen=True)
class Budgets:
    cpu_s: float = 2.0          # CPU seconds (RLIMIT_CPU): stops busy loops, never sleepers
    wall_s: float = 5.0         # wall clock, enforced from outside: stops sleepers and hung I/O
    memory_mb: int = 256        # address space (RLIMIT_AS); below ~256 MiB `import numpy` fails
    pids: int = 16              # the run's tasks (processes and threads), the main one included:
                                # RLIMIT_NPROC under its own UID, else the parent's count of its tree
    file_mb: int = 8            # the largest single file (RLIMIT_FSIZE): not a disk quota
    disk_mb: int = 16           # the workspace in total, measured after the run
    output_bytes: int = 16_384  # kept per stream; the rest is counted and dropped as it streams
    open_files: int = 64        # RLIMIT_NOFILE
    output_kill_bytes: int = 1 << 20   # total output (both streams) that ends the run: exit output_limit

    def exceeding(self, cap: "Budgets") -> list[str]:
        """The fields of this budget that are larger than `cap` (a policy's maximum)."""
        return [k for k, v in asdict(self).items() if v > getattr(cap, k)]


# One vocabulary for every rung (the lab's wrapper returns the same names). PRIMER §3.
EXIT_REASONS = ("ok", "error", "wall_timeout", "cpu_time", "memory", "file_too_large", "pids",
                "open_files", "output_limit", "disk_limit", "killed", "denied", "sandbox_error")
# Where an exit reason came from: the parent enforced or measured it; the kernel delivered a signal (which
# the code could also send itself); or the untrusted code's exit status / stderr said so (forgeable).
REASON_SOURCES = ("parent", "signal", "code")

HINTS = {   # what the model is told, so it can recover instead of retrying blindly (scaling primer §5.6)
    "error": "The code raised an exception; read stderr, fix the code and try again.",
    "wall_timeout": "The code ran past its wall-clock budget; do less work or avoid waiting on I/O.",
    "cpu_time": "The code used its whole CPU budget; use a cheaper algorithm or smaller input.",
    "memory": "The code ran out of memory; process the data in chunks.",
    "file_too_large": "A file grew past the per-file limit; write less data.",
    "pids": "The code tried to start too many processes; do the work in one process.",
    "open_files": "The code opened too many files at once; close files after use.",
    "output_limit": "The code printed far more than the output budget and was stopped; print a summary.",
    "disk_limit": "The workspace grew past its disk budget; write less to disk.",
    "killed": "The sandbox killed the code; do not retry the same code.",
    "denied": "Policy refused this execution; do not retry it unchanged.",
    "sandbox_error": "The sandbox itself failed to start; retrying the same call may succeed.",
}


@dataclass
class ExecutionRequest:
    code: str
    budgets: Budgets = field(default_factory=Budgets)
    files: dict[str, str] = field(default_factory=dict)     # inputs written into the workspace
    env: dict[str, str] = field(default_factory=dict)       # explicit, non-secret settings only
    egress: tuple[str, ...] = ()     # hosts the code SAYS it needs: input to policy review, not enforcement
    principal: str = "agent:unknown"                         # who asked: the agent's identity
    idempotency_key: str | None = None


@dataclass
class Usage:
    cpu_s: float = 0.0          # the leader and the descendants it reaped (wait4), not killed grandchildren
    wall_s: float = 0.0
    max_rss_mb: float = 0.0     # peak resident memory of the leader (wait4 ru_maxrss)
    disk_bytes: int = 0
    stdout_bytes: int = 0       # bytes the code wrote, not bytes kept
    stderr_bytes: int = 0


@dataclass
class ExecutionResult:
    exit_reason: str
    returncode: int | None
    stdout: str
    stderr: str
    truncated: bool
    usage: Usage
    artifacts: dict[str, str] = field(default_factory=dict)   # workspace path -> digest of its bytes
    over_budget: list[str] = field(default_factory=list)      # budgets found exceeded after the run
    isolation: dict[str, Any] = field(default_factory=dict)   # what was actually enforced, and where
    reason_source: str = "parent"                             # REASON_SOURCES: how far to trust exit_reason
    swept: dict[str, Any] = field(default_factory=dict)       # processes/files of the sandbox UID removed after
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_reason == "ok" and not self.over_budget

    def result_hash(self) -> str:
        return digest([self.exit_reason, self.stdout, self.stderr, self.artifacts])

    def as_tool_result(self) -> dict[str, Any]:
        """agent-core's tool-result shape: the exit reason is the `error`, truncated output is `data`."""
        data = {"stdout": self.stdout, "stderr": self.stderr[-2000:], "truncated": self.truncated,
                "artifacts": self.artifacts, "usage": asdict(self.usage)}
        if self.ok:
            return {"ok": True, "data": data}
        reason = self.exit_reason if self.exit_reason != "ok" else "disk_limit"
        return {"ok": False, "error": reason, "data": data,
                "message": f"execution ended with {reason}" + (f" ({', '.join(self.over_budget)})"
                                                                if self.over_budget else ""),
                "hint": HINTS.get(reason, "Do not retry the same code.")}


def denied(reasons: list[str]) -> ExecutionResult:
    """The result of an execution that policy refused: nothing ran."""
    return ExecutionResult("denied", None, "", "; ".join(reasons), False, Usage())


class InFlight(RuntimeError):
    """Another delivery of the same key is running right now; the caller should retry later, not run it."""


class ResultStore:
    """Results by idempotency key, so a redelivered step returns the stored result instead of running again.

    What it guarantees: within **one process**, a second delivery of a key — after the first finished, or
    while it is still running (it raises ``InFlight`` instead of starting a second run) — does not execute
    the code again. What it does not: survive a crash (the dict is in memory; a durable store would write
    the ``PENDING`` claim and the result to a database with a conditional put), dedupe across replicas, or
    make the code's own side effects idempotent — end to end that needs the key forwarded downstream (e.g.
    the egress proxy adding an ``Idempotency-Key`` header), as the scaling primer's §5.4 recipe does.
    """

    PENDING = object()

    def __init__(self) -> None:
        self.results: dict[str, Any] = {}
        self._lock = threading.Lock()

    def run_once(self, key: str, run: Callable[[], ExecutionResult]) -> tuple[ExecutionResult, bool]:
        """Return (result, replayed): replayed is True when the key was seen and nothing ran."""
        with self._lock:
            seen = self.results.get(key)
            if seen is self.PENDING:
                raise InFlight(f"key {key!r} is already running")
            if seen is not None:
                return seen, True
            self.results[key] = self.PENDING          # claim the key before running
        try:
            result = run()
        except BaseException:
            with self._lock:
                self.results.pop(key, None)           # nothing stored: a later delivery may run it
            raise
        with self._lock:
            self.results[key] = result
        return result, False
