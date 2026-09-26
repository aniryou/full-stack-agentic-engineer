"""sandboxcore — run model-generated code without ambient authority (T0, standard library only)."""
from .contract import (Budgets, ExecutionRequest, ExecutionResult, ResultStore, Usage,
                       denied, digest, idempotency_key)
from .executor import ProcessSandbox, SandboxConfig, UnsafeExecutor, running_as_root
from .threats import PROBES, PROBES_BY_NAME, LoopbackTrap, Probe, Verdict, run_probe

__all__ = [
    "Budgets", "ExecutionRequest", "ExecutionResult", "Usage", "ResultStore",
    "denied", "digest", "idempotency_key",
    "ProcessSandbox", "SandboxConfig", "UnsafeExecutor", "running_as_root",
    "PROBES", "PROBES_BY_NAME", "Probe", "Verdict", "LoopbackTrap", "run_probe",
]
