"""sandboxlab — run model-generated code without ambient authority, at every rung of the ladder.

Modules: ``wrapper`` (the execution contract, enforced inside any sandbox), ``process`` (the T0
executors), ``probes`` (attacks against stand-ins, with verdicts), ``docker`` + ``seccomp``
(hardened containers, gVisor), ``k8s`` (pod-per-execution, warm pools, admission policy),
``proxy`` (egress allowlist + credential brokering), ``agent`` (a 07.1 loop with sandboxed tools),
``audit``, ``bench``, ``report``, ``gke``. See README.md.
"""
__version__ = "0.1.0"
