"""k8sgpu — Kubernetes for GPUs, hands-on.

Modules, in the order the notebooks use them:

* ``manifests`` — typed builders for Job, JobSet, LeaderWorkerSet, Kueue objects, DRA claims, GKE ComputeClass
* ``lint``      — a GPU pod-spec linter (the mistakes that cost GPU-hours)
* ``machines``  — what a GPU node can give a pod (allocatable, per-GPU share, stranded GPUs)
* ``scenarios`` — the kind lab's objects, steps and answer key
* ``kindsim``   — a predictor for Kueue admission/preemption, TAS placement and kube-scheduler filters
* ``kindlab``   — drive a real kind cluster step by step and compare with the prediction
* ``pending``   — "why is my pod Pending?" from kubectl JSON (fixtures included)
* ``capacity``  — on-demand vs Spot vs flex-start vs reservations; ComputeClass fallback; cold start
* ``gke``       — the GKE manifests and offline readers for the Terraform

Only PyYAML is required. See the lab README for the tiers (T0 offline, kind with Docker, T3 GKE).
"""
from . import capacity, lint, machines, manifests, pending  # noqa: F401

__version__ = "0.1.0"
