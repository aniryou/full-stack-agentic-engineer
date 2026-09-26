"""curves.py — read a router's load and specialisation from numbers, no torch needed.

One idea: two statistics tell you whether an MoE is healthy for serving. *Load balance* — the
busiest expert's share over the mean (1.0 is perfect; it sets the slowest expert-parallel rank and
the hottest expert's weight traffic) and the count of dead experts (memory with no work).
*Specialisation* — how much of the router's choice a property of the input explains (mutual
information over the entropy of the expert choice: 0 = unrelated, 1 = the property decides the expert).
Ask it of more than one property: in the toy the current token explains most of the choice and the
domain almost none — experts rarely split along the lines a person would name. These
functions read the bundled curves (``fixtures/tinymoe_curves.json``) or a fresh torch run alike.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tinymoe_curves.json"


def load_stats(load: np.ndarray, dead_share: float = 0.02) -> dict:
    """``load``: per-expert share of assignments (sums to 1). Returns max/mean, dead experts
    (share below ``dead_share``) and the entropy in bits, normalised to [0, 1]."""
    p = np.asarray(load, float)
    p = p / p.sum()
    nz = p[p > 0]
    return {"max_over_mean": float(p.max() * len(p)), "dead": int((p < dead_share).sum()),
            "entropy": float(-(nz * np.log2(nz)).sum() / np.log2(len(p)))}


def specialisation(counts) -> float:
    """I(X; expert) / H(expert) from ``[values of X, experts]`` co-occurrence counts: the share of
    the router's choice that X explains (0 = unrelated, 1 = X decides the expert)."""
    c = np.asarray(counts, float)
    pj = c / c.sum()
    px, pe = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
    nz = pj > 0
    mi = float((pj[nz] * np.log2(pj[nz] / (px @ pe)[nz])).sum())
    he = float(-(pe[pe > 0] * np.log2(pe[pe > 0])).sum())
    return mi / he if he else 0.0


def load_bundled(path: Path = FIXTURE) -> dict:
    """The recorded runs: ``{"_label", "task", "runs": [{config, steps, ce, aux, load, domain_expert, ...}]}``."""
    return json.loads(Path(path).read_text())


def table(runs: list[dict]) -> str:
    """One line per run from recorded dicts (bundled or ``dataclasses.asdict(Run)``)."""
    out = [f"{'balance':8s} {'seed':>4s} {'final CE':>9s} {'max/mean':>9s} {'dead':>5s} "
           f"{'by domain':>10s} {'by token':>9s}"]
    for r in runs:
        s = load_stats(r["load"][-1])
        out.append(f"{r['config']['balance']:8s} {r['config']['seed']:>4d} {r['ce'][-1]:9.3f} "
                   f"{s['max_over_mean']:9.2f} {s['dead']:5d} {specialisation(r['domain_expert']):10.2f} "
                   f"{specialisation(r['token_expert']):9.2f}")
    return "\n".join(out)
