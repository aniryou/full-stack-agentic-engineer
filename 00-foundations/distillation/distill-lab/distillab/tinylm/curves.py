"""curves.py — the recorded tiny-distillation run, for machines without torch, and helpers to show any run.

One idea: a training curve is evidence only with its provenance. ``train.run()`` returns curves labelled
``"measured"`` on the machine that ran them; :func:`load_recorded` returns a copy of one such run, recorded on a
CPU with this code, and labels it **illustrative** — it shows the shape to expect (the four students separating
on accuracy, agreement and length), not what your machine will produce.
"""
from __future__ import annotations

import json
from importlib import resources

from ..report import curve, table

RECORDED = "tinylm_recorded_run.json"
COLS = ["accuracy", "full", "length", "agree", "kl", "rkl", "accept"]


def load_recorded() -> dict:
    data = json.loads(resources.files("distillab.assets").joinpath(RECORDED).read_text())
    data["source"] = "recorded run (illustrative): " + data.get("machine", "?")
    return data


def label(run: dict) -> str:
    return "MEASURED on this machine" if run.get("source") == "measured" else run.get("source", "")


def summary_rows(run: dict) -> list:
    """One row per model: the teacher's evaluation and each student's final evaluation."""
    rows = [{"model": "teacher", **{c: run["teacher"]["eval"][c] for c in COLS}}]
    for m, s in run["students"].items():
        rows.append({"model": m, **{c: s["final"][c] for c in COLS}})
    return rows


def show(run: dict, curves: bool = True) -> str:
    """Text summary: sizes, data, the final table and (optionally) each student's curve."""
    c = run["config"]
    sq = run["data"]["seqkd"]
    out = [f"[{label(run)}] teacher {run['params']['teacher']:,} parameters, student {run['params']['student']:,}; "
           f"{c['student_steps']} steps x batch {c['student_batch']} per student; timings (s) {run['timing_s']}",
           f"labelled set: {run['data']['labelled']} problems, {run['data']['labelled_full']:.0%} with the full working. "
           f"SeqKD data: {sq['kept']} of {sq['generated']} teacher samples kept by the verifier "
           f"(full scratchpad {sq['full_before']:.0%} -> {sq['full_after']:.0%})",
           table(summary_rows(run), ["model"] + COLS, "Final evaluation (held-out problems)")]
    if curves:
        for m, s in run["students"].items():
            out.append(curve(s["curve"], "step", ["accuracy", "full", "agree", "kl"], f"{m}"))
    return "\n\n".join(out)


EXPERIMENTS = "tinylm_recorded_experiments.json"


def load_experiments() -> dict:
    """Recorded follow-up runs from the same teacher (GKD with β = 1 and β = 0, SeqKD without the verifier filter),
    labelled illustrative like :func:`load_recorded`."""
    data = json.loads(resources.files("distillab.assets").joinpath(EXPERIMENTS).read_text())
    for e in data["experiments"].values():
        e["source"] = "recorded run (illustrative): " + data.get("machine", "?")
    return data["experiments"]
