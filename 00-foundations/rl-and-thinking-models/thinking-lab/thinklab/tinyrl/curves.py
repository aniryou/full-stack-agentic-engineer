"""curves.py — the recorded tiny-RL run, for machines without torch, and helpers to show any run.

One idea: a training curve is evidence only with its provenance. ``run()`` in :mod:`thinklab.tinyrl.train`
returns curves labelled ``"measured"`` on the machine that ran them; ``load_recorded()`` returns a copy of
one such run, recorded on a CPU with this code, and labels it **illustrative** — it shows the shape to
expect (reward and completion length rising together as the policy learns to use its scratchpad), not
what your machine will produce.
"""
from __future__ import annotations

import json
from importlib import resources

from ..report import curve, table


def load_recorded() -> dict:
    data = json.loads(resources.files("thinklab.data").joinpath("tinyrl_recorded_run.json").read_text())
    data["source"] = "recorded run (illustrative): " + data.get("machine", "?")
    return data


def show(run: dict) -> str:
    """Text summary of a run: SFT loss, accuracy by scratchpad length before/after, the RL curves."""
    label = "MEASURED on this machine" if run.get("source") == "measured" else run.get("source", "")
    k = run["config"]["k"]
    rows = []
    for j in range(k + 1):
        b, a = run["before"], run["after"]
        rows.append({"scratchpad length": j, "share before": b["scratch_hist"][str(j)] if str(j) in b["scratch_hist"] else b["scratch_hist"].get(j, 0),
                     "acc before": b["acc_by_scratch"].get(str(j), b["acc_by_scratch"].get(j)),
                     "share after": a["scratch_hist"].get(str(j), a["scratch_hist"].get(j, 0)),
                     "acc after": a["acc_by_scratch"].get(str(j), a["acc_by_scratch"].get(j))})
    out = [f"[{label}] {run['params']:,} parameters; SFT {run['timing_s']['sft']} s, GRPO {run['timing_s']['rl']} s",
           table(run["sft"], ["step", "loss"], "SFT warm-up (cross-entropy on completion tokens)"),
           table(rows, None, f"Samples by scratchpad length (of {run['config']['eval_prompts']}) and accuracy per length"),
           curve(run["rl"], "step", ["reward", "length", "frac_zero_std", "kl"], "GRPO"),
           f"accuracy {run['before']['accuracy']:.3f} -> {run['after']['accuracy']:.3f}; "
           f"mean completion {run['before']['mean_length']:.2f} -> {run['after']['mean_length']:.2f} tokens"]
    return "\n\n".join(out)
