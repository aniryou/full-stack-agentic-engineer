#!/usr/bin/env python3
"""make_figures.py: the charts of the topic's PRIMER.md that this core computes (FIGURE-STYLE.md §7).

Each chart is a spec (a plain dict) that one builder below computes with the core's own functions, from the
inputs the primer names: four empty 8-GPU nodes and 1-GPU pods of 8 cores and 64 GiB (§3.4), the illustrative
reclaim rate of 0.005 per node-hour with a 24-hour job and a 15-minute restart (§7.2), and a 16-node gang with a
300 s boot, a 3% grant chance per minute and seed 0 (§7.3). No value is copied out of the prose; the marks are the
primer's pinned points, labelled the way the primer prints them (``tests/test_docs.py`` holds the same format
strings). ``tools/orchestration/svgchart.py`` renders each spec into ``../figures/<name>.svg`` beside the primer,
and ``tests/test_figures.py`` checks that every committed chart is what the core computes now.

    python3 tools/make_figures.py          # rewrite every chart (the output is the same bytes on every machine)

Standard library only.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from gpusched import GPU, NodePool, Scheduler, gpu_pod, make_cluster  # noqa: E402
from gpusched.autoscaler import expected_runtime_h, gang_survival, provision  # noqa: E402
from gpusched.scheduler import fragmentation, stranded_gpus  # noqa: E402

TOPIC = LAB.parent
FIG_DIR = TOPIC / "figures"
DOC = TOPIC / "PRIMER.md"
REPO = next(p for p in LAB.parents if (p / "tools" / "orchestration" / "svgchart.py").is_file())


def _load_svgchart():
    path = REPO / "tools" / "orchestration" / "svgchart.py"
    spec = importlib.util.spec_from_file_location("svgchart", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


svgchart = _load_svgchart()


def sig(v: float, digits: int = 6) -> float:
    """A value rounded to `digits` significant digits: the chart needs no more, and the file stays small."""
    return float(f"{v:.{digits}g}")


# ---------------------------------------------------------------- §3.4: fragmentation, measured

DEFAULT_SCORE = ("LeastAllocated", (("cpu", 1), ("memory", 1)))          # upstream default: spread on cpu+memory
BINPACK_SCORE = ("MostAllocated", (("cpu", 1), ("memory", 1), (GPU, 5)))  # the primer's gpu-binpack profile
NODES, PODS, SHAPE = 4, 16, 8          # four empty 8-GPU nodes, sixteen 1-GPU pods, then one 8-GPU pod
LAST = 24                              # from the 25th pod on only one node has room, and the two lines coincide


def arrivals(strategy: str, resources) -> tuple[list[int], tuple[int, float]]:
    """Stranded GPUs for an 8-GPU pod after each 1-GPU pod (8 cores, 64 GiB) that the scheduler binds, up to
    LAST pods; and (stranded, fragmentation) as they were after the primer's sixteen pods."""
    cluster = make_cluster(hosts=NODES)
    sched = Scheduler(cluster, strategy, resources)
    out, at_example = [stranded_gpus(cluster, SHAPE)], None
    for i in range(LAST):
        sched.submit(gpu_pod(f"p{i}", 1))
        sched.run()
        out.append(stranded_gpus(cluster, SHAPE))
        if i + 1 == PODS:
            at_example = (stranded_gpus(cluster, SHAPE), fragmentation(cluster, SHAPE))
    return out, at_example


def spec_stranded_arrivals() -> dict:
    """stranded_gpus(cluster, 8) as 1-GPU pods arrive one at a time, under the default score and under
    MostAllocated with a GPU weight of 5, with the primer's state after sixteen pods marked."""
    spread, (s16, f16) = arrivals(*DEFAULT_SCORE)
    packed, (p16, g16) = arrivals(*BINPACK_SCORE)
    xs = list(range(len(spread)))
    return {
        "kind": "line",
        "title": ("Stranded GPUs for an 8-GPU pod as 1-GPU pods arrive one at a time at four empty 8-GPU nodes, "
                  "under the default score and under MostAllocated with a GPU weight of 5."),
        "width": 880, "height": 380,
        "x": {"label": "1-GPU pods placed (8 cores, 64 GiB each)", "min": 0, "max": xs[-1],
              "ticks": list(range(0, xs[-1] + 1, 4))},
        "y": {"label": "stranded GPUs for an 8-GPU pod", "min": -5, "max": 32,
              "ticks": [0, 4, 8, 12, 16, 20, 24, 28, 32]},
        "series": [
            {"name": "default (LeastAllocated, cpu+memory)", "x": xs, "y": spread, "style": "warm", "markers": True,
             "dash": True},
            {"name": "MostAllocated, GPU weight 5", "x": xs, "y": packed, "style": "accent", "markers": True},
        ],
        "marks": [
            {"kind": "vline", "x": PODS, "label": f"{PODS} pods, then one 8-GPU pod", "style": "muted",
             "anchor": "end"},
            {"kind": "point", "x": PODS, "y": s16, "label": f"{s16} stranded, {f16:.0%}", "style": "warm"},
            {"kind": "point", "x": PODS, "y": p16, "label": f"{p16} stranded, {g16:.0%}", "style": "accent",
             "dy": 18},
        ],
        "legend": "top-right",
        "note": "computed by gpusched.scheduler.Scheduler and stranded_gpus (k8s-gpu-core)",
    }


# ---------------------------------------------------------------- §7.2: Spot and gangs

RATE, WORK_H, RESTART_H = 0.005, 24, 0.25     # illustrative reclaims per node-hour, the job, R = 15 min
TABLE_ROWS = (1, 4, 16)                       # the rows of the primer's table


def spec_spot_gang() -> dict:
    """expected_runtime_h(24, N, 0.005, 0.25) for gangs of 1 to 16 nodes, with gang_survival() in the labels of
    the table's three rows."""
    ns = list(range(1, TABLE_ROWS[-1] + 1))
    hours = [sig(expected_runtime_h(WORK_H, n, RATE, RESTART_H)) for n in ns]

    def label(n):
        h, s = expected_runtime_h(WORK_H, n, RATE, RESTART_H), gang_survival(n, WORK_H, RATE)
        return f"{n} node{'s' if n > 1 else ''}: {h:.1f} h, survives {s:.1%}"
    place = {1: {"anchor": "start", "dy": 24}, 4: {"anchor": "start", "dy": 16}, 16: {"anchor": "end", "dy": -6}}
    return {
        "kind": "line",
        "title": ("Expected wall-clock hours of a 24-hour job on Spot against the number of nodes in the gang, when "
                  "every reclaim restarts the whole gang, at 0.005 reclaims per node-hour and a 15-minute restart."),
        "width": 880, "height": 400,
        "x": {"label": "nodes in the gang", "min": 1, "max": ns[-1], "ticks": [1, 4, 8, 12, 16]},
        "y": {"label": "expected wall-clock (h)", "min": 0, "max": 84, "ticks": [0, 20, 40, 60, 80]},
        "series": [
            {"name": "expected wall-clock", "x": ns, "y": hours, "style": "accent", "markers": True},
        ],
        "marks": [
            {"kind": "hline", "y": WORK_H, "label": f"{WORK_H} h of work", "style": "muted"},
        ] + [
            {"kind": "point", "x": n, "y": sig(expected_runtime_h(WORK_H, n, RATE, RESTART_H)), "label": label(n),
             "style": "accent", **place[n]} for n in TABLE_ROWS
        ],
        "legend": "none",
        "note": "computed by gpusched.autoscaler.expected_runtime_h and gang_survival (k8s-gpu-core); illustrative rate",
    }


# ---------------------------------------------------------------- §7.3: queued provisioning

GANG, POOL_KW, TICK_S, SEED = 16, dict(gpus_per_node=8, boot_s=300, stockout=0.97), 60, 0


def billed(run: dict) -> tuple[list[float], list[int]]:
    """The nodes a pool has created (and bills) against time in hours, from 0 to the gang start: a step series."""
    xs, ys = [0.0], [0]
    for t in sorted(set(run["created_s"])):
        xs.append(sig(t / 3600))
        ys.append(sum(c <= t for c in run["created_s"]))
    xs.append(sig(run["gang_start_s"] / 3600))
    ys.append(len(run["created_s"]))
    return xs, ys


def spec_queued_provisioning() -> dict:
    """provision() for a 16-node gang on an ordinary pool and on a queued pool, seed 0: the nodes each pool bills
    while the gang waits for its last node. The area under each step line is its waiting_node_h."""
    ordinary = provision(NodePool("a3", **POOL_KW), GANG, tick_s=TICK_S, seed=SEED)
    queued = provision(NodePool("a3-queued", queued=True, **POOL_KW), GANG, tick_s=TICK_S, seed=SEED)
    ox, oy = billed(ordinary)
    qx, qy = billed(queued)
    start_h = ordinary["gang_start_s"] / 3600
    return {
        "kind": "line",
        "title": ("Nodes billed against time while a 16-node gang waits for its last node, in one simulated run: "
                  "the ordinary pool creates each node at its grant, and the queued pool creates all 16 nodes together."),
        "width": 880, "height": 360,
        "x": {"label": "time after the request (h)", "min": 0, "max": math.ceil(start_h * 2) / 2,
              "ticks": [0, 0.5, 1, 1.5, 2, 2.5]},
        "y": {"label": "nodes billed", "min": 0, "max": 22, "ticks": [0, 4, 8, 12, 16]},
        "series": [
            {"name": "ordinary pool", "x": ox, "y": oy, "style": "warm", "step": True, "fill": True},
            {"name": "queued pool", "x": qx, "y": qy, "style": "accent", "step": True, "fill": True},
        ],
        "marks": [
            {"kind": "vline", "x": sig(start_h), "label": f"gang start: {start_h:.2f} h", "style": "muted",
             "anchor": "end"},
            {"kind": "text", "x": sig(start_h / 2), "y": 8, "style": "warm",
             "text": (f"ordinary pool: {ordinary['waiting_node_h']:.1f} node-hours "
                      f"({ordinary['waiting_gpu_h']:.0f} GPU-hours)")},
            {"kind": "text", "x": sig(qx[1] - 0.04), "y": 17.5, "style": "accent", "anchor": "end",
             "text": f"queued pool: {queued['waiting_node_h']:.1f} node-hours"},
        ],
        "legend": "top-left",
        "note": f"computed by gpusched.autoscaler.provision, seed {SEED} (k8s-gpu-core); simulated",
    }


FIGURES = {
    "stranded-gpus-as-pods-arrive.svg": spec_stranded_arrivals,
    "spot-gang-wall-clock.svg": spec_spot_gang,
    "queued-provisioning-billed-nodes.svg": spec_queued_provisioning,
}


def main() -> None:
    for name, build in FIGURES.items():
        out = svgchart.write_figure(FIG_DIR / name, build())
        print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
