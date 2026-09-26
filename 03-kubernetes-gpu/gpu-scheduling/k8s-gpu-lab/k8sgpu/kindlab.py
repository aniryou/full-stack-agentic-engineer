"""Drive the kind cluster: apply a scenario step by step, observe, compare with the prediction.

The one idea: a prediction you can check is worth more than a demo you watch. For each step
the runner prints the exact ``kubectl`` command, applies it, waits for Kueue and the
scheduler to settle, then reads back Workload conditions and pod placements and compares
them with ``kindsim``'s prediction for the same step. With ``dry_run=True`` (or no cluster)
nothing is executed: you get the commands and the predictions — the T0 path.

Safety: the runner only talks to the lab's kind context (``kind-gpu-lab``) unless you pass
another one explicitly, and it only deletes objects in the lab's namespaces.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from . import kindsim
from . import manifests as m
from .scenarios import EXPECTED, KUBE_CONTEXT, Scenario, scenario as get_scenario

LAB_NAMESPACES = ["team-a", "team-b", "zoo", "fleet"]
WORKLOAD_KINDS = "jobs.batch,jobsets.jobset.x-k8s.io,leaderworkersets.leaderworkerset.x-k8s.io"
POD_OWNER_LABELS = {"Job": "batch.kubernetes.io/job-name", "JobSet": "jobset.sigs.k8s.io/jobset-name",
                    "LeaderWorkerSet": "leaderworkerset.sigs.k8s.io/name"}
LWS_GROUP_LABEL = "leaderworkerset.sigs.k8s.io/group-index"


@dataclass
class Kubectl:
    context: str = KUBE_CONTEXT
    dry_run: bool = False
    echo: Callable[[str], None] = print
    binary: str = "kubectl"
    commands: list[str] = field(default_factory=list)

    def argv(self, *args: str) -> list[str]:
        return [self.binary, "--context", self.context, *args]

    def run(self, *args: str, stdin: str | None = None, check: bool = True, quiet: bool = False) -> str:
        line = "kubectl " + " ".join(args)
        self.commands.append(line)
        if not quiet:
            self.echo(f"$ {line}")
        if self.dry_run:
            return ""
        p = subprocess.run(self.argv(*args), input=stdin, capture_output=True, text=True, timeout=120)
        if check and p.returncode != 0:
            raise RuntimeError(f"{line} failed: {p.stderr.strip()}")
        return p.stdout

    def get_json(self, *args: str) -> dict:
        out = self.run("get", *args, "-o", "json", quiet=True)
        return json.loads(out) if out else {"items": []}


def cluster_status(context: str = KUBE_CONTEXT) -> tuple[bool, str]:
    """(usable, explanation): kubectl present, context reachable, fake GPUs advertised, Kueue installed."""
    if not shutil.which("kubectl"):
        return False, "kubectl not found: running the T0 path (predictions + commands)"
    k = Kubectl(context, echo=lambda s: None)
    try:
        nodes = k.get_json("nodes").get("items", [])
    except Exception as e:  # noqa: BLE001 - any failure means "not usable"
        return False, f"context {context!r} not reachable ({str(e)[:120]})"
    gpus = sum(int((n.get("status") or {}).get("allocatable", {}).get(m.GPU, 0) or 0) for n in nodes)
    if not gpus:
        return False, f"{len(nodes)} node(s) but no {m.GPU} capacity: run deploy/kind/up.sh (or fake-gpus.sh)"
    try:
        k.run("get", "crd", "clusterqueues.kueue.x-k8s.io", quiet=True)
    except Exception:  # noqa: BLE001
        return False, "Kueue CRDs missing: run deploy/kind/install-addons.sh"
    return True, f"{context}: {len(nodes)} nodes, {gpus} fake GPUs, Kueue installed"


# ---- observing ------------------------------------------------------------------------------------
def _conditions(obj: dict) -> dict[str, dict]:
    return {c["type"]: c for c in (obj.get("status") or {}).get("conditions") or []}


def workload_state(wl: dict) -> tuple[str, str]:
    """(state, message) of a Kueue Workload object, in kindsim's vocabulary."""
    conds = _conditions(wl)
    if conds.get("Admitted", {}).get("status") == "True":
        return "admitted", ""
    pre = conds.get("Preempted")
    if pre and pre.get("status") == "True":
        return "preempted", f"{pre.get('reason')}: {pre.get('message', '')}"
    qr = conds.get("QuotaReserved")
    if qr and qr.get("status") == "True":
        return "reserved", "waiting for admission checks"
    return "pending", (qr or {}).get("message", "")


def observe(k: Kubectl, sc: Scenario) -> dict[str, dict]:
    """The cluster's current outcome for a scenario's objects, shaped like ``kindsim`` output."""
    nodes = {n["metadata"]["name"]: n for n in k.get_json("nodes").get("items", [])}
    host_of = {name: (n["metadata"].get("labels") or {}).get(m.TOPOLOGY_HOST, name) for name, n in nodes.items()}
    pods = k.get_json("pods", "-A").get("items", [])
    wls = k.get_json("workloads.kueue.x-k8s.io", "-A").get("items", [])
    out: dict[str, dict] = {}
    for _, (_, o) in sc.objects.items():
        kind, ns, name = o["kind"], o["metadata"].get("namespace", "default"), o["metadata"]["name"]
        key = f"{kind}/{ns}/{name}"
        mine = [p for p in pods if p["metadata"].get("namespace") == ns
                and (p["metadata"].get("labels") or {}).get(POD_OWNER_LABELS[kind]) == name
                and not p["metadata"].get("deletionTimestamp")]
        queued = m.QUEUE_LABEL in (o["metadata"].get("labels") or {})
        groups = sorted({(p["metadata"].get("labels") or {}).get(LWS_GROUP_LABEL, "0") for p in mine}) \
            if kind == "LeaderWorkerSet" else [None]
        for g in groups or ["0"]:
            gkey = key if g is None else f"{key}#{g}"
            gpods = mine if g is None else [p for p in mine if (p["metadata"].get("labels") or {}).get(LWS_GROUP_LABEL) == g]
            d: dict = {}
            if queued:
                prefix = f"{kind.lower()}-{name}-" + (f"{g}-" if g is not None else "")
                wl = next((w for w in wls if w["metadata"].get("namespace") == ns and (
                    any(r.get("name") == name and r.get("kind") == kind for r in w["metadata"].get("ownerReferences") or [])
                    if g is None else w["metadata"]["name"].startswith(prefix))), None)
                if wl is None:
                    continue
                d["state"], msg = workload_state(wl)
                if msg:
                    d["message"] = msg
            else:
                unsched = [p for p in gpods if _conditions(p).get("PodScheduled", {}).get("reason") == "Unschedulable"]
                d["state"] = "unschedulable" if unsched else "running"
                if unsched:
                    d["message"] = _conditions(unsched[0])["PodScheduled"].get("message", "")
            if d["state"] in ("admitted", "running"):
                d["hosts"] = dict(sorted(Counter(host_of.get(p["spec"].get("nodeName"), "?")
                                                 for p in gpods if p["spec"].get("nodeName")).items()))
            out[gkey] = d
    return out


def compare(observed: dict[str, dict], predicted: dict[str, dict]) -> list[str]:
    """Differences between what the cluster did and what the predictor said."""
    problems = []
    for key, p in predicted.items():
        o = observed.get(key)
        if o is None:
            problems.append(f"{key}: not observed yet")
        elif o["state"] != p["state"]:
            problems.append(f"{key}: state {o['state']} (predicted {p['state']}) {o.get('message', '')[:160]}")
        elif p.get("hosts") not in (None, "any") and o.get("hosts") != p.get("hosts"):
            problems.append(f"{key}: hosts {o.get('hosts')} (predicted {p['hosts']})")
    return problems


# ---- running -----------------------------------------------------------------------------------------
def reset(k: Kubectl, timeout_s: int = 180) -> None:
    """Delete every lab workload (not the queues) and wait for the pods to go."""
    for ns in LAB_NAMESPACES:
        for kind in WORKLOAD_KINDS.split(","):   # one kind at a time: a missing CRD must not abort the rest
            k.run("delete", kind, "--all", "-n", ns, "--ignore-not-found", "--wait=true", check=False,
                  quiet=kind != "jobs.batch")
    if k.dry_run:
        return
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        left = [p for p in k.get_json("pods", "-A").get("items", []) if p["metadata"]["namespace"] in LAB_NAMESPACES]
        if not left:
            return
        time.sleep(3)


def settle(k: Kubectl, sc: Scenario, *, stable_s: float = 12.0, timeout_s: float = 240.0,
           expect_keys: list[str] | None = None, poll_s: float = 3.0) -> dict[str, dict]:
    """Poll until the observed outcome stops changing for ``stable_s`` (and expected objects exist)."""
    if k.dry_run:
        return {}
    start, last, since = time.time(), None, time.time()
    while True:
        cur = observe(k, sc)
        if cur != last:
            last, since = cur, time.time()
        present = all(key in cur for key in (expect_keys or []))
        if present and time.time() - since >= stable_s:
            return cur
        if time.time() - start > timeout_s:
            return cur
        time.sleep(poll_s)


def run_scenario(key: str, *, k: Kubectl | None = None, dry_run: bool = False, do_reset: bool = True,
                 printer: Callable[[str], None] = print) -> dict:
    """Apply a scenario on the cluster step by step; print predictions, observations and differences."""
    sc = get_scenario(key)
    k = k or Kubectl(dry_run=dry_run, echo=printer)
    live = not k.dry_run
    nodes = kindsim.nodes_from_k8s(k.get_json("nodes")["items"]) if live else None
    sim = kindsim.new_sim(sc, nodes)
    report = {"scenario": sc.id, "steps": []}
    printer(f"== {sc.id}: {sc.title}\n   {sc.story}")
    if do_reset:
        reset(k)
    for i, st in enumerate(sc.steps):
        printer(f"\n-- step {i + 1}/{len(sc.steps)}: {st.action} {st.file or st.target} {('- ' + st.note) if st.note else ''}")
        path = f"deploy/kind/workloads/{st.file}"
        if st.action == "apply":
            k.run("apply", "-f", path)
        elif st.action == "delete":
            k.run("delete", "-f", path, "--wait=true")
        elif st.action == "scale":
            kind, ns, name = st.target.split("/")
            k.run("scale", f"{kind.lower()}.leaderworkerset.x-k8s.io" if kind == "LeaderWorkerSet" else kind.lower(),
                  name, "-n", ns, f"--replicas={st.replicas}")
        kindsim.run_step(sim, sc, i)
        predicted = sim.outcome()
        observed = settle(k, sc, stable_s=12.0 if st.action == "apply" else 30.0,
                          expect_keys=[kk for kk in predicted if st.action != "apply" or kk in _keys_of(st)])
        if live:   # follow the scheduler's random choices so later predictions start from reality
            for kk, o in observed.items():
                w = sim.workloads.get(kk)
                if w and not w.queue and w.any_node and o.get("hosts"):
                    sim_pin(sim, w, o["hosts"])
            predicted = sim.outcome()
        printer("   predicted:\n" + _indent(kindsim.describe(predicted)))
        step = {"step": i + 1, "predicted": predicted}
        if live:
            printer("   observed:\n" + _indent(kindsim.describe(observed)))
            diffs = compare(observed, predicted)
            printer("   " + ("OK: matches the prediction" if not diffs else "DIFFERENT:\n" + _indent("\n".join(diffs))))
            step.update(observed=observed, differences=diffs)
            exp = EXPECTED.get(sc.id, {}).get(i + 1)
            if exp:
                step["answer_key"] = kindsim.check_expectation(observed, exp)
        report["steps"].append(step)
    return report


def _keys_of(st) -> list[str]:
    out = []
    for o in m.load_all(kindsim.KIND_DIR / "workloads" / st.file):
        out.append(f"{o['kind']}/{o['metadata'].get('namespace', 'default')}/{o['metadata']['name']}"
                   + ("#0" if o["kind"] == "LeaderWorkerSet" else ""))
    return out


def sim_pin(sim: kindsim.Sim, w: kindsim.Workload, hosts: dict[str, int]) -> None:
    """Move a plain (non-Kueue) workload's pods to the hosts the real scheduler picked."""
    by_host = {n.host: n.name for n in sim.nodes}
    per_pod = w.podsets[0].gpus
    sim._unbind(w)
    placement = {by_host[h]: pods for h, pods in hosts.items() if h in by_host}
    sim._bind(w, placement, {node: pods * per_pod for node, pods in placement.items()})
    w.any_node = False


def _indent(text: str, n: int = 6) -> str:
    return "\n".join(" " * n + line for line in text.splitlines())
