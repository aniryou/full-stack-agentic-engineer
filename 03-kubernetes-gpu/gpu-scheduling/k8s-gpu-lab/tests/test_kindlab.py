"""The live-cluster code path, exercised against a fake kubectl.

No Docker here, so a fake ``Kubectl`` keeps a simulated cluster (``kindsim.Sim``) and answers
``kubectl get ... -o json`` with documents shaped like the real ones: Nodes with labels, taints
and allocatable GPUs; Pods with the owner labels Job/JobSet/LWS controllers set, ``nodeName``
and ``PodScheduled`` conditions; Kueue Workloads with owner references, names and conditions.
This checks that ``observe`` reads those shapes correctly and that ``run_scenario`` drives apply,
delete, scale and reset without errors.
"""
import json

import pytest

from k8sgpu import kindlab, kindsim, scenarios
from k8sgpu import manifests as m


class FakeCluster(kindlab.Kubectl):
    def __init__(self, sc):
        super().__init__(dry_run=False, echo=lambda s: None)
        self.sim = kindsim.new_sim(sc)

    # -- kubectl verbs the runner uses ------------------------------------------------------------
    def run(self, *args, stdin=None, check=True, quiet=False):
        self.commands.append("kubectl " + " ".join(args))
        a = list(args)
        if a[:2] == ["apply", "-f"]:
            for o in m.load_all(a[2]):
                self.sim.apply(o)
        elif a[:2] == ["delete", "-f"]:
            for o in m.load_all(a[2]):
                meta = o["metadata"]
                self.sim.delete(f"{o['kind']}/{meta['namespace']}/{meta['name']}")
        elif a[0] == "delete" and "--all" in a:
            ns = a[a.index("-n") + 1]
            for key in [k for k, w in self.sim.workloads.items() if w.namespace == ns]:
                self.sim.delete(key.split("#")[0])
        elif a[0] == "scale":
            ns = a[a.index("-n") + 1]
            replicas = int(next(x for x in a if x.startswith("--replicas=")).split("=")[1])
            self.sim.scale(f"LeaderWorkerSet/{ns}/{a[2]}", replicas)
        elif a[0] == "get":
            return json.dumps(self.get_json(*a[1:]))
        return ""

    def get_json(self, *args):
        what = args[0]
        if what == "nodes":
            return {"items": [self._node(n) for n in self.sim.nodes]}
        if what == "pods":
            return {"items": [p for key, w in self.sim.workloads.items() for p in self._pods(key, w)]}
        if what.startswith("workloads"):
            return {"items": [self._workload(key, w) for key, w in self.sim.workloads.items() if w.queue]}
        return {"items": []}

    # -- documents in the shapes the real API returns -----------------------------------------------
    @staticmethod
    def _node(n):
        return {"metadata": {"name": n.name, "labels": dict(n.labels)},
                "spec": {"taints": [dict(t) for t in n.taints]},
                "status": {"allocatable": {m.GPU: str(n.gpus)}}}

    @staticmethod
    def _split(key):
        base, _, group = key.partition("#")
        kind, ns, name = base.split("/")
        return kind, ns, name, group or None

    def _pods(self, key, w):
        kind, ns, name, group = self._split(key)
        labels = {kindlab.POD_OWNER_LABELS[kind]: name}
        if group is not None:
            labels[kindlab.LWS_GROUP_LABEL] = group
        pods, i = [], 0
        for node, count in w.placement.items():
            for _ in range(count):
                pods.append({"metadata": {"name": f"{name}-{i}", "namespace": ns, "labels": labels},
                             "spec": {"nodeName": node}, "status": {"phase": "Running"}})
                i += 1
        if w.state == "unschedulable":
            pods.append({"metadata": {"name": f"{name}-x", "namespace": ns, "labels": labels}, "spec": {},
                         "status": {"phase": "Pending", "conditions": [
                             {"type": "PodScheduled", "status": "False", "reason": "Unschedulable", "message": w.message}]}})
        if w.queue and w.state != "admitted" and kind == "LeaderWorkerSet":   # grouped pods exist, gated
            pods.append({"metadata": {"name": f"{name}-{group}", "namespace": ns, "labels": labels},
                         "spec": {"schedulingGates": [{"name": "kueue.x-k8s.io/admission"}]}, "status": {"phase": "Pending"}})
        return pods

    def _workload(self, key, w):
        kind, ns, name, group = self._split(key)
        wname = f"{kind.lower()}-{name}-" + (f"{group}-" if group is not None else "") + "1a2b3"
        conds = []
        if w.state == "admitted":
            conds = [{"type": "QuotaReserved", "status": "True"}, {"type": "Admitted", "status": "True"}]
        elif w.state == "preempted":
            conds = [{"type": "QuotaReserved", "status": "False", "message": w.message},
                     {"type": "Preempted", "status": "True", "reason": w.preempted, "message": "Preempted to accommodate"}]
        else:
            conds = [{"type": "QuotaReserved", "status": "False", "reason": "Pending", "message": w.message}]
        return {"metadata": {"name": wname, "namespace": ns,
                             "ownerReferences": [{"kind": kind, "name": name}]},
                "status": {"conditions": conds}}


@pytest.mark.parametrize("sid", ["s1", "s2", "s3", "s4", "s5", "s6", "k1"])
def test_run_scenario_against_a_fake_cluster(sid):
    sc = scenarios.scenario(sid)
    fake = FakeCluster(sc)
    report = kindlab.run_scenario(sid, k=fake, printer=lambda s: None, stable_s=0, poll_s=0)
    assert len(report["steps"]) == len(sc.steps)
    for step in report["steps"]:
        assert step["differences"] == [], (sid, step["step"], step["differences"])
        if "answer_key" in step:
            assert step["answer_key"] == [], (sid, step["step"])
    applied = [c for c in fake.commands if c.startswith("kubectl apply -f")]
    assert len(applied) == sum(s.action == "apply" for s in sc.steps)
    assert any(c.startswith("kubectl delete jobs.batch --all -n team-a") for c in fake.commands)


def test_observe_reads_workload_conditions_and_placements():
    sc = scenarios.scenario("s4")
    fake = FakeCluster(sc)
    for i in range(len(sc.steps)):
        kindsim.run_step(fake.sim, sc, i)
    obs = kindlab.observe(fake, sc)
    assert obs["Job/team-a/a-low-2"]["state"] == "preempted" and "InClusterQueue" in obs["Job/team-a/a-low-2"]["message"]
    assert obs["Job/team-a/a-high"] == {"state": "admitted", "hosts": {"host-a2-2": 1}}


def test_workload_state_vocabulary():
    assert kindlab.workload_state({"status": {"conditions": [{"type": "Admitted", "status": "True"}]}}) == ("admitted", "")
    st, msg = kindlab.workload_state({"status": {"conditions": [
        {"type": "QuotaReserved", "status": "False", "message": "couldn't assign flavors"}]}})
    assert st == "pending" and msg.startswith("couldn't")
    assert kindlab.workload_state({"status": {"conditions": [{"type": "QuotaReserved", "status": "True"}]}})[0] == "reserved"
