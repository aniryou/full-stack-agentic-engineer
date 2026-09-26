"""The answer key against the upstream sources, not against the predictor.

``test_kindsim`` checks the predictor against ``scenarios.EXPECTED`` and ``test_kindlab`` runs the
live-cluster code against a fake cluster built from that same predictor, so neither is independent
evidence for the strings the answer key expects from a real cluster. This file is: every format
string below is copied from the pinned upstream source (file and tag in the comment), filled in by
hand with the kind lab's numbers, and every ``says`` in the answer key must be one of them.
"""
from k8sgpu import scenarios

# kubernetes v1.34.1 pkg/scheduler/framework/types.go: NoNodeAvailableMsg = "0/%v nodes are available";
# FitError.Error(): reasons are "%v %v" (count, reason), sort.Strings'ed, joined with ", ", then "."; then
# " " + the PostFilter message, which DefaultPreemption builds as "preemption: " + another such message.
NO_NODE = "0/{} nodes are available: {}."
# plugins/tainttoleration/taint_toleration.go: "node(s) had untolerated taint {%s: %s}" (key, value)
TAINT = "node(s) had untolerated taint {{{}: {}}}"
# plugins/nodeaffinity/node_affinity.go: ErrReasonPod
AFFINITY = "node(s) didn't match Pod's node affinity/selector"
# plugins/noderesources/fit.go: fmt.Sprintf("Insufficient %v", rName)
INSUFFICIENT = "Insufficient {}"
# framework/preemption/preemption.go and plugins/defaultpreemption/default_preemption.go
NOT_HELPFUL = "Preemption is not helpful for scheduling"
NO_VICTIMS = "No preemption victims found for incoming pod"

# kueue v0.19.6 pkg/scheduler/flavorassigner/flavorassigner.go (fitsResourceQuota / fitsMaxCapacity)
UNUSED = "insufficient unused quota for {} in flavor {}, {} more needed"
MAX_CAP = ("insufficient quota for {} in flavor {}, previously considered podsets requests ({}) + current podset "
           "request ({}) > maximum capacity ({})")
# kueue v0.19.6 pkg/cache/scheduler/tas_flavor_snapshot.go: "topology %q allows to fit only %d out of %d %s(s)"
TAS_ONLY = 'topology "{}" allows to fit only {} out of {} pod(s)'
# kueue v0.19.6 pkg/scheduler/preemption: Preempted condition reasons
PREEMPT_REASONS = {"InClusterQueue", "InCohortReclamation"}


def fit_error(n: int, reasons: dict[str, int], verdicts: dict[str, int]) -> str:
    hist = ", ".join(sorted(f"{c} {r}" for r, c in reasons.items()))
    pre = ", ".join(sorted(f"{c} {r}" for r, c in verdicts.items()))
    return NO_NODE.format(n, hist) + " preemption: " + NO_NODE.format(n, pre)


CP = TAINT.format("node-role.kubernetes.io/control-plane", "")
GPU_T = TAINT.format("nvidia.com/gpu", "present")
GPU = INSUFFICIENT.format("nvidia.com/gpu")

# The kind cluster: 1 control plane, 1 untainted system worker, 4 fake 4-GPU nodes with 1 GPU free each (s6).
UPSTREAM_SAYS = {
    ("s6", "Job/zoo/no-toleration"): fit_error(6, {AFFINITY: 1, CP: 1, GPU_T: 4}, {NOT_HELPFUL: 6}),
    ("s6", "Job/zoo/wrong-accelerator"): fit_error(6, {CP: 1, AFFINITY: 5}, {NOT_HELPFUL: 6}),
    # 5 GPUs > 4 allocatable: Unresolvable on every GPU node, so preemption cannot help anywhere
    ("s6", "Job/zoo/too-big"): fit_error(6, {AFFINITY: 1, CP: 1, GPU: 4}, {NOT_HELPFUL: 6}),
    # 2 GPUs <= 4 allocatable: resolvable by preemption, but every pod has priority 0 -> no victims
    ("s6", "Job/zoo/fragmented"): fit_error(6, {AFFINITY: 1, CP: 1, GPU: 4}, {NOT_HELPFUL: 2, NO_VICTIMS: 4}),
    ("s6", "Job/team-a/quota-too-big"): MAX_CAP.format("nvidia.com/gpu", "gpu-l4", 0, 16, 12),
    ("s6", "Job/team-a/topology-too-tight"): TAS_ONLY.format("gke-default", 1, 2),
    ("s2", "JobSet/team-a/gang-waits"): TAS_ONLY.format("gke-default", 2, 4),
    ("s3", "LeaderWorkerSet/team-b/llm-multihost#1"): UNUSED.format("nvidia.com/gpu", "gpu-l4", 4),
    ("s5", "Job/team-a/a-job-4"): UNUSED.format("nvidia.com/gpu", "gpu-l4", 4),
}


def test_every_expected_message_is_an_upstream_format():
    seen = set()
    for sid, steps in scenarios.EXPECTED.items():
        for exp in steps.values():
            for key, e in exp.items():
                says = e.get("says")
                if not says:
                    continue
                if says in PREEMPT_REASONS:
                    continue
                assert UPSTREAM_SAYS.get((sid, key)) == says, (sid, key, says)
                seen.add((sid, key))
    assert seen == set(UPSTREAM_SAYS)


def test_histogram_is_sorted_as_strings():
    # sort.Strings on "count reason": "1 node(s) didn't..." < "1 node(s) had..." < "4 node(s) had..."
    msg = UPSTREAM_SAYS[("s6", "Job/zoo/no-toleration")]
    assert msg.index("1 node(s) didn't") < msg.index("1 node(s) had") < msg.index("4 node(s) had")
