# k8s-gpu-core

**How Kubernetes turns GPUs into schedulable integers — and how to place, share, queue and scale
them.** A small, deterministic simulator of the pieces that decide where GPU work runs: the device
plugin and kubelet, kube-scheduler's filter/score/preempt cycle, gang and topology-aware placement
(Kueue TAS), Kueue quotas with cohort borrowing and reclaim, and a cluster autoscaler for GPU pools.
Pure standard library, about a thousand lines you can read in two sittings, plus five fill-in
notebooks.

**Tier: T0.** Everything runs on a laptop, Colab CPU or CI — no cluster, no GPU, no network. The
concepts are the whole point; the detailed lab next door, [`../k8s-gpu-lab`](../k8s-gpu-lab), takes
the same ideas to real manifests, a kind cluster with fake GPUs and Kueue, and GKE (T1–T3). The
concept primer both share is [`../PRIMER.md`](../PRIMER.md).

## Quick start

```bash
cd 03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core
python3 -m pip install -r requirements.txt   # only to run the notebooks/tests
python3 -m pytest -q                          # 33 tests, well under a second
python3 -m jupyterlab notebooks               # do the exercises
```

On Colab, open any notebook from the layer README's Colab links; its first cell clones the repo,
`cd`s here and `pip install -e .`s the package.

The library itself needs **nothing installed**:

```python
from gpusched import Scheduler, gpu_pod, make_cluster, stranded_gpus

cluster = make_cluster(hosts=4)                      # 4 nodes x 8 GPUs
sched = Scheduler(cluster)                           # kube-scheduler defaults: LeastAllocated on cpu+memory
sched.submit(*[gpu_pod(f"infer-{i}", 1) for i in range(16)])
sched.run()
print(cluster.show())                                # 4/8 on every node: spread
print(sched.schedule_one(gpu_pod("train", 8)).message)
# 0/4 nodes are available: 4 Insufficient nvidia.com/gpu. preemption: 0/4 nodes are available: ...
print(stranded_gpus(cluster, 8))                     # 16 free GPUs, none usable by an 8-GPU pod
```

## The library (seven files)

| File | What it teaches |
|------|-----------------|
| `gpusched/deviceplugin.py` | a GPU becomes an integer: `Register` → `ListAndWatch` → capacity vs allocatable → `Allocate`; unhealthy devices; time-slicing replicas; NVLink-aware preferred allocation |
| `gpusched/cluster.py` | nodes, pods, taints and tolerations, topology paths; the API server's rules for extended resources; the ExtendedResourceToleration admission plugin |
| `gpusched/plugins.py` | filters (first failure wins, upstream reason strings), `LeastAllocated` / `MostAllocated` scores in upstream integer arithmetic, `DefaultPreemption` victim selection and node choice |
| `gpusched/scheduler.py` | the one-pod-at-a-time cycle, the `FailedScheduling` message format, stranded GPUs and fragmentation |
| `gpusched/gang.py` | all-or-nothing placement; Kueue TAS `BestFit` / `LeastFreeCapacity`; required vs preferred topology levels |
| `gpusched/quota.py` | Kueue-style ClusterQueues, LocalQueues, cohorts, `borrowingLimit` / `lendingLimit`, classic preemption, StrictFIFO vs BestEffortFIFO |
| `gpusched/autoscaler.py` | bin-packing estimate, least-waste expander, scale-from-zero and scale-down, Spot reclaim and gang restarts, atomic queued provisioning, startup latency |

Read them in that order. Each module opens with the one idea it teaches. Where the simulator
simplifies the real system, the docstring says so (for example: preemption binds immediately
instead of nominating a node; Kueue is modelled with one resource group, flat cohorts and classic
preemption).

## The notebooks

Each has worked examples, then exercises with `# YOUR CODE HERE` and a check cell that prints ✅.
Solutions are in `solutions/`. Every notebook ends with **In a design review**: the two-minute
explanation and drill questions.

1. **`01_how_kubernetes_sees_a_gpu`** — device plugin → kubelet → node status; taints and tolerations; the integer rules; labels; a GPU failing under running pods; NVLink-aware allocation.
2. **`02_filter_score_and_fragmentation`** — the cycle; scores by hand; spread vs pack and stranded GPUs; reading `FailedScheduling`; preemption victims.
3. **`03_gangs_and_topology`** — the partial-placement deadlock; all-or-nothing; Kueue TAS BestFit; required vs preferred topology per workload.
4. **`04_queues_quotas_and_preemption`** — quota arithmetic with borrowing and lending; reclaim; preemption order; FIFO strategies; writing ClusterQueues for a serving/batch split.
5. **`05_autoscaling_and_obtainability`** — bin-packing estimate; scale-from-zero and the idle tail; Spot and gang restarts; queued provisioning; choosing a capacity type; startup latency.

## Regenerating notebooks

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
make check                                              # all of the above + tests
```

To redo an exercise, `git restore notebooks/<name>.ipynb` returns it to the committed blank.

## Where the numbers come from

Every number in the notebooks and in `../PRIMER.md` is computed by this package, with inputs
stated next to it. Durations, prices, stockout and preemption rates are **illustrative inputs**
(prices and obtainability: [`../../../COMPUTE.md`](../../../COMPUTE.md)); outputs are
**simulated**. Formulas and reason strings are pinned to upstream sources in `tests/`.

MIT licensed.
