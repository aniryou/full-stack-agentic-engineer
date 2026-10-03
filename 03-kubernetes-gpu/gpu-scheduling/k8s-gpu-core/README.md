# k8s-gpu-core — how Kubernetes turns GPUs into schedulable integers

After this core, you can predict where GPU work lands, and why. You can explain these things:

- How the device plugin turns GPUs into integers.
- How kube-scheduler filters, scores and preempts.
- How gangs and topology-aware placement (Kueue TAS) prevent deadlock.
- How Kueue quotas borrow and reclaim across a cohort.
- How a cluster autoscaler brings GPU pools up from zero.

The core is a small, deterministic simulator, with five fill-in notebooks. It uses only the standard library. It has
about a thousand lines, and you can read them in two sessions.

**Tier T0** (laptop, Colab CPU or CI, with no cluster, no GPU and no network, at no cost). The concepts are the whole
point. The detailed lab next to this core, [`../k8s-gpu-lab`](../k8s-gpu-lab), takes the same ideas to three places:

- Real manifests.
- A kind cluster with fake GPUs and Kueue (T0 + Docker, a laptop with Docker, also at no cost).
- GKE (T3, the optional Google Cloud deployment).

The core and the lab share one concept primer, [`../PRIMER.md`](../PRIMER.md).

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1 (what Kubernetes sees).
2. Run the tests (see "Run it"). There are 59 tests, and they take ~25 s.
3. Open [`notebooks/01_how_kubernetes_sees_a_gpu.ipynb`](notebooks/01_how_kubernetes_sees_a_gpu.ipynb). The check
   cell of each exercise prints ✅ when your answer is correct.

## Run it

```bash
cd 03-kubernetes-gpu/gpu-scheduling/k8s-gpu-core
python3 -m pip install -r requirements.txt   # only to run the notebooks/tests
python3 -m pytest -q                          # 59 tests, ~25 s
python3 -m jupyterlab notebooks               # do the exercises
```

On Colab, open any notebook from the Colab links in the layer README. Its first cell clones the repo and does a
`cd` to this folder. Then it runs `pip install -e .` for the package.

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

## What you get: the library (seven files)

| File | What it teaches |
|------|-----------------|
| `gpusched/deviceplugin.py` | how a GPU becomes an integer: `Register`, then `ListAndWatch`, then the difference between capacity and allocatable, then `Allocate`. It also teaches unhealthy devices, time-slicing replicas and NVLink-aware preferred allocation. |
| `gpusched/cluster.py` | nodes, pods, taints and tolerations, topology paths, the API server's rules for extended resources, and the ExtendedResourceToleration admission plugin |
| `gpusched/plugins.py` | filters (the first failure wins, with upstream reason strings), `LeastAllocated` / `MostAllocated` scores in upstream integer arithmetic, and `DefaultPreemption` victim selection and node choice |
| `gpusched/scheduler.py` | the one-pod-at-a-time cycle, the `FailedScheduling` message format, stranded GPUs and fragmentation |
| `gpusched/gang.py` | all-or-nothing placement, Kueue TAS `BestFit` / `LeastFreeCapacity`, and the difference between required and preferred topology levels |
| `gpusched/quota.py` | Kueue-style ClusterQueues, LocalQueues, cohorts, `borrowingLimit` / `lendingLimit`, classic preemption, and StrictFIFO against BestEffortFIFO |
| `gpusched/autoscaler.py` | the bin-packing estimate, the least-waste expander, scale-from-zero and scale-down, Spot reclaim and gang restarts, atomic queued provisioning, and startup latency |

Read them in that order. Each module starts with the one idea that it teaches. Where the simulator
simplifies the real system, the docstring says so. The notebook that uses the module says it again.
These are the simplifications:

- Preemption binds immediately. It does not nominate a node.
- The simulator models Kueue with one resource group, flat cohorts, classic preemption and no admission
  checks. It admits a preemptor in the same step in which it evicts the victims of that preemptor.
- Gangs have one pod shape (no leader, no slices).
- The least-waste expander ranks idle GPUs, not CPU and memory.
- `simulate()` is one pool with no utilisation threshold.

## The notebooks

Each notebook has worked examples, then exercises with `# YOUR CODE HERE` and a check cell that prints ✅.
The solutions are in `solutions/`. Each notebook ends with **In a design review**: the two-minute
explanation and drill questions.

1. **`01_how_kubernetes_sees_a_gpu`**: the path from the device plugin to the kubelet to the node status. Then taints and tolerations, the integer rules and labels. Then a GPU that fails while pods run on it, NVLink-aware allocation, and time-slicing replicas and `failRequestsGreaterThanOne`.
2. **`02_filter_score_and_fragmentation`**: the cycle, scores by hand, spread against pack and stranded GPUs, how to read `FailedScheduling`, and preemption victims.
3. **`03_gangs_and_topology`**: the partial-placement deadlock, all-or-nothing, Kueue TAS BestFit, and required against preferred topology for each workload.
4. **`04_queues_quotas_and_preemption`**: quota arithmetic with borrowing and lending, reclaim, preemption order and FIFO strategies. Then how to write ClusterQueues for a serving/batch split, and how to predict a reclaim (borrowers only, minimised in reverse).
5. **`05_autoscaling_and_obtainability`**: the bin-packing estimate, scale-from-zero and the idle tail, Spot and gang restarts, and queued provisioning. Then how to select a capacity type, and startup latency.

## Regenerating notebooks

`tools/build_notebooks.py` makes `notebooks/` and `solutions/` from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks). Edit the sources. Then run these commands:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
make check                                              # all of the above + tests
```

To do an exercise again, run `git restore notebooks/<name>.ipynb`. It returns the notebook to the committed blank.

## Caveats: where the numbers come from

This package calculates every worked number in the notebooks and in `../PRIMER.md`. The notebooks and the primer state
the inputs next to each number. The package does not calculate the product facts (defaults, versions, discounts, MIG
profiles). The primer cites them in its Sources and Verify list. The durations, prices, stockout rates and
preemption rates are **illustrative inputs** (for prices and obtainability, see
[`COMPUTE.md`](../../../COMPUTE.md)). The outputs are **simulated**.

The tests in `tests/` pin the formulas and reason strings to upstream sources. `tests/test_docs.py` also
validates the primer's DRA manifest with `kubernetes-validate`. `requirements.txt` and the `dev` extra install
`kubernetes-validate`. Without `kubernetes-validate`, the test skips.

MIT licensed.
