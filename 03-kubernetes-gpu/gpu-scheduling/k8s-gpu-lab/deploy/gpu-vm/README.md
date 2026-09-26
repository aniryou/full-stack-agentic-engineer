# deploy/gpu-vm — the real device plugin on one GPU VM you control (T1/T2)

**What it does.** Turns one rented GPU VM into a one-node Kubernetes cluster with the pieces kind
can only fake: [k3s](https://k3s.io) (Kubernetes 1.34 channel), the **NVIDIA device plugin**
(it answers the kubelet's `ListAndWatch`/`Allocate`, primer §1.2), **GPU Feature Discovery**
(the `nvidia.com/gpu.*` node labels, primer §1.3) and, optionally, **time-slicing** (one GPU
advertised as several `nvidia.com/gpu`, primer §9). The scheduling lessons are the same as on
kind; what you gain is the device path: a pod that asks for `nvidia.com/gpu: 1` gets
`/dev/nvidia*`, the driver libraries and a working `nvidia-smi`. The NVIDIA GPU Operator
(primer §2) would also install and manage the driver and container toolkit; a GPU VM image
already has both, so the device plugin chart (with GFD) is the smallest real stack — on a
kubeadm cluster, or to manage drivers in-cluster, use the GPU Operator instead (verify its
k3s/containerd settings).

**Cost.** Whatever the VM costs while it runs: a 1-GPU VM on Lambda, a GCP `g2-standard-4` L4
VM (Spot is cheapest), or any box with an NVIDIA GPU. Prices and obtainability:
`COMPUTE.md` at the repo root. One hour covers everything below. RunPod and Vast.ai give
you a *container*, not a VM: no systemd and no kubelet of your own, so k3s does not fit there.

**Needs, on the VM.** Ubuntu (or another systemd Linux) with the **NVIDIA driver** and the
**NVIDIA Container Toolkit** installed (`nvidia-smi` and `nvidia-container-runtime` on `PATH`;
Lambda Stack and GCP Deep Learning VM images come with both - verify), `sudo`, `curl`,
[`helm`](https://helm.sh/docs/intro/install/), and this repository checked out.

## Run it

```bash
# on the GPU VM, from the lab root (03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab)
DRY_RUN=1 deploy/gpu-vm/up.sh                  # read the whole procedure first; nothing runs
deploy/gpu-vm/up.sh                            # k3s + device plugin + GFD; runs the smoke Job
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
kubectl apply -f deploy/gpu-vm/20-time-sliced.yaml && kubectl get pods -o wide
#   on a 1-GPU VM without sharing: one pod Running, three Pending with "Insufficient nvidia.com/gpu"
kubectl delete -f deploy/gpu-vm/20-time-sliced.yaml
TIME_SLICING_REPLICAS=4 deploy/gpu-vm/up.sh    # re-run: the plugin now advertises 4 per GPU
kubectl apply -f deploy/gpu-vm/20-time-sliced.yaml && kubectl get pods -o wide
kubectl logs -l job-name=time-sliced            # 1-GPU VM: four pods, one GPU UUID: shared, not isolated
deploy/gpu-vm/down.sh                          # uninstall k3s - then STOP THE VM
```

| File | What |
|---|---|
| `up.sh` | checks the driver and runtime; installs k3s (`INSTALL_K3S_CHANNEL` from `../versions.env`), which finds the NVIDIA runtime and registers the `nvidia` RuntimeClass; installs the device plugin Helm chart (pinned `NVDP_CHART_VERSION`) with `runtimeClassName=nvidia` and `gfd.enabled=true`, plus a time-slicing config when `TIME_SLICING_REPLICAS > 1`; prints allocatable GPUs and GFD labels; runs `10-smoke.yaml` |
| `10-smoke.yaml`, `20-time-sliced.yaml` | generated from `k8sgpu/gpuvm.py`: `nvidia-smi -L` Jobs with `runtimeClassName: nvidia`, pinned to a node GFD labelled (`nvidia.com/gpu.product` exists) |
| `down.sh` | `k3s-uninstall.sh` |

## What to look at

* **Capacity comes from the plugin, not a patch.** `kubectl get node -o yaml`: `nvidia.com/gpu`
  in `capacity`/`allocatable` equals the GPUs `nvidia-smi` lists — or `GPUs x replicas` with
  time-slicing (`k8sgpu.gpuvm.time_sliced_allocatable`; the plugin's README: 8 GPUs x 10 = 80).
* **Labels come from GFD.** `nvidia.com/gpu.product`, `.memory` (MiB), `.count`, `.replicas`,
  `nvidia.com/cuda.driver-version.*`: the vocabulary a mixed fleet selects on, where GKE uses
  `cloud.google.com/gke-accelerator`.
* **Time-slicing is sharing without isolation.** On a 1-GPU VM all four pods print the same GPU
  UUID; nothing limits one pod's memory. With several GPUs the plugin hands out replicas from the
  least-loaded GPUs first, so pods spread over the GPUs before they double up. A container that
  requests `nvidia.com/gpu: 2` therefore gets two different GPUs on an idle multi-GPU node, but can
  get two slices of the *same* GPU once the node is busy (and always on a 1-GPU VM); the plugin's
  `failRequestsGreaterThanOne` option rejects such requests instead (primer §9).
* **The pod spec did not change.** Same integer limit and toleration as on kind and GKE; the one
  k3s-specific line is `runtimeClassName: nvidia`.

## Clean up

`deploy/gpu-vm/down.sh`, then **stop or delete the VM** — the GPU bills until you do.

## Verify list

That your VM image ships the driver and the NVIDIA Container Toolkit; the k3s NVIDIA-runtime
detection and `nvidia` RuntimeClass (k3s docs, *Advanced options: NVIDIA Container Runtime*);
device plugin chart values (`runtimeClassName`, `gfd.enabled`, `config.map`) for the pinned
chart; the `NVIDIA_DISABLE_REQUIRE` escape hatch the smoke Job uses for older drivers.
