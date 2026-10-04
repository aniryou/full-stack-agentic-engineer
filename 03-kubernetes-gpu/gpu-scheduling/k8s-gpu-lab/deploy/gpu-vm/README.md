# deploy/gpu-vm — the real device plugin on one GPU VM you control (T1/T2)

**What it does.** These scripts change one rented GPU VM into a one-node Kubernetes cluster. The cluster has
the parts that kind can only fake:

- [k3s](https://k3s.io) (Kubernetes 1.34 channel).
- The **NVIDIA device plugin**. It answers the `ListAndWatch`/`Allocate` calls of the kubelet (primer §1.2).
- **GPU Feature Discovery**, which supplies the `nvidia.com/gpu.*` node labels (primer §1.3).
- Optionally, **time-slicing**. The device plugin advertises one GPU as several `nvidia.com/gpu` (primer §9).

The scheduling lessons are the same as on kind. The gain is the device path. A pod that asks for
`nvidia.com/gpu: 1` gets `/dev/nvidia*`, the driver libraries and a `nvidia-smi` that works.

The NVIDIA GPU Operator (primer §2) can also install and manage the driver and the container toolkit. A GPU VM
image already has both. Thus the device plugin chart (with GFD) is the smallest real stack. On a kubeadm
cluster, or to manage drivers in the cluster, use the GPU Operator instead (verify its k3s/containerd settings).

**Cost.** The cost is the price of the VM while it runs. The VM can be one of these:

- A 1-GPU VM on Lambda.
- A GCP `g2-standard-4` L4 VM (Spot has the lowest cost).
- Any box with an NVIDIA GPU.

For prices and availability, see [`COMPUTE.md`](../../../../../COMPUTE.md). One hour is sufficient for all the
steps in this file. RunPod and Vast.ai give you a *container*, not a VM. In a container, you have no systemd and no
kubelet of your own. Thus, k3s does not fit there.

**Clean up.** `deploy/gpu-vm/down.sh` uninstalls k3s. Then **stop or delete the VM**. You pay for the GPU until
you do.

**Needs, on the VM.**

- Ubuntu (or another systemd Linux) with the **NVIDIA driver** and the **NVIDIA Container Toolkit** installed.
  `nvidia-smi` and `nvidia-container-runtime` must be on `PATH`. Lambda Stack and GCP Deep Learning VM images
  have both (verify).
- `sudo`.
- `curl`.
- [`helm`](https://helm.sh/docs/intro/install/).
- This repository, checked out.

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
| `up.sh` | It examines the driver and the runtime. It installs k3s (`INSTALL_K3S_CHANNEL` from `../versions.env`). k3s finds the NVIDIA runtime and registers the `nvidia` RuntimeClass. It installs the device plugin Helm chart (pinned `NVDP_CHART_VERSION`) with `runtimeClassName=nvidia` and `gfd.enabled=true`. When `TIME_SLICING_REPLICAS > 1`, it also installs a time-slicing config. It prints the allocatable GPUs and the GFD labels. It runs `10-smoke.yaml` |
| `10-smoke.yaml`, `20-time-sliced.yaml` | `k8sgpu/gpuvm.py` generates them. They are `nvidia-smi -L` Jobs with `runtimeClassName: nvidia`. Each Job goes only to a node that has a GFD label (`nvidia.com/gpu.product` exists) |
| `down.sh` | `k3s-uninstall.sh` |

## What to look at

* **Capacity comes from the plugin, not a patch.** Run `kubectl get node -o yaml`. The `nvidia.com/gpu` value
  in `capacity`/`allocatable` is equal to the number of GPUs that `nvidia-smi` lists. With time-slicing, it is
  `GPUs x replicas` (`k8sgpu.gpuvm.time_sliced_allocatable`). The README of the plugin gives this example:
  8 GPUs x 10 = 80.
* **Labels come from GFD.** GFD writes `nvidia.com/gpu.product`, `.memory` (MiB), `.count`, `.replicas` and
  `nvidia.com/cuda.driver-version.*`. These labels are the vocabulary that a mixed fleet uses to select nodes.
  GKE uses `cloud.google.com/gke-accelerator` for the same purpose.
* **Time-slicing is sharing without isolation.** On a 1-GPU VM, all four pods print the same GPU UUID, and
  nothing limits the memory of one pod. With several GPUs, the plugin gives replicas from the least-loaded GPUs first,
  so pods spread over the GPUs before two pods share a GPU. Thus, on an idle multi-GPU node, a container that
  requests `nvidia.com/gpu: 2` gets two different GPUs. But when the node is busy, the container can get two
  slices of the *same* GPU, and on a 1-GPU VM it always gets them. The `failRequestsGreaterThanOne` option of
  the plugin rejects such requests instead (primer §9).
* **The pod spec did not change.** The pod has the same integer limit and toleration as on kind and GKE. The
  only line specific to k3s is `runtimeClassName: nvidia`.

## Clean up

Run `deploy/gpu-vm/down.sh`. Then **stop or delete the VM**. You pay for the GPU until you do.

## Verify list

Make sure of these items:

- That your VM image has the driver and the NVIDIA Container Toolkit.
- That k3s finds the NVIDIA runtime and registers the `nvidia` RuntimeClass (k3s docs, *Advanced options: NVIDIA
  Container Runtime*).
- The values of the device plugin chart (`runtimeClassName`, `gfd.enabled`, `config.map`) for the pinned chart.
- The `NVIDIA_DISABLE_REQUIRE` escape hatch that the smoke Job uses for older drivers.
