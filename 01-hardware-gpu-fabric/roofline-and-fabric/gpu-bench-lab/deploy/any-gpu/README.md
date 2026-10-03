# Run gpubench on any GPU (T1/T2)

**What it does.** It puts the measurement suite of the lab on a real GPU. The GPU can be on Colab, on Kaggle, in a
rented container or VM, or in any Linux box with Docker. Thus, the GPU cells of the notebooks and `gpubench run`
measure real hardware.

**Cost.** It is free on Colab and Kaggle. For a rented GPU, you pay by the hour while the machine exists (see the
table that follows). The default (quick) run of the suite takes a few minutes on one GPU.

**Clean up.** After the suite writes the report (`ls results/`), stop or terminate the pod/VM. Also delete each
volume that you attached.

The lab needs Python ≥ 3.10 and numpy. For the GPU tiers, it also needs PyTorch with CUDA and an NVIDIA driver. The
table gives four ways to get to the GPU tiers, with the lowest cost first. The prices are approximate (September 2026, verify).
[`COMPUTE.md`](../../../../../COMPUTE.md) keeps the current comparison.

| Where | GPUs | Cost | Gets you |
|---|---|---|---|
| Google Colab | 1× T4 16 GB (free tier, not guaranteed) | free | T1: every GPU cell of notebooks 01, 02, 04 |
| Kaggle notebooks | 2× T4 (or 1× P100), ~30 GPU-hours/week | free | T2 over **PCIe** (no NVLink): the P2P matrix of notebook 03 |
| RunPod / Vast.ai (containers) | RTX 4090 ~$0.3–0.4/hr, A100/H100 ~$1–3/hr, multi-GPU SXM pods | per second/hour | NVLink P2P on SXM machines, FP8 on H100 |
| Lambda (VMs) | A100 ~$2/hr, H100 ~$3.3/hr | per hour | a full VM: drivers, Docker, `nvidia-smi topo -m` |

## 1 · Colab or Kaggle (no install)

Open a notebook from its Colab link in the layer README ([`01-hardware-gpu-fabric/README.md`](../../../../README.md)).
Then select *Runtime*, then *Change runtime type*, then *T4 GPU*. The first cell clones this repository, changes
into the lab and runs `pip install -e .`. PyTorch is already there. Thus `get_backend("auto")` selects the GPU.

On Kaggle, select *Settings*, then *Accelerator*, then *GPU T4 x2*. Set *Internet on*. Then run these commands in a
cell:

```bash
!git clone --depth 1 https://github.com/aniryou/full-stack-agentic-engineer.git
%cd full-stack-agentic-engineer/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab
!pip install -q -e .
!python -m gpubench run --suite inventory,gemm,transfer,p2p --out results
```

Expect the two T4s to connect to each other over PCIe through the CPU (`PHB`). The 15.8 GB/s (Gen3 x16) of the
model assumes direct peer access. If the VM does not permit direct peer access, the copies go through host memory as
staged copies. Their speed is about half the link or less. Notebook 03 explains how to find which case you have.

## 2 · Any Linux GPU box, plain pip

```bash
git clone --depth 1 https://github.com/aniryou/full-stack-agentic-engineer.git
cd full-stack-agentic-engineer/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[gpu,dev]'            # PyPI torch (2.11+) is a CUDA 13 build: needs an R580+ driver (verify)
# older driver (nvidia-smi shows < 580)? install a CUDA 12.6 build first (Maxwell..Hopper, no Blackwell):
#   pip install torch --index-url https://download.pytorch.org/whl/cu126 && pip install -e '.[dev]'
nvidia-smi && python -m gpubench info  # the driver and PyTorch both see the GPU?
python -m gpubench run --out results   # add --full for bigger sizes
python -m jupyterlab notebooks
```

This list shows which PyTorch build each driver can run (verify against the PyTorch install matrix):

- From 2.11 on, the default build on PyPI is for CUDA 13. It needs an **R580+** driver.
- The `cu126` index serves CUDA 12.6 builds. They run on R525+, but they have no Blackwell kernels.
- Blackwell GPUs (B200, RTX 50xx, RTX PRO 6000) need a CUDA 12.8+ build and an R570+ driver.

If `gpubench info` uses the numpy fallback on a machine where `nvidia-smi` works, the message names your driver and
the CUDA version of PyTorch. A mismatch between these two versions is the cause.

On RunPod or Vast, you are already in a container with a GPU. Select a PyTorch template. Then run the same commands.
If the template has PyTorch, do not use `[gpu]`. You cannot change the driver there. Thus, select a template with a
CUDA version that the host driver supports.

Also, on these platforms, `nvidia-smi topo -m` shows only the GPUs that the platform gives to your container.

## 3 · Docker (`run.sh`)

This path needs Docker and the NVIDIA Container Toolkit, so that `docker run --gpus all` works.

```bash
deploy/any-gpu/run.sh                 # build the image, check the GPU is visible, run the suite
deploy/any-gpu/run.sh --full          # extra arguments go to `gpubench run`
DRY_RUN=1 deploy/any-gpu/run.sh       # print the commands only
BASE=pytorch/pytorch:<tag> deploy/any-gpu/run.sh   # another CUDA/PyTorch base image
```

What it does:

1. It builds `gpubench:local` from [`Dockerfile`](Dockerfile) (a `pytorch/pytorch` runtime image plus this lab).
2. It runs `gpubench info` in a container as a smoke test.
3. It runs the suite with `--gpus all --ipc=host --ulimit memlock=-1` and writes the report to `./results` on the
   host.

The CUDA version of the base image must agree with the host driver *and* the GPU (verify). The default `cuda12.6` image runs
on R525+ drivers, but it has no Blackwell kernels. On a B200 or an RTX 50xx / RTX PRO 6000, use
`BASE=pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime` (R580+ driver, Turing and newer only).

## 4 · GCP

See [`../gcp/`](../gcp/). Its Terraform makes one Spot L4 VM. The VM runs the suite when it boots, uploads the
report to a bucket and then stops.

## Cost and clean-up

Colab and Kaggle cost nothing. On all other platforms, you pay while the machine exists, not while it works. When
the suite writes the report (`ls results/`), stop or terminate the pod/VM. Also delete each volume that you attached.
The default (quick) run of the suite takes a few minutes on one GPU.
