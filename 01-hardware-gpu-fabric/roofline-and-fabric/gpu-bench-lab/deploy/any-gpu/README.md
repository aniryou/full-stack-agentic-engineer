# Run gpubench on any GPU (T1/T2)

The lab needs Python ≥ 3.10, numpy, and — for the GPU tiers — PyTorch with CUDA and an NVIDIA
driver. Four ways to get there, cheapest first. Prices are approximate (September 2026, verify);
[`COMPUTE.md`](../../../../../COMPUTE.md) keeps the current comparison.

| Where | GPUs | Cost | Gets you |
|---|---|---|---|
| Google Colab | 1× T4 16 GB (free tier, not guaranteed) | free | T1: every GPU cell of notebooks 01, 02, 04 |
| Kaggle notebooks | 2× T4 (or 1× P100), ~30 GPU-hours/week | free | T2 over **PCIe** (no NVLink): notebook 03's P2P matrix |
| RunPod / Vast.ai (containers) | RTX 4090 ~$0.3–0.4/hr; A100/H100 ~$1–3/hr; multi-GPU SXM pods | per second/hour | NVLink P2P on SXM machines; FP8 on H100 |
| Lambda (VMs) | A100 ~$2/hr, H100 ~$3.3/hr | per hour | a full VM: drivers, Docker, `nvidia-smi topo -m` |

## 1 · Colab or Kaggle (no install)

Open a notebook from the Colab badge in the layer README ([`01-hardware-gpu-fabric/README.md`](../../../../README.md)), then
*Runtime → Change runtime type → T4 GPU*. The first cell clones this repository, changes into the lab
and runs `pip install -e .`; PyTorch is already there, so `get_backend("auto")` picks the GPU.

On Kaggle: *Settings → Accelerator → GPU T4 x2* and *Internet on*, then in a cell:

```bash
!git clone --depth 1 https://github.com/aniryou/full-stack-agentic-engineer.git
%cd full-stack-agentic-engineer/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab
!pip install -q -e .
!python -m gpubench run --suite inventory,gemm,transfer,p2p --out results
```

Expect the two T4s to reach each other over PCIe through the CPU (`PHB`). The model's 15.8 GB/s
(Gen3 x16) assumes direct peer access; if the VM does not allow it, copies are staged through host
memory at about half the link or less — notebook 03 explains how to tell which you got.

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

Which PyTorch build a driver can run (verify against the PyTorch install matrix): PyPI's default from
2.11 on is built for CUDA 13 and needs an **R580+** driver; the `cu126` index serves CUDA 12.6 builds
that run on R525+ but carry no Blackwell kernels; Blackwell GPUs (B200, RTX 50xx, RTX PRO 6000) need a
CUDA 12.8+ build and an R570+ driver. If `gpubench info` falls back to numpy on a machine where
`nvidia-smi` works, the message names your driver and PyTorch's CUDA version — that mismatch is the cause.

On RunPod or Vast you are already inside a container with a GPU: choose a PyTorch template, then run
the same commands (skip `[gpu]` if the template has PyTorch). You cannot change the driver there — pick
a template whose CUDA version the host driver supports — and `nvidia-smi topo -m` shows only the GPUs
given to your container.

## 3 · Docker (`run.sh`)

Needs Docker and the NVIDIA Container Toolkit (so `docker run --gpus all` works).

```bash
deploy/any-gpu/run.sh                 # build the image, check the GPU is visible, run the suite
deploy/any-gpu/run.sh --full          # extra arguments go to `gpubench run`
DRY_RUN=1 deploy/any-gpu/run.sh       # print the commands only
BASE=pytorch/pytorch:<tag> deploy/any-gpu/run.sh   # another CUDA/PyTorch base image
```

What it does: builds `gpubench:local` from [`Dockerfile`](Dockerfile) (a `pytorch/pytorch` runtime
image plus this lab), runs `gpubench info` in a container as a smoke test, then runs the suite with
`--gpus all --ipc=host --ulimit memlock=-1` and writes the report to `./results` on the host.
The base image's CUDA must suit the host driver *and* the GPU (verify): the default `cuda12.6` image
runs on R525+ drivers but has no Blackwell kernels; on a B200 or an RTX 50xx / RTX PRO 6000 use
`BASE=pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime` (R580+ driver, Turing and newer only).

## 4 · GCP

See [`../gcp/`](../gcp/): Terraform for one Spot L4 VM that runs the suite on boot, uploads the
report to a bucket and powers off.

## Cost and clean-up

Colab and Kaggle cost nothing. Everything else bills while the machine exists, not while it works:
stop or terminate the pod/VM when the report is written (`ls results/`), and delete any volume you
attached. The suite's default (quick) run takes a few minutes on one GPU.
