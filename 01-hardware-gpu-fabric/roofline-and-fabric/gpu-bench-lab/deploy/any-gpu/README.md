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

Open a notebook from the Colab badge in the layer README (`01-hardware-gpu-fabric/README.md`), then
*Runtime → Change runtime type → T4 GPU*. The first cell clones this repository, changes into the lab
and runs `pip install -e .`; PyTorch is already there, so `get_backend("auto")` picks the GPU.

On Kaggle: *Settings → Accelerator → GPU T4 x2* and *Internet on*, then in a cell:

```bash
!git clone --depth 1 https://github.com/aniryou/full-stack-agentic-engineer.git
%cd full-stack-agentic-engineer/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab
!pip install -q -e .
!python -m gpubench run --suite inventory,gemm,transfer,p2p --out results
```

Expect the two T4s to reach each other over PCIe through the CPU (`PHB`); if peer access is not
available in the VM, copies are staged through host memory — notebook 03 explains the number.

## 2 · Any Linux GPU box, plain pip

```bash
git clone --depth 1 https://github.com/aniryou/full-stack-agentic-engineer.git
cd full-stack-agentic-engineer/01-hardware-gpu-fabric/roofline-and-fabric/gpu-bench-lab
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[gpu,dev]'            # torch from PyPI brings its own CUDA runtime; the driver must be recent enough
nvidia-smi && python -m gpubench info  # the driver and PyTorch both see the GPU?
python -m gpubench run --out results   # add --full for bigger sizes
python -m jupyterlab notebooks
```

On RunPod or Vast you are already inside a container with a GPU: choose a PyTorch template, then run
the same commands (skip `[gpu]` if the template has PyTorch). You cannot change the driver there,
and `nvidia-smi topo -m` shows only the GPUs given to your container.

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
The base image's CUDA must suit the host driver (verify: CUDA 12.x images run on R525+ drivers;
CUDA 13.x images need R580+ and drop pre-Turing GPUs).

## 4 · GCP

See [`../gcp/`](../gcp/): Terraform for one Spot L4 VM that runs the suite on boot, uploads the
report to a bucket and powers off.

## Cost and clean-up

Colab and Kaggle cost nothing. Everything else bills while the machine exists, not while it works:
stop or terminate the pod/VM when the report is written (`ls results/`), and delete any volume you
attached. The suite's default (quick) run takes a few minutes on one GPU.
