# deploy/any-gpu — run the lab on whatever GPU you can get

**What it does:** takes the T1/T2 paths of the lab (real kernel timings, NCCL collectives,
nccl-tests, the container probe) to any NVIDIA GPU: a free Colab or Kaggle notebook, a rented
container (RunPod, Vast.ai) or VM (Lambda, GCP), or your own machine. Nothing here is GCP-specific.

**Cost:** free on Colab (one T4) and Kaggle (two T4s, 30 GPU-hours/week); roughly $0.3–0.7/hr for a
rented 24 GB GPU and $2–25 for an hour on a multi-GPU NVLink box. Prices move — see the repo's
compute guide and check the provider (verify).

**Cleanup:** stop or terminate the notebook/pod/VM when you are done (billing runs while it exists),
then delete `out/` and the `nccl-tests/` build directory if you ran the scripts locally.

| File | What it is |
|---|---|
| `probe.sh` | prints what a container sees of its GPU (device nodes, injected driver files, versions) in sections that `python -m gpurt.container --log` explains |
| `run_nccl_tests.sh` | builds NVIDIA/nccl-tests against the NCCL you already have (system or PyTorch's pip wheel) and sweeps `all_reduce`/`all_gather` on every local GPU; `DRY_RUN=1` prints the steps |
| `Dockerfile.nccl-tests` | the same in a CUDA *devel* image (nvcc + NCCL inside) |
| `Dockerfile.lab` | the lab on `python:3.12-slim` — no CUDA in the base image: the CUDA userland comes from pip wheels, `libcuda` from the host |

## 1. Any GPU box, plain pip (VM, rented container, your workstation)

```bash
git clone --depth 1 https://github.com/aniryou/full-stack-agentic-engineer.git
cd full-stack-agentic-engineer/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab
pip install -e ".[dev,gpu]"                 # numba-cuda[cu12]: NVIDIA's CUDA target for Numba
python -m gpurt.env                         # tier, GPUs, numba mode
python -m gpurt.container                   # how this process sees the GPU, and the compat verdicts
python -m gpurt.kernels.bench --quick --json out/kernels.json   # notebook 02's measurements
```

With two or more GPUs and a CUDA build of PyTorch (T2):

```bash
torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl --op all_reduce -e 256M --json out/ar.json | tee out/ar.log
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl --op all_gather -e 64M  # which transport?
bash deploy/any-gpu/run_nccl_tests.sh       # the reference tool, same busbw definition
python -m gpurt.nccltests out/all_reduce_2gpu.log
```

`ar.log` (our sweep) and `all_reduce_2gpu.log` (nccl-tests) print the same table layout, so notebook 04
reads either. They should agree on the plateau busbw; if ours is lower at small sizes, that is the
Python-side launch overhead of `torch.distributed` — the α term.

## 2. Colab (one T4, free)

*Runtime → Change runtime type → T4 GPU*, then open any notebook through the Colab links in the layer
README; the first cell clones the repo and installs the lab. If `from numba import cuda;
cuda.is_available()` is `False` on a GPU runtime, run `!pip install -q "numba-cuda[cu12]"` and restart
the session (whether Colab preinstalls numba-cuda changes over time — verify). Colab gives one GPU: T1 only.

## 3. Kaggle, two T4s (free T2 — collectives over PCIe)

1. New notebook → *Settings → Accelerator → GPU T4 x2*; turn *Internet* on (needs a verified phone
   number on the account — verify current rules).
2. First cell:
   ```python
   !git clone --depth 1 https://github.com/aniryou/full-stack-agentic-engineer.git
   %cd full-stack-agentic-engineer/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab
   !pip install -q -e ".[gpu]"
   !nvidia-smi topo -m          # two T4s on PCIe: no NVLink row, P2P may or may not be enabled
   ```
3. NCCL through PyTorch (preinstalled on Kaggle):
   ```python
   !torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl --op all_reduce -e 256M | tee /kaggle/working/ar.log
   !NCCL_DEBUG=INFO torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl --op all_reduce -b 1M -e 1M -n 5 2>&1 | grep -E "via|Channel" | head
   ```
4. nccl-tests against the same NCCL (the script finds PyTorch's `nvidia-nccl-cu12` wheel if there is no
   system NCCL; needs `nvcc`, present in Kaggle's GPU image — verify):
   ```python
   !OUT=/kaggle/working/out bash deploy/any-gpu/run_nccl_tests.sh
   ```
5. Kernels on one T4: `!python -m gpurt.kernels.bench --quick`.
6. Download `/kaggle/working/*.log` and feed them to notebooks 03–04 (`gpurt.nccltests.parse`).

## 4. Docker on a machine you control (driver + NVIDIA Container Toolkit installed)

```bash
docker run --rm --gpus all -v "$PWD/deploy/any-gpu":/w nvidia/cuda:12.8.1-base-ubuntu24.04 bash /w/probe.sh > out/probe.log
python -m gpurt.container --log out/probe.log      # bind-mounted libcuda.so.<driver>, nvidia-smi, /dev/nvidia*

# CDI instead of the runtime hook (VERIFY the flags for your Docker/Podman version):
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
docker run --rm --device nvidia.com/gpu=all -v "$PWD/deploy/any-gpu":/w nvidia/cuda:12.8.1-base-ubuntu24.04 bash /w/probe.sh

docker build -f deploy/any-gpu/Dockerfile.lab -t gpurt-lab .          # from the lab root
docker run --rm --gpus all gpurt-lab                                    # pip-wheel userland + injected driver
docker build -f deploy/any-gpu/Dockerfile.nccl-tests -t nccl-tests deploy/any-gpu
docker run --rm --gpus all --shm-size=1g nccl-tests all_reduce_perf -b 8 -e 256M -f 2 -g 2
```

Run the probe **without** `--gpus all` once: no device nodes, no driver files — the container runtime,
not the image, decides whether a container has a GPU. NCCL's shared-memory transport needs more than
Docker's default 64 MB `/dev/shm`: pass `--shm-size=1g` or `--ipc=host`.

## 5. RunPod, Vast.ai, Lambda

* **RunPod / Vast.ai** hand you a *container* that is already running with GPUs attached: you cannot
  change the driver or run Docker inside. Pick a template whose driver supports your wheels' CUDA
  version (`python -m gpurt.container` tells you), then use recipe 1. Multi-GPU pods with NVLink make
  the T2 collective sweeps meaningful (`nvidia-smi topo -m` shows `NV#` between GPUs).
* **Lambda** (and GCP VMs) give a full VM with the driver installed: recipe 1, or recipe 4 with Docker.

## What to bring back to the notebooks

| Output | Notebook |
|---|---|
| `gpurt.kernels.bench --json` | 02 — memory-bound kernels on a real GPU |
| `gpurt.dist.bench` / `run_nccl_tests.sh` logs | 03 — collectives, 04 — busbw and the α-β fit |
| `probe.sh` output | 05 — how a container sees a GPU |

Concepts behind each step: [the primer](../../../PRIMER.md) §1 (compatibility), §5 (collectives),
§6 (containers).
