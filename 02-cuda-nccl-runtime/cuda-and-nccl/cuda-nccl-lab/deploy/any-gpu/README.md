# deploy/any-gpu — run the lab on whatever GPU you can get

**What it does:** it moves the T1/T2 paths of the lab to any NVIDIA GPU. These paths are the time
measurements of real kernels, the NCCL collectives, nccl-tests and the container probe. The GPU can be
in a free Colab or Kaggle notebook, a rented container (RunPod, Vast.ai), a rented VM (Lambda, GCP) or
your own machine. Nothing in this folder is specific to GCP.

**Cost:** Colab (one T4) and Kaggle (two T4s, 30 GPU-hours/week) are free. A rented 24 GB GPU costs
approximately $0.3–0.7/hr. One hour on a multi-GPU NVLink box costs $2–25. Prices change. Read the
compute guide of the repository, and examine the prices of the provider (verify).

**Cleanup:** after your work, stop or delete the notebook, the pod or the VM. Billing continues
while it exists. If you ran the scripts locally, then delete `out/` (gitignored) and the nccl-tests
build in `~/.cache/nccl-tests-v2.20.0`. If you started the exporter, run `docker rm -f dcgm-exporter`.

| File | What it is |
|---|---|
| `probe.sh` | It prints what a container sees of its GPU: device nodes, injected driver files and versions. It prints them in sections that `python -m gpurt.container --log` explains. |
| `run_nccl_tests.sh` | It builds NVIDIA/nccl-tests against the NCCL that you already have (the system NCCL or the pip wheel of PyTorch). Then it does a sweep of `all_reduce`/`all_gather` on every local GPU. `DRY_RUN=1` prints the steps. |
| `Dockerfile.nccl-tests` | The same build in a CUDA *devel* image. The image contains nvcc and NCCL. |
| `Dockerfile.lab` | The lab on `python:3.12-slim`. The base image has no CUDA. The CUDA userland comes from pip wheels, and `libcuda` comes from the host. |
| `dcgm-counters.csv` | The collectors for dcgm-exporter: the stock defaults, and also the fields that `gpurt.dcgm` must have (clock-event reasons, SM active, SM occupancy). |

## 1. Any GPU box, plain pip (VM, rented container, your workstation)

```bash
git clone --depth 1 https://github.com/aniryou/full-stack-agentic-engineer.git
cd full-stack-agentic-engineer/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab
pip install -e ".[dev,gpu]"                 # numba-cuda[cu12]: NVIDIA's CUDA target for Numba
mkdir -p out                                # results land here (gitignored); the notebooks look for them
python -m gpurt.env                         # tier, GPUs, numba mode
python -m gpurt.container                   # how this process sees the GPU, and the compat verdicts
python -m gpurt.kernels.bench --quick --json out/kernels.json   # notebook 02 prints this file if present
```

`gpurt.kernels.bench` measures the time of real kernels. Thus it must have the GPU and `numba-cuda`. Without
them, it stops with one line and exit status 2 (`python -m gpurt.env` tells you why). On a laptop, the T0
path of notebook 02 is the equivalent.

With two or more GPUs and a CUDA build of PyTorch (T2):

```bash
torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl --op all_reduce -e 256M --json out/ar.json | tee out/ar.log
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl --op all_gather -e 64M  # which transport?
bash deploy/any-gpu/run_nccl_tests.sh       # the reference tool, same busbw definition
python -m gpurt.nccltests out/all_reduce_2gpu.log
```

`ar.log` (our sweep) and `all_reduce_2gpu.log` (nccl-tests) print the same table layout. Thus notebook 04
reads either log from `out/`. Expect the two logs to agree on the plateau busbw. If our busbw is lower at
small sizes, the cause is the launch overhead of `torch.distributed` on the Python side. This overhead is
the α term.

## 2. Colab (one T4, free)

In Colab, open the menu *Runtime*, then *Change runtime type*, then select *T4 GPU*. Then open a notebook
through the Colab links in the layer README. The first cell clones the repository and installs the lab.

Numba cannot use the GPU if numba-cuda or its NVVM is not available, or if the driver and the toolkit do not
agree. In that case, `gpurt.kernels` finds the problem before it selects its mode. It then uses the
simulator and prints the reason. To repair this, run `!pip install -q "numba-cuda[cu12]"`. Then start
the session again.

Colab preinstalls numba-cuda at some times and not at others (verify). Colab gives
one GPU, thus it supports T1 only.

## 3. Kaggle, two T4s (free T2 — collectives over PCIe)

1. Make a new notebook. In *Settings*, set *Accelerator* to *GPU T4 x2*. Set *Internet* to on. For
   this setting, the account must have a verified phone number (verify current rules).
2. Put this code in the first cell:
   ```python
   !git clone --depth 1 https://github.com/aniryou/full-stack-agentic-engineer.git
   %cd full-stack-agentic-engineer/02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab
   !pip install -q -e ".[gpu]"
   !nvidia-smi topo -m          # two T4s on PCIe: no NVLink row, P2P may or may not be enabled
   ```
3. Run NCCL through PyTorch. Kaggle preinstalls PyTorch:
   ```python
   !torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl --op all_reduce -e 256M | tee /kaggle/working/ar.log
   !NCCL_DEBUG=INFO torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl --op all_reduce -b 1M -e 1M -n 5 2>&1 | grep -E "via|Channel" | head
   ```
4. Run nccl-tests against the same NCCL. If there is no system NCCL, the script finds the
   `nvidia-nccl-cu12` wheel of PyTorch. The script must have `nvcc`. The GPU image of Kaggle contains
   `nvcc` (verify):
   ```python
   !OUT=/kaggle/working/out bash deploy/any-gpu/run_nccl_tests.sh
   ```
5. Run the kernels on one T4: `!python -m gpurt.kernels.bench --quick --json /kaggle/working/out/kernels.json`.
6. Download `/kaggle/working/out/`. Put it in the `out/` folder of the lab at home. Notebook 04 reads the
   logs, and notebook 02 reads the kernel JSON. `gpurt.nccltests.parse` works on any single log.

The two T4s are on **PCIe with no NVLink**. `nvidia-smi topo -m` shows a PCIe path between them, such
as `PIX`, `PHB` or `SYS`, and never `NV#`. Thus PCIe Gen3 x16 sets the upper limit of the busbw plateau.
This limit is 15.75 GB/s per direction on paper, and less in practice. The plateau is lower again if
NCCL must go through host memory (`via SHM` in `NCCL_DEBUG=INFO`).

Measure the plateau, and do not trust these limits. Exercise 4.5 of notebook 04 compares your plateau
with the link. The same all-reduce on an NVLink box is one to two orders of magnitude faster.

## 4. Docker on a machine you control (driver + NVIDIA Container Toolkit installed)

```bash
mkdir -p out
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

Run the probe one time **without** `--gpus all`. The output then has no device nodes and no driver
files. The container runtime decides if a container has a GPU. The image does not decide this. The
shared-memory transport of NCCL must have more than the default 64 MB `/dev/shm` of Docker. Pass
`--shm-size=1g` or `--ipc=host`.

## 5. RunPod, Vast.ai, Lambda

* **RunPod / Vast.ai** give you a *container* that already runs, with GPUs attached. You cannot change
  the driver or run Docker in it. Select a template whose driver supports the CUDA version of your
  wheels (`python -m gpurt.container` tells you). Then use recipe 1. On multi-GPU pods with NVLink, the
  T2 collective sweeps give useful results (`nvidia-smi topo -m` shows `NV#` between GPUs).
* **Lambda** (and GCP VMs) give a full VM with the driver installed. Use recipe 1, or recipe 4 with Docker.

## 6. DCGM, MIG and MPS on a GPU VM you control (T1/T2)

For this section, you must have a *VM* (Lambda, a GCP GPU VM, your workstation) with Docker and the
NVIDIA Container Toolkit. RunPod/Vast containers cannot run Docker or change GPU modes.

**DCGM metrics without Kubernetes.** The stock exporter does not export some fields that the lab reads.
Thus give it the collectors file of the lab. The profiling (`DCGM_FI_PROF_*`) fields must have `SYS_ADMIN`:

```bash
mkdir -p out
docker run -d --name dcgm-exporter --gpus all --cap-add SYS_ADMIN -p 9400:9400 \
  -v "$PWD/deploy/any-gpu/dcgm-counters.csv":/etc/dcgm-exporter/lab-counters.csv:ro \
  -e DCGM_EXPORTER_COLLECTORS=/etc/dcgm-exporter/lab-counters.csv \
  nvcr.io/nvidia/k8s/dcgm-exporter:4.6.1-4.8.4-distroless     # VERIFY: tag (the Helm chart's default, 2026-09-26)
python -m gpurt.kernels.bench --quick &                        # something to watch
sleep 20 && curl -s localhost:9400/metrics > out/dcgm.prom
python -m gpurt.dcgm < out/dcgm.prom                           # notebook 06 reads out/*.prom too
docker rm -f dcgm-exporter
```

On a consumer GPU (RTX 4090), it is possible that the profiling fields are not available. The profiling
support of DCGM is mainly for data-center GPUs (verify). `gpurt.dcgm.report` tells you which rules then
cannot fire.

**MIG by hand (A100/H100 VM, root).** For MIG mode, the GPU must be idle. On some systems, you must
also reset the GPU:

```bash
sudo nvidia-smi -i 0 -mig 1                  # enable MIG mode on GPU 0 (then reset/reboot if it asks)
sudo nvidia-smi mig -lgip                    # the GPU-instance profiles this GPU offers, with IDs
sudo nvidia-smi mig -i 0 -cgi 1g.10gb,1g.10gb,3g.40gb -C   # profile names from -lgip (A100 80GB shown; 40GB: 1g.5gb, 3g.20gb)
nvidia-smi -L                                # MIG devices, each with its own UUID
CUDA_VISIBLE_DEVICES=MIG-<uuid> python -m gpurt.kernels.bench --quick   # one slice: its own SMs and memory
sudo nvidia-smi mig -dci && sudo nvidia-smi mig -dgi && sudo nvidia-smi -i 0 -mig 0   # undo
```

**MPS and plain time-slicing.** Without MPS, two processes on one GPU take turns. This is
time-slicing, the default. With the MPS daemon, their kernels can run at the same time:

```bash
python -m gpurt.kernels.bench --quick > out/solo.txt                        # alone
(python -m gpurt.kernels.bench --quick > out/ts_a.txt & python -m gpurt.kernels.bench --quick > out/ts_b.txt; wait)
nvidia-cuda-mps-control -d                                                  # start MPS (same user as the clients)
(python -m gpurt.kernels.bench --quick > out/mps_a.txt & python -m gpurt.kernels.bench --quick > out/mps_b.txt; wait)
echo quit | nvidia-cuda-mps-control                                         # stop MPS
```

Compare the copy bandwidth and the launch overhead in the three pairs (primer §7). Time-slicing
decreases the throughput of each process to approximately one half. When no process fills the GPU
alone, MPS gets back some of that throughput.

## What to bring back to the notebooks

Put everything in the `out/` directory of the lab (gitignored). The notebooks look for the files there.

| Output | Notebook |
|---|---|
| `gpurt.kernels.bench --json out/kernels.json` | 02: memory-bound kernels on a real GPU |
| `gpurt.dist.bench` / `run_nccl_tests.sh` logs (`out/*.log`) | 04: busbw and the α-β fit. The notebook parses every log that it finds. |
| `probe.sh` output | 05: how a container sees a GPU (`python -m gpurt.container --log out/probe.log`) |
| `out/dcgm.prom` | 06: DCGM fields, their values, and the alert rules that can fire |

For the concepts behind each step, read [the primer](../../../PRIMER.md) §1 (compatibility), §5
(collectives) and §6 (containers).
