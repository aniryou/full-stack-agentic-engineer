# deploy/any-gpu — run the lab on whatever GPU you can get

**What it does:** takes the T1/T2 paths of the lab (real kernel timings, NCCL collectives,
nccl-tests, the container probe) to any NVIDIA GPU: a free Colab or Kaggle notebook, a rented
container (RunPod, Vast.ai) or VM (Lambda, GCP), or your own machine. Nothing here is GCP-specific.

**Cost:** free on Colab (one T4) and Kaggle (two T4s, 30 GPU-hours/week); roughly $0.3–0.7/hr for a
rented 24 GB GPU and $2–25 for an hour on a multi-GPU NVLink box. Prices move — see the repo's
compute guide and check the provider (verify).

**Cleanup:** stop or terminate the notebook/pod/VM when you are done (billing runs while it exists),
then delete `out/` (gitignored) and the nccl-tests build in `~/.cache/nccl-tests-v2.20.0` if you ran
the scripts locally; `docker rm -f dcgm-exporter` if you started it.

| File | What it is |
|---|---|
| `probe.sh` | prints what a container sees of its GPU (device nodes, injected driver files, versions) in sections that `python -m gpurt.container --log` explains |
| `run_nccl_tests.sh` | builds NVIDIA/nccl-tests against the NCCL you already have (system or PyTorch's pip wheel) and sweeps `all_reduce`/`all_gather` on every local GPU; `DRY_RUN=1` prints the steps |
| `Dockerfile.nccl-tests` | the same in a CUDA *devel* image (nvcc + NCCL inside) |
| `Dockerfile.lab` | the lab on `python:3.12-slim` — no CUDA in the base image: the CUDA userland comes from pip wheels, `libcuda` from the host |
| `dcgm-counters.csv` | dcgm-exporter collectors: the stock defaults plus the fields `gpurt.dcgm` needs (clock-event reasons, SM active, SM occupancy) |

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

With two or more GPUs and a CUDA build of PyTorch (T2):

```bash
torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl --op all_reduce -e 256M --json out/ar.json | tee out/ar.log
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 -m gpurt.dist.bench --backend nccl --op all_gather -e 64M  # which transport?
bash deploy/any-gpu/run_nccl_tests.sh       # the reference tool, same busbw definition
python -m gpurt.nccltests out/all_reduce_2gpu.log
```

`ar.log` (our sweep) and `all_reduce_2gpu.log` (nccl-tests) print the same table layout, so notebook 04
reads either from `out/`. They should agree on the plateau busbw; if ours is lower at small sizes, that is the
Python-side launch overhead of `torch.distributed` — the α term.

## 2. Colab (one T4, free)

*Runtime → Change runtime type → T4 GPU*, then open any notebook through the Colab links in the layer
README; the first cell clones the repo and installs the lab. If Numba cannot use the GPU (numba-cuda or
its NVVM missing, or a driver/toolkit mismatch), `gpurt.kernels` notices before choosing its mode, falls
back to the simulator and prints why: run `!pip install -q "numba-cuda[cu12]"` and restart the session
(whether Colab preinstalls numba-cuda changes over time — verify). Colab gives one GPU: T1 only.

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
5. Kernels on one T4: `!python -m gpurt.kernels.bench --quick --json /kaggle/working/out/kernels.json`.
6. Download `/kaggle/working/out/` and put it in the lab's `out/` at home: notebook 04 reads the logs,
   notebook 02 the kernel JSON (`gpurt.nccltests.parse` works on any single log).

The two T4s sit on **PCIe with no NVLink** (`nvidia-smi topo -m` shows a PCIe path such as `PIX`, `PHB`
or `SYS` between them, never `NV#`). The busbw plateau is therefore bounded by PCIe Gen3 x16 — 15.75 GB/s
per direction on paper, less in practice — and lower still if NCCL has to go through host memory (`via
SHM` in `NCCL_DEBUG=INFO`). Measure it rather than trusting these bounds; notebook 04's exercise 4.5
compares your plateau with the link. The same all-reduce on an NVLink box is one to two orders of
magnitude faster.

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

Run the probe **without** `--gpus all` once: no device nodes, no driver files — the container runtime,
not the image, decides whether a container has a GPU. NCCL's shared-memory transport needs more than
Docker's default 64 MB `/dev/shm`: pass `--shm-size=1g` or `--ipc=host`.

## 5. RunPod, Vast.ai, Lambda

* **RunPod / Vast.ai** hand you a *container* that is already running with GPUs attached: you cannot
  change the driver or run Docker inside. Pick a template whose driver supports your wheels' CUDA
  version (`python -m gpurt.container` tells you), then use recipe 1. Multi-GPU pods with NVLink make
  the T2 collective sweeps meaningful (`nvidia-smi topo -m` shows `NV#` between GPUs).
* **Lambda** (and GCP VMs) give a full VM with the driver installed: recipe 1, or recipe 4 with Docker.

## 6. DCGM, MIG and MPS on a GPU VM you control (T1/T2)

Needs a *VM* (Lambda, a GCP GPU VM, your workstation) with Docker and the NVIDIA Container Toolkit;
RunPod/Vast containers cannot run Docker or change GPU modes.

**DCGM metrics without Kubernetes.** The stock exporter leaves out fields the lab reads, so give it the
lab's collectors file; profiling (`DCGM_FI_PROF_*`) fields need `SYS_ADMIN`:

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

On a consumer GPU (RTX 4090) the profiling fields may be missing — DCGM's profiling support is
data-center-GPU oriented (verify); `gpurt.dcgm.report` says which rules then cannot fire.

**MIG by hand (A100/H100 VM, root).** MIG mode needs the GPU idle, and on some systems a reset:

```bash
sudo nvidia-smi -i 0 -mig 1                  # enable MIG mode on GPU 0 (then reset/reboot if it asks)
sudo nvidia-smi mig -lgip                    # the GPU-instance profiles this GPU offers, with IDs
sudo nvidia-smi mig -i 0 -cgi 1g.10gb,1g.10gb,3g.40gb -C   # profile names from -lgip (A100 80GB shown; 40GB: 1g.5gb, 3g.20gb)
nvidia-smi -L                                # MIG devices, each with its own UUID
CUDA_VISIBLE_DEVICES=MIG-<uuid> python -m gpurt.kernels.bench --quick   # one slice: its own SMs and memory
sudo nvidia-smi mig -dci && sudo nvidia-smi mig -dgi && sudo nvidia-smi -i 0 -mig 0   # undo
```

**MPS and plain time-slicing.** Two processes on one GPU without MPS take turns (time-slicing, the
default); with the MPS daemon their kernels can run concurrently:

```bash
python -m gpurt.kernels.bench --quick > out/solo.txt                        # alone
(python -m gpurt.kernels.bench --quick > out/ts_a.txt & python -m gpurt.kernels.bench --quick > out/ts_b.txt; wait)
nvidia-cuda-mps-control -d                                                  # start MPS (same user as the clients)
(python -m gpurt.kernels.bench --quick > out/mps_a.txt & python -m gpurt.kernels.bench --quick > out/mps_b.txt; wait)
echo quit | nvidia-cuda-mps-control                                         # stop MPS
```

Compare the copy bandwidth and launch overhead in the three pairs (primer §7): time-slicing roughly
halves each process's throughput; MPS recovers some of it when neither process fills the GPU alone.

## What to bring back to the notebooks

Put everything in the lab's `out/` directory (gitignored); the notebooks look there.

| Output | Notebook |
|---|---|
| `gpurt.kernels.bench --json out/kernels.json` | 02 — memory-bound kernels on a real GPU |
| `gpurt.dist.bench` / `run_nccl_tests.sh` logs (`out/*.log`) | 04 — busbw and the α-β fit (it parses every log it finds) |
| `probe.sh` output | 05 — how a container sees a GPU (`python -m gpurt.container --log out/probe.log`) |
| `out/dcgm.prom` | 06 — DCGM fields, readings and which alert rules can fire |

Concepts behind each step: [the primer](../../../PRIMER.md) §1 (compatibility), §5 (collectives),
§6 (containers).
