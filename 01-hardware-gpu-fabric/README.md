# 01 · Hardware — GPUs, NVLink, NICs, storage, cooling

The physical layer: silicon, interconnect, and the datacenter around it. Everything above this
layer is ultimately bounded by what lives here.

**Covers:** GPU SKUs & memory (H100 / H200 / B200, HBM), NVLink / NVSwitch, scale-up vs
scale-out fabrics, RDMA NICs (InfiniBand, RoCE), network topology, storage, power and cooling.

**Signal keywords:** NVLink, NVSwitch, InfiniBand, RoCE, HBM, memory bandwidth, scale-up/scale-out,
rail-optimized, GPUDirect, RDMA, power/cooling, TCO, fabric.

## Current contents
- **`gpu-primer/`** — why a GPU is shaped the way it is, from first principles (primer + exercises).
- **`gpu-deployment/`** — GPU deployment architecture for scale-up: the scale-up (NVLink) vs
  scale-out (InfiniBand/RoCE) fabric distinction and what it means for LLM serving (primer + exercises).

_Drop more hardware / interconnect / datacenter material here._

<!-- colab-links:start -->
## Run in Colab

One-time Colab setup is in [`../COLAB.md`](../COLAB.md). Exercises are under `notebooks/` / `exercises/`; worked answers under `solutions/`.

_No notebooks yet._
<!-- colab-links:end -->
