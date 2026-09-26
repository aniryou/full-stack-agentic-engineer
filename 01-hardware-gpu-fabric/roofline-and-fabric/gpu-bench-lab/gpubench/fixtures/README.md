# Fixtures: sample `nvidia-smi` output (illustrative)

These files are **sample output in the documented format** of `nvidia-smi topo -m` and
`nvidia-smi --query-gpu=... --format=csv`, written for parsing practice and tests. They are not
measurements of any real machine; the anomalies in `inventory_hgx_h100_8gpu.csv` (a power-capped GPU,
a link trained at x8, memory held by another process) are planted for notebook 03's exercises.

| File | Shape |
|---|---|
| `topo_hgx_h100_8gpu.txt` | 8 GPUs on NVSwitch (NV18 between every pair), 8 NICs, two NUMA nodes |
| `topo_pcie_4gpu_2socket.txt` | 4 PCIe GPUs, two per socket behind a PCIe switch, one NIC |
| `topo_a2_highgpu_2g.txt` | 2 A100s joined by NVLink (NV12), in the shape of a GCP `a2-highgpu-2g` |
| `topo_2x_t4_pcie.txt` | 2 T4s through the CPU root complex (PHB), in the shape of a Kaggle 2×T4 box |
| `inventory_hgx_h100_8gpu.csv` | an 8×H100 inventory with three planted problems |
| `inventory_colab_t4.csv` | one T4 at idle (PCIe Gen1 of Gen3 — power saving, not a fault) |

To analyse your own machine instead: `python -m gpubench topo` and `python -m gpubench inventory`.
