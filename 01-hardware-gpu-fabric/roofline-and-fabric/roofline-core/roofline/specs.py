"""Spec-sheet literacy: a dated catalogue of accelerators, as plain data.

The one idea: a datasheet is a handful of numbers -- peak FLOP/s per precision,
memory capacity and bandwidth, cache, links and power -- and every performance
argument in this layer is arithmetic on them. Two traps are built into the data
model so you cannot fall into them silently:

* Peaks are stored DENSE. Headline tensor numbers marked with an asterisk
  assume 2:4 structured sparsity and are exactly 2x the dense rate
  (``from_sparse``). LLM inference runs dense.
* Scale-up links are stored as marketed (NVLink / Infinity Fabric: the
  bidirectional total) and exposed per direction, which is what sets a
  transfer's time.

Snapshot: September 2026, from vendor datasheets and cloud docs. Every number
is (verify): SKUs, clocks and power limits get revised, a cloud shape is not
the board, and derived entries are flagged in ``notes``.
Units: TFLOP/s = 1e12 FLOP/s, GB = 1e9 bytes, TB/s = 1e12 bytes/s.
"""
from __future__ import annotations

from dataclasses import dataclass, field

AS_OF = "2026-09"


@dataclass(frozen=True)
class Device:
    name: str
    vendor: str
    arch: str
    memory_gb: float                 # capacity as marketed (DRAM "GB" is really GiB; see primer 1)
    memory_tbs: float                # memory bandwidth, TB/s
    tflops: dict = field(default_factory=dict)   # DENSE peak, TFLOP/s, by precision
    memory_type: str = "HBM"
    l2_mb: float | None = None       # last-level on-chip cache (L2, or AMD Infinity Cache)
    scaleup_gbs: float | None = None  # GPU<->GPU scale-up bandwidth per GPU, as marketed
    scaleup_domain: int = 1          # accelerators in one scale-up (NVLink/xGMI/ICI) domain
    host_link: str = ""
    tdp_w: float | None = None
    year: int = 0                    # first availability, approximate
    gcp: str = ""                    # how it is offered on Google Cloud, if at all
    notes: str = ""

    def supports(self, precision: str) -> bool:
        return precision in self.tflops

    def peak(self, precision: str = "bf16") -> float:
        """Dense peak in FLOP/s. Raises if the part has no such path (T4 has no bf16)."""
        if precision not in self.tflops:
            have = ", ".join(self.tflops)
            raise KeyError(f"{self.name} has no {precision} path (it has: {have})")
        return self.tflops[precision] * 1e12

    def bandwidth(self) -> float:
        """Memory bandwidth in bytes/s."""
        return self.memory_tbs * 1e12

    def ridge(self, precision: str = "bf16") -> float:
        """FLOP per byte at which this device stops being memory-bound."""
        return self.peak(precision) / self.bandwidth()

    @property
    def scaleup_gbs_per_dir(self) -> float | None:
        """Marketed scale-up figures are bidirectional totals; a transfer sees half."""
        return None if self.scaleup_gbs is None else self.scaleup_gbs / 2


def from_sparse(headline_tflops: float) -> float:
    """Convert a 2:4-sparsity headline (the asterisked number) to the dense rate."""
    return headline_tflops / 2


def peak_from_clock(sms: int, flops_per_clk_per_sm: int, boost_ghz: float) -> float:
    """A peak is just units x work-per-clock x clock. Returns TFLOP/s.

    A100: 108 SMs x 2048 dense fp16 FLOP/clk/SM x 1.41 GHz = 311.9 TFLOP/s.
    Sustained clocks under power/thermal limits are lower, so a peak is a ceiling.
    """
    return sms * flops_per_clk_per_sm * boost_ghz * 1e9 / 1e12


# SMs, dense fp16 tensor FLOP/clk/SM, boost GHz -- the per-SM rate doubles A100 -> H100.
CLOCKS = {
    "t4": (40, 1024, 1.59),
    "l4": (58, 1024, 2.04),
    "a100-80gb": (108, 2048, 1.41),
    "h100-sxm": (132, 4096, 1.83),
}


DEVICES: dict[str, Device] = {
    "t4": Device(name="NVIDIA T4", vendor="NVIDIA", arch="Turing", year=2018,
                 memory_gb=16, memory_tbs=0.32, memory_type="GDDR6", l2_mb=4,
                 tflops={"fp32": 8.1, "fp16": 65, "int8": 130, "int4": 260},
                 host_link="PCIe Gen3 x16", tdp_w=70, gcp="N1 + T4 (1-4 per VM)",
                 notes="no bf16/tf32/fp8 paths; no NVLink"),
    "l4": Device(name="NVIDIA L4", vendor="NVIDIA", arch="Ada Lovelace", year=2023,
                 memory_gb=24, memory_tbs=0.30, memory_type="GDDR6", l2_mb=48,
                 tflops={"fp32": 30.3, "tf32": 60, "fp16": 121, "bf16": 121, "fp8": 242.5, "int8": 242.5},
                 host_link="PCIe Gen4 x16", tdp_w=72, gcp="G2 (g2-standard-4 = 1 L4); Cloud Run",
                 notes="datasheet headlines are sparse (242/485); no NVLink"),
    "a100-40gb": Device(name="NVIDIA A100 SXM 40GB", vendor="NVIDIA", arch="Ampere", year=2020,
                        memory_gb=40, memory_tbs=1.555, l2_mb=40,
                        tflops={"fp32": 19.5, "tf32": 156, "fp16": 312, "bf16": 312, "int8": 624},
                        scaleup_gbs=600, scaleup_domain=8, host_link="PCIe Gen4 x16", tdp_w=400,
                        gcp="A2 (a2-highgpu-1g..8g)"),
    "a100-80gb": Device(name="NVIDIA A100 SXM 80GB", vendor="NVIDIA", arch="Ampere", year=2020,
                        memory_gb=80, memory_tbs=2.039, l2_mb=40,
                        tflops={"fp32": 19.5, "tf32": 156, "fp16": 312, "bf16": 312, "int8": 624},
                        scaleup_gbs=600, scaleup_domain=8, host_link="PCIe Gen4 x16", tdp_w=400,
                        gcp="A2 ultra (a2-ultragpu-1g..8g)"),
    "h100-sxm": Device(name="NVIDIA H100 SXM", vendor="NVIDIA", arch="Hopper", year=2022,
                       memory_gb=80, memory_tbs=3.35, l2_mb=50,
                       tflops={"fp32": 67, "tf32": 494.7, "fp16": 989.4, "bf16": 989.4,
                               "fp8": 1978.9, "int8": 1978.9},
                       scaleup_gbs=900, scaleup_domain=8, host_link="PCIe Gen5 x16", tdp_w=700,
                       gcp="A3 (a3-highgpu-8g; A3 Mega with GPUDirect-TCPXO)",
                       notes="datasheet headlines are sparse (1,979 bf16 / 3,958 fp8)"),
    "h200": Device(name="NVIDIA H200 SXM", vendor="NVIDIA", arch="Hopper", year=2024,
                   memory_gb=141, memory_tbs=4.8, memory_type="HBM3e", l2_mb=50,
                   tflops={"fp32": 67, "tf32": 494.7, "fp16": 989.4, "bf16": 989.4,
                           "fp8": 1978.9, "int8": 1978.9},
                   scaleup_gbs=900, scaleup_domain=8, host_link="PCIe Gen5 x16", tdp_w=700,
                   gcp="A3 Ultra (RDMA)", notes="H100 compute, more and faster memory"),
    "b200": Device(name="NVIDIA B200 (HGX)", vendor="NVIDIA", arch="Blackwell", year=2025,
                   memory_gb=180, memory_tbs=8.0, memory_type="HBM3e",
                   tflops={"fp32": 75, "tf32": 1125, "fp16": 2250, "bf16": 2250, "fp8": 4500,
                           "fp4": 9000, "int8": 4500},
                   scaleup_gbs=1800, scaleup_domain=8, host_link="PCIe Gen5 x16 (verify)", tdp_w=1000,
                   gcp="A4 (8 GPUs)", notes="192 GB physical, 180 GB usable in HGX (verify)"),
    "gb200": Device(name="NVIDIA GB200 (per GPU, NVL72)", vendor="NVIDIA", arch="Blackwell", year=2025,
                    memory_gb=186, memory_tbs=8.0, memory_type="HBM3e",
                    tflops={"fp32": 90, "tf32": 1250, "fp16": 2500, "bf16": 2500, "fp8": 5000,
                            "fp4": 10000, "int8": 5000},
                    scaleup_gbs=1800, scaleup_domain=72, host_link="NVLink-C2C to Grace", tdp_w=1200,
                    gcp="A4X (GB200 NVL72)",
                    notes="rack-scale NVLink domain of 72 GPUs; 186 GB = 13.4 TB HBM / 72 GPUs (verify)"),
    "gb300": Device(name="NVIDIA GB300 (per GPU, NVL72)", vendor="NVIDIA", arch="Blackwell Ultra",
                    year=2025, memory_gb=288, memory_tbs=8.0, memory_type="HBM3e",
                    tflops={"fp16": 2500, "bf16": 2500, "fp8": 5000, "fp4": 15000},
                    scaleup_gbs=1800, scaleup_domain=72, host_link="NVLink-C2C to Grace", tdp_w=1400,
                    gcp="A4X Max (GB300 NVL72)",
                    notes="1.5x Blackwell dense FP4; INT8/FP64 cut; 1,400 W TDP (verify)"),
    "rtx-pro-6000": Device(name="NVIDIA RTX PRO 6000 Blackwell Server", vendor="NVIDIA", arch="Blackwell",
                           year=2025, memory_gb=96, memory_tbs=1.6, memory_type="GDDR7", l2_mb=128,
                           tflops={"fp32": 120, "fp16": 500, "bf16": 500, "fp8": 1000, "fp4": 2000},
                           host_link="PCIe Gen5 x16", tdp_w=600, gcp="G4; Cloud Run",
                           notes="tensor rates derived from the 4 PFLOPS sparse-FP4 headline (verify)"),
    "mi300x": Device(name="AMD Instinct MI300X", vendor="AMD", arch="CDNA 3", year=2023,
                     memory_gb=192, memory_tbs=5.3, l2_mb=256,
                     tflops={"fp32": 163.4, "tf32": 653.7, "fp16": 1307.4, "bf16": 1307.4,
                             "fp8": 2614.9, "int8": 2614.9},
                     scaleup_gbs=896, scaleup_domain=8, host_link="PCIe Gen5 x16", tdp_w=750,
                     notes="l2_mb is the 256 MB Infinity Cache; xGMI 7 x 128 GB/s"),
    "mi325x": Device(name="AMD Instinct MI325X", vendor="AMD", arch="CDNA 3", year=2024,
                     memory_gb=256, memory_tbs=6.0, memory_type="HBM3e", l2_mb=256,
                     tflops={"fp32": 163.4, "tf32": 653.7, "fp16": 1307.4, "bf16": 1307.4,
                             "fp8": 2614.9, "int8": 2614.9},
                     scaleup_gbs=896, scaleup_domain=8, host_link="PCIe Gen5 x16", tdp_w=1000),
    "mi355x": Device(name="AMD Instinct MI355X", vendor="AMD", arch="CDNA 4", year=2025,
                     memory_gb=288, memory_tbs=8.0, memory_type="HBM3e", l2_mb=256,
                     tflops={"fp32": 157.3, "fp16": 2500, "bf16": 2500, "fp8": 5000, "fp4": 10000,
                             "int8": 5000},
                     scaleup_gbs=1075, scaleup_domain=8, host_link="PCIe Gen5 x16", tdp_w=1400,
                     notes="fp4 = MXFP4; xGMI 7 x 153.6 GB/s (verify)"),
    "tpu-v5e": Device(name="Google TPU v5e", vendor="Google", arch="TPU v5e", year=2023,
                      memory_gb=16, memory_tbs=0.819,
                      tflops={"bf16": 197, "int8": 394},
                      scaleup_gbs=200, scaleup_domain=256, host_link="host PCIe",
                      gcp="Cloud TPU / GKE (ct5lp)", notes="ICI 1,600 Gbps per chip, 2D torus"),
    "tpu-v6e": Device(name="Google TPU v6e (Trillium)", vendor="Google", arch="TPU v6e", year=2024,
                      memory_gb=32, memory_tbs=1.64,
                      tflops={"bf16": 918, "int8": 1836},
                      scaleup_gbs=448, scaleup_domain=256, host_link="host PCIe",
                      gcp="Cloud TPU / GKE (ct6e)", notes="ICI 3,584 Gbps per chip"),
    "tpu-v7": Device(name="Google TPU7x (Ironwood)", vendor="Google", arch="TPU v7", year=2026,
                     memory_gb=192, memory_tbs=7.37, memory_type="HBM3e",
                     tflops={"bf16": 2307, "fp8": 4614},
                     scaleup_gbs=1200, scaleup_domain=9216, host_link="host PCIe",
                     gcp="GKE (TPU7x), GA 2026-04",
                     notes="bf16 assumed half of the fp8 headline (verify); ICI 1.2 TB/s bidirectional"),
}


def get(name: str) -> Device:
    """Look a device up by key ('h100-sxm') or by a unique fragment of its name ('H100')."""
    key = name.lower().strip()
    if key in DEVICES:
        return DEVICES[key]
    hits = [d for k, d in DEVICES.items() if key in k or key in d.name.lower()]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"{name!r} matches {len(hits)} devices; use one of {sorted(DEVICES)}")


def table(keys=None, precision: str = "bf16") -> str:
    """The landscape as a Markdown table (dense peaks; ridge computed, not quoted).

    The main TF column and the ridge use `precision`, falling back to bf16 then fp16 for
    parts without it (a TPU has no fp16, a T4 no bf16 or fp8).
    """
    main = "bf16/fp16" if precision == "bf16" else f"{precision} (else bf16/fp16)"
    rows = [f"| Device | Mem GB | TB/s | {main} TF | fp8 TF | fp4 TF | Ridge FLOP/B | Scale-up GB/s (domain) | TDP W |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k in keys or DEVICES:
        d = DEVICES[k]
        p = next(x for x in (precision, "bf16", "fp16") if d.supports(x))
        fmt = lambda x: "–" if x is None else f"{x:,.0f}" if x >= 100 else f"{x:g}"
        up = "PCIe only" if d.scaleup_gbs is None else f"{d.scaleup_gbs:,.0f} ({d.scaleup_domain})"
        rows.append(f"| {d.name} | {d.memory_gb:g} | {d.memory_tbs:g} | {fmt(d.tflops.get(p))} | "
                    f"{fmt(d.tflops.get('fp8'))} | {fmt(d.tflops.get('fp4'))} | {d.ridge(p):.0f} | "
                    f"{up} | {fmt(d.tdp_w)} |")
    return "\n".join(rows)
