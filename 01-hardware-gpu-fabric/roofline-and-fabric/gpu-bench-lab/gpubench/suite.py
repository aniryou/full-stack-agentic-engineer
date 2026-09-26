"""Run the whole suite on this machine and write one report — what the CLI and the GCP VM call.

The suite adapts to what it finds. The numpy backend measures the CPU: GEMM by size and dtype,
STREAM with one thread and with every usable core, host memcpy with an α-β fit, and disk reads
cold and warm. The torch backend measures the GPU: tensor-core dtypes (FP8 where supported), HBM
STREAM, host↔device pinned/pageable sweeps with α-β fits (back to back, plus a synchronised
latency sweep of pinned copies), P2P when two or more GPUs are visible, and a streamed disk→GPU
load. Nothing that cannot run is faked: it lands in ``report.skipped`` with the reason — including
"cold" reads of a file that lives on tmpfs, which is RAM. Rooflines are built from the
measurements and set beside the datasheet (GPU) or an ISA estimate (CPU), each labelled with its
source.
"""
from __future__ import annotations

import os
import shutil

from . import __version__, gemm, inventory, loading, membw, p2p, topo, transfer
from .backends import get_backend
from .measure import measure
from .report import Report
from .roofline import measured_roofline, noise_warnings, spec_roofline
from .specs import cpu_peak_flops, lookup

ALL = ("inventory", "gemm", "stream", "transfer", "p2p", "load")


def _cpu_estimate(dtype: str, info: dict) -> dict | None:
    cores, ghz = info.get("usable_cpus") or info.get("logical_cpus"), info.get("ghz_nominal")
    if not (cores and ghz) or dtype not in ("float64", "float32"):
        return None
    b = 8 if dtype == "float64" else 4
    return {"peak_flops": cpu_peak_flops(cores, ghz, info.get("simd_bits", 128), 2, b), "source": "estimate",
            "formula": f"{cores} cpus × {ghz} GHz × {info.get('simd_bits', 128) // (8 * b)} lanes × 2 (FMA) × 2 FMA units",
            "caveat": "estimate (verify): FMA-unit count and AVX clock are assumptions; vCPUs may be hyperthreads"}


def rooflines(be, gemms, streams) -> dict:
    """Measured roofline per dtype, with the spec (GPU) or ISA estimate (CPU) beside it."""
    out, desc = {}, be.describe()
    spec = lookup(desc.get("name", "")) if be.is_gpu else None
    for dtype in dict.fromkeys(m.params["dtype"] for m in gemms):
        r = measured_roofline(gemms, streams, dtype)
        entry = {"measured": r.to_dict()}
        if spec is not None and spec.peak_flops(dtype):
            s = spec_roofline(spec, dtype)
            entry["spec"] = s.to_dict()
            entry["fraction_of_spec"] = {"flops": r.peak_flops / s.peak_flops, "bandwidth": r.peak_bw / s.peak_bw}
        elif not be.is_gpu:
            est = _cpu_estimate(dtype, desc)
            if est:
                entry["estimate"] = est
                entry["fraction_of_estimate"] = r.peak_flops / est["peak_flops"]
        out[dtype] = entry
    return out


def run_suite(backend="auto", quick: bool = True, suites=ALL, workdir=None, log=print, tiny: bool = False) -> Report:
    """Run ``suites`` and return a Report. ``tiny=True`` is a plumbing check for CI and tests:
    real measurements, but of sizes too small to mean anything (the report says so)."""
    be = get_backend(backend, verbose=False) if isinstance(backend, str) else backend
    rep = Report.new(be, quick=quick, tiny=tiny, gpubench=__version__, suites=list(suites))
    if tiny:
        rep.meta["warning"] = "tiny run: plumbing check only — sizes far too small to characterise the machine"
    kw = dict(repeats=2, min_time=0.001) if tiny else {}
    log(f"gpubench {__version__}: backend={be.name} device={be.device} quick={quick} tiny={tiny}")

    if "inventory" in suites:
        rows, raw = inventory.query_gpus()
        if rows:
            rep.analyses["inventory"] = {"gpus": [{k: v for k, v in r.items() if k != "_units"} for r in rows],
                                         "findings": [str(f) for f in inventory.health(rows)], "raw": raw}
        else:
            rep.skipped.append({"what": "nvidia-smi inventory", "reason": raw})
        text = topo.run_nvidia_smi()
        if text:
            try:
                rep.analyses["topology"] = {"summary": topo.parse(text).summary(), "raw": text}
            except ValueError as e:
                rep.analyses["topology"] = {"raw": text, "parse_error": str(e)}

    gemms, streams = [], []
    if "gemm" in suites:
        log("GEMM sweep ...")
        gemms = gemm.sweep(be, sizes=[64, 128] if tiny else None, quick=quick, skipped=rep.skipped, **kw)
        rep.extend(gemms)
    if "stream" in suites:
        log("STREAM ...")
        n = (1 << 16) if tiny else membw.stream_elems(be)
        threads = [1] if be.is_gpu else sorted({1, inventory.usable_cpus()})
        for t in threads:
            ms = membw.stream_suite(be, n, threads=t, **kw)
            streams += ms
            rep.extend(ms)
    if gemms and streams:
        rep.analyses["roofline"] = rooflines(be, gemms, streams)
        warns = noise_warnings(gemms, streams, be.is_gpu)
        if warns:
            rep.meta["noise_warnings"] = warns
            for w in warns:
                log(f"WARNING: {w}")

    if "transfer" in suites:
        log("transfers ...")
        sweep = transfer.sizes(4 << 10, (256 << 10) if tiny else (64 << 20) if quick else (1 << 30), 4)
        if be.is_gpu:
            ms = transfer.hostdevice_sweep(be, sweep, **kw)
            # α of a copy you wait for (one synchronised copy per sample), small sizes, pinned only
            lat = transfer.sizes(4 << 10, (64 << 10) if tiny else (4 << 20), 4)
            ms += transfer.hostdevice_sweep(be, lat, pinned=(True,), latency=True,
                                            **(kw or dict(repeats=20, min_time=0.0)))
        else:
            ms = transfer.memcpy_sweep(be, sweep, **kw)
        rep.extend(ms)
        rep.analyses["alpha_beta"] = {transfer.series_label(key): transfer.fit(s).to_dict()
                                      for key, s in transfer.series(ms).items()}

    if "p2p" in suites:
        count = be.describe().get("device_count", 0) if be.is_gpu else 0
        if count >= 2:
            log("P2P ...")
            size = (1 << 20) if tiny else (64 << 20) if quick else (256 << 20)
            rep.extend(p2p.measure_matrix(be, size, **kw))
            rep.extend(p2p.measure_matrix(be, size, bidirectional=True, **kw))
        else:
            rep.skipped.append({"what": "p2p", "reason": f"needs >= 2 GPUs with the torch backend (have {count})"})

    if "load" in suites:
        log("weights loading ...")
        tmp = workdir or loading.default_workdir()      # not /tmp if that is tmpfs (RAM)
        os.makedirs(tmp, exist_ok=True)
        path = os.path.join(tmp, "synthetic.safetensors")
        try:
            info = loading.synthetic_checkpoint(path, (4 << 20) if tiny else (256 << 20) if quick else (2 << 30),
                                                hidden=256 if tiny else 1024, vocab=4096 if tiny else 32000)
            rep.analyses["checkpoint"] = info
            rep.extend(loading.bench_load(path, threads=(4,) if quick else (4, 8), skipped=rep.skipped,
                                          repeats=2 if tiny else 3))
            if be.is_gpu:
                cold_ok, why = loading.cold_read_possible(path)
                if not cold_ok:
                    rep.skipped.append({"what": "load.to_device cold", "reason": why})
                for cold in ((True, False) if cold_ok else (False,)):
                    setup = (lambda: loading.drop_page_cache(path)) if cold else None
                    op = be.make_file_to_device(path, setup=setup)
                    rep.extend([measure(be, op, "load.to_device", {"nbytes": info["bytes"],
                                                                   "cache": "cold" if cold else "warm"},
                                        repeats=3, min_time=0.0)])
        finally:
            if os.path.exists(path):
                os.remove(path)
            if workdir is None:
                shutil.rmtree(tmp, ignore_errors=True)
    return rep
