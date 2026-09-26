"""Read nccl-tests output (``all_reduce_perf`` and friends) and re-check its numbers yourself.

The one idea: nccl-tests' table is the lingua franca for collective performance, and every column
can be recomputed: ``algbw = size / time`` and ``busbw = algbw x factor(op, nranks)``
(``gpurt.dist.busbw``). Parsing it into rows lets you (1) verify the factor instead of trusting it,
(2) fit the α-β model to find the latency floor and the half-bandwidth message size, and
(3) compare the plateau busbw with the link's peak (NVLink, PCIe, the NIC) — primer §5.

Handles the current format (``size count type redop root`` then out-of-place and in-place
``time algbw busbw #wrong``) and the older one without ``redop``/``root`` and with a float ``error``
column; ignores interleaved ``NCCL INFO`` lines. Our own ``gpurt.dist`` sweeps print a one-block
variant of the same table, which parses too.

    python -m gpurt.nccltests all_reduce.log            # summary: peak busbw, α-β fit, S½
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

from .dist import busbw as bw
from .dist.alphabeta import AlphaBeta, fit

_DEVICE = re.compile(r"#\s+Rank\s+(\d+)\s+Group\s+(\d+)\s+Pid\s+(\d+)\s+on\s+(\S+)\s+device\s+(\d+)\s+\[([^\]]+)\]\s+(.*)")
_HEADER_INT = re.compile(r"\b(nThread|nGpus|minBytes|maxBytes)\s+(\d+)")
_HEADER_COLON = re.compile(r"\b(warmup iters|agg iters|iters|validation|graph):\s*(\d+)")
_STARTING = re.compile(r"Collective test starting:\s*(\S+)")
_BACKEND = re.compile(r"#\s*backend:\s*(\S+)\s+op:\s*(\S+)\s+nranks:\s*(\d+)")


@dataclass
class NcclRow:
    size: int
    count: int
    dtype: str
    redop: str | None
    root: int | None
    time_us: float  # out-of-place (or the only block)
    algbw: float
    busbw: float
    wrong: float | None  # None when validation was off ("N/A")
    ip_time_us: float | None = None  # in-place block, when present
    ip_algbw: float | None = None
    ip_busbw: float | None = None
    ip_wrong: float | None = None


@dataclass
class NcclTestsResult:
    op: str | None
    nranks: int | None
    header: dict = field(default_factory=dict)
    devices: list[dict] = field(default_factory=list)
    rows: list[NcclRow] = field(default_factory=list)
    avg_busbw: float | None = None
    out_of_bounds: int | None = None
    backend: str = "nccl-tests"


def _num(tok: str) -> float | None:
    return None if tok.upper() == "N/A" else float(tok)


def parse(text: str, op: str | None = None, nranks: int | None = None) -> NcclTestsResult:
    """Parse one nccl-tests run. ``op`` is taken from the "Collective test starting" line when present;
    ``nranks`` from the "Using devices" list (pass it when parsing output without that list)."""
    res = NcclTestsResult(op=bw.canonical(op) if op else None, nranks=nranks)
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            if m := _DEVICE.match(s):
                res.devices.append({"rank": int(m[1]), "group": int(m[2]), "pid": int(m[3]), "host": m[4],
                                    "device": int(m[5]), "bus_id": m[6], "name": m[7].strip()})
            elif "nThread" in s:
                res.header.update({k: int(v) for k, v in _HEADER_INT.findall(s)})
                res.header.update({k: int(v) for k, v in _HEADER_COLON.findall(s)})
            elif "Avg bus bandwidth" in s:
                res.avg_busbw = float(s.split(":")[-1])
            elif "Out of bounds values" in s:
                res.out_of_bounds = int(s.split(":")[-1].split()[0])
            elif (m := _STARTING.search(s)) and res.op is None:
                res.op = bw.canonical(m[1])
            elif m := _BACKEND.match(s):
                res.backend = m[1]
                res.op = res.op or bw.canonical(m[2])
                res.nranks = res.nranks or int(m[3])
            continue
        tok = s.split()
        if len(tok) < 7 or not (tok[0].isdigit() and tok[1].isdigit()):
            continue  # NCCL INFO lines, warnings, anything else
        rest = tok[3:]
        redop = root = None
        if len(rest) in (6, 10):  # current format: redop and root columns present
            redop, root, rest = rest[0], int(rest[1]), rest[2:]
        if len(rest) not in (4, 8):
            continue
        row = NcclRow(int(tok[0]), int(tok[1]), tok[2], redop, root,
                      float(rest[0]), float(rest[1]), float(rest[2]), _num(rest[3]))
        if len(rest) == 8:
            row.ip_time_us, row.ip_algbw, row.ip_busbw, row.ip_wrong = (
                float(rest[4]), float(rest[5]), float(rest[6]), _num(rest[7]))
        res.rows.append(row)
    if res.nranks is None:
        if res.devices:
            res.nranks = len(res.devices)
        elif "nGpus" in res.header:
            res.nranks = res.header["nGpus"] * res.header.get("nThread", 1)  # single process only
    return res


def split_runs(text: str) -> list[str]:
    """One log often holds several runs (all_reduce_perf, then all_gather_perf, ...). Split at each
    "Collective test starting" line, or else at each "# nThread" header."""
    lines = text.splitlines()
    marks = [i for i, line in enumerate(lines) if "Collective test starting" in line]
    if not marks:
        marks = [i for i, line in enumerate(lines) if line.lstrip().startswith("# nThread")]
    if len(marks) < 2:
        return [text]
    marks[0] = 0  # any preamble belongs to the first run
    return ["\n".join(lines[a:b]) for a, b in zip(marks, marks[1:] + [len(lines)])]


def parse_many(text: str, nranks: int | None = None) -> list[NcclTestsResult]:
    """Parse every run in a log; chunks without result rows are dropped."""
    return [r for r in (parse(chunk, nranks=nranks) for chunk in split_runs(text)) if r.rows]


def recheck(res: NcclTestsResult, rel_tol: float = 0.02, abs_tol: float = 0.011) -> list[str]:
    """Recompute algbw and busbw from size and time; return a list of disagreements (empty = consistent).
    Printed values are rounded (two decimals, time to 3-4 significant digits), hence the tolerances."""
    if res.op is None or res.nranks is None:
        raise ValueError("need op and nranks to recheck (pass them to parse())")
    factor = bw.bus_factor(res.op, res.nranks)
    issues = []
    for r in res.rows:
        for label, t, alg, bus in (("out-of-place", r.time_us, r.algbw, r.busbw),
                                    ("in-place", r.ip_time_us, r.ip_algbw, r.ip_busbw)):
            if t is None:
                continue
            want_alg = r.size / t / 1e3  # bytes/µs -> GB/s
            for name, got, want in (("algbw", alg, want_alg), ("busbw", bus, want_alg * factor)):
                if abs(got - want) > max(abs_tol, rel_tol * abs(want)):
                    issues.append(f"size {r.size} {label}: {name} printed {got:.2f}, recomputed {want:.2f}")
    return issues


def rows_for_fit(res: NcclTestsResult, in_place: bool = False) -> list[dict]:
    out = []
    for r in res.rows:
        t = r.ip_time_us if in_place else r.time_us
        if t is not None and r.size > 0:
            out.append({"size": r.size, "time_us": t})
    return out


def fit_alpha_beta(res: NcclTestsResult, in_place: bool = False) -> AlphaBeta:
    pts = rows_for_fit(res, in_place)
    return fit([p["size"] for p in pts], [p["time_us"] * 1e-6 for p in pts])


def summarize(res: NcclTestsResult) -> dict:
    if not res.rows:
        raise ValueError("no result rows found")
    peak = max(res.rows, key=lambda r: r.busbw)
    ab = fit_alpha_beta(res)
    out = {"op": res.op, "nranks": res.nranks, "backend": res.backend, "points": len(res.rows),
           "peak_busbw_gbps": peak.busbw, "peak_at_bytes": peak.size,
           "min_time_us": min(r.time_us for r in res.rows), "alpha_us": ab.alpha_s * 1e6,
           "algbw_asymptote_gbps": ab.bw_Bps / 1e9, "n_half_bytes": ab.n_half, "fit_rel_rms": ab.rel_rms,
           "avg_busbw_reported": res.avg_busbw,
           "wrong_total": sum((r.wrong or 0) + (r.ip_wrong or 0) for r in res.rows)}
    if res.op and res.nranks:
        out["busbw_asymptote_gbps"] = ab.peak_busbw_gbps(res.op, res.nranks)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Summarise nccl-tests output.")
    p.add_argument("path")
    p.add_argument("--op", help="collective, if the log does not say (e.g. all_reduce)")
    p.add_argument("--nranks", type=int)
    a = p.parse_args(argv)
    with open(a.path, encoding="utf-8") as f:
        text = f.read()
    runs = parse_many(text, a.nranks) if a.op is None and len(split_runs(text)) > 1 else [parse(text, a.op, a.nranks)]
    for res in runs:
        _print_summary(res)
    return 0


def _print_summary(res: NcclTestsResult) -> None:
    s = summarize(res)
    print(f"{s['op']} over {s['nranks']} ranks ({s['backend']}), {s['points']} sizes")
    print(f"  peak busbw {s['peak_busbw_gbps']:.2f} GB/s at {s['peak_at_bytes']} B; latency floor {s['min_time_us']:.1f} µs")
    print(f"  α-β fit: α = {s['alpha_us']:.1f} µs, algbw -> {s['algbw_asymptote_gbps']:.2f} GB/s"
          f"{', busbw -> %.2f GB/s' % s['busbw_asymptote_gbps'] if 'busbw_asymptote_gbps' in s else ''}; "
          f"S½ = {s['n_half_bytes'] / 1024:.0f} KiB")
    if res.op and res.nranks:
        issues = recheck(res)
        print("  recheck: " + ("algbw/busbw consistent with size/time and the factor" if not issues
                              else f"{len(issues)} disagreement(s), first: {issues[0]}"))


if __name__ == "__main__":
    raise SystemExit(main())
