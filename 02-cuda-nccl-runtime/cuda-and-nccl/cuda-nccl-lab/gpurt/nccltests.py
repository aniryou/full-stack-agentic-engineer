"""Read nccl-tests output (``all_reduce_perf`` and friends) and re-check its numbers yourself.

The one idea: nccl-tests' table is the lingua franca for collective performance, and every column
can be recomputed: ``algbw = size / time`` and ``busbw = algbw x factor(op, nranks)``
(``gpurt.dist.busbw``). Parsing it into rows lets you (1) verify the factor instead of trusting it,
(2) fit the α-β model to find the latency floor and the half-bandwidth message size, and
(3) compare the plateau busbw with the link's peak (NVLink, PCIe, the NIC) — primer §5.

Layouts: the current one (v2.20.0: ``size count type redop root`` then out-of-place and in-place
``time algbw busbw #wrong``), v2.9-era logs with ``redop`` but no ``root``, older ones with neither
and a float ``error`` column, and the optional per-iteration (``-I 1``: ``i_min i_max i_p99 i_cv%``
after each block), timestamp (``-S 1``) and tuning (``-U 1``) columns, which are skipped. The
column-header line decides the layout when the log has one; otherwise the column count does. A
numeric row that fits no layout is counted in ``skipped`` (and reported), never dropped silently.
Interleaved ``NCCL INFO`` lines are ignored. Our own ``gpurt.dist`` sweeps print a one-block variant
of the same table, which parses too.

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
_HEADER_COLON = re.compile(r"\b(warmup iters|agg iters|iters|validation|graph|unalign):\s*(\d+)")
_STARTING = re.compile(r"Collective test starting:\s*(\S+)")
_VERSION = re.compile(r"#\s*nccl-tests version\s+(\S+)")
_COLUMNS = re.compile(r"#\s*size\s+count\s+type\b(.*)")
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
    skipped: list[str] = field(default_factory=list)  # numeric rows whose columns fit no known layout


def _num(tok: str) -> float | None:
    return None if tok.upper() == "N/A" else float(tok)


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return tok.upper() == "N/A"


@dataclass(frozen=True)
class _Layout:
    redop: bool  # a redop column after type
    root: bool  # a root column after redop
    per_iter: bool = False  # i_min i_max i_p99 i_cv% after each time/algbw/busbw/#wrong block


def _layout_from_columns(tail: str) -> _Layout:
    """From the column-header line: ``#  size  count  type  [redop  [root]]  time ...``."""
    cols = tail.split()
    return _Layout("redop" in cols, "root" in cols, "i_min" in cols)


def _guess_layout(rest: list[str]) -> _Layout | None:
    """No header seen: decide from the tokens after ``type`` (4 or 8 numbers per row, plus prefixes)."""
    if not rest:
        return None
    if _is_number(rest[0]):
        return _Layout(False, False) if len(rest) in (4, 8) else None
    if len(rest) in (5, 9):
        return _Layout(True, False)  # v2.9-era: redop, no root
    if len(rest) in (6, 10):
        return _Layout(True, True)
    return None


def parse(text: str, op: str | None = None, nranks: int | None = None) -> NcclTestsResult:
    """Parse one nccl-tests run. ``op`` is taken from the "Collective test starting" line when present;
    ``nranks`` from the "Using devices" list (pass it when parsing output without that list)."""
    res = NcclTestsResult(op=bw.canonical(op) if op else None, nranks=nranks)
    layout: _Layout | None = None
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
            elif m := _VERSION.match(s):
                res.header["nccl_tests_version"] = m[1]
            elif m := _COLUMNS.match(s):
                layout = _layout_from_columns(m[1])
            elif "Avg bus bandwidth" in s:
                res.avg_busbw = float(s.split(":")[-1].split()[0])
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
        row = _parse_row(tok, layout or _guess_layout(tok[3:]))
        if row is None:
            res.skipped.append(s)
        else:
            res.rows.append(row)
    if res.nranks is None:
        if res.devices:
            res.nranks = len(res.devices)
        elif "nGpus" in res.header:
            res.nranks = res.header["nGpus"] * res.header.get("nThread", 1)  # single process only
    return res


def _parse_row(tok: list[str], lay: _Layout | None) -> NcclRow | None:
    if lay is None:
        return None
    i = 3
    redop = tok[i] if lay.redop else None
    i += lay.redop
    try:
        root = int(tok[i]) if lay.root else None
        i += lay.root
        nums = tok[i:]
        width = 8 if lay.per_iter else 4  # one block: time algbw busbw #wrong [i_min i_max i_p99 i_cv%]
        if len(nums) < 4 or not all(_is_number(t) for t in nums[:4]):
            return None
        row = NcclRow(int(tok[0]), int(tok[1]), tok[2], redop, root,
                      float(nums[0]), float(nums[1]), float(nums[2]), _num(nums[3]))
        ip = nums[width:width + 4]
        if len(ip) == 4 and all(_is_number(t) for t in ip):  # the in-place block; anything after is skipped
            row.ip_time_us, row.ip_algbw, row.ip_busbw, row.ip_wrong = (
                float(ip[0]), float(ip[1]), float(ip[2]), _num(ip[3]))
    except ValueError:
        return None
    return row


def split_runs(text: str) -> list[str]:
    """One log often holds several runs (all_reduce_perf, then all_gather_perf, ...). Split at the run
    markers — "# nccl-tests version" (the first line current nccl-tests prints), "# Collective test
    starting" (also echoed by older job scripts) or the "# nThread" header — using whichever kind
    occurs most often. Chunks without result rows (e.g. an echoed marker) are dropped by parse_many."""
    lines = text.splitlines()
    marks: list[int] = []
    for key in ("# nccl-tests version", "# Collective test starting", "# nThread"):
        found = [i for i, line in enumerate(lines) if line.lstrip().startswith(key)]
        if len(found) > len(marks):
            marks = found
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
           "avg_busbw_reported": res.avg_busbw, "skipped_rows": len(res.skipped),
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
    else:
        print("  recheck skipped: the log does not name the collective or the rank count (pass --op / --nranks)")
    if res.skipped:
        print(f"  WARNING: {len(res.skipped)} numeric row(s) skipped, unrecognised columns; first: {res.skipped[0][:80]!r}")


if __name__ == "__main__":
    raise SystemExit(main())
