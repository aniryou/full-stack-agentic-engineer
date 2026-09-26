#!/usr/bin/env python3
"""Compile every gpurt Numba kernel to PTX for real GPUs — no GPU needed, but numba-cuda (and its NVVM).

The tests run the kernels in Numba's CUDA simulator, which never compiles them: a kernel that uses
something the real CUDA target rejects would still pass. This check closes that gap by asking
numba-cuda's compiler for PTX at compute capability 7.5 (T4) and 8.9 (L4) and by refusing float64
arithmetic in kernels meant to be float32 (a Python float literal is float64 to Numba).

    python3 tools/check_ptx.py             # exit 0: all compiled; 1: failures; 2: skipped (no numba-cuda/NVVM)
    python3 tools/check_ptx.py --cc 9.0    # other targets

``tests/test_ptx.py`` runs it when numba-cuda is installed (``pip install -e ".[gpu]"``) and skips otherwise.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

os.environ["NUMBA_ENABLE_CUDASIM"] = "0"  # the real CUDA target: must be set before numba.cuda is imported
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKIP = 2


def cases():
    from numba import float32

    from gpurt.kernels import bench, elementwise as ew, matmul as mm, reduction as red, softmax as sm, transpose as tr

    v1, v2 = float32[::1], float32[:, ::1]
    return [
        ("vec_add", ew.vec_add, (v1, v1, v1)),
        ("saxpy", ew.saxpy, (float32, v1, v1, v1)),
        ("copy", ew.copy, (v1, v1)),
        *[(f"block_sum[{t},atomic={a}]", red.make_block_sum(t, a), (v1, v1)) for t in (32, 256) for a in (False, True)],
        ("matmul_naive", mm.matmul_naive, (v2, v2, v2)),
        ("matmul_tiled", mm.matmul_tiled, (v2, v2, v2)),  # the module's default instance (tile 16)
        *[(f"matmul_tiled[{t}]", mm.make_matmul_tiled(t), (v2, v2, v2)) for t in (8, 32)],
        ("copy2d", tr.copy2d, (v2, v2)),
        ("transpose_naive", tr.transpose_naive, (v2, v2)),
        ("transpose_tiled", tr.transpose_tiled, (v2, v2)),  # the default instance (pad 1)
        ("transpose_tiled[pad=0]", tr.make_transpose_tiled(0), (v2, v2)),
        *[(f"softmax_fused[{t}]", sm.make_softmax_fused(t), (v2, v2)) for t in (32, 256)],
        *[(f"{n}[{t}]", k, (v2, v1)) for t in (32, 256) for n, k in zip(("row_max", "row_sum"), sm.make_row_reductions(t))],
        ("sub_exp", sm.sub_exp, (v2, v1, v2)),
        ("divide_rows", sm.divide_rows, (v2, v1, v2)),
        ("_empty", bench._empty, ()),
    ]


def module_level_kernels() -> set[str]:
    """Names of CUDA kernels defined at module level in gpurt.kernels — each must have a case above."""
    import gpurt.kernels as K

    names = set()
    for mod in ("elementwise", "matmul", "reduction", "softmax", "transpose", "bench"):
        m = __import__(f"gpurt.kernels.{mod}", fromlist=["_"])
        names |= {n for n, v in vars(m).items() if type(v).__name__.endswith("CUDADispatcher")}
    assert not K.SIMULATOR
    return names


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cc", nargs="+", default=["7.5", "8.9"], help="compute capabilities, e.g. 7.5 8.9 9.0")
    a = p.parse_args(argv)
    if importlib.util.find_spec("numba_cuda") is None:
        print("SKIP: numba-cuda is not installed (pip install -e '.[gpu]'); nothing compiled")
        return SKIP
    from numba import cuda
    from numba.cuda.cudadrv import nvvm

    if not nvvm.is_available():
        print("SKIP: numba-cuda is installed but NVVM is not (it ships with the cuda-nvcc wheels of numba-cuda[cu12])")
        return SKIP
    all_cases = cases()
    missing = module_level_kernels() - {name for name, _, _ in all_cases}  # a new kernel needs a case here
    bad = 0
    if missing:
        bad += 1
        print(f"FAIL: kernels without a PTX case in tools/check_ptx.py: {sorted(missing)}")
    for cc in a.cc:
        major, minor = (int(x) for x in cc.split("."))
        for name, disp, sig in all_cases:
            try:
                ptx, _ = cuda.compile_ptx(disp.py_func, sig, cc=(major, minor))
            except Exception as e:  # noqa: BLE001  (report every failure, then fail)
                bad += 1
                print(f"FAIL sm_{major}{minor} {name}: {type(e).__name__}: {str(e)[:300]}")
                continue
            target = re.search(r"^\.target\s+(\S+)", ptx, re.M)
            f64 = sorted(set(re.findall(r"\b(?:add|sub|mul|fma|div|ex2|lg2|sqrt|rcp)\.\S*f64\b", ptx)))
            if f64:
                bad += 1
                print(f"FAIL sm_{major}{minor} {name}: float64 arithmetic in a float32 kernel: {f64}")
            else:
                print(f"ok   {target[1] if target else '?':7} {name}")
    print(f"{len(all_cases)} kernels x {len(a.cc)} targets, failures: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
