"""Pins for trained and sampled numbers: the reference run's value, and how far another CPU may land from it.

A seeded run of a tiny numpy model is one CPU's run. numpy's bundled OpenBLAS picks a GEMM kernel per
microarchitecture (Katmai, Nehalem, Sandybridge, Haswell — AMD Zen uses it too — SkylakeX, …), its exp, tanh and
reductions dispatch per SIMD level, and the kernels round differently in the last bit. Hard-label training and
the reasoning toy's table policy contract that difference away and land on the same model everywhere; soft-target
training, the drafts and on-policy GKD do not — hundreds of Adam steps at lr 0.02 grow it into a slightly
different, equally valid model, and a sample drawn from that model is a different sample. So a trained or sampled
number is pinned to the reference run's value within a tolerance, and `near()` hands the pinned value back, so
that a primer row is formatted from the pin (a recomputation on a rounding boundary cannot flip a digit) and must
still appear verbatim in PRIMER.md. Closed-form numbers stay exact.

Every tolerance is at least twice the largest deviation measured on 2026-09-27 over 36 runs of
`tools/host_sensitivity.py` — five OpenBLAS kernels × four numpy SIMD levels on one machine, plus sixteen runs with
the last bit of every np.exp / np.tanh result flipped at random — and never below 0.005 on a rate (half a unit in
the third decimal). The largest spreads: the on-policy students of §4 and §8 (rule accuracy ±0.05 at β = 0, ±0.07
at β = 1; the §8 rare-slice counts ±11 and ±19 of 101), the 4- and 8-unit drafts of §7 (α ±0.05 and ±0.14) and
the small KL of a near-converged student, which the last few wrong contexts dominate (within a factor of 2.8; pinned
within a factor of 4). Rerun the tool after changing a model, a seed or a training loop, and re-pin from its table.
"""
from __future__ import annotations


def near(x, pinned, tol=None, *, factor=None):
    """Assert that the recomputed `x` is within `tol` of `pinned` — or, for a KL, within a `factor` of it — and
    return `pinned`, so the caller formats the primer's fragment from the pinned value."""
    x = float(x)
    if factor is not None:
        ok, how = pinned / factor <= x <= pinned * factor, f"within a factor of {factor} of"
    else:
        ok, how = abs(x - pinned) <= tol, f"within ±{tol} of"
    assert ok, (f"recomputed {x:.4g} is not {how} the pinned {pinned}: this host moved the run further than the "
                f"kernels measured in tools/host_sensitivity.py; rerun it and re-pin (tests/pins.py)")
    return pinned
