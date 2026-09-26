"""Reliability at scale: failures are a rate, and rates add.

The one idea: if each GPU fails independently at rate 1/M, a job spanning N GPUs
fails at rate N/M. A GPU that runs for years between failures still gives a
16K-GPU synchronous job an interruption every few hours. Training answers with
checkpoints spaced by the Young/Daly interval sqrt(2 x checkpoint_time x MTBF);
inference answers with independent replicas, and the question becomes how many
spares keep enough of them up.

Model: independent, exponentially distributed failures (constant rate). Real
fleets have infant mortality and correlated failures (a switch, a PDU), so treat
results as first-order.
"""
from __future__ import annotations

import math

HOURS_PER_YEAR = 8760


def component_mtbf_from_observation(interruptions: int, hours: float, n_components: int) -> float:
    """Back out a per-component MTBF (hours) from a fleet's observed failure count."""
    return n_components * hours / interruptions


def cluster_mtbf(component_mtbf_h: float, n: int) -> float:
    """Rates add: N components with MTBF M fail, as a group, every M / N hours."""
    return component_mtbf_h / n


def failures_per_day(n: int, component_mtbf_h: float) -> float:
    return 24 * n / component_mtbf_h


def p_survive(hours: float, n: int, component_mtbf_h: float) -> float:
    """Probability that none of n components fails within `hours`: exp(-n t / M)."""
    return math.exp(-n * hours / component_mtbf_h)


def young_daly_interval(checkpoint_s: float, mtbf_s: float, higher_order: bool = False) -> float:
    """Optimal compute time between checkpoints.

    Young (1974): sqrt(2 delta M). Daly (2006) adds correction terms, valid for delta < 2M:
    sqrt(2 delta M) [1 + (1/3) sqrt(delta / 2M) + (1/9)(delta / 2M)] - delta.
    """
    base = math.sqrt(2 * checkpoint_s * mtbf_s)
    if not higher_order:
        return base
    if checkpoint_s >= 2 * mtbf_s:
        return mtbf_s
    r = checkpoint_s / (2 * mtbf_s)
    return base * (1 + math.sqrt(r) / 3 + r / 9) - checkpoint_s


def wasted_fraction(interval_s: float, checkpoint_s: float, mtbf_s: float,
                    restart_s: float = 0.0) -> float:
    """First-order share of wall-clock lost: checkpoint overhead delta / tau plus, per
    failure, half an interval of lost work and the restart: (tau / 2 + R) / M.
    At tau = sqrt(2 delta M) the first two terms are equal and sum to sqrt(2 delta / M)."""
    return checkpoint_s / interval_s + (interval_s / 2 + restart_s) / mtbf_s


# -- inference: replicas are failure domains ----------------------------------------------
def availability(mtbf_h: float, mttr_h: float) -> float:
    """Steady-state fraction of time up: MTBF / (MTBF + MTTR)."""
    return mtbf_h / (mtbf_h + mttr_h)


def replica_availability(gpu_mtbf_h: float, gpus_per_replica: int, mttr_h: float) -> float:
    """A TP=g replica is down when any of its g GPUs is: its MTBF is M / g."""
    return availability(cluster_mtbf(gpu_mtbf_h, gpus_per_replica), mttr_h)


def p_at_least(k: int, n: int, a: float) -> float:
    """Probability that at least k of n independent replicas (each up with prob a) are up."""
    return sum(math.comb(n, i) * a ** i * (1 - a) ** (n - i) for i in range(k, n + 1))


def replicas_for(k: int, a: float, target: float) -> int:
    """Smallest n >= k with P(at least k up) >= target -- 'N + how many spares?'."""
    n = k
    while p_at_least(k, n, a) < target:
        n += 1
    return n
