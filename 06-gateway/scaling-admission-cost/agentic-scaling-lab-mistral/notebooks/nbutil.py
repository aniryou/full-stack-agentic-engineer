"""Helpers for the practice notebooks."""

from __future__ import annotations

import inspect

import matplotlib.pyplot as plt
import numpy as np


class NotAttempted(Exception):
    pass


def todo(*_a, **_k):
    """Placeholder body for an exercise: replace the call with your implementation."""
    raise NotAttempted()


def check(name, fn):
    """Print PASS / FAIL / not attempted instead of raising, so a notebook runs end to end."""
    try:
        fn()
        print(f"PASS  {name}")
    except NotAttempted:
        print(f"----  {name}: not attempted yet")
    except AssertionError as e:
        print(f"FAIL  {name}: {e}")


async def acheck(name, coro_fn):
    """Same as check() for an async check function: `await acheck("name", _d)`."""
    try:
        await coro_fn()
        print(f"PASS  {name}")
    except NotAttempted:
        print(f"----  {name}: not attempted yet")
    except AssertionError as e:
        print(f"FAIL  {name}: {e}")


def cdf(ax, values, label):
    v = np.sort(np.asarray([x for x in values if x is not None and not np.isnan(x)]))
    if v.size:
        ax.plot(v, np.linspace(0, 1, v.size), label=f"{label} (p95 {np.percentile(v, 95):.1f}s)")


def latency_cdfs(results, title="Turn latency"):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, r in results.items():
        done = r.samples[r.samples.outcome == "completed"]
        cdf(ax, done.latency_s, name)
    ax.set_xlabel("seconds"), ax.set_ylabel("share of completed turns"), ax.set_title(title), ax.grid(alpha=.3), ax.legend(fontsize=8)
    plt.tight_layout()


def timeline(result, title):
    """In-flight turns, the backend's load signal (pool utilisation on the API; batch per replica on a fleet), degrade level."""
    tl = result.timeline
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(tl.t, tl.inflight), axes[0].set_ylabel("in flight")
    if result.setup.server is not None:
        axes[1].plot(tl.t, tl.batch), axes[1].set_ylabel("batch / replica")
        axes[1].axhline(result.setup.server.target_batch, ls="--", c="grey")
    else:
        axes[1].plot(tl.t, tl.pool_utilisation), axes[1].set_ylabel("pool util."), axes[1].axhline(1, ls="--", c="grey")
    axes[2].step(tl.t, tl.level, where="post"), axes[2].set_ylabel("degrade level"), axes[2].set_xlabel("virtual seconds")
    axes[0].set_title(title)
    for ax in axes:
        ax.grid(alpha=.3)
    plt.tight_layout()
