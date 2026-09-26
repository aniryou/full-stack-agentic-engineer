"""Kueue-style quota: ClusterQueues, cohorts, borrowing and lending, and reclaim preemption.

The one idea: the scheduler decides WHERE a pod runs; a quota system decides WHETHER a job may
start at all. Kueue keeps a job suspended until its whole request fits its ClusterQueue's quota
- borrowing idle quota from the cohort if allowed - and only then lets its pods be created.
When the owner of lent quota comes back, borrowers are preempted. Admission is all-or-nothing
per job, which is also what makes Kueue a gang admitter.

Simplifications: one resource group per ClusterQueue (all resources share a flavor), flat
cohorts, classic preemption (not Fair Sharing), no admission checks, and a preemptor admitted in
the same step its victims are evicted (real Kueue marks the victims Evicted and admits the
preemptor in a later cycle, once their quota is released).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

_seq = itertools.count(1)


@dataclass
class Quota:
    nominal: int
    borrowing_limit: int | None = None     # None: may borrow all the cohort lends
    lending_limit: int | None = None       # None: lends all its unused nominal quota


@dataclass
class ClusterQueue:
    name: str
    quotas: dict                            # {flavor: {resource: Quota}}; flavors tried in order
    cohort: str | None = None               # spec.cohortName
    queueing_strategy: str = "BestEffortFIFO"      # or "StrictFIFO"
    within_cluster_queue: str = "Never"            # Never | LowerPriority | LowerOrNewerEqualPriority
    reclaim_within_cohort: str = "Never"           # Never | LowerPriority | Any
    borrow_within_cohort: str = "Never"            # Never | LowerPriority
    max_priority_threshold: int | None = None
    usage: dict = field(default_factory=dict)      # {(flavor, resource): amount}

    def quota(self, flavor: str, res: str) -> Quota | None:
        return self.quotas.get(flavor, {}).get(res)

    def guaranteed(self, flavor: str, res: str) -> int:
        """nominal - lendingLimit: quota never lent to the cohort (Kueue's localQuota)."""
        q = self.quota(flavor, res)
        return max(0, q.nominal - q.lending_limit) if q and q.lending_limit is not None else 0


@dataclass(eq=False)
class Workload:
    name: str
    queue: str                  # a LocalQueue ("namespace/name") or a ClusterQueue name
    requests: dict              # summed over all its pods, e.g. {"nvidia.com/gpu": 16}
    priority: int = 0
    seq: int = field(default_factory=lambda: next(_seq))
    cq: str | None = None
    flavor: str | None = None
    admitted: int | None = None     # admission order while admitted
    evictions: int = 0


class Kueue:
    def __init__(self, cluster_queues, local_queues: dict | None = None):
        self.cqs = {cq.name: cq for cq in cluster_queues}
        self.local_queues = dict(local_queues or {})   # {"team-a/gpus": "team-a-cq"}
        self.pending: list[Workload] = []
        self.running: list[Workload] = []
        self.events: list[tuple] = []
        self._order = itertools.count(1)

    # -- quota arithmetic (pkg/cache/scheduler/resource_node.go, flat cohort) ---------------------
    def members(self, cq: ClusterQueue) -> list:
        return [c for c in self.cqs.values() if cq.cohort and c.cohort == cq.cohort]

    def available(self, cq: ClusterQueue, flavor: str, res: str) -> int:
        """How much more of (flavor, res) this ClusterQueue can use right now."""
        q = cq.quota(flavor, res)
        if q is None:
            return 0
        use = cq.usage.get((flavor, res), 0)
        if not cq.cohort:
            return q.nominal - use
        pool = used = 0
        for c in self.members(cq):                     # the cohort holds what members lend
            if c.quota(flavor, res):
                g = c.guaranteed(flavor, res)
                pool += c.quota(flavor, res).nominal - g
                used += max(0, c.usage.get((flavor, res), 0) - g)
        g = cq.guaranteed(flavor, res)
        from_cohort = pool - used
        if q.borrowing_limit is not None:
            from_cohort = min(from_cohort, (q.nominal - g) - max(0, use - g) + q.borrowing_limit)
        return max(0, g - use) + from_cohort

    def fits(self, wl: Workload, cq: ClusterQueue, flavor: str, borrow: bool = True) -> bool:
        for res, amount in wl.requests.items():
            q = cq.quota(flavor, res)
            if q is None or amount > self.available(cq, flavor, res):
                return False
            if not borrow and cq.usage.get((flavor, res), 0) + amount > q.nominal:
                return False
        return True

    def borrowing(self, cq: ClusterQueue, flavor: str, resources) -> bool:
        return any(cq.usage.get((flavor, r), 0) > cq.quota(flavor, r).nominal
                   for r in resources if cq.quota(flavor, r))

    # -- the queue --------------------------------------------------------------------------------
    def submit(self, *wls: Workload) -> None:
        for wl in wls:
            wl.cq = self.local_queues.get(wl.queue, wl.queue)
            self.pending.append(wl)

    def schedule(self, max_steps: int = 10_000) -> list:
        """Admit (and preempt) until nothing changes; return the events of this call."""
        start = len(self.events)
        for _ in range(max_steps):
            if not self._step():
                break
        return self.events[start:]

    def _heads(self) -> list:
        """Candidates in Kueue's order: no borrowing needed first, then priority, then age.
        StrictFIFO offers only its head; BestEffortFIFO lets later workloads past a blocked one."""
        heads = []
        for cq in self.cqs.values():
            queue = sorted((w for w in self.pending if w.cq == cq.name), key=lambda w: (-w.priority, w.seq))
            heads += queue[:1] if cq.queueing_strategy == "StrictFIFO" else queue

        def needs_borrowing(w):
            cq = self.cqs[w.cq]
            return not any(all(cq.quota(f, r) and cq.usage.get((f, r), 0) + a <= cq.quota(f, r).nominal
                               for r, a in w.requests.items()) for f in cq.quotas)
        return sorted(heads, key=lambda w: (needs_borrowing(w), -w.priority, w.seq))

    def _step(self) -> bool:
        for wl in self._heads():
            cq = self.cqs[wl.cq]
            flavor = next((f for f in cq.quotas if self.fits(wl, cq, f)), None)   # fit, borrowing allowed
            if flavor:
                self._admit(wl, cq, flavor)
                return True
            for flavor in cq.quotas:                                               # else: preempt
                targets = self.preemption_targets(wl, cq, flavor)
                if targets:
                    for v in targets:
                        self._evict(v, wl)
                    self._admit(wl, cq, flavor)
                    return True
        return False

    # -- classic preemption (site/content/en/docs/concepts/preemption.md) ------------------------
    def preemption_targets(self, wl: Workload, cq: ClusterQueue, flavor: str) -> list | None:
        if any(cq.quota(flavor, r) is None for r in wl.requests):
            return None
        within_nominal = all(a <= cq.quota(flavor, r).nominal for r, a in wl.requests.items())
        borrow_preempt = cq.borrow_within_cohort == "LowerPriority"
        if not (within_nominal or borrow_preempt):
            return None
        cands = []
        for v in self.running:
            if v.flavor != flavor or not set(v.requests) & set(wl.requests):
                continue
            if v.cq == cq.name:
                p = cq.within_cluster_queue
                ok = (p in ("LowerPriority", "LowerOrNewerEqualPriority") and v.priority < wl.priority) or \
                     (p == "LowerOrNewerEqualPriority" and v.priority == wl.priority and v.seq > wl.seq)
            else:
                p = cq.reclaim_within_cohort
                ok = self.cqs[v.cq] in self.members(cq) and (p == "Any" or (p == "LowerPriority" and v.priority < wl.priority))
            if ok:
                cands.append(v)
        cands.sort(key=lambda v: (v.cq == cq.name, v.priority, -v.admitted))   # others, low prio, newest
        own = [v for v in cands if v.cq == cq.name]
        if not cands:
            return None
        if len(own) == len(cands):
            return self._greedy(wl, cq, flavor, cands, borrow=True)
        if borrow_preempt:
            t = cq.max_priority_threshold
            eligible = [v for v in cands if v.cq == cq.name or
                        (v.priority < wl.priority and (t is None or v.priority <= t))]
            found = self._greedy(wl, cq, flavor, eligible, borrow=True)
            if found:
                return found
        if within_nominal:
            found = self._greedy(wl, cq, flavor, cands, borrow=False)
            if found:
                return found
        return self._greedy(wl, cq, flavor, own, borrow=True)

    def _greedy(self, wl, cq, flavor, cands, borrow) -> list | None:
        """Remove candidates in order until wl fits, then give back every one it did not need."""
        removed = []
        for v in cands:
            if self.fits(wl, cq, flavor, borrow):
                break
            if v.cq != cq.name and not self.borrowing(self.cqs[v.cq], flavor, wl.requests):
                continue                                # only borrowers can be reclaimed from
            self._charge(v, -1)
            removed.append(v)
        ok = self.fits(wl, cq, flavor, borrow)
        if ok:
            for v in reversed(list(removed)):
                self._charge(v, +1)
                if self.fits(wl, cq, flavor, borrow):
                    removed.remove(v)
                else:
                    self._charge(v, -1)
        for v in removed:
            self._charge(v, +1)                          # undo the dry run; the caller evicts
        return removed if ok and removed else None

    # -- bookkeeping --------------------------------------------------------------------------------
    def _charge(self, wl: Workload, sign: int) -> None:
        cq = self.cqs[wl.cq]
        for res, amount in wl.requests.items():
            cq.usage[(wl.flavor, res)] = cq.usage.get((wl.flavor, res), 0) + sign * amount

    def _admit(self, wl: Workload, cq: ClusterQueue, flavor: str) -> None:
        self.pending.remove(wl)
        wl.flavor, wl.admitted = flavor, next(self._order)
        self._charge(wl, +1)
        self.running.append(wl)
        note = "borrowing" if self.borrowing(cq, flavor, wl.requests) else ""
        self.events.append(("admitted", wl.name, cq.name, flavor, note))

    def _evict(self, wl: Workload, by: Workload) -> None:
        self._charge(wl, -1)
        self.running.remove(wl)
        wl.flavor, wl.admitted, wl.evictions = None, None, wl.evictions + 1
        self.pending.append(wl)                          # requeued; its pods are deleted
        self.events.append(("preempted", wl.name, wl.cq, by.name))

    def finish(self, name: str) -> None:
        wl = next(w for w in self.running if w.name == name)
        self._charge(wl, -1)
        self.running.remove(wl)
        self.events.append(("finished", wl.name, wl.cq, wl.flavor, ""))

    def explain(self, wl: Workload) -> str:
        """Why a pending workload is not admitted, in the words of Kueue's flavor assigner."""
        cq = self.cqs[wl.cq]
        msgs = []
        for flavor in cq.quotas:
            for res, amount in sorted(wl.requests.items()):
                if cq.quota(flavor, res) is None:
                    msgs.append(f"resource {res} unavailable in ClusterQueue")
                elif amount > self.available(cq, flavor, res):
                    msgs.append(f"insufficient unused quota for {res} in flavor {flavor}, "
                                f"{amount - self.available(cq, flavor, res)} more needed")
        return "couldn't assign flavors to pod set main: " + "; ".join(msgs) if msgs else "fits"

    def table(self, res: str = "nvidia.com/gpu") -> str:
        return "\n".join(f"{cq.name:<14} {f:<10} used {cq.usage.get((f, res), 0):>3} / nominal {q.nominal:>3}  "
                         f"borrowed {max(0, cq.usage.get((f, res), 0) - q.nominal):>3}  "
                         f"available {self.available(cq, f, res):>3}"
                         for cq in self.cqs.values() for f in cq.quotas if (q := cq.quota(f, res)))
