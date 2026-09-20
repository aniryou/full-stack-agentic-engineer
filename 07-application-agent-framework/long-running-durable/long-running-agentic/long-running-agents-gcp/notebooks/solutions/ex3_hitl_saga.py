"""Solutions — Practice 03 (human-in-the-loop + saga)."""

import hmac

from lragents.core import RunStatus, StepKind, StepRecord, StepStatus


def approve(loop, run_id, token, approved, approver, comment=""):
    run = loop.store.acquire_lease(run_id, loop.worker_id, loop.lease_ttl_s)
    try:
        if run.status != RunStatus.WAITING_HUMAN:
            return run                                        # already decided → no-op
        w = run.waiting_on
        if not hmac.compare_digest(str(w["token"]), token):
            raise PermissionError("bad token")
        human = StepRecord(index=run.next_index(), kind=StepKind.HUMAN, status=StepStatus.DONE, name="approval",
                           input={"tool": w["tool"], "args": w["args"]})
        human.finish(output={"approved": approved, "approver": approver, "comment": comment})
        run.append(human)
        run.waiting_on, run.status = None, RunStatus.RUNNING
        if approved:                                          # journal the approved call; never re-ask the model
            idx = run.next_index()
            run.append(StepRecord(index=idx, kind=StepKind.TOOL, status=StepStatus.STARTED, name=w["tool"],
                                  input=w["args"], idempotency_key=f"{run.run_id}:{idx}"))
        run = loop.store.save(run)
        loop._enqueue_next(run)
        return run
    finally:
        loop.store.release_lease(run_id, loop.worker_id)


def saga_next(order, saga):
    if saga["phase"] == "forward":
        if saga["cursor"] >= len(order):
            return "__done__", "succeeded"
        return order[saga["cursor"]], "action"
    if saga["comp_cursor"] < 0:
        return "__done__", "failed"
    return order[saga["comp_cursor"]], "compensate"
