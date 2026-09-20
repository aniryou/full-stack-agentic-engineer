"""Solutions — Practice 01 (durable loop)."""

from lragents.core import Run, RunStatus, StepKind, StepRecord, StepStatus, ToolContext, new_run_id


def run_idempotent(store, key, fn):
    if key in store:
        return store.get(key), True
    result = fn()
    store.put(key, result)
    return result, False


class MiniDurableLoop:
    """A ~40-line durable loop: one step per call, write-ahead intent, memoised tools."""

    def __init__(self, store, llm, tools, idem, faults):
        self.store, self.llm, self.tools, self.idem, self.faults = store, llm, tools, idem, faults

    def start(self, goal):
        return self.store.create(Run(run_id=new_run_id("mini"), goal=goal, status=RunStatus.RUNNING))

    def step(self, run_id):
        run = self.store.get(run_id)
        if run.status.terminal:
            return run
        pending = run.pending_step()
        if pending is None:                                   # ask the model, journal the decision
            d = self.llm.decide("", [{"role": "user", "content": run.goal}], self.tools.specs())
            rec = StepRecord(index=run.next_index(), kind=StepKind.LLM, status=StepStatus.DONE, name="decide")
            rec.finish(output={"kind": d.kind, "tool": d.tool_name, "args": d.tool_args, "text": d.text})
            run.append(rec)
            if d.kind == "final":
                run.result, run.status = d.text, RunStatus.SUCCEEDED
                return self.store.save(run)
            idx = run.next_index()
            pending = StepRecord(index=idx, kind=StepKind.TOOL, status=StepStatus.STARTED, name=d.tool_name,
                                 input=d.tool_args, idempotency_key=f"{run.run_id}:{idx}")
            run.append(pending)
            run = self.store.save(run)                        # checkpoint #1: intent before side effect
            pending = run.journal[-1]
        tool = self.tools.get(pending.name)
        ctx = ToolContext(run.run_id, pending.index, pending.idempotency_key, run.state)
        out, replayed = run_idempotent(self.idem, pending.idempotency_key, lambda: tool.fn(pending.input, ctx))
        self.faults.maybe_crash("after_side_effect")
        pending.finish(output={"result": out, "replayed": replayed})
        return self.store.save(run)                           # checkpoint #2: result
