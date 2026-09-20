"""Solutions — Practice 02 (fan-out / fan-in)."""

from lragents.core import run_idempotent, transact
from lragents.core.transport import Envelope
from lragents.patterns import FanOutFanIn


def make_mutate(subtask_id, result):
    def mutate(run):
        fan = run.state["fan"]
        if subtask_id not in fan["results"]:                  # duplicate delivery guard
            fan["results"][subtask_id] = result
            fan["completed"] += 1
        return fan["completed"] == fan["expected"]            # late duplicates also see all_done
    return mutate


class MyFan(FanOutFanIn):
    def handle_subtask(self, run_id, subtask_id, task):
        key = f"{run_id}:sub:{subtask_id}"
        result, _ = run_idempotent(self.idem, key, lambda: self.worker(task))   # side effect OUTSIDE the txn
        run, all_done = transact(self.store, run_id, make_mutate(subtask_id, result))
        if all_done:
            self.dispatcher.enqueue(Envelope(run_id, 1, "aggregate"))             # named → de-duplicated
        return run
