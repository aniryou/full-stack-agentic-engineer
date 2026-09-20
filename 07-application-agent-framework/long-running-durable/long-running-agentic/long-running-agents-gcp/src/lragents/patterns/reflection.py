"""Pattern 6 — Reflection (evaluator–optimizer) loop.

generate → critique (score + feedback) → revise → ... until the critic's score
clears a threshold or the iteration cap is hit. Two design points that matter
for long-running use:

* **Stop conditions live in code.** ``threshold`` and ``max_iters`` are checked
  by the runner; the prompt merely asks for a score. A model asked to "keep
  improving until it's perfect" never stops.
* **Every iteration is a checkpoint.** Drafts and critiques are stored in the
  run so a crash resumes at iteration k, not at zero, and so the whole history
  is available for evaluation/replay later.

Use separate model configurations for generator and critic (or at least
separate prompts with ``include_contents='none'`` in ADK terms): a critic that
sees the generator's reasoning tends to rationalise it.
"""

from __future__ import annotations

import json
import time
from typing import Callable

from ..core.budget import BudgetExceeded, charge, check_budget
from ..core.llm import LLM, PriceCard
from ..core.models import Budget, Run, RunStatus, StepKind, StepRecord, StepStatus, new_run_id
from ..core.store import RunStore
from ..core.transport import Dispatcher, Envelope

GEN_PROMPT = "Produce the best possible draft for the task. If feedback is provided, revise the previous draft to address it."
CRITIC_PROMPT = 'Score the draft from 0 to 10 for the task and give concrete feedback. Reply ONLY with JSON: {"score": <int>, "feedback": "<text>"}'


class ReflectionLoop:
    def __init__(self, *, store: RunStore, generator: LLM, critic: LLM, dispatcher: Dispatcher, threshold: int = 8,
                 max_iters: int = 3, price: PriceCard | None = None, worker_id: str = "worker-1",
                 lease_ttl_s: float = 60.0, clock: Callable[[], float] = time.time) -> None:
        self.store, self.generator, self.critic, self.dispatcher = store, generator, critic, dispatcher
        self.threshold, self.max_iters = threshold, max_iters
        self.price = price or PriceCard()
        self.worker_id, self.lease_ttl_s, self.clock = worker_id, lease_ttl_s, clock

    def start(self, task: str, budget: Budget | None = None, run_id: str | None = None) -> Run:
        run = Run(run_id=run_id or new_run_id("refl"), goal=task, status=RunStatus.RUNNING, budget=budget or Budget(),
                  state={"iter": 0, "draft": None, "feedback": None, "scores": []})
        run = self.store.create(run)
        self.dispatcher.enqueue(Envelope(run.run_id, 0, "step"))
        return run

    def handle(self, env: Envelope) -> Run:
        return self.step(env.run_id)

    def step(self, run_id: str) -> Run:
        run = self.store.acquire_lease(run_id, self.worker_id, self.lease_ttl_s)
        try:
            if run.status.terminal:
                return run
            try:
                check_budget(run, self.clock())
            except BudgetExceeded as e:
                run.status, run.error = RunStatus.FAILED, f"budget: {e}"
                return self.store.save(run)
            st = run.state
            if st["draft"] is None or st["feedback"] is not None:          # generate or revise
                msgs = [{"role": "user", "content": f"TASK: {run.goal}"}]
                if st["draft"] is not None:
                    msgs.append({"role": "user", "content": f"PREVIOUS DRAFT:\n{st['draft']}\nFEEDBACK:\n{st['feedback']}"})
                d = self.generator.decide(GEN_PROMPT, msgs, [])
                self._record(run, "generate" if st["draft"] is None else "revise", d.text, d)
                st["draft"], st["feedback"] = d.text, None
            else:                                                          # critique
                d = self.critic.decide(CRITIC_PROMPT, [{"role": "user", "content": f"TASK: {run.goal}\nDRAFT:\n{st['draft']}"}], [])
                try:
                    verdict = json.loads(d.text or "{}")
                    score = int(verdict.get("score", 0))
                except (ValueError, TypeError):
                    verdict, score = {"score": 0, "feedback": "critic returned invalid JSON"}, 0
                self._record(run, "critique", verdict, d)
                st["scores"].append(score)
                st["iter"] += 1
                if score >= self.threshold or st["iter"] >= self.max_iters:    # deterministic stop
                    run.result = {"draft": st["draft"], "score": score, "iterations": st["iter"]}
                    run.status = RunStatus.SUCCEEDED
                    return self.store.save(run)
                st["feedback"] = verdict.get("feedback", "")
            run = self.store.save(run)
            self.dispatcher.enqueue(Envelope(run.run_id, run.next_index(), "step"))
            return run
        finally:
            self.store.release_lease(run_id, self.worker_id)

    def _record(self, run: Run, name: str, output, d) -> None:
        cost = charge(run, d.tokens_in, d.tokens_out, self.price)
        rec = StepRecord(index=run.next_index(), kind=StepKind.LLM, status=StepStatus.DONE, name=name,
                         tokens_in=d.tokens_in, tokens_out=d.tokens_out, cost_usd=cost)
        rec.finish(output=output)
        run.append(rec)
