"""Reusable long-running-agent patterns built on the engine.

* :mod:`hitl` — approval gates (suspend / resume / timeout)
* :mod:`reflection` — bounded evaluator–optimizer loop with per-iteration checkpoints
* :mod:`orchestrator_worker` — planner fan-out into durable child runs, fan-in aggregation
* :mod:`saga` — compensating side effects for multi-system actions
"""
