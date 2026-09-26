# Working in this repository

This file is the brief every contributor (human or agent) follows so the lab stays coherent.

## Layout

    agentlab/            the library: small, readable, tested; docstrings name the notebook that teaches them
      llm/             FakeLLM (offline model), message/response types, optional Gemini adapter
      agents/          tools, budgets, agent loop, workflows, sessions/state, context builder, runner  ← CORE
      mcp/             teaching subset of MCP (2026-07-28 shape) + gateway
      auth/            toy OAuth 2.1 authorization server, PKCE, resource indicators, token exchange
      evals/           golden sets, trajectory metrics, judges, gates
      observability/   tracer (gen_ai.* attributes), metrics, price table
      reliability/     retry/backoff, circuit breaker, bulkhead, fallbacks
      security/        prompt-injection defences, screening, action policy
      estimation/      cost / throughput / latency arithmetic
    notebooks_src/     percent-format sources (see notebooks_src/README.md)
    notebooks/         GENERATED exercise notebooks (do not edit by hand)
    solutions/         GENERATED solution notebooks
    tests/             pytest; `tests/test_notebooks.py` executes all solutions (marked slow)
    tools/             build_notebooks.py, run_notebooks.py

## Rules

1. **Do not modify `agentlab/llm/` or `agentlab/agents/`** (the core) unless you own that change; every notebook
   depends on them. If you need something from the core, note it in your report instead.
2. Dependencies: standard library + `pydantic` + `httpx` only. No network calls, no API keys, no sleeping
   longer than a few hundred milliseconds in tests or notebooks (inject clocks/sleepers where timing matters).
3. Every module gets a `tests/test_<module>.py`; run it with `python3 -m pytest -q tests/test_<module>.py`.
4. Every notebook source follows `notebooks_src/README.md` and the pattern in `notebooks_src/00_setup_and_fake_llm.py`:
   heading + the concept it teaches (see `docs/PRIMER_MAP.md`) + "you will" list → worked examples → 4–7 exercises each with a check cell →
   a *The one-minute version* cell. Build and verify with:

       python3 tools/build_notebooks.py notebooks_src/NN_name.py
       python3 tools/run_notebooks.py solutions/NN_name.ipynb            # must PASS
       python3 tools/run_notebooks.py notebooks/NN_name.ipynb --expect-fail   # must stop at the first exercise

5. Solution blocks are complete statements at statement level (never inside an expression).
6. Prefer teaching the mechanism over API coverage: a learner should be able to explain *why* after each exercise.
7. Tone: precise, warm, no filler. Comments say why, not what.

## Core API cheat-sheet (agentlab.agents / agentlab.llm)

    from agentlab.llm import FakeLLM, KeywordPlanner, Rule, scripted, call, calls, text, count_tokens
    from agentlab.agents import (tool, FunctionTool, SideEffect, ToolContext, ToolResult, ToolPermanentError,
        ToolTransientError, IdempotencyStore, Identity, ToolRegistry,
        LlmAgent, AgentTool, SequentialAgent, ParallelAgent, LoopAgent, InvocationContext, Budget, BudgetExceeded,
        Session, Event, SessionStatus, InMemorySessionStore, JsonFileSessionStore, VersionConflict,
        TaskRecord, TaskStatus, TaskStore, ContextBuilder, context_report, Runner, RunResult, Paused)

    @tool(side_effect=SideEffect.IRREVERSIBLE, required_scope="cards:write")   # schema from the signature
    def block_card(card_id: str, reason: str = "lost") -> dict: ...

    agent = LlmAgent("assistant", llm, "instruction {state_key}", tools=[...], sub_agents=[...], output_key="answer")
    runner = Runner(agent, store=InMemorySessionStore(), tracer=None)
    result = await runner.run("session-1", "block my card", user=Identity("u1", scopes={"cards:write"}))
    result.paused / result.pending / result.text / result.session.events
    await runner.approve("session-1", True)          # resumes a paused irreversible tool call
    async for ev in runner.stream("session-1", "…"): ...   # events as they happen

    session.messages()   # model-facing view derived from the event log
    session.state        # typed working state; keys "app:", "user:", "temp:" or plain
    InvocationContext(session=s, budget=Budget(max_steps=8), user=Identity(...), tracer=t, confirm=async_hook)

    Tracer contract expected by the loop:  with tracer.span(name, kind="model", **attrs) as span: span.set(**more)
