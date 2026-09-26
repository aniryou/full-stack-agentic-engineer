import asyncio
import json
import math
import re
from pathlib import Path

import pytest

from agentlab.agents import Budget, LlmAgent, Runner, tool
from agentlab.llm import FakeLLM, KeywordPlanner, Rule, StreamChunk, Usage, call, calls, text
from agentlab.observability import (DEFAULT_PRICES, GEN_AI_CACHED_TOKENS, GEN_AI_FINISH_REASONS, GEN_AI_INPUT_MESSAGES,
                                  GEN_AI_INPUT_TOKENS, GEN_AI_OPERATION_NAME, GEN_AI_OUTPUT_TOKENS,
                                  GEN_AI_REQUEST_MODEL, TOOL_CALL_ID, TOOL_NAME, AlertRule, Price, PriceTable, RedactingExporter,
                                  Tracer, TraceSummary, agent_metrics, cost_per_resolved, evaluate_alerts, now_ms,
                                  percentile, redact, reported_latency_ms, rule, summarize_latencies, tokens_per_task,
                                  ttft_and_tps, wrong_tool_rate)


# ---------------------------------------------------------------- helpers
class FakeClock:
    """Injectable clock so durations are exact."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float) -> None:
        self.t += seconds


@tool
def get_balance(account_id: str) -> dict:
    """Balance for an account."""
    return {"account_id": account_id, "balance": 1234.5, "owner_email": "jane@example.com"}


def bank_agent(llm=None):
    planner = KeywordPlanner([Rule(r"balance", "get_balance", {"account_id": "acc-1"})])
    return LlmAgent("assistant", llm or FakeLLM(policy=planner), "You are a bank assistant.", tools=[get_balance])


def model_span(tracer, clock, model="fake-flash", inp=1000, out=200, cached=0, seconds=0.5):
    with tracer.span("model.generate", kind="model", **{GEN_AI_REQUEST_MODEL: model}) as s:
        clock.tick(seconds)
        s.set(**{GEN_AI_INPUT_TOKENS: inp, GEN_AI_OUTPUT_TOKENS: out, GEN_AI_CACHED_TOKENS: cached})
    return s


# ------------------------------------------------------------------ tracer
def test_spans_nest_and_group_into_implicit_traces():
    tracer = Tracer()
    with tracer.span("agent", kind="agent") as root:
        with tracer.span("model.generate", kind="model") as m:
            assert tracer.current_span() is m
        with tracer.span("tool x", kind="tool") as t:
            pass
    with tracer.span("second turn", kind="agent") as other:
        pass
    assert m.parent_id == root.id and t.parent_id == root.id and root.parent_id is None
    assert m.trace_id == t.trace_id == root.trace_id
    assert other.trace_id != root.trace_id, "a new root span starts a new implicit trace"
    assert [t.name for t in tracer.traces()] == ["agent", "second turn"]
    assert tracer.current_span() is None


def test_start_trace_groups_several_root_level_spans():
    tracer = Tracer()
    with tracer.start_trace("conversation 1", session="s1") as turn:
        with tracer.span("agent", kind="agent"):
            pass
        with tracer.span("agent", kind="agent"):
            pass
    (trace,) = tracer.traces()
    assert trace.root is turn and trace.root.kind == "internal" and trace.root.attrs["session"] == "s1"
    assert len(trace.children(turn)) == 2 and len(trace.spans) == 3


async def test_spans_nest_correctly_inside_gather():
    tracer = Tracer()

    async def work(i):
        with tracer.span(f"tool {i}", kind="tool") as s:
            await asyncio.sleep(0)
            with tracer.span("inner"):
                await asyncio.sleep(0)
            return s

    with tracer.span("agent", kind="agent") as root:
        children = await asyncio.gather(*(work(i) for i in range(3)))
    assert {c.parent_id for c in children} == {root.id}
    inner = [s for s in tracer.spans if s.name == "inner"]
    assert sorted(s.parent_id for s in inner) == sorted(c.id for c in children)
    assert len({s.trace_id for s in tracer.spans}) == 1
    assert tracer.current_span() is None


def test_span_records_exception_and_reraises():
    tracer = Tracer()
    with pytest.raises(RuntimeError):
        with tracer.span("boom", kind="tool"):
            raise RuntimeError("upstream 503")
    (s,) = tracer.spans
    assert s.status == "error" and s.exception == "RuntimeError: upstream 503" and s.finished
    assert tracer.current_span() is None, "the context is restored even when the block raises"


def test_attributes_operation_name_and_kind_validation():
    tracer = Tracer()
    with tracer.span("tool get_balance", kind="tool", **{TOOL_NAME: "get_balance"}) as s:
        s.set(**{"tool.ok": True})
    assert s.attrs[GEN_AI_OPERATION_NAME] == "execute_tool" and s.attrs["tool.ok"] is True
    with tracer.span("model.generate", kind="model", **{GEN_AI_OPERATION_NAME: "generate_content"}) as m:
        pass
    assert m.attrs[GEN_AI_OPERATION_NAME] == "generate_content", "an explicit operation name is kept"
    with pytest.raises(ValueError):
        with tracer.span("x", kind="database"):
            pass


async def test_tracer_attached_to_runner_sees_model_and_tool_spans():
    tracer = Tracer()
    runner = Runner(bank_agent(), tracer=tracer)
    await runner.run("s1", "what is my balance?")
    (trace,) = tracer.traces()
    kinds = [s.kind for s in trace.spans]
    assert kinds == ["agent", "model", "tool", "model"]
    model_spans = trace.by_kind("model")
    assert model_spans[0].attrs["gen_ai.response.finish_reasons"] == ["tool_calls"]
    assert model_spans[1].attrs[GEN_AI_INPUT_TOKENS] > model_spans[0].attrs[GEN_AI_INPUT_TOKENS]
    assert trace.by_kind("tool")[0].attrs["tool.ok"] is True
    tree = tracer.render_tree()
    assert "tool get_balance [tool]" in tree and "model.generate [model]" in tree and "$" in tree


def test_attribute_names_follow_the_otel_genai_conventions():
    # Names as registered in open-telemetry/semantic-conventions-genai, September 2026 (verify).
    assert GEN_AI_CACHED_TOKENS == "gen_ai.usage.cache_read.input_tokens"
    assert GEN_AI_FINISH_REASONS == "gen_ai.response.finish_reasons"
    assert GEN_AI_INPUT_MESSAGES == "gen_ai.input.messages"
    assert (TOOL_NAME, TOOL_CALL_ID) == ("gen_ai.tool.name", "gen_ai.tool.call.id")


async def test_loop_writes_convention_attributes_on_model_and_tool_spans():
    tracer = Tracer()
    await Runner(bank_agent(), tracer=tracer).run("s1", "what is my balance?")
    (trace,) = tracer.traces()
    m = trace.by_kind("model")[0].attrs
    assert isinstance(m[GEN_AI_FINISH_REASONS], list), "finish_reasons is a string array"
    assert GEN_AI_CACHED_TOKENS in m and "gen_ai.usage.cached_tokens" not in m
    assert "gen_ai.response.finish_reason" not in m
    t = trace.by_kind("tool")[0].attrs
    assert t[TOOL_NAME] == "get_balance" and t[TOOL_CALL_ID] and "tool.name" not in t


def test_no_lab_code_or_notebook_writes_the_pre_convention_attribute_names():
    # The capstone once wrote gen_ai.usage.cached_tokens / gen_ai.response.finish_reason; the
    # library no longer reads those names, so such a span would lose its cached-token price.
    root = Path(__file__).resolve().parents[1]
    offenders = [str(p.relative_to(root)) for d in ("agentlab", "notebooks_src") for p in (root / d).rglob("*.py")
                 if re.search(r'"gen_ai\.usage\.cached_tokens"|"gen_ai\.response\.finish_reason"', p.read_text())]
    assert offenders == []


def test_spans_are_read_by_their_convention_names_only():
    tracer = Tracer(clock=FakeClock())
    with tracer.span("model.generate", kind="model", **{GEN_AI_REQUEST_MODEL: "fake-flash"}) as s:
        s.set(**{GEN_AI_INPUT_TOKENS: 1000, GEN_AI_OUTPUT_TOKENS: 10, GEN_AI_CACHED_TOKENS: 400,
                 GEN_AI_FINISH_REASONS: ["stop"]})
    from agentlab.observability import span_usage
    assert span_usage(s).cached_tokens == 400
    assert "(cached 400)" in tracer.render_tree() and "stop" in tracer.render_tree()


def test_render_tree_and_json_export_with_exact_durations():
    clock = FakeClock()
    tracer = Tracer(clock=clock)
    with tracer.span("agent assistant", kind="agent"):
        model_span(tracer, clock, inp=1000, out=200, seconds=0.5)
        with tracer.span("tool get_balance", kind="tool", **{TOOL_NAME: "get_balance"}) as t:
            clock.tick(0.02)
            t.set(**{"tool.ok": False, "tool.error": "timeout"})
    lines = tracer.render_tree().splitlines()
    assert lines[0].startswith("trace ") and "1200 tokens" in lines[0] and "520.0 ms" in lines[0]
    assert lines[1] == "└─ agent assistant [agent] 520.0 ms"
    assert "├─ model.generate [model] 500.0 ms · in 1000 (cached 0) · out 200 · $0.001100" in lines[2]
    assert "└─ tool get_balance [tool] 20.0 ms · error=timeout" in lines[3]
    exported = json.loads(tracer.to_json())
    assert [d["name"] for d in exported] == ["agent assistant", "model.generate", "tool get_balance"]
    assert exported[1]["parent_id"] == exported[0]["id"] and exported[2]["duration_ms"] == pytest.approx(20.0)


# --------------------------------------------------------------- redaction
def test_redact_scrubs_emails_cards_and_bearer_tokens_recursively():
    s = "mail jane.doe@example.com card 4111 1111 1111 1111 auth Bearer eyJhbGci.payload-sig"
    assert redact(s) == "mail <email> card <card> auth Bearer <token>"
    assert redact("4111111111111111 and 4111-1111-1111-1111") == "<card> and <card>"
    assert redact("order 12345 at 10:30") == "order 12345 at 10:30", "short numbers are not cards"
    nested = redact({"user": ["a@b.co", 7, None], "meta": {"note": "Bearer x", "ok": True}})
    assert nested == {"user": ["<email>", 7, None], "meta": {"note": "Bearer <token>", "ok": True}}
    custom = rule("account", r"ACC-\d+", "<account>")
    assert redact("ACC-42 a@b.co", rules=[custom]) == "<account> a@b.co", "only the given rules apply"


def test_redacting_exporter_scrubs_copies_and_drops_attributes():
    tracer = Tracer()
    with tracer.span("tool create_case", kind="tool", **{"case.details": "refund to jane@example.com", GEN_AI_INPUT_MESSAGES: "secret"}):
        pass
    seen = []
    exporter = RedactingExporter(drop_attrs=(GEN_AI_INPUT_MESSAGES,), sink=seen.append)
    out = exporter.export(tracer)
    assert out[0]["attrs"]["case.details"] == "refund to <email>" and GEN_AI_INPUT_MESSAGES not in out[0]["attrs"]
    assert tracer.spans[0].attrs["case.details"] == "refund to jane@example.com", "the tracer keeps the original"
    assert seen == out == exporter.exported
    assert exporter.scrub({"events": [{"content": "card 5500000000000004"}]}) == {"events": [{"content": "card <card>"}]}


# -------------------------------------------------------------------- cost
def test_price_table_cost_is_exact_and_splits_cached_input():
    usage = Usage(input_tokens=1000, output_tokens=200, cached_tokens=400)
    # pro: 600 × $2.00/M + 400 × $0.20/M + 200 × $12.00/M
    assert math.isclose(DEFAULT_PRICES.cost(usage, "gemini-3.1-pro"), 0.0012 + 0.00008 + 0.0024, rel_tol=1e-12)
    # flash: 600 × $0.50/M + 400 × $0.05/M + 200 × $3.00/M
    assert math.isclose(DEFAULT_PRICES.cost(usage, "gemini-3-flash"), 0.0003 + 0.00002 + 0.0006, rel_tol=1e-12)
    assert DEFAULT_PRICES.cost(usage, "fake-flash") == DEFAULT_PRICES.cost(usage, "gemini-3-flash")
    assert DEFAULT_PRICES.cost(usage, "fake-pro") == DEFAULT_PRICES.cost(usage, "gemini-3.1-pro")
    uncached = Usage(input_tokens=1000, output_tokens=200)
    assert DEFAULT_PRICES.cost(uncached, "gemini-3-flash") > DEFAULT_PRICES.cost(usage, "gemini-3-flash")
    assert DEFAULT_PRICES.breakdown(usage, "gemini-3.5-flash-lite") == pytest.approx({"input": 0.00018, "cached_input": 0.000012, "output": 0.0005})
    assert "illustrative" in DEFAULT_PRICES.note
    with pytest.raises(KeyError):
        DEFAULT_PRICES.cost(usage, "gemini-unknown")
    custom = PriceTable({"m": Price(1.0, 0.1, 10.0)})
    assert custom.cost(Usage(input_tokens=1_000_000, output_tokens=0), "m") == 1.0 and "m" in custom and "x" not in custom


def test_trace_summary_tokens_cost_latency_and_top_costs():
    clock = FakeClock()
    tracer = Tracer(clock=clock)
    with tracer.start_trace("turn", session="s1", resolved=True):
        with tracer.span("agent", kind="agent"):
            model_span(tracer, clock, inp=1000, out=100, seconds=0.4)          # 0.0005 + 0.0003 = 0.0008
            with tracer.span("tool get_balance", kind="tool", **{"tool.ok": True}):
                clock.tick(0.05)
            model_span(tracer, clock, inp=2000, out=300, cached=1000, seconds=0.6)  # 0.0005 + 0.00005 + 0.0009 = 0.00145
    (summary,) = TraceSummary.from_tracer(tracer)
    assert (summary.steps, summary.tool_calls, summary.errors) == (2, 1, 0)
    assert (summary.input_tokens, summary.cached_tokens, summary.output_tokens) == (3000, 1000, 400)
    assert math.isclose(summary.cost_usd, 0.0008 + 0.00145, rel_tol=1e-12)
    assert summary.wall_ms == pytest.approx(1050.0) and summary.attrs == {"session": "s1", "resolved": True}
    assert summary.longest_span == ("agent", pytest.approx(1050.0))
    assert summary.top_costs(1) == [("model.generate #2", pytest.approx(0.00145))]
    assert summary.latency_by_kind["model"]["p95"] == pytest.approx(600.0)
    assert tokens_per_task([summary]) == pytest.approx({"count": 1, "mean": 3400, "p50": 3400, "p95": 3400, "p99": 3400, "max": 3400})


# ------------------------------------------------------------------ metrics
def test_percentile_is_nearest_rank():
    assert percentile([15, 20, 35, 40, 50], 30) == 20      # ceil(0.3 × 5) = 2nd value
    assert percentile([15, 20, 35, 40, 50], 40) == 20
    assert percentile([15, 20, 35, 40, 50], 50) == 35
    assert percentile([15, 20, 35, 40, 50], 100) == 50
    assert percentile(range(1, 11), 95) == 10 and percentile(range(1, 11), 0) == 1
    assert percentile([3, 1, 2], 50) == 2, "input need not be sorted"
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_summarize_latencies_by_kind_with_reported_latency():
    clock = FakeClock()
    tracer = Tracer(clock=clock)
    for seconds, reported in ((0.001, 420.0), (0.002, 900.0)):
        with tracer.span("model.generate", kind="model") as s:
            clock.tick(seconds)
            s.set(**{"model.latency_ms": reported})
    with tracer.span("tool x", kind="tool"):
        clock.tick(0.05)
    wall = summarize_latencies(tracer.spans)
    assert wall["model"]["p95"] == pytest.approx(2.0) and wall["tool"]["max"] == pytest.approx(50.0) and wall["model"]["count"] == 2
    reported = summarize_latencies(tracer.spans, latency=reported_latency_ms)
    assert reported["model"]["p50"] == 420.0 and reported["model"]["max"] == 900.0
    assert reported["tool"]["max"] == pytest.approx(50.0), "tools without a reported latency fall back to wall time"


def test_ttft_and_tps_from_synthetic_chunks():
    chunks = [
        StreamChunk(text="Hello ", t_ms=1400.0),
        StreamChunk(text="agentic ", t_ms=1500.0),
        StreamChunk(text="world", t_ms=1600.0),
        StreamChunk(done=True, usage=Usage(input_tokens=50, output_tokens=30), t_ms=1600.0),
    ]
    stats = ttft_and_tps(chunks, start_ms=1000.0)
    assert stats.ttft_ms == 400.0
    assert stats.tokens_per_sec == pytest.approx(150.0)        # 30 tokens over the 200 ms generation window
    assert stats.output_tokens == 30 and stats.generation_ms == 200.0
    single = ttft_and_tps([StreamChunk(text="ok", t_ms=5.0), StreamChunk(done=True, usage=Usage(output_tokens=1), t_ms=5.0)], start_ms=0.0)
    assert single.tokens_per_sec == 0.0, "no generation window → no throughput claim"
    with pytest.raises(ValueError):
        ttft_and_tps([StreamChunk(done=True)], start_ms=0.0)


async def test_ttft_and_tps_from_fake_llm_stream():
    llm = FakeLLM(responses=["one two three four five six seven eight"], simulate_time=True, time_scale=0.01, ttft_ms=400)
    start = now_ms()
    chunks = [c async for c in llm.stream([{"role": "user", "content": "count"}])]
    stats = ttft_and_tps(chunks, start)
    assert 3.5 <= stats.ttft_ms < 200, f"400 simulated ms at 1% scale should be ≈ 4 real ms, got {stats.ttft_ms}"
    assert stats.tokens_per_sec > 0 and stats.output_tokens == chunks[-1].usage.output_tokens


def test_alert_rules_fire_on_threshold_breaches():
    rules = [
        AlertRule("cost per task", "cost_per_task", threshold=0.01),
        AlertRule("slow model", "p95_model_latency_ms", threshold=2000),
        AlertRule("wrong tools", "wrong_tool_rate", threshold=0.05, severity="ticket"),
        AlertRule("resolution collapse", "resolution_rate", threshold=0.8, comparator="<"),
    ]
    metrics = {"cost_per_task": 0.02, "p95_model_latency_ms": 900.0, "wrong_tool_rate": 0.25, "resolution_rate": 0.9}
    fired = evaluate_alerts(rules, metrics)
    assert [a.rule.name for a in fired] == ["cost per task", "wrong tools"]
    assert str(fired[1]).startswith("[ticket] wrong tools: wrong_tool_rate=0.25 > 0.05")
    with pytest.raises(KeyError):
        evaluate_alerts([AlertRule("typo", "cost_per_tsk", 1)], metrics)


async def test_agent_metrics_expose_looping_agent_cost_drift():
    healthy_tracer, looping_tracer = Tracer(), Tracer()
    healthy = Runner(bank_agent(), tracer=healthy_tracer)
    await healthy.run("h1", "balance please")

    def repeat_until_stopped(messages, tools):
        # keeps asking for the same lookup until the loop's duplicate detector refuses it
        if any(m.get("role") == "tool" and "duplicate_call" in m.get("content", "") for m in messages):
            return text("Your balance is 1234.5")
        return calls(call("get_balance", account_id="acc-1"))

    looping = Runner(bank_agent(FakeLLM(policy=repeat_until_stopped)), tracer=looping_tracer,
                     budget_factory=lambda: Budget(max_steps=10))
    result = await looping.run("l1", "balance please")
    healthy_m = agent_metrics(healthy_tracer, [healthy.store.get("h1")])
    looping_m = agent_metrics(looping_tracer, [result.session])
    assert healthy_m["wrong_tool_rate"] == 0.0 and looping_m["wrong_tool_rate"] > 0.0
    assert looping_m["steps_per_task"] > healthy_m["steps_per_task"] and looping_m["cost_per_task"] > healthy_m["cost_per_task"]
    assert wrong_tool_rate([result.session]) == pytest.approx(1 / 3)     # 3 calls, the third rejected as a duplicate
    summaries = TraceSummary.from_tracer(looping_tracer)
    assert cost_per_resolved(summaries, lambda s: False) == math.inf
    assert cost_per_resolved(summaries, lambda s: True) == pytest.approx(summaries[0].cost_usd)
