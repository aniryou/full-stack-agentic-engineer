# %% [markdown]
# # 09 · Tracing, cost and latency metrics
#
# You cannot manage what you cannot see — and with agents, what you cannot see is *which step* burned
# the tokens, *which tool* took the time, and *which conversation* wrote a card number into a log. This
# notebook attaches a tracer to the runner, reads the span tree, and turns spans into the numbers an
# operator watches: dollars per conversation, p95 latency by span kind, time-to-first-token and tokens
# per second — then alert rules that fire when a looping agent drifts.
#
# **Primer sections:** 4.2 (observability: spans with `gen_ai.*` attributes, redaction, retention),
# 5.2 (token and latency anchors), 5.3 A (cost arithmetic: pro vs flash, caching).
#
# In this notebook you will:
# 1. attach a `Tracer` to a `Runner`, run three conversations and read the span tree;
# 2. compute cost per conversation, p95 latency by span kind, the most expensive step, and TTFT / tokens per second from a stream;
# 3. redact customer data before export and write alert rules that catch a looping agent's cost drift.

# %%
from agentlab.agents import Budget, ContextBuilder, LlmAgent, Runner, SideEffect, ToolContext, tool
from agentlab.llm import FakeLLM, KeywordPlanner, Rule, StreamChunk, Usage, call, calls, text
from agentlab.observability import (DEFAULT_PRICES, AlertRule, Price, RedactingExporter, Tracer, TraceSummary,
                                  agent_metrics, evaluate_alerts, now_ms, percentile, redact, reported_latency_ms,
                                  rule, summarize_latencies, ttft_and_tps)

# %% [markdown]
# ## 1. A traced runner
#
# The agent loop opens a span per agent turn, per model call and per tool call, and stamps them with
# OpenTelemetry GenAI attribute names (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, …) so any
# real backend understands them. Tools can enrich their own span through `ctx.span` — `create_case`
# records the case details, which is exactly how customer data ends up in traces (section 5).

# %%
@tool
def get_balance(account_id: str) -> dict:
    """Current balance for an account."""
    return {"account_id": account_id, "balance": 1234.5, "currency": "SGD"}


@tool
def list_transactions(account_id: str, limit: int = 5) -> list:
    """Most recent transactions, newest first."""
    return [{"merchant": "ACME Sports", "amount": 89.9}, {"merchant": "MRT top-up", "amount": 20.0}][:limit]


@tool(side_effect=SideEffect.REVERSIBLE)
def create_case(subject: str, details: str, ctx: ToolContext) -> dict:
    """Open a support case (annotates its span with the details)."""
    if ctx.span is not None:
        ctx.span.set(**{"case.subject": subject, "case.details": details})
    return {"case_id": "CASE-1001", "subject": subject}


RULES = [
    Rule(r"balance", "get_balance", {"account_id": "acc-1"}),
    Rule(r"transactions?|charges?", "list_transactions", {"account_id": "acc-1", "limit": 5}),
    Rule(r"open a case|not mine|dispute", "create_case", lambda t: {"subject": "Disputed charge", "details": t}),
]
POLICY_TEXT = "Bank assistant policy: " + "verify identity before disclosing balances; never move money without confirmation. " * 12


def make_agent(policy=None, model_name="fake-flash"):
    """A bank assistant with a long, stable system prefix — the part a context cache can serve."""
    llm = FakeLLM(policy=policy or KeywordPlanner(RULES), model_name=model_name)
    context = ContextBuilder(instruction="You are a careful bank assistant.", static_context=[POLICY_TEXT])
    return LlmAgent("bank-assistant", llm, "You are a careful bank assistant.", tools=[get_balance, list_transactions, create_case], context=context)


tracer = Tracer()
runner = Runner(make_agent(), tracer=tracer)
conversations = {
    "conv-1": "What's my balance?",
    "conv-2": "Show my balance and my recent transactions",
    "conv-3": "Open a case: the ACME charge is not mine. Card 4111 1111 1111 1111, reach me at jane.doe@example.com",
}
for sid, message in conversations.items():
    with tracer.start_trace(f"turn {sid}", session=sid):      # one trace per conversation turn
        result = await runner.run(sid, message)
    print(f"{sid}: {result.text[:72]}…")

# %%
tracer.print_tree()

# %% [markdown]
# Read one model span's attributes: model, tokens (with the cached share), finish reason, and the latency
# the model reported. The second model call of `conv-2` shows `cached_tokens > 0` — the system prefix was
# identical to the first call's, so the simulated context cache served it.

# %%
conv2 = tracer.traces()[1]
for s in conv2.by_kind("model"):
    print({k: v for k, v in s.attrs.items() if k.startswith("gen_ai") or k == "model.latency_ms"})

# %% [markdown]
# ## 2. Cost per conversation
#
# `TraceSummary` reduces a trace to the operator's numbers; the price table bills uncached input, cached
# input and output separately. The default table is **illustrative — verify against the official pricing page**.

# %%
summaries = TraceSummary.from_tracer(tracer, DEFAULT_PRICES, latency=reported_latency_ms)
for s in summaries:
    print(s.as_row())
print("\nprice table note:", DEFAULT_PRICES.note)

# %% [markdown]
# The same arithmetic as Primer §5.3 A, at small scale. Take a typical turn — 6,000 input tokens of which
# 4,500 are a stable prefix, 300 output tokens — and price it four ways. Per turn the numbers look like
# rounding errors; at a million turns a day they are budget lines, and the two levers are visible:
# **model tier** (≈4×) and **caching** (≈2×).

# %%
turn = Usage(input_tokens=6000, output_tokens=300)
cached_turn = Usage(input_tokens=6000, output_tokens=300, cached_tokens=4500)
print(f"{'scenario':32s} {'per turn':>10s} {'per 1M turns/day':>18s}")
for label, usage, model in (("flash, nothing cached", turn, "gemini-3-flash"), ("flash, 75% prefix cached", cached_turn, "gemini-3-flash"),
                            ("pro, nothing cached", turn, "gemini-3.1-pro"), ("pro, 75% prefix cached", cached_turn, "gemini-3.1-pro")):
    c = DEFAULT_PRICES.cost(usage, model)
    print(f"{label:32s} ${c:>9.5f} ${c * 1_000_000:>17,.0f}")
print("\nbreakdown of 'flash, 75% cached':", {k: round(v, 6) for k, v in DEFAULT_PRICES.breakdown(cached_turn, 'gemini-3-flash').items()})

# %% [markdown]
# And from real traces: the identical conversation on a pro-tier model (same prompts, same cache warm-up)
# costs a multiple before any scale — the multiple is the ratio of the price rows, weighted by where the
# tokens went.

# %%
pro_tracer = Tracer()
pro_runner = Runner(make_agent(model_name="fake-pro"), tracer=pro_tracer)
for sid in ("conv-1", "conv-2"):                      # conv-1 warms the cache, exactly as it did for flash
    await pro_runner.run(sid, conversations[sid])
flash_cost, pro_cost = summaries[1].cost_usd, TraceSummary.from_tracer(pro_tracer)[1].cost_usd
print(f"conv-2 on flash ${flash_cost:.6f} vs pro ${pro_cost:.6f}  (×{pro_cost / flash_cost:.1f})")

# %% [markdown]
# ### Exercise 2.1 — implement the cost formula
#
# `cost_usd(usage, price)` returns dollars for one call, where `price` is a `Price` in **$ per million
# tokens** with fields `input`, `cached_input`, `output`. Bill `input_tokens − cached_tokens` at the input
# rate, `cached_tokens` at the cached rate, and `output_tokens + thinking_tokens` at the output rate.

# %% exercise
def cost_usd(usage: Usage, price: Price) -> float:
    ### BEGIN SOLUTION
    uncached = usage.input_tokens - usage.cached_tokens
    return (uncached * price.input + usage.cached_tokens * price.cached_input
            + (usage.output_tokens + usage.thinking_tokens) * price.output) / 1_000_000
    ### END SOLUTION

# %% check
flash, pro = DEFAULT_PRICES.price_for("gemini-3-flash"), DEFAULT_PRICES.price_for("gemini-3.1-pro")
assert abs(cost_usd(turn, flash) - 0.0039) < 1e-12, cost_usd(turn, flash)                    # 6000×0.5 + 300×3 per million
assert abs(cost_usd(cached_turn, flash) - 0.001875) < 1e-12, cost_usd(cached_turn, flash)    # 1500×0.5 + 4500×0.05 + 300×3
assert abs(cost_usd(cached_turn, pro) - 0.0075) < 1e-12, cost_usd(cached_turn, pro)
thinking = Usage(input_tokens=1000, output_tokens=100, thinking_tokens=400)
assert abs(cost_usd(thinking, pro) - DEFAULT_PRICES.cost(thinking, "gemini-3.1-pro")) < 1e-12, "thinking tokens bill as output"
for s in summaries:
    u = Usage(input_tokens=s.input_tokens, output_tokens=s.output_tokens, cached_tokens=s.cached_tokens)
    assert abs(cost_usd(u, flash) - s.cost_usd) < 1e-12, (s.name, cost_usd(u, flash), s.cost_usd)
print("✅ cost formula matches the price table on every traced conversation")

# %% [markdown]
# ## 3. Latency by span kind, and the most expensive step
#
# Percentiles are computed per **span kind**, because "p95 latency" of a turn mixes model calls, tool
# calls and orchestration into one number nobody can act on. `FakeLLM` does not actually sleep, so we
# read the latency it *reports* (`model.latency_ms`); in production the span duration is the truth.

# %%
for kind, stats in summarize_latencies(tracer.spans, latency=reported_latency_ms).items():
    print(f"{kind:9s} n={stats['count']}  p50={stats['p50']:8.1f} ms  p95={stats['p95']:8.1f} ms  max={stats['max']:8.1f} ms")
priciest = max(summaries, key=lambda s: s.cost_usd)
step, step_cost = priciest.top_costs(1)[0]
span_name, span_ms = priciest.longest_span
print(f"\nmost expensive conversation: {priciest.name} (${priciest.cost_usd:.6f}); costliest step: {step} (${step_cost:.6f})")
print(f"longest span in it (wall time): {span_name} {span_ms:.1f} ms")

# %% [markdown]
# ### Exercise 3.1 — nearest-rank percentile
#
# `my_percentile(values, p)` returns the smallest observed value below which `p` percent of the sample
# lies, with **no interpolation**: sort, take rank `ceil(p/100 × n)` (at least 1), return the value at
# that rank. An interpolated p95 can be a latency nobody experienced; nearest-rank never invents a number.

# %% exercise
import math

def my_percentile(values, p: float) -> float:
    ### BEGIN SOLUTION
    data = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(data)))
    return data[rank - 1]
    ### END SOLUTION

# %% check
assert my_percentile([15, 20, 35, 40, 50], 40) == 20 and my_percentile([15, 20, 35, 40, 50], 50) == 35
assert my_percentile([15, 20, 35, 40, 50], 100) == 50 and my_percentile([15, 20, 35, 40, 50], 0) == 15
assert my_percentile(list(range(1, 101)), 95) == 95 and my_percentile([3, 1, 2], 50) == 2
model_latencies = [reported_latency_ms(s) for s in tracer.spans if s.kind == "model"]
for p in (50, 90, 95, 99):
    assert my_percentile(model_latencies, p) == percentile(model_latencies, p), p
print(f"✅ nearest-rank percentile; p95 model latency here = {my_percentile(model_latencies, 95):.1f} ms")

# %% [markdown]
# ## 4. Time-to-first-token and tokens per second
#
# Users feel two latencies: the wait before the first token (**TTFT**) and the typing speed after it
# (**tokens/s**). Both come from stream chunk timestamps and the usage in the final chunk. `FakeLLM` with
# `simulate_time=True, time_scale=0.01` sleeps for 1% of the simulated time, so the notebook stays fast.

# %%
stream_llm = FakeLLM(responses=["Your balance is SGD 1234.5 and your last charge was ACME Sports for 89.90. " * 6],
                     simulate_time=True, time_scale=0.01, ttft_ms=400, tokens_per_sec=150)
start = now_ms()
chunks = []
async for chunk in stream_llm.stream([{"role": "user", "content": "balance and last charge?"}]):
    chunks.append(chunk)
stats = ttft_and_tps(chunks, start)
print(f"{len(chunks)} chunks; measured TTFT {stats.ttft_ms:.1f} ms real ≈ {stats.ttft_ms / 0.01:,.0f} ms simulated (configured 400)")
print(f"{stats.output_tokens} output tokens over {stats.generation_ms:.1f} ms real → {stats.tokens_per_sec:,.0f} tok/s real; "
      f"the sleep granularity dominates at this scale, so read the shape, not the digits")

# %% [markdown]
# ### Exercise 4.1 — implement `ttft_and_tps`
#
# `my_ttft_and_tps(chunks, start_ms)` returns `(ttft_ms, tokens_per_sec)`:
# * TTFT = `t_ms` of the first chunk that carries content (`text` or `tool_call`) minus `start_ms`;
# * tokens/s = `output_tokens` from the last chunk's `usage`, divided by the seconds between that first
#   content chunk and the **last** chunk (`0.0` if the window is empty).

# %% exercise
def my_ttft_and_tps(chunks: list[StreamChunk], start_ms: float) -> tuple[float, float]:
    ### BEGIN SOLUTION
    first = next(c for c in chunks if c.text or c.tool_call is not None)
    last = chunks[-1]
    output_tokens = last.usage.output_tokens if last.usage else 0
    window_s = (last.t_ms - first.t_ms) / 1000.0
    return first.t_ms - start_ms, (output_tokens / window_s if window_s > 0 else 0.0)
    ### END SOLUTION

# %% check
synthetic = [StreamChunk(text="Hello ", t_ms=1400.0), StreamChunk(text="agentic ", t_ms=1500.0), StreamChunk(text="world", t_ms=1600.0),
             StreamChunk(done=True, usage=Usage(input_tokens=50, output_tokens=30), t_ms=1600.0)]
assert my_ttft_and_tps(synthetic, 1000.0) == (400.0, 150.0), my_ttft_and_tps(synthetic, 1000.0)
tool_first = [StreamChunk(tool_call=call("get_balance", account_id="acc-1"), t_ms=250.0), StreamChunk(done=True, usage=Usage(output_tokens=12), t_ms=250.0)]
assert my_ttft_and_tps(tool_first, 0.0) == (250.0, 0.0), "a tool call is a first token too; an empty window claims no throughput"
mine = my_ttft_and_tps(chunks, start)
assert abs(mine[0] - stats.ttft_ms) < 1e-9 and abs(mine[1] - stats.tokens_per_sec) < 1e-6
print(f"✅ TTFT {mine[0]:.1f} ms, {mine[1]:,.0f} tok/s — matches the library on the real stream")

# %% [markdown]
# ## 5. Redaction before export
#
# Traces carry customer data: the case details above contain a card number and an email. Redaction
# happens **at the export boundary**, field by field, so what leaves the process is clean while the
# in-memory span keeps what the tool wrote. `drop_attrs` removes whole attributes (raw prompts,
# retrieved passages) that should never leave at all.

# %%
exporter = RedactingExporter(drop_attrs=("gen_ai.prompt",))
case_span = next(s for s in tracer.spans if s.name == "tool create_case")
exported = exporter.export(tracer)
exported_case = next(d for d in exported if d["name"] == "tool create_case")
print("in memory:", case_span.attrs["case.details"])
print("exported: ", exported_case["attrs"]["case.details"])
print("\nthe session log needs the same treatment:")
print("  ", exporter.scrub(runner.store.get("conv-3").to_dict())["events"][0]["payload"]["content"])

# %% [markdown]
# ### Exercise 5.1 — a redaction rule for card numbers
#
# Build `card_rule = rule("card", pattern, "<card>")` whose regex matches a 16-digit card number written
# as `4111111111111111`, `4111 1111 1111 1111` or `4111-1111-1111-1111` — and nothing shorter, longer,
# or date-like. (`\b` word boundaries and a repeated group of four digits get you there.)

# %% exercise
### BEGIN SOLUTION
card_rule = rule("card", r"\b(?:\d{4}[ -]?){3}\d{4}\b", "<card>")
### END SOLUTION

# %% check
for hit in ("4111111111111111", "4111 1111 1111 1111", "4111-1111-1111-1111", "5500 0000 0000 0004"):
    assert redact(f"card {hit} ok", rules=[card_rule]) == "card <card> ok", hit
for miss in ("order 12345678", "2026-09-05 10:30", "call 6123 4567", "12345678901234567", "acc-1 balance 1234.5"):
    assert redact(miss, rules=[card_rule]) == miss, miss
assert "4111" not in redact(case_span.attrs["case.details"], rules=[card_rule])
print("✅ card numbers redacted, everything else untouched")

# %% [markdown]
# ## 6. Alert rules on a looping agent
#
# A model stuck in a loop re-issues the same call until the runtime's duplicate detector refuses it. Each
# extra step costs a model call, so **cost per task** drifts up and the **wrong-tool rate** (unknown,
# invalid or duplicate calls over all calls) rises — two numbers worth alerting on, alongside p95 model
# latency. `agent_metrics` computes them from a tracer plus the session logs. Notice how caching softens
# the cost drift (the repeated prompt prefix is cheap) while steps and wrong-tool rate double outright.

# %%
def repeat_until_stopped(messages, tools):
    """Re-issues the same lookup until a tool result says 'duplicate_call', then answers."""
    if any(m.get("role") == "tool" and "duplicate_call" in m.get("content", "") for m in messages):
        return text("Your balance is SGD 1234.5.")
    return calls(call("get_balance", account_id="acc-1"))


healthy_tracer, looping_tracer = Tracer(), Tracer()
healthy = Runner(make_agent(), tracer=healthy_tracer)
looping = Runner(make_agent(policy=repeat_until_stopped), tracer=looping_tracer, budget_factory=lambda: Budget(max_steps=10))
for i in range(3):
    await healthy.run(f"h{i}", "What's my balance?")
    await looping.run(f"l{i}", "What's my balance?")
healthy_metrics = agent_metrics(healthy_tracer, [healthy.store.get(f"h{i}") for i in range(3)], latency=reported_latency_ms)
looping_metrics = agent_metrics(looping_tracer, [looping.store.get(f"l{i}") for i in range(3)], latency=reported_latency_ms)
print(f"{'metric':24s} {'healthy':>12s} {'looping':>12s}")
for key in healthy_metrics:
    print(f"{key:24s} {healthy_metrics[key]:12.5f} {looping_metrics[key]:12.5f}")

# %%
rules = [  # thresholds anchored on the healthy baseline: +25% cost per task is drift worth a look
    AlertRule("cost per task drift", "cost_per_task", threshold=1.25 * healthy_metrics["cost_per_task"]),
    AlertRule("steps per task", "steps_per_task", threshold=3),
    AlertRule("slow model p95", "p95_model_latency_ms", threshold=2000),
]
for name, metrics in (("healthy", healthy_metrics), ("looping", looping_metrics)):
    fired = evaluate_alerts(rules, metrics)
    print(f"{name}: {[str(a) for a in fired] or 'no alerts'}")

# %% [markdown]
# ### Exercise 6.1 — an alert rule for the wrong-tool rate
#
# Define `wrong_tool_alert`, an `AlertRule` on the `wrong_tool_rate` metric that fires when more than 5%
# of tool calls are wrong (unknown tool, invalid arguments, or a duplicate). Give it a clear name and
# severity `"ticket"` — a rising wrong-tool rate is a prompt or schema problem, not a 3 a.m. page.

# %% exercise
### BEGIN SOLUTION
wrong_tool_alert = AlertRule("wrong-tool rate above 5%", "wrong_tool_rate", threshold=0.05, comparator=">", severity="ticket")
### END SOLUTION

# %% check
assert wrong_tool_alert.metric == "wrong_tool_rate" and wrong_tool_alert.severity == "ticket"
assert evaluate_alerts([wrong_tool_alert], healthy_metrics) == [], "the healthy agent must not fire it"
(fired,) = evaluate_alerts([wrong_tool_alert], looping_metrics)
assert abs(fired.observed - 1 / 3) < 1e-9, fired.observed          # 3 calls per task, the third refused as a duplicate
assert not wrong_tool_alert.fires(0.05) and wrong_tool_alert.fires(0.0501), "strictly above 5%"
print("✅", fired)

# %% [markdown]
# ### Exercise 6.2 — why traces need retention rules
#
# Write `retention_explanation`: two or three sentences a reviewer would accept, covering *what* is in a
# trace that makes it sensitive, *why* keeping it forever is a liability rather than an asset, and *what*
# the rule looks like (a time limit, and what survives it).

# %% exercise
### BEGIN SOLUTION
retention_explanation = (
    "A trace holds the customer's own words, tool arguments and retrieved records, so it is personal data "
    "even after field-level redaction — regulations such as GDPR and PDPA require a purpose and a time limit "
    "for keeping it, and every retained day widens the blast radius of a breach or a subpoena. "
    "Keep raw traces only as long as debugging and eval triage need them (days to a few weeks), then delete "
    "them and retain only aggregated metrics and redacted, reviewed golden cases."
)
### END SOLUTION

# %% check
assert isinstance(retention_explanation, str) and len(retention_explanation) > 120, "write the real paragraph"
low = retention_explanation.lower()
assert any(k in low for k in ("personal", "pii", "customer", "sensitive")), "say what makes a trace sensitive"
assert any(k in low for k in ("delete", "expire", "days", "weeks", "ttl", "time limit")), "say how long, and what happens after"
print("✅ retention rule explained")

# %% [markdown]
# ## The one-minute version
#
# When asked *"how would you operate this?"*, answer with the span tree in your head:
#
# * **One trace per turn, spans per agent / model / tool**, attributes in the `gen_ai.*` convention so the
#   platform's tracing backend (Cloud Trace, an OTel collector) works unchanged. Tool spans carry `tool.ok`
#   and `tool.error`; model spans carry tokens, cached tokens and finish reason.
# * **Cost is arithmetic on those spans**: uncached input, cached input, output — priced per model. Say the
#   two levers out loud: model tier is ≈4×, prompt caching is ≈2×, and both come from prompt layout and
#   routing decisions, not from negotiation.
# * **Latency per span kind**, nearest-rank percentiles, TTFT and tokens/s for streaming UX. A single
#   "p95 of the turn" hides whether the model or a tool is slow.
# * **Redaction at the export boundary, retention by rule.** Traces are customer data; scrub fields before
#   they leave the process, keep raw traces days not years, keep aggregates and reviewed golden cases.
# * **Alerts on the agent's economics**: cost per task, steps per task, wrong-tool rate, p95 model latency.
#   A looping agent is a cost incident first and a quality incident second; both show up here before the invoice.
