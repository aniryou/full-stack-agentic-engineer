# %% [markdown]
# # 04 · Context engineering and caching
#
# The context window is a budget, not a bucket. Every model call re-reads everything you put in front of it, so *what* you
# include, *in which order*, and *how much of it is byte-identical across calls* decide latency, cost and — past a point —
# accuracy. The lab's `ContextBuilder` makes those decisions explicit: a cache-friendly layout, tool-result truncation,
# compaction of old turns and a hard token cap. This notebook measures each of them.
#
# **Concept map:** see [docs/PRIMER_MAP.md](../docs/PRIMER_MAP.md); deeper in this repo: the [scaling primer](../../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §5.5 (context engineering for scale).
#
# In this notebook you will:
# 1. read a `ContextBuilder` layout and account for every token in it by role;
# 2. measure prefix caching across turns, and the cost difference a stable prefix makes at production volume;
# 3. write a better summarizer, a token-budget enforcer, a memory provider and a tool-scoping rule — the four levers you own.

# %%
import json
import re

from agentlab.llm import FakeLLM, count_tokens, messages_tokens
from agentlab.agents import ContextBuilder, Event, InvocationContext, LlmAgent, Session, ToolRegistry, context_report, tool

POLICY = ("Returns policy. Items may be returned within 30 days with proof of purchase; refunds go to the original payment "
          "method within 5 business days; sale items are exchange-only; gift cards are non-refundable. ") * 8

# %% [markdown]
# ## 1. The layout
#
# `ContextBuilder.build(session)` assembles the prompt in this order:
#
#     [system: instruction + static reference material]   ← identical for every user and every turn
#     [system: known facts about this user]                ← identical across this user's turns
#     [system: summary of old turns]                       ← changes only when compaction moves
#     [recent turns … current turn]                        ← changes every turn
#
# Stable material first, volatile last — because a context cache matches an exact **leading** prefix.

# %%
def memory_from_state(session: Session) -> list[str]:
    return [f"{k[5:]}: {v}" for k, v in sorted(session.state.items()) if k.startswith("user:")]


builder = ContextBuilder(instruction="You are the returns assistant for an online shoe shop. Customer tier: {tier}.",
                         static_context=[POLICY], memory_provider=memory_from_state, max_recent_turns=2)
session = Session(id="ctx-1", state={"tier": "gold", "user:name": "Anil", "user:country": "SG"})
for q, a in [("Can I return shoes?", "Yes, within 30 days with proof of purchase."), ("Sale items?", "Exchange only."),
             ("And gift cards?", "Non-refundable, sorry."), ("How long does a refund take?", None)]:
    session.append(Event(kind="user", payload={"content": q}))
    if a:
        session.append(Event(kind="model", payload={"content": a}))
prompt = builder.build(session)
for m in prompt:
    print(f"{m['role']:10s} {count_tokens(m['content']):5d} tokens  {m['content'][:66]!r}")

# %% [markdown]
# `context_report` is the first thing to look at when a prompt is expensive: tokens by role. Here the reference material
# dominates — which is fine **if** it is cached.

# %%
print(json.dumps(context_report(prompt), indent=1))
print("cacheable prefix (instruction + static material):", builder.cacheable_prefix_tokens(session), "tokens")

# %% [markdown]
# ## 2. Caching, measured
#
# `FakeLLM` reports `cached_tokens` for the longest leading prefix it has seen before, the way a real context cache bills.
# Run four turns through two builders: one with the layout above, one whose instruction starts with a per-turn timestamp —
# the classic way to destroy a cache without noticing.

# %%
QUESTIONS = ["Can I return shoes?", "What about sale items?", "And gift cards?", "How long does a refund take?"]


async def run_turns(builder: ContextBuilder, volatile_prefix: bool = False) -> list[tuple[int, int]]:
    llm = FakeLLM(responses=["Noted."] * len(QUESTIONS))
    s = Session(id="cache-demo", state={"tier": "gold", "user:name": "Anil", "now": "t0"})
    usage = []
    for i, q in enumerate(QUESTIONS):
        if volatile_prefix:
            s.state["now"] = f"2026-09-05T10:0{i}:00"
        s.append(Event(kind="user", payload={"content": q}))
        r = await llm.generate(builder.build(s))
        s.append(Event(kind="model", payload=r.as_message(), usage=r.usage))
        usage.append((r.usage.input_tokens, r.usage.cached_tokens))
    return usage


stable = ContextBuilder(instruction="You are the returns assistant. Customer tier: {tier}.", static_context=[POLICY], memory_provider=memory_from_state)
volatile = ContextBuilder(instruction="Current time: {now}. You are the returns assistant. Customer tier: {tier}.", static_context=[POLICY], memory_provider=memory_from_state)
for label, b, vol in (("stable prefix first", stable, False), ("timestamp first", volatile, True)):
    print(label)
    for i, (inp, cached) in enumerate(await run_turns(b, volatile_prefix=vol), 1):
        print(f"  turn {i}: input={inp:4d} cached={cached:4d}  cached share={cached / inp:4.0%}")

# %% [markdown]
# ### What that is worth
#
# Illustrative per-million-token prices — **verify against the current Vertex AI price list before quoting anyone** — with
# a cached input token at a steep discount to a fresh one. Multiply by a realistic day.

# %%
PRICE_INPUT_PER_M = 0.30        # USD per 1M fresh input tokens  — illustrative, verify
PRICE_CACHED_PER_M = 0.075      # USD per 1M cached input tokens — illustrative, verify (roughly 75% off)
TURNS_PER_DAY = 2_000_000
prompt_tokens, cacheable_tokens = 3_000, 2_400


def daily_input_cost(prompt_tokens: int, cached_tokens: int, turns: int) -> float:
    fresh = prompt_tokens - cached_tokens
    return turns * (fresh * PRICE_INPUT_PER_M + cached_tokens * PRICE_CACHED_PER_M) / 1_000_000


no_cache = daily_input_cost(prompt_tokens, 0, TURNS_PER_DAY)
with_cache = daily_input_cost(prompt_tokens, cacheable_tokens, TURNS_PER_DAY)
print(f"{TURNS_PER_DAY:,} turns/day × {prompt_tokens:,} input tokens: ${no_cache:,.0f}/day uncached vs ${with_cache:,.0f}/day "
      f"with {cacheable_tokens / prompt_tokens:.0%} of the prompt cached → ${no_cache - with_cache:,.0f}/day from message order alone")

# %% [markdown]
# ## 3. Bounding what tools put in the window
#
# Tool output is the largest uncontrolled input. `max_tool_result_chars` truncates it **in the model's view only** — the
# event log keeps the full result — and appends a marker that tells the model how to get more: *call again with a narrower
# request*. The lesson for tool design: give the model paging and filters, not everything.

# %%
big = Session(id="big")
big.append(Event(kind="user", payload={"content": "show my transactions"}))
big.append(Event(kind="model", payload={"content": "", "tool_calls": [{"id": "c1", "name": "list_transactions", "args": {}}]}))
big.append(Event(kind="tool_result", payload={"id": "c1", "name": "list_transactions", "ok": True,
                                              "content": json.dumps([{"id": i, "memo": "coffee"} for i in range(300)])}))
shaped = ContextBuilder(instruction="You are a bank assistant.", max_tool_result_chars=200).build(big)
tool_msg = [m for m in shaped if m["role"] == "tool"][0]
print("in the log:", len(big.events[-1].payload["content"]), "chars | in the prompt:", len(tool_msg["content"]), "chars")
print("tail of what the model sees:", tool_msg["content"][-105:])

# %% [markdown]
# ## 4. Compaction: old turns become a summary
#
# `max_recent_turns` keeps that many user turns verbatim; everything older is folded into one system message by the
# `summarizer`. The default `naive_summarizer` keeps the user's asks and the tool names — and loses everything the assistant
# decided or looked up, which is exactly what a later turn tends to need (Exercise 6.1 fixes that).

# %%
long = Session(id="long")
for i in range(1, 13):
    long.append(Event(kind="user", payload={"content": f"Question {i} about my account"}))
    long.append(Event(kind="model", payload={"content": f"Answer {i}: noted."}))
long.append(Event(kind="user", payload={"content": "So what did we decide?"}))
compacted = ContextBuilder(instruction="Assist.", max_recent_turns=3).build(long)
print(f"{len(long.messages())} history messages → {len(compacted)} in the prompt")
print(compacted[1]["content"][:200], "…")
print("report:", context_report(compacted))

# %% [markdown]
# `max_input_tokens` is the hard cap behind that: if the prompt is still too large, the oldest *recent* turns are dropped —
# never the current one, never the system prefix.

# %%
for cap in (None, 110, 90):
    msgs = ContextBuilder(instruction="Assist.", max_recent_turns=6, max_input_tokens=cap).build(long)
    print(f"max_input_tokens={str(cap):5s} → {len(msgs):2d} messages, {messages_tokens(msgs):3d} tokens, last={msgs[-1]['content']!r}")

# %% [markdown]
# ## 5. Tool-set scoping: schemas are tokens too
#
# Every tool schema is re-sent on every model call. A support agent with nine tools pays for nine schemas per step even
# when the current stage can legitimately use three. Scoping the tool set by stage cuts tokens **and** removes temptation:
# a refund tool cannot be misfired during triage if it is not offered.

# %%
@tool
def find_customer(email: str) -> dict:
    """Find a customer profile by email. Use before any account-specific action."""
    return {}


@tool
def get_order_status(order_id: str, include_items: bool = False) -> dict:
    """Fulfilment status of an order. Use when the user asks where an order is or when it arrives."""
    return {}


@tool
def search_policy(query: str) -> dict:
    """Search the returns and refunds policy. Use to check eligibility before promising anything."""
    return {}


@tool
def issue_refund(order_id: str, amount: float, reason: str) -> dict:
    """Refund an order (irreversible). Use only after eligibility is confirmed."""
    return {}


@tool
def create_return_label(order_id: str, carrier: str = "SpeedPost") -> dict:
    """Create a prepaid return shipping label for an order."""
    return {}


@tool
def hold_shipment(order_id: str) -> dict:
    """Put an unshipped order on hold; can be released later."""
    return {}


@tool
def send_confirmation(customer_id: str, channel: str = "email") -> dict:
    """Send the customer a confirmation of what was done."""
    return {}


@tool
def log_disposition(case_id: str, outcome: str) -> dict:
    """Record the case outcome for reporting."""
    return {}


@tool
def escalate_to_human(reason: str) -> dict:
    """Hand the conversation to a human agent. Use whenever you are unsure or the user asks for a person."""
    return {}


ALL_TOOLS = [find_customer, get_order_status, search_policy, issue_refund, create_return_label, hold_shipment,
             send_confirmation, log_disposition, escalate_to_human]
STAGE_OF = {
    "find_customer": {"triage"}, "get_order_status": {"triage", "resolve"}, "search_policy": {"triage", "resolve"},
    "issue_refund": {"resolve"}, "create_return_label": {"resolve"}, "hold_shipment": {"resolve"},
    "send_confirmation": {"close"}, "log_disposition": {"close"}, "escalate_to_human": {"*"},
}


def schema_tokens(tools) -> int:
    return count_tokens(json.dumps(ToolRegistry(list(tools)).schemas()))


print(f"all {len(ALL_TOOLS)} tools: {schema_tokens(ALL_TOOLS)} schema tokens on every model call")
print(f"triage tools picked by hand: {schema_tokens([find_customer, get_order_status, search_policy, escalate_to_human])} schema tokens")

# %% [markdown]
# ## 6. Exercises
#
# ### Exercise 6.1 — a summarizer that keeps what matters
#
# Write `better_summarizer(messages) -> str` for the old turns. It must keep (1) every identifier matching
# `[A-Z]{2,5}-\d{2,}` (order, card and claim numbers) from **any** role — tool results included — each mentioned once, and
# (2) the content of the **last assistant message** verbatim, the most recent decision. It must still compress: under half
# the tokens of the messages it replaces. Keep the user's asks too, briefly, if you like.

# %% exercise
def better_summarizer(messages: list[dict]) -> str:
    ### BEGIN SOLUTION
    ids: list[str] = []
    for m in messages:
        for found in re.findall(r"\b[A-Z]{2,5}-\d{2,}\b", str(m.get("content", ""))):
            if found not in ids:
                ids.append(found)
    asks = [str(m["content"])[:60] for m in messages if m.get("role") == "user"]
    last_decision = next((str(m["content"]) for m in reversed(messages) if m.get("role") == "assistant" and m.get("content")), "")
    parts = []
    if asks:
        parts.append("Earlier asks: " + " | ".join(asks[-3:]))
    if ids:
        parts.append("Identifiers mentioned: " + ", ".join(ids))
    if last_decision:
        parts.append("Last decision: " + last_decision)
    return " ".join(parts) or "(no earlier context)"
    ### END SOLUTION

# %% check
hist = Session(id="ids")
script = [
    ("My card was stolen", json.dumps({"ok": True, "data": {"card": "CARD-7781", "customer": "C-10", "status": "active",
                                                            "recent": [{"merchant": "cafe", "amount": 4.5}] * 6}}),
     "I have blocked CARD-7781 and ordered a replacement."),
    ("Also my order never arrived", json.dumps({"ok": True, "data": {"order": "ORD-10442", "status": "lost", "carrier": "SpeedPost",
                                                                     "events": [{"scan": "depot", "day": d} for d in range(6)]}}),
     "ORD-10442 is marked lost; I have re-sent it with tracking."),
    ("Thanks. Can you also check my claim?", json.dumps({"ok": True, "data": {"claim": "CLM-55", "status": "approved", "amount": 300,
                                                                              "history": [{"step": s, "ok": True} for s in range(6)]}}),
     "Claim CLM-55 is approved: SGD 300 will be paid on Friday."),
]
for i, (q, tool_content, answer) in enumerate(script):
    hist.append(Event(kind="user", payload={"content": q}))
    hist.append(Event(kind="model", payload={"content": "", "tool_calls": [{"id": f"c{i}", "name": "lookup", "args": {}}]}))
    hist.append(Event(kind="tool_result", payload={"id": f"c{i}", "name": "lookup", "ok": True, "content": tool_content}))
    hist.append(Event(kind="model", payload={"content": answer}))
for q in ("ok", "hello?"):
    hist.append(Event(kind="user", payload={"content": q}))
    hist.append(Event(kind="model", payload={"content": "Sure."}))
hist.append(Event(kind="user", payload={"content": "When exactly will the claim money arrive, and on which card?"}))
old_msgs = [m for t in ContextBuilder._split_turns(hist.messages())[:-3] for m in t]
naive = ContextBuilder(instruction="Assist.", max_recent_turns=3).build(hist)[1]["content"]
better = ContextBuilder(instruction="Assist.", max_recent_turns=3, summarizer=better_summarizer).build(hist)[1]["content"]
assert better.startswith("# Summary of earlier conversation"), better[:60]
for ident in ("CARD-7781", "ORD-10442", "CLM-55"):
    assert ident not in naive, "the naive summary should have lost the ids (check the fixture)"
    assert ident in better, f"{ident} did not survive compaction"
assert "Claim CLM-55 is approved: SGD 300 will be paid on Friday." in better, "keep the last decision verbatim"
summary_tokens = count_tokens(better_summarizer(old_msgs))
assert summary_tokens < 0.5 * messages_tokens(old_msgs), f"a summary must compress: {summary_tokens} vs {messages_tokens(old_msgs)} tokens"
print(f"✅ old turns: {messages_tokens(old_msgs)} tokens → summary: {summary_tokens} tokens, with every id and the last decision kept")

# %% [markdown]
# ### Exercise 6.2 — reorder a prompt for caching
#
# `build_prompt_bad` puts the volatile parts first and gets no cache hits. Write `build_prompt_good(policy, facts, history, now)`
# that returns the **same information** in a cache-friendly order: a system message with the policy; a system message with
# the user facts; the history as-is; then a final message carrying `now` (a system message `"Current time: …"` at the end
# is fine). The check runs four turns and requires a cached share of at least 60% from the second turn on.

# %% exercise
def build_prompt_bad(policy: str, facts: list[str], history: list[dict], now: str) -> list[dict]:
    return ([{"role": "system", "content": f"Current time: {now}"}]
            + [{"role": "system", "content": "Known about this user:\n" + "\n".join(facts)}]
            + history
            + [{"role": "system", "content": policy}])


def build_prompt_good(policy: str, facts: list[str], history: list[dict], now: str) -> list[dict]:
    ### BEGIN SOLUTION
    return ([{"role": "system", "content": policy}]
            + [{"role": "system", "content": "Known about this user:\n" + "\n".join(facts)}]
            + history
            + [{"role": "system", "content": f"Current time: {now}"}])
    ### END SOLUTION

# %% check
async def cached_shares(build) -> list[float]:
    llm = FakeLLM(responses=["ok"] * len(QUESTIONS))
    history: list[dict] = []
    shares = []
    for i, q in enumerate(QUESTIONS):
        history.append({"role": "user", "content": q})
        msgs = build(POLICY, ["name: Anil", "tier: gold"], history, f"2026-09-05T10:0{i}:00")
        flat = json.dumps(msgs)
        assert POLICY in flat and "Anil" in flat and q in flat and f"10:0{i}" in flat, "keep all the information"
        r = await llm.generate(msgs)
        history.append({"role": "assistant", "content": r.text})
        shares.append(r.usage.cached_tokens / r.usage.input_tokens)
    return shares


bad_shares, good_shares = await cached_shares(build_prompt_bad), await cached_shares(build_prompt_good)
assert max(bad_shares) == 0.0, bad_shares
assert all(s >= 0.6 for s in good_shares[1:]), f"cached share from turn 2 on: {[f'{s:.0%}' for s in good_shares]}"
print("✅ cached share per turn — volatile first:", [f"{s:.0%}" for s in bad_shares], "| stable first:", [f"{s:.0%}" for s in good_shares])

# %% [markdown]
# ### Exercise 6.3 — scope the tool set by stage
#
# Implement `scope_tools(tools, stage)` using `STAGE_OF`: return, in the original order, the tools whose stage set contains
# `stage` **or** `"*"` (always-available tools such as `escalate_to_human`). The check asserts the triage and close sets, a
# schema-token drop of at least 50% for triage, and that an agent built from the scoped list offers only those tools to the
# model.

# %% exercise
def scope_tools(tools: list, stage: str) -> list:
    ### BEGIN SOLUTION
    return [t for t in tools if STAGE_OF.get(t.spec.name, set()) & {stage, "*"}]
    ### END SOLUTION

# %% check
triage = scope_tools(ALL_TOOLS, "triage")
assert [t.spec.name for t in triage] == ["find_customer", "get_order_status", "search_policy", "escalate_to_human"], [t.spec.name for t in triage]
assert [t.spec.name for t in scope_tools(ALL_TOOLS, "close")] == ["send_confirmation", "log_disposition", "escalate_to_human"]
assert schema_tokens(triage) <= 0.5 * schema_tokens(ALL_TOOLS), (schema_tokens(triage), schema_tokens(ALL_TOOLS))
probe = FakeLLM(responses=["I need your order number to start."])
probe_session = Session(id="probe")
probe_session.append(Event(kind="user", payload={"content": "my order is late"}))
await LlmAgent("triage", probe, "Triage the request.", tools=triage).run_to_completion(InvocationContext(session=probe_session))
assert probe.calls[0]["tools"] == [t.spec.name for t in triage], probe.calls[0]["tools"]
print(f"✅ triage offers {len(triage)} of {len(ALL_TOOLS)} tools: {schema_tokens(triage)} vs {schema_tokens(ALL_TOOLS)} schema tokens per call")

# %% [markdown]
# ### Exercise 6.4 — a hard token budget that never drops the wrong thing
#
# Implement `enforce_budget(messages, max_tokens)` for a prompt shaped `[system, turn, turn, …, current turn]`, where a turn
# starts at a user message. Drop the **oldest whole turns** first until `messages_tokens(result) <= max_tokens`; **never**
# drop the first (system) message or the current turn — the last user message and anything after it — even if those alone
# exceed the budget. Return a new list; do not mutate the input.

# %% exercise
def enforce_budget(messages: list[dict], max_tokens: int) -> list[dict]:
    ### BEGIN SOLUTION
    system, rest = messages[:1], messages[1:]
    turns: list[list[dict]] = []
    for m in rest:
        if m.get("role") == "user" or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)
    while len(turns) > 1 and messages_tokens(system + [m for t in turns for m in t]) > max_tokens:
        turns = turns[1:]                                    # the oldest turn goes first; the current one never does
    return system + [m for t in turns for m in t]
    ### END SOLUTION

# %% check
convo = [{"role": "system", "content": "You are a bank assistant. " * 8}]
for i in range(1, 11):
    convo.append({"role": "user", "content": f"Question {i}: " + "details " * 10})
    convo.append({"role": "assistant", "content": f"Answer {i}: " + "explanation " * 10})
convo.append({"role": "user", "content": "Final question: what is my balance?"})
before = json.dumps(convo)
full_tokens = messages_tokens(convo)
budget = full_tokens // 2
trimmed = enforce_budget(convo, budget)
assert json.dumps(convo) == before, "do not mutate the input"
assert messages_tokens(trimmed) <= budget, (messages_tokens(trimmed), budget)
assert trimmed[0] is convo[0] and trimmed[-1] is convo[-1], "keep the system message and the current turn"
assert trimmed[1]["role"] == "user", "drop whole turns: the first kept history message must start a turn"
k = len(convo) - len(trimmed)
assert k > 0 and trimmed == convo[:1] + convo[1 + k:], "drop the OLDEST turns and keep the newest, in order"
assert messages_tokens(convo[:1] + convo[1 + k - 2:]) > budget, "you dropped more than necessary"
assert enforce_budget(convo, 10) == [convo[0], convo[-1]], "system + current turn survive even when they exceed the budget"
print(f"✅ {full_tokens} tokens → {messages_tokens(trimmed)} (budget {budget}); dropped the {k // 2} oldest turns, kept the newest")

# %% [markdown]
# ### Exercise 6.5 — a memory provider
#
# Write `user_memory(session) -> list[str]` returning at most **three** facts of the form `"<key>: <value>"` from the
# session's `user:`-prefixed state keys (key shown without the prefix), in a **stable order** (sorted by key) so the injected
# message is byte-identical across turns and stays cacheable. Ignore `app:`, `temp:` and plain keys.

# %% exercise
def user_memory(session: Session) -> list[str]:
    ### BEGIN SOLUTION
    facts = sorted((k[len("user:"):], v) for k, v in session.state.items() if k.startswith("user:"))
    return [f"{k}: {v}" for k, v in facts[:3]]
    ### END SOLUTION

# %% check
mem_session = Session(id="mem", state={"user:tier": "gold", "user:name": "Anil", "user:language": "en", "user:city": "Singapore",
                                      "app:version": "3.2", "temp:draft": "SECRET", "order_id": "ORD-1"})
mem_session.append(Event(kind="user", payload={"content": "hi"}))
mem_builder = ContextBuilder(instruction="Assist.", memory_provider=user_memory)
mem_prompt = mem_builder.build(mem_session)
assert len(mem_prompt) == 3 and mem_prompt[1]["role"] == "system", [m["role"] for m in mem_prompt]
lines = mem_prompt[1]["content"].splitlines()
assert lines[0] == "# Known about this user", lines[0]
assert lines[1:] == ["- city: Singapore", "- language: en", "- name: Anil"], lines[1:]
assert "SECRET" not in mem_prompt[1]["content"] and "3.2" not in mem_prompt[1]["content"]
assert mem_builder.build(mem_session)[1] == mem_prompt[1], "must be byte-identical across calls"
empty = Session(id="nomem")
empty.append(Event(kind="user", payload={"content": "hi"}))
assert len(ContextBuilder(instruction="Assist.", memory_provider=user_memory).build(empty)) == 2, "no facts → no memory message"
print("✅ injected:", mem_prompt[1]["content"].replace("\n", " | "))

# %% [markdown]
# ### Exercise 6.6 — say it in one paragraph
#
# In `context_strategy`, state the cache-friendly ordering (what goes first, what goes last, and why) and explain why a long
# context window is a **capability, not a strategy** — give at least one cost reason and one quality or latency reason.

# %% exercise
### BEGIN SOLUTION
context_strategy = (
    "Order the prompt from stable to volatile: system instruction and static reference material first, then per-user "
    "memory, then a summary of old turns, then recent turns, with anything per-request (timestamps, the current question) "
    "last — a context cache matches an exact leading prefix, so every volatile byte moved forward un-caches everything "
    "after it. A million-token window is a capability, not a strategy: every token is re-read and paid for on every call, "
    "latency grows with uncached input, and attention degrades as the window fills, so the job is still to retrieve, "
    "summarise and scope what the model sees each turn."
)
### END SOLUTION

# %% check
_c = context_strategy.lower()
assert len(context_strategy) > 200, "write a real paragraph"
assert ("stable" in _c or "prefix" in _c) and "first" in _c, "state the ordering and why it matters"
assert any(w in _c for w in ("cost", "paid", "price", "$", "bill")), "give a cost reason"
assert any(w in _c for w in ("attention", "accuracy", "degrade", "lost in the middle", "distract", "quality", "latency")), "give a quality or latency reason"
print("✅", context_strategy[:110], "…")

# %% [markdown]
# ## The one-minute version
#
# When context comes up, do not say "we use a long-context model". Say how the prompt is built: *"stable to volatile —
# instruction and reference material first so the prefix caches across every user; per-user memory next; a summary of old
# turns; then the recent turns and the current request. Tool results are truncated in the model's view with a pointer to
# fetch more, old turns are compacted by a summarizer that keeps decisions and identifiers, there is a hard token cap, and
# the tool set is scoped to the stage so we are not paying for nine schemas to use three."* Then quantify it: cached share
# per turn, tokens by role, dollars per day at the customer's volume. That is what context *engineering* means — and why a
# bigger window changes the ceiling, not the plan.
