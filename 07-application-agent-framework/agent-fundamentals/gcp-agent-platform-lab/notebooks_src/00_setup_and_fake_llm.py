# %% [markdown]
# # 00 · Setup and the fake model
#
# Everything in this lab runs **offline**: a scripted `FakeLLM` stands in for Gemini so you
# can practise the mechanics that design reviews actually probe — tool contracts, loops, state,
# identity, evaluation, cost — without an API key. When you have a key, `agentlab.llm.gemini.GeminiLLM`
# is a drop-in replacement (see `docs/GEMINI_ADAPTER.md`).
#
# **Primer sections:** 0 (how the loop is scored), 2.1 (the single-agent loop), 5.2 (token anchors).
#
# In this notebook you will:
# 1. drive a `FakeLLM` three ways (scripted queue, policy function, `KeywordPlanner`);
# 2. see how token usage and *context caching* are reported;
# 3. write your own planner policy — the same shape a real tool-calling model returns.

# %%
from agentlab.llm import FakeLLM, KeywordPlanner, Rule, call, calls, scripted, text, count_tokens

# %% [markdown]
# ## 1. A scripted model
#
# `scripted(...)` returns one entry per model call, in order. Strings become plain text answers;
# `call(...)` / `calls(...)` become tool-call turns — exactly the two shapes a function-calling model produces.

# %%
llm = scripted(
    call("get_balance", account_id="acc-1"),          # 1st call: the model asks for a tool
    "Your balance is SGD 1,234.50.",                   # 2nd call: the model answers
)
r1 = await llm.generate([{"role": "user", "content": "What's my balance?"}], tools=[{"name": "get_balance"}])
print("tool calls:", [(tc.name, tc.args) for tc in r1.tool_calls], "| finish:", r1.finish_reason)
r2 = await llm.generate([
    {"role": "user", "content": "What's my balance?"},
    r1.as_message(),
    {"role": "tool", "name": "get_balance", "tool_call_id": r1.tool_calls[0].id, "content": '{"ok":true,"data":{"balance":1234.5}}'},
])
print("text:", r2.text, "| usage:", r2.usage)

# %% [markdown]
# Every response carries `usage` (input/output/cached tokens) and a simulated `latency_ms`.
# The token count is an estimate (≈ 4 characters per token) — enough for budgets and cost maths to behave realistically.

# %%
print("tokens in 'Hello, agentic world':", count_tokens("Hello, agentic world"))
print("latency of the last call (ms):", round(r2.latency_ms, 1))

# %% [markdown]
# ## 2. Context caching, simulated
#
# Real models bill the *stable prefix* of a prompt at a steep discount when it repeats.
# `FakeLLM` mimics that: identical leading messages across calls are reported as `cached_tokens`.
# This is why prompt **layout** (stable material first) is a cost lever — Primer §2.5 and §5.3.

# %%
policy_text = "Refund policy: " + "items may be returned within 30 days with proof of purchase. " * 40
cache_llm = FakeLLM(responses=["ok", "ok", "ok"])
system = {"role": "system", "content": policy_text}
for turn in ("Can I return shoes?", "What about electronics?", "And gift cards?"):
    r = await cache_llm.generate([system, {"role": "user", "content": turn}])
    print(f"{turn:28s} input={r.usage.input_tokens:5d}  cached={r.usage.cached_tokens:5d}  "
          f"share={r.usage.cached_tokens / r.usage.input_tokens:.0%}")

# %% [markdown]
# ### Exercise 2.1 — break the cache, then explain it
#
# Change *one thing* about how the messages are built so that **no** tokens are cached on the second and third turns,
# without changing the policy text itself. (Hint: what happens if the volatile part comes *before* the stable part?)
#
# Then write one sentence in `explanation` on why that ordering is expensive in production.

# %% exercise
def build_uncacheable_messages(turn: str) -> list[dict]:
    """Return messages for one turn such that the prefix cache never hits."""
    ### BEGIN SOLUTION
    # Putting the volatile user turn first changes the very first message every call,
    # so no leading prefix ever repeats.
    return [{"role": "user", "content": turn}, {"role": "system", "content": policy_text}]
    ### END SOLUTION

### BEGIN SOLUTION
explanation = ("The cache keys on an exact leading prefix; if the volatile turn comes first, the long policy text is "
               "re-billed at full price on every call, which at scale is the difference between $3.8k and $6.7k a day.")
### END SOLUTION

# %% check
probe = FakeLLM(responses=["ok"] * 3)
cached = []
for t in ("Can I return shoes?", "What about electronics?", "And gift cards?"):
    r = await probe.generate(build_uncacheable_messages(t))
    cached.append(r.usage.cached_tokens)
assert cached == [0, 0, 0], f"expected no cache hits, got {cached}"
assert isinstance(explanation, str) and len(explanation) > 40, "write a real sentence"
print("✅ cache broken as intended:", cached)

# %% [markdown]
# ## 3. A policy-driven model
#
# A **policy** is any function `(messages, tools) -> ModelResponse | str | list[ToolCall]`.
# `KeywordPlanner` is a ready-made one: keywords in the latest user message → tool calls;
# once tool results are present → a templated final answer. It is deliberately dumb — the point of the lab is
# the *harness* around the model, which is where production systems succeed or fail.

# %%
planner = KeywordPlanner(
    rules=[
        Rule(r"balance", "get_balance", lambda t: {"account_id": "acc-1"}),
        Rule(r"block .*card", "block_card", lambda t: {"card_id": "card-9"}),
    ],
    answer_template="Done. {results}",
)
p_llm = FakeLLM(policy=planner)
tools = [{"name": "get_balance"}, {"name": "block_card"}]
r = await p_llm.generate([{"role": "user", "content": "Block my card and show my balance"}], tools=tools)
print("two rules matched → two parallel tool calls:", [tc.name for tc in r.tool_calls])

# %% [markdown]
# ### Exercise 3.1 — extract arguments with a regex
#
# Write a `Rule` whose `args` function pulls the order id out of messages like
# *"where is order ORD-10442?"* and *"status of ORD-7?"*, returning `{"order_id": "ORD-10442"}`.
# If no id is present, return `{"order_id": None}` (the tool's schema validation will then produce a useful error — Notebook 01).

# %% exercise
import re

def order_args(user_text: str) -> dict:
    ### BEGIN SOLUTION
    m = re.search(r"\b(ORD-\d+)\b", user_text, flags=re.IGNORECASE)
    return {"order_id": m.group(1).upper() if m else None}
    ### END SOLUTION

order_rule = Rule(r"order", "get_order_status", order_args)

# %% check
assert order_args("where is order ORD-10442?") == {"order_id": "ORD-10442"}
assert order_args("status of ord-7?") == {"order_id": "ORD-7"}
assert order_args("where is my order?") == {"order_id": None}
o_llm = FakeLLM(policy=KeywordPlanner([order_rule]))
resp = await o_llm.generate([{"role": "user", "content": "where is order ORD-10442?"}], tools=[{"name": "get_order_status"}])
assert resp.tool_calls[0].args == {"order_id": "ORD-10442"}
print("✅ order rule works")

# %% [markdown]
# ### Exercise 3.2 — write a policy from scratch
#
# Implement `triage_policy(messages, tools)` with this behaviour:
#
# * If the messages after the last user turn contain a **tool result** → return the text
#   `"Resolved: <content of the last tool result>"`.
# * Else, if the last user message mentions `"refund"` → return a single tool call `issue_refund(order_id="ORD-1", amount=20.0)`.
# * Else → return the text `"How can I help?"`.
#
# Use the helpers `text(...)` and `call(...)`. This is the same decision shape every function-calling model produces on each step.

# %% exercise
def triage_policy(messages, tools):
    ### BEGIN SOLUTION
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    after = messages[last_user + 1:]
    results = [m for m in after if m.get("role") == "tool"]
    if results:
        return text(f"Resolved: {results[-1]['content']}")
    if last_user >= 0 and "refund" in str(messages[last_user]["content"]).lower():
        return call("issue_refund", order_id="ORD-1", amount=20.0)
    return text("How can I help?")
    ### END SOLUTION

# %% check
t_llm = FakeLLM(policy=triage_policy)
a = await t_llm.generate([{"role": "user", "content": "I want a refund"}])
assert a.tool_calls and a.tool_calls[0].name == "issue_refund" and a.tool_calls[0].args["amount"] == 20.0
b = await t_llm.generate([{"role": "user", "content": "I want a refund"}, a.as_message(),
                          {"role": "tool", "name": "issue_refund", "content": '{"ok":true}'}])
assert b.text == 'Resolved: {"ok":true}', b.text
c = await t_llm.generate([{"role": "user", "content": "hello"}])
assert c.text == "How can I help?"
print("✅ triage policy behaves like a tool-calling model")

# %% [markdown]
# ## 4. What the model recorded
#
# `FakeLLM.calls` keeps every request it received. Exercises in later notebooks assert on it
# ("did the agent send the tool result back?", "how many model calls did this turn cost?").

# %%
print("calls made to t_llm:", t_llm.call_count)
print("tools offered on the last call:", t_llm.calls[-1]["tools"])

# %% [markdown]
# ## The one-minute version
#
# When someone asks *"how does the agent decide what to do?"*, the answer is the policy shape you just wrote:
# the model returns either text or structured tool calls against a schema, and **the runtime** validates, executes,
# appends results and enforces budgets. The intelligence is bounded by the harness — which is the subject of Notebook 01.
