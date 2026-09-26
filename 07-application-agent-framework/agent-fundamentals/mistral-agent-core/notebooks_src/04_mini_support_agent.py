# %% [markdown]
# # 04 · A mini support agent (capstone)
#
# Put the three ideas together into a small but honest support agent for a bank:
#
# * it answers **only** from tool results (never invents a balance);
# * reads are free; a **card block** is irreversible, so it needs human approval;
# * anything it can't do becomes a **case** for a human.
#
# This is the same shape as the bank agent in gcp-agent-platform-lab's capstone (notebook
# 14), shrunk to what fits on one screen. The production version (identity, MCP, evals,
# tracing) is `gcp-agent-platform-lab`, next to this lab in
# 07-application-agent-framework/agent-fundamentals/ — this is the concept underneath it.

# %%
from agentcore import Agent, FakeLLM, ToolError, call, calls, text, tool

# -- the systems of record (fakes) --------------------------------------------
ACCOUNTS = {"a1": {"balance": 1234.5, "currency": "SGD"}}
CARDS = {"card-1": {"status": "active", "account": "a1"}}
CASES = []

@tool
def get_balance(account_id: str) -> dict:
    """Return the balance for an account."""
    if account_id not in ACCOUNTS:
        raise ToolError("no such account", kind="not_found", hint="Confirm the account id.")
    return {"account_id": account_id, **ACCOUNTS[account_id]}

@tool(confirm=True)                       # irreversible → the loop will ask a human
def block_card(card_id: str) -> dict:
    """Block a card. Irreversible; requires approval."""
    CARDS[card_id]["status"] = "blocked"
    return {"card_id": card_id, "status": "blocked"}

# %% [markdown]
# ## Exercise 4.1 — the escalation tool
#
# Write `raise_case(summary: str, queue: str)` that appends `{"summary","queue"}` to the
# `CASES` list and returns `{"case_id": ..., "queue": ...}` where `case_id` is
# `"CASE-" + <new length of CASES>`. This is the "hand to a human" path for anything out
# of scope.

# %% exercise
@tool
def raise_case(summary: str, queue: str) -> dict:
    """Open a case for a human team."""
    ### BEGIN SOLUTION
    CASES.append({"summary": summary, "queue": queue})
    return {"case_id": f"CASE-{len(CASES)}", "queue": queue}
    ### END SOLUTION

# %% check
CASES.clear()
out = raise_case.run({"summary": "dispute on ORD-1", "queue": "billing"})
assert out["data"]["case_id"] == "CASE-1" and len(CASES) == 1
print("✅ raise_case works:", out["data"])

# %% [markdown]
# ## The agent
# A scripted model walks three intents. Read the transcript for each.

# %%
support = Agent(
    tools=[get_balance, block_card, raise_case],
    instruction="You are a bank support agent. Use tools for every fact; never guess.",
    llm=FakeLLM([]),   # replaced per-run below
)

def run_once(script, message, on_confirm=None):
    support.llm = FakeLLM(script)
    r = support.run(message, on_confirm=on_confirm)
    print(f"\n=== {message} ===")
    print(r.transcript())
    return r

# intent 1: a balance question, answered from the tool
run_once([call("get_balance", account_id="a1"), "Your balance is SGD 1,234.50."],
         "what's my balance on a1?")

# %% [markdown]
# ## Exercise 4.2 — the approval gate
#
# Run a "block my card" conversation twice with the same script
# `[call("block_card", card_id="card-1"), "Your card is now blocked."]`, keeping the two
# results as `declined` and `approved`:
#
# * once with `on_confirm` returning **False** — assert the card is still `"active"`
#   and the tool result the model saw contains `"declined"`;
# * once with `on_confirm` returning **True** — assert the card becomes `"blocked"`.

# %% exercise
script = [call("block_card", card_id="card-1"), "Your card is now blocked."]
### BEGIN SOLUTION
CARDS["card-1"]["status"] = "active"
declined = run_once(list(script), "block card-1", on_confirm=lambda name, args: False)
assert CARDS["card-1"]["status"] == "active"
assert any("declined" in m["content"] for m in declined.messages if m["role"] == "tool")

approved = run_once(list(script), "block card-1", on_confirm=lambda name, args: True)
assert CARDS["card-1"]["status"] == "blocked"
### END SOLUTION

# %% check
def block_results(r):
    return [m["content"] for m in r.messages if m["role"] == "tool" and m["name"] == "block_card"]
assert CARDS["card-1"]["status"] == "blocked", "after approval the card must be blocked"
assert block_results(declined) and all("declined" in c for c in block_results(declined)), \
    "the declined run must reach block_card and get a declined result"
assert block_results(approved) and '"blocked"' in block_results(approved)[0] \
    and "declined" not in block_results(approved)[0], \
    "the card must be blocked by the agent's approved block_card call, not by hand"
print("✅ irreversible action gated on human approval")

# %% [markdown]
# ## Exercise 4.3 — never act without a tool
#
# The one property that makes this agent trustworthy: every factual claim comes from a
# tool result, and every state change comes from a tool call. Write `used_a_tool(result)`
# that returns True if the transcript contains at least one `tool` message. Then confirm
# the balance answer used one. (In real evals this is a *trajectory* check — did the
# agent look before it spoke — and it is graded as an absolute gate.)

# %% exercise
def used_a_tool(result) -> bool:
    ### BEGIN SOLUTION
    return any(m["role"] == "tool" for m in result.messages)
    ### END SOLUTION

# %% check
r = run_once([call("get_balance", account_id="a1"), "SGD 1,234.50."], "balance on a1?")
assert used_a_tool(r) is True
# and an out-of-scope ask should escalate, not invent an answer
esc = run_once([call("raise_case", summary="wants a loan", queue="lending"),
                "I've raised this with our lending team."], "can I get a loan?")
assert used_a_tool(esc) and any(m["name"] == "raise_case" for m in esc.messages if m["role"] == "tool")
print("✅ the agent acts only through tools")

# %% [markdown]
# ## Where to go next
# You now have the whole core: a model, tools, the loop, state, budgets, and a human in
# the loop for irreversible actions. The step-up — async and parallel tool calls, MCP
# servers and a policy gateway, OAuth identity propagation, evaluation gates, and
# tracing — is `gcp-agent-platform-lab`, next to this lab in this repo. Same concepts;
# more machinery.
#
# ## The one-minute version
# Walk this design as: unit of work (a support interaction), every fact from a tool,
# irreversible actions gated by a human, out-of-scope work escalated as a case, and the
# trustworthiness property ("never acts without a tool") checked as an absolute eval gate.
