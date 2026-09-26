# %% [markdown]
# # 02 · Workflows and multi-agent systems
#
# Once you have one reliable loop, the question is how to compose several. The rule: **use code where the
# control flow is known, use the model where judgement is needed.** Workflow agents (`Sequential`, `Parallel`, `Loop`) are
# code; delegation (`AgentTool`) is a model deciding to call another agent. Both compound cost and failure, so this
# notebook also makes you *measure* what an extra agent costs before you add one.
#
# **Concept map:** see [docs/PRIMER_MAP.md](../docs/PRIMER_MAP.md); deeper in this repo: the [scaling primer](../../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §1.7 (multi-agent multiplies everything).
#
# In this notebook you will:
# 1. wire agents together through session state (`output_key` + `{placeholder}` instructions) in sequence, in parallel and in a loop;
# 2. delegate through `AgentTool`, read the `delegation` events, and measure what delegation costs in model calls and tokens;
# 3. quantify compounded reliability and decide — with numbers — when a second agent earns its keep.

# %%
import json
import random

from agentlab.llm import FakeLLM, call, calls, scripted, text
from agentlab.agents import (Budget, Event, InvocationContext, LlmAgent, LoopAgent, ParallelAgent, SequentialAgent, Session,
                           SideEffect, tool)


def new_session(message: str, session_id: str = "s1") -> Session:
    s = Session(id=session_id, tenant="t1", user="u1")
    s.append(Event(kind="user", payload={"content": message}))
    return s

# %% [markdown]
# ## 1. Sequential: a pipeline that talks through state
#
# A `SequentialAgent` runs its children in order **on the same session**. Each child writes its final text to
# `session.state[output_key]`; the next child's instruction reads it through `{placeholders}` — the `ContextBuilder`
# formats the instruction with `session.state` on every model call. The transcript is shared too, so later stages also
# *see* earlier answers; the state key is the explicit, typed hand-off you can assert on.

# %%
classify = LlmAgent("classify", scripted("shipping_delay"), "Classify the user's issue into one snake_case label.", output_key="issue_type")
draft = LlmAgent("draft", scripted("Sorry about the delay on your parcel — it ships tomorrow with tracking."),
                 "Issue type: {issue_type}. Draft a two-sentence reply.", output_key="reply")
pipeline = SequentialAgent("support_pipeline", [classify, draft])

session = new_session("My parcel is a week late.")
await pipeline.run_to_completion(InvocationContext(session=session))
print("state after the pipeline:", json.dumps(session.state, indent=1, ensure_ascii=False))
print("draft's system prompt was:", repr(draft.llm.calls[0]["messages"][0]["content"]))

# %% [markdown]
# ## 2. Parallel: fan out on branch copies, merge through state
#
# A `ParallelAgent` gives each child a **branch copy** of the session (`session.branch(name)`), runs them concurrently, then
# copies changed state keys back to the parent and records a `note` with `merged_state_keys`. Children never touch the same
# session object — no lost updates — and because all branches share one `InvocationContext`, they share **one budget**:
# three branches cannot spend three budgets.

# %%
checks = ParallelAgent("kyc_checks", [
    LlmAgent("sanctions", scripted("clear"), "Screen the applicant against sanctions lists.", output_key="sanctions"),
    LlmAgent("credit", scripted("score 712 — approve"), "Pull the applicant's credit summary.", output_key="credit"),
    LlmAgent("identity", scripted("document verified"), "Verify the identity document.", output_key="identity"),
])
par_session = new_session("Onboard applicant A-77")
par_ctx = InvocationContext(session=par_session)
par_events = await checks.run_to_completion(par_ctx)
print("merged state:", par_session.state)
print("note:", [e.payload for e in par_events if e.kind == "note"])
print("one budget across all three branches:", par_ctx.budget.summary())

# %% [markdown]
# ## 3. Loop: iterate until code says stop
#
# A `LoopAgent` repeats its children until `until(session)` is true or `max_iterations` is reached. **The exit criterion is
# code, not prompt** — a critic model may say "approved" a dozen ways, the predicate decides what counts — and
# `max_iterations` guarantees termination whatever the models do. Each iteration also stamps `temp:loop_iteration` into
# state, which the Runner clears at the end of the turn.

# %%
calls_seen = {"writer": 0, "critic": 0}


def writer_policy(messages, tools):
    calls_seen["writer"] += 1
    return text(f"Draft v{calls_seen['writer']}: we are sorry and will refund you in full.")


def critic_policy(messages, tools):
    calls_seen["critic"] += 1
    return text("APPROVED" if calls_seen["critic"] >= 3 else f"REVISE: draft {calls_seen['critic']} is too vague")


writer = LlmAgent("writer", FakeLLM(policy=writer_policy), "Write or revise the reply. Latest review: {verdict}", output_key="draft")
critic = LlmAgent("critic", FakeLLM(policy=critic_policy), "Review this draft: {draft}. Answer APPROVED or REVISE: <reason>.", output_key="verdict")
review_loop = LoopAgent("review", [writer, critic], max_iterations=5, until=lambda s: s.state.get("verdict") == "APPROVED")

loop_session = new_session("Please reply to this complaint.")
loop_session.state["verdict"] = "(no review yet)"
await review_loop.run_to_completion(InvocationContext(session=loop_session, budget=Budget(max_steps=20)))
print("final draft:", loop_session.state["draft"])
print("exit:", [e.payload for e in loop_session.events if e.kind == "note"][-1])

# %% [markdown]
# ## 4. Delegation: an agent as a tool
#
# `sub_agents=[...]` wraps each child in an `AgentTool`: the coordinator sees a tool named after the child whose only
# argument is `request`, and whose description is the child's `description`. The child runs in **its own child session**
# (`"<parent id>/<child name>"`), so its transcript never enters the parent's context — only the final answer comes back,
# as a tool result — and the parent log gets a `delegation` event with the request, the answer and how many model steps the
# child took.

# %%
billing = LlmAgent("billing", scripted("The September invoice is SGD 42.10, due on the 15th."), "You answer billing questions.",
                   description="Invoices, balances and payment dates")
cards = LlmAgent("cards", scripted("Card ending 1234 is blocked; a replacement ships in 3 days."), "You handle card blocks and replacements.",
                 description="Card blocks and replacements")
coordinator = LlmAgent("coordinator",
                       scripted(call("cards", request="Block the card ending 1234"),
                                call("billing", request="What does the customer owe this month?"),
                                "Card 1234 is blocked (replacement in 3 days) and your September invoice is SGD 42.10."),
                       "Route each request to the right specialist, then answer the user.", sub_agents=[billing, cards])

del_session = new_session("Block my card ending 1234 and tell me what I owe.", session_id="s-77")
await coordinator.run_to_completion(InvocationContext(session=del_session))
print("tools the coordinator was offered:", coordinator.llm.calls[0]["tools"])
for e in del_session.events:
    if e.kind == "delegation":
        print(f"delegation → {e.payload['child_session']:14s} steps={e.payload['steps']}  answer={e.payload['answer'][:48]!r}")
print("what the coordinator saw come back:", [m["content"][:60] for m in coordinator.llm.calls[2]["messages"] if m["role"] == "tool"])
print("final:", del_session.last_final_text())

# %% [markdown]
# ### What delegation costs
#
# Every hop is at least one extra model call, and the child re-reads its own system prompt plus the request. Compare the
# same task done by one agent with one tool against a coordinator with one specialist. Two counters matter:
# `session.usage()` sums only the events **in that session** — the specialist's calls live in its child session — while
# the shared `Budget` counts every model call in the invocation, so it is the honest number.

# %%
blocked: list[str] = []


@tool(side_effect=SideEffect.REVERSIBLE)
def block_card(card_id: str) -> dict:
    """Block a card; it can be unblocked."""
    blocked.append(card_id)
    return {"blocked": card_id}


TASK = "Block my card 1234."


async def run_single_agent() -> InvocationContext:
    llm = scripted(call("block_card", card_id="1234"), "Card 1234 is blocked.")
    agent = LlmAgent("assistant", llm, "You are a bank assistant.", tools=[block_card])
    ctx = InvocationContext(session=new_session(TASK, "single"))
    await agent.run_to_completion(ctx)
    return ctx


async def run_coordinator() -> InvocationContext:
    specialist = LlmAgent("cards", scripted(call("block_card", card_id="1234"), "Card 1234 is blocked."),
                          "You are the cards specialist.", tools=[block_card], description="Card blocks")
    coord = LlmAgent("coordinator", scripted(call("cards", request="block card 1234"), "Card 1234 is blocked."),
                     "Route to specialists.", sub_agents=[specialist])
    ctx = InvocationContext(session=new_session(TASK, "multi"))
    await coord.run_to_completion(ctx)
    return ctx


for label, ctx in (("single agent", await run_single_agent()), ("coordinator + 1", await run_coordinator())):
    print(f"{label:16s} model calls={ctx.budget.steps_used}  tokens={ctx.budget.tokens_used:4d}   "
          f"(session.usage() alone sees {ctx.session.usage().total} tokens)   final={ctx.session.last_final_text()!r}")

# %% [markdown]
# ## 5. Compounded reliability
#
# A chain of agents is a chain of probabilities. If each hop does the right thing with probability *p*, the chain succeeds
# with *p^N* — 95% per hop looks fine until five hops make it 77%. Simulate it before you believe it.

# %%
def simulate_chain(p: float, hops: int, trials: int = 20_000, seed: int = 7) -> float:
    rng = random.Random(seed)
    successes = sum(all(rng.random() < p for _ in range(hops)) for _ in range(trials))
    return successes / trials


p, hops = 0.95, 5
print(f"p={p} hops={hops}: empirical {simulate_chain(p, hops):.3f}  vs  p**N = {p ** hops:.3f}")
for n in (1, 2, 3, 5, 8):
    print(f"  {n} hops → {0.95 ** n:4.0%} at p=0.95   {0.99 ** n:4.0%} at p=0.99")

# %% [markdown]
# ### When *not* to go multi-agent
#
# Before adding a second LLM agent, check the list. If none applies, the extra agent is cost and failure surface with no
# return:
#
# - [ ] **The control flow is already known.** Then it is a workflow (`Sequential`/`Parallel`/`Loop`), not a second agent.
# - [ ] **It is really a tool.** A deterministic function with a schema is cheaper, testable and cannot hallucinate.
# - [ ] **The sub-task fits in the parent's context.** Splitting context only pays when the parent's window is the constraint.
# - [ ] **Same tools, same permissions.** A specialist that shares the parent's tool set and identity isolates nothing.
# - [ ] **Latency budget is tight.** Each hop adds a model call; hand-offs serialise unless the work is truly parallel.
# - [ ] **You cannot evaluate the parts separately.** If there is no per-agent golden set, you cannot tell which agent broke.
#
# What *does* justify an agent: a genuinely different context (long documents, noisy tool output), a different tool or
# permission boundary, independent work that can run in parallel, or a different model/owner with its own evals.

# %% [markdown]
# ## 6. Exercises
#
# ### Exercise 6.1 — a three-stage pipeline wired through state
#
# Build `build_pipeline(extract_llm, enrich_llm, respond_llm)` returning a `SequentialAgent` of three `LlmAgent`s:
#
# | stage name | `output_key` | its instruction must contain |
# |---|---|---|
# | `extract` | `order_id` | — |
# | `enrich`  | `order_facts` | `{order_id}` |
# | `respond` | `reply` | `{order_id}` **and** `{order_facts}` |
#
# The check runs it with scripted models, then inspects `respond_llm.calls[0]` to prove that stage 3's system prompt really
# contained what stages 1 and 2 produced — the state hand-off is the contract, not the shared transcript.

# %% exercise
def build_pipeline(extract_llm, enrich_llm, respond_llm) -> SequentialAgent:
    ### BEGIN SOLUTION
    extract = LlmAgent("extract", extract_llm, "Extract the order id from the user's message; reply with the id only.",
                       output_key="order_id")
    enrich = LlmAgent("enrich", enrich_llm, "Order {order_id}: summarise its shipping facts in one line.",
                      output_key="order_facts")
    respond = LlmAgent("respond", respond_llm,
                       "Order {order_id}. Facts: {order_facts}. Write a friendly two-sentence status update.",
                       output_key="reply")
    return SequentialAgent("order_status", [extract, enrich, respond])
    ### END SOLUTION

# %% check
ex_llm = scripted("ORD-10442")
en_llm = scripted("shipped 2026-09-01 via SpeedPost, ETA 2026-09-04")
re_llm = scripted("Good news: ORD-10442 shipped on 1 September via SpeedPost and should arrive by the 4th.")
pipe = build_pipeline(ex_llm, en_llm, re_llm)
assert isinstance(pipe, SequentialAgent) and [a.name for a in pipe.sub_agents] == ["extract", "enrich", "respond"], pipe
p_session = new_session("Where is my order ORD-10442?")
await pipe.run_to_completion(InvocationContext(session=p_session))
assert p_session.state.get("order_id") == "ORD-10442", p_session.state
assert p_session.state.get("order_facts", "").startswith("shipped 2026-09-01"), p_session.state
assert p_session.state.get("reply", "").startswith("Good news"), p_session.state
enrich_prompt = en_llm.calls[0]["messages"][0]["content"]
respond_prompt = re_llm.calls[0]["messages"][0]["content"]
assert en_llm.calls[0]["messages"][0]["role"] == "system" and "ORD-10442" in enrich_prompt, enrich_prompt
assert "ORD-10442" in respond_prompt and "SpeedPost" in respond_prompt, respond_prompt
assert "{order_id}" not in respond_prompt and "{order_facts}" not in respond_prompt, "placeholders were not filled"
print("✅ stage 3's system prompt:", repr(respond_prompt[:95]))

# %% [markdown]
# ### Exercise 6.2 — the exit criterion lives in code
#
# Write `approved(session) -> bool` — true when the critic's `output_key` (`verdict`) equals `"APPROVED"` — and
# `build_review_loop(writer_llm, critic_llm, max_iterations=5)` returning a `LoopAgent` over a `writer` (output_key `draft`)
# and a `critic` (output_key `verdict`) that exits on your predicate. The check runs two critics: one that approves on the
# third pass, one that never approves.

# %% exercise
def approved(session: Session) -> bool:
    ### BEGIN SOLUTION
    return session.state.get("verdict") == "APPROVED"
    ### END SOLUTION


def build_review_loop(writer_llm, critic_llm, max_iterations: int = 5) -> LoopAgent:
    ### BEGIN SOLUTION
    writer = LlmAgent("writer", writer_llm, "Write or revise the reply. Latest review: {verdict}", output_key="draft")
    critic = LlmAgent("critic", critic_llm, "Review this draft: {draft}. Answer APPROVED or REVISE: <reason>.", output_key="verdict")
    return LoopAgent("review", [writer, critic], max_iterations=max_iterations, until=approved)
    ### END SOLUTION

# %% check
def make_critic(approve_on):
    n = {"calls": 0}

    def policy(messages, tools):
        n["calls"] += 1
        return text("APPROVED" if approve_on is not None and n["calls"] >= approve_on else "REVISE: tighten the wording")
    return FakeLLM(policy=policy)


assert approved(Session(id="t", state={"verdict": "APPROVED"})) is True
assert approved(Session(id="t", state={"verdict": "REVISE: too long"})) is False and approved(Session(id="t")) is False
writer_llm = FakeLLM(policy=lambda m, t: text("Draft: we are sorry and will refund you."))
critic_llm = make_critic(approve_on=3)
loop = build_review_loop(writer_llm, critic_llm, max_iterations=5)
assert isinstance(loop, LoopAgent) and callable(loop.until) and loop.max_iterations == 5
assert [a.output_key for a in loop.sub_agents] == ["draft", "verdict"], [a.output_key for a in loop.sub_agents]
s_ok = new_session("reply to the complaint")
await loop.run_to_completion(InvocationContext(session=s_ok, budget=Budget(max_steps=30)))
exit_note = [e.payload for e in s_ok.events if e.kind == "note" and "loop_exit" in e.payload][-1]
assert exit_note == {"loop_exit": "condition met", "iterations": 3}, exit_note
assert critic_llm.call_count == 3 and s_ok.state["verdict"] == "APPROVED"
never = build_review_loop(FakeLLM(policy=lambda m, t: text("Draft.")), make_critic(approve_on=None), max_iterations=4)
s_never = new_session("reply to the complaint")
await never.run_to_completion(InvocationContext(session=s_never, budget=Budget(max_steps=30)))
exit_never = [e.payload for e in s_never.events if e.kind == "note" and "loop_exit" in e.payload][-1]
assert exit_never == {"loop_exit": "max_iterations", "iterations": 4}, exit_never
print("✅ exits: condition met after 3 iterations; max_iterations after 4 when the critic never approves")

# %% [markdown]
# ### Exercise 6.3 — a coordinator and its child sessions
#
# Write `build_coordinator(coordinator_llm, billing_llm, cards_llm) -> LlmAgent` whose specialists are named exactly
# `billing` and `cards`; give each a one-line `description` — that text becomes the tool description the coordinator reads
# when deciding whom to call. The check scripts the coordinator to delegate to both in one turn and asserts two `delegation`
# events whose `child_session` ids are `"<session id>/billing"` and `"<session id>/cards"`.

# %% exercise
def build_coordinator(coordinator_llm, billing_llm, cards_llm) -> LlmAgent:
    ### BEGIN SOLUTION
    billing = LlmAgent("billing", billing_llm, "You answer billing questions.", description="Invoices, balances and payment dates")
    cards = LlmAgent("cards", cards_llm, "You handle card blocks and replacements.", description="Card blocks and replacements")
    return LlmAgent("coordinator", coordinator_llm, "Route each part of the request to the right specialist, then answer.",
                    sub_agents=[billing, cards])
    ### END SOLUTION

# %% check
c_llm = scripted(calls(call("billing", request="what does the customer owe?"), call("cards", request="block card 1234")),
                 "Card 1234 is blocked; you owe SGD 42.10.")
coord = build_coordinator(c_llm, scripted("SGD 42.10, due on the 15th."), scripted("Card 1234 is blocked."))
assert isinstance(coord, LlmAgent) and sorted(coord.registry.names()) == ["billing", "cards"], coord.registry.names()
assert all(coord.registry.get(n).spec.description.strip() for n in ("billing", "cards")), "give the specialists a description"
c_session = new_session("Block card 1234 and tell me what I owe", session_id="s-42")
await coord.run_to_completion(InvocationContext(session=c_session))
deleg = [e for e in c_session.events if e.kind == "delegation"]
assert len(deleg) == 2, [e.kind for e in c_session.events]
assert sorted(e.payload["child_session"] for e in deleg) == ["s-42/billing", "s-42/cards"], [e.payload for e in deleg]
assert all(e.payload["steps"] == 1 for e in deleg), "each specialist answered in one model call"
assert c_session.last_final_text().startswith("Card 1234 is blocked"), c_session.last_final_text()
assert sorted(c_llm.calls[0]["tools"]) == ["billing", "cards"], c_llm.calls[0]["tools"]
print("✅ delegations:", [(e.payload["child_session"], e.payload["answer"]) for e in deleg])

# %% [markdown]
# ### Exercise 6.4 — reliability arithmetic
#
# Implement `compounded_reliability(p, hops)` (the probability that every one of `hops` steps succeeds) and
# `hops_allowed(p, target, max_hops=100)` — the largest number of hops for which the end-to-end success rate is still
# **at least** `target` (0 if even one hop falls short; capped at `max_hops`). Prefer a loop over `log` division: the
# floating-point boundary cases bite.

# %% exercise
def compounded_reliability(p: float, hops: int) -> float:
    ### BEGIN SOLUTION
    return p ** hops
    ### END SOLUTION


def hops_allowed(p: float, target: float, max_hops: int = 100) -> int:
    ### BEGIN SOLUTION
    n = 0
    while n < max_hops and compounded_reliability(p, n + 1) >= target:
        n += 1
    return n
    ### END SOLUTION

# %% check
assert abs(compounded_reliability(0.95, 5) - 0.7738) < 1e-3, compounded_reliability(0.95, 5)
assert compounded_reliability(0.99, 1) == 0.99 and compounded_reliability(0.5, 0) == 1.0
assert hops_allowed(0.95, 0.90) == 2, hops_allowed(0.95, 0.90)
assert hops_allowed(0.99, 0.95) == 5, hops_allowed(0.99, 0.95)
assert hops_allowed(0.99, 0.90) == 10, hops_allowed(0.99, 0.90)
assert hops_allowed(0.50, 0.90) == 0, hops_allowed(0.50, 0.90)
assert hops_allowed(1.0, 0.99) == 100, hops_allowed(1.0, 0.99)
emp = simulate_chain(0.95, 5)
assert abs(emp - compounded_reliability(0.95, 5)) < 0.02, (emp, compounded_reliability(0.95, 5))
print(f"✅ for a 95% end-to-end target you can afford {hops_allowed(0.99, 0.95)} hops at p=0.99 but only "
      f"{hops_allowed(0.95, 0.95)} at p=0.95")

# %% [markdown]
# ### Exercise 6.5 — simplicity as a decision
#
# `run_claims_multi()` below answers a claims question with a coordinator and two specialists (`policy` owns `get_policy`,
# `payout` owns `estimate_payout`). Rewrite it as **one** agent with both tools: implement `build_claims_single()` returning
# an `LlmAgent` whose scripted model calls both tools in one turn and then answers with `CLAIM_ANSWER`. The check runs both
# designs on the same task and asserts the single agent calls the same tools with fewer model calls **and** fewer tokens.

# %% exercise
@tool
def get_policy(policy_id: str) -> dict:
    """Coverage details of a policy."""
    return {"policy_id": policy_id, "covers": ["water damage"], "excess": 500}


@tool
def estimate_payout(claim_amount: float, excess: float) -> dict:
    """Payout after the excess is deducted."""
    return {"payout": max(0.0, claim_amount - excess)}


CLAIM_TASK = "Policy P-9: is water damage covered, and what would I get on a SGD 2,000 claim?"
CLAIM_ANSWER = "Water damage is covered under P-9; after the SGD 500 excess you would receive SGD 1,500."


async def run_claims_multi() -> InvocationContext:
    policy_agent = LlmAgent("policy", scripted(call("get_policy", policy_id="P-9"), "P-9 covers water damage; the excess is SGD 500."),
                            "You are the policy specialist.", tools=[get_policy], description="Policy coverage questions")
    payout_agent = LlmAgent("payout", scripted(call("estimate_payout", claim_amount=2000.0, excess=500.0), "The payout would be SGD 1,500."),
                            "You are the payout specialist.", tools=[estimate_payout], description="Payout estimates")
    coord = LlmAgent("coordinator",
                     scripted(call("policy", request="Is water damage covered under P-9, and what is the excess?"),
                              call("payout", request="Payout on a SGD 2,000 claim with a SGD 500 excess"),
                              CLAIM_ANSWER),
                     "Route to specialists, then answer.", sub_agents=[policy_agent, payout_agent])
    ctx = InvocationContext(session=new_session(CLAIM_TASK, "claims-multi"))
    await coord.run_to_completion(ctx)
    return ctx


def build_claims_single() -> LlmAgent:
    ### BEGIN SOLUTION
    llm = scripted(calls(call("get_policy", policy_id="P-9"), call("estimate_payout", claim_amount=2000.0, excess=500.0)),
                   CLAIM_ANSWER)
    return LlmAgent("claims", llm, "You answer policy coverage and payout questions.", tools=[get_policy, estimate_payout])
    ### END SOLUTION

# %% check
multi = await run_claims_multi()
single_agent = build_claims_single()
assert isinstance(single_agent, LlmAgent) and sorted(single_agent.registry.names()) == ["estimate_payout", "get_policy"], single_agent.registry.names()
single = InvocationContext(session=new_session(CLAIM_TASK, "claims-single"))
await single_agent.run_to_completion(single)
assert single.session.last_final_text() == CLAIM_ANSWER, single.session.last_final_text()
single_tools = sorted(e.payload["name"] for e in single.session.events if e.kind == "tool_call")
assert single_tools == ["estimate_payout", "get_policy"], single_tools
assert all(e.payload["ok"] for e in single.session.events if e.kind == "tool_result"), "both tool calls must succeed"
assert single.budget.steps_used < multi.budget.steps_used, (single.budget.steps_used, multi.budget.steps_used)
assert single.budget.tokens_used < multi.budget.tokens_used, (single.budget.tokens_used, multi.budget.tokens_used)
print(f"✅ single agent: {single.budget.steps_used} model calls / {single.budget.tokens_used} tokens   "
      f"vs coordinator design: {multi.budget.steps_used} model calls / {multi.budget.tokens_used} tokens — same answer, same tools")

# %% [markdown]
# ### Exercise 6.6 — when does another agent earn its keep?
#
# Fill `conditions_for_another_agent` with **three** one-sentence conditions under which adding an agent is the right call
# (think: context isolation, a different tool or permission boundary, genuinely parallel work, a different model or owner).
# The check looks for three real sentences that touch at least two of those ideas.

# %% exercise
### BEGIN SOLUTION
conditions_for_another_agent = [
    "The sub-task needs a large or noisy context (long documents, big tool outputs) that would crowd out the parent's context window.",
    "The work sits behind a different tool set or permission boundary, so isolating it limits the blast radius of a mistake.",
    "Several sub-tasks are independent and can run in parallel, cutting wall-clock latency instead of adding a serial hop.",
]
### END SOLUTION

# %% check
assert isinstance(conditions_for_another_agent, list) and len(conditions_for_another_agent) == 3, "exactly three conditions"
assert all(isinstance(c, str) and len(c.split()) >= 8 for c in conditions_for_another_agent), "write real sentences"
_joined = " ".join(conditions_for_another_agent).lower()
_ideas = {"context isolation": ("context", "window"),
          "tool/permission boundary": ("permission", "tool set", "toolset", "scope", "boundary", "blast radius"),
          "parallel work": ("parallel", "latency", "concurrent"),
          "different model/owner": ("different model", "cheaper model", "team", "owner", "specialis")}
hits = [name for name, words in _ideas.items() if any(w in _joined for w in words)]
assert len(hits) >= 2, f"touch at least two of the ideas; found {hits}"
print("✅ conditions cover:", ", ".join(hits))

# %% [markdown]
# ## The one-minute version
#
# Start from the single agent and justify every addition. *"I begin with one agent and a small tool set. The moment the
# control flow is known — classify then draft, run three checks at once, revise until a checker passes — I move it into a
# workflow agent so the sequence is code, not prompt. I add a second LLM agent only when a sub-task needs its own context,
# its own permissions, or genuinely parallel work — and I budget for it: each hop is another model call, the child cannot see
# the parent's context, and reliability compounds as p^N (95% per hop is 77% after five). The budget is shared across the
# tree so delegation can't escape it, and the coordinator's session records delegation events so I can trace who decided
# what."* That framing — simplicity as a decision, backed by numbers — is what separates an architect from a framework user.
