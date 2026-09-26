# %% [markdown]
# # 02 · A thinking model on one GPU: size it, switch it, parse it, budget it
#
# **Tier:** T1 — `Qwen/Qwen3-0.6B` (or 1.7B) served by vLLM v0.30.0 with `--reasoning-parser qwen3` on
# a free Colab/Kaggle T4. Start it with [`deploy/any-gpu/serve.sh`](../deploy/any-gpu/serve.sh), then
# `export THINKLAB_URL=http://127.0.0.1:8000` and every cell below measures the real model. T0 (the
# default) uses bundled responses in vLLM's documented format (illustrative) and an in-process fake
# server whose model and timing are **simulated**.
#
# ## The one-minute version
#
# * A "thinking model" is ordinary weights plus four conventions. The **chat template** opens a
#   `<think>` block; a server-side **reasoning parser** splits the output into `reasoning` and
#   `content`; the **sampling settings** differ (Qwen3: T=0.6, top-p 0.95 when thinking, never
#   greedy); and a **switch** (`chat_template_kwargs.enable_thinking` for Qwen3) turns it off.
# * vLLM v0.30.0 returns `message.reasoning` / `delta.reasoning`. The field used to be
#   `reasoning_content`, which SGLang and the DeepSeek API still use. A client that reads only one
#   of them silently gets nothing from the other kind of server.
# * `max_tokens` counts reasoning tokens. It is not a thinking budget: set too low, it returns
#   `content: null` with `finish_reason: "length"`. Budgets that end the thinking and still get an
#   answer are `thinking_token_budget` (vLLM) or Qwen's two-call recipe.
# * Thinking tokens are output tokens: billed as output, decoded one step at a time, and held in
#   KV. On Qwen3-0.6B an 8K-token trace holds about as many bytes of KV as the whole model's weights.
#
# Concepts: PRIMER §5 "Thinking models" and §7 "What thinking does to serving" ([`PRIMER.md`](../../PRIMER.md));
# sampling knobs in the 04 serving-engine PRIMER §6 "Sampling and structured output".

# %%
import json
from importlib import resources
from thinklab import env, parsers
from thinklab.report import table
from thinklab.thinking import budget as B
from thinklab.thinking.client import SAMPLING, ThinkingClient, request_body
from thinklab.thinking.evalset import make_evalset, verify
from thinklab.thinking.ttc import wilson_interval

print(env.describe())
target = env.connect(time_scale=0.02)        # THINKLAB_URL -> real server; else the fake one, 50x faster than simulated time
print(target)
client = ThinkingClient(target.url, headers=target.headers)
LABEL = target.label
print("model:", client.model, "|", LABEL)
SAMPLES = resources.files("thinklab.data").joinpath("samples")

# %% [markdown]
# ## Worked example: size it before you serve it
#
# Weights = parameters × 2 bytes (fp16 on a T4, which has no bf16). KV per token = 2 (K and V) ×
# layers × KV heads × head_dim × 2 bytes. The parameter counts and T4 block predictions below are
# the 04 lab's `servelab.sizing` numbers for these configs (`tests/test_reuse.py` reproduces them
# from that lab when it is present). Always read `head_dim` from `config.json`: Qwen3-0.6B's is
# 128, not hidden / heads = 64.

# %%
QWEN3 = {   # layers, KV heads, head_dim (config.json); params from servelab.sizing.param_count
    "Qwen3-0.6B": (28, 8, 128, 596_049_920), "Qwen3-1.7B": (28, 8, 128, 1_720_574_976),
    "Qwen3-4B": (36, 8, 128, 4_022_468_096), "R1-Distill-Qwen-1.5B": (28, 2, 128, 1_777_088_000)}
rows = []
for name, (L, kv, hd, params) in QWEN3.items():
    per_tok = 2 * L * kv * hd * 2
    rows.append({"model": name, "weights GB (fp16)": round(params * 2 / 1e9, 2), "KV B/token": per_tok,
                 "KV of one 8K trace GB": round(per_tok * 8192 / 1e9, 2)})
print(table(rows, title="Thinking-model sizing (arithmetic)"))
print("T4 predictions (servelab.sizing, gpu_memory_utilization 0.92, max_model_len 8192): Qwen3-0.6B 6,969 blocks x 16 ="
      " 111,504 KV tokens -> 13.6 concurrent 8K requests; Qwen3-1.7B 5,721 blocks -> 11.2 (predicted; calibrate on the startup log)")

# %% [markdown]
# ## Exercise 2.1 — the KV of a thinking trace
#
# Write `kv_bytes(layers, kv_heads, head_dim, tokens, bytes_per_elem=2)`. Then answer the design
# question: how many *full* 8K-token thinking traces of Qwen3-0.6B fit in the 111,504 KV-token
# slots the T4 has left after weights?

# %% exercise
def kv_bytes(layers: int, kv_heads: int, head_dim: int, tokens: int, bytes_per_elem: int = 2) -> int:
    ### BEGIN SOLUTION
    return 2 * layers * kv_heads * head_dim * bytes_per_elem * tokens
    ### END SOLUTION

traces_that_fit = None
### BEGIN SOLUTION
traces_that_fit = 111_504 // 8192
### END SOLUTION

# %% check
assert kv_bytes(28, 8, 128, 1) == 114_688 and kv_bytes(36, 8, 128, 1) == 147_456
assert kv_bytes(28, 8, 128, 8192) == 939_524_096           # 0.94 GB, next to 1.19 GB of weights
assert kv_bytes(28, 2, 128, 1) == 28_672                   # R1-Distill-1.5B: 2 KV heads -> 4x less per token
assert traces_that_fit == 13
print("✅ 112 KiB per token; an 8K trace = 0.94 GB of KV for a 1.19 GB model; 13 whole traces fit a T4")

# %% [markdown]
# ## Worked example: what comes back
#
# The same question answered with thinking on and off, as vLLM v0.30.0 returns it, and the SGLang
# spelling. These files are sample output in the documented format (illustrative). The field
# names are real; the text was written for this lab.

# %%
on = json.loads(SAMPLES.joinpath("vllm_chat_qwen3_thinking.json").read_text())
off = json.loads(SAMPLES.joinpath("vllm_chat_qwen3_nothinking.json").read_text())
sg = json.loads(SAMPLES.joinpath("sglang_chat_reasoning_content.json").read_text())
for name, d in (("vLLM thinking on", on), ("vLLM thinking off", off), ("SGLang", sg)):
    m, u = d["choices"][0]["message"], d["usage"]
    print(f"{name:17} fields={sorted(k for k in m if k != 'role')}")
    print(f"{'':17} reasoning={str(parsers.reasoning_of(m))[:70]!r}...")
    print(f"{'':17} content={str(m['content'])[:60]!r} | completion {u['completion_tokens']} tokens, "
          f"reasoning {(u.get('completion_tokens_details') or {}).get('reasoning_tokens', 'n/a')}")

# %% [markdown]
# Two things to notice. `completion_tokens` includes the reasoning (132 = 103 reasoning + 29
# answer): that is what you pay for and what the engine decodes. And the no-thinking answer is
# *wrong* (281); in these illustrative samples, as on real models, thinking buys accuracy on
# multi-step arithmetic.
#
# ## Exercise 2.2 — parse raw output the way `--reasoning-parser deepseek_r1` does
#
# Without a server-side parser (for example `/v1/completions`), you split the text yourself.
# DeepSeek-R1-style templates put `<think>\n` into the *prompt*, so the output usually starts
# mid-thought and contains only `</think>`. Return `(reasoning, content)`:
#
# * drop an opening `<think>` if present;
# * no `</think>` at all: everything is reasoning and content is `None` (the model was cut off
#   while thinking);
# * otherwise reasoning is what precedes the first `</think>` and content what follows (`None` if
#   empty).

# %% exercise
def split_r1(text: str) -> tuple:
    ### BEGIN SOLUTION
    before, sep, after = text.partition("<think>")
    body = after if sep else before
    if "</think>" not in body:
        return body, None
    reasoning, _, content = body.partition("</think>")
    return reasoning, content or None
    ### END SOLUTION

# %% check
raw = SAMPLES.joinpath("deepseek_r1_raw.txt").read_text()
r, c = split_r1(raw)
assert r.startswith("Okay") and "\\boxed{4}" in c and "</think>" not in r + c
assert split_r1("<think>a</think>b") == ("a", "b") and split_r1("still thinking") == ("still thinking", None)
assert split_r1("x</think>") == ("x", None)
for t in (raw, "<think>a</think>b", "no end", "<think>\n\n</think>\n\nhi"):
    s = parsers.split_deepseek_r1(t)
    assert split_r1(t) == (s.reasoning, s.content)
print("✅ R1 semantics: no opening tag needed; no closing tag = all reasoning")

# %% [markdown]
# ## Exercise 2.3 — follow a stream: when does the *answer* start?
#
# A streamed response sends `delta.reasoning` chunks, then `delta.content` chunks. For a user
# waiting for an answer, the moment that matters is the first *content* chunk, not the first
# token. Given the parsed chunk payloads of the bundled stream, return the full reasoning text, the
# full content text, and the 0-based index (among chunks that carry any text) of the first content
# chunk. Read the reasoning under either field name.

# %% exercise
def follow_stream(chunks: list) -> tuple:
    ### BEGIN SOLUTION
    reasoning, content, first_content, i = [], [], None, 0
    for ch in chunks:
        for c in ch.get("choices") or []:
            d = c.get("delta") or {}
            r = d.get("reasoning") or d.get("reasoning_content")
            t = d.get("content")
            if not r and not t:
                continue
            if r:
                reasoning.append(r)
            if t:
                content.append(t)
                if first_content is None:
                    first_content = i
            i += 1
    return "".join(reasoning), "".join(content), first_content
    ### END SOLUTION

# %% check
events = [json.loads(l[5:]) for l in SAMPLES.joinpath("vllm_stream_qwen3.sse").read_text().splitlines()
          if l.startswith("data:") and l[5:].strip() != "[DONE]"]
R, C, k = follow_stream(events)
assert R.startswith("Okay, 17 * 23") and C == "The answer is \\boxed{291}." and k == 14
renamed = [json.loads(json.dumps(e).replace('"reasoning"', '"reasoning_content"')) for e in events]
assert follow_stream(renamed) == (R, C, k)
print(f"✅ 14 reasoning chunks before the answer starts at chunk {k}; works for both field names")

# %% [markdown]
# ## Worked example: thinking on vs off on the eval set
#
# Twenty generated problems (arithmetic, weekday, ordering and counting, each checked by a
# program), with the model card's sampling for each mode. Against the fake server, the model's
# answers are **simulated**: it knows the right answer and gives it with a probability that grows
# with its thinking. Against a real server this is a measurement. Accuracy on 20 problems carries a
# wide interval, so report it.

# %%
problems = make_evalset(20, seed=0)
results = {}
for mode, kw in (("thinking off", {"thinking": False}), ("thinking on", {"thinking": True})):
    comps = client.chat_many([p.messages() for p in problems], max_tokens=6000, **kw)
    ok = [verify(p, c.content) for p, c in zip(problems, comps)]
    lo, hi = wilson_interval(sum(ok), len(ok))
    results[mode] = comps
    print(f"[{LABEL}] {mode:13} accuracy {sum(ok)}/{len(ok)} (95% CI {lo:.2f}-{hi:.2f}) | mean reasoning "
          f"{sum(c.reasoning_tokens or 0 for c in comps) / len(comps):.0f} tokens | mean output "
          f"{sum(c.completion_tokens for c in comps) / len(comps):.0f} tokens | sampling {SAMPLING[kw['thinking']]}")

# %% [markdown]
# ## Exercise 2.4 — the request bodies for three switches
#
# Build the JSON bodies vLLM v0.30.0 expects (model `"m"`, one user message `msgs`) for:
#
# 1. `off`: thinking disabled through the chat template, with the non-thinking sampling
#    (temperature 0.7, top_p 0.8, top_k 20);
# 2. `budget`: thinking on with a 256-token thinking budget (vLLM's request field) and the
#    thinking sampling (0.6, 0.95, 20);
# 3. `effort_none`: `reasoning_effort` set to `"none"`, no other switch. For Qwen3 vLLM turns
#    this into `enable_thinking=False`. It is an on/off switch, not a length control.

# %% exercise
msgs = [{"role": "user", "content": "What is 17 * 23 - 100?"}]
### BEGIN SOLUTION
off_body = {"model": "m", "messages": msgs, "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.7, "top_p": 0.8, "top_k": 20}
budget_body = {"model": "m", "messages": msgs, "thinking_token_budget": 256, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
effort_none_body = {"model": "m", "messages": msgs, "reasoning_effort": "none"}
### END SOLUTION

# %% check
assert off_body == request_body(msgs, model="m", thinking=False)
assert budget_body == request_body(msgs, model="m", budget=256)
assert effort_none_body["reasoning_effort"] == "none" and "chat_template_kwargs" not in effort_none_body
r = client.chat(msgs, max_tokens=4000, reasoning_effort="none", recommended=False)
assert r.ok and not r.reasoning, "reasoning_effort='none' should switch Qwen3-style thinking off"
print("✅ bodies match; reasoning_effort='none' returned no reasoning from", LABEL.lower(), "server")

# %% [markdown]
# ## Worked example: the `max_tokens` trap, and two budgets that work
#
# One hard problem, three ways to bound it at 64 thinking tokens.

# %%
hard = next(p for p in make_evalset(40, seed=3) if p.difficulty == 4)
print(hard.question)
for name, fn in (("max_tokens=64 (trap)", lambda: B.truncate(client, hard.messages(), 64)),
                 ("thinking_token_budget=64", lambda: B.native(client, hard.messages(), 64, 2000)),
                 ("two-call recipe, 64", lambda: B.two_call(client, hard.messages(), 64, 2000))):
    c = fn()
    print(f"[{LABEL}] {name:26} -> {B.classify(c):16} finish={c.finish_reason:6} reasoning={c.reasoning_tokens} "
          f"output={c.completion_tokens} answer={c.answer!r}")

# %% [markdown]
# The trap spends 64 tokens and returns no answer. Both budgets end the thinking and get an answer
# from what was thought so far. Qwen3's report says answering from truncated thinking "emerges
# naturally" rather than being trained, so its accuracy is a curve you measure (next exercise). The
# two-call recipe costs a second round trip and re-prefills the thinking. The native budget needs
# `--reasoning-parser` on the server (and optionally `--reasoning-config` for the phrase it forces
# before `</think>`).
#
# ## Exercise 2.5 — choose a budget from a sweep
#
# `rows` is an accuracy-vs-budget sweep (`budget=None` is unlimited thinking). Write
# `pick_budget(rows, tolerance)`: the *smallest* budget whose accuracy is within `tolerance`
# (absolute) of the unlimited row. Then report output tokens per correct answer for it and for
# unlimited thinking.

# %% exercise
def pick_budget(rows: list, tolerance: float = 0.05):
    ### BEGIN SOLUTION
    full = next(r for r in rows if r.budget is None)
    ok = [r for r in rows if r.budget is not None and r.accuracy >= full.accuracy - tolerance]
    return min(ok, key=lambda r: r.budget) if ok else full
    ### END SOLUTION

# %% check
sweep = B.sweep(client, make_evalset(30, seed=5), [None, 128, 256, 512, 1024, 2048], max_tokens=6000)
print(table([r.as_dict() for r in sweep], title=f"[{LABEL}] accuracy vs thinking budget (30 problems)"))
best = pick_budget(sweep, 0.05)
full = next(r for r in sweep if r.budget is None)
fake = [B.SweepRow("native", b, 30, a, 0, t, 0) for b, a, t in ((None, .7, 1000), (256, .5, 300), (512, .66, 500), (1024, .69, 800))]
assert pick_budget(fake, 0.05).budget == 512 and pick_budget(fake, 0.01).budget == 1024 and pick_budget(fake, 0.3).budget == 256
per_correct = B.expected_cost_tokens(sweep)
print(f"✅ smallest budget within 5 points of unlimited: {best.budget} "
      f"({per_correct[best.budget]:.0f} output tokens per correct answer vs {per_correct[None]:.0f} unlimited)")

# %% [markdown]
# ## On a real GPU (T1)
#
# ```bash
# # Colab / Kaggle T4 (fp16; vLLM picks its Triton attention backend on sm_75 by itself):
# pip install -q "vllm==0.30.0"
# vllm serve Qwen/Qwen3-0.6B --dtype half --max-model-len 8192 --gpu-memory-utilization 0.85 \
#   --reasoning-parser qwen3 \
#   --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.</think>"}' &
# export THINKLAB_URL=http://127.0.0.1:8000        # then re-run this notebook: every number is measured
# # DeepSeek-R1-Distill-Qwen-1.5B always thinks; no system prompt, temperature 0.6:
# vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --dtype half --max-model-len 8192 --reasoning-parser deepseek_r1
# ```
#
# Things to check on the real model: that fp16 outputs on a T4 are sane (Qwen3 is trained in bf16;
# verify there are no NaNs or garbage); that `usage.completion_tokens_details.reasoning_tokens` is
# filled (it needs the parser); how accuracy moves between thinking off and on and across budgets
# on *these* problems; and how much of the budget the model uses. `deploy/any-gpu/README.md` has the
# whole recipe, including Qwen3-1.7B and Qwen3-4B on a 24 GB card.

# %%
target.stop()

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "We serve a hybrid thinking model with vLLM's reasoning parser, so clients get
# `reasoning` and `content` as separate fields. Our client reads both field names, because SGLang
# and the DeepSeek API still call it `reasoning_content`. Thinking is a per-request product decision:
# Qwen3's `enable_thinking` switch in `chat_template_kwargs`, the model card's sampling for each mode,
# and never greedy decoding. We bound cost with `thinking_token_budget`, not `max_tokens`: a
# `max_tokens` cap cuts the answer off, returns `content: null` and still bills every token. On a T4,
# Qwen3-0.6B holds 111K KV tokens after its weights. An 8K-token trace takes 0.94 GB of KV, so a
# handful of long thinkers fills the cache. Budgets and length distributions are capacity inputs,
# not just quality knobs."
#
# **Drill 1.** *Our client shows empty reasoning since the vLLM upgrade. Why?* vLLM renamed the
# field from `reasoning_content` to `reasoning`, and the old attribute now reads as empty. Read
# `reasoning` first and fall back to `reasoning_content`.
#
# **Drill 2.** *Does `reasoning_effort="low"` make Qwen3 think less?* No. For Qwen3 vLLM maps any
# effort other than `"none"` to `enable_thinking=True`. Only templates that use the effort (gpt-oss:
# low/medium/high in its system prompt) change behaviour. Bound length with a budget instead.
#
# **Drill 3.** *We set `max_tokens=512` to cap cost and accuracy collapsed. What happened?* The
# requests that needed more than 512 tokens of thinking ended inside `<think>` with no answer
# (`finish_reason: "length"`, `content: null`). Use `thinking_token_budget` so the model is forced to
# answer, and measure the accuracy-vs-budget curve to choose the value.
