# %% [markdown]
# # 04 · Sampling and structured output
#
# **Tier:** T0 — CPU only, no network, a few seconds. The same parameters are fields of vLLM's `SamplingParams`
# and of its OpenAI-compatible API; `vllm-serving-lab` sends them to a real server (T1).
#
# ## The one-minute version
# The forward pass produces **logits** — one score per vocabulary entry. The sampler turns them into one token
# through a fixed pipeline: mask what is not allowed (structured output) → apply repetition/presence/frequency
# **penalties** → if `temperature == 0` take the argmax → divide by **temperature** → cut the tail with
# **min-p**, **top-k**, **top-p** → draw, with the request's **own seeded generator**. None of this changes the
# model; it reshapes one distribution per step. **Structured output** is the same mechanism with a grammar behind
# it: a finite-state machine says which tokens are legal next, the rest get `-inf`. The model never learns the
# schema; it is fenced in — which guarantees *syntax*, not *sense*.
#
# Primer: §6 *Sampling and structured output* (`../../PRIMER.md`); background:
# `00-foundations/transformers/docs/transformer-primer.md` §7.4.

# %%
import numpy as np

from minengine import ChoiceFSM, Engine, SamplingParams, TinyLM, decode, encode
from minengine.sampler import (EOS, VOCAB, apply_penalties, log_softmax, min_p_filter, probs, sample,
                               top_k_filter, top_p_filter)

model = TinyLM()


def show(p, k=6):
    top = np.argsort(-p)[:k]
    return "  ".join(f"{decode([t]) if t < 256 else 'EOS'!r}:{p[t]:.3f}" for t in top)

# %% [markdown]
# ## Worked example 1 — one distribution, several knobs

# %%
flat = model.forward_dense(encode("The engine "))[-1]            # many plausible next letters
peaked = model.forward_dense(encode("When memory runs ou"))[-1]   # three plausible next letters
for name, lg in [("flat  ", flat), ("peaked", peaked)]:
    print(name, show(probs(lg)))
    for T in [0.5, 1.0, 1.5]:
        q = probs(lg / T)
        print(f"   temperature {T}: top-1 {q.max():.2f}, entropy {-(q * np.log(q)).sum():.2f} nats")
    kept = lambda x: int(np.isfinite(x).sum())
    print(f"   tokens kept: top_k=5 -> {kept(top_k_filter(lg, 5))}, top_p=0.9 -> {kept(top_p_filter(lg, 0.9))}, "
          f"min_p=0.1 -> {kept(min_p_filter(lg, 0.1))}")

# %% [markdown]
# **top-k** keeps five tokens whatever the shape of the distribution. **top-p** and **min-p** adapt: many tokens
# when the model is unsure, few when it is confident — which is why they are the usual choice. Temperature
# changes the shape, not the ranking: the argmax is the same at every temperature.
#
# ## Worked example 2 — greedy loops, penalties break them

# %%
eng = Engine(model, num_blocks=64, block_size=16)
for label, p in [("greedy                    ", SamplingParams(max_tokens=40, temperature=0)),
                 ("greedy + repetition 1.3   ", SamplingParams(max_tokens=40, temperature=0, repetition_penalty=1.3)),
                 ("greedy + presence/freq 0.5", SamplingParams(max_tokens=40, temperature=0, presence_penalty=0.5,
                                                              frequency_penalty=0.5)),
                 ("temperature 0.8, top_p 0.9", SamplingParams(max_tokens=40, temperature=0.8, top_p=0.9, seed=1))]:
    print(label, repr(eng.generate(["The engine "], p)[0].text))

# %% [markdown]
# ## Worked example 3 — seeds: reproducible, whatever else is in the batch
# Each request gets its own random generator (seeded from `seed`, or from the engine's seed and the request's
# index). So a seeded request's output does not depend on which other requests shared its steps — the property
# you need to replay a production incident.

# %%
sp = SamplingParams(max_tokens=16, temperature=0.9, seed=7)
alone = Engine(model, num_blocks=64, block_size=4).generate(["The engine "], sp)[0]
crowd = Engine(model, num_blocks=64, block_size=4).generate(
    ["Each step", "The engine ", "When memory"], [SamplingParams(max_tokens=30), sp, SamplingParams(temperature=1.4)])[1]
other = Engine(model, num_blocks=64, block_size=4).generate(["The engine "], SamplingParams(max_tokens=16, temperature=0.9, seed=8))[0]
print("alone     :", repr(alone.text))
print("in a crowd:", repr(crowd.text), "- identical:", alone.text == crowd.text)
print("seed 8    :", repr(other.text))

# %% [markdown]
# (On a GPU, bitwise batch invariance is harder than this: batch size changes kernel choices and reduction order,
# so logits differ in the last bits. vLLM offers a batch-invariant mode for that, at some speed cost — verify.)
#
# ## Worked example 4 — logprobs are the model's, not the sampler's
# vLLM V1 returns logprobs of the **raw** logits by default (`logprobs_mode="raw_logprobs"`), before temperature,
# penalties or masks. They tell you what the model believed, even when the sampler overruled it.

# %%
o = Engine(model, num_blocks=64, block_size=4).generate(
    ["The engine "], SamplingParams(max_tokens=5, temperature=0.3, logprobs=3, seed=0))[0]
for t, (lp, top) in zip(o.token_ids, o.logprobs):
    print(f"sampled {decode([t])!r} logprob {lp:6.2f} | top-3 raw: " +
          ", ".join(f"{decode([k])!r} {v:.2f}" for k, v in top.items()))

# %% [markdown]
# ## Worked example 5 — structured output: a JSON answer from a model that has never seen JSON

# %%
fsm = ChoiceFSM(['{"answer": "yes"}', '{"answer": "no"}'])
print("allowed first tokens:", [decode([t]) for t in np.flatnonzero(fsm.allowed(fsm.start()))])
print("allowed after '{\"answer\": \"':", [decode([t]) for t in np.flatnonzero(fsm.allowed(b'{"answer": "'))])
eng = Engine(model, num_blocks=64, block_size=16)
outs = eng.generate(["Is the engine running? "] * 4, [SamplingParams(max_tokens=40, fsm=fsm, seed=s) for s in range(4)])
for o in outs:
    forced = sum(lp for lp, _ in o.logprobs)
    print(f"{o.text!r:22} {o.finish_reason:17} model's own log-probability of this text: {forced:7.1f}")

# %% [markdown]
# Every output is valid — and the model's own log-probability of it is astronomically low: it would never have
# produced `{` unprompted. The mask **forces** syntax. It cannot supply meaning: if the model does not know the
# answer, the grammar makes it say `"yes"` or `"no"` anyway. Real engines compile JSON schemas or regexes into
# token-level automata (xgrammar, llguidance, outlines — verify current vLLM backends); with a BPE vocabulary each
# token spans several characters, so computing the mask per step over ~100k tokens is the engineering problem.
#
# ## Exercise 4.1 — top-p (nucleus) filtering
# Keep the smallest set of highest-probability tokens whose total probability reaches `p`; set the rest to `-inf`.
# The top token is always kept.

# %% exercise
def my_top_p(logits, p):
    ### BEGIN SOLUTION
    order = np.argsort(-logits, kind="stable")
    pr = probs(logits)[order]
    keep = np.zeros(len(logits), bool)
    keep[order[(np.cumsum(pr) - pr) < p]] = True
    return np.where(keep, logits, -np.inf)
    ### END SOLUTION

# %% check
L = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
assert np.isfinite(my_top_p(L, 0.8)).tolist() == [True, True, False, False]
assert np.isfinite(my_top_p(L, 0.81)).tolist() == [True, True, True, False]
for lg in [flat, peaked]:
    for p in [0.3, 0.9, 0.99]:
        assert np.array_equal(np.isfinite(my_top_p(lg, p)), np.isfinite(top_p_filter(lg, p)))
print("✅ top-p keeps", int(np.isfinite(my_top_p(flat, 0.9)).sum()), "tokens of the flat distribution and",
      int(np.isfinite(my_top_p(peaked, 0.9)).sum()), "of the peaked one")

# %% [markdown]
# ## Exercise 4.2 — min-p
# Keep the tokens whose probability is at least `min_p` times the **top** token's probability.

# %% exercise
def my_min_p(logits, min_p):
    ### BEGIN SOLUTION
    pr = probs(logits)
    return np.where(pr >= min_p * pr.max(), logits, -np.inf)
    ### END SOLUTION

# %% check
assert np.isfinite(my_min_p(L, 0.2)).tolist() == [True, True, True, False]      # threshold 0.1
for lg in [flat, peaked]:
    assert np.array_equal(np.isfinite(my_min_p(lg, 0.1)), np.isfinite(min_p_filter(lg, 0.1)))
print("✅ min-p's threshold scales with the model's confidence")

# %% [markdown]
# ## Exercise 4.3 — the repetition penalty
# vLLM's (and Hugging Face's) rule: for every token that appears in the prompt **or** the output so far, divide
# its logit by `penalty` if it is positive, multiply it by `penalty` if it is negative (both push it down).

# %% exercise
def my_repetition_penalty(logits, seen_ids, penalty):
    ### BEGIN SOLUTION
    x = np.array(logits, float)
    seen = np.unique(np.asarray(list(seen_ids), int))
    x[seen] = np.where(x[seen] > 0, x[seen] / penalty, x[seen] * penalty)
    return x
    ### END SOLUTION

# %% check
x = np.array([2.0, -2.0, 1.0, 0.5])
assert np.allclose(my_repetition_penalty(x, [0, 1, 1], 2.0), [1.0, -4.0, 1.0, 0.5])
ids = encode("The engine tofofof")
assert np.allclose(my_repetition_penalty(flat, ids, 1.3),
                   apply_penalties(flat, SamplingParams(repetition_penalty=1.3), prompt_ids=ids[:11], output_ids=ids[11:]))
print("✅ repetition penalty matches the engine's")

# %% [markdown]
# ## Exercise 4.4 — your own FSM: an integer from 0 to 255
# Write an FSM with the engine's protocol — `start()`, `allowed(state)` (a boolean mask over the `VOCAB` = 257
# token ids; byte `b"7"[0]` is the digit 7, `EOS` ends the output), `advance(state, token)`. Legal outputs:
# `0`–`255` in decimal, no leading zeros, then EOS. Use the digits typed so far (a string) as the state.

# %% exercise
class ByteIntFSM:
    ### BEGIN SOLUTION
    def start(self):
        return ""

    def allowed(self, state):
        mask = np.zeros(VOCAB, bool)
        if state:
            mask[EOS] = True                              # a non-empty number may end here
        if state == "0":
            return mask                                   # no leading zeros
        for d in "0123456789":
            if int(state + d) <= 255:
                mask[ord(d)] = True
        return mask

    def advance(self, state, token):
        return state + chr(token) if token < 256 else state
    ### END SOLUTION

# %% check
f = ByteIntFSM()
digits = lambda m: "".join(chr(t) for t in np.flatnonzero(m) if t < 256)
assert digits(f.allowed("")) == "0123456789" and not f.allowed("")[EOS]
assert digits(f.allowed("25")) == "012345" and f.allowed("25")[EOS]
assert digits(f.allowed("0")) == "" and f.allowed("0")[EOS]
assert digits(f.allowed("255")) == ""
outs = Engine(model, num_blocks=64, block_size=16).generate(
    ["A number: "] * 12, [SamplingParams(max_tokens=8, fsm=f, seed=s) for s in range(12)])
values = [o.text for o in outs]
assert all(v.isdigit() and 0 <= int(v) <= 255 and (v == "0" or v[0] != "0") for v in values), values
print("✅ every output is a legal integer:", values)

# %% [markdown]
# The model's corpus contains no digits at all, so it has no opinion about which number to pick, and it prefers
# to stop after one digit (its corpus has an end-of-text, but no digit ever follows a digit). The grammar
# guaranteed the *format*. The *value* is the model's job.
#
# ## Exercise 4.5 — which settings are deterministic?
# Ignoring exact ties between logits, which of these make the sampled token independent of the seed for **any**
# logits? Fill in `True` / `False`.

# %% exercise
deterministic = {"temperature=0": None, "top_k=1": None, "min_p=1.0": None, "top_p=0.01": None,
                 "temperature=0.01": None}
### BEGIN SOLUTION
deterministic = {"temperature=0": True,        # argmax
                 "top_k=1": True,              # only the top token survives
                 "min_p=1.0": True,            # only tokens as likely as the top one survive
                 "top_p=0.01": False,          # a near-flat distribution keeps several tokens below 1%
                 "temperature=0.01": False}    # sharp, but near-ties still split
### END SOLUTION

# %% check
settings = {"temperature=0": SamplingParams(temperature=0), "top_k=1": SamplingParams(top_k=1),
            "min_p=1.0": SamplingParams(min_p=1.0), "top_p=0.01": SamplingParams(top_p=0.01),
            "temperature=0.01": SamplingParams(temperature=0.01)}
near_tie = -np.arange(VOCAB) * 1e-4                       # an adversarial, almost flat distribution
for name, p in settings.items():
    draws = {sample(near_tie, p, np.random.default_rng(s))[0] for s in range(100)}
    assert (len(draws) == 1) == deterministic[name], name
print("✅ only argmax-equivalent settings are seed-independent for every distribution")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "The engine's sampler is a pipeline over logits: grammar mask, penalties, then greedy
# or temperature, min-p, top-k, top-p, and a draw from the request's own seeded generator. It reshapes the model's
# distribution per step; it never changes the model. For production we default to temperature with top-p or min-p
# because they adapt to the model's confidence, we fix seeds when we need replayable outputs, and we log raw
# logprobs because they show what the model believed. For machine-readable output we use structured output: the
# engine compiles the JSON schema into an automaton and masks illegal tokens, so the output always parses. That
# guarantees syntax, not correctness — a model that does not know the answer will still produce a well-formed one,
# so we validate semantics downstream and watch the logprobs of constrained fields."
#
# **Drill questions**
# 1. *Why prefer top-p or min-p over top-k?* — They adapt to the distribution: few tokens when the model is sure,
#    many when it is not; top-k keeps the same number either way.
# 2. *Structured output is on and the JSON always parses, but the values are wrong. Why?* — The mask enforces
#    syntax only; the model's distribution over legal tokens can still be poor. Check logprobs, improve the prompt.
# 3. *How do you make a sampled request reproducible?* — Set `seed`: the request gets its own generator, so
#    batch composition does not change its draws (modulo GPU numerics).
