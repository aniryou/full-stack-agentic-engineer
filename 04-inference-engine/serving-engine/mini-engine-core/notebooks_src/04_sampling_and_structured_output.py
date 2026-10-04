# %% [markdown]
# # 04 · Sampling and structured output
#
# **Tier:** T0. It needs a CPU only and no network, and it runs in a few seconds. The same parameters are fields of
# vLLM's `SamplingParams` and of its OpenAI-compatible API. `vllm-serving-lab` sends them to a real server (T1).
#
# ## The one-minute version
# The forward pass produces **logits**: one score per vocabulary entry. The sampler turns them into one token
# through a pipeline of steps in a set order:
#
# 1. It masks the tokens that the grammar does not permit (structured output).
# 2. It applies the repetition, presence and frequency **penalties**.
# 3. If `temperature == 0`, it takes the argmax.
# 4. It divides by the **temperature**.
# 5. It cuts the tail with **min-p**, **top-k** and **top-p**.
# 6. It draws a token with the request's **own seeded generator**.
#
# None of this changes the model. It changes the shape of one distribution per step. **Structured output** is the
# same mechanism with a grammar behind it. A finite-state machine says which tokens are legal next, and the other
# tokens get $-\infty$. The model never learns the schema. The mask holds the model inside the grammar, and that
# guarantees *syntax*, not *sense*.
#
# With real, multi-character tokens, the calculation of that mask for each grammar state is the high-cost part. In
# this notebook, you build one.
#
# Primer: §6 *Sampling and structured output* (`../../PRIMER.md`). Background:
# `00-foundations/transformers/docs/transformer-primer.md` §7.4.

# %%
import json
import re

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
# **top-k** keeps five tokens for any shape of the distribution. **top-p** and **min-p** adapt. They keep many
# tokens when the model is unsure, and few tokens when it is confident. That is why they are the usual choice.
# Temperature changes the shape, not the order of the tokens: the argmax is the same at each temperature.
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
# Each request gets its own random generator. The seed of that generator comes from `seed`, or from the engine's
# seed and the index of the request. Thus the output of a seeded request does not depend on the other requests that
# shared its steps. You need this property to replay a production incident.

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
# (On a GPU, bitwise batch invariance is more difficult than this. The batch size changes the kernel choices and the
# reduction order, so the logits are different in the last bits. For that, vLLM has a batch-invariant mode, at
# some cost in speed, verify.)
#
# ## Worked example 4 — logprobs are the model's, not the sampler's
# vLLM V1 returns logprobs of the **raw** logits by default (`logprobs_mode="raw_logprobs"`). These are the logprobs
# before temperature, penalties or masks. They tell you what the model believed, even when the sampler went against
# the model.

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
    print(f"{o.text!r:22} {o.finish_reason:22} model's own log-probability of this text: {forced:7.1f}")

# %% [markdown]
# Each output is valid. But the model's own log-probability of each output is far below zero: the probability is many
# orders of magnitude below 1. Without the mask, the model never produces `{`. The mask **forces** syntax. It cannot
# supply meaning. If the model does not know the answer, the grammar makes it say `"yes"` or `"no"` all the same.
#
# Real engines compile JSON schemas or regexes into token-level automata (xgrammar, llguidance, outlines). Make sure
# that these are the current vLLM backends (verify). With a BPE vocabulary, each token contains several characters.
# Thus the calculation of the mask for each step over ~100k tokens is the problem that an engineer must solve.
#
# ## Worked example 6 — a grammar over characters, a vocabulary of multi-character tokens
# Take the JSON schema `{"type": "object", "properties": {"n": {"type": "integer", "minimum": 0}},
# "required": ["n"]}` with no optional whitespace. The legal outputs are `{"n": 0}`, `{"n": 7}`, `{"n": 2026}`, …
# (JSON does not permit leading zeros).
#
# A structured-output backend compiles the schema into an automaton over
# **characters**. The next cell writes that automaton by hand. But the vocabulary, like the vocabulary of any BPE
# tokenizer, mixes single characters with merges such as `{"n": ` or `1}`. Such a merge crosses several automaton
# states in one token.

# %%
PREFIX = '{"n": '


def char_step(state, ch):
    """Character-level DFA for {"n": 0|[1-9][0-9]*}: the next state, or None if `ch` is illegal here."""
    if state < len(PREFIX):                                  # states 0-5: inside the fixed prefix
        return state + 1 if ch == PREFIX[state] else None
    if state == 6:                                           # 6: the number's first digit
        return 7 if ch in "123456789" else (8 if ch == "0" else None)
    if state == 7:                                           # 7: more digits, or close
        return 7 if ch.isdigit() else (9 if ch == "}" else None)
    if state == 8:                                           # 8: after a lone 0 only "}" may follow
        return 9 if ch == "}" else None
    return None                                              # 9: accepting; nothing may follow


STATES, ACCEPT = range(10), 9
VOCAB_TXT = list('{}":, ') + list("0123456789") + list("anxy") + [
    '{"', '":', '": ', '"n', 'n"', '{"n": ', '"}', '12', '42', '2026', '00', '07', '1}', '0}', ', "', 'yes']
state = 0
for ch in PREFIX:
    state = char_step(state, ch)
print(f"{len(VOCAB_TXT)} tokens; the single token {PREFIX!r} takes the DFA from state 0 to state {state}")
print("in state 6, a quote ->", char_step(6, '"'), "(illegal: a number must come next)")

# %% [markdown]
# A token is legal in a state only if **every** character of it is legal in turn, from that state. Thus the mask
# depends on the state, and one token can move the automaton several states at once. Exercise 4.6 precomputes the
# whole table.
#
# ## Exercise 4.1 — top-p (nucleus) filtering
# Keep the smallest set of highest-probability tokens whose total probability gets to $p$. Set the other tokens to
# $-\infty$. Always keep the top token.

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
# Use the rule of vLLM (and of Hugging Face). Find each token that is in the prompt **or** in the output until now.
# If its logit is positive, divide the logit by `penalty`. If its logit is negative, multiply the logit by
# `penalty`. Both operations push the logit down.

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
# Write an FSM with the engine's protocol:
#
# * `start()`
# * `allowed(state)`: a boolean mask over the `VOCAB` = 257 token ids. Byte `b"7"[0]` is the digit 7, and `EOS` ends
#   the output.
# * `advance(state, token)`
#
# The legal outputs are the integers from `0` to `255` in decimal, with no leading zeros, then EOS. Use the
# digits typed until now (a string) as the state.

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
# The model's corpus contains no digits at all. Thus the model has no opinion about which number to select. It
# prefers to stop after one digit: its corpus has an end-of-text, but no digit ever follows a digit. The grammar
# guaranteed the *format*. The *value* is the model's job.
#
# ## Exercise 4.5 — which settings are deterministic?
# Ignore exact ties between logits. Which of these settings make the sampled token independent of the seed for
# **any** logits? Fill in `True` / `False`.

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
# ## Exercise 4.6 — compile the token mask table
# Use `char_step`, `STATES` and `VOCAB_TXT` from worked example 6. For each automaton state `s` and each token `t`,
# precompute the state after you feed **all** of `t`'s characters from `s`. If a character is illegal, the result is
# `None`. Return `table[s]`, a list aligned with the vocabulary. The engine's mask in state `s` then becomes
# `table[s][t] is not None`, and the advance after a sampled token is one lookup.
#
# The purpose of xgrammar and llguidance is to do this ahead of time for ~100k tokens and thousands of states. For
# the rest, they do it sufficiently fast to overlap the GPU's forward pass.

# %% exercise
def compile_token_table(vocab, states, step):
    ### BEGIN SOLUTION
    table = {}
    for s in states:
        row = []
        for tok in vocab:
            q = s
            for ch in tok:
                q = step(q, ch)
                if q is None:
                    break
            row.append(q)
        table[s] = row
    return table
    ### END SOLUTION

# %% check
table = compile_token_table(VOCAB_TXT, STATES, char_step)
allowed = lambda s: {VOCAB_TXT[t] for t, q in enumerate(table[s]) if q is not None}
assert allowed(0) == {"{", '{"', '{"n": '} and table[0][VOCAB_TXT.index('{"n": ')] == 6
assert table[6][VOCAB_TXT.index("1}")] == ACCEPT                          # one token: a digit AND the close
assert "07" in allowed(7) and "07" not in allowed(6)                      # same token, legal in one state only
assert not any('"}' in allowed(s) for s in STATES)                        # looks like JSON, never legal here
print("tokens allowed per state:", {s: len(allowed(s)) for s in STATES}, f"(of {len(VOCAB_TXT)})")
rng, runs = np.random.default_rng(0), []
for _ in range(300):                                  # a "model" with no opinion: uniform over the legal tokens
    state, toks = 0, []
    while state != ACCEPT and len(toks) < 12:         # max_tokens = 12
        t = int(rng.choice([t for t, q in enumerate(table[state]) if q is not None]))
        toks.append(t)
        state = table[state][t]
    runs.append(("".join(VOCAB_TXT[t] for t in toks), tuple(toks), state == ACCEPT))
done = [x for x, _, ok in runs if ok]
cut = [x for x, _, ok in runs if not ok]
assert all(re.fullmatch(r'\{"n": (0|[1-9][0-9]*)\}', x) and isinstance(json.loads(x)["n"], int) for x in done)
assert cut and all(re.fullmatch(r'\{"n": (0|[1-9][0-9]*)', x) for x in cut)   # legal so far, but unfinished
spellings = {}
for x, toks, _ in runs:
    spellings.setdefault(x, set()).add(toks)
top = max(spellings, key=lambda x: len(spellings[x]))
print(f"✅ {len(done)} of 300 constrained outputs parse as the schema; {len(cut)} hit max_tokens mid-number, "
      f"e.g. {cut[0]!r} - a legal prefix that does not parse")
print(f"   {top!r} came out as {len(spellings[top])} different token sequences, e.g.",
      " / ".join("|".join(VOCAB_TXT[t] for t in seq) for seq in sorted(spellings[top], key=len)[:2]))

# %% [markdown]
# This toy already shows three things that real backends handle:
#
# * **The mask is per state**: `07` is legal after `{"n": 1` and illegal immediately after `{"n": `. Thus the engine
#   needs the state of the automaton for each request, in each step.
# * **One text, many token sequences**: the grammar permits all of them. Thus the mask can push a constrained model
#   into tokenizations that it rarely saw in training. Also, a client that tokenizes the returned text again gets
#   different ids. Different ids give different block names in the prefix cache (primer §5).
# * **A length cap cuts a legal prefix**: the grammar mask guarantees that the output *so far* is legal, not that it
#   finishes. Examine `finish_reason` (vLLM: `length`) before you parse the output.
#
# ## In a design review
# **The two-minute version.** "The engine's sampler is a pipeline over logits. It applies the grammar mask and the
# penalties. Then it takes the greedy token, or it applies temperature, min-p, top-k and top-p and draws from the
# request's own seeded generator. It changes the shape of the model's distribution per step, but it never changes
# the model.
#
# "For production, our default is temperature with top-p or min-p, because they adapt to the model's confidence. We
# set the seed when we need outputs that we can replay. We log raw logprobs, because they show what the model
# believed.
#
# "For machine-readable output, we use structured output. The engine compiles the JSON schema into a character
# automaton. It precomputes which multi-character tokens of the vocabulary are legal in each state, and it masks
# the other tokens. Thus an output that finishes always parses. An output that max_tokens cuts does not parse, so we
# examine finish_reason.
#
# "Structured output guarantees syntax, not correctness. A model that does not know the answer still produces a
# well-formed answer. Thus we make sure downstream that the semantics are correct, and we monitor the logprobs of
# constrained fields."
#
# **Drill questions**
# 1. *Why prefer top-p or min-p over top-k?* They adapt to the distribution. They keep few tokens when the model is
#    sure, and many tokens when it is not. But top-k keeps the same number in both cases.
# 2. *Structured output is on and the JSON always parses, but the values are incorrect. Why?* The mask enforces
#    syntax only. The model's distribution over legal tokens can still be poor. Examine the logprobs, and improve
#    the prompt.
# 3. *How do you make a sampled request reproducible?* Set `seed`. The request gets its own generator. Thus the
#    composition of the batch does not change its draws (except for GPU numerics).
