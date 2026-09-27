"""teacher.py — teacher data through any OpenAI-compatible server: sample, verify, deduplicate, bill, write.

One idea: sequence-level distillation's dataset is a *pipeline* with a yield at every stage, and the bill is
paid on everything the teacher generated, not on what you keep (PRIMER §3 "Sequence-level distillation:
learning from the teacher's outputs"):

    prompts × n samples  →  verifier (rejection sampling)  →  deduplicate  →  (length cap)  →  JSONL
         generated              kept if right                  distinct         short enough     what SFT reads

Everything here runs against vLLM at T1 (``vllm serve Qwen/Qwen2.5-1.5B-Instruct``) or this lab's fake teacher
at T0, through :class:`distillab.client.Client`. The JSONL rows use the dataset formats TRL's ``SFTTrainer``
reads (TRL 1.14.0 ``docs/source/dataset_formats.md``, verify): *conversational* ``{"messages": [{"role":
"user", ...}, {"role": "assistant", ...}]}`` or *prompt-completion* ``{"prompt": [user message], "completion":
[assistant message]}`` (loss on the completion only). A thinking teacher's reasoning goes in the assistant
message's ``reasoning_content`` field or inline as ``<think>…</think>`` — the two ways TRL's Qwen3 training
template accepts it. :func:`api_cost` re-implements the 06 scaling lab's ``cost_per_call`` (USD per million
tokens in / out / cached; its price table was checked on 5 Sep 2026 — verify before quoting).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from . import data as D

# USD per 1M tokens (input, output, cached input), as the 06 scaling lab's scalelab.capacity.PRICES lists them
# ("checked 5 Sep 2026"; verify before quoting). Re-implemented here: labs are standalone.
PRICES = {
    "gemini-3.5-flash": (1.50, 9.00, 0.15),
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03),
    "gemini-3.1-pro-preview": (2.00, 12.00, 0.20),
}


def api_cost(model: str, input_tokens: float, output_tokens: float, cached_tokens: float = 0.0) -> float:
    """Dollars for one call at the table's prices: uncached input + cached input + output, per million."""
    inp, out, cached = PRICES[model]
    return ((input_tokens - cached_tokens) * inp + cached_tokens * cached + output_tokens * out) / 1e6


@dataclass
class Sample:
    problem_id: str
    kind: str
    difficulty: int
    question: str
    prompt: str
    content: str | None
    reasoning: str | None
    correct: bool
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None
    index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def generate(client, problems: list, *, n: int = 4, temperature: float = 0.7, max_tokens: int = 512,
             thinking: bool | None = None, seed: int = 0, workers: int = 8) -> list:
    """``n`` teacher samples per problem, each scored by the verifier on its *content*. One request per sample
    (``n=1``, seed ``seed + i``): an API reports ``usage`` for the whole request, and the per-sample lengths are
    what the length filter and the trace statistics need (vLLM's prefix cache makes the repeated prompt cheap)."""
    jobs = [(p, i) for p in problems for i in range(n)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(lambda job: client.chat(job[0].messages(), n=1, temperature=temperature,
                                                   max_tokens=max_tokens, thinking=thinking,
                                                   seed=seed + job[1])[0], jobs))
    samples = []
    for (p, i), c in zip(jobs, outs):
        if not c.ok:
            raise RuntimeError(f"teacher request failed for {p.id}: {c.error}")
        samples.append(Sample(p.id, p.kind, p.difficulty, p.question, p.prompt, c.content, c.reasoning,
                              D.verify(p, c.content), c.finish_reason, c.prompt_tokens, c.completion_tokens,
                              c.reasoning_tokens, i))
    return samples


def filter_verified(samples: list) -> list:
    """Rejection sampling: keep what the verifier accepts (and what finished — a cut-off answer is not data)."""
    return [s for s in samples if s.correct and s.finish_reason != "length"]


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def dedup(samples: list) -> list:
    """Drop repeats of the same (question, reasoning, content) — the same trace twice teaches nothing new and
    over-weights that problem. Near-duplicates (different phrasing, same working) survive: say so if it matters."""
    seen, out = set(), []
    for s in samples:
        key = (s.question, _norm(s.reasoning), _norm(s.content))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def length_cap(samples: list, max_completion_tokens: int) -> list:
    return [s for s in samples if s.completion_tokens <= max_completion_tokens]


def token_bill(samples: list) -> dict:
    """What the teacher generated and read, whatever survives the filters (you pay for all of it)."""
    return {"requests": len({(s.problem_id) for s in samples}), "samples": len(samples),
            "prompt_tokens": sum(s.prompt_tokens for s in samples),
            "completion_tokens": sum(s.completion_tokens for s in samples),
            "reasoning_tokens": sum(s.reasoning_tokens or 0 for s in samples)}


def funnel(samples: list, max_completion_tokens: int | None = None) -> tuple:
    """The whole pipeline with the count at each stage. Returns ``(kept, rows)``."""
    rows = [{"stage": "generated", "samples": len(samples)}]
    s1 = filter_verified(samples)
    rows.append({"stage": "verified (right and finished)", "samples": len(s1)})
    s2 = dedup(s1)
    rows.append({"stage": "deduplicated", "samples": len(s2)})
    if max_completion_tokens is not None:
        s2 = length_cap(s2, max_completion_tokens)
        rows.append({"stage": f"length <= {max_completion_tokens} tokens", "samples": len(s2)})
    covered = len({s.problem_id for s in s2})
    rows.append({"stage": "problems with at least one kept sample", "samples": covered})
    return s2, rows


# --- JSONL in TRL's formats --------------------------------------------------------------------------------

def assistant_message(s: Sample, reasoning: str = "field") -> dict:
    """``reasoning="field"``: ``reasoning_content`` beside ``content``; ``"inline"``: ``<think>…</think>`` at the
    start of ``content``; ``"drop"``: the answer only (what a model that should not think is trained on)."""
    msg = {"role": "assistant", "content": s.content or ""}
    if s.reasoning and reasoning == "field":
        msg["reasoning_content"] = s.reasoning
    elif s.reasoning and reasoning == "inline":
        msg["content"] = f"<think>\n{s.reasoning}\n</think>\n\n{s.content or ''}"
    elif reasoning not in ("field", "inline", "drop"):
        raise ValueError(reasoning)
    return msg


def to_messages(samples: list, reasoning: str = "field") -> list:
    return [{"messages": [{"role": "user", "content": s.prompt}, assistant_message(s, reasoning)]} for s in samples]


def to_prompt_completion(samples: list, reasoning: str = "field") -> list:
    return [{"prompt": [{"role": "user", "content": s.prompt}], "completion": [assistant_message(s, reasoning)]}
            for s in samples]


def check_rows(rows: list) -> list:
    """Problems that would make TRL's SFTTrainer reject or mis-train a row (empty = fine): the keys of one of
    the two conversational formats, roles, string contents, and an assistant turn to learn from."""
    problems = []
    for i, r in enumerate(rows):
        if "messages" in r:
            msgs = r["messages"]
            if not msgs or msgs[-1].get("role") != "assistant":
                problems.append(f"row {i}: the last message must be the assistant's")
        elif {"prompt", "completion"} <= set(r):
            msgs = r["prompt"] + r["completion"]
            if not r["completion"] or r["completion"][0].get("role") != "assistant":
                problems.append(f"row {i}: completion must be an assistant message")
        else:
            problems.append(f"row {i}: neither 'messages' nor 'prompt' + 'completion'")
            continue
        for m in msgs:
            if m.get("role") not in ("system", "user", "assistant", "tool"):
                problems.append(f"row {i}: bad role {m.get('role')!r}")
            if not isinstance(m.get("content"), str):
                problems.append(f"row {i}: content must be a string")
    return problems


def write_jsonl(rows: list, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def read_jsonl(path: str | Path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# --- logit targets through an API: top-k only ---------------------------------------------------------

def topk_target(top: list) -> tuple:
    """A truncated logit-KD target from one position's API ``top_logprobs`` ``[(token, logprob), ...]``:
    the top-k probabilities renormalised to sum to 1, and the mass the API did not return (which the
    renormalisation silently gives to the top k). Returns ``({token: p}, missing_mass)``."""
    probs = {t: math.exp(lp) for t, lp in top}
    total = sum(probs.values())
    return {t: p / total for t, p in probs.items()}, max(0.0, 1.0 - total)


def by_kind(samples: list) -> list:
    """Accuracy of the teacher's samples per (kind, difficulty): where the verifier throws the most away."""
    n, ok = Counter(), Counter()
    for s in samples:
        n[(s.kind, s.difficulty)] += 1
        ok[(s.kind, s.difficulty)] += s.correct
    return [{"kind": k, "difficulty": d, "samples": n[(k, d)], "correct": round(ok[(k, d)] / n[(k, d)], 3)}
            for k, d in sorted(n)]
