"""traces.py — distilling reasoning: collect a thinking teacher's traces, filter them, and see what a student inherits.

One idea: a student trained on reasoning traces inherits more than answers — it inherits the *procedure* and the
teacher's *thinking-length distribution*, so the serving workload of a thinking model (rl-and-thinking-models
PRIMER §7) comes with it (PRIMER §5 "Distilling reasoning"). The two filters you apply to the traces decide
both what the student can solve and how long it will think:

    verifier filter    keep correct traces only — raises the accuracy the student imitates, and shifts the kept
                       lengths toward whatever lengths were more often right
    length cap         keep traces with at most L reasoning tokens — shorter student outputs (cheaper to serve),
                       but the problems that *need* long traces lose their examples first (the hard ones)

At T1 :func:`collect` reads a thinking model served with ``vllm serve Qwen/Qwen3-1.7B --reasoning-parser qwen3``
(``message.reasoning``; ``reasoning_content`` elsewhere) with Qwen3's thinking sampling (temperature 0.6,
top-p 0.95, top-k 20); ``deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`` works the same with ``--reasoning-parser
deepseek_r1`` and its own template (``<｜User｜>``/``<｜Assistant｜>``: SeqKD into a Qwen student is fine, logit KD
across the two templates is not). At T0 the traces come from this lab's fake thinking teacher, bundled as
``assets/samples/traces_thinker_illustrative.jsonl`` (simulated; illustrative). :func:`to_sft_rows` writes them in
the two forms TRL's Qwen3 training template accepts (``reasoning_content`` or inline ``<think>``).
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from importlib import resources

from . import data as D
from . import teacher as TE

QWEN3_THINKING = {"temperature": 0.6, "top_p": 0.95, "extra": {"top_k": 20}}   # Qwen3 model card (verify)


@dataclass
class Trace:
    problem_id: str
    kind: str
    difficulty: int
    question: str
    prompt: str
    reasoning: str | None
    content: str | None
    correct: bool
    reasoning_tokens: int
    completion_tokens: int
    finish_reason: str | None = "stop"

    def to_dict(self) -> dict:
        return asdict(self)


def from_samples(samples: list) -> list:
    return [Trace(s.problem_id, s.kind, s.difficulty, s.question, s.prompt, s.reasoning, s.content, s.correct,
                  s.reasoning_tokens or 0, s.completion_tokens, s.finish_reason) for s in samples]


def collect(client, problems: list, *, n: int = 4, max_tokens: int = 4096, seed: int = 0, workers: int = 8) -> list:
    """``n`` thinking traces per problem with the Qwen3 thinking settings (one request per trace)."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(p, i) for p in problems for i in range(n)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(lambda j: client.chat(j[0].messages(), n=1, thinking=True, max_tokens=max_tokens,
                                                 seed=seed + j[1], temperature=QWEN3_THINKING["temperature"],
                                                 top_p=QWEN3_THINKING["top_p"], extra=QWEN3_THINKING["extra"])[0], jobs))
    out = []
    for (p, i), c in zip(jobs, outs):
        if not c.ok:
            raise RuntimeError(f"trace request failed for {p.id}: {c.error}")
        out.append(Trace(p.id, p.kind, p.difficulty, p.question, p.prompt, c.reasoning, c.content,
                         D.verify(p, c.content), c.reasoning_tokens or 0, c.completion_tokens, c.finish_reason))
    return out


BUNDLED = "traces_thinker_illustrative.jsonl"


def load_bundled() -> list:
    """The bundled traces: simulated by this lab's fake thinking teacher, in vLLM's response fields (illustrative)."""
    text = resources.files("distillab.assets").joinpath("samples", BUNDLED).read_text()
    return [Trace(**json.loads(line)) for line in text.splitlines() if line.strip()]


def pctl(xs: list, q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return float(xs[i])


def length_stats(traces: list, name: str = "") -> dict:
    lens = [t.reasoning_tokens for t in traces]
    return {"set": name, "traces": len(traces), "mean": round(statistics.fmean(lens), 1) if lens else float("nan"),
            "p50": pctl(lens, 0.5), "p90": pctl(lens, 0.9), "p99": pctl(lens, 0.99), "max": max(lens, default=0),
            "correct": round(sum(t.correct for t in traces) / max(1, len(traces)), 3)}


def filter_traces(traces: list, *, correct_only: bool = True, max_reasoning_tokens: int | None = None,
                  unique: bool = True) -> list:
    out = [t for t in traces if t.finish_reason != "length" and (t.correct or not correct_only)
           and (max_reasoning_tokens is None or t.reasoning_tokens <= max_reasoning_tokens)]
    if unique:
        seen, keep = set(), []
        for t in out:
            k = (t.question, TE._norm(t.reasoning), TE._norm(t.content))
            if k not in seen:
                seen.add(k)
                keep.append(t)
        out = keep
    return out


def trade(traces: list, caps: list) -> list:
    """For each length cap: what survives among the *correct* traces, how long it is, and which problems keep
    at least one example — overall and at the hardest difficulty (the problems a cap starves first)."""
    correct = filter_traces(traces, correct_only=True, unique=False)
    problems = {t.problem_id for t in traces}
    hard = {t.problem_id for t in traces if t.difficulty == max(x.difficulty for x in traces)}
    rows = []
    for cap in caps:
        kept = [t for t in correct if cap is None or t.reasoning_tokens <= cap]
        covered = {t.problem_id for t in kept}
        rows.append({"cap": "none" if cap is None else cap, "kept traces": len(kept),
                     "share of correct": round(len(kept) / max(1, len(correct)), 3),
                     "mean reasoning tokens": round(statistics.fmean([t.reasoning_tokens for t in kept]), 1) if kept else 0.0,
                     "problems covered": round(len(covered) / max(1, len(problems)), 3),
                     "hardest covered": round(len(covered & hard) / max(1, len(hard)), 3)})
    return rows


def accuracy_by_length(traces: list, edges: list) -> list:
    """Teacher accuracy by reasoning length bin [edges[i], edges[i+1]). In the fake teacher longer traces hold more
    re-checks, so they are more often right; in real models hard problems also take longer, which can make long
    traces *less* often right — read this table per difficulty before concluding anything."""
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        b = [t for t in traces if lo <= t.reasoning_tokens < hi]
        if b:
            rows.append({"reasoning tokens": f"{lo}-{hi - 1}", "traces": len(b),
                         "correct": round(sum(t.correct for t in b) / len(b), 3),
                         "mean difficulty": round(statistics.fmean(t.difficulty for t in b), 2)})
    return rows


def compare(clients: dict, problems: list, *, n: int = 2, max_tokens: int = 4096, seed: int = 0) -> list:
    """Before and after, measured the same way: for each served model (``{"student before": Client, "student
    after": Client, "teacher": Client}``), accuracy on ``problems`` with a 95% Wilson interval and the reasoning
    length it produces. At T1 the clients are vLLM servers of the base student, the SFT'd student and the teacher."""
    from .agreement import wilson_interval
    rows = []
    for name, client in clients.items():
        tr = collect(client, problems, n=n, max_tokens=max_tokens, seed=seed)
        k = sum(t.correct for t in tr)
        lo, hi = wilson_interval(k, len(tr))
        st = length_stats(tr, name)
        rows.append({"model": name, "traces": len(tr), "accuracy": round(k / len(tr), 3), "95% CI": f"{lo:.2f}-{hi:.2f}",
                     "mean reasoning": st["mean"], "p90 reasoning": st["p90"]})
    return rows


def to_sft_rows(traces: list, reasoning: str = "field") -> list:
    """Conversational rows for TRL's SFTTrainer; ``reasoning`` = "field" (``reasoning_content``) or "inline"."""
    return [{"messages": [{"role": "user", "content": t.prompt},
                          TE.assistant_message(TE.Sample(t.problem_id, t.kind, t.difficulty, t.question, t.prompt,
                                                         t.content, t.reasoning, t.correct, t.finish_reason, 0,
                                                         t.completion_tokens, t.reasoning_tokens), reasoning)]}
            for t in traces]


def reasoning_bill(traces: list, price_out_per_m: float) -> dict:
    """Thinking tokens are billed as output tokens: the trace set's generation cost at a $/M output price."""
    out = sum(t.completion_tokens for t in traces)
    return {"traces": len(traces), "output tokens": out, "reasoning tokens": sum(t.reasoning_tokens for t in traces),
            "cost $": round(out * price_out_per_m / 1e6, 4)}
