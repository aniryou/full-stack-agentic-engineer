"""parsers.py — split a thinking model's output into *reasoning* and *content*, the way servers do.

One idea: a thinking model emits one token stream; the reasoning block is only a convention of its
chat template (``<think>…</think>`` for Qwen3 and DeepSeek-R1, Harmony *channels* for gpt-oss).
A server with a reasoning parser (vLLM ``--reasoning-parser qwen3`` / ``deepseek_r1``; SGLang
``--reasoning-parser qwen3`` / ``deepseek-r1``) cuts the stream at those markers and returns two
fields. Three details decide whether a client gets it right:

* **Where the stream starts.** R1-style templates put ``<think>\\n`` in the *prompt*, so the output
  often has no opening tag: everything before ``</think>`` is reasoning (vLLM's
  ``DeepSeekR1ReasoningParser``). Qwen3's parser starts *in* reasoning when thinking is on, and
  returns everything as content when ``enable_thinking=False`` (vLLM v0.30.0 ``Qwen3Parser``).
* **The field name.** vLLM ≥ the rename returns ``message.reasoning`` / ``delta.reasoning`` (it was
  ``reasoning_content``, which SGLang, the DeepSeek API and the Qwen docs still use). A client that
  reads only one of them silently gets an empty string from the other kind of server: read both.
* **Streaming.** A tag can be split across two network chunks; the parser must hold back a partial
  tag instead of leaking ``</thi`` into the reasoning.

No imports beyond the standard library; the server-side reference implementations are
``vllm/reasoning/basic_parsers.py``, ``deepseek_r1_reasoning_parser.py`` and ``qwen3_engine_reasoning_parser.py``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

THINK_START, THINK_END = "<think>", "</think>"
REASONING_FIELDS = ("reasoning", "reasoning_content")     # vLLM 0.30 first, then SGLang / DeepSeek API / older vLLM


@dataclass
class Split:
    reasoning: str | None
    content: str | None

    @property
    def thought(self) -> bool:
        return bool(self.reasoning and self.reasoning.strip())


def split_deepseek_r1(text: str, start: str = THINK_START, end: str = THINK_END) -> Split:
    """vLLM's ``BaseThinkingReasoningParser.extract_reasoning`` (used by ``deepseek_r1``):
    drop an opening tag if present; if no closing tag, *everything* is reasoning (the model was cut
    off while thinking); otherwise reasoning is before the first closing tag and content after it
    (``None`` when empty)."""
    before, sep, after = text.partition(start)
    body = after if sep else before
    if end not in body:
        return Split(body, None)
    reasoning, _, content = body.partition(end)
    return Split(reasoning, content or None)


def split_qwen3(text: str, enable_thinking: bool = True) -> Split:
    """Qwen3 semantics: thinking off → the whole output is content (no parsing at all); thinking on
    → as ``deepseek_r1`` (the parser starts in the reasoning state, so a missing ``<think>`` is fine).
    A ``<tool_call>`` inside reasoning also ends it in vLLM's Qwen3 parser — not modelled here."""
    if not enable_thinking:
        return Split(None, text)
    return split_deepseek_r1(text)


_HARMONY = re.compile(r"<\|channel\|>(\w+)<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|<\|call\|>|<\|start\|>|$)", re.S)


def split_harmony(text: str) -> Split:
    """gpt-oss (Harmony format): messages carry a *channel*. ``analysis`` is the chain of thought,
    ``final`` is the answer shown to the user (``commentary`` carries tool calls and preambles)."""
    analysis, final = [], []
    for channel, body in _HARMONY.findall(text):
        (analysis if channel == "analysis" else final if channel == "final" else []).append(body)
    return Split("".join(analysis) or None, "".join(final) or None)


PARSERS = {
    # name as passed to `vllm serve --reasoning-parser` (underscores; SGLang uses hyphens)
    "qwen3": split_qwen3,
    "deepseek_r1": split_deepseek_r1,
    "openai_gptoss": split_harmony,
}


def split(text: str, parser: str = "qwen3", **kw) -> Split:
    """Parse a raw completion the way ``--reasoning-parser <parser>`` would."""
    try:
        fn = PARSERS[parser]
    except KeyError:
        raise ValueError(f"unknown parser {parser!r}; known: {sorted(PARSERS)}") from None
    return fn(text, **kw) if kw else fn(text)


def reasoning_of(obj) -> str | None:
    """The reasoning text of a response ``message`` or streaming ``delta`` (dict or object), whichever
    field name the server used. Returns ``None`` when neither is present or both are empty."""
    for name in REASONING_FIELDS:
        v = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if v:
            return v
    return None


class StreamSplitter:
    """Incremental ``<think>…</think>`` splitter for raw text streams (``/v1/completions`` or a server
    without a reasoning parser). ``feed(chunk)`` returns ``(reasoning_delta, content_delta)``; a
    possible partial tag at the end of a chunk is held back until the next chunk decides it.

    ``starts_in_reasoning=True`` models templates that open the think block in the prompt (R1, Qwen3
    thinking mode)."""

    def __init__(self, starts_in_reasoning: bool = True, start: str = THINK_START, end: str = THINK_END):
        self.start, self.end = start, end
        self.state = "reasoning" if starts_in_reasoning else "content"
        self.buf = ""
        self.reasoning, self.content = [], []

    def _hold(self, s: str, tags: tuple) -> int:
        """Length of the longest suffix of ``s`` that is a proper prefix of one of ``tags``."""
        best = 0
        for tag in tags:
            for n in range(min(len(tag) - 1, len(s)), 0, -1):
                if s.endswith(tag[:n]):
                    best = max(best, n)
                    break
        return best

    def feed(self, chunk: str) -> tuple:
        s = self.buf + chunk
        r_out, c_out = [], []
        while s:
            if self.state == "reasoning":
                i_end = s.find(self.end)
                if i_end >= 0:
                    r_out.append(s[:i_end].replace(self.start, ""))
                    s, self.state = s[i_end + len(self.end):], "content"
                    continue
                hold = self._hold(s, (self.end, self.start))
                r_out.append(s[: len(s) - hold].replace(self.start, ""))
                s = s[len(s) - hold:]
                break
            i_start = s.find(self.start)
            if i_start >= 0:
                c_out.append(s[:i_start])
                s, self.state = s[i_start + len(self.start):], "reasoning"
                continue
            if self.end in s:                  # a stray duplicate </think> in content: drop it (as vLLM's Qwen3 parser does)
                s = s.replace(self.end, "")
                continue
            hold = self._hold(s, (self.start, self.end))
            c_out.append(s[: len(s) - hold])
            s = s[len(s) - hold:]
            break
        self.buf = s
        r, c = "".join(r_out), "".join(c_out)
        self.reasoning.append(r)
        self.content.append(c)
        return r, c

    def close(self) -> Split:
        """End of stream: flush what was held back (an unfinished tag is just text)."""
        if self.buf:
            (self.reasoning if self.state == "reasoning" else self.content).append(self.buf)
            self.buf = ""
        r, c = "".join(self.reasoning), "".join(self.content)
        return Split(r or None, c or None)


# --- answers ----------------------------------------------------------------------------------
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER = re.compile(r"(?:final answer|answer)\s*(?:is\b|[:=])\s*\**\s*([-\w./]+)", re.I)


def extract_answer(content: str | None) -> str | None:
    """The final answer in a completion's *content* (never its reasoning): the last ``\\boxed{…}``,
    else the last "Answer: …", else ``None``. Normalised to lower case without surrounding
    punctuation, so ``\\boxed{ 42 }`` and ``Answer: 42.`` compare equal."""
    if not content:
        return None
    boxed = _BOXED.findall(content)
    raw = boxed[-1] if boxed else (_ANSWER.findall(content) or [None])[-1]
    if raw is None:
        return None
    return raw.strip().strip(".,;:*").lower() or None
