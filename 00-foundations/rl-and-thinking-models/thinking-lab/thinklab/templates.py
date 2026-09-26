"""templates.py — what a thinking model's chat template does to the *next* turn's prompt.

One idea: the prefix cache reuses KV blocks only for an identical token prefix (04 PRIMER §5
"Prefix caching": full 16-token blocks, chained hashes). Thinking-model templates **drop the
reasoning of earlier turns** when they render history (Qwen3; gpt-oss: "CoT is dropped during all
previous turns"). So turn N *generated* ``<think>…</think>answer`` into the KV cache, but turn
N + 1's prompt contains only ``answer`` for that turn: the cached prefix ends right after turn N's
assistant header, the thinking KV is never reused, and the answer is prefilled again. Inside one
tool-calling loop (assistant turns after the last real user message) the thinking is kept, so the
prefix keeps matching there.

``render_qwen3`` re-implements the history logic of Qwen3's chat template in Python (the Jinja
source is ``qwen3/docs/source/assets/qwen3_nonthinking.jinja`` in the Qwen3 repo: ``last_query_index``,
reasoning kept only after it). It renders *strings*; ``tokens`` is a toy whitespace tokenizer so
block arithmetic can be shown without a real tokenizer — a real run counts real tokens
(``usage.prompt_tokens_details.cached_tokens`` with ``--enable-prompt-tokens-details``).
"""
from __future__ import annotations

import re

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def _split_reasoning(msg: dict) -> tuple:
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content")
    if reasoning is None:
        reasoning = msg.get("reasoning")
    if reasoning is None and "</think>" in content:
        reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
        content = content.split("</think>")[-1].lstrip("\n")
    return reasoning or "", content


def _is_real_query(msg: dict) -> bool:
    c = msg.get("content")
    return (msg.get("role") == "user" and isinstance(c, str)
            and not (c.startswith("<tool_response>") and c.endswith("</tool_response>")))


def render_qwen3(messages: list, add_generation_prompt: bool = True, enable_thinking: bool = True,
                 keep_all_reasoning: bool = False) -> str:
    """Render ChatML the way Qwen3's template does (no tools). Assistant turns *before* the last real
    user query lose their reasoning; later ones (the current tool loop) keep it. With
    ``enable_thinking=False`` the generation prompt pre-fills an empty think block.
    ``keep_all_reasoning=True`` is the counterfactual "interleaved thinking" rendering."""
    last_query = max((i for i, m in enumerate(messages) if _is_real_query(m)), default=len(messages) - 1)
    out = []
    for i, m in enumerate(messages):
        role = m["role"]
        if role in ("user", "system"):
            out.append(f"{IM_START}{role}\n{m.get('content') or ''}{IM_END}\n")
        elif role == "assistant":
            reasoning, content = _split_reasoning(m)
            if (i > last_query and (i == len(messages) - 1 or reasoning)) or (keep_all_reasoning and reasoning):
                out.append(f"{IM_START}assistant\n<think>\n{reasoning.strip(chr(10))}\n</think>\n\n{content.lstrip(chr(10))}")
            else:
                out.append(f"{IM_START}assistant\n{content}")
            out.append(f"{IM_END}\n")
        elif role == "tool":
            out.append(f"{IM_START}user\n<tool_response>\n{m.get('content') or ''}\n</tool_response>{IM_END}\n")
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n"))
    return "".join(out)


_TOK = re.compile(r"<\|[a-z_]+\|>|</?think>|\w+|[^\w\s]|\n")


def tokens(text: str) -> list:
    """A toy tokenizer: special markers, words, punctuation and newlines are one token each."""
    return _TOK.findall(text)


def common_prefix(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def cached_tokens(prev_context: list, next_prompt: list, block_size: int = 16) -> int:
    """Tokens of ``next_prompt`` a block-hash prefix cache can serve from ``prev_context`` (the previous
    request's prompt + generated tokens): the common prefix rounded *down* to whole blocks. vLLM also
    never serves the very last prompt token from cache (it must compute at least one), so a full hit
    is capped at ``len(next_prompt) − 1`` before rounding (04 PRIMER §5)."""
    n = min(common_prefix(prev_context, next_prompt), len(next_prompt) - 1)
    return (n // block_size) * block_size


def turn_reuse(history: list, turn_output: dict, next_user: str, block_size: int = 16,
               keep_all_reasoning: bool = False) -> dict:
    """One multi-turn step: the tokens turn N put in the KV cache, the prompt of turn N + 1, and how
    much of that prompt the prefix cache can serve. ``turn_output`` = {"reasoning", "content"}."""
    prompt_n = render_qwen3(history)
    generated = f"<think>\n{turn_output['reasoning']}\n</think>\n\n{turn_output['content']}{IM_END}\n"
    context_n = tokens(prompt_n + generated)
    nxt = history + [{"role": "assistant", "content": turn_output["content"], "reasoning": turn_output["reasoning"]},
                     {"role": "user", "content": next_user}]
    prompt_next = tokens(render_qwen3(nxt, keep_all_reasoning=keep_all_reasoning))
    hit = cached_tokens(context_n, prompt_next, block_size)
    return {"context_tokens": len(context_n), "reasoning_tokens": len(tokens(turn_output["reasoning"])),
            "next_prompt_tokens": len(prompt_next), "common_prefix": common_prefix(context_n, prompt_next),
            "cached": hit, "prefilled": len(prompt_next) - hit}
