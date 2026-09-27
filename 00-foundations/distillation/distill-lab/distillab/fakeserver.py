"""fakeserver.py — a fake vLLM teacher: answers the generated problems with scratchpads, log-probs and usage.

One idea: every T1 path in this lab talks to a teacher through the OpenAI-compatible API that vLLM serves, so
a T0 stand-in must speak the same protocol with the same field names — and be honest that everything it
says is **simulated**. This server answers like ``vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-logprobs 20``
(v0.30.0 request and response shapes) for the problems of :mod:`distillab.data`:

* ``POST /v1/chat/completions`` — the canonical scratchpad of :func:`distillab.data.steps` in one of three
  phrasings, with a *slip* (one wrong intermediate result, so a wrong final answer) whose probability grows
  with difficulty and temperature; ``n``, ``seed``, ``temperature``, ``max_tokens`` (``finish_reason:
  "length"`` when it runs out); ``logprobs`` / ``top_logprobs`` (at most ``--max-logprobs``, default 20, as
  vLLM enforces); ``chat_template_kwargs.enable_thinking`` for a thinking teacher: the working goes to
  ``message.reasoning`` (vLLM 0.30.0's field; ``reasoning_content`` elsewhere) with a heavy-tailed number of
  re-checks, each of which may catch a slip, so longer traces are more often right;
  ``usage.completion_tokens_details.reasoning_tokens``.
* ``POST /v1/completions`` — the scoring path of on-policy distillation: ``echo`` with ``max_tokens: 0`` and
  ``prompt_logprobs`` returns the teacher's log-probability of every token of a text you send (the first
  entry is ``None``), which is all the per-token reward r_t = log π_teacher − log π_student needs (PRIMER §4).
* ``GET /metrics`` (vLLM names: generation/prompt tokens, successes), ``/health``, ``/v1/models``,
  ``/version`` — which says ``"simulated": true``.

How its log-probabilities are made up: the simulated teacher puts 0.9 on the next token of its own canonical
completion while a text follows it, 0.1 spread over 49 alternatives at the first token that leaves it, and
1/50 on every token after that. Its "tokenizer" splits text into words and punctuation (leading spaces
attached), not BPE: token counts are the right order of magnitude, not a Qwen tokenizer's. ``profile="student"``
serves a weaker model (more slips, its own phrasing) to play the student in the notebooks. Every response
carries ``x-distillab-simulated: true``. The 04 lab's ``servelab.fakeserver`` and thinking-lab's fake are this
server's siblings; ``llm-d-inference-sim`` is the upstream tool for load (it has no task knowledge).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
import socket
import threading
import time
import uuid
import zlib

from aiohttp import web

from . import data as D
from . import metrics as M

SIM = {"x-distillab-simulated": "true"}
TOKEN = re.compile(r"\n+|[^\S\n]*\w+|[^\S\n]*[^\w\s]|[^\S\n]+")
PHRASINGS = ("Let's work it out step by step.\n", "Working:\n", "Step by step:\n")
CHECKS = ("Let me double-check that step.", "Checking the result again from the start.",
          "Verifying each intermediate value once more.", "Recomputing to be sure.")
PROFILES = {
    # slip = P(one wrong step) at temperature 1 for a difficulty-4 problem; greedy keeps greedy_share of it
    "teacher": {"model": "Qwen/Qwen2.5-1.5B-Instruct", "slip": 0.30, "greedy_share": 0.35, "on_path": 0.9},
    "student": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "slip": 0.65, "greedy_share": 0.5, "on_path": 0.75},
    "thinker": {"model": "Qwen/Qwen3-1.7B", "slip": 0.55, "greedy_share": 0.35, "on_path": 0.9},
}
MAX_LOGPROBS = 20
VOCAB = 151_936          # ids are hashed into Qwen's vocab_size so they look like real ids; they are not


def tokens(text: str) -> list:
    """The fake tokenizer: newlines on their own, words and punctuation with leading spaces attached (a BPE-like
    pre-tokenisation); ``"".join`` restores the text."""
    return TOKEN.findall(text or "")


def token_id(tok: str) -> int:
    return zlib.crc32(tok.encode()) % VOCAB


def _rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def slip(p: D.Problem, lines: list, rng: random.Random) -> tuple:
    """One plausible mistake, carried through to the answer: ``(lines, wrong answer)``. A running sum mod 5 is
    knocked off at a random step and every later sum shifts with it; otherwise the final step's result changes
    (a number by a small amount, a weekday or a name to another one)."""
    lines = list(lines)
    if p.kind == "modsum":
        head, sums = lines[0].split(": ")
        vals = list(map(int, sums.split()))
        j, off = rng.randrange(len(vals)), rng.choice([1, 2, 3, 4])
        vals = vals[:j] + [(v + off) % D.MODSUM_BASE for v in vals[j:]]
        lines[0] = f"{head}: {' '.join(map(str, vals))}"
        return lines, str(vals[-1])
    last = lines[-1]
    if p.answer.lstrip("-").isdigit():
        m = list(re.finditer(r"-?\d+", last))[-1]
        v = int(m.group(0))
        wrong = str(v + rng.choice([1, 2]) if p.kind == "count" else v + rng.choice([-10, -2, -1, 1, 2, 10]))
        lines[-1] = last[: m.start()] + wrong + last[m.end():]
        return lines, wrong
    pool = D.DAYS if p.kind == "days" else [n.lower() for n in D.NAMES]
    wrong = rng.choice([x for x in pool if x != p.answer])
    lines[-1] = re.sub(p.answer, wrong.capitalize(), last, flags=re.I)
    return lines, wrong


class Generation:
    """One simulated completion: its text parts and the log-probability of each token."""

    def __init__(self, card: dict, p: D.Problem | None, question: str, *, temperature: float, seed, index: int,
                 thinking: bool):
        # greedy decoding is deterministic: every greedy sample of a question is the same (slip included)
        rng = _rng(card["model"], question, "greedy") if temperature == 0 else \
            _rng(card["model"], question, seed, index, round(temperature, 3))
        self.thinking = thinking
        if p is None:
            self.reasoning, self.content, self.correct = None, "I can only answer the lab's generated problems.", None
        else:
            difficulty = p.difficulty / 4
            p_slip = card["slip"] * difficulty * (card["greedy_share"] + (1 - card["greedy_share"]) * min(temperature, 2))
            lines = D.steps(p)
            slipped = rng.random() < p_slip
            checks = 0
            if thinking:                          # heavy-tailed re-checking; each check may catch the slip
                checks = min(40, int(rng.lognormvariate(math.log(2 + 2 * p.difficulty), 0.8)))
                for _ in range(checks):
                    if slipped and rng.random() < 0.25:
                        slipped = False
            answer = p.answer
            if slipped:
                lines, answer = slip(p, lines, rng)
            phrase = PHRASINGS[0] if temperature == 0 else rng.choice(PHRASINGS)
            if thinking:
                body = [phrase.strip()] + lines
                for c in range(checks):
                    body += [CHECKS[c % len(CHECKS)]] + lines[-1:]
                self.reasoning = "\n".join(body)
                self.content = f"The answer is \\boxed{{{answer}}}."
            else:
                self.reasoning = None
                self.content = phrase + "\n".join(lines + [f"Answer: \\boxed{{{answer}}}"])
            self.correct = answer == p.answer
        self.checks = checks if p is not None else 0
        on = card["on_path"]
        self.logprobs = []                         # the sampler's view of its own tokens (confident, not certain)
        for t in tokens(self.reasoning or "") + tokens(self.content or ""):
            self.logprobs.append((t, math.log(on) if rng.random() < 0.85 else math.log(rng.uniform(0.3, on))))

    def text_tokens(self) -> tuple:
        return tokens(self.reasoning or ""), tokens(self.content or "")


def score_text(card: dict, question: str, completion: str) -> list:
    """The simulated teacher's log-probability of each token of ``completion`` after ``question`` (see the
    module docstring): high on its own canonical path (any of its phrasings, then the right working), low at
    the first token that leaves it, uniform after."""
    p = D.find(question)
    paths = [tokens(ph + D.scratchpad(p)) for ph in PHRASINGS] if p else []
    out, alive = [], paths
    for i, t in enumerate(tokens(completion)):
        nxt = {c[i] for c in alive if i < len(c)}
        if alive and t in nxt:
            out.append(math.log(card["on_path"] / len(nxt)))
            alive = [c for c in alive if i < len(c) and c[i] == t]
        elif alive:
            out.append(math.log((1 - card["on_path"]) / 49))
            alive = []
        else:
            out.append(math.log(1 / 50))
    return out


class FakeTeacher:
    """The simulated teacher behind an aiohttp app. ``start()`` serves it on a daemon thread and returns the
    base URL (also a context manager); ``serve_forever()`` for ``python -m distillab fake``."""

    def __init__(self, profile: str = "teacher", *, model: str | None = None, host: str = "127.0.0.1", port: int = 0,
                 max_logprobs: int = MAX_LOGPROBS, reasoning_field: str = "reasoning"):
        self.card = dict(PROFILES[profile])
        self.profile = profile
        self.model = model or self.card["model"]
        self.host, self.port, self.max_logprobs, self.field = host, port, max_logprobs, reasoning_field
        self.reg = M.Registry({"model_name": self.model, "engine": "0"})
        for name in (M.PROMPT_TOKENS, M.GENERATION_TOKENS):
            self.reg.counter(name, 0.0)
        self.reg.gauge(M.RUNNING, 0)
        self.reg.gauge(M.WAITING, 0)
        self._counter = 0
        self._loop = self._thread = self._stopping = self._runner = None
        self._ready = threading.Event()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/version", self._version)
        app.router.add_get("/v1/models", self._models)
        app.router.add_get("/metrics", self._metrics)
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_post("/v1/completions", self._completions)
        return app

    async def _health(self, request):
        return web.Response(status=200, headers=SIM)

    async def _version(self, request):
        return web.json_response({"version": "distillab-fakeserver", "simulated": True, "profile": self.profile})

    async def _models(self, request):
        return web.json_response({"object": "list", "data": [{"id": self.model, "object": "model",
                                                              "owned_by": "distillab (simulated)", "max_model_len": 4096}]},
                                 headers=SIM)

    async def _metrics(self, request):
        return web.Response(text=self.reg.render(), content_type="text/plain", headers=SIM)

    def _seed(self, body):
        if body.get("seed") is not None:
            return body["seed"]
        self._counter += 1
        return f"auto-{self._counter}"

    def _count(self, prompt_toks: int, gens: list, reasons: list) -> None:
        self.reg.counter(M.PROMPT_TOKENS, prompt_toks * len(gens))
        self.reg.counter(M.GENERATION_TOKENS, sum(len(g.logprobs) for g in gens))
        for r in reasons:
            self.reg.counter(M.REQUEST_SUCCESS, 1, {"finished_reason": r})

    def _check_logprobs(self, k) -> str | None:
        if k is not None and k > self.max_logprobs:
            return f"Requested sample logprobs of {k}, which is greater than max allowed: {self.max_logprobs}"
        return None

    def _top(self, tok: str, lp: float, k: int, rng: random.Random) -> list:
        """``k`` alternatives: the token itself plus made-up neighbours sharing the rest of the mass."""
        out = [{"token": tok, "logprob": lp, "bytes": list(tok.encode())}]
        rest = max(1e-9, 1 - math.exp(lp))
        for j in range(1, k):
            alt = f" alt{j}"
            out.append({"token": alt, "logprob": math.log(rest * 0.5 ** j), "bytes": list(alt.encode())})
        return sorted(out, key=lambda e: -e["logprob"])[:k]

    async def _chat(self, request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error(400, "body is not JSON")
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return _error(400, "messages must be a non-empty list")
        k = body.get("top_logprobs") if body.get("logprobs") else None
        if (err := self._check_logprobs(k)):
            return _error(400, err)
        question = next((m.get("content") or "" for m in reversed(msgs) if m.get("role") == "user"), "")
        ctk = body.get("chat_template_kwargs") or {}
        thinking = bool(ctk.get("enable_thinking", self.profile == "thinker"))
        temperature = float(body.get("temperature", 1.0))
        n = int(body.get("n") or 1)
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
        seed = self._seed(body)
        p = D.find(question)
        prompt_toks = len(tokens("".join(m.get("content") or "" for m in msgs))) + 4 * len(msgs)
        choices, gens, reasons = [], [], []
        for i in range(n):
            g = Generation(self.card, p, question, temperature=temperature, seed=seed, index=i, thinking=thinking)
            r_toks, c_toks = g.text_tokens()
            finish = "stop"
            if max_tokens is not None and len(r_toks) + len(c_toks) > int(max_tokens):
                finish, cut = "length", int(max_tokens)
                c_toks = c_toks[: max(0, cut - len(r_toks))]
                r_toks = r_toks[:cut]
                g.logprobs = g.logprobs[:cut]
            reasoning = "".join(r_toks).strip() or None if thinking else None
            content = "".join(c_toks) or None
            msg = {"role": "assistant", "content": content, self.field: reasoning, "tool_calls": []}
            lp = None
            if body.get("logprobs"):
                rng = _rng("top", question, seed, i)
                lp = {"content": [{"token": t, "logprob": v, "bytes": list(t.encode()),
                                   "top_logprobs": self._top(t, v, int(k or 0), rng) if k else []}
                                  for t, v in g.logprobs]}
            choices.append({"index": i, "message": msg, "logprobs": lp, "finish_reason": finish, "stop_reason": None})
            gens.append(g)
            reasons.append(finish)
        self._count(prompt_toks, gens, reasons)
        comp = sum(len(g.logprobs) for g in gens)
        usage = {"prompt_tokens": prompt_toks, "completion_tokens": comp, "total_tokens": prompt_toks + comp}
        if thinking:
            usage["completion_tokens_details"] = {"reasoning_tokens": sum(len(tokens(c["message"][self.field] or ""))
                                                                          for c in choices)}
        return web.json_response({"id": "chatcmpl-" + uuid.uuid4().hex[:16], "object": "chat.completion",
                                  "created": int(time.time()), "model": self.model, "choices": choices,
                                  "usage": usage}, headers=SIM)

    async def _completions(self, request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error(400, "body is not JSON")
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            return _error(400, "this fake server takes one prompt string")
        for key in ("logprobs", "prompt_logprobs"):
            if (err := self._check_logprobs(body.get(key))):
                return _error(400, err.replace("sample logprobs", "prompt logprobs" if key == "prompt_logprobs" else "sample logprobs"))
        if body.get("prompt_logprobs") is not None and body.get("stream"):
            return _error(400, "prompt_logprobs is not compatible with streaming")
        max_tokens = int(body.get("max_tokens", 16))
        echo = bool(body.get("echo"))
        if echo and max_tokens == 0:              # vLLM: echo without generation still samples one token
            max_tokens = 1
        seed = self._seed(body)
        p = D.find(prompt)
        ptoks = tokens(prompt)
        qend = prompt.find(p.question) + len(p.question) if p else len(prompt)
        qend = prompt.find(D.SUFFIX, qend) + len(D.SUFFIX) if p and D.SUFFIX in prompt[qend:] else qend
        head = "<|im_start|>assistant\n"            # a ChatML-rendered prompt: the answer starts after this header
        qend = prompt.find(head, qend) + len(head) if head in prompt[qend:] else qend
        pl = body.get("prompt_logprobs")
        if pl is None and echo:
            pl = body.get("logprobs")
        prompt_logprobs = None
        if pl is not None:
            prompt_logprobs, pos = [None], len(ptoks[0]) if ptoks else 0
            scored = score_text(self.card, prompt[:qend], prompt[qend:])
            j = 0
            head_rng = _rng("prompt", prompt[:qend])
            for t in ptoks[1:]:
                start = pos
                pos += len(t)
                if start >= qend and j < len(scored):
                    v = scored[j]
                    j += 1
                else:
                    v = math.log(head_rng.uniform(0.02, 0.6))
                prompt_logprobs.append({str(token_id(t)): {"logprob": v, "rank": 1 if v > math.log(0.5) else 2,
                                                           "decoded_token": t}})
        g = Generation(self.card, p, prompt, temperature=float(body.get("temperature", 1.0)), seed=seed, index=0,
                       thinking=False)
        gen_toks = tokens(g.content)[:max_tokens]
        text = "".join(gen_toks)
        choice = {"index": 0, "text": (prompt if echo else "") + text,
                  "logprobs": None, "finish_reason": "length" if len(gen_toks) >= max_tokens else "stop",
                  "prompt_logprobs": prompt_logprobs}
        if body.get("logprobs") is not None:
            choice["logprobs"] = {"tokens": gen_toks, "token_logprobs": [v for _, v in g.logprobs[: len(gen_toks)]],
                                  "text_offset": [], "top_logprobs": None}
        g.logprobs = g.logprobs[: len(gen_toks)]
        self._count(len(ptoks), [g], [choice["finish_reason"]])
        return web.json_response({"id": "cmpl-" + uuid.uuid4().hex[:16], "object": "text_completion",
                                  "created": int(time.time()), "model": self.model, "choices": [choice],
                                  "usage": {"prompt_tokens": len(ptoks), "completion_tokens": len(gen_toks),
                                            "total_tokens": len(ptoks) + len(gen_toks)}}, headers=SIM)

    # -- lifecycle -------------------------------------------------------------------------------------------
    async def _serve(self):
        self._stopping = asyncio.Event()
        self._runner = web.AppRunner(self.app(), access_log=None)
        await self._runner.setup()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        self.port = sock.getsockname()[1]
        await web.SockSite(self._runner, sock).start()
        self._ready.set()
        try:
            await self._stopping.wait()
        finally:
            await self._runner.cleanup()

    def start(self) -> str:
        def run():
            self._loop = asyncio.new_event_loop()
            self._loop.run_until_complete(self._serve())
            self._loop.close()
        self._thread = threading.Thread(target=run, name="distillab-fakeserver", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("fake teacher did not start")
        return self.url

    def stop(self) -> None:
        if self._loop and self._stopping and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stopping.set)
        if self._thread:
            self._thread.join(timeout=10)

    def serve_forever(self) -> None:
        print(f"fake vLLM teacher (simulated, profile {self.profile}) serving {self.model} on "
              f"http://{self.host}:{self.port or '<auto>'}")
        try:
            asyncio.run(self._serve())
        except KeyboardInterrupt:
            pass

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"object": "error", "message": message, "type": "BadRequestError", "code": status},
                             status=status, headers=SIM)
