# %% [markdown]
# # 07 · The agent's own API: streaming, tasks, idempotency, limits
#
# "API discussions" here mean the interface the agent presents to the channel or to other
# systems (Primer §3.5). This notebook builds that API **in-process on `agentlab.agents.Runner`** — no web
# framework, just the shapes: a `request(method, path, headers, body)` entry point that returns a status,
# headers and either JSON or a stream of SSE-style events. Every design point from the primer's sketch is here:
# events (not tokens) on the stream, a task handle for long-running work, `Idempotency-Key` on POSTs because
# clients retry, `429` with `Retry-After`, and `X-Agent-Version` because behaviour is part of the contract.
#
# **Primer sections:** 3.5 (the agent's own API), 2.4 (state and durable tasks), 3.2 (MCP Tasks shape), 4.4 (backpressure).
#
# In this notebook you will:
# 1. drive the API end to end: create a session, stream a turn, pause for approval, answer it, run a turn as a background task;
# 2. see a retry double-execute a turn, then make the API idempotent;
# 3. write the four mechanisms yourself: event→SSE, idempotency dedupe, the job-polling handler, a token bucket.

# %%
import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Callable
from urllib.parse import parse_qs, urlsplit

from agentlab.agents import (Event, Identity, InMemorySessionStore, LlmAgent, Runner, SessionNotFound, SessionStatus, SideEffect,
                           TaskRecord, TaskStatus, TaskStore, tool)
from agentlab.auth import AuthorizationServer, identity_from_claims
from agentlab.llm import FakeLLM, KeywordPlanner, Rule

# %% [markdown]
# ## 1. An agent worth an API
#
# Three tools with three latency/risk profiles: a quick read, an **irreversible** write (the Runner pauses for
# approval), and a slow report (the reason a `202` path exists). The model is a `KeywordPlanner`, so the whole
# notebook runs offline in a couple of seconds.

# %%
CALLS = {"get_balance": 0, "issue_refund": 0, "generate_statement": 0}


@tool
def get_balance(account_id: str) -> dict:
    """Current balance of an account."""
    CALLS["get_balance"] += 1
    return {"account_id": account_id, "balance": 1234.5, "currency": "SGD"}


@tool(side_effect=SideEffect.IRREVERSIBLE)
def issue_refund(order_id: str, amount: float) -> dict:
    """Refund an order — money moves, so the Runner asks a human first."""
    CALLS["issue_refund"] += 1
    return {"order_id": order_id, "refunded": amount}


@tool
async def generate_statement(account_id: str) -> dict:
    """Render a PDF statement (slow: ~0.3 s here, minutes in production)."""
    CALLS["generate_statement"] += 1
    await asyncio.sleep(0.3)
    return {"account_id": account_id, "pages": 4, "url": "gs://statements/acc-1/2026-08.pdf"}


planner = KeywordPlanner([
    Rule(r"balance", "get_balance", {"account_id": "acc-1"}),
    Rule(r"refund", "issue_refund", {"order_id": "ORD-1", "amount": 20.0}),
    Rule(r"statement", "generate_statement", {"account_id": "acc-1"}),
], answer_template="Done. {results}")
llm = FakeLLM(policy=planner)
INSTRUCTION = "You are a bank assistant. Be brief."
agent = LlmAgent("bank-assistant", llm, INSTRUCTION, tools=[get_balance, issue_refund, generate_statement])
runner = Runner(agent, store=InMemorySessionStore())

# The version is part of the contract: prompt hash + model. Change either and clients pinning the old one must know.
AGENT_VERSION = f"{agent.name}@{hashlib.sha1(INSTRUCTION.encode()).hexdigest()[:8]}+{llm.model_name}"
print("X-Agent-Version:", AGENT_VERSION)

# %% [markdown]
# Callers authenticate with a bearer token for the agent service; tenant and user come from the verified claims,
# never from a header the client could set. The IdP is the one from Notebook 06.

# %%
AGENT_URL = "https://assistant.bank.example"
authz = AuthorizationServer("https://idp.bank.example")
verify_agent_token = authz.verifier(AGENT_URL)
ALICE = {"Authorization": "Bearer " + authz.issue_access_token("alice", AGENT_URL, "chat", extra={"tenant": "bank-sg"})}
BOB = {"Authorization": "Bearer " + authz.issue_access_token("bob", AGENT_URL, "chat", extra={"tenant": "bank-sg"})}

# %% [markdown]
# ## 2. The API, in one class
#
# `Response.body` is a dict for JSON or an async iterator of SSE strings for a stream. Four mechanisms are
# **injected** so that the exercises can replace them one at a time: `serialize` (event → SSE), `idempotency`
# (dedupe), `task_view` (what `GET /tasks/{id}` returns) and `bucket` (rate limiting). The defaults are the
# naive versions; you will notice what each one lacks.

# %%
@dataclass
class Response:
    status: int
    headers: dict = field(default_factory=dict)
    body: Any = None                       # dict | AsyncIterator[str]


class ApiError(Exception):
    def __init__(self, status: int, error: str, detail: str = "", headers: dict | None = None):
        super().__init__(f"{status} {error}: {detail}")
        self.status, self.error, self.detail, self.headers = status, error, detail, headers or {}


def raw_sse(ev: Event) -> str:
    """Naive serializer: the raw event kind and payload. (Exercise 2.1 replaces it.)"""
    return f"event: {ev.kind}\ndata: {json.dumps(ev.payload, default=str)}\n\n"


def minimal_task_view(rec: TaskRecord) -> Response:
    """Naive polling view: status only. (Exercise 2.3 replaces it.)"""
    return Response(200, {}, {"task_id": rec.id, "status": rec.status.value})


ROUTES = [
    ("POST", re.compile(r"^/v1/sessions$"), "create_session"),
    ("POST", re.compile(r"^/v1/sessions/(?P<session_id>[^/]+)/messages$"), "post_message"),
    ("GET", re.compile(r"^/v1/sessions/(?P<session_id>[^/]+)/events$"), "list_events"),
    ("GET", re.compile(r"^/v1/tasks/(?P<task_id>[^/]+)$"), "get_task"),
    ("POST", re.compile(r"^/v1/tasks/(?P<task_id>[^/]+)/input$"), "post_input"),
    ("POST", re.compile(r"^/v1/tasks/(?P<task_id>[^/]+)/cancel$"), "cancel_task"),
]
NEEDS_IDEMPOTENCY_KEY = {"post_message", "post_input"}     # the POSTs whose retry would repeat side effects


async def collect(resp: Response) -> list[str]:
    """Drain a streamed body (what an HTTP client does as bytes arrive)."""
    return [chunk async for chunk in resp.body] if not isinstance(resp.body, dict) else []

# %%
class AgentApi:
    """The agent service (Primer §3.5) on top of a Runner. Sessions are the Runner's; turns are TaskRecords."""

    def __init__(self, runner: Runner, verify_token: Callable[[str], dict], *, version: str,
                 serialize: Callable[[Event], str] = raw_sse, idempotency=None, task_view=minimal_task_view, bucket=None):
        self.runner, self.verify_token, self.version = runner, verify_token, version
        self.serialize, self.idempotency, self.task_view, self.bucket = serialize, idempotency, task_view, bucket
        self.tasks = TaskStore()
        self._jobs: dict[str, asyncio.Task] = {}

    # -- entry point -----------------------------------------------------------------
    async def request(self, method: str, path: str, headers: dict | None = None, body: dict | None = None) -> Response:
        hdrs = {k.lower(): v for k, v in (headers or {}).items()}
        try:
            user = self._authenticate(hdrs)
            self._check_version(hdrs)
            self._rate_limit(user)
            handler_name, params = self._route(method, path)
            key = hdrs.get("idempotency-key")
            if handler_name in NEEDS_IDEMPOTENCY_KEY and not key:
                raise ApiError(400, "idempotency_key_required", "POST requires an Idempotency-Key header")
            handler = getattr(self, handler_name)
            if key and self.idempotency is not None:
                return await self._deduplicated(key, f"{method} {path} {json.dumps(body, sort_keys=True)}", handler, user, hdrs, body or {}, params)
            return self._stamp(await handler(user, hdrs, body or {}, **params))
        except ApiError as e:
            return self._stamp(Response(e.status, {"content-type": "application/json", **e.headers}, {"error": e.error, "detail": e.detail}))

    def _stamp(self, resp: Response) -> Response:
        resp.headers = {"x-agent-version": self.version, **resp.headers}
        return resp

    # -- cross-cutting concerns --------------------------------------------------------
    def _authenticate(self, hdrs: dict) -> Identity:
        auth = hdrs.get("authorization", "")
        if not auth.startswith("Bearer "):
            raise ApiError(401, "unauthorized", "bearer token required", {"www-authenticate": "Bearer"})
        try:
            return identity_from_claims(self.verify_token(auth[7:]))
        except Exception as e:  # noqa: BLE001
            raise ApiError(401, "invalid_token", str(e), {"www-authenticate": 'Bearer error="invalid_token"'})

    def _check_version(self, hdrs: dict) -> None:
        pinned = hdrs.get("x-agent-version")
        if pinned and pinned != self.version:      # a router would send this to the pinned release; one deployment cannot
            raise ApiError(409, "version_mismatch", f"this deployment serves {self.version}, not {pinned}")

    def _rate_limit(self, user: Identity) -> None:
        if self.bucket is None:
            return
        allowed, retry_after = self.bucket.try_take(user.tenant)
        if not allowed:
            raise ApiError(429, "rate_limited", f"tenant {user.tenant} over quota", {"retry-after": str(math.ceil(retry_after))})

    def _route(self, method: str, path: str) -> tuple[str, dict]:
        for verb, pattern, name in ROUTES:
            m = pattern.match(urlsplit(path).path)
            if m and verb == method:
                params = dict(m.groupdict())
                params["query"] = {k: v[0] for k, v in parse_qs(urlsplit(path).query).items()}
                return name, params
        raise ApiError(404, "not_found", f"{method} {path}")

    async def _deduplicated(self, key, fingerprint, handler, user, hdrs, body, params) -> Response:
        cached = self.idempotency.begin(key, fingerprint)          # raises 409 on conflict / in flight
        if cached is not None:
            replay = Response(cached.status, {**cached.headers, "idempotent-replayed": "true"}, cached.body)
            if isinstance(cached.body, list):
                replay.body = self._replay(cached.body)
            return self._stamp(replay)
        try:
            resp = await handler(user, hdrs, body, **params)
        except BaseException:
            self.idempotency.abort(key)                             # let the client retry a request that never happened
            raise
        resp = self._stamp(resp)
        if isinstance(resp.body, dict):
            self.idempotency.complete(key, resp)
        else:
            resp.body = self._recording(resp.body, key, resp.status, resp.headers)   # complete only once fully sent
        return resp

    async def _recording(self, source: AsyncIterator[str], key: str, status: int, headers: dict) -> AsyncIterator[str]:
        chunks = []
        async for chunk in source:
            chunks.append(chunk)
            yield chunk
        self.idempotency.complete(key, Response(status, headers, chunks))

    @staticmethod
    async def _replay(chunks: list[str]) -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    # -- handlers ----------------------------------------------------------------------
    async def create_session(self, user, hdrs, body, query) -> Response:
        session_id = body.get("session_id") or f"s_{int(time.time() * 1000) % 10_000_000}"   # customers often key by case id
        try:
            self.runner.store.get(session_id)
            return Response(200, {}, {"session_id": session_id})                      # already there: creating is idempotent
        except SessionNotFound:
            self.runner.store.get_or_create(session_id, tenant=user.tenant, user=user.subject)
            return Response(201, {}, {"session_id": session_id})

    async def post_message(self, user, hdrs, body, session_id, query) -> Response:
        message = body.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ApiError(400, "invalid_request", "body.message must be a non-empty string")
        session = self._session(session_id, user)
        if session.status == SessionStatus.AWAITING_APPROVAL:
            raise ApiError(409, "awaiting_approval", "answer the pending approval before sending a new message")
        rec = self.tasks.create(TaskRecord(kind="turn", checkpoint={"session_id": session_id, "message": message}, poll_interval_ms=100))
        if hdrs.get("prefer") == "respond-async":
            self._jobs[rec.id] = asyncio.create_task(self._run_turn(rec.id, session_id, message, user))
            return Response(202, {"location": f"/v1/tasks/{rec.id}"}, {"task_id": rec.id, "status": "working"})
        return Response(200, {"content-type": "text/event-stream", "x-task-id": rec.id}, self._stream(rec.id, session_id, message, user))

    async def _stream(self, task_id, session_id, message, user) -> AsyncIterator[str]:
        try:
            async for ev in self.runner.stream(session_id, message, user=user):
                yield self.serialize(ev)
        except Exception as e:  # noqa: BLE001
            self._finish(task_id, TaskStatus.FAILED, error={"message": str(e)})
            return
        self._settle(task_id, self.runner.store.get(session_id))

    async def _run_turn(self, task_id, session_id, message, user) -> None:
        try:
            result = await self.runner.run(session_id, message, user=user)
        except Exception as e:  # noqa: BLE001
            self._finish(task_id, TaskStatus.FAILED, error={"message": str(e)})
            return
        self._settle(task_id, result.session, error=result.error)

    async def get_task(self, user, hdrs, body, task_id, query) -> Response:
        return self.task_view(self._task(task_id))

    async def post_input(self, user, hdrs, body, task_id, query) -> Response:
        rec = self._task(task_id)
        if rec.status != TaskStatus.INPUT_REQUIRED or "approved" not in body:
            raise ApiError(409, "no_input_expected", f"task is {rec.status.value}; body needs {{'approved': bool}}")
        result = await self.runner.approve(rec.checkpoint["session_id"], bool(body["approved"]), user=user)
        self._settle(task_id, result.session, error=result.error)
        return self.task_view(self._task(task_id))

    async def cancel_task(self, user, hdrs, body, task_id, query) -> Response:
        rec = self._task(task_id)
        job = self._jobs.get(task_id)
        if job is not None and not job.done():
            job.cancel()                                             # the Runner never persisted the half-turn: the log is intact
        if not rec.status.terminal:
            self._finish(task_id, TaskStatus.CANCELLED)
        return Response(202, {}, {"task_id": task_id, "status": "cancelled"})

    async def list_events(self, user, hdrs, body, session_id, query) -> Response:
        session = self._session(session_id, user)
        after, limit = int(query.get("after", 0)), int(query.get("limit", 50))
        page = session.events[after:after + limit]
        return Response(200, {}, {"events": [ev.to_dict() for ev in page], "next_after": after + len(page), "total": len(session.events)})

    # -- helpers -----------------------------------------------------------------------
    def _session(self, session_id: str, user: Identity):
        try:
            session = self.runner.store.get(session_id)
        except SessionNotFound:
            raise ApiError(404, "session_not_found", session_id)
        if session.tenant != user.tenant or session.user != user.subject:   # a session belongs to one principal
            raise ApiError(404, "session_not_found", session_id)
        return session

    def _task(self, task_id: str) -> TaskRecord:
        try:
            return self.tasks.get(task_id)
        except KeyError:
            raise ApiError(404, "task_not_found", task_id)

    def _settle(self, task_id: str, session, error: str | None = None) -> None:
        """Map the Runner's outcome onto the task state machine (working → input_required | completed | failed)."""
        if error:
            self._finish(task_id, TaskStatus.FAILED, error={"message": error})
        elif session.status == SessionStatus.AWAITING_APPROVAL:
            call = session.pending["tool_call"]
            self._finish(task_id, TaskStatus.INPUT_REQUIRED, input_requests={"approval": {
                "message": f"Approve {call['name']}({', '.join(f'{k}={v!r}' for k, v in call['args'].items())})?", "tool_call": call}})
        else:
            self._finish(task_id, TaskStatus.COMPLETED, result={"text": session.last_final_text(), "usage": asdict(session.usage())})

    def _finish(self, task_id: str, status: TaskStatus, **fields) -> None:
        rec = self.tasks.get(task_id)
        rec.mark(status)
        for name, value in fields.items():
            setattr(rec, name, value)
        self.tasks.put(rec)


api = AgentApi(runner, verify_agent_token, version=AGENT_VERSION)

# %% [markdown]
# ## 3. Driving it
#
# A session bound to Alice's tenant and user; then a streamed turn. The stream carries **events** — model
# turns, tool calls, tool results, the final answer — so a client can render progress, not just text.

# %%
created = await api.request("POST", "/v1/sessions", ALICE, {"session_id": "case-4711"})
print(created.status, created.body, "| headers:", created.headers)
print("without a token:", (await api.request("POST", "/v1/sessions", {}, {})).status)
print("bob asking for alice's session:", (await api.request("GET", "/v1/sessions/case-4711/events", BOB)).status)

# %%
resp = await api.request("POST", "/v1/sessions/case-4711/messages", {**ALICE, "Idempotency-Key": "k-1"}, {"message": "what is my balance?"})
print(resp.status, resp.headers)
for chunk in await collect(resp):
    print(chunk, end="")

# %% [markdown]
# **Approvals are a task state, not a modal dialog.** A refund pauses the Runner (Notebook 02); on the API that
# is a task in `input_required`. The client answers with `POST /v1/tasks/{id}/input`, which resumes the same
# step. A second message in the meantime is a `409`.

# %%
resp = await api.request("POST", "/v1/sessions/case-4711/messages", {**ALICE, "Idempotency-Key": "k-2"}, {"message": "refund my order"})
events = await collect(resp)
task_id = resp.headers["x-task-id"]
print("last event on the stream:", events[-1].splitlines()[0])
print("GET task →", (await api.request("GET", f"/v1/tasks/{task_id}", ALICE)).body)
print("another message now →", (await api.request("POST", "/v1/sessions/case-4711/messages", {**ALICE, "Idempotency-Key": "k-3"}, {"message": "hello?"})).status)
answered = await api.request("POST", f"/v1/tasks/{task_id}/input", {**ALICE, "Idempotency-Key": "k-4"}, {"approved": True})
print("after approval →", answered.status, answered.body, "| refunds executed:", CALLS["issue_refund"])
print("final text:", runner.store.get("case-4711").last_final_text())

# %% [markdown]
# **Long-running work gets a handle.** `Prefer: respond-async` returns `202` and a task id immediately; the turn
# runs in the background on a durable `TaskRecord` (the MCP Tasks shape) and the client polls. Notice how little
# the naive `GET /tasks/{id}` tells you — Exercise 2.3 fixes that.

# %%
accepted = await api.request("POST", "/v1/sessions/case-4711/messages", {**ALICE, "Idempotency-Key": "k-5", "Prefer": "respond-async"}, {"message": "send me a statement"})
print(accepted.status, accepted.headers.get("location"), accepted.body)
while (polled := await api.request("GET", accepted.headers["location"], ALICE)).body["status"] == "working":
    await asyncio.sleep(0.05)
print("polled →", polled.body)
print("the event log is queryable:", (await api.request("GET", "/v1/sessions/case-4711/events?after=0&limit=3", ALICE)).body["total"], "events")

# %% [markdown]
# **The version is part of the contract.** Every response says which prompt+model release served it; a client
# pinning a different one gets a `409` instead of silently different behaviour.

# %%
print("served by:", resp.headers["x-agent-version"])
print("pinned to an old release:", (await api.request("GET", "/v1/tasks/x", {**ALICE, "X-Agent-Version": "bank-assistant@deadbeef+fake-flash"})).body)

# %% [markdown]
# **Clients retry — and without idempotency the agent runs twice.** The same request with the same key executes
# a second time: the model is called again and the tool runs again. On a refund that is money moved twice.

# %%
before = (llm.call_count, CALLS["get_balance"])
for _ in range(2):                                       # the client's timeout fired; it sends the identical request again
    await collect(await api.request("POST", "/v1/sessions/case-4711/messages", {**ALICE, "Idempotency-Key": "k-6"}, {"message": "balance again please"}))
print(f"model calls +{llm.call_count - before[0]} (two per turn), get_balance ran +{CALLS['get_balance'] - before[1]}: the turn executed twice")

# %% [markdown]
# ### Exercise 2.1 — event → SSE
#
# Implement `event_to_sse(ev)` producing the typed event names from the primer's sketch:
#
# | `Event.kind` | SSE `event:` |
# |---|---|
# | `model` | `message.delta` |
# | `tool_call` | `tool.call` |
# | `tool_result` | `tool.result` |
# | `approval_required` | `approval.required` |
# | `final` | `message.final` |
# | `error` | `error` |
# | anything else | the kind itself |
#
# Format: `id: <ev.id>\nevent: <name>\ndata: <json>\n\n` where the JSON object is the payload plus `"agent"` and
# `"step"`. The `id:` line is what lets a client resume with `Last-Event-ID` after a dropped connection.

# %% exercise
SSE_EVENT_NAMES = {"model": "message.delta", "tool_call": "tool.call", "tool_result": "tool.result",
                   "approval_required": "approval.required", "final": "message.final", "error": "error"}


def event_to_sse(ev: Event) -> str:
    ### BEGIN SOLUTION
    data = {"agent": ev.agent, "step": ev.step, **ev.payload}
    return f"id: {ev.id}\nevent: {SSE_EVENT_NAMES.get(ev.kind, ev.kind)}\ndata: {json.dumps(data, default=str)}\n\n"
    ### END SOLUTION

# %% check
def parse_sse(chunk: str) -> dict:
    fields = dict(line.split(": ", 1) for line in chunk.strip("\n").splitlines())
    fields["data"] = json.loads(fields["data"])
    return fields

sample = Event(kind="tool_call", agent="bank-assistant", step=1, payload={"id": "call_1", "name": "get_balance", "args": {"account_id": "acc-1"}})
parsed = parse_sse(event_to_sse(sample))
assert event_to_sse(sample).endswith("\n\n") and parsed["id"] == sample.id and parsed["event"] == "tool.call", parsed
assert parsed["data"]["name"] == "get_balance" and parsed["data"]["agent"] == "bank-assistant" and parsed["data"]["step"] == 1
assert parse_sse(event_to_sse(Event(kind="final", payload={"text": "hi"})))["event"] == "message.final"
assert parse_sse(event_to_sse(Event(kind="note", payload={})))["event"] == "note"
api.serialize = event_to_sse
typed = [parse_sse(c)["event"] for c in await collect(await api.request("POST", "/v1/sessions/case-4711/messages", {**ALICE, "Idempotency-Key": "k-7"}, {"message": "balance?"}))]
assert typed == ["message.delta", "tool.call", "tool.result", "message.delta", "message.final"], typed
print("✅ typed events on the stream:", typed)

# %% [markdown]
# ### Exercise 2.2 — idempotency dedupe
#
# Implement `IdempotencyCache` so that `AgentApi._deduplicated` works:
#
# * `begin(key, fingerprint)` reserves the key and returns `None` the first time; returns the stored `Response`
#   for a replay with the same fingerprint; raises `ApiError(409, "idempotency_conflict", …)` if the key was used
#   with a **different** fingerprint, and `ApiError(409, "idempotency_in_progress", …)` if the first request has
#   not completed yet;
# * `complete(key, response)` stores the response; `abort(key)` drops the reservation.
#
# (A production store puts a TTL on entries and keys them per tenant; keep the in-memory dict here.)

# %% exercise
class IdempotencyCache:
    def __init__(self):
        self._entries: dict[str, list] = {}          # key -> [fingerprint, Response | None]

    def begin(self, key: str, fingerprint: str) -> Response | None:
        ### BEGIN SOLUTION
        entry = self._entries.get(key)
        if entry is None:
            self._entries[key] = [fingerprint, None]
            return None
        if entry[0] != fingerprint:
            raise ApiError(409, "idempotency_conflict", "Idempotency-Key reused with a different request")
        if entry[1] is None:
            raise ApiError(409, "idempotency_in_progress", "the original request has not completed")
        return entry[1]
        ### END SOLUTION

    def complete(self, key: str, response: Response) -> None:
        ### BEGIN SOLUTION
        self._entries[key][1] = response
        ### END SOLUTION

    def abort(self, key: str) -> None:
        ### BEGIN SOLUTION
        self._entries.pop(key, None)
        ### END SOLUTION

# %% check
api.idempotency = IdempotencyCache()
H = {**ALICE, "Idempotency-Key": "k-8"}
first = await collect(await api.request("POST", "/v1/sessions/case-4711/messages", H, {"message": "balance once"}))
calls_after_first = (llm.call_count, CALLS["get_balance"])
replayed = await api.request("POST", "/v1/sessions/case-4711/messages", H, {"message": "balance once"})
assert replayed.headers.get("idempotent-replayed") == "true" and await collect(replayed) == first
assert (llm.call_count, CALLS["get_balance"]) == calls_after_first, "the replay must not execute anything"
conflict = await api.request("POST", "/v1/sessions/case-4711/messages", H, {"message": "a different message"})
assert conflict.status == 409 and conflict.body["error"] == "idempotency_conflict", conflict.body
pending = await api.request("POST", "/v1/sessions/case-4711/messages", {**ALICE, "Idempotency-Key": "k-9"}, {"message": "balance"})
in_flight = await api.request("POST", "/v1/sessions/case-4711/messages", {**ALICE, "Idempotency-Key": "k-9"}, {"message": "balance"})
assert in_flight.status == 409 and in_flight.body["error"] == "idempotency_in_progress", in_flight.body
await collect(pending)
assert (await api.request("POST", "/v1/sessions/case-4711/messages", {**ALICE, "Idempotency-Key": "k-9"}, {"message": "balance"})).headers.get("idempotent-replayed") == "true"
print("✅ replay returns the recorded response; conflicts and in-flight duplicates are 409; nothing ran twice")

# %% [markdown]
# ### Exercise 2.3 — the job polling handler
#
# Implement `task_view(rec)` — what `GET /v1/tasks/{id}` returns — with proper HTTP semantics for polling:
#
# * `working` → **202** with a `Retry-After` header of `ceil(poll_interval_ms / 1000)` seconds and body
#   `{"task_id", "status", "status_message"}`;
# * `input_required` → 200, body adds `"input_requests": rec.input_requests`;
# * `completed` → 200, body adds `"result": rec.result`;
# * `failed` → 200, body adds `"error": rec.error`;
# * `cancelled` → 200, the base body.

# %% exercise
def task_view(rec: TaskRecord) -> Response:
    ### BEGIN SOLUTION
    body = {"task_id": rec.id, "status": rec.status.value, "status_message": rec.status_message}
    if rec.status == TaskStatus.WORKING:
        return Response(202, {"retry-after": str(math.ceil(rec.poll_interval_ms / 1000))}, body)
    if rec.status == TaskStatus.INPUT_REQUIRED:
        body["input_requests"] = rec.input_requests
    elif rec.status == TaskStatus.COMPLETED:
        body["result"] = rec.result
    elif rec.status == TaskStatus.FAILED:
        body["error"] = rec.error
    return Response(200, {}, body)
    ### END SOLUTION

# %% check
api.task_view = task_view
working = task_view(TaskRecord(status=TaskStatus.WORKING, poll_interval_ms=1500))
assert working.status == 202 and working.headers.get("retry-after") == "2" and working.body["status"] == "working"
assert task_view(TaskRecord(status=TaskStatus.FAILED, error={"message": "boom"})).body["error"] == {"message": "boom"}
assert "result" not in task_view(TaskRecord(status=TaskStatus.CANCELLED)).body
accepted = await api.request("POST", "/v1/sessions/case-4711/messages", {**ALICE, "Idempotency-Key": "k-10", "Prefer": "respond-async"}, {"message": "statement please"})
polled = await api.request("GET", accepted.headers["location"], ALICE)
assert polled.status == 202 and polled.headers["retry-after"] == "1", (polled.status, polled.headers)
while polled.status == 202:
    await asyncio.sleep(0.05)
    polled = await api.request("GET", accepted.headers["location"], ALICE)
assert polled.status == 200 and polled.body["status"] == "completed" and "pages" in polled.body["result"]["text"], polled.body
refund = await api.request("POST", "/v1/sessions/case-4711/messages", {**ALICE, "Idempotency-Key": "k-11"}, {"message": "refund again"})
await collect(refund)
view = await api.request("GET", f"/v1/tasks/{refund.headers['x-task-id']}", ALICE)
assert view.status == 200 and view.body["status"] == "input_required" and view.body["input_requests"]["approval"]["tool_call"]["name"] == "issue_refund"
await api.request("POST", f"/v1/tasks/{refund.headers['x-task-id']}/input", {**ALICE, "Idempotency-Key": "k-12"}, {"approved": False})
print("✅ 202 + Retry-After while working, 200 with result / input_requests when settled")

# %% [markdown]
# ### Exercise 2.4 — a token bucket
#
# Implement `TokenBucket.try_take(key)` → `(allowed, retry_after_s)`. Each key (tenant) has its own bucket that
# holds at most `capacity` tokens and refills continuously at `refill_per_s`. Taking succeeds when at least one
# token is available; otherwise report how long until one is. Use `self.clock()` for time so the check can
# drive it without sleeping.

# %% exercise
class TokenBucket:
    def __init__(self, capacity: int, refill_per_s: float, clock: Callable[[], float] = time.monotonic):
        self.capacity, self.rate, self.clock = capacity, refill_per_s, clock
        self._state: dict[str, tuple[float, float]] = {}       # key -> (tokens, last refill time)

    def try_take(self, key: str) -> tuple[bool, float]:
        ### BEGIN SOLUTION
        now = self.clock()
        tokens, last = self._state.get(key, (float(self.capacity), now))
        tokens = min(float(self.capacity), tokens + (now - last) * self.rate)
        if tokens >= 1.0:
            self._state[key] = (tokens - 1.0, now)
            return True, 0.0
        self._state[key] = (tokens, now)
        return False, (1.0 - tokens) / self.rate
        ### END SOLUTION

# %% check
fake_now = [1000.0]
bucket = TokenBucket(capacity=3, refill_per_s=2.0, clock=lambda: fake_now[0])
assert [bucket.try_take("bank-sg")[0] for _ in range(3)] == [True, True, True]
allowed, wait = bucket.try_take("bank-sg")
assert not allowed and abs(wait - 0.5) < 1e-6, (allowed, wait)
assert bucket.try_take("other-tenant")[0], "buckets are per key"
fake_now[0] += 0.5
assert bucket.try_take("bank-sg")[0] and not bucket.try_take("bank-sg")[0]
fake_now[0] += 10
assert [bucket.try_take("bank-sg")[0] for _ in range(4)] == [True, True, True, False], "refill is capped at capacity"
api.bucket = TokenBucket(capacity=2, refill_per_s=1.0, clock=lambda: fake_now[0])
statuses = [(await api.request("GET", "/v1/sessions/case-4711/events", ALICE)).status for _ in range(3)]
limited = await api.request("GET", "/v1/sessions/case-4711/events", ALICE)
assert statuses == [200, 200, 429] and limited.status == 429 and limited.headers["retry-after"] == "1", (statuses, limited.headers)
api.bucket = None
print("✅ 429 with Retry-After once the tenant's bucket is empty")

# %% [markdown]
# ### Exercise 2.5 — which endpoints must be idempotent, and why
#
# Fill `IDEMPOTENCY` with one of `"required"`, `"recommended"`, `"inherent"` or `"safe"` for each endpoint, and
# write `why` — one or two sentences on the rule behind your choices. Think about what a **retried** request
# would do to the world.

# %% exercise
ENDPOINTS = ["POST /v1/sessions", "POST /v1/sessions/{id}/messages", "POST /v1/tasks/{id}/input",
             "POST /v1/tasks/{id}/cancel", "GET /v1/tasks/{id}", "GET /v1/sessions/{id}/events"]
### BEGIN SOLUTION
IDEMPOTENCY = {
    "POST /v1/sessions": "recommended",              # a retry creates an orphan session: wasteful, not dangerous (client-chosen ids make it inherent)
    "POST /v1/sessions/{id}/messages": "required",   # a retry re-runs the model and re-executes tools: refunds twice
    "POST /v1/tasks/{id}/input": "required",         # a retried approval must not approve a *later* pending call
    "POST /v1/tasks/{id}/cancel": "inherent",        # cancelling a cancelled task changes nothing
    "GET /v1/tasks/{id}": "safe",
    "GET /v1/sessions/{id}/events": "safe",
}
why = ("Clients and proxies retry on timeouts, so any POST whose repeat would repeat a side effect — running the model, "
       "executing a tool, approving an action — must carry an Idempotency-Key and return the recorded response; "
       "reads and naturally idempotent state transitions do not need one.")
### END SOLUTION

# %% check
assert set(IDEMPOTENCY) == set(ENDPOINTS) and set(IDEMPOTENCY.values()) <= {"required", "recommended", "inherent", "safe"}
assert IDEMPOTENCY["POST /v1/sessions/{id}/messages"] == "required" and IDEMPOTENCY["POST /v1/tasks/{id}/input"] == "required"
assert IDEMPOTENCY["GET /v1/tasks/{id}"] == "safe" and IDEMPOTENCY["GET /v1/sessions/{id}/events"] == "safe"
assert IDEMPOTENCY["POST /v1/tasks/{id}/cancel"] in ("inherent", "required")
assert "retr" in why.lower() and ("side effect" in why.lower() or "twice" in why.lower() or "repeat" in why.lower())
print("✅", why)

# %% [markdown]
# ## The one-minute version
#
# Sketch the API in five lines and make five points: the stream carries **events**, not tokens, so the client can
# show tool progress and approval prompts; long-running work returns a **task handle** that mirrors the MCP Tasks
# shape (working → input_required → completed | failed | cancelled), and approvals are a state in that machine;
# **idempotency on every POST** because clients retry and a retried turn is a second refund; **`429` with
# `Retry-After`** per tenant because backpressure is cheaper than an incident; and the **prompt+model version in
# the headers** because behaviour changes when they do. Then mention what is behind it: the session store with
# optimistic concurrency, the durable task record, and the queryable event log that support and audit will need.
