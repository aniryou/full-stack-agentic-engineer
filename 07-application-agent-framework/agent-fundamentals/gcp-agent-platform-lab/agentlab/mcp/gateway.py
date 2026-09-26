"""An egress gateway between agents and MCP servers (notebooks 05 and 06). Teaching subset.

The agent still addresses the server's canonical URL; every request goes
through the gateway, which does what Agent Gateway does on Google Cloud, in
miniature:

* identifies the *agent* (``X-Agent-Identity``, standing in for the SPIFFE id
  an mTLS gateway extracts) — the principal policy and audit key on;
* checks the mirrored headers against the body *before* routing, so the tool
  name it authorises is the one the server will execute;
* evaluates a deny-by-default ``Policy`` (agent × server × tool glob, plus an
  optional predicate over the arguments — the CEL condition);
* screens serialised arguments and results (Model Armor's job): prompt
  injection and sensitive data are blocked with an error naming the finding;
* forbids token passthrough: ``Authorization`` is forwarded only when the
  verifier confirms the token's ``aud`` is the destination server; anything
  else is stripped and audited;
* writes an audit record per call and keeps per-tool counters.

Prompts and screening reduce how often the agent tries the wrong thing; the
policy here is what stops it from succeeding (§4.5).
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Callable, Mapping

from ..auth.oauth import OAuthError
from . import protocol as p
from .protocol import McpError
from .transport import HttpTriple, InProcessTransport, Transport

UNKNOWN_DESTINATION = -32090
POLICY_DENIED = -32091
SCREENING_BLOCKED = -32092
IDENTITY_REQUIRED = -32093

Screen = Callable[[str], list[str]]
TokenVerifier = Callable[[str], Mapping[str, Any]]


# ------------------------------------------------------------------ policy
@dataclass(frozen=True)
class Rule:
    """One line of policy. Globs on agent, server and tool; ``condition(args)`` narrows further."""
    agent: str = "*"
    server: str = "*"
    tool: str = "*"
    effect: str = "allow"                                    # "allow" | "deny"
    condition: Callable[[dict[str, Any]], bool] | None = None
    name: str = ""

    def matches(self, agent: str, server: str, tool: str, args: Mapping[str, Any]) -> bool:
        if not (fnmatchcase(agent, self.agent) and fnmatchcase(server, self.server) and fnmatchcase(tool, self.tool)):
            return False
        if self.condition is None:
            return True
        try:
            return bool(self.condition(dict(args)))
        except Exception:  # noqa: BLE001 - a predicate that crashes must fail closed
            return False


@dataclass
class Decision:
    allowed: bool
    reason: str
    rule: Rule | None = None


@dataclass
class Policy:
    """Deny by default; an explicit deny beats any allow (IAM semantics)."""
    rules: list[Rule] = field(default_factory=list)

    def decide(self, agent: str, server: str, tool: str, args: Mapping[str, Any]) -> Decision:
        matched = [r for r in self.rules if r.matches(agent, server, tool, args)]
        denies = [r for r in matched if r.effect == "deny"]
        if denies:
            return Decision(False, f"denied by rule {denies[0].name or denies[0]}", denies[0])
        allows = [r for r in matched if r.effect == "allow"]
        if allows:
            return Decision(True, f"allowed by rule {allows[0].name or allows[0]}", allows[0])
        return Decision(False, f"no rule allows {agent} -> {server}/{tool} (deny by default)")

    def reaches(self, agent: str, server: str) -> bool:
        """May the agent talk to the server at all (discovery, listing, task polling)?"""
        return any(r.effect == "allow" and fnmatchcase(agent, r.agent) and fnmatchcase(server, r.server) for r in self.rules)


# ------------------------------------------------------------------- audit
@dataclass
class AuditRecord:
    agent: str
    server: str
    method: str
    tool: str | None
    decision: str            # allow | deny | blocked_args | blocked_result | rejected | error
    status: int
    latency_ms: float
    token: str = "none"      # none | forwarded | stripped:<reason>
    detail: str = ""
    ts: float = field(default_factory=time.time)


class Gateway:
    def __init__(self, registry: Mapping[str, Transport], policy: Policy, token_verifier: TokenVerifier | None = None,
                 screen: Screen | None = None):
        self.registry = dict(registry)
        self.policy = policy
        self.token_verifier = token_verifier
        self.screen = screen
        self.audit: list[AuditRecord] = []
        self.counters: dict[str, Counter[str]] = defaultdict(Counter)

    def transport_for(self, server_name: str) -> InProcessTransport:
        """What an agent is handed: a transport addressed to the server's URL whose bytes go through the gateway."""
        return InProcessTransport(self, base_url=self.registry[server_name].base_url)

    # -- the proxy path -----------------------------------------------------
    async def handle(self, method: str, url: str, headers: Mapping[str, str], body: bytes) -> HttpTriple:
        t0 = time.perf_counter()
        hdrs = {k.lower(): v for k, v in headers.items()}
        agent = hdrs.get(p.HEADER_AGENT_IDENTITY.lower(), "")
        server = self._resolve(url)
        rpc_method, tool, args, request_id = self._describe(method, body, hdrs)
        try:
            if not agent:
                raise McpError(IDENTITY_REQUIRED, "agent identity required (X-Agent-Identity)", http_status=401)
            if server is None:
                raise McpError(UNKNOWN_DESTINATION, f"no registered server for {url}", http_status=404)
            if method == "POST":
                self._check_mirrored_headers(hdrs, rpc_method, tool)
                self._enforce_policy(agent, server, rpc_method, tool, args)
                self._screen_or_raise(json.dumps(args, sort_keys=True), "blocked_args")
            forward, token_note = self._credential_policy(hdrs, server)
            status, rheaders, rbody = await self.registry[server].request(method, url, forward, body)
            if method == "POST" and status == 200 and tool is not None:
                # Screening a result cannot undo a side effect; it keeps poisoned content out of the model's context.
                self._screen_or_raise(rbody.decode("utf-8", "replace"), "blocked_result")
            self._record(agent, server, rpc_method, tool, "allow" if status < 400 else "error", status, t0, token_note)
            return status, rheaders, rbody
        except McpError as e:
            self._record(agent, server or "?", rpc_method, tool, _decision_for(e), e.http_status, t0, "none", e.message)
            return e.http_status, {"content-type": "application/json"}, json.dumps(p.error(request_id, e)).encode()

    # -- steps --------------------------------------------------------------
    def _resolve(self, url: str) -> str | None:
        """Longest registry URL that prefixes the request URL. The registry, not the agent, knows where servers live."""
        best: tuple[int, str] | None = None
        for name, transport in self.registry.items():
            base = transport.base_url.rstrip("/")
            if (url == base or url.startswith(base + "/")) and (best is None or len(base) > best[0]):
                best = (len(base), name)
        return best[1] if best else None

    @staticmethod
    def _describe(method: str, body: bytes, hdrs: Mapping[str, str]) -> tuple[str, str | None, dict[str, Any], Any]:
        """(rpc method, tool name, arguments, id) from the body — or from headers when the body cannot be parsed."""
        if method != "POST":
            return method, None, {}, None
        try:
            payload = json.loads(body)
            params = payload.get("params") or {}
            tool = params.get("name") if payload.get("method") == "tools/call" else None
            return str(payload.get("method")), tool, dict(params.get("arguments") or {}), payload.get("id")
        except (ValueError, AttributeError):
            return hdrs.get(p.HEADER_METHOD.lower(), "?"), hdrs.get(p.HEADER_NAME.lower()), {}, None

    @staticmethod
    def _check_mirrored_headers(hdrs: Mapping[str, str], rpc_method: str, tool: str | None) -> None:
        """Decide policy on the same name the server will execute: reject header/body disagreement here."""
        if hdrs.get(p.HEADER_METHOD.lower()) != rpc_method:
            raise McpError(p.HEADER_MISMATCH, f"HeaderMismatch: Mcp-Method={hdrs.get(p.HEADER_METHOD.lower())!r} but body says {rpc_method!r}")
        if rpc_method == "tools/call" and hdrs.get(p.HEADER_NAME.lower()) != tool:
            raise McpError(p.HEADER_MISMATCH, f"HeaderMismatch: Mcp-Name={hdrs.get(p.HEADER_NAME.lower())!r} but body says {tool!r}")

    def _enforce_policy(self, agent: str, server: str, rpc_method: str, tool: str | None, args: Mapping[str, Any]) -> None:
        if tool is not None:
            decision = self.policy.decide(agent, server, tool, args)
            if not decision.allowed:
                raise McpError(POLICY_DENIED, f"policy: {decision.reason}", http_status=403)
        elif not self.policy.reaches(agent, server):
            raise McpError(POLICY_DENIED, f"policy: {agent} may not reach {server}", http_status=403)

    def _screen_or_raise(self, text: str, decision: str) -> None:
        """``decision`` is the audit label: ``blocked_args`` or ``blocked_result``."""
        findings = self.screen(text) if self.screen else []
        if findings:
            raise McpError(SCREENING_BLOCKED, f"blocked by screening: {'; '.join(findings)}",
                           data={"decision": decision, "findings": findings}, http_status=403)

    def _credential_policy(self, hdrs: Mapping[str, str], server: str) -> tuple[dict[str, str], str]:
        """Forward ``Authorization`` only when the token was minted for this destination; otherwise strip it.

        A gateway that cannot verify a token cannot vouch for it, so 'no verifier' also means 'strip'.
        Forwarding a token bound to another audience would turn the gateway into a laundering step.
        """
        forward = {k: v for k, v in hdrs.items() if k != "authorization"}
        auth = hdrs.get("authorization", "")
        if not auth:
            return forward, "none"
        if self.token_verifier is None:
            return forward, "stripped:no verifier configured"
        try:
            claims = self.token_verifier(auth[7:].strip() if auth.lower().startswith("bearer ") else auth)
        except (OAuthError, ValueError) as e:
            return forward, f"stripped:invalid token ({e})"
        expected = self.registry[server].base_url
        if claims.get("aud") != expected:
            return forward, f"stripped:audience {claims.get('aud')!r} is not {expected!r}"
        forward["authorization"] = auth
        return forward, "forwarded"

    def _record(self, agent: str, server: str, method: str, tool: str | None, decision: str, status: int,
                t0: float, token: str, detail: str = "") -> None:
        self.audit.append(AuditRecord(agent, server, method, tool, decision, status, (time.perf_counter() - t0) * 1000, token, detail))
        self.counters[f"{server}/{tool or method}"][decision] += 1
        if token.startswith("stripped"):
            self.counters[f"{server}/{tool or method}"]["token_stripped"] += 1


def _decision_for(e: McpError) -> str:
    if e.code == POLICY_DENIED:
        return "deny"
    if e.code == SCREENING_BLOCKED:
        return e.data["decision"]
    return "rejected"
