"""Deny-by-default policy engine for tool calls (the runtime policy enforcement point).

Evaluation order (first failure wins, and every failure is a *reason* in the decision so the
audit log — and the model — can see why):

1. tool known?                      → else DENY
2. caller principal allowed?        → IAM-style member matching (principal:// or principalSet://)
3. authority mode allowed?          → own / delegated / any
4. required scopes present?         → from the delegated/own token
5. argument constraints             → max/min/enum/pattern/length
6. egress allowlist for URL args    → parsed host, https, no private ranges
7. budgets                          → per-invocation counters
8. confirmation                     → CONFIRM unless the `unless` expression holds
→ ALLOW

The ``unless`` expression is a deliberately tiny, safe language (comparisons, boolean logic,
attribute access on ``args``/``ctx``, literals) — the same role CEL plays in IAM Conditions.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..guardrails.sanitize import EgressPolicy
from ..identity.delegation import AuthorityContext, AuthorityMode
from ..identity.principals import member_matches
from .model import Authority, Policy, Tier, ToolPolicy


class Effect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"


@dataclass
class ToolCallRequest:
    tool: str
    args: dict[str, Any]
    authority: AuthorityContext
    invocation_id: str | None = None
    counters: dict[str, int] = field(default_factory=dict)  # per-invocation counters (caller-owned)
    confirmed_by: str | None = None  # set when a human already approved this exact call


@dataclass
class Decision:
    effect: Effect
    reasons: list[str] = field(default_factory=list)
    tool_policy: ToolPolicy | None = None
    hint: str | None = None

    @property
    def allowed(self) -> bool:
        return self.effect is Effect.ALLOW

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"{self.effect.value}: {'; '.join(self.reasons) or 'ok'}"


# ---- safe expression evaluation ------------------------------------------------------------
class _Attr:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as e:
            raise AttributeError(name) from e


_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Not,
    ast.UnaryOp,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Set,
)


def safe_eval(expression: str, *, args: dict[str, Any], ctx: dict[str, Any]) -> bool:
    """Evaluate a restricted boolean expression over ``args`` and ``ctx``."""
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax in policy expression: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in {"args", "ctx", "True", "False", "None"}:
            raise ValueError(f"unknown name in policy expression: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("private attribute access not allowed")
    env = {"args": _Attr(args), "ctx": _Attr(ctx), "__builtins__": {}}
    try:
        return bool(eval(compile(tree, "<policy>", "eval"), env))  # noqa: S307 - AST-restricted
    except AttributeError:
        return False


class PolicyEngine:
    def __init__(self, policy: Policy):
        self.policy = policy

    def tool_policy(self, tool: str) -> ToolPolicy | None:
        return self.policy.tools.get(tool)

    def evaluate(self, req: ToolCallRequest) -> Decision:
        tp = self.tool_policy(req.tool)
        if tp is None:
            return Decision(Effect.DENY, [f"tool {req.tool!r} is not in policy (default deny)"])
        reasons: list[str] = []
        auth = req.authority

        members = self.policy.resolve_members(tp.allow)
        if not any(member_matches(m, auth.agent) for m in members):
            reasons.append(f"agent {auth.agent.short_name} not in allow list for {req.tool}")

        if tp.authority is Authority.DELEGATED and auth.mode is not AuthorityMode.DELEGATED:
            reasons.append(f"{req.tool} requires delegated (on-behalf-of-user) authority")
        elif tp.authority is Authority.OWN and auth.mode is not AuthorityMode.OWN:
            reasons.append(f"{req.tool} must run under the agent's own authority")

        missing = set(tp.required_scopes) - set(auth.scopes)
        if missing:
            reasons.append(f"missing scopes {sorted(missing)}")

        for arg, constraint in tp.constraints.items():
            if arg in req.args:
                problem = constraint.check(req.args[arg])
                if problem:
                    reasons.append(f"arg {arg}={req.args[arg]!r} {problem}")

        if tp.tier is Tier.EXTERNAL or tp.egress_hosts:
            egress = EgressPolicy(allowed_hosts=set(tp.egress_hosts))
            for arg in tp.url_args:
                if arg in req.args and isinstance(req.args[arg], str):
                    d = egress.check(req.args[arg])
                    if not d.allowed:
                        reasons.append(f"egress blocked for {arg}: {d.reason}")

        b = self.policy.budgets
        if req.counters.get("tool_calls", 0) >= b.max_tool_calls_per_invocation:
            reasons.append(
                f"budget exceeded: {b.max_tool_calls_per_invocation} tool calls per invocation"
            )
        if (
            tp.tier is Tier.DESTRUCTIVE
            and req.counters.get("destructive_calls", 0) >= b.max_destructive_calls_per_invocation
        ):
            reasons.append(
                f"budget exceeded: {b.max_destructive_calls_per_invocation} destructive calls per invocation"
            )

        if reasons:
            return Decision(Effect.DENY, reasons, tp)

        if tp.confirmation.required and not req.confirmed_by:
            ctx = {
                "authority": auth.mode.value,
                "user": auth.user.subject if auth.user else None,
                "agent": auth.agent.short_name,
            }
            if tp.confirmation.unless and safe_eval(tp.confirmation.unless, args=req.args, ctx=ctx):
                return Decision(
                    Effect.ALLOW, [f"confirmation waived: {tp.confirmation.unless}"], tp
                )
            hint = tp.confirmation.hint or f"Approve {req.tool} with arguments {req.args}?"
            return Decision(Effect.CONFIRM, ["human confirmation required"], tp, hint=hint)

        if req.confirmed_by:
            return Decision(Effect.ALLOW, [f"confirmed by {req.confirmed_by}"], tp)
        return Decision(Effect.ALLOW, [], tp)

    def dry_run(
        self, plan: list[tuple[str, dict[str, Any]]], authority: AuthorityContext
    ) -> list[Decision]:
        """Evaluate a whole plan (plan → check → act) with budget accounting."""
        counters: dict[str, int] = {}
        out: list[Decision] = []
        for tool, args in plan:
            d = self.evaluate(
                ToolCallRequest(tool=tool, args=args, authority=authority, counters=dict(counters))
            )
            out.append(d)
            if d.effect is not Effect.DENY:
                counters["tool_calls"] = counters.get("tool_calls", 0) + 1
                if d.tool_policy and d.tool_policy.tier is Tier.DESTRUCTIVE:
                    counters["destructive_calls"] = counters.get("destructive_calls", 0) + 1
        return out
