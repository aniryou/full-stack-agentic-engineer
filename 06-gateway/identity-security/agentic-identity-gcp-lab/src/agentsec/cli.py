"""Command-line entry points.

agentsec demo            run the offline end-to-end demo (policy, confirmation, audit)
agentsec mcp-serve       serve the tickets MCP server (resource server) with uvicorn
agentsec policy-check    evaluate a tool call against a policy file without running anything
agentsec token-demo      mint / exchange / verify tokens and show the delegation chain
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .logging_utils import quiet_logs


def _cmd_demo(args: argparse.Namespace) -> int:
    from .agents import REFUNDS, LocalStack, Step, reset_demo_state
    from .runtime import confirm, run_turn, seed_session

    async def main() -> None:
        reset_demo_state()
        stack = LocalStack.create()
        user = {"subject": "u-ana", "email": "ana@customer.example", "tenant": "acme"}
        scopes = ["customers:read", "orders:read", "payments:refund", "agent:invoke"]
        await seed_session(
            stack.runner, user_id="u-ana", session_id="demo", user=user, scopes=scopes
        )
        stack.script(
            Step.call("lookup_customer", email="ana@customer.example"),
            Step.call("search_knowledge", query="refund policy"),
            Step.call("run_sql", query="select * from customers"),
            Step.call(
                "issue_refund",
                order_id="O-5002",
                amount=35.0,
                currency="USD",
                reason="duplicate charge",
            ),
            Step.call(
                "issue_refund",
                order_id="O-5001",
                amount=120.0,
                currency="USD",
                reason="event cancelled",
            ),
            Step.say("I refunded the 35 USD order; the 120 USD refund is awaiting your approval."),
        )
        r = await run_turn(
            stack.runner,
            user_id="u-ana",
            session_id="demo",
            message="Please refund my recent orders.",
        )
        print(r.summary())
        for p in r.pending_confirmations:
            print(f"\n[front-end] confirmation UI shows: {json.dumps(p.payload, indent=2)}")
            r2 = await confirm(
                stack.runner,
                user_id="u-ana",
                session_id="demo",
                pending=p,
                confirmed=not args.reject,
            )
            print(r2.summary())
        print("\nRefunds issued:", json.dumps(REFUNDS, indent=2))
        print("\nAudit timeline:")
        for line in stack.audit.timeline():
            print(" ", line)

    asyncio.run(main())
    return 0


def _cmd_mcp_serve(args: argparse.Namespace) -> int:
    from .audit import AuditLog, JsonLinesSink
    from .config import Settings
    from .identity import TokenIssuer
    from .mcp import build_server, serve

    settings = Settings.from_env()
    resource_url = args.resource_url or settings.mcp_audience
    issuer = TokenIssuer(issuer=settings.sts_issuer)
    audit = AuditLog([JsonLinesSink(sys.stdout)])
    server = build_server(
        issuer, resource_url=resource_url, audit=audit, require_dpop=args.require_dpop
    )
    print(
        f"tickets MCP server on http://{args.host}:{args.port}/mcp  (audience={resource_url}, issuer={issuer.issuer})",
        file=sys.stderr,
    )
    print(
        "NOTE: the issuer's signing key is generated at start-up; clients must obtain tokens from this process's STS (see notebooks) — in production the AS is Auth Manager / your IdP.",
        file=sys.stderr,
    )
    serve(server.app(require_dpop=args.require_dpop), host=args.host, port=args.port)
    return 0


def _cmd_policy_check(args: argparse.Namespace) -> int:
    from .identity import AgentIdentity, AuthorityContext, UserPrincipal
    from .policy import Policy, PolicyEngine, ToolCallRequest

    policy = Policy.from_yaml(args.policy)
    engine = PolicyEngine(policy)
    agent = AgentIdentity.parse(args.agent)
    scopes = set(args.scopes.split(",")) if args.scopes else set()
    if args.user:
        authority = AuthorityContext.delegated(
            agent, UserPrincipal(subject=args.user, email=args.user), scopes
        )
    else:
        authority = AuthorityContext.own(agent, scopes)
    call_args = json.loads(args.args) if args.args else {}
    decision = engine.evaluate(ToolCallRequest(tool=args.tool, args=call_args, authority=authority))
    print(f"{decision.effect.value.upper()}  {'; '.join(decision.reasons) or 'ok'}")
    return 0 if decision.allowed else 2


def _cmd_token_demo(args: argparse.Namespace) -> int:
    import jwt

    from .identity import (
        AgentIdentity,
        DPoP,
        LocalRuntimeCA,
        TokenIssuer,
        UserPrincipal,
        jwk_thumbprint,
        public_jwk,
    )

    agent = AgentIdentity.for_agent_engine(
        project_number="987654321098",
        location="us-central1",
        engine_id="support-agent",
        org_id="123456789012",
    )
    ca = LocalRuntimeCA()
    cert = ca.issue(agent)
    issuer = TokenIssuer()
    user = UserPrincipal(subject="u-ana", email="ana@customer.example")
    id_token = issuer.mint_user_id_token(user, audience="https://app.acme.example")
    agent_token = issuer.mint_agent_token(cert, audience=issuer.issuer)
    exchanged = issuer.exchange(
        subject_token=id_token,
        subject_token_audience="https://app.acme.example",
        actor_token=agent_token,
        actor_token_audience=issuer.issuer,
        presented_thumbprint=cert.thumbprint,
        audience="https://tickets.acme.example/mcp",
        scope="tickets:read",
    )
    claims = jwt.decode(exchanged["access_token"], options={"verify_signature": False})
    print("agent SPIFFE ID :", agent.spiffe_id)
    print("cert thumbprint :", cert.thumbprint)
    print("delegated token :", json.dumps(claims, indent=2))
    key = DPoP.generate_key()
    print("DPoP jkt        :", jwk_thumbprint(public_jwk(key)))
    return 0


def main(argv: list[str] | None = None) -> int:
    quiet_logs(logging.ERROR)
    parser = argparse.ArgumentParser(
        prog="agentsec", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo", help="offline end-to-end demo")
    p.add_argument(
        "--reject", action="store_true", help="reject the confirmation instead of approving"
    )
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("mcp-serve", help="serve the tickets MCP server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--resource-url", default=None, help="canonical server URI (audience); defaults to settings"
    )
    p.add_argument("--require-dpop", action="store_true")
    p.set_defaults(func=_cmd_mcp_serve)

    p = sub.add_parser("policy-check", help="evaluate a tool call against a policy")
    p.add_argument("--policy", default="policies/support-agent.yaml")
    p.add_argument("--agent", required=True, help="principal:// or spiffe:// agent identity")
    p.add_argument("--tool", required=True)
    p.add_argument("--args", default="{}", help="JSON arguments")
    p.add_argument("--user", default=None, help="act on behalf of this user (delegated authority)")
    p.add_argument("--scopes", default="", help="comma-separated scopes")
    p.set_defaults(func=_cmd_policy_check)

    p = sub.add_parser("token-demo", help="mint/exchange/verify demo")
    p.set_defaults(func=_cmd_token_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
