# %% [markdown]
# # 06 · OAuth and identity propagation
#
# "Secure agentic workflows with MCP, tool calling and OAuth" is one question: **when the agent calls a tool,
# who is it acting as, and how does the system of record know?** This notebook walks the identity chain
# hop by hop, with a toy OAuth 2.1 authorization server (HS256 JWTs, in-process — a real IdP signs
# with asymmetric keys and publishes JWKS) and the MCP server from Notebook 05. At every hop ask the four
# questions a design review wants answered: *what token is on the wire, who issued it, what audience, what scope
# — and where is it validated?*
#
# **Concept map:** see [docs/PRIMER_MAP.md](../docs/PRIMER_MAP.md); deeper in this repo: the [identity primer](../../../../06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md) §3.5 (delegation mechanics) and §7.1 (the MCP server as an OAuth 2.1 resource server).
#
# In this notebook you will:
# 1. run the MCP authorization chain: 401 → protected-resource metadata → AS metadata → PKCE → an audience-bound token;
# 2. watch that token fail at another server, step up a scope, and get exchanged (RFC 8693) for a downstream credential
#    that still carries the user's `sub` — then see the system of record enforce per-user ACLs on it;
# 3. reproduce the confused deputy with a shared service account, and map verified claims onto agentlab's `Identity`.

# %%
from agentlab.agents import InMemorySessionStore, LlmAgent, Runner, SideEffect, ToolContext, ToolPermanentError, tool
from agentlab.auth import (ACCESS_TOKEN_TYPE, AuthorizationServer, InvalidGrant, InvalidScope, InvalidToken, MixUpDetected,
                         OAuthClient, WrongAudience, authorize_with_pkce, challenge_for, discover_and_authorize,
                         identity_from_claims, make_verifier, parse_www_authenticate, peek_claims, step_up)
from agentlab.llm import call, scripted
from agentlab.mcp import Forbidden, InProcessTransport, McpClient, McpServer, Unauthorized

# %% [markdown]
# ## 0. The cast
#
# * **IdP** — `AuthorizationServer("https://idp.bank.example")`, the enterprise authorization server.
# * **assistant-app** — the agent's OAuth client (a `client_id` plus a redirect URI).
# * **orders-mcp** — the MCP server; a *resource server* for user tokens and an OAuth client of its own when it
#   exchanges them for downstream credentials (registered with the downstream scopes it may delegate).
# * **ledger** — the system of record behind the MCP server. It never sees the MCP token; it sees an exchanged one.
#
# URLs are canonical identities: they become token audiences.

# %%
AGENT_APP_URL = "https://assistant.bank.example"
ORDERS_URL = "https://orders.mcp.bank.example/mcp"
PAYMENTS_URL = "https://payments.mcp.bank.example/mcp"
LEDGER_URL = "https://ledger.core.bank.example/api"

authz = AuthorizationServer("https://idp.bank.example", scopes_supported=("orders:read", "orders:write", "ledger:read"))
authz.register_client("assistant-app", redirect_uris=[f"{AGENT_APP_URL}/oauth/callback"])
authz.register_client("orders-mcp", resource_url=ORDERS_URL, delegated_scopes=["ledger:read"])
oauth_client = OAuthClient("assistant-app", f"{AGENT_APP_URL}/oauth/callback", resolve_issuer={authz.issuer: authz}.__getitem__)

ORDERS = {"ORD-1": {"id": "ORD-1", "customer": "alice", "status": "shipped", "total": 128.0},
          "ORD-2": {"id": "ORD-2", "customer": "bob", "status": "processing", "total": 42.5}}


def show(title: str, token: str) -> None:
    """Print the claims of a JWT. peek_claims does NOT verify — display only, never for decisions."""
    claims = peek_claims(token)
    print(f"{title}\n   iss={claims['iss']}  sub={claims['sub']}  aud={claims['aud']}  scope={claims['scope']!r}"
          + (f"  act={claims['act']}" if "act" in claims else ""))

# %% [markdown]
# ## 1. Hop 1 — the user signs in
#
# Alice signs in to the assistant through SSO (OIDC). What the app holds afterwards is a token **for the app**
# — its audience is the assistant, not any tool. Let us mint that directly (the stand-in for the login).

# %%
app_token = authz.issue_access_token("alice", audience=AGENT_APP_URL, scope="openid profile chat")
show("app token (what the assistant holds after SSO):", app_token)

# %% [markdown]
# ## 2. Hop 2 — the agent calls the MCP server
#
# The orders server verifies tokens (`token_verifier`), knows its authorization server, and enforces a scope
# per tool (`required_scope`). Business entitlement — *is this Alice's order?* — is checked in the tool, from
# verified claims, never from the prompt.

# %%
@tool(required_scope="orders:read")
def get_order(order_id: str, ctx: ToolContext) -> dict:
    """Read one of the signed-in user's orders."""
    order = ORDERS.get(order_id)
    if order is None or order["customer"] != ctx.user.subject:          # entitlement check lives in the server
        raise ToolPermanentError(f"no order {order_id} for {ctx.user.subject}", type="not_found")
    return order


@tool(required_scope="orders:write", side_effect=SideEffect.IRREVERSIBLE, requires_confirmation=False)
def cancel_order(order_id: str, ctx: ToolContext) -> dict:
    """Cancel one of the signed-in user's orders."""
    order = get_order.fn(order_id, ctx)
    order["status"] = "cancelled"
    return order


orders_server = McpServer("orders", [get_order, cancel_order], resource_url=ORDERS_URL, token_verifier=authz.verify,
                          authorization_servers=[authz.issuer], scopes_supported=["orders:read", "orders:write"])
mcp = McpClient(InProcessTransport(orders_server), bearer_token=app_token)
try:
    await mcp.call_tool("get_order", {"order_id": "ORD-1"})
except Unauthorized as e:
    first_401 = e
print("HTTP", first_401.http_status, "|", first_401.message)
print("WWW-Authenticate:", first_401.headers["www-authenticate"])

# %% [markdown]
# The app token is a valid token from the right issuer for the right user — and it is still rejected, because
# its **audience** is the assistant. A token is not a universal key; it is a statement *to one audience*.
# The 401 does not just say no: it says where the server's metadata lives.
#
# ## 3. The discovery chain, hop by hop
#
# 401 → Protected Resource Metadata (RFC 9728) → AS metadata (RFC 8414) → authorization code + PKCE with a
# resource indicator (RFC 8707) → check `iss` (RFC 9207) → redeem the code → an audience-bound token.

# %%
challenge = parse_www_authenticate(first_401.headers["www-authenticate"])
print("1. challenge:", challenge)
prm = await mcp.fetch_json(challenge["resource_metadata"])
print("2. protected resource metadata:", prm)
as_meta = authz.metadata()                                   # in production: GET {issuer}/.well-known/oauth-authorization-server
print("3. AS metadata: token_endpoint =", as_meta["token_endpoint"], "| PKCE:", as_meta["code_challenge_methods_supported"])
verifier = make_verifier()
response = authz.authorize(client_id="assistant-app", redirect_uri=f"{AGENT_APP_URL}/oauth/callback", scope="orders:read",
                           resource=prm["resource"], code_challenge=challenge_for(verifier), subject="alice")
print("4. authorization response:", {k: (v[:8] + "…" if k == "code" else v) for k, v in response.items()})
assert response["iss"] == as_meta["issuer"], "mix-up attack: the code came from a different issuer"
token_response = authz.token(grant_type="authorization_code", code=response["code"], code_verifier=verifier,
                             client_id="assistant-app", resource=prm["resource"])
print("5. token response:", {k: v for k, v in token_response.items() if k != "access_token"})
show("   access token:", token_response["access_token"])

# %% [markdown]
# `discover_and_authorize` does the same five steps in one call. From here on the client presents the new token
# and the server's audience check passes:

# %%
try:
    await mcp.call_tool("get_order", {"order_id": "ORD-1"})
except Unauthorized as e:
    grant = await discover_and_authorize(oauth_client, e.headers, subject="alice", scopes={"orders:read"}, fetch_json=mcp.fetch_json)
mcp.bearer_token = grant.access_token
print("alice's own order:", (await mcp.call_tool("get_order", {"order_id": "ORD-1"}))["structuredContent"])
print("bob's order:      ", (await mcp.call_tool("get_order", {"order_id": "ORD-2"}))["structuredContent"])

# %% [markdown]
# The `iss` check matters: an attacker who can redirect the browser to a look-alike authorization server would
# otherwise obtain a code the client redeems at the real one. `authorize_with_pkce` refuses:

# %%
class Impostor(AuthorizationServer):
    def metadata(self):
        return {**super().metadata(), "issuer": authz.issuer}   # claims to be our IdP, but signs its own iss into responses

impostor = Impostor("https://idp-bank-example.attacker.net")
impostor.register_client("assistant-app", [f"{AGENT_APP_URL}/oauth/callback"])
try:
    authorize_with_pkce(OAuthClient("assistant-app", f"{AGENT_APP_URL}/oauth/callback", resolve_issuer=lambda iss: impostor),
                        authz.issuer, ORDERS_URL, "alice", {"orders:read"})
except MixUpDetected as mixup:
    print("refused:", mixup)

# %% [markdown]
# ## 4. The same token at a different server
#
# The payments server trusts the same IdP and the same user. The token still fails, because the audience is the
# orders server. This is the property that makes a stolen or leaked token useless elsewhere — and the reason
# MCP forbids servers from accepting or forwarding tokens issued for anything else.

# %%
payments_server = McpServer("payments", [get_order], resource_url=PAYMENTS_URL, token_verifier=authz.verify, authorization_servers=[authz.issuer])
try:
    await McpClient(InProcessTransport(payments_server), bearer_token=grant.access_token).call_tool("get_order", {"order_id": "ORD-1"})
except Unauthorized as e:
    print("payments →", e.http_status, e.message)

# %% [markdown]
# ## 5. Insufficient scope → step-up
#
# `cancel_order` needs `orders:write`. The server answers 403 with the scope it requires; the client re-authorizes
# for the **union** of scopes (the user consents again) and retries.

# %%
try:
    await mcp.call_tool("cancel_order", {"order_id": "ORD-1"})
except Forbidden as e:
    print("403 |", e.headers["www-authenticate"])
    grant = step_up(oauth_client, e.headers, grant, subject="alice")
mcp.bearer_token = grant.access_token
show("stepped-up token:", grant.access_token)
print("cancel →", (await mcp.call_tool("cancel_order", {"order_id": "ORD-1"}))["structuredContent"]["status"])

# %% [markdown]
# ## 6. At the MCP server: exchange, never forward
#
# The ledger is the system of record. It accepts only tokens minted **for it** (`aud = LEDGER_URL`) and enforces
# per-user ACLs itself. So the MCP server cannot pass the user's token through (wrong audience, and forbidden by
# MCP anyway); it performs an RFC 8693 **token exchange**: the IdP verifies the inbound token, checks that
# `orders-mcp` is its audience (you can exchange what you *received*, not what you observed), and issues a
# downstream token with the **same `sub`**, the ledger as audience, and an `act` claim naming the server.
#
# The ledger below still has a legacy `ledger:admin` path for a shared service account. Keep an eye on it.

# %%
class Ledger:
    """System of record: verifies aud, logs who asked, enforces ownership. Legacy admin scope bypasses ownership."""

    def __init__(self, verify, owners: dict):
        self.verify, self.owners = verify, owners
        self.balances = {"acc-1": 1234.5, "acc-2": 88.0}
        self.log = []

    def read_balance(self, token: str, account_id: str) -> dict:
        claims = self.verify(token)                       # signature, expiry, issuer and aud == LEDGER_URL
        self.log.append({"sub": claims["sub"], "act": claims.get("act", {}).get("sub"), "account": account_id})
        if "ledger:admin" not in claims["scope"].split() and self.owners.get(account_id) != claims["sub"]:
            raise PermissionError(f"{claims['sub']} may not read {account_id}")
        return {"account_id": account_id, "balance": self.balances[account_id]}


ledger = Ledger(verify=authz.verifier(LEDGER_URL), owners={"acc-1": "alice", "acc-2": "bob"})
SERVICE_ACCOUNT_TOKEN = authz.issue_access_token("orders-svc", LEDGER_URL, "ledger:admin")   # the shared credential


@tool(required_scope="orders:read")
def get_balance(account_id: str, ctx: ToolContext) -> dict:
    """Account balance from the ledger, on behalf of the signed-in user (token exchange)."""
    downstream = authz.token_exchange(ctx.user.token, audience=LEDGER_URL, scope="ledger:read", client_id="orders-mcp")
    try:
        return ledger.read_balance(downstream["access_token"], account_id)
    except PermissionError as err:
        raise ToolPermanentError(str(err), type="forbidden", hint="Tell the user this account is not theirs.")


@tool(required_scope="orders:read")
def get_balance_shared(account_id: str) -> dict:
    """ANTI-PATTERN: the same lookup with the server's shared service account."""
    return ledger.read_balance(SERVICE_ACCOUNT_TOKEN, account_id)


orders_server.add_tool(get_balance)
orders_server.add_tool(get_balance_shared)
print("alice, acc-1 via exchange →", (await mcp.call_tool("get_balance", {"account_id": "acc-1"}))["structuredContent"])
exchanged = authz.token_exchange(grant.access_token, audience=LEDGER_URL, scope="ledger:read", client_id="orders-mcp")
show("the downstream token the ledger saw:", exchanged["access_token"])
print("issued_token_type:", exchanged["issued_token_type"] == ACCESS_TOKEN_TYPE, "| lifetime ≤ the user's token:", exchanged["expires_in"] <= grant.expires_in)

# %% [markdown]
# ## 7. The confused deputy
#
# Bob signs in and asks for **Alice's** account. Through the delegated path the ledger sees `sub=bob` and
# refuses. Through the shared service account the ledger sees `sub=orders-svc` with admin scope and answers —
# the MCP server has just been used as a confused deputy: a privileged component tricked into spending its own
# broad credentials on behalf of a less-privileged requester. The defence is structural: the agent and the
# server never hold credentials broader than the user they serve, and every downstream call carries the
# requester's identity.

# %%
bob = McpClient(InProcessTransport(orders_server))
try:
    await bob.call_tool("get_balance", {"account_id": "acc-1"})
except Unauthorized as e:
    bob_grant = await discover_and_authorize(oauth_client, e.headers, subject="bob", scopes={"orders:read"}, fetch_json=bob.fetch_json)
bob.bearer_token = bob_grant.access_token
delegated = await bob.call_tool("get_balance", {"account_id": "acc-1"})
shared = await bob.call_tool("get_balance_shared", {"account_id": "acc-1"})
print("delegated token →", delegated["structuredContent"])
print("service account →", shared["structuredContent"])
print("\nwhat the ledger logged (who it thinks asked):")
for row in ledger.log[-2:]:
    print("  ", row)

# %% [markdown]
# ## 8. Verified claims → `Identity` → local tools
#
# Inside the agent runtime the same claims become a agentlab `Identity`; local tools enforce `required_scope`
# against it exactly as the MCP server did (Notebook 01). The token rides along so a tool layer can exchange it
# downstream — never to forward it.

# %%
read_only = identity_from_claims(authz.verify(bob_grant.access_token, ORDERS_URL), token=bob_grant.access_token)
read_write = identity_from_claims(authz.verify(grant.access_token, ORDERS_URL), token=grant.access_token)
print("bob  :", read_only.subject, sorted(read_only.scopes))
print("alice:", read_write.subject, sorted(read_write.scopes))


@tool(required_scope="orders:write", side_effect=SideEffect.REVERSIBLE)
def add_note(order_id: str, note: str) -> dict:
    """Attach a note to an order."""
    return {"order_id": order_id, "note": note}


async def run_as(identity):
    agent = LlmAgent("assistant", scripted(call("add_note", order_id="ORD-2", note="call back"), "Noted."), "You help with orders.", tools=[add_note])
    result = await Runner(agent, InMemorySessionStore()).run(f"s-{identity.subject}", "add a note", user=identity)
    return next(ev.payload for ev in result.session.events if ev.kind == "tool_result")

print("bob   →", (await run_as(read_only))["content"])
print("alice →", (await run_as(read_write))["content"])

# %% [markdown]
# ### Exercise 8.1 — implement the audience check
#
# A downstream service that does not hold the signing key validates tokens by **introspection** (RFC 7662):
# it asks the IdP, which answers `{"active": false}` for anything invalid or expired, else the claims. Implement
# `verify_by_introspection(token, audience)`:
#
# * call `authz.introspect(token)`; if not `active`, raise `InvalidToken("inactive token")`;
# * if `aud` is not `audience`, raise `WrongAudience(...)`;
# * return the claims without the `active` key.

# %% exercise
def verify_by_introspection(token: str, audience: str) -> dict:
    ### BEGIN SOLUTION
    info = authz.introspect(token)
    if not info.get("active"):
        raise InvalidToken("inactive token")
    if info.get("aud") != audience:
        raise WrongAudience(f"token audience {info.get('aud')!r} is not {audience!r}")
    return {k: v for k, v in info.items() if k != "active"}
    ### END SOLUTION

# %% check
good = authz.issue_access_token("alice", LEDGER_URL, "ledger:read")
claims = verify_by_introspection(good, LEDGER_URL)
assert claims["sub"] == "alice" and "active" not in claims
for bad, expected in ((grant.access_token, WrongAudience), (authz.issue_access_token("alice", LEDGER_URL, "ledger:read", ttl_s=-1), InvalidToken), ("garbage", InvalidToken)):
    try:
        verify_by_introspection(bad, LEDGER_URL)
        raise AssertionError(f"should have rejected {bad[:12]}…")
    except expected:
        pass
print("✅ audience, expiry and garbage all rejected; a good token yields claims")

# %% [markdown]
# ### Exercise 8.2 — implement token exchange
#
# Re-implement the IdP's exchange policy in a subclass. `MyIdP.token_exchange(subject_token, audience, scope=None, *, client_id)` must:
#
# 1. look up the caller with `self.registered_client(client_id)` and verify the subject token with `self.verify(subject_token)`;
# 2. raise `InvalidGrant` unless the caller's `resource_url` equals the subject token's `aud` (only the recipient may exchange);
# 3. compute the requested scopes (`scope.split()`, or the subject token's scopes when `scope` is `None`) and raise
#    `InvalidScope` unless they are all within the subject token's scopes **or** the caller's `delegated_scopes`;
# 4. build `act = {"sub": client_id}`, nesting the subject token's existing `act` under it if there is one;
# 5. mint with `self.issue_access_token(sub, audience, " ".join(sorted(requested)), ttl_s=..., extra={"act": act}, client_id=client_id)`
#    where the lifetime is `min(self.token_ttl_s, exp - now)` so the delegated token never outlives the user's;
# 6. return `{"access_token": ..., "issued_token_type": ACCESS_TOKEN_TYPE, "token_type": "Bearer"}`.

# %% exercise
class MyIdP(AuthorizationServer):
    def token_exchange(self, subject_token, audience, scope=None, *, client_id):
        ### BEGIN SOLUTION
        caller = self.registered_client(client_id)
        claims = self.verify(subject_token)
        if caller.resource_url is None or caller.resource_url != claims["aud"]:
            raise InvalidGrant("only the recipient of a token may exchange it")
        granted = set(claims["scope"].split())
        requested = set(scope.split()) if scope else granted
        if not requested <= granted | set(caller.delegated_scopes):
            raise InvalidScope(f"{client_id} may not request {sorted(requested - granted)}")
        act = {"sub": client_id}
        if claims.get("act"):
            act["act"] = claims["act"]
        ttl = min(self.token_ttl_s, int(claims["exp"] - self.clock()))
        token = self.issue_access_token(claims["sub"], audience, " ".join(sorted(requested)), ttl_s=ttl, extra={"act": act}, client_id=client_id)
        return {"access_token": token, "issued_token_type": ACCESS_TOKEN_TYPE, "token_type": "Bearer"}
        ### END SOLUTION

# %% check
idp = MyIdP("https://idp.test")
idp.register_client("orders-mcp", resource_url=ORDERS_URL, delegated_scopes=["ledger:read"])
idp.register_client("ledger-svc", resource_url=LEDGER_URL)
idp.register_client("bystander", resource_url="https://bystander.test")
user_tok = idp.issue_access_token("alice", ORDERS_URL, "orders:read orders:write", ttl_s=600)
hop1 = idp.token_exchange(user_tok, audience=LEDGER_URL, scope="ledger:read", client_id="orders-mcp")
c1 = idp.verify(hop1["access_token"], audience=LEDGER_URL)
assert c1["sub"] == "alice" and c1["act"] == {"sub": "orders-mcp"} and c1["scope"] == "ledger:read", c1
assert c1["exp"] - c1["iat"] <= 600 and hop1["issued_token_type"] == ACCESS_TOKEN_TYPE
hop2 = idp.token_exchange(hop1["access_token"], audience="https://core.test", client_id="ledger-svc")
assert peek_claims(hop2["access_token"])["act"] == {"sub": "ledger-svc", "act": {"sub": "orders-mcp"}}
for kwargs, expected in (({"client_id": "bystander"}, InvalidGrant), ({"client_id": "orders-mcp", "scope": "ledger:admin"}, InvalidScope)):
    try:
        idp.token_exchange(user_tok, audience=LEDGER_URL, **kwargs)
        raise AssertionError(f"should have rejected {kwargs}")
    except expected:
        pass
print("✅ same sub, nested act, bounded lifetime, recipient-only, no widening")

# %% [markdown]
# ### Exercise 8.3 — implement the step-up
#
# `handle_insufficient_scope(client, error, grant, subject)` receives the `Forbidden` error from the MCP client.
# Parse its `WWW-Authenticate` header with `parse_www_authenticate`; if `error` is not `insufficient_scope`,
# re-raise the original error. Otherwise call `authorize_with_pkce(client, grant.issuer, grant.resource, subject,
# <current scopes ∪ required scopes>)` and return the new grant.

# %% exercise
def handle_insufficient_scope(client, error, grant, subject):
    ### BEGIN SOLUTION
    challenge = parse_www_authenticate(error.headers.get("www-authenticate"))
    if challenge.get("error") != "insufficient_scope":
        raise error
    required = set(challenge.get("scope", "").split())
    return authorize_with_pkce(client, grant.issuer, grant.resource, subject, grant.scopes | required)
    ### END SOLUTION

# %% check
fresh = McpClient(InProcessTransport(orders_server))
try:
    await fresh.call_tool("get_order", {"order_id": "ORD-2"})
except Unauthorized as e:
    bob_ro = await discover_and_authorize(oauth_client, e.headers, subject="bob", scopes={"orders:read"}, fetch_json=fresh.fetch_json)
fresh.bearer_token = bob_ro.access_token
try:
    await fresh.call_tool("cancel_order", {"order_id": "ORD-2"})
    raise AssertionError("read-only token should not cancel")
except Forbidden as err:
    bob_rw = handle_insufficient_scope(oauth_client, err, bob_ro, "bob")
assert bob_rw.scopes == {"orders:read", "orders:write"} and bob_rw.resource == ORDERS_URL
fresh.bearer_token = bob_rw.access_token
assert (await fresh.call_tool("cancel_order", {"order_id": "ORD-2"}))["structuredContent"]["status"] == "cancelled"
try:
    handle_insufficient_scope(oauth_client, Forbidden(-32003, "policy denied", http_status=403), bob_ro, "bob")
    raise AssertionError("a plain 403 is not a step-up")
except Forbidden:
    pass
print("✅ step-up re-authorizes for", sorted(bob_rw.scopes))

# %% [markdown]
# ### Exercise 8.4 — the ACL check in the system of record
#
# Close the confused-deputy hole. `StrictLedger.authorize_read(claims, account_id)` must raise `PermissionError`
# unless `claims["sub"]` owns the account — no admin bypass, no exceptions for service accounts. (Whoever needs to
# act for a user must arrive with that user's `sub`, via exchange.)

# %% exercise
class StrictLedger(Ledger):
    def authorize_read(self, claims: dict, account_id: str) -> None:
        ### BEGIN SOLUTION
        if self.owners.get(account_id) != claims["sub"]:
            raise PermissionError(f"{claims['sub']} may not read {account_id}")
        ### END SOLUTION

    def read_balance(self, token: str, account_id: str) -> dict:
        claims = self.verify(token)
        self.authorize_read(claims, account_id)
        return {"account_id": account_id, "balance": self.balances[account_id]}

# %% check
strict = StrictLedger(verify=authz.verifier(LEDGER_URL), owners={"acc-1": "alice", "acc-2": "bob"})
alice_ledger = authz.token_exchange(grant.access_token, audience=LEDGER_URL, scope="ledger:read", client_id="orders-mcp")["access_token"]
bob_ledger = authz.token_exchange(bob_grant.access_token, audience=LEDGER_URL, scope="ledger:read", client_id="orders-mcp")["access_token"]
assert strict.read_balance(alice_ledger, "acc-1")["balance"] == 1234.5
for tok, acct in ((alice_ledger, "acc-2"), (bob_ledger, "acc-1"), (SERVICE_ACCOUNT_TOKEN, "acc-1"), (SERVICE_ACCOUNT_TOKEN, "acc-2")):
    try:
        strict.read_balance(tok, acct)
        raise AssertionError(f"{peek_claims(tok)['sub']} must not read {acct}")
    except PermissionError:
        pass
try:
    strict.read_balance(grant.access_token, "acc-1")
    raise AssertionError("a token for the orders server must not work at the ledger")
except WrongAudience:
    pass
print("✅ ownership enforced in the system of record; the service account can no longer read anyone's balance")

# %% [markdown]
# ### Exercise 8.5 — the rule, in one sentence
#
# Write `the_rule`: one sentence that states where authorisation decisions happen and what the agent may never
# hold. It should mention the system of record, the prompt, and credentials.

# %% exercise
### BEGIN SOLUTION
the_rule = ("Authorisation happens in the system of record from the user's delegated identity, never in the prompt, "
            "and the agent never holds credentials broader than the user it is serving.")
### END SOLUTION

# %% check
rule = the_rule.lower()
assert "system of record" in rule or "systems of record" in rule, "where does authorisation happen?"
assert "prompt" in rule and ("never" in rule or "not" in rule), "say what the prompt is not"
assert "credential" in rule or "token" in rule, "say what the agent may not hold"
print("✅", the_rule)

# %% [markdown]
# ## The one-minute version
#
# Draw the chain — user → agent → gateway → MCP server → system of record — and narrate one token per hop:
# the app token (aud: assistant) is useless at the tool; the agent runs the discovery chain and gets a token
# whose `aud` is the MCP server's canonical URL, with PKCE and an `iss` check on the way; the server validates
# audience and scope and answers 403 with the scope to step up to; it never forwards that token but exchanges it
# (RFC 8693) for a downstream credential with the same `sub` and an `act` claim; the system of record validates
# *its* audience and enforces the user's entitlements. Two identities travel with every call: the user's
# (what may be done) and the agent's (which tools it may reach, what audit keys on).
#
# Then name the anti-pattern before someone else does: one shared service account for all users, or the
# user's token passed straight through. Say why it fails — per-user authorisation is lost, audit is blind, and
# a confused deputy is one prompt injection away — and where a legacy system with no notion of the user forces
# you to enforce entitlements in the MCP server and state it as a limitation.
