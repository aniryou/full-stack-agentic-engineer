# %% [markdown]
# # 04 · Egress and secrets: the proxy the sandbox talks to, so it never holds a key
#
# **Tier:** T0 — laptop / Colab CPU / CI, free, seconds. Everything runs over loopback servers this notebook
# starts. The proxy runs as a real container behind a NetworkPolicy in the lab (`../sandbox-lab`, notebook
# 03); the logic is all here.
#
# ## The one-minute version
# The sandbox has no network of its own and no secrets. When code legitimately needs an outside host, it
# talks to an **egress proxy** inside the trust boundary; the proxy checks the host against an **allowlist**
# and, for allowed hosts, **injects the credential** on the way out. The secret lives only in the proxy —
# the same "gateway path" the identity primer describes (the agent never sees the raw credential), one layer
# down, and the header-injection pattern managed services like E2B use host-side. To keep it there the proxy
# must also **not follow redirects** (a 302 would replay the injected header to a host nobody allowed),
# **drop the caller's own credential headers**, and **redact** an upstream that echoes the credential back.
# Two honest edges: a default-deny egress policy also blocks **DNS**, and the safe shape keeps it blocked —
# the sandbox reaches only the proxy (by `hostAliases`), and the proxy resolves names; and HTTPS through a
# `CONNECT` tunnel can't have headers injected without terminating TLS, so the teaching proxy brokers plain
# HTTP and refuses `CONNECT`. Enforcement is the **network**, not an env var: `HTTP_PROXY` is advisory, and
# code that opens its own socket ignores it — which worked example 5 shows. Primer: `../PRIMER.md` §4
# (network and secrets). Reuses the identity primer's token-exchange/gateway pattern (§3.5, §5).

# %%
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sandboxcore import EgressProxy, ProxyPolicy, ProxyServer


class Upstream:
    """A loopback 'API' that records what arrives; /echo reflects the Authorization header, /redirect 302s."""

    def __init__(self, redirect_to=None):
        self.seen = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.seen.append((self.path, self.headers.get("Authorization")))
                if self.path.startswith("/redirect") and redirect_to:
                    self.send_response(302)
                    self.send_header("Location", redirect_to)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = f"upstream saw Authorization={self.headers.get('Authorization')}".encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


attacker = Upstream()                                              # "localhost": NOT allowlisted
api = Upstream(redirect_to=f"http://localhost:{attacker.port}/steal")   # "127.0.0.1": allowlisted
policy = ProxyPolicy(allowlist=("127.0.0.1",), inject={"127.0.0.1": {"Authorization": "Bearer sk-SECRET"}})
print("allowlisted API on 127.0.0.1:%d, attacker on localhost:%d" % (api.port, attacker.port))

# %% [markdown]
# ## Worked example 1 — allowlist and credential injection
# The sandbox sends no credential; the proxy attaches one for the allowed host. The upstream here *echoes*
# the header back, as a misbehaving API might — the proxy redacts the value before the body reaches the
# sandbox. An off-allowlist host is refused with 403 before any request goes out.

# %%
proxy = EgressProxy(policy)
print("allowed ->", proxy.fetch("GET", f"http://127.0.0.1:{api.port}/echo")[:2])
print("         the API itself received:", api.seen[-1][1])
print("blocked ->", proxy.fetch("GET", f"http://localhost:{attacker.port}/x")[:2])
for e in proxy.events:
    print(f"  log: host={e.host:10} decision={e.decision:5} status={e.status} injected={e.injected} {e.note}")

# %% [markdown]
# The log records the header **name** it injected, never the value — so the audit trail is safe to keep.
#
# ## Worked example 2 — a redirect is not followed
# The allowed API answers `302 Location: http://localhost:<attacker>/steal`. Python's default `urlopen`
# would follow it *and copy the injected `Authorization` header onto the new request* — handing the
# credential to a host nobody allowed, while the log shows only the allowed one. The proxy returns the 302
# to the caller instead; asking for the new URL is a new request, checked against the allowlist again.

# %%
print("redirect ->", proxy.fetch("GET", f"http://127.0.0.1:{api.port}/redirect")[:1], proxy.events[-1].note)
print("the attacker received:", attacker.seen or "nothing")

# %% [markdown]
# ## Worked example 3 — the proxy refuses CONNECT (a real request)
# The teaching proxy brokers plain HTTP so it can read and rewrite headers. An HTTPS `CONNECT` tunnel is
# opaque — you cannot inject a header without terminating TLS with a certificate the sandbox trusts. Start the
# proxy as a server and send it both kinds of request the way a sandboxed client would.

# %%
import http.client

with ProxyServer(policy) as srv:
    c = http.client.HTTPConnection(srv.host, srv.port, timeout=5)
    c.request("CONNECT", f"127.0.0.1:{api.port}")
    print("CONNECT ->", c.getresponse().status)
    c.close()
    c = http.client.HTTPConnection(srv.host, srv.port, timeout=5)
    c.request("GET", f"http://127.0.0.1:{api.port}/echo", headers={"Proxy-Authorization": "Basic sandbox"})
    r = c.getresponse()
    print("GET via the proxy ->", r.status, r.read())
    c.close()
print("Production options: (a) terminate TLS at the proxy with a sandbox-trusted CA; (b) the proxy is the")
print("TLS client and the sandbox speaks plain HTTP to it over loopback / a pinned ClusterIP.")

# %% [markdown]
# ## Worked example 4 — the network is the enforcement, not the env var
# `HTTP_PROXY` only asks a well-behaved client to use the proxy. What *forces* traffic through the proxy is
# the network layer: a default-deny egress NetworkPolicy that opens only the proxy — and **not DNS**. The
# rendered pod finds the proxy through `hostAliases` pointing at the proxy Service's pinned ClusterIP, so it
# needs no resolver, and the proxy resolves the allowlisted names itself.

# %%
from sandboxcore import SandboxPolicy

objs = SandboxPolicy(egress_allowlist=("api.github.com",)).render_k8s()
allow = next(o for o in objs if o["metadata"]["name"] == "sandbox-egress-to-proxy")
print("egress rules the sandbox pod is allowed:")
for rule in allow["spec"]["egress"]:
    print(f"  to={rule['to']} ports={[p.get('port') for p in rule.get('ports', [])]}")
pod = next(o for o in objs if o["kind"] == "Job")["spec"]["template"]["spec"]
print("dnsPolicy:", pod["dnsPolicy"], "| hostAliases:", pod["hostAliases"])

# %% [markdown]
# ## Worked example 5 — what a process sandbox does with a socket (the undeclared exfiltration)
# The agent's policy check reads the hosts the model *declares* it needs. A hijacked model that declares
# `attacker.example` is refused before running — but only because it said so. The same model can just not
# declare anything and open a socket. The process sandbox has no network control, so this **leaks**, and
# the audit log records an ordinary `allow`. That is the whole case for the network layer.

# %%
from sandboxcore import SCENARIO_OUTCOMES, LoopbackTrap, SandboxAgent, injection_scenarios

pol = SandboxPolicy(egress_allowlist=())
with LoopbackTrap() as trap:
    scen = injection_scenarios(pol, trap=(trap.host, trap.port))
    declared = SandboxAgent(scen["exfiltrate_declared"], pol)
    declared.run("look this up")
    undeclared = SandboxAgent(scen["exfiltrate_undeclared"], pol)
    res = undeclared.run("summarise this page")
    hit = trap.hit
print("declared   :", declared.audit.events[0].decision, "|", SCENARIO_OUTCOMES["exfiltrate_declared"])
print("undeclared :", res.executions[0].exit_reason, res.executions[0].stdout.strip(), "| trap hit:", hit,
      "| audit:", [e.decision for e in undeclared.audit.events])
print("            ", SCENARIO_OUTCOMES["exfiltrate_undeclared"])

# %% [markdown]
# ## Exercise 4.1 — the allowlist check
# Implement `proxy_allows(policy, url)` returning True only for a plain `http://` URL whose host is on the
# allowlist. (The proxy calls this before doing anything; deny-by-default means an unknown host never gets a
# request.) Watch for look-alike hosts and for schemes the proxy cannot broker.

# %%
import urllib.parse

# %% exercise
def proxy_allows(policy, url):
    ### BEGIN SOLUTION
    parts = urllib.parse.urlparse(url)
    return parts.scheme == "http" and policy.allows(parts.hostname or "")
    ### END SOLUTION

# %% check
pol = ProxyPolicy(allowlist=("api.github.com",))
cases = ["http://api.github.com/repos", "http://evil.example/steal", "http://api.github.com.evil.example/x",
         "https://api.github.com/user", "http://API.github.com@evil.example/x", "ftp://api.github.com/"]
for url in cases:
    want = EgressProxy(pol, opener=lambda *a, **k: None).fetch("GET", url)[0] != 403
    assert proxy_allows(pol, url) == want, url
print("✅ exact-host allowlist, http only; look-alikes and userinfo tricks do not match")

# %% [markdown]
# ## Exercise 4.2 — the headers that leave the proxy
# Implement `outbound_headers(policy, host, caller_headers)`: drop every caller header whose lower-cased name
# is in `sandboxcore.proxy.DROP_HEADERS` (hop-by-hop headers and anything carrying a credential), then add
# the injected headers for `host` from `policy.inject`. Return `(headers, injected_names)`. The check compares
# with the proxy's own implementation on hostile input.

# %% exercise
from sandboxcore.proxy import DROP_HEADERS


def outbound_headers(policy, host, caller_headers):
    ### BEGIN SOLUTION
    headers = {k: v for k, v in caller_headers.items() if k.lower() not in DROP_HEADERS}
    injected = []
    for name, value in policy.inject.get(host, {}).items():
        headers[name] = value
        injected.append(name)
    return headers, injected
    ### END SOLUTION

# %% check
pol = ProxyPolicy(allowlist=("h",), inject={"h": {"Authorization": "Bearer REAL"}})
hostile = {"authorization": "Bearer FAKE-from-sandbox", "Proxy-Authorization": "Basic sandbox",
           "Cookie": "session=stolen", "Connection": "close", "Accept": "*/*"}
mine = outbound_headers(pol, "h", hostile)
assert mine == EgressProxy(pol).outbound_headers("h", hostile), mine
assert mine[0]["Authorization"] == "Bearer REAL" and "authorization" not in mine[0]
print("✅ the proxy owns the credential; the sandbox can neither set one nor forward one it found")

# %% [markdown]
# ## Exercise 4.3 — predict the undeclared exfiltration under each rung
# For each isolation level, predict whether worked example 5's raw socket reaches the attacker: `True`
# (leaks) or `False`. Levels: `"process_sandbox"`, `"process_plus_empty_netns"` (an `unshare -n` network
# namespace with only loopback-to-itself), `"container_network_none"`, and
# `"pod_default_deny_to_proxy_only"`. The check verifies the first against a live run here and the others
# against the documented behaviour (primer §2, §4–§5).

# %% exercise
leaks = {"process_sandbox": None, "process_plus_empty_netns": None,
         "container_network_none": None, "pod_default_deny_to_proxy_only": None}
### BEGIN SOLUTION
leaks = {"process_sandbox": True, "process_plus_empty_netns": False,
         "container_network_none": False, "pod_default_deny_to_proxy_only": False}
### END SOLUTION

# %% check
import hashlib
assert leaks["process_sandbox"] == hit, "worked example 5 measured this one on this machine"
assert hashlib.sha256(repr(sorted(leaks.items())).encode()).hexdigest()[:10] == "cf76374136", (
    "only the process sandbox has no network layer: re-read primer §2's ladder table")
print("✅ the process sandbox leaks a raw socket; every rung that owns the network does not")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "The sandbox holds no secrets and has no network. When code needs an allowed
# host, it goes through an egress proxy inside the boundary: the proxy checks the host against an allowlist
# and injects the credential outbound, so the secret lives in one hardened place and the sandboxed code
# never sees it — the identity primer's gateway path, one layer down. The proxy doesn't follow redirects,
# strips any credential the caller sends, and redacts an upstream that echoes the key. Enforcement is the
# network, not an environment variable or the tool call: a default-deny egress NetworkPolicy opens only the
# proxy, because `HTTP_PROXY` is advisory and the hosts a model declares are its own claim — a process
# sandbox lets a raw socket straight out. I keep DNS closed too: the pod finds the proxy through hostAliases,
# and the proxy resolves names. And I broker plain HTTP so I can inject headers — HTTPS needs TLS termination
# at the proxy with a trusted CA. The audit log records which header was injected, never its value."
#
# **Drill questions**
# 1. *Where does the API key live, and who can read it?* — Only in the proxy. The sandboxed code cannot read
#    or set it; the proxy attaches it on the way out to allowed hosts and redacts it from responses.
# 2. *Is `HTTP_PROXY=...` enough to force traffic through the proxy?* — No. It is advisory; code can open a
#    raw socket. The NetworkPolicy (deny-all egress except the proxy) is what enforces it.
# 3. *Why not just allow DNS everywhere?* — Broad DNS is an exfiltration channel (data in query names). Let
#    the proxy resolve names; the sandbox reaches only the proxy, by hostAliases.
# 4. *The allowed API returns a 302 to another host. What must the proxy do?* — Hand the 3xx back, not
#    follow it: following replays the injected credential to an unchecked host.
