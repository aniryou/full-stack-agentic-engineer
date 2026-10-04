# %% [markdown]
# # 04 · Egress and secrets: the proxy the sandbox talks to, so it never holds a key
#
# **Tier:** T0: a laptop, a Colab CPU or CI, free, seconds. Everything runs over loopback servers that this notebook
# starts. In the lab, the proxy runs as a real container behind a NetworkPolicy (`../sandbox-lab`, notebook 03). The
# logic is all here.
#
# ## The one-minute version
# The sandbox has no network of its own and no secrets. When code legitimately needs an outside host, it talks to an
# **egress proxy** inside the trust boundary. The proxy checks the host against an **allowlist**. For the hosts on the
# allowlist, it **injects the credential** on the way out.
#
# The secret lives only in the proxy. This is the same "gateway path" that the identity primer describes (the agent
# never sees the raw credential), one layer down. It is also the header-injection pattern that managed services like
# E2B use on the host side.
#
# To keep the secret there, the proxy must also do these things:
#
# - **not follow redirects**. If the proxy follows a 302, it replays the injected header to a host that nobody put on
#   the allowlist.
# - **drop the caller's own credential headers**.
# - **redact** an upstream that echoes the credential back.
#
# Two honest edges:
#
# - A default-deny egress policy also blocks **DNS**, and the safe shape keeps it blocked. The sandbox reaches only
#   the proxy (by `hostAliases`), and the proxy resolves names.
# - The proxy cannot inject headers into HTTPS through a `CONNECT` tunnel unless it terminates TLS. Thus the proxy in
#   this notebook brokers plain HTTP and refuses `CONNECT`.
#
# The **network** is the enforcement, not an env var. `HTTP_PROXY` is advisory, and code that opens its own socket
# ignores it. Worked example 5 shows this. Primer: `../PRIMER.md` §4 (network and secrets). This notebook uses the
# token-exchange and gateway pattern of the identity primer again (§3.5, §5).

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
# The sandbox sends no credential. The proxy attaches one for the host on the allowlist. The upstream here *echoes*
# the header back, as an API that behaves badly can do. The proxy redacts the value before the body reaches the
# sandbox. The proxy refuses a host that is not on the allowlist with 403, before any request goes out.

# %%
proxy = EgressProxy(policy)
print("allowed ->", proxy.fetch("GET", f"http://127.0.0.1:{api.port}/echo")[:2])
print("         the API itself received:", api.seen[-1][1])
print("blocked ->", proxy.fetch("GET", f"http://localhost:{attacker.port}/x")[:2])
for e in proxy.events:
    print(f"  log: host={e.host:10} decision={e.decision:5} status={e.status} injected={e.injected} {e.note}")

# %% [markdown]
# The log records the **name** of the header that the proxy injected, never the value. Thus the audit trail is safe
# to keep.
#
# ## Worked example 2 — a redirect is not followed
# The API on the allowlist answers `302 Location: http://localhost:<attacker>/steal`. The default `urlopen` of Python
# follows it *and copies the injected `Authorization` header onto the new request*. That gives the credential to a
# host that nobody put on the allowlist, while the log shows only the permitted host. Instead, the proxy returns the
# 302 to the caller. A request for the new URL is a new request, and the proxy examines it against the allowlist
# again.

# %%
print("redirect ->", proxy.fetch("GET", f"http://127.0.0.1:{api.port}/redirect")[:1], proxy.events[-1].note)
print("the attacker received:", attacker.seen or "nothing")

# %% [markdown]
# ## Worked example 3 — the proxy refuses CONNECT (a real request)
# The proxy in this notebook brokers plain HTTP, so it can read and rewrite headers. An HTTPS `CONNECT` tunnel is
# opaque. You cannot inject a header unless you terminate TLS with a certificate that the sandbox trusts. Start the
# proxy as a server. Then send it the two kinds of request, as a sandboxed client does.

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
# `HTTP_PROXY` only asks a well-behaved client to use the proxy. The network layer *forces* traffic through the proxy.
# That layer is a default-deny egress NetworkPolicy that opens only the proxy, and **not DNS**. The rendered pod finds
# the proxy through `hostAliases`, which point at the pinned ClusterIP of the proxy Service. Thus the pod needs no
# resolver, and the proxy itself resolves the names on the allowlist.

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
# The policy check of the agent reads the hosts that the model *declares* it needs. A hijacked model declares
# `attacker.example`, and the agent refuses it before the run. But this occurs only because the model said so. The
# same model can declare nothing and open a socket.
#
# The process sandbox has no network control, so this **leaks**, and the audit log records an ordinary `allow`. That
# is the full case for the network layer.

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
# Implement `proxy_allows(policy, url)`. It returns True only for a plain `http://` URL whose host is on the
# allowlist. The proxy calls this function before it does anything else. Deny-by-default means that an unknown host
# never gets a request. Look out for look-alike hosts and for schemes that the proxy cannot broker.

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
# Implement `outbound_headers(policy, host, caller_headers)`. First, drop every caller header whose lower-case name
# is in `sandboxcore.proxy.DROP_HEADERS`. These are the hop-by-hop headers and all headers that carry a credential.
# Then add the injected headers for `host` from `policy.inject`. Return `(headers, injected_names)`. The check
# compares your function with the implementation of the proxy itself on hostile input.

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
# For each isolation level, predict if the raw socket of worked example 5 reaches the attacker: `True` (leaks) or
# `False`. The levels are `"process_sandbox"`, `"process_plus_empty_netns"` (an `unshare -n` network namespace with
# only loopback-to-itself), `"container_network_none"` and `"pod_default_deny_to_proxy_only"`. The check compares
# the first level with a live run here. It compares the other levels with the documented behaviour (primer §2,
# §4–§5).

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
# **The two-minute version.** "The sandbox holds no secrets and has no network. When code needs a host on the
# allowlist, it goes through an egress proxy inside the boundary. The proxy checks the host against an allowlist and
# injects the credential outbound. Thus the secret lives in one hardened place, and the sandboxed code never sees it.
#
# "This is the gateway path of the identity primer, one layer down. The proxy does not follow redirects, removes each
# credential that the caller sends, and redacts an upstream that echoes the key.
#
# "The network is the enforcement, not an environment variable or the tool call. A default-deny egress NetworkPolicy
# opens only the proxy. The reason is that `HTTP_PROXY` is advisory, and the hosts that a model declares are its own
# claim. A process sandbox lets a raw socket go straight out. I also keep DNS closed: the pod finds the proxy through
# hostAliases, and the proxy resolves names.
#
# "Also, I broker plain HTTP, so I can inject headers. HTTPS needs TLS termination at the proxy with a trusted CA.
# The audit log records which header the proxy injected, never its value."
#
# **Drill questions**
# 1. *Where does the API key live, and who can read it?* Only in the proxy. The sandboxed code cannot read or set
#    it. The proxy attaches it on the way out to hosts on the allowlist, and it redacts the key from responses.
# 2. *Is `HTTP_PROXY=...` sufficient to force traffic through the proxy?* No. It is advisory, and code can open a
#    raw socket. The NetworkPolicy (deny-all egress except the proxy) enforces it.
# 3. *Why not just permit DNS everywhere?* Broad DNS is an exfiltration channel (data in query names). Let the
#    proxy resolve names. The sandbox reaches only the proxy, by hostAliases.
# 4. *The API on the allowlist returns a 302 to a different host. What must the proxy do?* Give the 3xx back, and
#    do not follow it. If the proxy follows it, it replays the injected credential to an unchecked host.
