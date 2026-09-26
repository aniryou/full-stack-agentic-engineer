# %% [markdown]
# # 04 · Egress and secrets: the proxy the sandbox talks to, so it never holds a key
#
# **Tier:** T0 — laptop / Colab CPU / CI, free, seconds. The proxy runs as a real container behind a
# NetworkPolicy in the lab (`../sandbox-lab`, notebook 03); the logic is all here, over a loopback server.
#
# ## The one-minute version
# The sandbox has no network of its own and no secrets. When code legitimately needs an outside host, it
# talks to an **egress proxy** inside the trust boundary; the proxy checks the host against an **allowlist**
# and, for allowed hosts, **injects the credential** on the way out. The secret lives only in the proxy —
# the same "gateway path" the identity primer describes (the agent never sees the raw credential), one layer
# down, and the header-injection pattern managed services like E2B use host-side. Two honest edges: a
# default-deny egress policy also blocks **DNS**, so you re-open DNS explicitly or let the proxy resolve
# names; and HTTPS through a `CONNECT` tunnel can't have headers injected without terminating TLS, so the
# teaching proxy brokers plain HTTP and refuses `CONNECT`. Enforcement is the **network**, not an env var:
# `HTTP_PROXY` is advisory; the NetworkPolicy is what makes the proxy the only way out. Primer:
# `../PRIMER.md` §4 (network and secrets). Reuses the identity primer's token-exchange/gateway pattern (§3.5, §5).

# %%
import io

from sandboxcore import EgressProxy, ProxyPolicy

# %% [markdown]
# ## Worked example 1 — allowlist and credential injection
# We use a fake upstream (`echo_opener`) that reports which headers arrived, so you can see the proxy add an
# `Authorization` the caller never set. An off-allowlist host is refused with 403 before any request goes out.

# %%
class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def echo_opener(req, timeout=5):
    return _Resp(f"upstream saw Authorization={req.headers.get('Authorization', '<none>')}".encode())


policy = ProxyPolicy(allowlist=("api.github.com",),
                     inject={"api.github.com": {"Authorization": "Bearer sk-SECRET"}})
proxy = EgressProxy(policy, opener=echo_opener)
print("allowed  ->", proxy.fetch("GET", "http://api.github.com/user"))
print("blocked  ->", proxy.fetch("GET", "http://data-exfil.example/x")[:2])
for e in proxy.events:
    print(f"  log: host={e.host:22} decision={e.decision:5} status={e.status} injected={e.injected}")

# %% [markdown]
# The sandboxed code sent no credential and never learns one; the proxy attached it. The log records the
# header **name** it injected, never the value — so the audit trail is safe to keep.
#
# ## Worked example 2 — the proxy refuses CONNECT (and why)
# The teaching proxy brokers plain HTTP so it can read and rewrite headers. An HTTPS `CONNECT` tunnel is
# opaque — you cannot inject a header without terminating TLS with a certificate the sandbox trusts. So the
# proxy refuses `CONNECT` and says what a production proxy does instead.

# %%
print("A CONNECT would return 405: the proxy cannot inject into an opaque TLS tunnel.")
print("Production options: (a) terminate TLS at the proxy with a sandbox-trusted CA; (b) the proxy is the")
print("TLS client and the sandbox speaks plain HTTP to it over loopback / a cluster ClusterIP.")

# %% [markdown]
# ## Worked example 3 — the network is the enforcement, not the env var
# `HTTP_PROXY` only asks a well-behaved client to use the proxy; malicious code ignores it and opens a raw
# socket. What *forces* traffic through the proxy is the network layer: a default-deny egress NetworkPolicy
# that opens only the proxy (and DNS). The rendered policy from notebook 03 does exactly this.

# %%
from sandboxcore import SandboxPolicy

allow = next(o for o in SandboxPolicy(egress_allowlist=("api.github.com",)).render_k8s()
             if o["metadata"]["name"] == "sandbox-egress-to-proxy")
print("egress rules the sandbox pod is allowed:")
for rule in allow["spec"]["egress"]:
    dst = rule["to"][0]
    ports = [p.get("port") for p in rule.get("ports", [])]
    print(f"  to={list(dst.values())[0]} ports={ports}")

# %% [markdown]
# ## Exercise 4.1 — the allowlist check
# Implement `proxy_allows(policy, url)` returning True only when the URL's host is on the allowlist. (The
# proxy calls this before doing anything; deny-by-default means an unknown host never gets a request.)

# %%
import urllib.parse

# %% exercise
def proxy_allows(policy, url):
    ### BEGIN SOLUTION
    host = urllib.parse.urlparse(url).hostname or ""
    return policy.allows(host)
    ### END SOLUTION

# %% check
pol = ProxyPolicy(allowlist=("api.github.com",))
assert proxy_allows(pol, "http://api.github.com/repos")
assert not proxy_allows(pol, "http://evil.example/steal")
assert not proxy_allows(pol, "http://api.github.com.evil.example/x")   # not an exact match
print("✅ exact-host allowlist; look-alike domains do not match")

# %% [markdown]
# ## Exercise 4.2 — inject a credential the sandbox never holds
# Implement `outbound_headers(policy, host, caller_headers)`: start from the caller's headers, then add the
# injected headers for `host` from `policy.inject`. Return `(headers, injected_names)`. The injected value
# must overwrite anything the caller tried to set for that name (the caller does not control it).

# %% exercise
def outbound_headers(policy, host, caller_headers):
    ### BEGIN SOLUTION
    headers = dict(caller_headers)
    injected = []
    for name, value in policy.inject.get(host, {}).items():
        headers[name] = value
        injected.append(name)
    return headers, injected
    ### END SOLUTION

# %% check
pol = ProxyPolicy(allowlist=("h",), inject={"h": {"Authorization": "Bearer REAL"}})
headers, injected = outbound_headers(pol, "h", {"Authorization": "Bearer FAKE-from-sandbox", "Accept": "*/*"})
assert headers["Authorization"] == "Bearer REAL"     # the sandbox's attempt was overwritten
assert injected == ["Authorization"] and headers["Accept"] == "*/*"
print("✅ the proxy owns the credential; the sandbox cannot set or read it")

# %% [markdown]
# ## Exercise 4.3 — the DNS trap
# A reviewer proposes "default-deny egress, no exceptions" as the strongest posture. What breaks, and what
# is the safer shape? Set `answer` to the letter.
#
# - **a** — nothing breaks; deny-all egress is complete on its own.
# - **b** — DNS breaks (deny-all blocks port 53 too); re-opening DNS broadly re-opens a DNS-exfiltration
#   channel, so the sandbox should reach only the proxy and let the proxy resolve names.
# - **c** — only HTTPS breaks; DNS is always allowed by Kubernetes.

# %% exercise
### BEGIN SOLUTION
answer = "b"
### END SOLUTION

# %% check
assert answer == "b"
print("✅ deny-all blocks DNS; prefer sandbox -> proxy only, proxy resolves and reaches the allowlist")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "The sandbox holds no secrets and has no network. When code needs an allowed
# host, it goes through an egress proxy inside the boundary: the proxy checks the host against an allowlist
# and injects the credential outbound, so the secret lives in one hardened place and the sandboxed code
# never sees it — the identity primer's gateway path, one layer down. Enforcement is the network, not an
# environment variable: a default-deny egress NetworkPolicy opens only the proxy and DNS, because
# `HTTP_PROXY` is advisory and malicious code ignores it. Two honest edges: deny-all also blocks DNS, so I
# let the proxy resolve names rather than re-opening port 53 broadly, which would itself be an exfiltration
# channel; and I broker plain HTTP so I can inject headers — HTTPS needs TLS termination at the proxy with a
# trusted CA. The audit log records which header was injected, never its value."
#
# **Drill questions**
# 1. *Where does the API key live, and who can read it?* — Only in the proxy. The sandboxed code cannot read
#    or set it; the proxy attaches it on the way out to allowed hosts.
# 2. *Is `HTTP_PROXY=...` enough to force traffic through the proxy?* — No. It is advisory; code can open a
#    raw socket. The NetworkPolicy (deny-all egress except the proxy and DNS) is what enforces it.
# 3. *Why not just allow DNS everywhere?* — Broad DNS is an exfiltration channel (data in query names). Let
#    the proxy resolve names; the sandbox reaches only the proxy.
