"""The egress proxy: allowlist, credential injection the sandbox never sees, and no CONNECT."""
import io

from sandboxcore import EgressProxy, ProxyPolicy


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def echo_opener(req, timeout=5):
    # a fake upstream that reports what header the proxy attached
    auth = req.headers.get("Authorization", "<none>")
    return _Resp(f"auth={auth}".encode())


def test_allowed_host_gets_the_injected_credential():
    pol = ProxyPolicy(allowlist=("api.github.com",),
                      inject={"api.github.com": {"Authorization": "Bearer SECRET"}})
    p = EgressProxy(pol, opener=echo_opener)
    status, body, injected = p.fetch("GET", "http://api.github.com/user")
    assert status == 200 and injected == ["Authorization"]
    assert body == b"auth=Bearer SECRET"       # the sandbox never set this; the proxy did


def test_blocked_host_is_denied_and_logged():
    p = EgressProxy(ProxyPolicy(allowlist=("api.github.com",)), opener=echo_opener)
    status, body, _ = p.fetch("GET", "http://evil.example/steal")
    assert status == 403 and b"denied" in body
    assert p.events[-1].decision == "deny" and p.events[-1].host == "evil.example"


def test_response_body_is_capped():
    def big_opener(req, timeout=5):
        return _Resp(b"x" * 5000)
    p = EgressProxy(ProxyPolicy(allowlist=("h",), max_bytes=1000), opener=big_opener)
    status, body, _ = p.fetch("GET", "http://h/big")
    assert status == 200 and len(body) <= 1000 + 40      # capped + a short truncation marker


def test_events_never_record_the_secret_value():
    pol = ProxyPolicy(allowlist=("h",), inject={"h": {"Authorization": "Bearer SECRET"}})
    p = EgressProxy(pol, opener=echo_opener)
    p.fetch("GET", "http://h/x")
    ev = p.events[-1]
    assert ev.injected == ["Authorization"]              # header NAME only
    assert "SECRET" not in repr(ev)                       # never the value
