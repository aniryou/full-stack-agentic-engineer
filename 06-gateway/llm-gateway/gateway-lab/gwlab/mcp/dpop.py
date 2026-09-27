"""DPoP proofs (RFC 9449): a signer, a proof builder and a verifier that also checks server nonces.

The one idea: a DPoP-bound token is useless without the private key that signs a fresh proof per request —
`htm` and `htu` pin the request, `iat` and `jti` its freshness, `ath` the token, `nonce` a value the *server*
chose (RFC 9449 §8 for the authorization server: 400 `use_dpop_nonce`; §9 for the resource server: 401 with
`WWW-Authenticate: DPoP error="use_dpop_nonce"`; the new value arrives in a `DPoP-Nonce` header). The client keeps
one nonce per server and uses the latest. The identity lab's `agentsec.identity.tokens.DPoP.verify()` checks
`jti`, `ath` and `cnf.jkt` but never issues or checks a nonce: that gap is what this module closes. DPoP is not
part of the MCP authorization spec; it is RFC 9449 layered on top.

Signers: `ES256Signer` with `cryptography` (a real asymmetric key, as RFC 9449 §4.2 requires) when it is
installed; otherwise `HmacStandInSigner`, a **labelled stand-in** that uses HMAC so the flow runs with the
standard library — RFC 9449 forbids MAC algorithms, so a proof it signs is not DPoP-conformant, and the fake
servers verify it only through a shared key registry that a real server could never have.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
import uuid

HMAC_LABEL = "HMAC stand-in: RFC 9449 §4.2 requires an asymmetric alg; proofs signed with it are NOT DPoP-conformant"
STAND_IN_KEYS: dict[str, bytes] = {}           # kid -> secret, shared with the fake servers (a stand-in only)


def b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def jwk_thumbprint(jwk: dict) -> str:
    """RFC 7638: SHA-256 over the required members in lexicographic order, base64url."""
    req = {"EC": ("crv", "kty", "x", "y"), "RSA": ("e", "kty", "n"), "oct": ("kid", "kty")}[jwk["kty"]]
    canon = json.dumps({k: jwk[k] for k in req}, separators=(",", ":"), sort_keys=True)
    return b64url(hashlib.sha256(canon.encode()).digest())


class ES256Signer:
    alg = "ES256"
    conformant = True
    label = "ES256 (P-256) via cryptography"

    def __init__(self):
        from cryptography.hazmat.primitives.asymmetric import ec
        self._ec = ec
        self.key = ec.generate_private_key(ec.SECP256R1())

    def jwk(self) -> dict:
        n = self.key.public_key().public_numbers()
        return {"kty": "EC", "crv": "P-256", "x": b64url(n.x.to_bytes(32, "big")), "y": b64url(n.y.to_bytes(32, "big"))}

    def sign(self, data: bytes) -> bytes:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
        r, s = decode_dss_signature(self.key.sign(data, self._ec.ECDSA(hashes.SHA256())))
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")         # JWS wants raw r || s

    @staticmethod
    def verify(jwk: dict, data: bytes, sig: bytes) -> bool:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
        pub = ec.EllipticCurvePublicNumbers(int.from_bytes(b64url_decode(jwk["x"]), "big"),
                                            int.from_bytes(b64url_decode(jwk["y"]), "big"), ec.SECP256R1()).public_key()
        try:
            pub.verify(encode_dss_signature(int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big")), data,
                       ec.ECDSA(hashes.SHA256()))
            return True
        except InvalidSignature:
            return False


class HmacStandInSigner:
    alg = "HS256"
    conformant = False
    label = HMAC_LABEL

    def __init__(self):
        self.kid = "standin-" + secrets.token_hex(4)
        self.secret = secrets.token_bytes(32)
        STAND_IN_KEYS[self.kid] = self.secret

    def jwk(self) -> dict:
        return {"kty": "oct", "kid": self.kid}                  # the secret is never in the proof

    def sign(self, data: bytes) -> bytes:
        return hmac.new(self.secret, data, hashlib.sha256).digest()


def default_signer():
    """ES256 when `cryptography` is installed, else the labelled HMAC stand-in."""
    try:
        import cryptography  # noqa: F401
        return ES256Signer()
    except ImportError:
        return HmacStandInSigner()


def htu(url: str) -> str:
    return url.split("#")[0].split("?")[0]


def ath(access_token: str) -> str:
    return b64url(hashlib.sha256(access_token.encode("ascii")).digest())


def make_proof(signer, method: str, url: str, access_token: str | None = None, nonce: str | None = None,
               now: float | None = None) -> str:
    header = {"typ": "dpop+jwt", "alg": signer.alg, "jwk": signer.jwk()}
    claims = {"jti": uuid.uuid4().hex, "htm": method.upper(), "htu": htu(url), "iat": int(now or time.time())}
    if access_token:
        claims["ath"] = ath(access_token)
    if nonce:
        claims["nonce"] = nonce
    signing_input = (b64url(json.dumps(header, separators=(",", ":")).encode()) + "." +
                     b64url(json.dumps(claims, separators=(",", ":")).encode()))
    return signing_input + "." + b64url(signer.sign(signing_input.encode()))


class ProofError(Exception):
    def __init__(self, error: str, description: str):
        super().__init__(f"{error}: {description}")
        self.error, self.description = error, description


def verify_proof(proof: str, *, method: str, url: str, access_token: str | None = None, nonce: str | None = None,
                 seen_jti: set | None = None, max_age: int = 300, algs=("ES256", "HS256")) -> tuple[dict, str]:
    """Verify one proof; returns (claims, jkt). `nonce` is the value this server currently expects (None = no
    nonce required). Raises ProofError("use_dpop_nonce", ...) when the nonce is missing or stale."""
    try:
        h64, c64, s64 = proof.split(".")
        header, claims = json.loads(b64url_decode(h64)), json.loads(b64url_decode(c64))
    except ValueError as e:
        raise ProofError("invalid_dpop_proof", f"malformed: {e}") from e
    if header.get("typ") != "dpop+jwt" or "jwk" not in header or header.get("alg") not in algs:
        raise ProofError("invalid_dpop_proof", "typ must be dpop+jwt with an embedded jwk and an accepted alg")
    jwk, data, sig = header["jwk"], (h64 + "." + c64).encode(), b64url_decode(s64)
    if header["alg"] == "ES256":
        if "d" in jwk or not ES256Signer.verify(jwk, data, sig):
            raise ProofError("invalid_dpop_proof", "bad signature")
    else:
        secret = STAND_IN_KEYS.get(jwk.get("kid", ""))
        if secret is None or not hmac.compare_digest(hmac.new(secret, data, hashlib.sha256).digest(), sig):
            raise ProofError("invalid_dpop_proof", "bad signature (stand-in)")
    if claims.get("htm") != method.upper() or claims.get("htu") != htu(url):
        raise ProofError("invalid_dpop_proof", "htm/htu do not match the request")
    if abs(time.time() - int(claims.get("iat", 0))) > max_age:
        raise ProofError("invalid_dpop_proof", "iat too old")
    if access_token is not None and claims.get("ath") != ath(access_token):
        raise ProofError("invalid_dpop_proof", "ath does not match the access token")
    if seen_jti is not None:
        if claims.get("jti") in seen_jti:
            raise ProofError("invalid_dpop_proof", "jti replayed")
        seen_jti.add(claims.get("jti"))
    if nonce is not None and claims.get("nonce") != nonce:
        raise ProofError("use_dpop_nonce", "a fresh server nonce is required")
    return claims, jwk_thumbprint(jwk)
