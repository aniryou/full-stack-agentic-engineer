"""Runtime-attested agent certificates (a local stand-in for Agent Identity's X.509 substrate).

On Google Cloud, the agent runtime provisions each agent an X.509 certificate whose SAN carries
the agent's SPIFFE ID, valid for 24 hours and rotated automatically. Access tokens are bound to
that certificate (``cnf`` → ``x5t#S256`` thumbprint, RFC 8705), so a token replayed from
anywhere that cannot present the certificate is rejected.

Locally we emulate the runtime CA: it issues short-lived certificates with the SPIFFE URI SAN,
and :func:`thumbprint` produces the value that tokens are bound to.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
from dataclasses import dataclass

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from .principals import AgentIdentity


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def thumbprint(cert: x509.Certificate) -> str:
    """RFC 8705 ``x5t#S256``: base64url(SHA-256(DER))."""
    der = cert.public_bytes(serialization.Encoding.DER)
    return _b64url(hashlib.sha256(der).digest())


@dataclass(frozen=True)
class AgentCertificate:
    agent: AgentIdentity
    private_key: rsa.RSAPrivateKey
    certificate: x509.Certificate

    @property
    def thumbprint(self) -> str:
        return thumbprint(self.certificate)

    @property
    def not_after(self) -> dt.datetime:
        return self.certificate.not_valid_after_utc

    @property
    def spiffe_id(self) -> str:
        sans = self.certificate.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        uris = sans.get_values_for_type(x509.UniformResourceIdentifier)
        return uris[0]

    def is_valid_at(self, when: dt.datetime | None = None) -> bool:
        when = when or dt.datetime.now(dt.UTC)
        return self.certificate.not_valid_before_utc <= when <= self.certificate.not_valid_after_utc

    def pem(self) -> str:
        return self.certificate.public_bytes(serialization.Encoding.PEM).decode()


class LocalRuntimeCA:
    """A tiny certificate authority standing in for the agent runtime's identity substrate."""

    def __init__(self, name: str = "agentsec-local-runtime-ca") -> None:
        self.name = name
        self._key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
        now = dt.datetime.now(dt.UTC)
        self.certificate = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(self._key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=1))
            .not_valid_after(now + dt.timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .sign(self._key, hashes.SHA256())
        )

    def issue(
        self, agent: AgentIdentity, ttl: dt.timedelta = dt.timedelta(hours=24)
    ) -> AgentCertificate:
        """Issue a certificate for one agent: SPIFFE URI SAN, short validity, fresh key pair."""
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        now = dt.datetime.now(dt.UTC)
        cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, agent.short_name)]))
            .issuer_name(self.certificate.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=1))
            .not_valid_after(now + ttl)
            .add_extension(
                x509.SubjectAlternativeName([x509.UniformResourceIdentifier(agent.spiffe_id)]),
                critical=False,
            )
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(self._key, hashes.SHA256())
        )
        return AgentCertificate(agent=agent, private_key=key, certificate=cert)

    def verify(self, cert: x509.Certificate) -> bool:
        """Check the certificate was signed by this CA (issuer signature + validity window)."""
        try:
            cert.verify_directly_issued_by(self.certificate)
        except Exception:
            return False
        now = dt.datetime.now(dt.UTC)
        return cert.not_valid_before_utc <= now <= cert.not_valid_after_utc
