"""Prompt/response screening with the shape of Model Armor.

Model Armor sanitises *user prompts* and *model responses* against four filter families:
prompt-injection & jailbreak, sensitive data (Sensitive Data Protection), malicious URIs, and
Responsible-AI categories. Each filter can be ``INSPECT_ONLY`` or ``INSPECT_AND_BLOCK``.
On Vertex AI you attach it per request (``modelArmorConfig.promptTemplateName`` /
``responseTemplateName``) or as project **floor settings** (baseline that applies to every
``generateContent`` call). Request-level templates take precedence over the floor, which takes
precedence over Gemini's built-in safety filters.

:class:`LocalScreener` is a deterministic, dependency-free stand-in with the same result
shape so the ADK plugin and notebooks behave identically offline. :class:`ModelArmorScreener`
calls the real API. Both return a :class:`ScreenResult`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class Enforcement(str, Enum):
    INSPECT_ONLY = "INSPECT_ONLY"
    INSPECT_AND_BLOCK = "INSPECT_AND_BLOCK"


@dataclass(frozen=True)
class Finding:
    filter: str  # pi_and_jailbreak | sdp | malicious_uri | rai
    severity: str  # LOW | MEDIUM | HIGH
    detail: str
    span: str | None = None


@dataclass
class ScreenResult:
    findings: list[Finding] = field(default_factory=list)
    enforcement: Enforcement = Enforcement.INSPECT_AND_BLOCK
    sanitized_text: str | None = None

    @property
    def matched(self) -> bool:
        return bool(self.findings)

    @property
    def blocked(self) -> bool:
        return self.matched and self.enforcement is Enforcement.INSPECT_AND_BLOCK

    def summary(self) -> str:
        if not self.findings:
            return "clean"
        return "; ".join(f"{f.filter}:{f.severity}({f.detail})" for f in self.findings)


class Screener(Protocol):
    def screen_prompt(self, text: str) -> ScreenResult: ...

    def screen_response(self, text: str) -> ScreenResult: ...


# ---- Local deterministic screener ------------------------------------------------------------
_INJECTION_PATTERNS = [
    (
        r"(?i)\bignore (all|any|the|your)? ?(previous|prior|above|earlier) (instructions|rules|guidance)",
        "override-instructions",
    ),
    (
        r"(?i)\b(disregard|forget) (your|the|all) (instructions|rules|system prompt)",
        "override-instructions",
    ),
    (r"(?i)\byou are now\b.*\b(unrestricted|developer mode|DAN)\b", "persona-jailbreak"),
    (
        r"(?i)\b(reveal|print|show|repeat) (me )?(your|the) (system prompt|instructions|hidden prompt)",
        "prompt-extraction",
    ),
    (r"(?i)\bsystem prompt\b", "prompt-reference"),
    (
        r"(?i)\b(new|updated|important) (instruction|instructions|task) (from|for) (the )?(admin|administrator|system|developer)\b",
        "authority-spoof",
    ),
    (r"(?i)\bAI assistant:?\s*(please )?(send|forward|email|post|upload)\b", "exfil-instruction"),
    (
        r"(?i)\b(send|forward|email|post|upload) (all|the|your) (data|credentials|tokens|secrets|conversation|context)\b",
        "exfil-instruction",
    ),
    (r"(?i)<\s*/?\s*(system|assistant|instruction|tool_call)\s*>", "role-tag-injection"),
    (r"(?i)\bcall the (\w+) tool with\b", "tool-steering"),
]
_SDP_PATTERNS = [
    (r"\bAIza[0-9A-Za-z\-_]{35}\b", "GOOGLE_API_KEY"),
    (r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b", "JWT"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "PRIVATE_KEY"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "US_SSN"),
    (
        r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2})[ -]?\d{4}[ -]?\d{4}[ -]?\d{3,4}\b",
        "CREDIT_CARD_NUMBER",
    ),
    (r"\b[STFG]\d{7}[A-Z]\b", "SINGAPORE_NRIC"),
]
_MALICIOUS_HOSTS = {"evil.example", "exfil.example", "attacker.example", "pastebin.example"}
_URL_RE = re.compile(r"https?://([A-Za-z0-9.-]+)(?::\d+)?(?:/[^\s\"'<>]*)?")


class LocalScreener:
    """Deterministic screener mirroring Model Armor's four filter families."""

    def __init__(
        self,
        enforcement: Enforcement = Enforcement.INSPECT_AND_BLOCK,
        *,
        malicious_hosts: set[str] | None = None,
        block_sdp_in_prompt: bool = False,
    ):
        self.enforcement = enforcement
        self.malicious_hosts = set(malicious_hosts or _MALICIOUS_HOSTS)
        self.block_sdp_in_prompt = block_sdp_in_prompt

    def _scan(self, text: str, *, direction: str) -> ScreenResult:
        findings: list[Finding] = []
        for pattern, label in _INJECTION_PATTERNS:
            m = re.search(pattern, text)
            if m:
                findings.append(Finding("pi_and_jailbreak", "HIGH", label, m.group(0)[:80]))
        for pattern, label in _SDP_PATTERNS:
            m = re.search(pattern, text)
            if m:
                sev = "HIGH" if direction == "response" or self.block_sdp_in_prompt else "MEDIUM"
                findings.append(Finding("sdp", sev, label, m.group(0)[:12] + "…"))
        for m in _URL_RE.finditer(text):
            host = m.group(1).lower()
            if host in self.malicious_hosts or any(
                host.endswith("." + h) for h in self.malicious_hosts
            ):
                findings.append(Finding("malicious_uri", "HIGH", host, m.group(0)[:80]))
        # Only HIGH findings block; MEDIUM ones are reported (inspect-only semantics per finding).
        blocking = [f for f in findings if f.severity == "HIGH"]
        result = ScreenResult(findings=findings, enforcement=self.enforcement)
        if not blocking:
            result.enforcement = Enforcement.INSPECT_ONLY if findings else self.enforcement
        return result

    def screen_prompt(self, text: str) -> ScreenResult:
        return self._scan(text, direction="prompt")

    def screen_response(self, text: str) -> ScreenResult:
        return self._scan(text, direction="response")


# ---- Real Model Armor ---------------------------------------------------------------------------
class ModelArmorScreener:
    """Calls Model Armor ``sanitizeUserPrompt`` / ``sanitizeModelResponse`` with a template.

    ``template`` is ``projects/P/locations/L/templates/T``. The client is created lazily so the
    package imports without the ``google-cloud-modelarmor`` extra installed.
    """

    def __init__(self, template: str, *, location: str | None = None, client: Any | None = None):
        self.template = template
        self.location = location or template.split("/locations/")[1].split("/")[0]
        self._client = client

    def _get_client(self):
        if self._client is None:
            from google.api_core.client_options import ClientOptions
            from google.cloud import modelarmor_v1  # lazy

            self._client = modelarmor_v1.ModelArmorClient(
                client_options=ClientOptions(
                    api_endpoint=f"modelarmor.{self.location}.rep.googleapis.com"
                )
            )
        return self._client

    @staticmethod
    def _to_result(sanitization_result: Any) -> ScreenResult:
        findings: list[Finding] = []
        match_state = getattr(sanitization_result, "filter_match_state", None)
        for name, fr in (getattr(sanitization_result, "filter_results", {}) or {}).items():
            state = None
            for attr in (
                "pi_and_jailbreak_filter_result",
                "sdp_filter_result",
                "malicious_uri_filter_result",
                "rai_filter_result",
            ):
                sub = getattr(fr, attr, None)
                if sub is not None and getattr(sub, "match_state", None) is not None:
                    state = sub.match_state
            if state is not None and "MATCH_FOUND" in str(state):
                findings.append(Finding(name, "HIGH", str(state)))
        result = ScreenResult(findings=findings, enforcement=Enforcement.INSPECT_AND_BLOCK)
        if match_state is not None and "MATCH_FOUND" not in str(match_state):
            result.findings = []
        return result

    def screen_prompt(self, text: str) -> ScreenResult:
        from google.cloud import modelarmor_v1  # lazy

        req = modelarmor_v1.SanitizeUserPromptRequest(
            name=self.template, user_prompt_data=modelarmor_v1.DataItem(text=text)
        )
        return self._to_result(
            self._get_client().sanitize_user_prompt(request=req).sanitization_result
        )

    def screen_response(self, text: str) -> ScreenResult:
        from google.cloud import modelarmor_v1  # lazy

        req = modelarmor_v1.SanitizeModelResponseRequest(
            name=self.template, model_response_data=modelarmor_v1.DataItem(text=text)
        )
        return self._to_result(
            self._get_client().sanitize_model_response(request=req).sanitization_result
        )
