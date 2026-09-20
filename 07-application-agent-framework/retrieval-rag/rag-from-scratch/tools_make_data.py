"""Write the bundled corpus (Markdown) and the evaluation labels (qrels.json).

Run once: `python tools_make_data.py`. Kept as a script so the data is easy to
inspect and regenerate; it is not part of the learning path.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "ragkit" / "data" / "corpus"
EVAL = HERE / "ragkit" / "data" / "eval"
CORPUS.mkdir(parents=True, exist_ok=True)
EVAL.mkdir(parents=True, exist_ok=True)

DOCS = {
"pto-policy": """# Paid Time Off Policy

## Annual Leave
Full-time employees receive 20 days of annual leave per calendar year. Leave
accrues monthly and is available to book once accrued. Up to 5 unused days may
be carried over into the following year; anything beyond that is forfeited.

## Requesting Leave
Submit leave requests through the TimeOff portal at least two weeks before the
intended start date. Your line manager approves requests. Bookings during
end-of-quarter freeze periods may be declined.

## Sick Leave
Employees receive 14 days of sick leave per year. For any absence of more than
two consecutive days, a medical certificate must be uploaded to the TimeOff
portal before the absence is approved.
""",

"remote-work": """# Remote Work Policy

## Default Arrangement
Employees may work remotely up to three days per week by default. Working fully
remote (every day) requires written director approval and a review after three
months.

## Core Hours
Regardless of location, everyone is expected to be online and reachable during
core hours of 10:00 to 16:00 SGT. Meetings should be scheduled inside this
window wherever possible.

## Equipment and Connectivity
A home office stipend of SGD 500 per year is available to cover a desk, chair,
or monitor. When accessing internal systems from outside the office you must
connect through the company VPN; see the VPN setup guide for details.
""",

"expense-policy": """# Expense Policy

## Submitting a Claim
Submit expense claims through Expensily within 30 days of the date on the
receipt. Claims submitted after 30 days require a written exception from your
line manager and may be rejected.

## Receipts
An itemised receipt is required for any single expense above SGD 20. For
expenses at or below SGD 20 a receipt is encouraged but not mandatory.

## Meal Limits
Meals are reimbursed up to SGD 60 per day for domestic travel and SGD 90 per
day for international travel. Alcohol is not reimbursable. Amounts above these
daily limits are the employee's own responsibility.
""",

"security-access": """# Access Control Standard (SEC-011)

## Scope
Policy SEC-011 governs how employees authenticate and how access to sensitive
systems is granted and reviewed.

## Authentication
All employees must register a hardware security key (FIDO2) for single sign-on.
Passwords alone are never sufficient for access to internal systems.

## Production Access
Access to production systems is granted only after approval from the on-call
security lead and is time-boxed to the duration of the task. Connecting to
internal or production systems requires the approved VPN client (see the VPN
setup guide). Access rights are reviewed every quarter and revoked if unused.
""",

"vpn-setup": """# VPN Setup Guide

## Client
The approved VPN client is Meridian Connect. Install it from the internal app
catalogue; personal or third-party VPN clients are not permitted for company
systems.

## Supported Systems
Meridian Connect supports macOS 13 or later, Windows 11, and Ubuntu 22.04 or
later. Split tunneling is disabled by policy, so all traffic is routed through
the VPN while connected. Connect to the gateway at vpn.meridian.internal.

## Troubleshooting
If you see error ERR_4290 when connecting, your device certificate has expired.
Re-enrol the device in the app catalogue to issue a fresh certificate, then
reconnect. Persistent failures should be raised with the IT service desk.
""",

"incident-response": """# Incident Response Runbook

## Severity Levels
Incidents are classified SEV-1, SEV-2, or SEV-3. A SEV-1 is a customer-facing
outage or data-loss event. A SEV-2 is major degradation with a workaround. A
SEV-3 is a minor issue with limited impact.

## Escalation
For a SEV-1, the on-call security lead must be paged within 15 minutes of
detection, and the incident commander opens a war room in the #incident-bridge
channel. SEV-2 incidents are handled during business hours.

## Postmortem
Every SEV-1 and SEV-2 requires a blameless postmortem, published within five
business days, listing the timeline, root cause, and follow-up actions.
""",

"api-rate-limits": """# API Rate Limits

## Limits by Tier
Rate limits depend on your plan. The Starter tier allows 60 requests per
minute, the Growth tier allows 600 requests per minute, and the Scale tier
allows 3000 requests per minute. Enterprise plans negotiate custom limits.

## Burst Allowance
Short bursts up to twice your per-minute limit are tolerated for up to 10
seconds before throttling begins, which smooths over brief spikes.

## Exceeding the Limit
When you exceed your limit the API responds with HTTP status 429 Too Many
Requests and a Retry-After header indicating how many seconds to wait. Clients
should honour Retry-After with exponential backoff rather than retrying
immediately.
""",

"billing-faq": """# Billing FAQ

## Plans and Prices
Meridian offers four plans: Starter (free), Growth at SGD 99 per month, Scale
at SGD 499 per month, and Enterprise (custom pricing). Each paid plan raises
usage limits and adds support options.

## Billing Cycle
Plans are billed monthly by default. Choosing annual billing gives two months
free compared with paying monthly. Upgrades take effect immediately; downgrades
take effect at the start of the next billing cycle.

## Refunds
A full refund is available within 14 days of a charge if you have not exceeded
the plan's usage limits. After 14 days, charges are non-refundable.
""",

"data-retention": """# Data Retention Policy (DR-07)

## Retention Periods
Under policy DR-07, application logs are retained for 90 days and database
backups for 35 days. Audit logs are retained for one year to support
compliance reviews.

## Account Deletion
When an account is deleted, associated customer data is purged from primary
systems and backups within 30 days. A deletion certificate is available on
request.

## Protection at Rest
All personally identifiable information (PII) is encrypted at rest using
AES-256. Access to decryption keys is restricted and logged.
""",
}

for doc_id, text in DOCS.items():
    (CORPUS / f"{doc_id}.md").write_text(text, encoding="utf-8")

# gold_docs is labelled at document level so recall is chunker-independent.
QRELS = [
    # --- lexical: exact IDs / codes / product terms (favour BM25) ---
    {"qid": "L1", "kind": "lexical",
     "question": "What does error ERR_4290 mean?",
     "gold_docs": ["vpn-setup"], "gold_section": "Troubleshooting"},
    {"qid": "L2", "kind": "lexical",
     "question": "What is policy SEC-011 about?",
     "gold_docs": ["security-access"], "gold_section": "Scope"},
    {"qid": "L3", "kind": "lexical",
     "question": "What retention periods does DR-07 define?",
     "gold_docs": ["data-retention"], "gold_section": "Retention Periods"},
    {"qid": "L4", "kind": "lexical",
     "question": "Which HTTP status code is returned when the rate limit is exceeded?",
     "gold_docs": ["api-rate-limits"], "gold_section": "Exceeding the Limit"},
    {"qid": "L5", "kind": "lexical",
     "question": "Which portal do I use to request time off?",
     "gold_docs": ["pto-policy"], "gold_section": "Requesting Leave"},
    {"qid": "L6", "kind": "lexical",
     "question": "Which application do I use to submit expense claims?",
     "gold_docs": ["expense-policy"], "gold_section": "Submitting a Claim"},
    {"qid": "L7", "kind": "lexical",
     "question": "How much is the home office stipend?",
     "gold_docs": ["remote-work"], "gold_section": "Equipment and Connectivity"},

    # --- semantic: paraphrases that avoid the doc's own wording (favour dense) ---
    {"qid": "S1", "kind": "semantic",
     "question": "How much can I get reimbursed for food each day when travelling abroad?",
     "gold_docs": ["expense-policy"], "gold_section": "Meal Limits"},
    {"qid": "S2", "kind": "semantic",
     "question": "Am I allowed to work from home every day of the week?",
     "gold_docs": ["remote-work"], "gold_section": "Default Arrangement"},
    {"qid": "S3", "kind": "semantic",
     "question": "How many vacation days do I get each year?",
     "gold_docs": ["pto-policy"], "gold_section": "Annual Leave"},
    {"qid": "S4", "kind": "semantic",
     "question": "Is my personal information kept safe while it is stored?",
     "gold_docs": ["data-retention"], "gold_section": "Protection at Rest"},
    {"qid": "S5", "kind": "semantic",
     "question": "What happens to my information after I close my account?",
     "gold_docs": ["data-retention"], "gold_section": "Account Deletion"},
    {"qid": "S6", "kind": "semantic",
     "question": "How long after a trip do I have to hand in my receipts?",
     "gold_docs": ["expense-policy"], "gold_section": "Submitting a Claim"},
    {"qid": "S7", "kind": "semantic",
     "question": "During which hours am I expected to be reachable?",
     "gold_docs": ["remote-work"], "gold_section": "Core Hours"},
    {"qid": "S8", "kind": "semantic",
     "question": "Do I need a doctor's note when I am off sick?",
     "gold_docs": ["pto-policy"], "gold_section": "Sick Leave"},

    # --- multi-hop: the answer needs two documents (for notebook 06) ---
    {"qid": "M1", "kind": "multihop",
     "question": "Which VPN client does the security policy require, and which operating systems does it support?",
     "gold_docs": ["security-access", "vpn-setup"], "gold_section": None},
    {"qid": "M2", "kind": "multihop",
     "question": "For a customer-facing outage, who must be paged and how quickly, and how is production access to fix it granted?",
     "gold_docs": ["incident-response", "security-access"], "gold_section": None},
    {"qid": "M3", "kind": "multihop",
     "question": "What per-minute request limit and what monthly price come with the Scale tier?",
     "gold_docs": ["api-rate-limits", "billing-faq"], "gold_section": None},
]

(EVAL / "qrels.json").write_text(json.dumps(QRELS, indent=2), encoding="utf-8")
print(f"Wrote {len(DOCS)} docs to {CORPUS} and {len(QRELS)} questions to {EVAL/'qrels.json'}")
