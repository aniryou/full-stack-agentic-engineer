# Incident Response Runbook

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
