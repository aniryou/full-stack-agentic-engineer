# API Rate Limits

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
