# deploy/local — the gateway and two fake providers in Docker Compose (T0 + Docker)

**Tier:** T0 + Docker: any laptop with Docker, no GPU, no account. **Cost:** free. **Cleanup:** `./down.sh` removes
the containers, the `gwdata` volume with keys, cache, ledger and spans, and the `gwlab:local` image.

Three containers come from one image, built from this lab:

| Service | Port | What it is |
|---|---|---|
| `gateway` | 8080 | `python -m gwlab gateway --config lab`. The provider keys come from its environment. The ledger and spans go to the `gwdata` volume. |
| `acme` | 8101 (internal) | an OpenAI-dialect fake provider that answers 30 % of requests with 503 before the first byte (`--fail-rate 0.3`) |
| `bolt` | 8102 (internal) | an Anthropic-dialect fake provider: the fallback in the `chat` alias |

Compose publishes only the port of the gateway. Thus, clients never reach a provider, and they never hold a provider key.

```bash
./up.sh           # DRY_RUN=1 ./up.sh prints the steps; builds gwlab:local, starts the stack, waits for /health
./smoke.sh        # a virtual key, ten streamed requests (see x-gwlab-target flip to bolt/haiku), the counters
python3 -m gwlab key --url http://localhost:8080 --tenant team-b        # another tenant's key
curl -s localhost:8080/debug/state -H 'Authorization: Bearer dev-admin-token' | python3 -m json.tool | head -40
docker compose -f docker-compose.yaml exec gateway python -m gwlab ledger --db /data/gateway.db
./down.sh
```

Notebook [`01_a_gateway_over_http`](../../notebooks/01_a_gateway_over_http.ipynb) talks to this stack when you set
`GWLAB_URL=http://localhost:8080`. If not, it starts the same processes in-process (`gwlab.stack.LocalStack`).
Without Docker, it prints these commands and a bundled sample output in the documented format (illustrative):
`gwlab/data/samples/compose_smoke.txt`.

Everything that the fakes report is **simulated** (their responses carry `x-gwlab-simulated: true`). Before this
stack runs on a machine other than your laptop, change `GWLAB_ADMIN_TOKEN` and `GWLAB_SALT_SECRET`. The first
protects `/admin/*`. The gateway calculates the `cache_salt` of each tenant from the second.
