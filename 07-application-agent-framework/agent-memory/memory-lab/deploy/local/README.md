# deploy/local — the memory service, a fake model server and Postgres + pgvector on your laptop

**Tier:** T0 + Docker. **Cost:** free. **Time:** about 5 minutes the first time (image build and a Postgres pull).

This folder starts the services in the table. All of them are bound to 127.0.0.1:

| Service | Port | What it is |
|---|---|---|
| `memory` | 8080 | `python -m memlab serve`: the memory service (scope from a bearer token, Idempotency-Key, forget, audit JSON lines) on a SQLite file in the `memlab-data` volume |
| `fake-llm` | 8000 | `python -m memlab fake --prompt-tokens-details`: chat with tool calls, `/v1/embeddings` and vLLM-named `/metrics`. It is **simulated**. |
| `postgres` (profile `pgvector`) | 5432 | `pgvector/pgvector:0.8.6-pg17` for the pgvector section of notebook 01 |

```bash
./up.sh                     # memory + fake-llm;  DRY_RUN=1 ./up.sh prints the commands only
./up.sh --pgvector          # ... plus Postgres + pgvector
export MEMLAB_TOKEN_KEY=local-dev-only-change-me
TOKEN=$(python -m memlab token --tenant acme --user u1)
curl -s -H "Authorization: Bearer $TOKEN" -H "Idempotency-Key: t1" -H "Content-Type: application/json" \
     -d '{"text": "Home city: the user lives in Lisbon.", "slot": "home_city", "value": "Lisbon"}' \
     http://127.0.0.1:8080/v1/memories
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"query": "which city is my home city?"}' http://127.0.0.1:8080/v1/memories/search
export MEMLAB_LLM_URL=http://127.0.0.1:8000                     # notebooks 02/03 use it (still simulated)
export MEMLAB_PG_DSN=postgresql://memlab:memlab-local-only@127.0.0.1:5432/memlab   # notebook 01, with psycopg
```

If there is no Docker daemon, `up.sh` prints the commands and exits 0. Then the notebooks run their T0 paths.
Where a container gives the answer on a machine with Docker, the notebooks show sample output in the documented
format (illustrative).

## What is simulated and what is real

The service, the SQLite store and Postgres are real. The model server is the fake of the lab. Its prefix-cache
hits obey the block rules of vLLM, and its TTFTs come from a roofline model. When you quote these numbers, say
"simulated". For measurements, run a real vLLM ([`../any-gpu/`](../any-gpu/)). Then point `MEMLAB_LLM_URL` at it.

The passwords and the token key in `compose.yaml` are local-only defaults (the ports are bound to 127.0.0.1).
For anything that you share with other people, set your own `MEMLAB_TOKEN_KEY` and `MEMLAB_PG_PASSWORD` in your
shell.

## Cost

It costs nothing more than the CPU of your laptop and about 1 GB of disk for the images.

## Cleanup

```bash
./down.sh        # docker compose down --volumes: stops everything and deletes the memory DB, audit log and Postgres data
docker image rm memlab:0.1.0 pgvector/pgvector:0.8.6-pg17   # optional: the images
```

When you delete the volumes, you do the local "forget everything". On a server, the same request is the
checklist of notebook 05, because backups, logs and caches are not in the volume.
