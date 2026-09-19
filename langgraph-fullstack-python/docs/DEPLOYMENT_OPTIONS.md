# Deployment options (input for Phase 10B)

Two realistic platforms for a single-instance public demo. Both are described
as *options to evaluate*, not as recommendations to buy: Phase 10A did not
create, configure or pay for anything.

## Option A - LangSmith Deployment (managed control plane, your cloud account)

The graph is already `langgraph.json`-compatible, the custom app is declared
under `http.app`, and `scripts/prepare_docker.py` produces the image build.

| Concern | Assessment |
|---|---|
| Configuration work | low: `langgraph.json` + `LANGSGRAPH_HTTP`/auth entries already exist; deployment is a push, not a server build |
| Postgres / Redis | provided and managed by the platform; this repository's compose file is then only for local parity |
| BGE-M3 (~2.2 GB) | the hard constraint: the managed image path must either allow a large custom image or the model weights must be mounted from object storage. `langchain/langgraph-api:3.11` + torch is a ~2-3 GB image. |
| Vector store | must live on a persistent mount; a container-local Chroma directory is lost on redeploy |
| Secrets | platform secret store; `DEEPSEEK_API_KEY` and `LANGGRAPH_DEMO_API_TOKEN` never enter the image |
| HTTPS / DNS | platform-provided |
| Cost shape | per-instance/month plus egress; the model spend (6-8 DeepSeek calls per answer) dominates at any real traffic |
| Ops | no OS patching; logs and metrics in one console |
| Risk | least control over memory limits and root filesystem, which is exactly where a 2 GB embedding model is awkward |

## Option B - one VPS with Docker Compose (this repository's artifacts)

Take the generated `docker/Dockerfile` and `docker/docker-compose.yml` to a
4 GB / 2 vCPU instance.

| Concern | Assessment |
|---|---|
| Configuration work | medium: install Docker, place `.env`, build the image, run the compose file, terminate TLS |
| Postgres / Redis | both in the compose file (`pgvector/pgvector:pg16`, `redis:6`) with healthchecks; volumes `langgraph-data` and `psychegraph-hf` |
| BGE-M3 | fits: it is a local directory/volume, and `scripts/prewarm_rag.py` pays the load cost once |
| Vector store | bind-mounted from the host, so re-indexing is a host operation |
| Secrets | `.env` on the host with restrictive permissions; the compose file passes it with `env_file` |
| HTTPS / DNS | your job: Caddy/nginx/Traefik in front; SSE needs buffering disabled |
| Cost shape | fixed monthly VPS price, no egress surprises; the model spend is unchanged |
| Ops | you own patching, backups, restarts and monitoring; `restart: unless-stopped` handles crashes, not upgrades |
| Risk | the manual steps are exactly the ones Phase 10A could not rehearse (no Docker on the development host) |

## Verification status of each promise

| Claim | Verified here? |
|---|---|
| `langgraph.json` loads the graph, the auth handler and the custom app | yes - `langgraph dev` runs end to end |
| Lockfile-pinned image dependencies resolve | partially - the export is generated and inspected, the pip resolve inside the image is not run |
| The Dockerfile builds | **no** - Docker is not installed on the development host |
| `docker compose up` brings Postgres/Redis/API to healthy | **no** - same reason |
| `/health`, limits, ownership behave in a container | **partially** - identical code paths are verified by tests and against the dev server, not inside a container |
| Peak RSS ~2 GB, ~12 s model load | yes, on the host, `scripts/bench_rss.py` |

## What Phase 10B has to decide first

1. **Where** the image runs (Option A or B) and **who** terminates TLS.
2. Whether the instance is truly public or token-gated: `LANGGRAPH_DEMO_API_TOKEN`
   protects the API, but a public web UI still needs rate limits tuned for real
   traffic and probably an IP-based bucket in front.
3. Whether BGE-M3 stays on CPU. Keeping it CPU-only is what makes a cheap VPS
   workable; a GPU instance changes both cost and image size.
4. How the vector store is produced and shipped (built on the host and mounted,
   or built inside a job).
5. A rollback and restore story for `langgraph-data` before the first real user
   types anything.
