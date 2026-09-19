# Deployment benchmark (Phase 10A)

Sizing numbers for the deployment decision. They were measured on the
development host - a Windows workstation, CPU only, no container - with
`scripts/bench_rss.py`, which is the reproducible part of this document:

```bash
uv run python scripts/bench_rss.py --skip-run   # no API key needed
uv run python scripts/bench_rss.py              # adds a full graph run
```

Raw output is written to `data/reports/phase10a_resources.json`.

## 1. Method

| Item | Value |
|---|---|
| Device | development workstation, CPU only (no CUDA), Windows |
| Python | 3.13.5 (`.venv`, `uv sync --frozen`) |
| Embedding model | `BAAI/bge-m3`, `EMBEDDING_DEVICE=cpu`, weights in the local Hugging Face cache |
| Vector store | local Chroma index at `RAG_VECTORSTORE_PATH` |
| RSS source | `K32GetProcessMemoryInfo` (Windows) / `/proc/self/status` (Linux) |
| LLM | DeepSeek `deepseek-flash`, 7 calls for one full answer |

The measurement is a **single process, cold start, one request**. RSS is the
whole process, including torch's arenas, the tokenizer, Chroma's sqlite handles
and the graph.

## 2. Memory ladder

| Stage | Elapsed | RSS | Delta |
|---|---|---|---|
| interpreter + settings import | ~0.0 s | 20.8 MB | - |
| `react_agent.graph` import (LangGraph, LangChain, chromadb, FastHTML) | ~0.0 s | 142.4 MB | +122 MB |
| BGE-M3 lazy load (weights into RAM) | 11.6 s | 805.2 MB | +663 MB |
| first retrieval (cold path, includes warm-up inference) | 0.42 s | 2003.4 MB | +1198 MB |
| second retrieval (same process) | 0.11 s | 2003.4 MB | +0 MB |
| full answer, 7 DeepSeek calls, 9 graph nodes | 39.5 s | **2046.5 MB** | +43 MB |

**Peak RSS ≈ 2.0 GB per worker.** The jump on the first retrieval is the
embedding model actually running: the weights are 2.2 GB on disk, and the first
forward pass materializes attention buffers and frameworks allocations that
stay resident afterwards.

Practical consequences:

* 2 GB of RAM is **not** enough (the process alone needs ~2 GB and the OOM
  killer does not warn politely). Plan 4 GB for one instance with headroom.
* The delta between load and first inference means "it answered one query fast"
  is not the same as "it fits": the peak appears only after real traffic.
* A container image carrying torch + transformers + chromadb is ~2-3 GB, so
  builds pull several GB and should use a local registry or build cache.

## 3. Latency

| Path | Measured | Note |
|---|---|---|
| Model load, cold process, weights already on disk | 11.6 s | up to ~25-27 s on a first, disk-cold load; the report in `data/reports/` is the source |
| Retrieval, warm model | 0.11 s | query embedding + Chroma search over the demo index |
| Retrieval inside a run | 0.60 s | logged by the evidence node as `evidence retrieval: available=True elapsed=0.602s` |
| Full answer via the SDK (7 DeepSeek calls) | 39.5 s | benchmark script |
| Full answer through the web UI (SSE, 9 nodes) | 73.1 s | Phase 10A live run, `langgraph dev`, 320 s budget |

The gap between 39.5 s and 73.1 s is the supervisor + specialist + critic
pipeline with the UI's task events enabled and the dev server's reload
overhead; model latency dominates both numbers, and neither is a promise.

Timeout guidance that follows from this: a request budget below ~120 s will cut
real answers off. The SSE stream keeps the connection alive, but a reverse proxy
must not apply a 30 s read timeout to `text/event-stream`.

## 4. Cache and index maintenance

| Item | Where | Cost of a miss | How to prepare |
|---|---|---|---|
| BGE-M3 weights (~2.2 GB) | `${HF_HOME}/hub` → volume `psychegraph-hf` | 2-5 min download, first request fails or stalls | `uv run python scripts/prewarm_rag.py`, or pre-bake the volume |
| Vector store | `RAG_VECTORSTORE_PATH` → `../data/vectorstore` | retrieval returns nothing, answers say they have no literature | index on the host before starting traffic |
| Postgres data | volume `langgraph-data` | conversations disappear | back up the volume |

`/health` answers exactly this question without leaking paths:

```json
{"status": "degraded",
 "graph_loaded": true,
 "vectorstore_available": false,
 "embedding_loaded": false}
```

`embedding_loaded: false` is normal on a fresh process (the model is lazy on
purpose: a container must start fast and must not download weights unasked).
`vectorstore_available: false` is the one to alert on, because it silently
downgrades every answer to "no local evidence".

## 5. What was not measured

* Any number **inside a Linux container**: no Docker/WSL2 on this host.
* RSS of Postgres + Redis (the compose file runs both next to the API).
* Concurrency: the demo allows one run per session, so these are single-request
  numbers, not throughput.
* Cold start of the image (build + first boot), which is where the unverified
  parts of Phase 10A live.

The honest summary for a sizing decision: **4 GB RAM / 2 vCPU is the floor for
one CPU-only instance, and 8 GB is comfortable** for the API plus Postgres and
Redis with headroom for the first-inference peak.

## 6. Container measurements (Docker Desktop on the development host)

Taken later, on the same machine but inside the WSL2 VM (32 vCPU, 16 GB VM RAM,
Linux containers). These are Linux-container numbers, which is what a VPS runs -
but they are **not** a VPS measurement, and the licensed production runtime may
differ from the development runtime used here.

| Measurement | Result |
|---|---|
| `docker build` - official image (`docker/Dockerfile`) | ✅ 5.4 min, **2.92 GB** |
| `docker build` - dev-runtime image (`docker/Dockerfile.dev`) | ✅ 3.5 min, **2.36 GB**, no nvidia/triton packages |
| App container memory, after a full answer (model loaded) | current **885 MiB**, cgroup peak **1.70 GiB** |
| Postgres container | 33.6 MiB |
| Redis container | 7.3 MiB |
| Cold prewarm in the container (download 2.2 GB + load) | **850 s**; HF volume grows to **4.56 GB** (safetensors + bin both fetched) |
| Warm prewarm after `docker compose restart` (no download) | **24.8 s** |
| `docker compose down` + `up` (volumes kept) | healthy in ~10 s, HF volume unchanged |
| First real answer inside the container (13-check smoke) | **56.6 s**, events `workflow, sources, answer, close` |
| Degraded path (index unavailable) | `/health` → `degraded`, `vectorstore_available: false`, answer still produced, no fabricated citations |
| Failure path (invalid model key) | events `workflow, error, close`, **0** answer frames, no traceback / key / provider text |

Sizing implication: the whole stack (app + Postgres + Redis) stays **under
2 GB**, so a 4 GB / 2 vCPU VPS has room, and 8 GB is comfortable. The binding
constraint is CPU (model load ~12-25 s, one answer ~40-80 s), not memory.
