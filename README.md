# PolyAI Vision Agent

An AI vision assistant that lets users upload an image and edit it through natural-language chat — blur, rotate, flip, or add noise to the whole image or to a specific detected object ("blur the second dog," "rotate the image 90 degrees left") — backed by object detection, a dedicated image-processing microservice, full observability, and a one-command containerized deployment.

## Architecture

The system is composed of seven containerized services:

| Service | Purpose | Internal port |
|---|---|---|
| `frontend` | Next.js chat UI | 3000 |
| `agent` | FastAPI + LangChain orchestrator; the ReAct loop that calls tools | 8000 |
| `yolo` | Object detection service (YOLO), stores/serves predictions | 8080 |
| `img-proc-mcp` | MCP server exposing image manipulation tools (`rotate`, `flip`, `blur`, `resize`, `crop`, `add_noise`) | 9000 (internal only, not published) |
| `prometheus` | Scrapes metrics from `yolo` and `node-exporter` | 9090 |
| `grafana` | Dashboards over Prometheus data, auto-provisioned | 3001 (host) → 3000 |
| `node-exporter` | Exposes host-level Linux metrics (CPU/memory/disk/network) | 9100 |

Two Docker networks separate concerns: `polyai-net` for the application tier and `monitoring-net` for observability, with `prometheus` bridging both so it can scrape application services as well as host metrics.

### Request flow

1. User uploads an image and sends a message from the frontend to the agent's `/chat` endpoint.
2. The agent's LLM (bound to three local tools — `detect_objects`, `get_detections`, `process_region`, plus `show_current_image`) decides what to do:
   - **Whole-image edits** (e.g. "rotate 90 degrees") call `process_region` directly with no bounding box — the transformed image becomes the new full image.
   - **Object-specific edits** (e.g. "blur the second dog") first call `get_detections`, which uploads the image to S3, asks `yolo` to detect objects, and returns a position-labeled list of bounding boxes (sorted left-to-right, labeled `leftmost`/`middle`/`rightmost`). The model matches the user's wording against this pre-computed `position` field rather than estimating position itself.
   - `process_region` then crops just the relevant region locally, sends only that (small) crop to the `img-proc-mcp` server for the actual transformation, and pastes the result back into the full-resolution image.
3. The edited image is returned directly in the `/chat` response — never routed back through the LLM, to avoid blowing the token budget on image data.

### Design decisions worth knowing

- **Full images go to Yolo via S3** (upload once, pass just the S3 key) since they can be large and Yolo may run on a separate host; **cropped regions go directly as base64** to the MCP server since they're small and don't need a storage round-trip.
- **The LLM never sees the raw MCP tools** (`rotate`/`flip`/`blur`/etc.) directly — only the high-level `process_region` abstraction, which decides internally which MCP tool to invoke. This keeps the model's tool surface small and prevents it from mismanaging large base64 payloads.
- **Per-request state uses `ContextVar`s**, not plain module-level dicts, so concurrent requests on the same process can never leak or clobber each other's in-flight image/result data.

## Getting started

### Prerequisites

- Docker and Docker Compose
- An AWS account with an IAM role granting `s3:PutObject`/`s3:GetObject` on your bucket, and `bedrock:InvokeModel` for whichever Bedrock model you use (if using a Bedrock model)
- A DockerHub account (for the CI/CD pipeline, if deploying via GitHub Actions)

### Environment variables

Set these in a `.env` file at the repository root:

| Variable | Example | Notes |
|---|---|---|
| `AWS_REGION` | `us-east-1` | |
| `AWS_S3_BUCKET` | `my-polyai-images` | Used for full-image uploads before Yolo detection |
| `MODEL` | `bedrock_converse:amazon.nova-lite-v1:0` | Must be one of the models in `ALLOWED_MODELS` in `services/agent/app.py` |
| `YOLO_TAG` / `IMG_PROC_MCP_TAG` / `AGENT_TAG` / `FRONTEND_TAG` | git commit SHA | Per-service image tags — never use `latest` |
| `AGENT_PUBLIC_URL` | `http://your-host:8000` | Baked into the frontend at build time (`NEXT_PUBLIC_AGENT_URL`) |
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins for the agent's CORS policy |

### Running the stack

```bash
docker compose up -d
```

This brings up all seven services. On first run, Grafana comes pre-configured with a Prometheus data source and the community "Node Exporter Full" dashboard (ID 1860) already provisioned — no manual UI setup required.

To rebuild a single service after a code change (Compose only recreates containers whose image actually changed):

```bash
docker compose build <service>
docker compose up -d <service>
```

### Verifying it's up

```bash
docker compose ps
curl http://localhost:8000/health
```

Grafana: `http://localhost:3001` — dashboards and data source should already be present.
Prometheus: `http://localhost:9090` — check **Status → Targets** to confirm `yolo` and `node-exporter` are being scraped.

## Project structure

```
.
├── docker-compose.yaml
├── .github/workflows/           # CI/CD: path-filtered build/test/deploy per service
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/     # auto-registers Prometheus
│       │   └── dashboards/      # points Grafana at the dashboards folder
│       └── dashboards/          # Node Exporter Full dashboard JSON
└── services/
    ├── agent/                   # FastAPI + LangChain ReAct agent
    ├── img-proc-mcp/            # MCP server: rotate/flip/blur/resize/crop/add_noise
    ├── yolo/                    # Object detection service
    └── frontend/                # Next.js chat UI
```

## API reference (agent service)

- `POST /chat` — body: `{ "messages": [{ "role": "user"|"assistant", "content": "...", "image_base64": "..." }] }`. Returns `{ "response": "...", "annotated_image_base64": "..." }`.
- `GET /health` — liveness check.

## Testing

The `img-proc-mcp` server has unit tests covering all six tools (dimension checks, error handling, and a pixel-diff assertion for noise) — run with:

```bash
cd services/img-proc-mcp
pytest tests/
```

## CI/CD

A single GitHub Actions workflow uses path-filtering (`dorny/paths-filter`) to detect which service(s) changed on a push, then conditionally tests, scans (Docker Scout, for `yolo`), and builds/pushes only the affected image(s) tagged with the git commit SHA. Deployment SSHes into the target instance, updates only the changed service's tag in `.env`, and runs `docker compose pull && up -d` — Compose only recreates containers whose image tag actually changed, making each service independently deployable from a single `docker-compose.yaml`.

## Known limitations

- `resize` and `crop` are implemented on the MCP server but not yet wired into natural-language requests — only `blur`, `rotate`, `flip`, and `add_noise` are reachable via chat.
- Small/cheap LLMs can occasionally skip calling a tool after several repetitive requests in one conversation despite explicit instructions; a bounded corrective retry mitigates most cases but isn't a hard guarantee.
- Object position resolution (`leftmost`/`middle`/`rightmost`) relies on Yolo's returned bounding boxes; this was previously broken by Yolo returning `box` as a JSON-encoded string rather than a native array (now fixed, see `_parse_box` in `app.py`).
- The Grafana provisioning files are checked into the repo, but any dashboard/data-source edits made through the Grafana UI directly won't persist unless the underlying JSON/YAML files are updated to match.
