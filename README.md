# PolyAI Vision Agent

An AI vision assistant that lets users upload an image and edit it through natural-language chat — blur, rotate, flip, or add noise to the whole image or to a specific detected object ("blur the second dog," "rotate the image 90 degrees left") — backed by object detection, a dedicated image-processing microservice, full observability, and two parallel deployment paths: Docker Compose on EC2, and a self-managed Kubernetes cluster with GitOps delivery via ArgoCD.

## Architecture

The application tier is composed of four containerized services, plus a monitoring stack and a log-shipping sidecar:

| Service | Purpose | Internal port |
|---|---|---|
| `frontend` | Next.js chat UI | 3000 |
| `agent` | FastAPI + LangChain orchestrator; the ReAct loop that calls tools; instrumented with Prometheus metrics | 8000 |
| `yolo` | Object detection service (YOLO), stores/serves predictions | 8080 |
| `img-proc-mcp` | MCP server exposing image manipulation tools (`rotate`, `flip`, `blur`, `resize`, `crop`, `add_noise`) | 9000 (internal only, not published) |
| `prometheus` | Scrapes metrics from `yolo`, `agent`, and (Compose only) `node-exporter` | 9090 |
| `grafana` | Dashboards over Prometheus data, auto-provisioned | 3001 (host) → 3000 |
| `node-exporter` | Host-level Linux metrics (CPU/memory/disk/network) — **Compose only**, intentionally excluded from the Kubernetes deployment | 9100 |
| `fluent-bit` | Tails container logs and ships them, compressed, to S3 — **Compose only** | — |

Two Docker networks separate concerns in Compose: `polyai-net` for the application tier and `monitoring-net` for observability, with `prometheus` bridging both so it can scrape application services as well as host metrics.

### Request flow

1. User uploads an image and sends a message from the frontend to the agent's `/chat` endpoint.
2. The agent's LLM (bound to four tools — `detect_objects`, `get_detections`, `process_region`, `show_current_image`) decides what to do:
   - **Whole-image edits** (e.g. "rotate 90 degrees") call `process_region` directly with no bounding box — the transformed image becomes the new full image.
   - **Object-specific edits** (e.g. "blur the second dog") first call `get_detections`, which uploads the image to S3, asks `yolo` to detect objects, and returns a position-labeled list of bounding boxes (sorted left-to-right, labeled `leftmost`/`middle`/`rightmost`). The model matches the user's wording against this pre-computed `position` field rather than estimating position itself.
   - `process_region` then crops just the relevant region locally, sends only that (small) crop to the `img-proc-mcp` server for the actual transformation, and pastes the result back into the full-resolution image.
3. The edited image is returned directly in the `/chat` response — never routed back through the LLM, to avoid blowing the token budget on image data.

### Design decisions worth knowing

- **Full images go to Yolo via S3** (upload once, pass just the S3 key) since they can be large and Yolo may run on a separate host; **cropped regions go directly as base64** to the MCP server since they're small and don't need a storage round-trip.
- **The LLM never sees the raw MCP tools** (`rotate`/`flip`/`blur`/etc.) directly — only the high-level `process_region` abstraction, which decides internally which MCP tool to invoke. This keeps the model's tool surface small and prevents it from mismanaging large base64 payloads.
- **Per-request state uses `ContextVar`s**, not plain module-level dicts, so concurrent requests on the same process can never leak or clobber each other's in-flight image, prediction, or token-usage data.

## Deployment models

This repo supports two independent, parallel deployment paths for `dev` and `prod` environments. Both are kept running simultaneously; Kubernetes is the direction the project is migrating toward.

### 1. Docker Compose (EC2)

Two long-running EC2 instances (`yolo-dev-*`, `yolo-prod-*`), each running the full stack via `docker-compose.yaml`. CI/CD builds and pushes each service's image on a path-filtered push, then SSHes in and runs `docker compose pull && up -d`.

### 2. Kubernetes (self-managed `kubeadm` cluster on EC2)

A 4-node cluster (2 control-plane, 2 worker, spanning two AZs) with `dev` and `prod` namespaces, deployed declaratively via **ArgoCD**:

- `yolo-dev` Application — tracks the `dev` branch, **auto-sync**, syncs the entire `infra/k8s/dev/` directory recursively.
- `yolo-prod` Application — tracks `main` (`HEAD`), **manual sync only** (a deliberate promotion gate — nothing reaches prod without an explicit sync trigger).
- `app-of-apps` — bootstraps both Applications above from `infra/k8s/argo/`.

Every service runs as a plain `Deployment` + `Service` (no Helm, no operators):

- **Probes & resources**: liveness/readiness HTTP (or `tcpSocket`, for `img-proc-mcp`, which has no HTTP health route) probes, and CPU/memory requests+limits on every container, sized per environment (`dev` lighter, `prod` heavier).
- **Autoscaling**: `HorizontalPodAutoscaler` (50% CPU target, min 1 / max 3 replicas) on `yolo`, `agent`, and `frontend`. Each Application's `ignoreDifferences` excludes `spec.replicas` on Deployments so ArgoCD doesn't fight the HPA for that field.
- **Persistence**: `prometheus` uses a statically-provisioned, EBS-backed `PersistentVolume` per environment (pinned to `us-east-1a` via `nodeAffinity`, since EBS volumes are single-AZ) — metrics history survives pod restarts. `grafana` and every other service are stateless.
- **Node-exporter is intentionally excluded** from Kubernetes (host-level metrics aren't meaningful in a scheduled, multi-tenant cluster context the way they are on a dedicated Compose host).

```bash
# One-time, cluster-scoped:
kubectl apply -f infra/k8s/base/storageclass.yaml   # if not already installed

# Bootstrap everything via ArgoCD:
kubectl apply -f infra/k8s/argo/app-of-apps.yaml

# Force an immediate sync (otherwise auto-sync/manual-sync policies apply):
kubectl patch application yolo-dev  -n argocd --type merge -p '{"operation":{"sync":{"revision":"HEAD","prune":true}}}'
kubectl patch application yolo-prod -n argocd --type merge -p '{"operation":{"sync":{"revision":"HEAD","prune":true}}}'
```

**Known asymmetry**: `yolo`'s CI workflow auto-commits its new image tag into `infra/k8s/<env>/yolo/deployment.yaml` on every build. `agent`, `frontend`, and `img-proc-mcp` don't have that automation yet — their Compose deployments update automatically via CI, but their **k8s** image tags need a manual commit after a rebuild until that pipeline is extended.

## Getting started (Docker Compose)

### Prerequisites

- Docker and Docker Compose
- An AWS account with an IAM role granting `s3:PutObject`/`s3:GetObject` on your image bucket, `s3:PutObject` on your logs bucket, and `bedrock:InvokeModel` for whichever Bedrock model you use (if using a Bedrock model)
- A DockerHub account (for the CI/CD pipeline, if deploying via GitHub Actions)

### Environment variables

Set these in a `.env` file at the repository root:

| Variable | Example | Notes |
|---|---|---|
| `AWS_REGION` | `us-east-1` | |
| `AWS_S3_BUCKET` | `my-polyai-images` | Used for full-image uploads before Yolo detection |
| `MODEL` | `bedrock_converse:amazon.nova-lite-v1:0` | Must be one of the models in `ALLOWED_MODELS` in `services/agent/app.py` |
| `YOLO_TAG` / `IMG_PROC_MCP_TAG` / `AGENT_TAG` / `FRONTEND_TAG` | git commit SHA | Per-service image tags — never use `latest` |
| `AGENT_PUBLIC_URL` | `http://your-host:8000` | Baked into the frontend at build time (`NEXT_PUBLIC_AGENT_URL`) — see [Known limitations](#known-limitations) |
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins for the agent's CORS policy |

### Running the stack

```bash
docker compose up -d
```

This brings up all eight services. On first run, Grafana comes pre-configured with a Prometheus data source and the community "Node Exporter Full" dashboard (ID 1860) already provisioned — no manual UI setup required.

To rebuild a single service after a code change (Compose only recreates containers whose image actually changed):

```bash
docker compose build <service>
docker compose up -d <service>
```

**Prometheus and Grafana don't hot-reload their config on a file change** — if you edit `monitoring/prometheus.yml` or a dashboard, restart the container explicitly:

```bash
docker compose restart prometheus
```

### Verifying it's up

```bash
docker compose ps
curl http://localhost:8000/health
```

Grafana: `http://localhost:3001` — dashboards and data source should already be present.
Prometheus: `http://localhost:9090` — check **Status → Targets** to confirm `yolo`, `agent`, and `node-exporter` are being scraped.

## Observability

### Metrics

- **`yolo`** and **`agent`** each expose a `/metrics` endpoint scraped by Prometheus every 15s.
- **Agent metrics** (added via `prometheus_client`):
  - `agent_chat_requests_total{status="success"|"error"}` — request count by outcome
  - `agent_chat_request_latency_seconds` — histogram, used for p50/p95/p99 via `histogram_quantile()`
  - `agent_chat_input_tokens_total` / `agent_chat_output_tokens_total` — cumulative LLM token usage, summed across every model call in a request's tool-calling loop (there can be several per `/chat` call)
- **Dashboards**: `monitoring/grafana/dashboards/` (Node Exporter Full, Compose only) and `infra/grafana/dashboards/agent.json` (request rate by status, latency percentiles, error rate, token throughput — portable across environments via a templated `${datasource}` variable, so the same JSON imports cleanly into any environment's Grafana).

### Logs (Fluent Bit → S3)

`fluent-bit.conf` tails every container's Docker JSON log file, adds a `host` field, compresses, and uploads to S3 (`sleeman-polyai-logs`, 90-day lifecycle expiry). Two things worth knowing if you touch this config:

- **`Parsers_File` must be set explicitly** in `[SERVICE]` — the `docker` parser referenced by `[INPUT]` isn't loaded by default, and without it Fluent Bit fails to start.
- **The source container ID is encoded directly in the S3 key** via `s3_key_format`'s `$TAG[5]` (Fluent Bit's tail input converts the source file path into a dot-separated tag; the container-ID path segment lands at index 5) — e.g. `logs/2026/07/20/<container_id>_083026.gz`. This is what makes per-container log queries possible without needing Docker's json-file driver to embed a friendly name (it doesn't).

### Observability MCP server (`services/observability-mcp/`)

A local, stdio-based MCP server (not part of the running stack — spawned on-demand by an AI coding assistant like Claude Code or GitHub Copilot Chat) that queries the logs and metrics above directly from chat:

| Tool | What it does |
|---|---|
| `list_log_sources` | Lists distinct container IDs that have shipped logs to S3 recently |
| `search_logs` | Time-windowed (relative or absolute ISO8601 range) log search, filterable by keyword and/or container ID |
| `get_service_logs` | Resolves a friendly compose service name (`yolo`, `agent`, ...) to its current container ID over SSH (`docker compose ps -q`), then searches its logs — since Docker's json-file driver never records the service name itself |
| `query_prometheus` | Raw PromQL instant query against either environment's Prometheus |
| `get_cpu_usage` | Convenience wrapper around `query_prometheus` for node-level CPU% via `node_exporter` |

**Setup**: `pip install -r services/observability-mcp/requirements.txt`, then register the server with your MCP client. For Claude Code, create a project-root `.mcp.json` (gitignored — it holds machine-specific IPs and SSH key paths):

```json
{
  "mcpServers": {
    "observability": {
      "command": "python3",
      "args": ["services/observability-mcp/app.py"],
      "env": {
        "DEV_PROMETHEUS_URL": "http://<dev-ec2-ip>:9090",
        "PROD_PROMETHEUS_URL": "http://<prod-ec2-ip>:9090",
        "DEV_S3_LOGS_BUCKET": "sleeman-polyai-logs",
        "PROD_S3_LOGS_BUCKET": "sleeman-polyai-logs",
        "AWS_REGION": "us-east-1",
        "DEV_SSH_HOST": "<dev-ec2-ip>",
        "PROD_SSH_HOST": "<prod-ec2-ip>",
        "DEV_SSH_KEY_PATH": "/path/to/yolo-dev-key.pem",
        "PROD_SSH_KEY_PATH": "/path/to/yolo-prod-key.pem",
        "SSH_USER": "ubuntu"
      }
    }
  }
}
```

(For VS Code + GitHub Copilot Chat, the equivalent file is `.vscode/mcp.json`, with a top-level `"servers"` key instead of `"mcpServers"` and each entry additionally specifying `"type": "stdio"`.)

Both EC2 instances' security groups need inbound access on port `9090` (Prometheus) for `query_prometheus`/`get_cpu_usage` to reach them from outside the VPC.

## Project structure

```
.
|-- docker-compose.yaml
|-- fluent-bit.conf
|-- .github/workflows/                 # CI/CD: path-filtered build/test/deploy per service
|-- monitoring/
|   |-- prometheus.yml
|   `-- grafana/
|       |-- provisioning/                # datasources + dashboards auto-registration
|       `-- dashboards/                  # Node Exporter Full dashboard JSON
|-- infra/
|   |-- k8s/
|   |   |-- base/                        # cluster-scoped resources (StorageClass)
|   |   |-- argo/                        # ArgoCD Application manifests + app-of-apps
|   |   |-- dev/                         # yolo, agent, frontend, img-proc-mcp, prometheus, grafana
|   |   `-- prod/                        # same services, prod-sized resources
|   `-- grafana/dashboards/              # agent.json - portable Grafana dashboard
`-- services/
    |-- agent/                           # FastAPI + LangChain ReAct agent, Prometheus-instrumented
    |-- img-proc-mcp/                    # MCP server: rotate/flip/blur/resize/crop/add_noise
    |-- observability-mcp/               # local MCP server: query S3 logs + Prometheus from chat
    |-- yolo/                            # Object detection service
    `-- frontend/                        # Next.js chat UI
```

## API reference (agent service)

- `POST /chat` — body: `{ "messages": [{ "role": "user"|"assistant", "content": "...", "image_base64": "..." }] }`. Returns `{ "response": "...", "annotated_image_base64": "..." }`.
- `GET /health` — liveness check.
- `GET /metrics` — Prometheus exposition format.

## Testing

The `img-proc-mcp` server has unit tests covering all six tools (dimension checks, error handling, and a pixel-diff assertion for noise) — run with:

```bash
cd services/img-proc-mcp
pytest tests/
```

## CI/CD

A single GitHub Actions workflow uses path-filtering (`dorny/paths-filter`) to detect which service(s) changed on a push, then conditionally tests, scans (Docker Scout, for `yolo`), and builds/pushes only the affected image(s) tagged with the git commit SHA. Compose deployment SSHes into the target instance, updates only the changed service's tag, and runs `docker compose pull && up -d`. `yolo`'s workflow additionally auto-commits the new tag into its Kubernetes manifests (see [Known asymmetry](#2-kubernetes-self-managed-kubeadm-cluster-on-ec2) above); the other services' k8s tags are updated manually for now.

## Known limitations

- `resize` and `crop` are implemented on the MCP server but not yet wired into natural-language requests — only `blur`, `rotate`, `flip`, and `add_noise` are reachable via chat.
- Small/cheap LLMs can occasionally skip calling a tool after several repetitive requests in one conversation despite explicit instructions; a bounded corrective retry mitigates most cases but isn't a hard guarantee.
- Object position resolution (`leftmost`/`middle`/`rightmost`) relies on Yolo's returned bounding boxes; this was previously broken by Yolo returning `box` as a JSON-encoded string rather than a native array (now fixed, see `_parse_box` in `app.py`).
- The Grafana provisioning files are checked into the repo, but any dashboard/data-source edits made through the Grafana UI directly won't persist unless the underlying JSON/YAML files are updated to match.
- **`NEXT_PUBLIC_AGENT_URL` isn't passed as a build-arg in `deploy-frontend.yml`**, so CI-built frontend images fall back to `http://localhost:8000` for browser-side API calls. Fine for local `kubectl port-forward` testing; needs fixing before the Kubernetes frontend is reachable by real end users.
- **Container-to-service-name mapping isn't available from S3 logs alone** — Docker's json-file log driver never records a friendly name, only the container ID. `get_service_logs` in the observability MCP server works around this with a live SSH lookup, but a raw `search_logs`/`list_log_sources` result will only ever show container IDs.
- Fluent Bit batches multiple containers' log lines within a single flush window into one uploaded object; an S3 object's *name* reflects one source container, but its *content* can occasionally include a few lines from another container active in the same window.
