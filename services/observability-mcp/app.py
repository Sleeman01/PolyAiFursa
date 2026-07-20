"""
Observability MCP server.

Exposes tools to query:
  - container logs shipped by Fluent Bit to S3
  - metrics from Prometheus (dev/prod)

Runs over stdio, spawned directly by an MCP client (e.g. VS Code Copilot
Chat via .vscode/mcp.json) - not a network service.

Environment variables (see .vscode/mcp.json):
  DEV_PROMETHEUS_URL, PROD_PROMETHEUS_URL   - e.g. http://<ec2-ip>:9090
  DEV_S3_LOGS_BUCKET, PROD_S3_LOGS_BUCKET   - S3 bucket holding Fluent Bit output
  AWS_REGION                                - e.g. us-east-1
  DEV_SSH_HOST, PROD_SSH_HOST               - EC2 public IP for each environment
  DEV_SSH_KEY_PATH, PROD_SSH_KEY_PATH       - local path to the .pem key
  SSH_USER                                  - defaults to "ubuntu"

Container identification: the S3 key itself encodes the source container ID
(fluent-bit.conf's s3_key_format uses $TAG[5], the container ID segment from
the tail input's path-derived tag) and the upload time. To resolve a
friendly compose service name (e.g. "yolo") to its current container ID,
this server SSHes into the relevant EC2 box and runs
`docker compose ps -q <service>`. If SSH env vars aren't set, service-name
lookups fail gracefully and the caller should use list_log_sources /
search_logs instead.
"""

import gzip
import io
import os
import re
from datetime import datetime, timedelta, timezone

import boto3
import paramiko
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("observability")

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
SSH_USER = os.environ.get("SSH_USER", "ubuntu")

ENV_CONFIG = {
    "dev": {
        "prometheus_url": os.environ.get("DEV_PROMETHEUS_URL", "").rstrip("/"),
        "bucket": os.environ.get("DEV_S3_LOGS_BUCKET", ""),
        "ssh_host": os.environ.get("DEV_SSH_HOST", ""),
        "ssh_key": os.environ.get("DEV_SSH_KEY_PATH", ""),
    },
    "prod": {
        "prometheus_url": os.environ.get("PROD_PROMETHEUS_URL", "").rstrip("/"),
        "bucket": os.environ.get("PROD_S3_LOGS_BUCKET", ""),
        "ssh_host": os.environ.get("PROD_SSH_HOST", ""),
        "ssh_key": os.environ.get("PROD_SSH_KEY_PATH", ""),
    },
}

_s3 = boto3.client("s3", region_name=AWS_REGION)

# Matches keys like: logs/2026/07/20/6961fc41ef60d942.../073522.gz-objectUYai0e4R
# The segment before the timestamp is the container ID, via s3_key_format's $TAG[5].
_KEY_RE = re.compile(
    r"/(\d{4})/(\d{2})/(\d{2})/([0-9a-f]{8,64})_(\d{2})(\d{2})(\d{2})\.gz"
)


def _env_cfg(environment: str) -> dict:
    environment = environment.strip().lower()
    if environment not in ENV_CONFIG:
        raise ValueError(f"environment must be 'dev' or 'prod', got '{environment}'")
    return ENV_CONFIG[environment]


def _parse_key(key: str):
    """Returns (container_id, upload_time_utc) parsed from the S3 key, or
    (None, None) if the key doesn't match the expected format."""
    m = _KEY_RE.search(key)
    if not m:
        return None, None
    y, mo, d, cid, h, mi, s = m.groups()
    t = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), tzinfo=timezone.utc)
    return cid, t


def _list_candidate_keys(bucket, start, end, buffer_minutes=5, container_id=None):
    """List S3 keys within [start - buffer, end + buffer], optionally
    filtered to a single container ID (matched by prefix, at the key level
    - no need to download objects that clearly aren't a match)."""
    start_b = start - timedelta(minutes=buffer_minutes)
    end_b = end + timedelta(minutes=buffer_minutes)
    day = start_b.date()
    last_day = end_b.date()
    keys = []
    paginator = _s3.get_paginator("list_objects_v2")
    while day <= last_day:
        prefix = f"logs/{day:%Y/%m/%d}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                cid, key_time = _parse_key(obj["Key"])
                if key_time is not None and not (start_b <= key_time <= end_b):
                    continue
                if container_id and cid and not cid.startswith(container_id):
                    continue
                keys.append(obj["Key"])
        day += timedelta(days=1)
    return keys


def _fetch_and_decompress(bucket: str, key: str) -> str:
    body = _s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with gzip.GzipFile(fileobj=io.BytesIO(body)) as f:
        return f.read().decode("utf-8", errors="replace")


def _resolve_window(minutes, start_time, end_time):
    """Returns (start, end) as tz-aware UTC datetimes."""
    now = datetime.now(timezone.utc)
    if start_time:
        start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end_time:
            end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
        else:
            end = start + timedelta(minutes=minutes or 10)
        return start, end
    minutes = minutes or 5
    return now - timedelta(minutes=minutes), now


def _resolve_container_id(environment: str, service: str) -> str:
    """SSH into the environment's EC2 box and resolve a compose service name
    (e.g. 'yolo') to its current (short) container ID."""
    cfg = _env_cfg(environment)
    if not cfg["ssh_host"] or not cfg["ssh_key"]:
        raise RuntimeError(
            f"SSH not configured for '{environment}' "
            f"(set {environment.upper()}_SSH_HOST / {environment.upper()}_SSH_KEY_PATH) - "
            "cannot resolve service name to container ID. "
            "Try list_log_sources or search_logs instead."
        )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            cfg["ssh_host"],
            username=SSH_USER,
            key_filename=cfg["ssh_key"],
            timeout=10,
        )
        stdin, stdout, stderr = client.exec_command(
            f"cd ~/PolyAiFursa && docker compose ps -q {service}"
        )
        container_id = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if not container_id:
            raise RuntimeError(
                f"No running container found for service '{service}' in '{environment}'. {err}"
            )
        return container_id
    finally:
        client.close()


@mcp.tool()
def list_log_sources(environment: str = "dev", minutes: int = 60) -> str:
    """List distinct containers that have shipped logs to S3 in the last N minutes.
    Returns container IDs, and when they were last seen.
    environment: 'dev' or 'prod'. minutes: how far back to look (default 60)."""
    cfg = _env_cfg(environment)
    if not cfg["bucket"]:
        return f"No S3 bucket configured for environment '{environment}'."

    start, end = _resolve_window(minutes, None, None)
    keys = _list_candidate_keys(cfg["bucket"], start, end)
    if not keys:
        return f"No log objects found for '{environment}' in the last {minutes} minute(s)."

    sources = {}  # container_id -> last_seen
    for key in keys:
        cid, key_time = _parse_key(key)
        if not cid:
            continue
        if key_time and (cid not in sources or key_time > sources[cid]):
            sources[cid] = key_time

    if not sources:
        return f"Found {len(keys)} log object(s) for '{environment}' but couldn't parse container IDs from key names."

    lines = [f"Containers shipping logs in '{environment}' (last {minutes}m):"]
    for cid, last_seen in sorted(sources.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  - {cid}  last_seen={last_seen.isoformat()}")
    return "\n".join(lines)


@mcp.tool()
def search_logs(
    environment: str = "dev",
    minutes: int = 5,
    keyword: str = "",
    container_id: str = "",
    start_time: str = "",
    end_time: str = "",
    max_lines: int = 200,
) -> str:
    """Fetch and filter container logs shipped to S3.
    environment: 'dev' or 'prod'.
    minutes: how far back to look from now (ignored if start_time is set).
    keyword: optional case-insensitive substring filter on raw log content.
    container_id: optional container ID (or prefix) to filter to one container.
    start_time / end_time: optional ISO8601 timestamps (e.g. '2026-07-01T12:00:00')
      for a specific historical window instead of 'last N minutes'.
    max_lines: cap on returned lines (default 200)."""
    cfg = _env_cfg(environment)
    if not cfg["bucket"]:
        return f"No S3 bucket configured for environment '{environment}'."

    start, end = _resolve_window(minutes, start_time or None, end_time or None)
    keys = _list_candidate_keys(cfg["bucket"], start, end, container_id=container_id or None)
    if not keys:
        return f"No log objects found for '{environment}' between {start} and {end}."

    keyword_l = keyword.lower() if keyword else None
    results = []
    for key in keys[:200]:
        try:
            text = _fetch_and_decompress(cfg["bucket"], key)
        except Exception:
            continue
        for line in text.splitlines():
            if keyword_l and keyword_l not in line.lower():
                continue
            results.append(line)
            if len(results) >= max_lines:
                break
        if len(results) >= max_lines:
            break

    if not results:
        return f"No matching log lines found in '{environment}' between {start} and {end}."
    return f"{len(results)} matching line(s) in '{environment}' between {start} and {end}:\n" + "\n".join(results)


@mcp.tool()
def get_service_logs(
    environment: str = "dev",
    service: str = "",
    minutes: int = 5,
    start_time: str = "",
    end_time: str = "",
    max_lines: int = 200,
) -> str:
    """Get logs for a named compose service (e.g. 'yolo', 'agent', 'frontend',
    'img-proc-mcp') by resolving its current container ID over SSH, then
    fetching matching log lines from S3.
    environment: 'dev' or 'prod'.
    minutes: how far back to look from now (ignored if start_time is set).
    start_time / end_time: optional ISO8601 timestamps for a historical window,
      e.g. start_time='2026-07-01T11:55:00', end_time='2026-07-01T12:05:00'.
    max_lines: cap on returned lines (default 200)."""
    if not service:
        return "Please specify a service name (e.g. 'yolo', 'agent', 'frontend', 'img-proc-mcp')."
    try:
        container_id = _resolve_container_id(environment, service)
    except Exception as e:
        return f"Could not resolve '{service}' to a container: {e}"
    return search_logs(
        environment=environment,
        minutes=minutes,
        container_id=container_id,
        start_time=start_time,
        end_time=end_time,
        max_lines=max_lines,
    )


@mcp.tool()
def query_prometheus(environment: str = "dev", promql: str = "") -> str:
    """Run a raw PromQL instant query against the environment's Prometheus.
    environment: 'dev' or 'prod'. promql: any valid PromQL expression."""
    if not promql:
        return "Please provide a PromQL expression."
    cfg = _env_cfg(environment)
    if not cfg["prometheus_url"]:
        return f"No Prometheus URL configured for environment '{environment}'."
    resp = requests.get(
        f"{cfg['prometheus_url']}/api/v1/query",
        params={"query": promql},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        return f"Prometheus query failed: {data}"
    result = data["data"]["result"]
    if not result:
        return f"No data returned for query: {promql}"
    lines = [f"Result for `{promql}` in '{environment}':"]
    for r in result:
        labels = ",".join(f'{k}="{v}"' for k, v in r["metric"].items())
        value = r["value"][1]
        lines.append(f"  {{{labels}}} = {value}")
    return "\n".join(lines)


@mcp.tool()
def get_cpu_usage(environment: str = "dev", minutes: int = 10) -> str:
    """Get current CPU usage percentage for the environment's instance, via
    node_exporter. environment: 'dev' or 'prod'. minutes: averaging window
    for the underlying rate() calculation (default 10)."""
    promql = (
        f'100 - (avg by (instance) '
        f'(rate(node_cpu_seconds_total{{mode="idle"}}[{minutes}m])) * 100)'
    )
    return query_prometheus(environment=environment, promql=promql)


if __name__ == "__main__":
    mcp.run(transport="stdio")
