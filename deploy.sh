#!/usr/bin/env bash
# deploy.sh — always build before up, so a changed .env or source file
# can never be masked by a stale cached/pulled image.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "==> Building services (only changed ones will actually rebuild)..."
docker compose build

echo "==> Bringing up the stack..."
docker compose up -d

echo "==> Done. Current status:"
docker compose ps
