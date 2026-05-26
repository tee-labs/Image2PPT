#!/usr/bin/env bash
# Production launcher — builds the frontend once, then runs uvicorn
# serving both the static bundle and the API from :8000.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-1}"

cd "$REPO"

if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "[start-prod] Installing web-layer deps (web/backend/requirements.txt)…"
  if ! python3 -m pip install -r web/backend/requirements.txt 2>/dev/null; then
    echo "[start-prod] Python is externally managed; retrying with --break-system-packages."
    python3 -m pip install --break-system-packages -r web/backend/requirements.txt
  fi
fi

if [ ! -d "web/frontend/node_modules" ]; then
  echo "[start-prod] Installing frontend deps…"
  (cd web/frontend && npm install)
fi

echo "[start-prod] Building frontend…"
(cd web/frontend && npm run build)

echo "[start-prod] uvicorn → http://${HOST}:${PORT}"
exec python3 -m uvicorn web.backend.app.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
