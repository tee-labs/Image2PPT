#!/usr/bin/env bash
# Dev launcher — runs the FastAPI backend and the Vite dev server in
# parallel. Use this during development. For production, run
# start-prod.sh instead (builds the frontend, lets uvicorn serve it).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

cd "$REPO"

# Backend deps are kept separate from the main project requirements —
# users who only use the CLI never need fastapi/sqlalchemy/etc. We
# install into the same Python that runs scripts/convert.py (so the
# subprocess pipeline keeps finding paddleocr et al.), but as a
# separate, opt-in step. Homebrew / Debian Python is externally
# managed (PEP 668); fall back to --break-system-packages there.
if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "[start] Installing web-layer deps (web/backend/requirements.txt)…"
  if ! python3 -m pip install -r web/backend/requirements.txt 2>/dev/null; then
    echo "[start] Python is externally managed; retrying with --break-system-packages."
    python3 -m pip install --break-system-packages -r web/backend/requirements.txt
  fi
fi

# Frontend deps.
if [ ! -d "web/frontend/node_modules" ]; then
  echo "[start] Installing frontend deps…"
  (cd web/frontend && npm install)
fi

cleanup() {
  echo "[start] stopping…"
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Backend on :8000
echo "[start] backend → http://localhost:8000"
(cd "$REPO" && python3 -m uvicorn web.backend.app.main:app --host 0.0.0.0 --port 8000 --reload) &

# Frontend on :5173, proxies /api + /ws to :8000
echo "[start] frontend → http://localhost:5173"
(cd "$REPO/web/frontend" && npm run dev -- --host) &

wait
