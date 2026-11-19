#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

# --- Фронтенд -------------------------------------------------------------
if command -v npm >/dev/null 2>&1; then
  if [ ! -f frontend/dist/index.html ] || [ -n "${REBUILD_UI:-}" ]; then
    echo "→ собираю фронтенд (frontend/dist)"
    (cd frontend && npm install --no-audit --no-fund && npm run build)
  fi
else
  echo "→ node/npm не найдены: API будет доступен, UI не собран"
fi

# --- Запуск ---------------------------------------------------------------
echo "→ http://${HOST}:${PORT}  (Swagger: /docs)"
exec python3 -m uvicorn backend.main:app --host "$HOST" --port "$PORT" --reload
