#!/usr/bin/env bash
# «Эксперт как плейлист» (ассистент Сэйн) — запуск одной командой, без docker.
#
#   ./run.sh              поднять всё и открыть http://127.0.0.1:8000
#   REBUILD_UI=1 ./run.sh пересобрать фронтенд
#   PORT=9000 ./run.sh    другой порт
#
# Структура проекта:
#   backend/   FastAPI + LangGraph + Chroma + ONNX (bge-m3) + gpt-6-luna
#   frontend/  React + TypeScript + Tailwind (сборка Vite → frontend/dist)
#
# Что делает скрипт:
#   1) создаёт .venv и ставит зависимости бэкенда — если окружения ещё нет;
#   2) создаёт .env с дефолтами (gpt-6-luna + bge-m3/ONNX) — если файла нет;
#   3) собирает фронтенд в frontend/dist — если есть node/npm и сборки нет;
#   4) запускает uvicorn: API + собранный UI на одном порту.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

# --- 1. Python-окружение -----------------------------------------------------
#if [ ! -d .venv ]; then
#  echo "→ создаю .venv и ставлю зависимости бэкенда"
#  python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv
#  .venv/bin/python -m pip install --upgrade pip >/dev/null
#  .venv/bin/python -m pip install -r requirements.txt
#fi

# --- 2. Конфигурация ---------------------------------------------------------
if [ ! -f .env ]; then
  echo "→ создаю .env из env.example (впишите LLM_API_KEY)"
  cp env.example .env
fi

# --- 3. Фронтенд -------------------------------------------------------------
if command -v npm >/dev/null 2>&1; then
  if [ ! -f frontend/dist/index.html ] || [ -n "${REBUILD_UI:-}" ]; then
    echo "→ собираю фронтенд (frontend/dist)"
    (cd frontend && npm install --no-audit --no-fund && npm run build)
  fi
else
  echo "→ node/npm не найдены: API будет доступен, UI не собран"
fi

# --- 4. Запуск ---------------------------------------------------------------
echo "→ http://${HOST}:${PORT}  (Swagger: /docs)"
exec python3 -m uvicorn backend.main:app --host "$HOST" --port "$PORT" --reload
