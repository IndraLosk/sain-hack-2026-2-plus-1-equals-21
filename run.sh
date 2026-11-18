#!/usr/bin/env bash
# Запуск бэкенда «Эксперт как плейлист» (локально, без Ollama).
# LLM доступна через OpenAI-совместимый прокси (LLM_BASE_URL/LLM_API_KEY/LLM_MODEL).
set -euo pipefail
cd "$(dirname "$0")"

#if [ ! -d .venv ]; then
#  echo "→ создаю .venv (python3)"
#  python3 -m venv .venv
#  .venv/bin/python -m pip install --upgrade pip
#  .venv/bin/python -m pip install -r requirements.txt
#fi

echo "→ http://127.0.0.1:8000  (Swagger: /docs)"
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"