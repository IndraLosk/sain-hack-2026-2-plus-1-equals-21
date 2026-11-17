#!/usr/bin/env bash
# Запуск бэкенда «Эксперт как плейлист» (macOS Apple Silicon, без docker).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "→ создаю .venv (python3.11)"
  python3.11 -m venv .venv || python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

# Ollama должна быть запущена и модель скачана
if ! curl -s -m 3 http://127.0.0.1:11434/api/tags > /dev/null; then
  echo "→ запускаю ollama serve"
  (nohup ollama serve > /tmp/ollama.log 2>&1 &)
  sleep 5
fi

echo "→ http://127.0.0.1:8000  (Swagger: /docs)"
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
