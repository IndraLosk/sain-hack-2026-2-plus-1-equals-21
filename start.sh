#!/bin/sh
# Запуск серверной части на стенде.
# Стенд: Python 3.11, зависимости из requirements.txt, порт в env PORT (на стенде 8000).
# Сервер слушает 0.0.0.0:$PORT и отдаёт и API (/api/v1/*, /run, /health), и фронт (web/index.html).
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"