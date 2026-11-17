#!/bin/sh
# Запуск серверной части на стенде.
# Стенд: Python 3.11, зависимости из requirements.txt, порт в env PORT (на стенде 8000).
# Сервер слушает 0.0.0.0:$PORT.
exec python3 main.py serve --host 0.0.0.0 --port "${PORT:-8000}"