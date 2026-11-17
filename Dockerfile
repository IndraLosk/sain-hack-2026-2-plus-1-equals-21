# Образ приложения Sain Case 4.
# Ставит зависимости и запускает весь проект через единый main.py.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Сначала зависимости — пока кэш образа общий.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Затем весь проект (main.py + пакеты db/ и backend/).
COPY main.py .
COPY db ./db
COPY backend ./backend

# Порт FastAPI
EXPOSE 8000

# По умолчанию поднимаем сервер через main.
# При необходимости переопределяется командой (например, make seed).
CMD ["python", "main.py", "serve"]