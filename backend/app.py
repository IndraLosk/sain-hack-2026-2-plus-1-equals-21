"""FastAPI-приложение кейса 4.

Сборщик приложения: регистрирует роутеры слоя routes/. Вся бизнес-логика
живёт в services/, пути — в routes/, набор агенту даёт client/ через getData.
БД в работе серверной части не участвует. Приложение само не стартует —
его поднимает корневой main.py (start.sh → команда serve).
"""

from fastapi import FastAPI

from .routes import root_router, run_router

app = FastAPI(title="Sain Case 4 — Эксперт как плейлист", version="0.1.0")

app.include_router(root_router)
app.include_router(run_router)