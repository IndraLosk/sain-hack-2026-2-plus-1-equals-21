"""Сервисный слой: бизнес-логика приложения.

Собирает обработчики данных и health-проверку. Используется роутами через
этот пакет. Не зависит от HTTP-части.
"""

from .services import (
    check_health,
    stats,
    list_experts,
    list_articles,
    list_requests,
    run_case,
)

__all__ = [
    "check_health",
    "stats",
    "list_experts",
    "list_articles",
    "list_requests",
    "run_case",
]