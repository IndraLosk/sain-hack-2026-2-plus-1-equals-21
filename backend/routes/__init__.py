"""Слой HTTP-путей. Тонкие обёртки: принимают запрос, зовут services, отдают ответ."""

from .root import router as root_router
from .data import router as data_router
from .run import router as run_router

__all__ = ["root_router", "data_router", "run_router"]