"""Пакет работы с базой данных: создание схемы и обработка (загрузка) данных.

Используется из единого main.py, сам ничего не запускает.
Зависит только от psycopg и не знает про backend/.
"""

from .db import (
    get_connection,
    create_schema,
    seed_from_json,
    truncate_all,
    row_counts,
    sample,
)
from . import cli

__all__ = [
    "get_connection",
    "create_schema",
    "seed_from_json",
    "truncate_all",
    "row_counts",
    "sample",
    "cli",
]