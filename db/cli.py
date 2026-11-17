"""CLI-команды работы с базой и приложением.

Вызываются из корневого main.py (единственная точка входа). Логика команд
живёт здесь, а не в main.py, чтобы тот оставался тонким композит-рутом.
Для вывода используется логгер, а не print.
"""

import logging

from db import get_connection, create_schema, seed_from_json, truncate_all, row_counts

log = logging.getLogger("sain-case4")


def seed(path: str) -> None:
    """Создать схему и загрузить данные из JSON."""
    with get_connection() as conn:
        log.info("Creating schema")
        create_schema(conn)
        log.info("Seeding from %s", path)
        seed_from_json(conn, path)
        conn.commit()
        for table, n in row_counts(conn).items():
            log.info("  %-18s %s", table, n)


def reset(path: str) -> None:
    """Очистить таблицы и залить данные заново."""
    with get_connection() as conn:
        log.info("Truncating all tables")
        truncate_all(conn)
        log.info("Creating schema")
        create_schema(conn)
        log.info("Seeding from %s", path)
        seed_from_json(conn, path)
        conn.commit()
        for table, n in row_counts(conn).items():
            log.info("  %-18s %s", table, n)


def status() -> None:
    """Показать число строк по таблицам."""
    with get_connection() as conn:
        for table, n in row_counts(conn).items():
            log.info("  %-18s %s", table, n)


def health() -> None:
    """Проверить подключение к БД."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    log.info("DB ok")