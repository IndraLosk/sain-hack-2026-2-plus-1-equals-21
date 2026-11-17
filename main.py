"""Единая точка входа проекта Sain Case 4 (composition root).

Связывает пакеты db/ (команды и данные) и backend/ (FastAPI-приложение).
Логика команд живёт в db/cli.py, запуск сервера — в backend. Здесь только
разбор аргументов и вызов нужного обработчика.

Команды:
  seed       создать схему и загрузить данные из JSON (по умолчанию case4-dev.json)
  reset      очистить таблицы и заново залить JSON
  status     показать число строк по таблицам
  health     проверить подключение к БД
  serve      запустить FastAPI-приложение (uvicorn)
"""

import argparse
import logging
import os
import sys

# Позволяет импортировать пакеты db/ и backend/ из корня проекта.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db import cli


def _handlers():
    return {
        "seed": lambda a: cli.seed(a.json),
        "reset": lambda a: cli.reset(a.json),
        "status": lambda a: cli.status(),
        "health": lambda a: cli.health(),
        "serve": lambda a: _serve(a),
    }


def _serve(args):
    import uvicorn  # локальный импорт: нужен только для запуска сервера

    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=False)


def main():
    parser = argparse.ArgumentParser(prog="sain-case4", description=__doc__)
    parser.add_argument(
        "command",
        choices=["seed", "reset", "status", "serve", "health"],
        help="что сделать",
    )
    parser.add_argument("--json", default="case4-dev.json", help="путь к JSON набора")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _handlers()[args.command](args)


if __name__ == "__main__":
    main()