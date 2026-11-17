"""Единая точка входа серверной части Sain Case 4.

Composition root: поднимает FastAPI-приложение (backend.app:app). Серверная
часть на стенде запускается через start.sh → python main.py serve.
БД в серверной части не используется: набор агенту даёт getData, все данные
приходят из API организаторов в /run.

Команда:
  serve   запустить FastAPI-приложение (uvicorn)
"""

import argparse
import os
import sys

# Позволяет импортировать пакет backend/ из корня проекта.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _serve(args):
    import uvicorn  # локальный импорт: нужен только для запуска сервера

    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=False)


def main():
    parser = argparse.ArgumentParser(prog="sain-case4", description=__doc__)
    parser.add_argument(
        "command",
        choices=["serve"],
        help="что сделать (для стенда нужна только serve)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    _serve(args)


if __name__ == "__main__":
    main()