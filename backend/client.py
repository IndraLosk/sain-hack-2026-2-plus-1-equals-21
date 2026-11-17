"""Клиент к API сервиса организаторов (хакатон Sain).

Приложение получает набор через GET /case/4/getData по dataset_code + токену.
На стенде переменные окружения уже заданы:
  HACK_API_URL    — базовый URL сервиса (по умолчанию https://hack-api.be-healthy.tech)
  HACK_TEAM_TOKEN — токен команды (разработка)
На онлайн-показе внутри /run токен берётся из заголовка входящего запроса,
а не из HACK_TEAM_TOKEN (см. инструкцию, шаг 5). Ключи в этом модуле не
выводятся и не логируются.
"""

import os

import httpx

# Сервис хакатона. На стенде задаётся организаторами, локально берётся дефолт.
_BASE = os.environ.get("HACK_API_URL", "https://hack-api.be-healthy.tech")

# dev-код набора кейса 4 (dataset_id), используется, если не передан явно.
DEV_DATASET_CODE = os.environ.get("HACK_DATASET_CODE", "case4-dev-v2")

# Файл токена команды из архива (локальная разработка). Не логируем его.
_TOKEN_FILE = os.environ.get("HACK_TOKEN_FILE", "api.token")


def _team_token() -> str:
    """Токен команды из env HACK_TEAM_TOKEN или файла api.token (не раскрывая)."""
    tok = os.environ.get("HACK_TEAM_TOKEN")
    if tok:
        return tok.strip()
    if os.path.exists(_TOKEN_FILE):
        with open(_TOKEN_FILE, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def fetch_dataset(dataset_code: str | None = None, run_token: str | None = None) -> dict:
    """Вызвать GET /case/4/getData и вернуть объект `dataset`.

    dataset_code: если не задан, берётся DEV_DATASET_CODE (для dev-проверки).
    run_token: если не задан, берётся токен команды (на стенде /run передаёт
    свой токен явно в run_token — он имеет приоритет). Токен не логируем.
    """
    code = dataset_code or DEV_DATASET_CODE
    token = run_token or _team_token()
    if not code:
        raise RuntimeError("dataset_code не задан")
    if not token:
        raise RuntimeError(
            "нет токена: передайте run_token (на /run) или задайте HACK_TEAM_TOKEN / api.token"
        )

    url = f"{_BASE.rstrip('/')}/case/4/getData"
    resp = httpx.get(
        url,
        params={"dataset_code": code},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload["dataset"]