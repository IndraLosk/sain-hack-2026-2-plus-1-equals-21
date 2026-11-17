"""Клиент к API сервиса организаторов (хакатон Sain).

Приложение получает набор через GET /case/4/getData по dataset_code + токену
прогона. На стенде переменные окружения уже заданы организаторами:
  HACK_API_URL    — базовый URL сервиса хакатона (https://hack-api.be-healthy.tech)
  HACK_TEAM_TOKEN — токен команды (разработка)
На онлайн-показе внутри /run токен берётся из заголовка входящего запроса,
а не из HACK_TEAM_TOKEN (см. инструкцию, шаг 5).
"""

import os

import httpx

# На стенде сервис хакатона доступен по HACK_API_URL.
_BASE = os.environ.get("HACK_API_URL", "")


def fetch_dataset(dataset_code: str, run_token: str) -> dict:
    """Вызвать GET /case/4/getData и вернуть объект `dataset`.

    Возвращает сам блок dataset (dict с ключами набора: meta, users, ...).
    Токен передаётся БЕЗ изменений: на dev это HACK_TEAM_TOKEN, на онлайн-показе —
    токен из входящего /run (их нельзя подменять).
    """
    if not _BASE:
        raise RuntimeError(
            "HACK_API_URL не задан: переменная окружения организаторов. "
            "На стенде она уже настроена; локально задайте вручную."
        )
    if not run_token:
        raise RuntimeError("run_token пуст: передайте Bearer-токен.")

    url = f"{_BASE.rstrip('/')}/case/4/getData"
    resp = httpx.get(
        url,
        params={"dataset_code": dataset_code},
        headers={"Authorization": f"Bearer {run_token}"},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload["dataset"]