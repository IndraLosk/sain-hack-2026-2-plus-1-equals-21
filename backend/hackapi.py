"""Клиент API хакатона: данные кейса 4 (`GET /case/4/getData`).

Набор запрашивается один раз и кешируется в памяти процесса (README стенда прямо просит
не дёргать getData на каждую запись). Если API недоступен или токен не задан —
используется локальная копия `data/case4-dev.json`, чтобы приложение работало всегда.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)

_cache: Dict[str, Any] = {"data": None, "at": 0.0, "source": ""}
_lock = asyncio.Lock()


def local_dataset() -> Dict[str, Any]:
    """Локальная копия набора (fallback, если API недоступен)."""
    path = Path(get_settings().dataset_path)
    if not path.exists():
        logger.warning("local dataset not found: %s", path)
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("dataset", raw)


async def fetch_dataset(*, token: Optional[str] = None, dataset_code: Optional[str] = None) -> Dict[str, Any]:
    """Возвращает набор кейса 4: сначала кеш, затем API хакатона, затем локальный файл.

    `token` — токен из заголовка Authorization (так стенд присылает токен проверки);
    если не передан, используется HACK_TEAM_TOKEN из окружения.
    """
    settings = get_settings()
    cache_key = dataset_code or settings.dataset_code
    fresh = _cache["data"] is not None and (time.time() - _cache["at"]) < settings.dataset_cache_ttl
    if fresh and _cache.get("key") == cache_key:
        return _cache["data"]

    async with _lock:
        if _cache["data"] is not None and _cache.get("key") == cache_key and (
            time.time() - _cache["at"]
        ) < settings.dataset_cache_ttl:
            return _cache["data"]

        data = await _fetch_remote(token=token, dataset_code=cache_key)
        source = "api"
        if not data:
            data = local_dataset()
            source = "local"
        if data:
            _cache.update({"data": data, "at": time.time(), "source": source, "key": cache_key})
            logger.info("case4 dataset ready: source=%s articles=%d requests=%d",
                        source, len(data.get("articles") or []), len(data.get("requests") or []))
        return data


async def _fetch_remote(*, token: Optional[str], dataset_code: str) -> Optional[Dict[str, Any]]:
    settings = get_settings()
    bearer = (token or "").strip() or settings.hack_team_token.strip()
    if not settings.hack_api_url.strip() or not bearer:
        return None
    if not bearer.lower().startswith("bearer "):
        bearer = f"Bearer {bearer}"
    url = f"{settings.hack_api_url.rstrip('/')}/case/4/getData"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(url, params={"dataset_code": dataset_code}, headers={"Authorization": bearer})
            resp.raise_for_status()
            payload = resp.json()
        data = payload.get("dataset") or {}
        return data or None
    except Exception as exc:  # noqa: BLE001 — токен и данные в лог не пишем
        logger.warning("getData failed (%s): %s", type(exc).__name__, getattr(exc, "status_code", ""))
        return None


def dataset_info() -> Dict[str, Any]:
    """Служебная информация о загруженном наборе (без секретов)."""
    data = _cache.get("data") or {}
    meta = data.get("meta") or {}
    return {
        "source": _cache.get("source") or "none",
        "dataset_id": meta.get("dataset_id"),
        "articles": len(data.get("articles") or []),
        "requests": len(data.get("requests") or []),
        "loaded_at": _cache.get("at") or None,
    }


__all__ = ["fetch_dataset", "local_dataset", "dataset_info"]
