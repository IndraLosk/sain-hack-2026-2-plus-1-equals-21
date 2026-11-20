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
_indexed_dataset_key: Optional[str] = None


def local_dataset() -> Dict[str, Any]:
    """Локальная копия набора (fallback, если API недоступен)."""
    path = Path(get_settings().dataset_path)
    if not path.exists():
        logger.warning("local dataset not found: %s", path)
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("dataset", raw)


async def fetch_dataset(
    *, token: Optional[str] = None, dataset_code: Optional[str] = None, allow_local_fallback: bool = True
) -> Dict[str, Any]:
    """Возвращает набор кейса 4; локальный fallback разрешён только для разработки.

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
        if not data and allow_local_fallback:
            data = local_dataset()
            source = "local"
        if data:
            _cache.update({"data": data, "at": time.time(), "source": source, "key": cache_key})
            logger.info("case4 dataset ready: source=%s articles=%d requests=%d",
                        source, len(data.get("articles") or []), len(data.get("requests") or []))
        return data or {}


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


async def ensure_dataset_indexed(
    *,
    token: Optional[str] = None,
    dataset_code: Optional[str] = None,
    dataset: Optional[Dict[str, Any]] = None,
    allow_local_fallback: bool = True,
) -> int:
    """Индексирует именно выданный набор и атомарно заменяет индекс при смене dataset_id."""
    from .config import get_catalog
    from .vectorstore import add_article_async, get_store

    global _indexed_dataset_key
    settings = get_settings()
    if dataset is None:
        dataset = await fetch_dataset(
            token=token, dataset_code=dataset_code, allow_local_fallback=allow_local_fallback
        )
    if not dataset:
        logger.warning("no dataset available for indexing (case4)")
        return 0

    code = dataset_code or settings.dataset_code
    dataset_key = str((dataset.get("meta") or {}).get("dataset_id") or code)
    store = get_store()
    if store.count() > 0 and _indexed_dataset_key == dataset_key:
        return 0
    if store.count() > 0:
        await asyncio.to_thread(store.reset)

    articles = dataset.get("articles") or []
    if not articles:
        logger.warning("dataset has no articles to index")
        return 0

    program_ids = [p.id for p in get_catalog().programs]
    preferred = "sleep_recovery" if "sleep_recovery" in program_ids else (program_ids[0] if program_ids else "sleep_recovery")
    indexed = 0
    for article in articles:
        sections = [
            {
                "section_id": str(s.get("section_id")),
                "title": str(s.get("title") or ""),
                "text": str(s.get("text") or ""),
            }
            for s in article.get("sections") or []
            if str(s.get("text") or "").strip()
        ]
        if not sections:
            continue
        year = int(article.get("year") or 2020)
        try:
            await add_article_async(
                program_id=preferred,
                title=str(article.get("title") or article.get("article_id")),
                published_date=f"{year:04d}-01-01",
                content="\n\n".join(s["text"] for s in sections),
                article_id=str(article.get("article_id")),
                sections=sections,
                extra_metadata={
                    "authors": ", ".join(article.get("authors") or []),
                    "doi": article.get("doi"),
                    "year": year,
                    "license": str((article.get("license") or {}).get("name") or ""),
                    "source": "case4",
                },
            )
            indexed += 1
        except Exception as exc:  # noqa: BLE001 — одна статья не должна ронять старт
            logger.warning("auto-index failed for article %s: %s", article.get("article_id"), exc)
    if indexed:
        _indexed_dataset_key = dataset_key
    logger.info("case4 auto-indexed %d articles (%d chunks)", indexed, store.count())
    return indexed


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


__all__ = ["fetch_dataset", "local_dataset", "dataset_info", "ensure_dataset_indexed"]
