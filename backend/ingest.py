"""Загрузка документов в векторную базу: ссылка на статью или файл.

Поддерживается:
* ссылка (html-страница статьи) — скачиваем, вытаскиваем заголовок и основной текст;
* файл `.txt` / `.md` — текст как есть;
* файл `.pdf` — текст через pypdf (если установлен), иначе понятная ошибка.

Дальше — обычный путь индексации: `vectorstore.add_article_async` (чанки + bge-m3 onnx).
Статус индексации хранится в SQLite (`backend/article_store.py`), чтобы UI мог
показать «в очереди / проиндексировано / ошибка».
"""
from __future__ import annotations

import asyncio
import html as html_lib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .article_store import update_article_status
from .vectorstore import add_article_async

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024  # 8 МБ — больше в память не тянем
USER_AGENT = "SainBot/2.0 (+knowledge-base ingestion; contact: hackathon)"

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.S | re.I)
_META_TITLE_RE = re.compile(
    r'<meta[^>]+(?:property|name)=["\'](?:og:title|twitter:title|title)["\'][^>]+content=["\'](.*?)["\']',
    re.S | re.I,
)
_MAIN_RE = re.compile(r"<(article|main)[^>]*>(.*?)</\1>", re.S | re.I)
_P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.S | re.I)


def _clean(text: str) -> str:
    text = html_lib.unescape(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _html_to_text(raw: str) -> Tuple[str, str]:
    """Возвращает (title, text) из html-страницы."""
    title = ""
    for pattern in (_META_TITLE_RE, _TITLE_RE, _H1_RE):
        match = pattern.search(raw)
        if match:
            title = _clean(_TAG_RE.sub(" ", match.group(1)))
            if title:
                break

    body = _SCRIPT_RE.sub(" ", raw)
    main = _MAIN_RE.search(body)
    if main:
        body = main.group(2)
    paragraphs = [_clean(_TAG_RE.sub(" ", p)) for p in _P_RE.findall(body)]
    paragraphs = [p for p in paragraphs if len(p) > 60]
    if paragraphs:
        return title, "\n\n".join(paragraphs)
    # fallback: весь видимый текст
    return title, _clean(_TAG_RE.sub(" ", body))


async def fetch_url(url: str) -> Tuple[str, str]:
    """Скачивает страницу и возвращает (title, text)."""
    if not re.match(r"^https?://", url or "", re.I):
        raise ValueError("ссылка должна начинаться с http:// или https://")
    async with httpx.AsyncClient(
        timeout=30.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        raw = response.content[:MAX_DOWNLOAD_BYTES]
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        return _pdf_title(url), _pdf_to_text(raw)
    charset = response.encoding or "utf-8"
    try:
        page = raw.decode(charset, errors="replace")
    except LookupError:  # pragma: no cover
        page = raw.decode("utf-8", errors="replace")
    title, text = _html_to_text(page)
    if len(text) < 200:
        raise ValueError("не удалось извлечь текст со страницы (возможно, нужен вход или это JS-приложение)")
    return title or url, text


def _pdf_title(url: str) -> str:
    name = url.rstrip("/").split("/")[-1]
    return name or "PDF-документ"


def _pdf_to_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ValueError("для PDF нужен пакет pypdf: pip install pypdf") from exc
    import io

    reader = PdfReader(io.BytesIO(raw))
    pages = [(page.extract_text() or "") for page in reader.pages[:60]]
    text = "\n\n".join(_clean(p) for p in pages if p and p.strip())
    if len(text) < 100:
        raise ValueError("в PDF не нашлось текстового слоя (похоже на скан)")
    return text


def file_to_text(filename: str, raw: bytes) -> Tuple[str, str]:
    """Текст из загруженного файла: .txt/.md/.csv как есть, .pdf через pypdf."""
    name = (filename or "document").strip()
    lower = name.lower()
    if lower.endswith(".pdf"):
        return name.rsplit(".", 1)[0], _pdf_to_text(raw)
    if lower.endswith((".txt", ".md", ".markdown", ".csv", ".json")):
        text = raw.decode("utf-8", errors="replace")
        return name.rsplit(".", 1)[0], _clean(text)
    # неизвестное расширение — пробуем как текст
    text = raw.decode("utf-8", errors="replace")
    if len(text.strip()) < 50:
        raise ValueError("поддерживаются .pdf, .txt, .md, .csv")
    return name, _clean(text)


async def index_text(
    *,
    article_id: str,
    program_id: str,
    title: str,
    text: str,
    published_date: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Индексирует текст в векторной БД и обновляет статус в SQLite."""
    date = published_date or datetime.now(timezone.utc).date().isoformat()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        date = datetime.now(timezone.utc).date().isoformat()
    try:
        _, chunks = await add_article_async(
            program_id=program_id,
            title=title[:500],
            published_date=date,
            content=text,
            article_id=article_id,
            sections=None,
            extra_metadata=extra_metadata or {},
        )
        await asyncio.to_thread(update_article_status, article_id, status="indexed", chunks=chunks)
        logger.info("document indexed: %s (%d chunks)", article_id, chunks)
        return chunks
    except Exception as exc:  # noqa: BLE001
        logger.exception("document indexing failed: %s", article_id)
        await asyncio.to_thread(update_article_status, article_id, status="failed", error=str(exc))
        raise


async def ingest_link(
    *,
    article_id: str,
    program_id: str,
    url: str,
    title_hint: str = "",
    fallback_text: str = "",
) -> int:
    """Скачивает ссылку и индексирует. Если страницу не отдать — берём title/комментарий."""
    try:
        title, text = await fetch_url(url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("link fetch failed (%s): %s", url, exc)
        text = (fallback_text or "").strip()
        if len(text) < 80:
            await asyncio.to_thread(
                update_article_status,
                article_id,
                status="failed",
                error=f"не удалось прочитать ссылку: {exc}",
            )
            raise ValueError(f"не удалось прочитать ссылку: {exc}") from exc
        title = title_hint or url
    return await index_text(
        article_id=article_id,
        program_id=program_id,
        title=title_hint or title,
        text=text,
        extra_metadata={"source_url": url},
    )


__all__ = ["fetch_url", "file_to_text", "index_text", "ingest_link"]
