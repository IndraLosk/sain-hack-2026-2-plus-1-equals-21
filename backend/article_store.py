"""Статьи, рекомендованные пользователями, и фрагменты, реально попавшие в ответ.

Зачем отдельная таблица, если есть векторная БД:
* векторная БД хранит только чанки и метаданные — там нет «человеческого» следа
  (кто прислал ссылку, когда, что именно проиндексировано);
* для экрана «Список статей» нужно показать пользователю ровно те куски,
  которые агент подставил в ответ, а не всю статью целиком.

Таблицы (тот же движок, что и backend/storage.py — SQLite по умолчанию):
* `articles`         — рекомендованные статьи: ссылка, заголовок, статус индексации;
* `article_chunks`   — фрагменты, которые агент реально использовал в ответе.

Все функции синхронные: вызываются из `asyncio.to_thread`, чтобы не блокировать
event loop (SQLAlchemy Core/ORM с обычным sqlite3-драйвером).
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import Float, Integer, String, Text, UniqueConstraint, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import get_settings

logger = logging.getLogger(__name__)

ARTICLE_STATUSES = {"pending", "indexed", "failed"}


class ArticleBase(DeclarativeBase):
    pass


class ArticleRow(ArticleBase):
    """Рекомендованная пользователем статья (ссылка + результат индексации)."""

    __tablename__ = "articles"
    __table_args__ = (UniqueConstraint("url", name="uq_articles_url"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: f"art_{uuid.uuid4().hex[:12]}")
    url: Mapped[str] = mapped_column(String(1024), index=True)
    title: Mapped[str] = mapped_column(String(500))
    program_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(128), index=True, nullable=True)
    comment: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)  # pending|indexed|failed
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    source_kind: Mapped[str] = mapped_column(String(16), default="link")  # link|file|text
    created_at: Mapped[float] = mapped_column(Float, default=time.time, index=True)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)


class ArticleChunkRow(ArticleBase):
    """Фрагмент статьи, который агент использовал при ответе."""

    __tablename__ = "article_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex)
    article_id: Mapped[str] = mapped_column(String(64), index=True)
    article_title: Mapped[str] = mapped_column(String(500), default="")
    section_id: Mapped[str] = mapped_column(String(128), default="")
    section_title: Mapped[str] = mapped_column(String(500), default="")
    text: Mapped[str] = mapped_column(Text, default="")
    query: Mapped[str] = mapped_column(Text, default="")
    session_id: Mapped[Optional[str]] = mapped_column(String(128), index=True, nullable=True)
    expert_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    similarity: Mapped[float] = mapped_column(Float, default=0.0)
    used_at: Mapped[float] = mapped_column(Float, default=time.time, index=True)


# --------------------------------------------------------------------------- #
# Движок (синхронный)
# --------------------------------------------------------------------------- #
_engine = None
_session_factory: Optional[sessionmaker[Session]] = None


def _sqlite_file() -> Path:
    url = get_settings().effective_database_url
    if url.startswith("sqlite"):
        raw = url.split("///", 1)[-1]
        return Path(raw)
    # PostgreSQL: отдельный файл не нужен, но схему создаёт тот же движок
    return Path(get_settings().sqlite_path)


def get_engine():
    """Синхронный движок к той же БД, что и storage.py (SQLite или PostgreSQL)."""
    global _engine, _session_factory
    if _engine is not None:
        return _engine
    url = get_settings().effective_database_url
    if url.startswith("sqlite"):
        path = _sqlite_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{path}", echo=False, future=True)
    else:
        sync_url = url.replace("+asyncpg", "+psycopg2").replace("+aiosqlite", "")
        _engine = create_engine(sync_url, echo=False, future=True, pool_pre_ping=True)
    ArticleBase.metadata.create_all(_engine)
    _session_factory = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def _session() -> Session:
    get_engine()
    assert _session_factory is not None
    return _session_factory()


def init_article_db() -> None:
    """Идемпотентно создаёт таблицы статей и использованных фрагментов."""
    get_engine()
    logger.info("article store ready")


def dispose_article_db() -> None:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


# --------------------------------------------------------------------------- #
# Статьи
# --------------------------------------------------------------------------- #
def _article_dict(row: ArticleRow) -> Dict[str, Any]:
    return {
        "id": row.id,
        "url": row.url,
        "title": row.title,
        "program_id": row.program_id,
        "user_id": row.user_id,
        "comment": row.comment or "",
        "status": row.status,
        "chunks": int(row.chunks or 0),
        "error": row.error or "",
        "source_kind": row.source_kind or "link",
        "created_at": float(row.created_at or 0),
        "updated_at": float(row.updated_at or 0),
    }


def upsert_article(
    *,
    url: str,
    title: str,
    program_id: str,
    user_id: Optional[str] = None,
    comment: str = "",
    source_kind: str = "link",
    status: str = "pending",
    article_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Сохраняет рекомендованную статью. Повторная ссылка обновляет запись."""
    url = (url or "").strip()[:1024]
    title = " ".join((title or "").split())[:500]
    now = time.time()
    with _session() as db:
        row = db.execute(select(ArticleRow).where(ArticleRow.url == url)).scalars().first() if url else None
        if row is None:
            row = ArticleRow(
                id=article_id or f"art_{uuid.uuid4().hex[:12]}",
                url=url,
                title=title or url,
                program_id=program_id,
                user_id=user_id,
                comment=(comment or "")[:2000],
                source_kind=source_kind,
                status=status,
                created_at=now,
                updated_at=now,
            )
            db.add(row)
        else:
            row.title = title or row.title
            row.program_id = program_id or row.program_id
            row.comment = (comment or row.comment or "")[:2000]
            row.source_kind = source_kind or row.source_kind
            row.status = status or row.status
            row.updated_at = now
            if user_id:
                row.user_id = user_id
        db.commit()
        db.refresh(row)
        return _article_dict(row)


def update_article_status(article_id: str, *, status: str, chunks: int = 0, error: str = "") -> None:
    with _session() as db:
        row = db.get(ArticleRow, article_id)
        if row is None:
            return
        row.status = status if status in ARTICLE_STATUSES else "pending"
        row.chunks = int(chunks or 0)
        row.error = (error or "")[:2000]
        row.updated_at = time.time()
        db.commit()


def list_articles(program_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    with _session() as db:
        stmt = select(ArticleRow).order_by(ArticleRow.created_at.desc()).limit(limit)
        if program_id:
            stmt = stmt.where(ArticleRow.program_id == program_id)
        rows = db.execute(stmt).scalars().all()
        return [_article_dict(r) for r in rows]


def get_article(article_id: str) -> Optional[Dict[str, Any]]:
    with _session() as db:
        row = db.get(ArticleRow, article_id)
        return _article_dict(row) if row is not None else None


def delete_article(article_id: str) -> bool:
    with _session() as db:
        row = db.get(ArticleRow, article_id)
        if row is None:
            return False
        db.execute(delete(ArticleChunkRow).where(ArticleChunkRow.article_id == article_id))
        db.delete(row)
        db.commit()
        return True


# --------------------------------------------------------------------------- #
# Фрагменты, которые реально использовал агент
# --------------------------------------------------------------------------- #
def record_used_chunks(
    *,
    query: str,
    session_id: Optional[str],
    expert_id: Optional[str],
    chunks: List[Dict[str, Any]],
) -> int:
    """Пишет фрагменты, подставленные агенту в контекст (дедуп по статье+разделу+тексту)."""
    if not chunks:
        return 0
    now = time.time()
    written = 0
    with _session() as db:
        for chunk in chunks:
            article_id = str(chunk.get("article_id") or "")
            if not article_id:
                continue
            text = " ".join(str(chunk.get("document") or "").split())[:4000]
            section_id = str(chunk.get("section_id") or "")
            existing = (
                db.execute(
                    select(ArticleChunkRow).where(
                        ArticleChunkRow.article_id == article_id,
                        ArticleChunkRow.section_id == section_id,
                        ArticleChunkRow.text == text,
                    )
                )
                .scalars()
                .first()
            )
            if existing is not None:
                existing.used_at = now
                existing.query = (query or existing.query or "")[:2000]
                existing.similarity = max(float(existing.similarity or 0.0), float(chunk.get("similarity") or 0.0))
                written += 1
                continue
            db.add(
                ArticleChunkRow(
                    article_id=article_id,
                    article_title=str(chunk.get("title") or "")[:500],
                    section_id=section_id[:128],
                    section_title=str(chunk.get("section_title") or "")[:500],
                    text=text,
                    query=(query or "")[:2000],
                    session_id=session_id,
                    expert_id=expert_id,
                    similarity=float(chunk.get("similarity") or 0.0),
                    used_at=now,
                )
            )
            written += 1
        db.commit()
    return written


def list_used_chunks(
    article_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    with _session() as db:
        stmt = select(ArticleChunkRow).order_by(ArticleChunkRow.used_at.desc()).limit(limit)
        if article_id:
            stmt = stmt.where(ArticleChunkRow.article_id == article_id)
        if session_id:
            stmt = stmt.where(ArticleChunkRow.session_id == session_id)
        rows = db.execute(stmt).scalars().all()
        return [
            {
                "id": r.id,
                "article_id": r.article_id,
                "article_title": r.article_title,
                "section_id": r.section_id,
                "section_title": r.section_title,
                "text": r.text,
                "query": r.query,
                "session_id": r.session_id,
                "expert_id": r.expert_id,
                "similarity": round(float(r.similarity or 0.0), 3),
                "used_at": float(r.used_at or 0),
            }
            for r in rows
        ]


def used_chunks_summary() -> Dict[str, int]:
    with _session() as db:
        articles = len(db.execute(select(ArticleRow.id)).scalars().all())
        chunks = len(db.execute(select(ArticleChunkRow.id)).scalars().all())
    return {"articles": articles, "used_chunks": chunks}


__all__ = [
    "init_article_db",
    "dispose_article_db",
    "upsert_article",
    "update_article_status",
    "list_articles",
    "get_article",
    "delete_article",
    "record_used_chunks",
    "list_used_chunks",
    "used_chunks_summary",
]
