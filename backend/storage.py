"""Персистентность: сессии, журнал «пробелов знаний», память о пользователе.

По умолчанию SQLite (`data/app.sqlite`, драйвер aiosqlite).
Если задан `DATABASE_URL=postgresql+asyncpg://...`, тот же код работает на PostgreSQL.

Таблицы:
* `sessions`       — диалоги (id, program_id, expert_id, user_id, время);
* `knowledge_gaps` — запросы, по которым в векторной БД ничего не нашлось;
* `user_memory`    — персонализация: устойчивые факты о пользователе.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import Float, Integer, String, Text, UniqueConstraint, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

from .config import get_settings

logger = logging.getLogger(__name__)

MEMORY_KEYS = {"sleep_schedule", "sleep_problem", "constraints", "goal", "context", "preference"}


class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(128), index=True, nullable=True)
    program_id: Mapped[str] = mapped_column(String(64), index=True)
    expert_id: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time, index=True)


class KnowledgeGapRow(Base):
    __tablename__ = "knowledge_gaps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(128), index=True, nullable=True)
    program_id: Mapped[str] = mapped_column(String(64), index=True)
    expert_id: Mapped[str] = mapped_column(String(64))
    query: Mapped[str] = mapped_column(Text)
    normalized_query: Mapped[str] = mapped_column(String(512), index=True)
    top_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    occurrences: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="open", index=True)  # open|indexed|rejected
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time, index=True)


class UserMemoryRow(Base):
    __tablename__ = "user_memory"
    __table_args__ = (UniqueConstraint("user_id", "key", name="uq_user_memory_user_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    key: Mapped[str] = mapped_column(String(64))
    value: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.6)
    source: Mapped[str] = mapped_column(String(32), default="llm")
    updated_at: Mapped[float] = mapped_column(Float, default=time.time, index=True)


# --------------------------------------------------------------------------- #
# Движок
# --------------------------------------------------------------------------- #
_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _engine, _sessionmaker
    if _engine is None:
        url = get_settings().effective_database_url
        if url.startswith("sqlite"):
            Path(url.split("///", 1)[-1]).parent.mkdir(parents=True, exist_ok=True)
            _engine = create_async_engine(url, echo=False, future=True)
        else:
            _engine = create_async_engine(
                url, echo=False, future=True, pool_size=10, max_overflow=20, pool_pre_ping=True
            )
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
        logger.info("database engine: %s", url.split("@")[-1])
    assert _sessionmaker is not None
    return _sessionmaker


async def init_db() -> None:
    """Создаёт схему (идемпотентно) и включает WAL для SQLite.

    WAL даёт параллельные чтения во время записи; файлы -wal/-shm при этом
    появляются рядом с БД и не должны попадать в git (см. .gitignore).
    """
    async with get_sessionmaker()() as db:
        conn = await db.connection()
        if conn.dialect.name == "sqlite":
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            await conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
        await conn.run_sync(Base.metadata.create_all)


async def dispose_db() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().replace("ё", "е").split())[:512]


# --------------------------------------------------------------------------- #
# Сессии
# --------------------------------------------------------------------------- #
async def touch_session(*, session_id: str, program_id: str, expert_id: str, user_id: Optional[str] = None) -> None:
    async with get_sessionmaker()() as db:
        row = await db.get(SessionRow, session_id)
        now = time.time()
        if row is None:
            db.add(SessionRow(id=session_id, user_id=user_id, program_id=program_id, expert_id=expert_id,
                              created_at=now, updated_at=now))
        else:
            row.program_id = program_id
            row.expert_id = expert_id
            row.updated_at = now
            if user_id:
                row.user_id = user_id
        await db.commit()


async def delete_session(session_id: str) -> None:
    async with get_sessionmaker()() as db:
        row = await db.get(SessionRow, session_id)
        if row is not None:
            await db.delete(row)
            await db.commit()


# --------------------------------------------------------------------------- #
# Журнал пробелов знаний
# --------------------------------------------------------------------------- #
async def record_knowledge_gap(
    *,
    session_id: str,
    program_id: str,
    expert_id: str,
    query: str,
    user_id: Optional[str] = None,
    top_similarity: float = 0.0,
) -> None:
    """Пишет запрос, по которому в векторной БД ничего не нашлось (с дедупликацией)."""
    normalized = _normalize(query)
    now = time.time()
    async with get_sessionmaker()() as db:
        existing = (
            await db.execute(
                select(KnowledgeGapRow).where(
                    KnowledgeGapRow.normalized_query == normalized,
                    KnowledgeGapRow.program_id == program_id,
                )
            )
        ).scalars().first()
        if existing is not None:
            existing.occurrences = (existing.occurrences or 1) + 1
            existing.updated_at = now
            existing.top_similarity = max(existing.top_similarity or 0.0, top_similarity)
        else:
            db.add(
                KnowledgeGapRow(
                    session_id=session_id,
                    user_id=user_id,
                    program_id=program_id,
                    expert_id=expert_id,
                    query=query[:4000],
                    normalized_query=normalized,
                    top_similarity=top_similarity,
                    created_at=now,
                    updated_at=now,
                )
            )
            logger.info("knowledge gap recorded: %s", normalized[:80])
        await db.commit()


async def list_knowledge_gaps(status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    async with get_sessionmaker()() as db:
        stmt = select(KnowledgeGapRow).order_by(KnowledgeGapRow.updated_at.desc()).limit(limit)
        if status:
            stmt = stmt.where(KnowledgeGapRow.status == status)
        rows = (await db.execute(stmt)).scalars().all()
    return [
        {
            "id": r.id,
            "query": r.query,
            "program_id": r.program_id,
            "expert_id": r.expert_id,
            "session_id": r.session_id,
            "occurrences": r.occurrences,
            "top_similarity": round(r.top_similarity, 3),
            "status": r.status,
            "created_at": r.created_at,
            "updated_at": r.updated_at,
        }
        for r in rows
    ]


async def set_gap_status(gap_id: str, status: str) -> bool:
    async with get_sessionmaker()() as db:
        row = await db.get(KnowledgeGapRow, gap_id)
        if row is None:
            return False
        row.status = status
        row.updated_at = time.time()
        await db.commit()
    return True


# --------------------------------------------------------------------------- #
# Память о пользователе (персонализация)
# --------------------------------------------------------------------------- #
async def upsert_memory(user_id: str, key: str, value: str, *, confidence: float = 0.6, source: str = "llm") -> None:
    key = (key or "").strip()[:64]
    value = " ".join((value or "").split())[:500]
    if not user_id or key not in MEMORY_KEYS or not value:
        return
    now = time.time()
    async with get_sessionmaker()() as db:
        row = (
            await db.execute(
                select(UserMemoryRow).where(UserMemoryRow.user_id == user_id, UserMemoryRow.key == key)
            )
        ).scalars().first()
        if row is None:
            db.add(UserMemoryRow(user_id=user_id, key=key, value=value, confidence=confidence,
                                 source=source, updated_at=now))
        else:
            row.value = value
            row.confidence = max(row.confidence or 0.0, confidence)
            row.updated_at = now
        await db.commit()


async def get_memory(user_id: Optional[str], limit: int = 12) -> List[Dict[str, Any]]:
    if not user_id:
        return []
    async with get_sessionmaker()() as db:
        rows = (
            await db.execute(
                select(UserMemoryRow)
                .where(UserMemoryRow.user_id == user_id)
                .order_by(UserMemoryRow.updated_at.desc())
                .limit(limit)
            )
        ).scalars().all()
    return [{"key": r.key, "value": r.value, "confidence": r.confidence, "updated_at": r.updated_at} for r in rows]


async def delete_memory(user_id: str) -> int:
    async with get_sessionmaker()() as db:
        rows = (
            await db.execute(select(UserMemoryRow).where(UserMemoryRow.user_id == user_id))
        ).scalars().all()
        for row in rows:
            await db.delete(row)
        await db.commit()
    return len(rows)


def memory_block(memory: List[Dict[str, Any]]) -> str:
    """Форматирует память о пользователе для системного промпта."""
    if not memory:
        return ""
    return "[USER_MEMORY]\n" + "\n".join(f"- {m['key']}: {m['value']}" for m in memory) + "\n[/USER_MEMORY]\n"


async def stats() -> Dict[str, Any]:
    async with get_sessionmaker()() as db:
        sessions = len((await db.execute(select(SessionRow.id))).scalars().all())
        open_gaps = len(
            (await db.execute(select(KnowledgeGapRow.id).where(KnowledgeGapRow.status == "open"))).scalars().all()
        )
        memory_facts = len((await db.execute(select(UserMemoryRow.id))).scalars().all())
    return {"sessions": sessions, "open_gaps": open_gaps, "memory_facts": memory_facts}


__all__ = [
    "Base",
    "init_db",
    "dispose_db",
    "touch_session",
    "delete_session",
    "record_knowledge_gap",
    "list_knowledge_gaps",
    "set_gap_status",
    "upsert_memory",
    "get_memory",
    "delete_memory",
    "memory_block",
    "stats",
    "MEMORY_KEYS",
]
