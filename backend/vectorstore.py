"""Векторное хранилище: ChromaDB + эмбеддинги BAAI/bge-m3 на ONNX Runtime (CPU).

Поиск гибридный: косинусная близость + бонус свежести, с жёстким порогом релевантности.
Если ничего не прошло порог — возвращается пустой список, и агент честно говорит,
что в базе нет подтверждённых исследований (а запрос уходит в журнал пробелов знаний).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import get_settings
from .embeddings import embed_texts

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Чанкование
# --------------------------------------------------------------------------- #
def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> List[str]:
    """Разбиение по ~500 символов с перекрытием 100, по границам предложений."""
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    ov = overlap if overlap is not None else settings.chunk_overlap
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    ov = max(0, min(ov, size - 1))
    text = " ".join((text or "").split())
    if not text:
        return []
    if len(text) <= size:
        return [text]

    sentences: List[str] = []
    buf = ""
    for part in text.replace("\n", " ").split(". "):
        part = part.strip()
        if not part:
            continue
        candidate = f"{buf}. {part}" if buf else part
        if len(candidate) <= size:
            buf = candidate
        else:
            if buf:
                sentences.append(buf)
            buf = part
    if buf:
        sentences.append(buf)

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + len(sentence) + 2 <= size:
            current = f"{current}. {sentence}"
        else:
            chunks.append(current)
            tail = current[-ov:] if ov else ""
            current = f"{tail} {sentence}".strip() if tail else sentence
    if current:
        chunks.append(current)

    # страховка: предложение длиннее окна
    fixed: List[str] = []
    for chunk in chunks:
        if len(chunk) <= size * 2:
            fixed.append(chunk)
            continue
        step = size - ov
        for start in range(0, len(chunk), step):
            fixed.append(chunk[start : start + size])
    return [c.strip() for c in fixed if c.strip()]


def parse_date(value: str) -> Optional[date]:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


@dataclass
class RetrievedChunk:
    document: str
    metadata: Dict[str, Any]
    distance: float

    @property
    def article_id(self) -> str:
        return str(self.metadata.get("article_id", ""))

    @property
    def section_id(self) -> str:
        return str(self.metadata.get("section_id", "") or "")

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))

    @property
    def published_date(self) -> str:
        return str(self.metadata.get("published_date", "") or "")

    @property
    def added_at(self) -> int:
        try:
            return int(self.metadata.get("added_at") or 0)
        except (TypeError, ValueError):
            return 0

    def is_new(self, freshness_days: int | None = None) -> bool:
        days = freshness_days if freshness_days is not None else get_settings().freshness_days
        added = self.added_at
        return bool(added) and (time.time() - added) <= days * 86400


# --------------------------------------------------------------------------- #
# Хранилище
# --------------------------------------------------------------------------- #
class VectorStore:
    """Обёртка над ChromaDB (persistent, локальный каталог data/chroma)."""

    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None) -> None:
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_dir
        self.collection_name = collection_name or settings.chroma_collection
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._get_collection()

    def _get_collection(self):
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "description": "scientific articles chunks (bge-m3 onnx)"},
        )

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:  # pragma: no cover
            return 0

    # ------------------------------------------------------------------ #
    def add_article(
        self,
        *,
        program_id: str,
        title: str,
        published_date: str,
        content: str,
        article_id: str | None = None,
        sections: Optional[List[Dict[str, str]]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int]:
        """Режет статью на чанки, считает эмбеддинги и пишет в Chroma."""
        settings = get_settings()
        article_id = article_id or f"art_{uuid.uuid4().hex[:12]}"
        added_at = int(time.time())
        self.delete_article(article_id)  # идемпотентность

        blocks: List[Tuple[str, str, str]] = []  # (section_id, section_title, text)
        if sections:
            for section in sections:
                blocks.append(
                    (
                        str(section.get("section_id") or f"sec_{uuid.uuid4().hex[:6]}"),
                        str(section.get("title") or ""),
                        str(section.get("text") or ""),
                    )
                )
        else:
            blocks.append(("sec_main", title, content))

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        for sec_id, sec_title, text in blocks:
            for idx, chunk in enumerate(chunk_text(text, settings.chunk_size, settings.chunk_overlap)):
                meta: Dict[str, Any] = {
                    "article_id": article_id,
                    "section_id": sec_id,
                    "section_title": sec_title,
                    "title": title,
                    "program_id": program_id,
                    "published_date": published_date,
                    "added_at": added_at,
                    "chunk_index": idx,
                }
                for key, value in (extra_metadata or {}).items():
                    if value is not None and key not in meta:
                        meta[key] = value
                documents.append(chunk)
                metadatas.append(meta)
                ids.append(f"{article_id}:{sec_id}:{idx}")

        if not documents:
            return article_id, 0

        self._collection.add(
            documents=documents, metadatas=metadatas, ids=ids, embeddings=embed_texts(documents)
        )
        logger.info("article %s indexed: %d chunks (program=%s)", article_id, len(ids), program_id)
        return article_id, len(ids)

    def delete_article(self, article_id: str) -> None:
        try:
            self._collection.delete(where={"article_id": article_id})
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------ #
    def _fresh_chunks(self, program_id: str | None) -> List[RetrievedChunk]:
        """Чанки статей, добавленных в базу за последние `freshness_days` дней.

        HNSW может не отдать только что вставленные векторы, а продукт требует
        «новое знание учитывается сразу» — такие чанки досчитываем явно.
        """
        settings = get_settings()
        threshold = int(time.time() - settings.freshness_days * 86400)
        where: Dict[str, Any] = {"added_at": {"$gte": threshold}}
        if program_id:
            where = {"$and": [{"program_id": program_id}, {"added_at": {"$gte": threshold}}]}
        try:
            res = self._collection.get(where=where, include=["documents", "metadatas"])
        except Exception as exc:  # pragma: no cover
            logger.warning("fresh chunks query failed: %s", exc)
            return []
        pairs = list(zip(res.get("documents") or [], res.get("metadatas") or []))
        pairs.sort(key=lambda p: int((p[1] or {}).get("added_at") or 0), reverse=True)
        pairs = pairs[: settings.recent_candidates_limit]
        return [
            RetrievedChunk(document=doc or "", metadata=dict(meta or {}), distance=0.0) for doc, meta in pairs
        ]

    def search(
        self,
        query: str,
        program_id: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> List[RetrievedChunk]:
        """Косинусный поиск + бонус свежести; всё ниже порога отбрасывается."""
        settings = get_settings()
        k = top_k or settings.retrieval_top_k
        threshold = settings.retrieval_min_similarity if min_similarity is None else min_similarity
        if not query.strip() or self.count() == 0:
            return []

        query_embedding = embed_texts([query], is_query=True)[0]
        where = {"program_id": program_id} if program_id else None
        try:
            n_candidates = min(max(k * 12, 60), max(self.count(), 1))
            res = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_candidates,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("chroma query failed: %s", exc)
            res = {}

        documents = list((res.get("documents") or [[]])[0])
        metadatas = list((res.get("metadatas") or [[]])[0])
        distances = list((res.get("distances") or [[]])[0])

        seen = {(str(m.get("article_id")), str(m.get("section_id"))) for m in metadatas if m}
        for chunk in self._fresh_chunks(program_id):
            key = (chunk.article_id, chunk.section_id)
            if key in seen:
                continue
            seen.add(key)
            sim = float(sum(a * b for a, b in zip(query_embedding, embed_texts([chunk.document])[0])))
            documents.append(chunk.document)
            metadatas.append(chunk.metadata)
            distances.append(max(0.0, 1.0 - sim))

        now = time.time()
        scored: List[Tuple[float, RetrievedChunk]] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            chunk = RetrievedChunk(document=doc or "", metadata=dict(meta or {}), distance=float(dist or 0.0))
            similarity = 1.0 - chunk.distance
            if similarity < threshold:
                continue
            age_days = (now - chunk.added_at) / 86400 if chunk.added_at else 365.0
            bonus = 0.25 if age_days <= settings.freshness_days else 0.05 if age_days <= 30 else 0.0
            scored.append((similarity + bonus, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)

        result: List[RetrievedChunk] = []
        per_article: Dict[str, int] = {}
        for _, chunk in scored:
            if per_article.get(chunk.article_id, 0) >= 2:  # не больше 2 чанков из одной статьи
                continue
            per_article[chunk.article_id] = per_article.get(chunk.article_id, 0) + 1
            result.append(chunk)
            if len(result) >= k:
                break
        return result

    # ------------------------------------------------------------------ #
    def list_articles(self, program_id: str | None = None) -> List[Dict[str, Any]]:
        where = {"program_id": program_id} if program_id else None
        try:
            res = self._collection.get(where=where, include=["metadatas"])
        except Exception:  # pragma: no cover
            return []
        grouped: Dict[str, Dict[str, Any]] = {}
        for meta in res.get("metadatas") or []:
            if not meta:
                continue
            aid = str(meta.get("article_id"))
            entry = grouped.setdefault(
                aid,
                {
                    "article_id": aid,
                    "program_id": str(meta.get("program_id", "")),
                    "title": str(meta.get("title", "")),
                    "published_date": str(meta.get("published_date", "")),
                    "added_at": int(meta.get("added_at") or 0),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
            entry["added_at"] = max(entry["added_at"], int(meta.get("added_at") or 0))
        articles = list(grouped.values())
        freshness = get_settings().freshness_days * 86400
        for art in articles:
            art["is_new"] = (time.time() - art["added_at"]) <= freshness
        articles.sort(key=lambda a: a["added_at"], reverse=True)
        return articles

    def article_ids(self) -> set[str]:
        try:
            res = self._collection.get(include=["metadatas"])
        except Exception:  # pragma: no cover
            return set()
        return {str(m.get("article_id")) for m in (res.get("metadatas") or []) if m}

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # pragma: no cover
            pass
        self._collection = self._get_collection()


# --------------------------------------------------------------------------- #
# Синглтон + асинхронные обёртки (CPU-bound работа — в отдельном потоке)
# --------------------------------------------------------------------------- #
_store_lock = threading.Lock()
_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = VectorStore()
    return _store


async def add_article_async(**kwargs: Any) -> Tuple[str, int]:
    return await asyncio.to_thread(lambda: get_store().add_article(**kwargs))


async def search_async(
    query: str,
    program_id: str | None = None,
    top_k: int | None = None,
    min_similarity: float | None = None,
) -> List[RetrievedChunk]:
    return await asyncio.to_thread(get_store().search, query, program_id, top_k, min_similarity)


__all__ = [
    "VectorStore",
    "RetrievedChunk",
    "get_store",
    "add_article_async",
    "search_async",
    "chunk_text",
    "parse_date",
    "today_iso",
]
