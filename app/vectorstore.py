"""Векторное хранилище: ChromaDB + эмбеддинги BAAI/bge-m3 (локально, offline-модель)."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import os

# офлайн-режим по умолчанию: модель bge-m3 уже в локальном кэше HF
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import get_settings

logger = logging.getLogger(__name__)

_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER = None


def get_embedder():
    """Ленивая загрузка sentence-transformers (BAAI/bge-m3) — один раз на процесс."""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        device = settings.embedding_device
        try:
            import torch

            if device.startswith("mps") and not torch.backends.mps.is_available():
                device = "cpu"
        except Exception:  # pragma: no cover
            device = "cpu"
        logger.info("loading embedding model %s on %s", settings.embedding_model, device)
        model = SentenceTransformer(settings.embedding_model, device=device)
        model.eval()
        _EMBEDDER = model
        return _EMBEDDER


def embed_texts(texts: Sequence[str], is_query: bool = False) -> List[List[float]]:
    """Считает эмбеддинги. bge-m3 не требует префиксов, но нормализуем вектора."""
    if not texts:
        return []
    model = get_embedder()
    vectors = model.encode(
        list(texts),
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


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

    # режем на предложения, затем собираем окна
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
            continue
        if len(current) + len(sentence) + 2 <= size:
            current = f"{current}. {sentence}"
        else:
            chunks.append(current)
            tail = current[-ov:] if ov else ""
            current = f"{tail} {sentence}".strip() if tail else sentence
    if current:
        chunks.append(current)

    # жёсткая страховка: если предложение длиннее окна
    fixed: List[str] = []
    for chunk in chunks:
        if len(chunk) <= size * 2:
            fixed.append(chunk)
            continue
        step = size - ov
        for start in range(0, len(chunk), step):
            fixed.append(chunk[start : start + size])
    return [c.strip() for c in fixed if c.strip()]


def _seed_added_at(published_date: str | None) -> int:
    """Дата «добавления в систему» для демо-статей: берём дату публикации (чтобы бейдж NEW был осмысленным)."""
    parsed = parse_date(published_date or "")
    if parsed is None:
        return int(time.time())
    return int(datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc).timestamp())


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
        if not added:
            return False
        return (time.time() - added) <= days * 86400

    def to_source(self) -> Dict[str, Any]:
        snippet = self.document.strip()
        if len(snippet) > 240:
            snippet = snippet[:237].rstrip() + "…"
        return {
            "article_id": self.article_id,
            "section_id": self.section_id or None,
            "title": self.title,
            "date": self.published_date or None,
            "is_new": self.is_new(),
            "snippet": snippet,
        }


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
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "description": "scientific articles chunks (bge-m3)"},
        )

    # ------------------------------------------------------------------ #
    @property
    def collection(self):
        return self._collection

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
        added_at: int | None = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, int]:
        """Режет статью на чанки, считает эмбеддинги и пишет в Chroma."""
        settings = get_settings()
        article_id = article_id or f"art_{uuid.uuid4().hex[:12]}"
        added_at = int(added_at if added_at is not None else time.time())

        # идемпотентность: удаляем предыдущие чанки этой статьи
        self.delete_article(article_id)

        prepared: List[Tuple[str, Dict[str, Any], str]] = []
        if sections:
            for section in sections:
                sec_id = str(section.get("section_id") or f"sec_{uuid.uuid4().hex[:6]}")
                sec_title = str(section.get("title") or "")
                text = str(section.get("text") or "")
                for idx, chunk in enumerate(chunk_text(text, settings.chunk_size, settings.chunk_overlap)):
                    meta = {
                        "article_id": article_id,
                        "section_id": sec_id,
                        "section_title": sec_title,
                        "title": title,
                        "program_id": program_id,
                        "published_date": published_date,
                        "added_at": added_at,
                        "chunk_index": idx,
                    }
                    prepared.append((chunk, meta, f"{article_id}:{sec_id}:{idx}"))
        else:
            sec_id = "sec_main"
            for idx, chunk in enumerate(chunk_text(content, settings.chunk_size, settings.chunk_overlap)):
                meta = {
                    "article_id": article_id,
                    "section_id": sec_id,
                    "section_title": title,
                    "title": title,
                    "program_id": program_id,
                    "published_date": published_date,
                    "added_at": added_at,
                    "chunk_index": idx,
                }
                prepared.append((chunk, meta, f"{article_id}:{sec_id}:{idx}"))

        if not prepared:
            return article_id, 0

        if extra_metadata:
            for _, meta, _ in prepared:
                for key, value in extra_metadata.items():
                    if value is not None and key not in meta:
                        meta[key] = value

        documents = [p[0] for p in prepared]
        metadatas = [p[1] for p in prepared]
        ids = [p[2] for p in prepared]
        embeddings = embed_texts(documents)
        self._collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
        logger.info("article %s indexed: %d chunks (program=%s)", article_id, len(ids), program_id)
        return article_id, len(ids)

    def delete_article(self, article_id: str) -> None:
        try:
            self._collection.delete(where={"article_id": article_id})
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        program_id: str | None = None,
        top_k: int | None = None,
    ) -> List[RetrievedChunk]:
        settings = get_settings()
        k = top_k or settings.retrieval_top_k
        if not query.strip() or self.count() == 0:
            return []
        where = {"program_id": program_id} if program_id else None
        query_embedding = embed_texts([query], is_query=True)[0]
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

        documents = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]

        # Свежие статьи добавляем в кандидаты явно: HNSW может не отдать только что
        # вставленные векторы, а продукт требует «новое знание учитывается сразу».
        seen_ids = {(str(m.get("article_id")), str(m.get("section_id"))) for m in metadatas if m}
        recent = self.recent_chunks(program_id=program_id)
        if recent:
            recent_embeddings = embed_texts([c.document for c in recent], is_query=True)
            for chunk, emb in zip(recent, recent_embeddings):
                key = (chunk.article_id, chunk.section_id)
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                sim = float(sum(a * b for a, b in zip(query_embedding, emb)))
                chunk.distance = max(0.0, 1.0 - sim)
                documents.append(chunk.document)
                metadatas.append(chunk.metadata)
                distances.append(chunk.distance)

        # гибридный ранжинг: близость + бонус свежести (added_at)
        now = time.time()
        scored: List[Tuple[float, RetrievedChunk]] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            meta = dict(meta or {})
            chunk = RetrievedChunk(document=doc or "", metadata=meta, distance=float(dist or 0.0))
            similarity = 1.0 - chunk.distance
            age_days = (now - chunk.added_at) / 86400 if chunk.added_at else 365.0
            freshness_bonus = 0.25 if age_days <= settings.freshness_days else 0.05 if age_days <= 30 else 0.0
            scored.append((similarity + freshness_bonus, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)

        # дедупликация: не больше 2 чанков из одной статьи
        result: List[RetrievedChunk] = []
        per_article: Dict[str, int] = {}
        for _, chunk in scored:
            if per_article.get(chunk.article_id, 0) >= 2:
                continue
            per_article[chunk.article_id] = per_article.get(chunk.article_id, 0) + 1
            result.append(chunk)
            if len(result) >= k:
                break
        return result

    # ------------------------------------------------------------------ #
    def recent_chunks(self, program_id: str | None = None, days: int | None = None) -> List[RetrievedChunk]:
        """Чанки статей, добавленных в базу за последние `days` дней.

        Нужны потому, что HNSW-индекс Chroma отдаёт только что добавленные векторы
        не сразу: без этого «свежая» статья не попадала в выдачу, хотя релевантна.
        """
        settings = get_settings()
        window = days if days is not None else settings.freshness_days
        threshold = int(time.time() - window * 86400)
        where: Dict[str, Any] = {"added_at": {"$gte": threshold}}
        if program_id:
            where = {"$and": [{"program_id": program_id}, {"added_at": {"$gte": threshold}}]}
        try:
            res = self._collection.get(where=where, include=["documents", "metadatas"])
        except Exception as exc:  # pragma: no cover
            logger.warning("recent_chunks failed: %s", exc)
            return []
        documents = res.get("documents") or []
        metadatas = res.get("metadatas") or []
        return [
            RetrievedChunk(document=doc or "", metadata=dict(meta or {}), distance=0.0)
            for doc, meta in zip(documents, metadatas)
        ]

    def list_articles(self, program_id: str | None = None) -> List[Dict[str, Any]]:
        where = {"program_id": program_id} if program_id else None
        try:
            res = self._collection.get(where=where, include=["metadatas"])
        except Exception:  # pragma: no cover
            return []
        metadatas = res.get("metadatas") or []
        grouped: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
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
        # одна запись на статью (чанки могли быть добавлены в разное время)
        merged: Dict[str, Dict[str, Any]] = {}
        for art in grouped.values():
            key = f"{art['article_id']}|{art['title']}|{art['published_date']}"
            if key in merged:
                merged[key]["chunks"] += art["chunks"]
                merged[key]["added_at"] = max(merged[key]["added_at"], art["added_at"])
            else:
                merged[key] = dict(art)
        articles = list(merged.values())
        for art in articles:
            art["is_new"] = (time.time() - art["added_at"]) <= get_settings().freshness_days * 86400
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
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "description": "scientific articles chunks (bge-m3)"},
        )


_store_lock = threading.Lock()
_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = VectorStore()
    return _store


async def add_article_async(**kwargs: Any) -> Tuple[str, int]:
    """CPU-bound индексация в отдельном потоке, чтобы не блокировать event loop."""
    return await asyncio.to_thread(lambda: get_store().add_article(**kwargs))


async def search_async(query: str, program_id: str | None = None, top_k: int | None = None) -> List[RetrievedChunk]:
    return await asyncio.to_thread(get_store().search, query, program_id, top_k)


def parse_date(value: str) -> Optional[date]:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


__all__ = [
    "VectorStore",
    "RetrievedChunk",
    "get_store",
    "add_article_async",
    "search_async",
    "chunk_text",
    "embed_texts",
    "get_embedder",
    "parse_date",
    "today_iso",
]
