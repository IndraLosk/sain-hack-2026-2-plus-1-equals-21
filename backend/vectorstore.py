"""Хранилище документов на чистых Python: BM25 (Okapi) + бонус свежести.

Замена ChromaDB + ONNX bge-m3 на поиск без внешних моделей и без векторов.
Подходит под ограничения стенда: офлайн, только табличные вычисления, без весов моделей
и без скачивания чего-либо в runtime. Пакеты — только стандартная библиотека.

Индекс живой: статьи хранятся в памяти, persist-копия (JSON) в каталоге `persist_dir`.
Поиск: токенизация запроса → BM25-оценка по всем чанкам → нормализация в 0..1 →
порог релевантности → бонус свежести → не более N чанков на статью.

Совместимость интерфейса сохранена: `RetrievedChunk.distance = 1 - normalized_score`,
поэтому логика агента (пороги, отказы, журнал пробелов знаний) не меняется.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import get_settings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Токенизация (русский + английский, латиница и кириллица)
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Нижний регистр, вырезаем токены (слова и числа), без пунктуации и стоп-слов."""
    if not text:
        return []
    STOP = {
        "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
        "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
        "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему",
        "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть",
        "the", "and", "of", "to", "in", "a", "is", "that", "it", "was", "for", "on", "with",
        "as", "are", "this", "be", "at", "from", "by", "an", "or", "we", "their", "which",
    }
    out = [t for t in _TOKEN_RE.findall((text or "").lower()) if t and t not in STOP]
    return out


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


# --------------------------------------------------------------------------- #
# RetrievedChunk (тот же интерфейс, что и у хромового слоя)
# --------------------------------------------------------------------------- #
@dataclass
class RetrievedChunk:
    document: str
    metadata: Dict[str, Any]
    distance: float  # 1 - normalized_score (0..1). 0 = лучший результат.

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


def _chunk_score_to_distance(score: float, max_score: float) -> float:
    """Нормализуем BM25-оценку в [0,1] и инвертируем в 'расстояние'."""
    if max_score <= 0:
        return 1.0
    norm = max(0.0, min(1.0, score / max_score))
    return 1.0 - norm


# --------------------------------------------------------------------------- #
# BM25-индекс (Okapi)
# --------------------------------------------------------------------------- #
class _BM25:
    """Лёгкий BM25-Okapi на чистом Python для набора чанков в памяти."""

    def __init__(self) -> None:
        self._docs: List[List[str]] = []  # токены каждого документа
        self._doc_len: List[int] = []

    def build(self, tokenized_docs: Sequence[List[str]]) -> None:
        self._docs = [list(d) for d in tokenized_docs]
        self._doc_len = [len(d) for d in self._docs]

    def search(self, query_tokens: Sequence[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
        """Возвращает BM25-оценки для каждого документа (0, если нет совпадений)."""
        if not self._docs:
            return []
        N = len(self._docs)
        avg_len = sum(self._doc_len) / N if N else 0.0
        scores = [0.0] * N
        query_tokens = list(query_tokens)
        if not query_tokens:
            return scores

        df: Dict[str, int] = {}
        for doc in self._docs:
            for token in set(doc):
                df[token] = df.get(token, 0) + 1

        for token in query_tokens:
            idf = math.log(1 + (N - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
            if idf <= 0:
                continue
            for i, doc in enumerate(self._docs):
                tf = doc.count(token)
                if tf == 0:
                    continue
                dl = self._doc_len[i]
                denom = tf + k1 * (1 - b + b * dl / avg_len) if avg_len else tf + k1
                scores[i] += idf * (tf * (k1 + 1)) / denom
        return scores


# --------------------------------------------------------------------------- #
# Хранилище
# --------------------------------------------------------------------------- #
class VectorStore:
    """In-memory BM25-индекс с JSON-persist. Публичный API как у Chroma-слоя."""

    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None) -> None:
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_dir
        self.collection_name = collection_name or settings.chroma_collection
        self._chunks: List[Dict[str, Any]] = []      # (document, meta)
        self._bm25 = _BM25()
        self._persist_path = Path(self.persist_dir) / (self.collection_name + ".json")
        self._lock = threading.Lock()
        self._load()

    # ---- persistence ----------------------------------------------------- #
    def _save(self) -> None:
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps({"chunks": self._chunks}, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("bm25 persist failed: %s", exc)

    def _load(self) -> None:
        try:
            if self._persist_path.exists():
                data = json.loads(self._persist_path.read_text(encoding="utf-8"))
                self._chunks = list(data.get("chunks") or [])
        except Exception as exc:  # pragma: no cover
            logger.warning("bm25 load failed, starting empty: %s", exc)
            self._chunks = []
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._bm25.build([tokenize(c.get("document", "")) for c in self._chunks])

    # ---- basics ----------------------------------------------------------- #
    def count(self) -> int:
        return len(self._chunks)

    def article_ids(self) -> set[str]:
        return {str(c.get("metadata", {}).get("article_id")) for c in self._chunks if c.get("metadata")}

    # ---- add / delete ------------------------------------------------------ #
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
        """Режет статью на чанки и добавляет их в BM25-индекс."""
        settings = get_settings()
        article_id = article_id or f"art_{uuid.uuid4().hex[:12]}"
        added_at = int(time.time())
        self.delete_article(article_id)  # идемпотентность

        blocks: List[Tuple[str, str, str]] = []
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

        new_chunks: List[Dict[str, Any]] = []
        with self._lock:
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
                    new_chunks.append({"document": chunk, "metadata": meta})
            self._chunks.extend(new_chunks)
            self._rebuild_index()
            self._save()
        logger.info("article %s indexed: %d chunks (program=%s)", article_id, len(new_chunks), program_id)
        return article_id, len(new_chunks)

    def delete_article(self, article_id: str) -> None:
        with self._lock:
            before = len(self._chunks)
            self._chunks = [
                c for c in self._chunks if str(c.get("metadata", {}).get("article_id")) != article_id
            ]
            if len(self._chunks) != before:
                self._rebuild_index()
                self._save()

    # ---- search ------------------------------------------------------------ #
    def search(
        self,
        query: str,
        program_id: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> List[RetrievedChunk]:
        """BM25-поиск + порог + бонус свежести; всё ниже порога отбрасывается."""
        settings = get_settings()
        k = top_k or settings.retrieval_top_k
        threshold = settings.retrieval_min_similarity if min_similarity is None else min_similarity
        if not query.strip() or self.count() == 0:
            return []

        with self._lock:
            scores = self._bm25.search(tokenize(query))
        # фильтр по программе
        candidate_idx = [
            i for i, c in enumerate(self._chunks)
            if program_id is None or (c.get("metadata") or {}).get("program_id") == program_id
        ]
        max_score = max((scores[i] for i in candidate_idx), default=0.0)
        now = time.time()
        scored: List[Tuple[float, RetrievedChunk]] = []
        for i in candidate_idx:
            score = scores[i]
            if score <= 0:
                continue
            chunk = RetrievedChunk(
                document=str(self._chunks[i].get("document") or ""),
                metadata=dict(self._chunks[i].get("metadata") or {}),
                distance=_chunk_score_to_distance(score, max_score),
            )
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

    # ---- articles ------------------------------------------------------------ #
    def list_articles(self, program_id: str | None = None) -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for c in self._chunks:
            meta = c.get("metadata") or {}
            if program_id and meta.get("program_id") != program_id:
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

    def reset(self) -> None:
        with self._lock:
            self._chunks = []
            self._rebuild_index()
            try:
                if self._persist_path.exists():
                    self._persist_path.unlink()
            except Exception:  # pragma: no cover
                pass
            self._save()


# --------------------------------------------------------------------------- #
# Синглтон + асинхронные обёртки (CPU-bound — в отдельном потоке)
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
    "tokenize",
]